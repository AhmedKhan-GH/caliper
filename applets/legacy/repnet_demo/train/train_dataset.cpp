#include "train_dataset.h"

#include "dsp.h"
#include "sgkf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tdata {

namespace {

// Lead order, matching Python SD_LEAD_ORDER.
const std::vector<std::string> kLeadOrder = {
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"};
const char* kLabelPos =
    "Preeclampsia or Other Hypertensive Disorders of Pregnancy";

// Trim leading/trailing ASCII whitespace (mirrors pandas skipinitialspace +
// header cleanup well enough for these clean CSV headers).
std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    std::stringstream ss(line);
    while (std::getline(ss, cur, ',')) out.push_back(trim(cur));
    return out;
}

// Read one ECG CSV; fill data (column-major leads x time). Returns time length
// T, or -1 on failure. Reads the 12 named lead columns in kLeadOrder.
int read_csv_leads(const std::string& path, std::vector<std::vector<double>>& leads) {
    std::ifstream f(path);
    if (!f) return -1;
    std::string header;
    if (!std::getline(f, header)) return -1;
    std::vector<std::string> cols = split_csv_line(header);
    // Map lead name -> column index.
    std::vector<int> col_idx(kLeadOrder.size(), -1);
    for (size_t li = 0; li < kLeadOrder.size(); ++li) {
        for (size_t c = 0; c < cols.size(); ++c) {
            if (cols[c] == kLeadOrder[li]) {
                col_idx[li] = static_cast<int>(c);
                break;
            }
        }
        if (col_idx[li] < 0) return -1;  // missing lead column
    }

    leads.assign(kLeadOrder.size(), {});
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<std::string> vals = split_csv_line(line);
        // pandas usecols would error on short rows; treat as parse failure.
        for (size_t li = 0; li < kLeadOrder.size(); ++li) {
            int ci = col_idx[li];
            if (ci >= static_cast<int>(vals.size())) return -1;
            const std::string& tok = vals[ci];
            double v;
            if (tok.empty()) {
                v = std::nan("");
            } else {
                try {
                    size_t pos = 0;
                    v = std::stod(tok, &pos);
                } catch (...) {
                    return -1;
                }
            }
            leads[li].push_back(v);
        }
    }
    int T = leads[0].empty() ? 0 : static_cast<int>(leads[0].size());
    for (const auto& l : leads)
        if (static_cast<int>(l.size()) != T) return -1;
    return T;
}

}  // namespace

torch::Tensor resample_to_5000(const torch::Tensor& x2500) {
    // scipy.signal.resample(x, 5000, axis=1) for even input length N=2500.
    // X = rfft(x) (1251 bins); copy bins [0, N/2]; halve the Nyquist bin
    // (index N/2) when upsampling an even-length signal; irfft to 5000 and
    // scale by num/N = 2.0.  Double precision throughout (matches golden <1e-3).
    TORCH_CHECK(x2500.size(-1) == 2500, "resample_to_5000 expects last dim 2500");
    const int64_t N = 2500;
    const int64_t num = 5000;
    torch::Tensor x = x2500.to(torch::kDouble);
    torch::Tensor X = torch::fft::rfft(x, /*n=*/N, /*dim=*/-1);  // (..., 1251)
    const int64_t nyq = N / 2 + 1;        // 1251 source bins
    const int64_t new_bins = num / 2 + 1; // 2501 destination bins

    std::vector<int64_t> yshape = X.sizes().vec();
    yshape.back() = new_bins;
    torch::Tensor Y = torch::zeros(yshape, X.options());
    Y.slice(-1, 0, nyq) = X.slice(-1, 0, nyq);
    // Even-length upsample: split the (single real) Nyquist component.
    Y.select(-1, N / 2) = Y.select(-1, N / 2) * 0.5;

    torch::Tensor y = torch::fft::irfft(Y, /*n=*/num, /*dim=*/-1);
    y = y * (static_cast<double>(num) / static_cast<double>(N));
    return y.to(torch::kFloat32);
}

Dataset load_and_preprocess(const std::string& data_dir,
                            std::function<void(int, int)> progress) {
    const std::string ekg_dir = data_dir + "/ekg_data";
    const std::string meta_path = data_dir + "/metadata.csv";

    // ---- read metadata.csv header to find columns ----
    std::ifstream mf(meta_path);
    if (!mf) throw std::runtime_error("cannot open metadata.csv: " + meta_path);
    std::string mheader;
    std::getline(mf, mheader);
    std::vector<std::string> mcols = split_csv_line(mheader);
    int c_ecg = -1, c_mrn = -1, c_label = -1;
    for (size_t i = 0; i < mcols.size(); ++i) {
        if (mcols[i] == "ECGTestID") c_ecg = static_cast<int>(i);
        else if (mcols[i] == "Pat_Obfus_MRN") c_mrn = static_cast<int>(i);
        else if (mcols[i] == "PatLabel") c_label = static_cast<int>(i);
    }
    if (c_ecg < 0 || c_mrn < 0 || c_label < 0)
        throw std::runtime_error("metadata.csv missing required columns");

    // Build the set of available ECGTestIDs (CSV present in ekg_data).
    // We test availability lazily by attempting to open the file.

    // Parse metadata rows IN FILE ORDER. We need the raw cell strings so we can
    // parse via int(float(...)) and detect blank patient ids.
    struct MetaRow {
        long ecg_id;
        long mrn;        // patient id (int via int(float))
        bool mrn_blank;
        int y;
    };
    std::vector<MetaRow> rows;
    std::string mline;
    while (std::getline(mf, mline)) {
        if (mline.empty()) continue;
        std::vector<std::string> v = split_csv_line(mline);
        int need = std::max({c_ecg, c_mrn, c_label});
        if (static_cast<int>(v.size()) <= need) continue;
        const std::string& ecg_s = v[c_ecg];
        if (ecg_s.empty()) continue;
        long ecg_id;
        try {
            ecg_id = static_cast<long>(std::stod(ecg_s));  // int(float(...))
        } catch (...) {
            continue;
        }
        const std::string& mrn_s = v[c_mrn];
        bool blank = mrn_s.empty() || mrn_s == "nan" || mrn_s == "None";
        long mrn = 0;
        if (!blank) {
            try {
                mrn = static_cast<long>(std::stod(mrn_s));
            } catch (...) {
                blank = true;
            }
        }
        int y = (v[c_label] == kLabelPos) ? 1 : 0;
        rows.push_back({ecg_id, mrn, blank, y});
    }

    // ---- iterate metadata rows in file order, load+resample+validate ----
    std::vector<torch::Tensor> X_list;  // each (12,5000) float32
    std::vector<int> y_list;
    std::vector<long> mrn_list;
    std::vector<bool> mrn_blank_list;
    std::vector<long> ecg_list;

    int total = static_cast<int>(rows.size());
    int done = 0;
    for (const auto& r : rows) {
        ++done;
        if (progress) progress(done, total);
        const std::string path = ekg_dir + "/" + std::to_string(r.ecg_id) + ".csv";
        std::vector<std::vector<double>> leads;
        int T = read_csv_leads(path, leads);
        if (T < 0) continue;  // file missing or parse failure -> skip record

        torch::Tensor arr;  // (12, T)
        {
            torch::Tensor t = torch::empty({12, T}, torch::kFloat32);
            auto a = t.accessor<float, 2>();
            for (int li = 0; li < 12; ++li)
                for (int j = 0; j < T; ++j)
                    a[li][j] = static_cast<float>(leads[li][j]);
            arr = t;
        }

        if (T == 5000) {
            // keep
        } else if (T == 2500) {
            arr = resample_to_5000(arr);  // (12,5000) float32
        } else {
            continue;  // skip
        }
        if (arr.size(0) != 12 || arr.size(1) != 5000) continue;

        X_list.push_back(arr);
        y_list.push_back(r.y);
        mrn_list.push_back(r.mrn);
        mrn_blank_list.push_back(r.mrn_blank);
        ecg_list.push_back(r.ecg_id);
    }

    // ---- stack X (N,12,5000) and apply filters in Python order ----
    const int N0 = static_cast<int>(X_list.size());
    std::vector<bool> keep(N0, true);

    // (a) drop rows with any non-finite value (over lead,time).
    for (int i = 0; i < N0; ++i) {
        if (!keep[i]) continue;
        if (!torch::isfinite(X_list[i]).all().item<bool>()) keep[i] = false;
    }
    // (b) drop flatline: any lead with population std (over time) < 1e-4.
    for (int i = 0; i < N0; ++i) {
        if (!keep[i]) continue;
        // std over time per lead, population (unbiased=false).
        torch::Tensor s = X_list[i].std(/*dim=*/1, /*unbiased=*/false);  // (12,)
        if ((s < 1e-4).any().item<bool>()) keep[i] = false;
    }
    // (c) drop rows with missing patient id.
    for (int i = 0; i < N0; ++i) {
        if (!keep[i]) continue;
        if (mrn_blank_list[i]) keep[i] = false;
    }

    std::vector<torch::Tensor> Xk;
    std::vector<int> yk;
    std::vector<long> mrnk;
    std::vector<long> ecgk;
    for (int i = 0; i < N0; ++i) {
        if (!keep[i]) continue;
        Xk.push_back(X_list[i]);
        yk.push_back(y_list[i]);
        mrnk.push_back(mrn_list[i]);
        ecgk.push_back(ecg_list[i]);
    }
    const int N = static_cast<int>(Xk.size());

    // ---- DSP preprocess each (12,5000) -> (12,2500) and stack ----
    Dataset ds;
    ds.X = torch::empty({N, 12, 2500}, torch::kFloat32);
    for (int i = 0; i < N; ++i) {
        torch::Tensor pp = dsp::preprocess_5k(Xk[i]);  // (12,2500)
        ds.X[i] = pp;
    }
    ds.y = std::move(yk);
    ds.ecg_ids = std::move(ecgk);

    // ---- groups_inv: numpy-unique-sorted encoding of patient ids ----
    std::vector<long> uniq(mrnk.begin(), mrnk.end());
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    std::map<long, int> remap;
    for (int i = 0; i < static_cast<int>(uniq.size()); ++i) remap[uniq[i]] = i;
    ds.groups_inv.resize(N);
    for (int i = 0; i < N; ++i) ds.groups_inv[i] = remap[mrnk[i]];
    ds.n_groups = static_cast<int>(uniq.size());

    return ds;
}

Split make_split(const Dataset& d, int split_i) {
    const int N = static_cast<int>(d.y.size());
    const uint32_t outer_seed = static_cast<uint32_t>(split_i * 7 + 1000);

    // Outer: 5-fold; test = fold0.
    auto outer = sgkf::stratified_group_kfold_test_folds(d.y, d.groups_inv, 5,
                                                         outer_seed);
    Split out;
    out.test = outer[0];
    std::sort(out.test.begin(), out.test.end());

    std::set<int> test_set(out.test.begin(), out.test.end());
    std::vector<int> dev;  // sorted complement
    dev.reserve(N - static_cast<int>(out.test.size()));
    for (int i = 0; i < N; ++i)
        if (!test_set.count(i)) dev.push_back(i);

    // Inner: re-encode dev groups (numpy-unique-sorted over dev patients).
    std::vector<int> y_dev;
    std::vector<int> dev_groups_orig;
    y_dev.reserve(dev.size());
    dev_groups_orig.reserve(dev.size());
    for (int s : dev) {
        y_dev.push_back(d.y[s]);
        dev_groups_orig.push_back(d.groups_inv[s]);
    }
    std::vector<int> uniq(dev_groups_orig.begin(), dev_groups_orig.end());
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    std::map<int, int> remap;
    for (int i = 0; i < static_cast<int>(uniq.size()); ++i) remap[uniq[i]] = i;
    std::vector<int> groups_inv_dev(dev_groups_orig.size());
    for (size_t i = 0; i < dev_groups_orig.size(); ++i)
        groups_inv_dev[i] = remap[dev_groups_orig[i]];

    auto inner = sgkf::stratified_group_kfold_test_folds(
        y_dev, groups_inv_dev, 8, outer_seed + 1);
    // val = dev[inner[0]]; train = dev minus val.
    std::set<int> val_local(inner[0].begin(), inner[0].end());
    for (int li = 0; li < static_cast<int>(dev.size()); ++li) {
        if (val_local.count(li))
            out.val.push_back(dev[li]);
        else
            out.train.push_back(dev[li]);
    }
    std::sort(out.val.begin(), out.val.end());
    std::sort(out.train.begin(), out.train.end());
    return out;
}

}  // namespace tdata
