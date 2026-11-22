// Tests of the pickle refusal and of the reader table and magic bytes it uses.
// Every pickle here is a few bytes built in code; none is ever loaded.
#include <caliper_builder/loaders/pickle.h>
#include <caliper_builder/loaders/readers.h>

#include <doctest/doctest.h>

#include "test_formats.h"

using namespace caliper_builder;
using namespace loaders_test;

namespace {

const std::string kFixtures = std::string(CALIPER_BUILDER_SOURCE_DIR) + "/loaders/tests/fixtures/";

std::string file(const std::string& dir, const std::string& name, const std::vector<uint8_t>& bytes) {
    write_bytes(dir + "/" + name, bytes);
    return dir + "/" + name;
}

std::vector<uint8_t> text(const std::string& s) { return std::vector<uint8_t>(s.begin(), s.end()); }

}  // namespace

TEST_CASE("reader names and extensions") {
    CHECK(reader_names().size() == 12);
    for (const std::string& n : reader_names()) {
        ReaderKind r;
        REQUIRE(reader_from_name(n, &r));
        CHECK(n == reader_name(r));
    }
    ReaderKind r;
    CHECK_FALSE(reader_from_name("excel", &r));
    REQUIRE(reader_for_extension("a/b/Data.TSV", &r));
    CHECK(r == ReaderKind::csv);
    REQUIRE(reader_for_extension("x.ndjson", &r));
    CHECK(r == ReaderKind::jsonl);
    REQUIRE(reader_for_extension("x.JPEG", &r));
    CHECK(r == ReaderKind::image);
    REQUIRE(reader_for_extension("w.safetensors", &r));
    CHECK(r == ReaderKind::safetensors);
    CHECK_FALSE(reader_for_extension("x.bin", &r));
    CHECK_FALSE(reader_for_extension("noext", &r));
    CHECK(reader_is_tabular(ReaderKind::arrow));
    CHECK_FALSE(reader_is_tabular(ReaderKind::npz));
    CHECK(reader_is_single_tensor(ReaderKind::wav));
    CHECK_FALSE(reader_is_single_tensor(ReaderKind::safetensors));
}

TEST_CASE("magic bytes identify the binary formats") {
    auto sniff = [](const std::vector<uint8_t>& b, uint64_t size = 1000) {
        return sniff_format(b.data(), b.size(), size);
    };
    CHECK(sniff({0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0}) == "npy");
    CHECK(sniff({'P', 'K', 3, 4, 0}) == "zip");
    CHECK(sniff({'P', 'K', 5, 6, 0}) == "zip");
    CHECK(sniff(text("PAR1xxxx")) == "parquet");
    CHECK(sniff(text("ARROW1")) == "arrow");
    CHECK(sniff({0x89, 'H', 'D', 'F', '\r', '\n', 0x1a, '\n'}) == "hdf5");
    CHECK(sniff({0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'}) == "png");
    CHECK(sniff({0xFF, 0xD8, 0xFF, 0xE0}) == "jpeg");
    CHECK(sniff({'R', 'I', 'F', 'F', 0x10, 0, 0, 0, 'W', 'A', 'V', 'E', 'f', 'm', 't', ' '}) == "wav");
    CHECK(sniff({0x80, 0x04, 0x95}) == "pickle");
    CHECK(sniff({0x80, 0x01}) == "");
    CHECK(sniff({20, 0, 0, 0, 0, 0, 0, 0, '{', '"'}) == "safetensors");
    CHECK(sniff({20, 0, 0, 0, 0, 0, 0, 0, '{', '"'}, 10) == "");  // the length runs past the file
    CHECK(sniff(text("a,b\n1,2\n")) == "");
    CHECK(sniff(text("{\"a\": 1}")) == "");
    CHECK(sniff({}) == "");
    ReaderKind r;
    REQUIRE(reader_for_format("zip", &r));
    CHECK(r == ReaderKind::npz);
    CHECK_FALSE(reader_for_format("pickle", &r));
}

TEST_CASE("pickle files are refused by extension") {
    const std::string dir = fresh_dir("pickle_ext");
    for (const char* name : {"m.pkl", "m.pickle", "m.pt", "M.PTH", "m.ckpt", "m.joblib"}) {
        CAPTURE(name);
        CHECK(has_pickle_extension(name));
        Error e = check_not_pickle(file(dir, name, text("anything")));
        CHECK(e.code == "loader.pickle_refused");
        CHECK(e.message.find("safetensors") != std::string::npos);
        CHECK(e.message.find("torch.load(weights_only=True)") != std::string::npos);
        CHECK(e.message.find("warning") != std::string::npos);
    }
    CHECK_FALSE(has_pickle_extension("m.bin"));
    CHECK_FALSE(has_pickle_extension("pkl"));
}

TEST_CASE("pickle files are refused by their bytes") {
    const std::string dir = fresh_dir("pickle_bytes");
    // A legacy torch .bin and a pickle of protocol 5 under another extension.
    CHECK(check_not_pickle(file(dir, "pytorch_model.bin", {0x80, 0x02, 0x8a, 0x0a})).code == "loader.pickle_refused");
    CHECK(check_not_pickle(file(dir, "weights.dat", {0x80, 0x05, 0x95, 0x00})).code == "loader.pickle_refused");
    CHECK(check_not_pickle(file(dir, "looks.npy", {0x80, 0x04, 0x95})).code == "loader.pickle_refused");
    // Protocols 0 and 1, judged only when the extension has no reader.
    CHECK(check_not_pickle(file(dir, "old.bin", text("(lp0\nI1\na."))).code == "loader.pickle_refused");
    CHECK(check_not_pickle(file(dir, "old1.bin", {']', 'q', 0, '.'})).code == "loader.pickle_refused");
    CHECK(check_not_pickle(file(dir, "note.csv", text("(a)\n(b)."))).ok());
    CHECK(check_not_pickle(file(dir, "plain.bin", text("hello world"))).ok());
    // The torch zip layout.
    CHECK(check_not_pickle(file(dir, "model.bin", make_zip({{"archive/data.pkl", {0x80, 0x02, '.'}},
                                                            {"archive/data/0", {0, 0, 0, 0}}})))
              .code == "loader.pickle_refused");
    CHECK(check_not_pickle(file(dir, "model.zip", make_zip({{"data.pkl", {'.'}}}))).code == "loader.pickle_refused");
    // An .npy with the object dtype, alone or inside an .npz.
    const std::vector<uint8_t> obj =
        make_npy("{'descr': '|O', 'fortran_order': False, 'shape': (1,), }", {0x80, 0x02, ']', '.'});
    Error e = check_not_pickle(file(dir, "obj.npy", obj));
    CHECK(e.code == "loader.pickle_refused");
    CHECK(e.message.find("object dtype") != std::string::npos);
    e = check_not_pickle(file(dir, "obj.npz", make_zip({{"ok.npy", make_npy("{'descr': '<f4', 'fortran_order': False, 'shape': (0,), }", {})},
                                                        {"obj.npy", obj}})));
    CHECK(e.code == "loader.pickle_refused");
    CHECK(e.message.find("obj.npy") != std::string::npos);
#ifdef CALIPER_BUILDER_HAVE_ZLIB
    e = check_not_pickle(file(dir, "objz.npz", make_zip({{"obj.npy", obj, true}})));
    CHECK(e.code == "loader.pickle_refused");
#endif
    CHECK(check_not_pickle(file(dir, "pk.npz", make_zip({{"blob", {0x80, 0x03, 'K', 1, '.'}}}))).code ==
          "loader.pickle_refused");
    CHECK(check_not_pickle(file(dir, "pk2.npz", make_zip({{"state.pickle", {'.'}}}))).code == "loader.pickle_refused");
}

TEST_CASE("files that are not pickles pass the check") {
    for (const char* name : {"numpy_savez.npz", "numpy_savez_compressed.npz", "numpy_fortran_f8.npy",
                             "real.safetensors"}) {
        CAPTURE(name);
        CHECK(check_not_pickle(kFixtures + name).ok());
    }
    const std::string dir = fresh_dir("pickle_none");
    CHECK(check_not_pickle(file(dir, "a.csv", text("x,y\n1,2\n"))).ok());
    CHECK(check_not_pickle(file(dir, "empty.bin", {})).ok());
    CHECK(check_not_pickle(file(dir, "broken.npz", {'P', 'K', 3, 4, 1, 2})).ok());
    CHECK(check_not_pickle(dir + "/missing.parquet").ok());
}
