#include "artifact_store.h"

#include "sha256.h"

#include <duckdb.hpp>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>

namespace fs = std::filesystem;

namespace caliper_host {

// All state behind the pimpl so no DuckDB type escapes the header. One
// connection, one mutex — every public method takes the lock, so calls from
// job threads serialize (artifacts.v1 contract).
struct ArtifactStore::Impl {
    std::unique_ptr<duckdb::DuckDB>     db;
    std::unique_ptr<duckdb::Connection> con;
    std::mutex mu;

    fs::path blob_dir;   // <root>/artifacts
    bool     opened = false;

    // Backing storage for the C ABI's "host-owned string, valid until the
    // next call" contract (path_of thunk in host_services.cpp returns
    // .c_str() of this).
    std::string last_path;

    // ts source: milliseconds, strictly monotonic per process so that two
    // puts in the same millisecond still order correctly for name->newest.
    int64_t last_ts = 0;
    int64_t next_ts() {
        int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
        last_ts = (now > last_ts) ? now : last_ts + 1;
        return last_ts;
    }

    // Caller holds mu. Resolve a digest-or-name to a digest; empty if unknown.
    std::string resolve(const std::string& key) {
        if (!opened) return {};
        auto stmt = con->Prepare(
            "SELECT digest FROM artifacts WHERE digest = $1 OR name = $1 "
            "ORDER BY ts DESC LIMIT 1");
        if (stmt->HasError()) return {};
        auto r = stmt->Execute(key.c_str());
        if (!r || r->HasError()) return {};
        auto chunk = r->Fetch();
        if (!chunk || chunk->size() == 0) return {};
        return chunk->GetValue(0, 0).ToString();
    }
};

ArtifactStore::ArtifactStore() : impl_(std::make_unique<Impl>()) {}
ArtifactStore::~ArtifactStore() = default;

bool ArtifactStore::open(const std::string& root_dir) {
    std::lock_guard<std::mutex> lk(impl_->mu);

    std::error_code ec;
    impl_->blob_dir = fs::path(root_dir) / "artifacts";
    fs::create_directories(impl_->blob_dir, ec);
    if (ec) return false;

    try {
        std::string db_path = (fs::path(root_dir) / "artifacts.duckdb").string();
        impl_->db  = std::make_unique<duckdb::DuckDB>(db_path.c_str());
        impl_->con = std::make_unique<duckdb::Connection>(*impl_->db);
    } catch (...) {
        return false;
    }

    auto r = impl_->con->Query(
        "CREATE TABLE IF NOT EXISTS artifacts ("
        "digest VARCHAR, name VARCHAR, run BIGINT, len BIGINT, ts BIGINT)");
    impl_->opened = r && !r->HasError();
    return impl_->opened;
}

void ArtifactStore::close() {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->con.reset();
    impl_->db.reset();
    impl_->opened = false;
}

bool ArtifactStore::put(const std::string& name, const void* bytes,
                        uint64_t len, uint64_t run, char out_digest[65]) {
    if (!bytes || !out_digest) return false;
    std::lock_guard<std::mutex> lk(impl_->mu);
    if (!impl_->opened) return false;

    std::string digest = sha256_hex(bytes, static_cast<size_t>(len));

    // Dedup: the blob file is written only if this content is new. Write via
    // a temp name + rename so a crash mid-write can't leave a half blob
    // claiming a valid digest (same atomic recipe as the MNIST cache).
    fs::path blob = impl_->blob_dir / digest;
    if (!fs::exists(blob)) {
        fs::path tmp = blob;
        tmp += ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out) return false;
            out.write(static_cast<const char*>(bytes),
                      static_cast<std::streamsize>(len));
            if (!out) {
                std::error_code ec;
                fs::remove(tmp, ec);
                return false;
            }
        }
        std::error_code ec;
        fs::rename(tmp, blob, ec);
        if (ec) {
            fs::remove(tmp, ec);
            return false;
        }
    }

    // Every put records an index row: names are history, newest wins.
    auto stmt = impl_->con->Prepare(
        "INSERT INTO artifacts VALUES ($1, $2, $3, $4, $5)");
    if (stmt->HasError()) return false;
    auto r = stmt->Execute(digest.c_str(), name.c_str(),
                           static_cast<int64_t>(run),
                           static_cast<int64_t>(len), impl_->next_ts());
    if (!r || r->HasError()) return false;

    std::memcpy(out_digest, digest.c_str(), 65);
    return true;
}

std::string ArtifactStore::path_of(const std::string& digest_or_name) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    std::string digest = impl_->resolve(digest_or_name);
    if (digest.empty()) return {};
    fs::path blob = impl_->blob_dir / digest;
    if (!fs::exists(blob)) return {};  // index row without blob = corrupt; inert
    impl_->last_path = blob.string();
    return impl_->last_path;
}

bool ArtifactStore::exists(const std::string& digest_or_name) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    std::string digest = impl_->resolve(digest_or_name);
    return !digest.empty() && fs::exists(impl_->blob_dir / digest);
}

std::vector<std::string> ArtifactStore::by_run(uint64_t run) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    std::vector<std::string> out;
    if (!impl_->opened) return out;
    auto stmt = impl_->con->Prepare(
        "SELECT DISTINCT digest FROM artifacts WHERE run = $1");
    if (stmt->HasError()) return out;
    auto r = stmt->Execute(static_cast<int64_t>(run));
    if (!r || r->HasError()) return out;
    while (auto chunk = r->Fetch()) {
        for (duckdb::idx_t i = 0; i < chunk->size(); i++)
            out.push_back(chunk->GetValue(0, i).ToString());
    }
    return out;
}

}  // namespace caliper_host
