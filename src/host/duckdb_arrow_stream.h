#pragma once
// caliper_host — the SQL→Arrow C stream producer, shared verbatim by the two
// host stores that expose a read surface across the ABI: caliper.data.v1
// (DataStore) and caliper.metrics.v1_1 (MetricsStore). ONE Arrow C stream
// contract (D3): the producer owns a fully materialized DuckDB result, so the
// stream outlives the originating store call and never touches the connection
// again — a consumer may drain it on any thread while the store keeps serving.
//
// Consumer protocol (Arrow C spec): drain via get_schema/get_next until an
// array whose release is NULL (end of stream), then release the stream exactly
// once; release nulls itself.
//
// This is an INTERNAL host header (src/host); no DuckDB type crosses the ABI.
#include <caliper/arrow_c.h>

#include <duckdb.hpp>
#include <duckdb/common/arrow/arrow_converter.hpp>

#include <cerrno>
#include <string>

namespace caliper_host {
namespace arrow_stream {

// Producer state: owns the materialized result + the client properties Arrow
// conversion needs. Heap-allocated, freed by release().
struct State {
    duckdb::unique_ptr<duckdb::MaterializedQueryResult> result;
    duckdb::ClientProperties props;
    std::string error;  // last get_next/get_schema failure, for get_last_error
};

inline int get_schema(ArrowArrayStream* self, ArrowSchema* out) {
    auto* st = static_cast<State*>(self->private_data);
    try {
        duckdb::ArrowConverter::ToArrowSchema(out, st->result->types,
                                              st->result->names, st->props);
        return 0;
    } catch (const std::exception& e) {
        st->error = e.what();
        return EIO;
    }
}

inline int get_next(ArrowArrayStream* self, ArrowArray* out) {
    auto* st = static_cast<State*>(self->private_data);
    try {
        auto chunk = st->result->Fetch();
        if (!chunk || chunk->size() == 0) {
            out->release = nullptr;  // spec: released empty array = stream end
            return 0;
        }
        duckdb::ArrowConverter::ToArrowArray(*chunk, out, st->props, {});
        return 0;
    } catch (const std::exception& e) {
        st->error = e.what();
        return EIO;
    }
}

inline const char* get_last_error(ArrowArrayStream* self) {
    auto* st = static_cast<State*>(self->private_data);
    return st->error.empty() ? nullptr : st->error.c_str();
}

inline void release(ArrowArrayStream* self) {
    if (!self->release) return;
    delete static_cast<State*>(self->private_data);
    self->private_data = nullptr;
    self->release = nullptr;  // spec: release must null itself
}

// Hand a materialized result out as a live ArrowArrayStream. Caller must have
// already verified `result` is a non-error MaterializedQueryResult; `props`
// come from the originating connection's context. On return `out` is a live
// stream owning `result`.
inline void fill(duckdb::unique_ptr<duckdb::MaterializedQueryResult> result,
                 duckdb::ClientProperties props, ArrowArrayStream* out) {
    auto* st = new State;
    st->result = std::move(result);
    st->props  = std::move(props);
    out->get_schema     = &get_schema;
    out->get_next       = &get_next;
    out->get_last_error = &get_last_error;
    out->release        = &release;
    out->private_data   = st;
}

}  // namespace arrow_stream
}  // namespace caliper_host
