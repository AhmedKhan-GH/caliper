#include "catalog_providers/feed_catalog.h"

#include "catalog_providers/catalog_util.h"
#include "feed_store.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

namespace caliper_host {

using namespace caliper_builder;
using catalog_detail::float_cell;

namespace {

std::vector<CatalogColumn> feed_columns() { return {{"seq", "BIGINT"}, {"t_ns", "BIGINT"}, {"value", "FLOAT"}}; }

// The text of a fixed, null-terminated ABI field.
std::string field(const char* s, std::size_t cap) { return std::string(s, strnlen(s, cap)); }

std::uint64_t buffered(std::uint32_t capacity, std::uint64_t newest) {
    return std::min<std::uint64_t>(newest, capacity);
}

}  // namespace

FeedCatalog::FeedCatalog(FeedStore& store) : store_(store) {}

std::int64_t FeedCatalog::steady_now_ns() {
#if defined(__APPLE__)
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    return static_cast<std::int64_t>(static_cast<__uint128_t>(mach_absolute_time()) * tb.numer / tb.denom);
#else
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
#endif
}

bool FeedCatalog::channel_of(const std::string& id, std::string* channel, std::uint32_t* index) const {
    const CatalogIdParse parsed = parse_catalog_id(id);
    if (parsed.error || parsed.kind != CatalogKind::feed) return false;
    const std::uint32_t n = store_.channel_count();
    for (std::uint32_t i = 0; i < n; ++i) {
        CaliperFeedChannelInfo info{};
        info.struct_size = sizeof info;
        if (!store_.channel_info(i, &info)) continue;
        if (field(info.id, sizeof info.id) == parsed.name) {
            *channel = parsed.name;
            *index = i;
            return true;
        }
    }
    return false;
}

bool FeedCatalog::entry_of(std::uint32_t index, CatalogEntry* out) const {
    CaliperFeedChannelInfo info{};
    info.struct_size = sizeof info;
    if (!store_.channel_info(index, &info)) return false;
    const std::string id = field(info.id, sizeof info.id);
    if (!catalog_detail::is_catalog_name(id)) return false;
    std::uint32_t capacity = 0;
    std::uint64_t newest = 0;
    if (!store_.ring_state(id, &capacity, &newest)) return false;
    CatalogEntry& e = *out;
    e = CatalogEntry{};
    e.kind = CatalogKind::feed;
    e.name = id;
    e.id = make_catalog_id(e.kind, id);
    e.title = catalog_detail::title_or(field(info.name, sizeof info.name), id);
    e.origin.service = "feed";
    e.origin.channel = catalog_detail::origin_value(id);
    e.origin.units = catalog_detail::origin_value(field(info.units, sizeof info.units));
    e.columns = feed_columns();
    e.rows = buffered(capacity, newest);
    if (info.nominal_hz > 0.0f && info.nominal_hz <= CatalogLimits::max_rate_hz)
        e.rate_hz = static_cast<double>(info.nominal_hz);
    e.wire = false;
    if (newest > 0) {
        CaliperFeedSample last{};
        std::uint64_t cursor = 0;
        if (store_.read(id.c_str(), &last, 1, &cursor) == 1) {
            const std::int64_t age = steady_now_ns() - last.t_ns;
            if (age >= 0)
                e.updated_at = catalog_detail::utc_text_ms(std::chrono::system_clock::now() -
                                                           std::chrono::duration_cast<std::chrono::system_clock::duration>(
                                                               std::chrono::nanoseconds(age)));
        }
    }
    return true;
}

CatalogListResult FeedCatalog::list(const CatalogFilter& filter) {
    if (filter.kind && *filter.kind != CatalogKind::feed) return {};
    std::vector<CatalogEntry> all;
    const std::uint32_t n = store_.channel_count();
    for (std::uint32_t i = 0; i < n; ++i) {
        CatalogEntry e;
        if (entry_of(i, &e)) all.push_back(std::move(e));
    }
    std::sort(all.begin(), all.end(), [](const CatalogEntry& a, const CatalogEntry& b) { return a.id < b.id; });
    return catalog_detail::filter_entries(all, filter);
}

CatalogDescribeResult FeedCatalog::describe(const std::string& id) {
    CatalogDescribeResult out;
    std::string channel;
    std::uint32_t index = 0;
    CatalogEntry e;
    if (channel_of(id, &channel, &index) && entry_of(index, &e)) out.entry = std::move(e);
    return out;
}

CatalogSampleResult FeedCatalog::sample(const std::string& id, const CatalogSampleRequest& request) {
    CatalogSampleResult out;
    std::string channel;
    std::uint32_t index = 0;
    std::uint32_t capacity = 0;
    std::uint64_t newest = 0;
    if (!channel_of(id, &channel, &index) || !store_.ring_state(channel, &capacity, &newest)) {
        out.error = Error("catalog.not_found", "No feed channel has the id " + id + ".", "id");
        return out;
    }
    const std::uint32_t limit = static_cast<std::uint32_t>(std::clamp<std::size_t>(request.limit, 1, 1000));
    std::vector<CaliperFeedSample> buf(limit);
    std::uint64_t cursor = 0;
    const std::uint32_t n = store_.read(channel.c_str(), buf.data(), limit, &cursor);
    CatalogSample& s = out.sample;
    s.id = id;
    s.kind = CatalogKind::feed;
    s.columns = feed_columns();
    s.rows = nlohmann::json::array();
    for (std::uint32_t i = 0; i < n; ++i)
        s.rows->push_back(nlohmann::json::array({buf[i].seq, buf[i].t_ns, float_cell(buf[i].value)}));
    const std::uint64_t total = std::max<std::uint64_t>(buffered(capacity, newest), n);
    s.total_rows = total;
    s.truncated = total > n;
    return out;
}

std::uint64_t FeedCatalog::revision() const { return 1 + store_.channel_count(); }

std::uint64_t FeedCatalog::generation(const std::string& id) {
    const CatalogIdParse parsed = parse_catalog_id(id);
    if (parsed.error || parsed.kind != CatalogKind::feed) return 0;
    std::uint64_t newest = 0;
    return store_.ring_state(parsed.name, nullptr, &newest) ? newest : 0;
}

ServedData FeedCatalog::read(const std::string& id, const ServedRequest& request) {
    ServedData out;
    const CatalogIdParse parsed = parse_catalog_id(id);
    std::uint32_t capacity = 0;
    std::uint64_t newest = 0;
    if (parsed.error || parsed.kind != CatalogKind::feed || !store_.ring_state(parsed.name, &capacity, &newest)) {
        out.error = Error("resolve.unavailable", "No feed channel has the id " + id + ".");
        return out;
    }
    if (newest == 0) {
        out.error = Error("resolve.unavailable", "The feed channel " + id + " has no samples yet.");
        return out;
    }
    const std::uint64_t held = buffered(capacity, newest);
    const std::uint64_t want = std::max<std::uint64_t>(
        1, std::min<std::uint64_t>({held, request.max_rows, request.max_cells / feed_columns().size()}));
    std::vector<CaliperFeedSample> buf(static_cast<std::size_t>(want));
    std::uint64_t cursor = 0;
    const std::uint32_t n = store_.read(parsed.name.c_str(), buf.data(), static_cast<std::uint32_t>(want), &cursor);
    out.tabular = true;
    out.columns = {{"seq", "BIGINT"}, {"t_ns", "BIGINT"}, {"value", "FLOAT"}};
    for (std::uint32_t i = 0; i < n; ++i)
        out.rows.push_back(nlohmann::json::array({buf[i].seq, buf[i].t_ns, float_cell(buf[i].value)}));
    out.generation = n > 0 ? buf[n - 1].seq : newest;
    out.truncated = held > n;
    return out;
}

}  // namespace caliper_host
