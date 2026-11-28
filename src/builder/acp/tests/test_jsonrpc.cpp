// The JSON-RPC peer alone, over an in-process pipe pair. One side is a peer;
// the other is either a second peer or the raw channel, so a test can write
// any bytes at all.
#include <caliper_builder/acp/jsonrpc.h>

#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#ifndef _WIN32

using namespace caliper_builder::acp;
using caliper_builder::Error;
using nlohmann::json;

namespace {

void write_raw(ByteChannel& ch, const std::string& text) {
    std::size_t done = 0;
    while (done < text.size()) {
        std::size_t n = 0;
        const IoStatus st = ch.write(text.data() + done, text.size() - done, &n, 1000);
        REQUIRE((st == IoStatus::ok || st == IoStatus::timeout));
        done += n;
    }
}

// Reads one line from a raw channel.
std::string read_line(ByteChannel& ch, int timeout_ms = 2000) {
    std::string line;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    char c = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        std::size_t got = 0;
        const IoStatus st = ch.read(&c, 1, &got, 50);
        if (st == IoStatus::ok) {
            if (c == '\n') return line;
            line += c;
        } else if (st != IoStatus::timeout) {
            break;
        }
    }
    return line;
}

struct Faults {
    std::mutex m;
    std::vector<Error> list;
    void add(const Error& e) {
        std::lock_guard<std::mutex> lock(m);
        list.push_back(e);
    }
    int count(const std::string& code) {
        std::lock_guard<std::mutex> lock(m);
        int n = 0;
        for (const auto& e : list) n += e.code == code;
        return n;
    }
};

template <class Pred>
bool eventually(Pred pred, int timeout_ms = 3000) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return pred();
}

}  // namespace

TEST_CASE("two peers call, notify and answer each other") {
    ChannelPair pair = make_pipe_channel_pair();
    REQUIRE(pair.error.ok());
    JsonRpcPeer a(std::move(pair.a));
    JsonRpcPeer b(std::move(pair.b));
    std::atomic<int> notes{0};
    b.on_request("echo", [&](const json& id, const json& params) { b.respond(id, params); });
    b.on_request("fail", [&](const json& id, const json&) {
        b.respond_error(id, rpc_code::invalid_params, "bad params", json{{"field", "x"}});
    });
    b.on_notification("note", [&](const json& p) { notes += p.value("n", 0); });
    a.start();
    b.start();

    RpcResult r = a.call_sync("echo", json{{"text", "line one\nline two"}, {"n", 3}}, 2000);
    REQUIRE(r.ok());
    CHECK(r.result["text"] == "line one\nline two");
    CHECK(r.result["n"] == 3);

    r = a.call_sync("fail", json::object(), 2000);
    CHECK(r.error.code == "rpc.remote_error");
    CHECK(r.rpc_code == rpc_code::invalid_params);
    CHECK(r.error.message == "bad params");
    CHECK(r.error.where == "fail");
    CHECK(r.error_data["field"] == "x");

    r = a.call_sync("no_such_method", json::object(), 2000);
    CHECK(r.error.code == "rpc.remote_error");
    CHECK(r.rpc_code == rpc_code::method_not_found);

    CHECK(a.notify("note", json{{"n", 2}}).ok());
    CHECK(a.notify("note", json{{"n", 5}}).ok());
    CHECK(a.notify("unknown_note", json{{"n", 100}}).ok());
    CHECK(eventually([&] { return notes == 7; }));
}

TEST_CASE("a handler may answer later from another thread") {
    ChannelPair pair = make_pipe_channel_pair();
    REQUIRE(pair.error.ok());
    JsonRpcPeer a(std::move(pair.a));
    JsonRpcPeer b(std::move(pair.b));
    std::mutex m;
    json saved_id;
    b.on_request("later", [&](const json& id, const json&) {
        std::lock_guard<std::mutex> lock(m);
        saved_id = id;
    });
    a.start();
    b.start();
    std::atomic<bool> done{false};
    RpcResult got;
    a.call_async("later", json::object(), 5000, [&](RpcResult r) {
        got = std::move(r);
        done = true;
    });
    REQUIRE(eventually([&] {
        std::lock_guard<std::mutex> lock(m);
        return !saved_id.is_null();
    }));
    std::thread([&] {
        std::lock_guard<std::mutex> lock(m);
        b.respond(saved_id, json{{"answer", 42}});
    }).join();
    REQUIRE(eventually([&] { return done.load(); }));
    CHECK(got.result["answer"] == 42);
    b.close();
    a.close();
}

TEST_CASE("calls time out, deadlines move and late answers are reported") {
    ChannelPair pair = make_pipe_channel_pair();
    REQUIRE(pair.error.ok());
    JsonRpcPeer a(std::move(pair.a));
    Faults faults;
    a.on_fault([&](const Error& e) { faults.add(e); });
    JsonRpcPeer b(std::move(pair.b));
    std::mutex m;
    std::vector<json> ids;
    b.on_request("slow", [&](const json& id, const json&) {
        std::lock_guard<std::mutex> lock(m);
        ids.push_back(id);
    });
    a.start();
    b.start();

    const int timeout_ms = 150;
    const auto t0 = std::chrono::steady_clock::now();
    RpcResult r = a.call_sync("slow", json::object(), timeout_ms);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    CHECK(r.error.code == "rpc.timeout");
    // The peer expires a call when its steady clock reaches the deadline, thus
    // never before the timeout.
    CHECK(ms >= timeout_ms);
    CHECK(ms < 2000);

    // A late answer to the expired call is dropped and reported.
    {
        std::lock_guard<std::mutex> lock(m);
        REQUIRE(ids.size() == 1);
        b.respond(ids[0], json::object());
    }
    CHECK(eventually([&] { return faults.count("rpc.unknown_response") == 1; }));

    // A call without a limit gets one later, as cancel() does for a prompt.
    std::atomic<bool> done{false};
    RpcResult got;
    const std::int64_t id = a.call_async("slow", json::object(), 0, [&](RpcResult rr) {
        got = std::move(rr);
        done = true;
    });
    REQUIRE(id > 0);
    // The peer has the call, and the call is still pending. The reader loop
    // sweeps expired calls once a poll, thus a wait of two polls gives the
    // sweep the time to run before the check.
    CHECK(eventually([&] {
        std::lock_guard<std::mutex> lock(m);
        return ids.size() == 2;
    }));
    std::this_thread::sleep_for(std::chrono::milliseconds(2 * JsonRpcOptions{}.poll_ms));
    CHECK_FALSE(done.load());
    CHECK(a.set_call_timeout(id, 100));
    CHECK(eventually([&] { return done.load(); }));
    CHECK(got.error.code == "rpc.timeout");
    CHECK_FALSE(a.set_call_timeout(id, 100));
    // Stop the handlers before the state they write to goes away.
    b.close();
    a.close();
}

TEST_CASE("closing fails pending calls and refuses new ones") {
    ChannelPair pair = make_pipe_channel_pair();
    REQUIRE(pair.error.ok());
    auto a = std::make_unique<JsonRpcPeer>(std::move(pair.a));
    JsonRpcPeer b(std::move(pair.b));
    b.on_request("never", [](const json&, const json&) {});
    std::atomic<int> closes{0};
    Error close_reason;
    a->on_close([&](const Error& e) {
        close_reason = e;
        ++closes;
    });
    a->start();
    b.start();
    std::atomic<bool> done{false};
    RpcResult got;
    a->call_async("never", json::object(), 0, [&](RpcResult r) {
        got = std::move(r);
        done = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    a->close();
    CHECK(done.load());
    CHECK(got.error.code == "rpc.closed");
    CHECK(closes == 1);
    CHECK(a->closed());
    CHECK(a->notify("x", json::object()).code == "rpc.closed");
    RpcResult after = a->call_sync("never", json::object(), 100);
    CHECK(after.error.code == "rpc.closed");
    a->close();  // twice is harmless
    CHECK(closes == 1);
    a.reset();
}

TEST_CASE("the other side closing ends the connection") {
    ChannelPair pair = make_pipe_channel_pair();
    REQUIRE(pair.error.ok());
    JsonRpcPeer a(std::move(pair.a));
    std::atomic<bool> closed{false};
    std::atomic<bool> failed{false};
    a.on_close([&](const Error&) { closed = true; });
    a.start();
    a.call_async("pending", json::object(), 0, [&](RpcResult r) { failed = r.error.code == "rpc.closed"; });
    pair.b->close_write();
    pair.b.reset();
    CHECK(eventually([&] { return closed.load() && failed.load(); }));
}

TEST_CASE("a last line without a newline is read at the end of file") {
    ChannelPair pair = make_pipe_channel_pair();
    REQUIRE(pair.error.ok());
    JsonRpcPeer a(std::move(pair.a));
    std::atomic<int> got{0};
    a.on_notification("n", [&](const json& p) { got = p.value("v", 0); });
    a.start();
    write_raw(*pair.b, R"({"jsonrpc":"2.0","method":"n","params":{"v":7}})");
    pair.b->close_write();
    CHECK(eventually([&] { return got == 7; }));
}

TEST_CASE("hostile lines are dropped and reported, and the peer carries on") {
    ChannelPair pair = make_pipe_channel_pair();
    REQUIRE(pair.error.ok());
    JsonRpcOptions opt;
    opt.max_message_bytes = 1000;
    opt.max_depth = 32;
    JsonRpcPeer a(std::move(pair.a), opt);
    Faults faults;
    a.on_fault([&](const Error& e) { faults.add(e); });
    std::atomic<int> good{0};
    a.on_notification("good", [&](const json&) { ++good; });
    a.start();
    ByteChannel& raw = *pair.b;

    write_raw(raw, "{not json\n");
    write_raw(raw, "\xff\xfe{\"a\":1}\n");
    write_raw(raw, "42\n");
    write_raw(raw, "[{\"jsonrpc\":\"2.0\",\"method\":\"good\"}]\n");
    write_raw(raw, "{\"jsonrpc\":\"1.0\",\"method\":\"good\"}\n");
    write_raw(raw, "{\"method\":\"good\"}\n");
    write_raw(raw, "{\"jsonrpc\":\"2.0\",\"id\":1.5,\"method\":\"good\"}\n");
    write_raw(raw, "{\"jsonrpc\":\"2.0\",\"id\":{\"x\":1},\"method\":\"good\"}\n");
    write_raw(raw, "{\"jsonrpc\":\"2.0\",\"method\":\"good\",\"params\":\"text\"}\n");
    write_raw(raw, "{\"jsonrpc\":\"2.0\",\"id\":77,\"result\":{}}\n");
    write_raw(raw, "{\"jsonrpc\":\"2.0\",\"id\":\"s\",\"result\":{}}\n");
    write_raw(raw, "{\"jsonrpc\":\"2.0\",\"id\":5,\"result\":{},\"error\":{}}\n");
    write_raw(raw, std::string(40, '[') + std::string(40, ']') + "\n");
    write_raw(raw, "{\"jsonrpc\":\"2.0\",\"method\":\"good\",\"params\":{\"s\":\"" + std::string(5000, 'x') + "\"}}\n");
    write_raw(raw, "\n   \n\r\n");
    write_raw(raw, "{\"jsonrpc\":\"2.0\",\"method\":\"good\"}\r\n");

    CHECK(eventually([&] { return good == 1; }));
    CHECK(faults.count("rpc.parse_error") == 2);
    CHECK(faults.count("rpc.invalid_message") == 8);
    CHECK(faults.count("rpc.unknown_response") == 2);
    CHECK(faults.count("rpc.too_deep") == 1);
    CHECK(faults.count("rpc.message_too_large") == 1);
    CHECK_FALSE(a.closed());
}

TEST_CASE("a request with a bad method or params gets an error answer") {
    ChannelPair pair = make_pipe_channel_pair();
    REQUIRE(pair.error.ok());
    JsonRpcPeer a(std::move(pair.a));
    a.on_request("thing", [&](const json& id, const json&) { a.respond(id, json::object()); });
    a.on_request("throws", [](const json&, const json&) { throw std::runtime_error("boom"); });
    a.start();
    ByteChannel& raw = *pair.b;

    write_raw(raw, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":7}\n");
    json r = json::parse(read_line(raw));
    CHECK(r["id"] == 1);
    CHECK(r["error"]["code"] == rpc_code::invalid_request);

    write_raw(raw, "{\"jsonrpc\":\"2.0\",\"id\":\"two\",\"method\":\"thing\",\"params\":5}\n");
    r = json::parse(read_line(raw));
    CHECK(r["id"] == "two");
    CHECK(r["error"]["code"] == rpc_code::invalid_params);

    write_raw(raw, "{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"absent\"}\n");
    r = json::parse(read_line(raw));
    CHECK(r["error"]["code"] == rpc_code::method_not_found);

    write_raw(raw, "{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"throws\"}\n");
    r = json::parse(read_line(raw));
    CHECK(r["error"]["code"] == rpc_code::internal_error);

    write_raw(raw, "{\"jsonrpc\":\"2.0\",\"id\":5,\"method\":\"thing\"}\n");
    r = json::parse(read_line(raw));
    CHECK(r["id"] == 5);
    CHECK(r["result"].is_object());
}

TEST_CASE("an oversized line split across many reads is dropped without keeping it") {
    ChannelPair pair = make_pipe_channel_pair();
    REQUIRE(pair.error.ok());
    JsonRpcOptions opt;
    opt.max_message_bytes = 4096;
    JsonRpcPeer a(std::move(pair.a), opt);
    Faults faults;
    a.on_fault([&](const Error& e) { faults.add(e); });
    std::atomic<int> good{0};
    a.on_notification("good", [&](const json&) { ++good; });
    a.start();
    std::thread writer([&] {
        const std::string chunk(32 * 1024, 'y');
        for (int i = 0; i < 64; ++i) write_raw(*pair.b, chunk);  // 2 MiB, no newline yet
        write_raw(*pair.b, "\n{\"jsonrpc\":\"2.0\",\"method\":\"good\"}\n");
    });
    writer.join();
    CHECK(eventually([&] { return good == 1; }));
    CHECK(faults.count("rpc.message_too_large") == 1);
}

TEST_CASE("a message written in pieces arrives once") {
    ChannelPair pair = make_pipe_channel_pair();
    REQUIRE(pair.error.ok());
    JsonRpcPeer a(std::move(pair.a));
    std::atomic<int> got{0};
    a.on_notification("piece", [&](const json& p) { got += p.value("v", 0); });
    a.start();
    const std::string msg = "{\"jsonrpc\":\"2.0\",\"method\":\"piece\",\"params\":{\"v\":1}}\n";
    for (char c : msg) {
        write_raw(*pair.b, std::string(1, c));
        if (c == ',') std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    CHECK(eventually([&] { return got == 1; }));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(got == 1);
}

TEST_CASE("outgoing messages are single lines and invalid UTF-8 is replaced") {
    ChannelPair pair = make_pipe_channel_pair();
    REQUIRE(pair.error.ok());
    JsonRpcPeer a(std::move(pair.a));
    a.start();
    CHECK(a.notify("text", json{{"s", "one\ntwo\r\nthree"}, {"bad", std::string("\xff\xfe ok")}}).ok());
    const std::string line = read_line(*pair.b);
    CHECK(line.find('\n') == std::string::npos);
    json j = json::parse(line);
    CHECK(j["params"]["s"] == "one\ntwo\r\nthree");
    CHECK(j["params"]["bad"].get<std::string>().find(" ok") != std::string::npos);
}

TEST_CASE("a handler cannot wait for a call on the reader thread") {
    ChannelPair pair = make_pipe_channel_pair();
    REQUIRE(pair.error.ok());
    JsonRpcPeer a(std::move(pair.a));
    JsonRpcPeer b(std::move(pair.b));
    std::atomic<bool> refused{false};
    b.on_request("nested", [&](const json& id, const json&) {
        RpcResult r = b.call_sync("anything", json::object(), 100);
        refused = r.error.code == "rpc.bad_argument";
        b.respond(id, json::object());
    });
    a.start();
    b.start();
    CHECK(a.call_sync("nested", json::object(), 2000).ok());
    CHECK(refused.load());
}

TEST_CASE("a peer that stops reading fills the queue, which is then refused") {
    ChannelPair pair = make_pipe_channel_pair();
    REQUIRE(pair.error.ok());
    JsonRpcOptions opt;
    opt.max_queued_bytes = 8192;
    opt.max_pending_calls = 2;
    JsonRpcPeer a(std::move(pair.a), opt);
    a.start();
    const json big = {{"s", std::string(1000, 'z')}};
    // Flood until the queue stays full: the pipe is full too, so nothing drains.
    int refused_in_a_row = 0;
    for (int i = 0; i < 100000 && refused_in_a_row < 10; ++i) {
        const Error e = a.notify("flood", big);
        if (e.ok()) {
            refused_in_a_row = 0;
        } else {
            CHECK(e.code == "rpc.queue_full");
            ++refused_in_a_row;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
    CHECK(refused_in_a_row == 10);
    CHECK_FALSE(a.flush(50));

    // The pending-call limit holds as well.
    int refused = 0;
    for (int i = 0; i < 3; ++i)
        a.call_async("c", json::object(), 0, [&](RpcResult r) { refused += r.error.code == "rpc.queue_full"; });
    CHECK(refused >= 1);
    a.close();
}

TEST_CASE("json_nesting_depth ignores brackets inside strings") {
    CHECK(json_nesting_depth("{\"a\":[1,[2]]}", 100) == 3);
    CHECK(json_nesting_depth("\"[[[[\"", 100) == 0);
    CHECK(json_nesting_depth("{\"a\":\"\\\"[[\"}", 100) == 1);
    CHECK(json_nesting_depth(std::string(500, '['), 10) == 11);
}

#endif  // _WIN32
