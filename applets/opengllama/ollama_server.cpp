#include "ollama_server.h"
#include "opengllama.h"

#include <httplib.h>
#include <llama.h>

#include <cstdio>
#include <ctime>
#include <sstream>

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned char)c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static std::string iso8601_now() {
    time_t t = time(nullptr);
    struct tm tm;
    gmtime_r(&t, &tm);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

static std::string json_unescape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            switch (s[i + 1]) {
                case '"':  out += '"';  ++i; break;
                case '\\': out += '\\'; ++i; break;
                case 'n':  out += '\n'; ++i; break;
                case 'r':  out += '\r'; ++i; break;
                case 't':  out += '\t'; ++i; break;
                case '/':  out += '/';  ++i; break;
                default:   out += s[i]; break;
            }
        } else {
            out += s[i];
        }
    }
    return out;
}

static std::string parse_json_string(const std::string& body, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    auto pos = body.find(needle);
    if (pos == std::string::npos) return "";
    pos = body.find('"', pos + needle.size());
    if (pos == std::string::npos) return "";
    pos++;
    auto end = body.find('"', pos);
    while (end != std::string::npos && end > 0 && body[end - 1] == '\\')
        end = body.find('"', end + 1);
    if (end == std::string::npos) return "";
    return json_unescape(body.substr(pos, end - pos));
}

static int parse_json_int(const std::string& body, const std::string& key, int default_val) {
    std::string needle = "\"" + key + "\"";
    auto pos = body.find(needle);
    if (pos == std::string::npos) return default_val;
    pos += needle.size();
    while (pos < body.size() && (body[pos] == ' ' || body[pos] == ':' || body[pos] == '\t'))
        pos++;
    size_t start = pos;
    while (pos < body.size() && (body[pos] >= '0' && body[pos] <= '9'))
        pos++;
    if (pos == start) return default_val;
    return std::atoi(body.substr(start, pos - start).c_str());
}

static bool parse_json_bool(const std::string& body, const std::string& key, bool default_val) {
    std::string needle = "\"" + key + "\"";
    auto pos = body.find(needle);
    if (pos == std::string::npos) return default_val;
    pos += needle.size();
    while (pos < body.size() && (body[pos] == ' ' || body[pos] == ':' || body[pos] == '\t'))
        pos++;
    if (pos + 4 <= body.size() && body.substr(pos, 4) == "true") return true;
    if (pos + 5 <= body.size() && body.substr(pos, 5) == "false") return false;
    return default_val;
}

struct ParsedMessage {
    std::string role;
    std::string content;
};

static std::vector<ParsedMessage> parse_messages(const std::string& body) {
    std::vector<ParsedMessage> msgs;
    size_t pos = 0;
    while (true) {
        auto role_pos = body.find("\"role\"", pos);
        if (role_pos == std::string::npos) break;

        std::string role = parse_json_string(body.substr(role_pos), "role");
        auto content_pos = body.find("\"content\"", role_pos);
        if (content_pos == std::string::npos) break;

        std::string content = parse_json_string(body.substr(content_pos), "content");
        msgs.push_back({role, content});
        pos = content_pos + 1;
    }
    return msgs;
}

static std::string apply_chat_template(const llama_model* model,
                                       const std::vector<ParsedMessage>& msgs) {
    std::vector<llama_chat_message> chat(msgs.size());
    for (size_t i = 0; i < msgs.size(); ++i) {
        chat[i].role = msgs[i].role.c_str();
        chat[i].content = msgs[i].content.c_str();
    }

    const char* tmpl = llama_model_chat_template(model, nullptr);
    std::string tmpl_fallback;
    if (!tmpl) {
        char desc[256];
        llama_model_desc(model, desc, sizeof(desc));
        std::string d(desc);
        if (d.find("gptoss") != std::string::npos || d.find("gpt-oss") != std::string::npos)
            tmpl_fallback = "gptoss";
        else
            tmpl_fallback = "chatml";
        tmpl = tmpl_fallback.c_str();
    }

    std::vector<char> buf(4096);
    int32_t n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true,
                                          buf.data(), (int32_t)buf.size());
    if (n > (int32_t)buf.size()) {
        buf.resize(n + 1);
        n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true,
                                      buf.data(), (int32_t)buf.size());
    }
    if (n < 0) {
        std::string result;
        for (auto& m : msgs) {
            if (m.role == "system") result += m.content + "\n";
            else if (m.role == "user") result += "User: " + m.content + "\n";
            else if (m.role == "assistant") result += "Assistant: " + m.content + "\n";
        }
        result += "Assistant:";
        return result;
    }
    return std::string(buf.data(), n);
}


OllamaServer::OllamaServer(OpenGllamaApplet* applet) : applet_(applet) {}

OllamaServer::~OllamaServer() {
    stop();
}

void OllamaServer::start(int port) {
    if (running_) return;

    port_ = port;
    server_ = new httplib::Server();
    setup_routes();

    running_ = true;
    server_thread_ = std::thread([this]() {
        std::fprintf(stderr, "[ollama-server] listening on 0.0.0.0:%d\n", port_);
        server_->listen("0.0.0.0", port_);
        running_ = false;
    });
}

void OllamaServer::stop() {
    if (server_) {
        server_->stop();
    }
    if (server_thread_.joinable()) server_thread_.join();
    delete server_;
    server_ = nullptr;
    running_ = false;
}

void OllamaServer::setup_routes() {
    server_->Get("/api/tags", [this](const httplib::Request&, httplib::Response& res) {
        std::string model_name = "opengllama";
        std::string now = iso8601_now();

        if (!applet_->is_model_loaded()) {
            res.set_content("{\"models\":[]}", "application/json");
            return;
        }

        std::string path = applet_->model_path();
        size_t slash = path.find_last_of('/');
        if (slash != std::string::npos) model_name = path.substr(slash + 1);
        size_t dot = model_name.find_last_of('.');
        if (dot != std::string::npos) model_name = model_name.substr(0, dot);

        std::string json = "{\"models\":[{"
            "\"name\":\"" + json_escape(model_name) + ":latest\","
            "\"model\":\"" + json_escape(model_name) + ":latest\","
            "\"modified_at\":\"" + now + "\","
            "\"size\":0,"
            "\"digest\":\"\","
            "\"details\":{\"format\":\"gguf\",\"family\":\"llama\"}"
            "}]}";
        res.set_content(json, "application/json");
    });

    server_->Post("/api/generate", [this](const httplib::Request& req, httplib::Response& res) {
        if (!applet_->is_model_loaded()) {
            res.status = 503;
            res.set_content("{\"error\":\"no model loaded\"}", "application/json");
            return;
        }
        if (request_active_.exchange(true)) {
            res.status = 503;
            res.set_content("{\"error\":\"inference already in progress\"}", "application/json");
            return;
        }

        std::string prompt = parse_json_string(req.body, "prompt");
        bool stream = !parse_json_bool(req.body, "stream", true) ? false : true;

        // Parse num_predict from options object (Ollama API convention)
        int num_predict = parse_json_int(req.body, "num_predict", 0);
        if (num_predict <= 0) num_predict = 2048;  // safety cap for non-streaming
        int saved_max_tokens = applet_->max_tokens_;
        applet_->max_tokens_ = num_predict;

        std::fprintf(stderr, "[ollama-server] /api/generate: prompt=%zu chars, stream=%s, num_predict=%d\n",
            prompt.size(), stream ? "true" : "false", num_predict);

        if (prompt.empty()) {
            request_active_ = false;
            res.status = 400;
            res.set_content("{\"error\":\"empty prompt\"}", "application/json");
            return;
        }

        std::string model_name = "opengllama";
        {
            std::string p = applet_->model_path();
            size_t s = p.find_last_of('/');
            if (s != std::string::npos) model_name = p.substr(s + 1);
            size_t d = model_name.find_last_of('.');
            if (d != std::string::npos) model_name = model_name.substr(0, d);
        }

        if (!stream) {
            std::string full_response;
            applet_->run_inference_blocking(prompt, [&](const std::string& piece) {
                full_response += piece;
                return true;
            });

            std::string json = "{\"model\":\"" + json_escape(model_name) + ":latest\","
                "\"created_at\":\"" + iso8601_now() + "\","
                "\"response\":\"" + json_escape(full_response) + "\","
                "\"done\":true}";
            res.set_content(json, "application/json");
            applet_->max_tokens_ = saved_max_tokens;
            request_active_ = false;
            return;
        }

        auto* active_flag = &request_active_;
        auto applet = applet_;
        int saved_mt = saved_max_tokens;
        std::string mn = model_name;

        res.set_chunked_content_provider("application/x-ndjson",
            [applet, prompt, mn, active_flag, saved_mt](size_t, httplib::DataSink& sink) -> bool {
                applet->run_inference_blocking(prompt, [&](const std::string& piece) {
                    std::string line = "{\"model\":\"" + json_escape(mn) + ":latest\","
                        "\"created_at\":\"" + iso8601_now() + "\","
                        "\"response\":\"" + json_escape(piece) + "\","
                        "\"done\":false}\n";
                    return sink.write(line.data(), line.size());
                });

                std::string done_line = "{\"model\":\"" + json_escape(mn) + ":latest\","
                    "\"created_at\":\"" + iso8601_now() + "\","
                    "\"response\":\"\","
                    "\"done\":true}\n";
                sink.write(done_line.data(), done_line.size());
                sink.done();
                applet->max_tokens_ = saved_mt;
                *active_flag = false;
                return true;
            });
    });

    server_->Post("/api/chat", [this](const httplib::Request& req, httplib::Response& res) {
        if (!applet_->is_model_loaded()) {
            res.status = 503;
            res.set_content("{\"error\":\"no model loaded\"}", "application/json");
            return;
        }
        if (request_active_.exchange(true)) {
            res.status = 503;
            res.set_content("{\"error\":\"inference already in progress\"}", "application/json");
            return;
        }

        auto msgs = parse_messages(req.body);
        bool stream = !parse_json_bool(req.body, "stream", true) ? false : true;

        if (msgs.empty()) {
            request_active_ = false;
            res.status = 400;
            res.set_content("{\"error\":\"no messages\"}", "application/json");
            return;
        }

        std::string prompt = apply_chat_template(applet_->model_, msgs);

        int num_predict = parse_json_int(req.body, "num_predict", 0);
        if (num_predict <= 0) num_predict = 2048;
        int saved_max_tokens = applet_->max_tokens_;
        applet_->max_tokens_ = num_predict;

        std::string model_name = "opengllama";
        {
            std::string p = applet_->model_path();
            size_t s = p.find_last_of('/');
            if (s != std::string::npos) model_name = p.substr(s + 1);
            size_t d = model_name.find_last_of('.');
            if (d != std::string::npos) model_name = model_name.substr(0, d);
        }

        if (!stream) {
            std::string full_response;
            applet_->run_inference_blocking(prompt, [&](const std::string& piece) {
                full_response += piece;
                return true;
            });

            std::string json = "{\"model\":\"" + json_escape(model_name) + ":latest\","
                "\"created_at\":\"" + iso8601_now() + "\","
                "\"message\":{\"role\":\"assistant\",\"content\":\"" + json_escape(full_response) + "\"},"
                "\"done\":true}";
            res.set_content(json, "application/json");
            applet_->max_tokens_ = saved_max_tokens;
            request_active_ = false;
            return;
        }

        auto* active_flag = &request_active_;
        auto applet = applet_;
        int saved_mt = saved_max_tokens;
        std::string mn = model_name;

        res.set_chunked_content_provider("application/x-ndjson",
            [applet, prompt, mn, active_flag, saved_mt](size_t, httplib::DataSink& sink) -> bool {
                applet->run_inference_blocking(prompt, [&](const std::string& piece) {
                    std::string line = "{\"model\":\"" + json_escape(mn) + ":latest\","
                        "\"created_at\":\"" + iso8601_now() + "\","
                        "\"message\":{\"role\":\"assistant\",\"content\":\"" + json_escape(piece) + "\"},"
                        "\"done\":false}\n";
                    return sink.write(line.data(), line.size());
                });

                std::string done_line = "{\"model\":\"" + json_escape(mn) + ":latest\","
                    "\"created_at\":\"" + iso8601_now() + "\","
                    "\"message\":{\"role\":\"assistant\",\"content\":\"\"},"
                    "\"done\":true}\n";
                sink.write(done_line.data(), done_line.size());
                sink.done();
                applet->max_tokens_ = saved_mt;
                *active_flag = false;
                return true;
            });
    });
}
