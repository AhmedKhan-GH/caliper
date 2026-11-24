// The model runner's viewer connection to one managed run (run_connection.h).
#include "run_connection.h"

#include "net/wire.h"
#include "net/ws.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <variant>

namespace caliper_runner {

namespace wire = caliper_net::wire;
using caliper_net::Json;
using nlohmann::json;

namespace {

Json to_net(const json& v) {
    Json out;
    const std::string text = v.dump();
    std::string err;
    if (!Json::parse(text.data(), text.size(), &out, &err)) return Json();
    return out;
}

json from_net(const Json& v) {
    json out = json::parse(v.dump(), nullptr, false);
    return out.is_discarded() ? json() : out;
}

bool token_char(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
}

}  // namespace

std::string stream_url_of(const std::string& link, std::string* error) {
    auto fail = [&](const std::string& why) {
        if (error) *error = "The stream link " + link.substr(0, link.find('?')) + " is not a link of this computer: " + why;
        return std::string();
    };
    std::string rest;
    if (link.rfind("http://", 0) == 0) {
        rest = link.substr(7);
    } else if (link.rfind("ws://", 0) == 0) {
        rest = link.substr(5);
    } else {
        return fail("it starts with neither http:// nor ws://.");
    }
    const std::size_t slash = rest.find_first_of("/?");
    const std::string authority = rest.substr(0, slash);
    const std::size_t colon = authority.rfind(':');
    if (colon == std::string::npos) return fail("it names no port.");
    const std::string host = authority.substr(0, colon);
    const std::string port = authority.substr(colon + 1);
    if (host != "127.0.0.1" && host != "localhost") return fail("its host is not 127.0.0.1.");
    if (port.empty() || port.size() > 5 || !std::all_of(port.begin(), port.end(), [](char c) { return c >= '0' && c <= '9'; }))
        return fail("its port is not a number.");
    const std::size_t q = link.find("token=");
    if (q == std::string::npos) return fail("it carries no token.");
    std::string token = link.substr(q + 6);
    token = token.substr(0, token.find('&'));
    if (token.size() < 22 || token.size() > 128 || !std::all_of(token.begin(), token.end(), token_char))
        return fail("its token is not valid.");
    return "ws://" + host + ":" + port + CALIPER_NET_PATH + "?token=" + token;
}

RunConnection::RunConnection(std::size_t max_values) : max_values_(std::max<std::size_t>(1, max_values)) {}

RunConnection::~RunConnection() = default;

bool RunConnection::open(const std::string& link, std::string* error) {
    const std::string url = stream_url_of(link, error);
    if (url.empty()) return false;
    interpretation_ = false;
    text_params_ = false;
    controls_ = json();
    interpretation_replies_.clear();
    client_ = std::make_unique<caliper_net::Client>();
    std::string err;
    if (!client_->connect(url, &err)) {
        if (error) *error = "The runner could not connect to the run's stream: " + err;
        client_.reset();
        state_ = State::closed;
        close_reason_ = "The connection failed: " + err;
        return false;
    }
    state_ = State::connecting;
    return true;
}

void RunConnection::wake() {
    if (client_) client_->wake();
}

void RunConnection::close() {
    if (client_ && state_ != State::closed) client_->close();
    client_.reset();
    if (state_ != State::closed) {
        state_ = State::closed;
        if (close_reason_.empty()) close_reason_ = "The runner closed the connection.";
    }
}

void RunConnection::send_bytes(const std::vector<std::uint8_t>& bytes) {
    if (client_) client_->send(bytes.data(), bytes.size(), true);
}

bool RunConnection::poll(int timeout_ms) {
    if (!client_ || state_ == State::closed) return false;
    bool changed = false;
    for (caliper_net::Event& e : client_->poll(timeout_ms)) {
        switch (e.type) {
            case caliper_net::Event::Type::Open: {
                state_ = State::open;
                changed = true;
                wire::Hello hello;
                hello.role = CALIPER_NET_ROLE_VIEWER;
                hello.capabilities = {CALIPER_NET_CAP_CONTROLS, CALIPER_NET_CAP_INTERVENTIONS, CALIPER_NET_CAP_INTERPRETATION, CALIPER_NET_CAP_TEXT_PARAMS};
                Json software = Json::object();
                software.set("name", "caliper-runner");
                software.set("version", "1");
                hello.software = software;
                std::vector<std::uint8_t> bytes;
                if (wire::encode(hello, &bytes)) send_bytes(bytes);
                break;
            }
            case caliper_net::Event::Type::Message:
                if (e.binary) {
                    const State previous_state = state_;
                    const std::size_t before = replies_.size();
                    const auto revision = interpretation_revision_;
                    const json old_controls = controls_;
                    std::uint64_t totals = 0;
                    for (const auto& s : series_) totals += s.second.total;
                    on_message(e.data);
                    std::uint64_t after = 0;
                    for (const auto& s : series_) after += s.second.total;
                    if (state_ != previous_state || interpretation_revision_ != revision ||
                        replies_.size() != before || controls_ != old_controls || after != totals) changed = true;
                }
                break;
            case caliper_net::Event::Type::Closed:
                state_ = State::closed;
                close_reason_ = e.reason.empty() ? "The run's stream closed (code " + std::to_string(e.code) + ")."
                                                 : "The run's stream closed: " + e.reason;
                changed = true;
                break;
            default:
                break;
        }
        if (!client_) break;
    }
    if (state_ == State::closed) client_.reset();
    return changed;
}

void RunConnection::on_message(const std::vector<std::uint8_t>& bytes) {
    wire::Message msg;
    const wire::DecodeResult r = wire::decode(bytes.data(), bytes.size(), &msg);
    if (!r.ok()) {
        const auto* op = r.frame.meta.find("op");
        const bool interpretation = op && op->is_string() && op->as_string() == CALIPER_NET_OP_INTERPRET;
        const bool missing_operation = !op || !op->is_string() || op->as_string().empty();
        if (r.type == CALIPER_NET_MSG_CONTROL_REPLY && (interpretation || missing_operation)) {
            const auto* id = r.frame.meta.find("request_id");
            std::uint64_t request = 0;
            const bool identified = id && id->get_u64(&request) && request <= std::numeric_limits<std::uint32_t>::max();
            // Only a known request ID can identify a reply without an operation.
            if (!identified && !interpretation) return;
            for (auto& [pending_id, entry] : interpretation_replies_) {
                auto& reply = entry.second;
                if ((identified && pending_id != request) || reply.status != CALIPER_NET_STATUS_QUEUED) continue;
                reply.status = CALIPER_NET_STATUS_ERROR;
                reply.error_code = CALIPER_NET_ERR_BAD_REQUEST;
                reply.error_message = "The producer sent an invalid interpretation reply. Check its protocol version.";
                ++interpretation_revision_;
            }
        }
        return;
    }
    if (const auto* w = std::get_if<wire::Welcome>(&msg)) {
        control_allowed_ = w->control.value_or(false);
        interpretation_ = std::find(w->capabilities.begin(), w->capabilities.end(), CALIPER_NET_CAP_INTERPRETATION) != w->capabilities.end();
        text_params_ = std::find(w->capabilities.begin(), w->capabilities.end(), CALIPER_NET_CAP_TEXT_PARAMS) != w->capabilities.end();
        if (const Json* id = w->session.find("id"); id && id->is_string()) session_ = id->as_string();
    } else if (const auto* v = std::get_if<wire::Views>(&msg)) {
        for (const wire::ViewInfo& info : v->list) {
            if (info.kind != CALIPER_NET_KIND_SERIES || subscribed_.count(info.name)) continue;
            wire::Subscribe sub;
            sub.sub = next_sub_++;
            sub.view = info.name;
            sub.request_id = next_request_++;
            std::vector<std::uint8_t> out;
            if (!wire::encode(sub, &out)) continue;
            subscribed_[info.name] = sub.sub;
            sub_views_[sub.sub] = info.name;
            send_bytes(out);
        }
    } else if (const auto* s = std::get_if<wire::Series>(&msg)) {
        const auto it = sub_views_.find(s->sub);
        const std::string name = it != sub_views_.end() ? it->second : s->view;
        if (name.empty() || !s->values.data || s->values.size % sizeof(float) != 0) return;
        const std::size_t m = s->values.size / sizeof(float);
        std::vector<float> vals(m);
        if (m) std::memcpy(vals.data(), s->values.data, s->values.size);
        Series& ser = series_[name];
        std::uint64_t start = s->start;
        std::size_t skip = 0;
        if (ser.values.empty()) ser.first = start;
        std::uint64_t end = ser.first + ser.values.size();
        if (start > end && start - end > max_values_) {   // a gap past what is kept: begin again
            ser.values.clear();
            ser.first = start;
            end = start;
        }
        if (start < ser.first) {
            skip = static_cast<std::size_t>(std::min<std::uint64_t>(m, ser.first - start));
            start = ser.first;
        }
        if (start > end) ser.values.resize(ser.values.size() + (start - end), std::numeric_limits<float>::quiet_NaN());
        for (std::size_t i = skip; i < m; ++i) {
            const std::uint64_t idx = s->start + i;
            if (idx < ser.first + ser.values.size())
                ser.values[static_cast<std::size_t>(idx - ser.first)] = vals[i];
            else
                ser.values.push_back(vals[i]);
        }
        ser.total = std::max<std::uint64_t>(ser.total, std::max<std::uint64_t>(s->total, ser.first + ser.values.size()));
        if (ser.values.size() > max_values_) {
            const std::size_t drop = ser.values.size() - max_values_;
            ser.values.erase(ser.values.begin(), ser.values.begin() + static_cast<std::ptrdiff_t>(drop));
            ser.first += drop;
        }
    } else if (const auto* c = std::get_if<wire::Controls>(&msg)) {
        controls_ = json{{"run", from_net(c->run)}, {"params", from_net(c->params)}, {"actions", from_net(c->actions)}};
        if (!c->interventions.is_null()) controls_["interventions"] = from_net(c->interventions);
    } else if (const auto* cr = std::get_if<wire::ControlReply>(&msg)) {
        Reply rep;
        rep.request_id = cr->request_id;
        rep.op = cr->op;
        rep.status = cr->status;
        rep.name = cr->name.value_or("");
        rep.message = cr->message.value_or("");
        rep.result = from_net(cr->result);
        if (cr->error.is_object()) {
            if (const Json* code = cr->error.find("code"); code && code->is_string()) rep.error_code = code->as_string();
            if (const Json* m = cr->error.find("message"); m && m->is_string()) rep.error_message = m->as_string();
        }
        if (rep.op == CALIPER_NET_OP_INTERPRET) {
            auto it = interpretation_replies_.find(rep.request_id);
            if (it != interpretation_replies_.end() && it->second.second.status == CALIPER_NET_STATUS_QUEUED) {
                it->second.second = rep;
                ++interpretation_revision_;
            }
        } else replies_.push_back(std::move(rep));
    } else if (const auto* p = std::get_if<wire::Ping>(&msg)) {
        std::vector<std::uint8_t> out;
        if (wire::encode(wire::Pong{p->nonce}, &out)) send_bytes(out);
    } else if (const auto* e = std::get_if<wire::Error>(&msg)) {
        if (e->sub) {   // a refused subscription: the next VIEWS tries again
            const auto it = sub_views_.find(*e->sub);
            if (it != sub_views_.end()) {
                subscribed_.erase(it->second);
                sub_views_.erase(it);
            }
        }
        if (e->request_id && e->in_reply_to && *e->in_reply_to == CALIPER_NET_MSG_CONTROL &&
            !e->sub && !e->view && !e->session) {
            Reply rep;
            rep.request_id = *e->request_id;
            rep.status = "error";
            rep.error_code = e->code;
            rep.error_message = e->message;
            auto it = interpretation_replies_.find(rep.request_id);
            if (it != interpretation_replies_.end()) {
                if (it->second.second.status == CALIPER_NET_STATUS_QUEUED) {
                    rep.op = CALIPER_NET_OP_INTERPRET;
                    it->second.second = rep;
                    ++interpretation_revision_;
                }
            } else replies_.push_back(std::move(rep));
        }
    } else if (std::get_if<wire::Bye>(&msg)) {
        close_reason_ = "The run's stream said goodbye.";
        close();
    }
}

std::uint32_t RunConnection::send_control(const std::string& op, const std::string& name, const json& value,
                                          std::uint32_t count, std::string* error) {
    if (!client_ || state_ != State::open) {
        if (error) *error = "The runner's connection to the run is not open.";
        return 0;
    }
    if (op == CALIPER_NET_OP_SET && !text_params_) {
        bool text_parameter = false;
        if (controls_.contains("params"))
            for (const auto& param : controls_["params"])
                if (param.value("name", "") == name && param.value("type", "") == CALIPER_NET_PARAM_TEXT)
                    text_parameter = true;
        if (text_parameter || (value.is_string() &&
            (value.get_ref<const std::string&>().empty() || value.get_ref<const std::string&>().size() > CALIPER_NET_NAME_MAX))) {
            if (error) *error = "The producer did not announce text_params.";
            return 0;
        }
    }
    wire::Control c;
    c.request_id = next_request_++;
    c.op = op;
    if (!name.empty()) c.name = name;
    if (op == CALIPER_NET_OP_SET) {
        try {
            c.value = to_net(value);
        } catch (const json::exception&) {
            if (error) *error = "The parameter value must contain valid UTF-8.";
            return 0;
        }
    }
    if (op == CALIPER_NET_OP_STEP) c.count = std::max<std::uint32_t>(1, count);
    if (op == CALIPER_NET_OP_INTERVENE) c.intervene = to_net(value);
    if (op == CALIPER_NET_OP_INTERPRET) {
        try {
            c.interpretation = to_net(value);
        } catch (const json::exception&) {
            if (error) *error = "The interpretation request must contain valid UTF-8.";
            return 0;
        }
    }
    std::vector<std::uint8_t> out;
    std::string err;
    if (!wire::encode(c, &out, &err)) {
        if (error) *error = "The request breaks a limit of the protocol: " + err;
        return 0;
    }
    if (!client_->send(out.data(), out.size(), true)) {
        if (error) *error = "The transport refused the request; check the connection and retry.";
        return 0;
    }
    return c.request_id;
}

json RunConnection::interpret(const std::string& method, const json& params, std::uint32_t request) {
    auto error = [](const std::string& code, const std::string& message) {
        return json{{"error", {{"code",code},{"message",message}}}};
    };
    if (request && (!params.is_object() || !params.empty()))
        return error("command.invalid_args", "A reply query accepts only request_id.");
    if (state_ != State::open) return error("run.not_attached", "The run connection is closed.");
    if (!interpretation_) return error("interpret.unsupported", "The producer did not announce interpretation.");
    const bool read = method.rfind("feature.",0) == 0 || method == "trace.read" || method == "trace.status";
    if (!read && !control_allowed_) return error("run.control_not_allowed", "The connection cannot change the run.");
    if (!request) {
        if (interpretation_replies_.size() >= 64) {
            auto it = std::find_if(interpretation_replies_.begin(), interpretation_replies_.end(), [](const auto& x) {
                return x.second.second.status != "queued";
            });
            if (it == interpretation_replies_.end()) return error("interpret.busy", "Too many requests await completion.");
            interpretation_replies_.erase(it);
        }
        std::string reason;
        request = send_control(CALIPER_NET_OP_INTERPRET, "", {{"version",1},{"method",method},{"params",params}}, 0, &reason);
        if (!request) return error("interpret.refused", reason);
        Reply pending;
        pending.request_id = request; pending.status = "queued";
        interpretation_replies_[request] = {method,pending};
    }
    auto it = interpretation_replies_.find(request);
    if (it == interpretation_replies_.end() || it->second.first != method)
        return error("interpret.unknown_request", "No matching request is retained.");
    const auto& reply = it->second.second;
    if (reply.status == "error") return error(reply.error_code, reply.error_message);
    return {{"request_id",request},{"pending",reply.status == "queued"},{"result",reply.result}};
}

std::vector<RunConnection::Reply> RunConnection::take_replies() {
    std::vector<Reply> out;
    out.swap(replies_);
    return out;
}

}  // namespace caliper_runner
