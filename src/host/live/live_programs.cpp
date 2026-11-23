// live_programs.cpp: ProgramSync and the ProgramLink of an Attachment (live_programs.h).
#include "live/live_programs.h"

#include "embed/embed_core.h"
#include "live/live_attach.h"

#include <algorithm>
#include <utility>

namespace caliper_live {

namespace wire = caliper_net::wire;
using caliper_net::Json;

// ---- AttachmentProgramLink ------------------------------------------------------------------

NetSource* AttachmentProgramLink::source() const {
    CaliperCore* core = attachment_.core();
    return core ? dynamic_cast<NetSource*>(caliper_live_stream(core)) : nullptr;
}

bool AttachmentProgramLink::ready() {
    NetSource* s = source();
    return s && s->programs_ready(nullptr);
}

uint64_t AttachmentProgramLink::epoch() {
    NetSource* s = source();
    return s ? s->program_epoch() : 0;
}

uint32_t AttachmentProgramLink::send(wire::Program msg) {
    NetSource* s = source();
    return s ? s->send_program(std::move(msg)) : 0;
}

std::vector<ProgramAnswer> AttachmentProgramLink::take() {
    NetSource* s = source();
    return s ? s->take_program_answers() : std::vector<ProgramAnswer>{};
}

bool AttachmentProgramLink::takes_pieces() {
    NetSource* s = source();
    return s && s->takes_pieces();
}

std::vector<ReceivedResource> AttachmentProgramLink::take_resources() {
    NetSource* s = source();
    return s ? s->take_resources() : std::vector<ReceivedResource>{};
}

}  // namespace caliper_live
