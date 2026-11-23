// frame_log.h: CALIPER_FRAME_LOG=<file>, an opt-in log of the app's frames, for a person who
// sees the visible app stall and wants to record it. Each frame appends one tab-separated line
// (the header line names the columns); the log stops after kMaxLines lines, and a buffered stream
// is flushed every kFlushLines lines and at the end. Unset, the app opens no file and measures
// nothing for it. Frame thread only.
#pragma once

#include <chrono>
#include <cstdio>
#include <string>

namespace caliper_host {

class FrameLog {
public:
    static constexpr int kMaxLines = 20000;
    static constexpr int kFlushLines = 60;

    FrameLog() = default;
    FrameLog(const FrameLog&) = delete;
    FrameLog& operator=(const FrameLog&) = delete;
    ~FrameLog() { close(); }

    // Opens the file that `path` names, for appending, and writes the header; false when it
    // cannot. An empty path leaves the log off.
    bool open(const std::string& path, const char* header) {
        close();
        if (path.empty()) return false;
        file_ = std::fopen(path.c_str(), "a");
        if (!file_) return false;
        t0_ = std::chrono::steady_clock::now();
        std::fprintf(file_, "%s\n", header);
        return true;
    }
    bool on() const { return file_ != nullptr; }
    // Milliseconds of the monotonic clock since open().
    double now_ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
    }
    // Appends one line (without its newline); after kMaxLines lines the log says so and closes.
    void line(const std::string& text) {
        if (!file_) return;
        std::fputs(text.c_str(), file_);
        std::fputc('\n', file_);
        if (++lines_ % kFlushLines == 0) std::fflush(file_);
        if (lines_ >= kMaxLines) {
            std::fprintf(file_, "# the log stops after %d frames\n", kMaxLines);
            close();
        }
    }
    void close() {
        if (file_) std::fclose(file_);
        file_ = nullptr;
    }

private:
    std::FILE* file_ = nullptr;
    std::chrono::steady_clock::time_point t0_{};
    int lines_ = 0;
};

}  // namespace caliper_host
