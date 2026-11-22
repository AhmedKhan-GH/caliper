// The child process on Windows (child_process.h). This code is written
// against the documented Win32 declarations and is not yet built or tested
// on Windows. CreateProcessW starts the child suspended, in a job object
// whose last handle ends it (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE), so that no
// child outlives the app, also in a crash; then it resumes. A forced end
// ends the whole job, so that a process the child started, such as the real
// interpreter behind a venv launcher, ends too. The output file takes
// stdout and stderr, NUL takes stdin, and the child has no console window.
#include "child_process.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdlib>
#include <cwchar>

namespace caliper_runner {

namespace {

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring out(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n);
    return out;
}

std::string narrow(const wchar_t* s, std::size_t len) {
    if (len == 0) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, s, static_cast<int>(len), nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<std::size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, s, static_cast<int>(len), out.data(), n, nullptr, nullptr);
    return out;
}

// One argument quoted as CommandLineToArgvW reads it back.
void append_quoted(std::wstring& cmd, const std::wstring& arg) {
    if (!cmd.empty()) cmd += L' ';
    if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        cmd += arg;
        return;
    }
    cmd += L'"';
    for (std::size_t i = 0;; ++i) {
        std::size_t slashes = 0;
        while (i < arg.size() && arg[i] == L'\\') {
            ++i;
            ++slashes;
        }
        if (i == arg.size()) {
            cmd.append(slashes * 2, L'\\');
            break;
        }
        if (arg[i] == L'"') {
            cmd.append(slashes * 2 + 1, L'\\');
            cmd += L'"';
        } else {
            cmd.append(slashes, L'\\');
            cmd += arg[i];
        }
    }
    cmd += L'"';
}

class WinChild final : public ChildProcess {
public:
    WinChild(HANDLE process, HANDLE job, DWORD pid) : process_(process), job_(job), pid_(pid) {}
    ~WinChild() override {
        if (process_) CloseHandle(process_);
        if (job_) CloseHandle(job_);   // the job's last handle: ends the child if it still runs
    }

    std::int64_t pid() const override { return pid_; }

    bool poll(std::int64_t* exit_code) override { return wait(0, exit_code); }

    void terminate() override { kill(); }

    void kill() override {
        // The job holds only the child and the processes it started, so it
        // may end also after the child ended, when a process it started runs
        // on.
        if (job_ && TerminateJobObject(job_, 1)) return;
        std::int64_t code = 0;
        if (!poll(&code)) TerminateProcess(process_, 1);
    }

    bool wait(int timeout_ms, std::int64_t* exit_code) override {
        if (ended_) {
            *exit_code = code_;
            return true;
        }
        if (WaitForSingleObject(process_, static_cast<DWORD>(timeout_ms < 0 ? 0 : timeout_ms)) != WAIT_OBJECT_0)
            return false;
        DWORD code = 0;
        GetExitCodeProcess(process_, &code);
        code_ = static_cast<std::int64_t>(code);
        ended_ = true;
        *exit_code = code_;
        return true;
    }

private:
    HANDLE process_ = nullptr;
    HANDLE job_ = nullptr;
    DWORD pid_ = 0;
    bool ended_ = false;
    std::int64_t code_ = 0;
};

}  // namespace

std::vector<std::pair<std::string, std::string>> current_environment() {
    std::vector<std::pair<std::string, std::string>> out;
    wchar_t* block = GetEnvironmentStringsW();
    if (!block) return out;
    for (const wchar_t* p = block; *p; p += wcslen(p) + 1) {
        const std::string kv = narrow(p, wcslen(p));
        const std::size_t eq = kv.find('=', 1);   // a leading '=' names a drive's folder
        if (eq == std::string::npos || kv[0] == '=') continue;
        out.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
    }
    FreeEnvironmentStringsW(block);
    return out;
}

std::int64_t current_process_id() { return static_cast<std::int64_t>(GetCurrentProcessId()); }

std::unique_ptr<ChildProcess> start_child(const ChildSpec& spec, std::string* error) {
    auto failed = [&](const std::string& why) -> std::unique_ptr<ChildProcess> {
        if (error) *error = "The program " + spec.program + " could not start: " + why;
        return nullptr;
    };
    if (spec.program.empty()) return failed("no program was named.");

    SECURITY_ATTRIBUTES inherit{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE out = CreateFileW(widen(spec.output.empty() ? std::string("NUL") : spec.output).c_str(), FILE_APPEND_DATA,
                             FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (out == INVALID_HANDLE_VALUE) return failed("the output file could not be opened.");
    HANDLE in = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);

    std::wstring cmd;
    append_quoted(cmd, widen(spec.program));
    for (const auto& a : spec.args) append_quoted(cmd, widen(a));
    std::wstring env;
    for (const auto& kv : spec.env) {
        env += widen(kv.first + "=" + kv.second);
        env += L'\0';
    }
    env += L'\0';

    STARTUPINFOW si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = in;
    si.hStdOutput = out;
    si.hStdError = out;
    PROCESS_INFORMATION pi{};
    const std::wstring cwd = widen(spec.cwd);
    const DWORD flags = CREATE_UNICODE_ENVIRONMENT | CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP | CREATE_SUSPENDED;
    const BOOL ok = CreateProcessW(widen(spec.program).c_str(), cmd.data(), nullptr, nullptr, TRUE, flags,
                                   env.data(), cwd.empty() ? nullptr : cwd.c_str(), &si, &pi);
    const DWORD err = ok ? 0 : GetLastError();
    CloseHandle(out);
    if (in != INVALID_HANDLE_VALUE) CloseHandle(in);
    if (!ok) return failed("CreateProcessW failed with error " + std::to_string(err) + ".");

    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
        AssignProcessToJobObject(job, pi.hProcess);
    }
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    return std::make_unique<WinChild>(pi.hProcess, job, pi.dwProcessId);
}

}  // namespace caliper_runner
