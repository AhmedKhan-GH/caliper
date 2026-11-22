#pragma once
// caliper_builder::programs: the compiler's reflection, the verdict of a
// check and the result of a compile, which sandbox.h (the one-pass form and
// stages) and graphics.h (the graphics form) share. Only the standard
// library, so that the browser module can compile the checks of graphics.h.
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace caliper_builder::programs {

// ---- reflection: the neutral form of caliper_wgsl_binding and
// caliper_wgsl_entry_point (tools/shaders/caliper-wgsl/include/caliper/wgsl.h),
// with the same numbers, so that a host without the compiler's header can
// still hold one.
enum class BindingKind : std::uint32_t {
    uniform_buffer = 1,
    storage_buffer = 2,
    texture = 3,
    uint_texture = 4,
    storage_texture = 5,
    sampler = 6,
};
enum class EntryStage : std::uint32_t { vertex = 0, fragment = 1, compute = 2 };

struct ReflectedBinding {
    std::uint32_t group = 0;
    std::uint32_t binding = 0;
    BindingKind kind = BindingKind::storage_buffer;
    bool read = false;            // CALIPER_WGSL_ACCESS_READ
    bool write = false;           // CALIPER_WGSL_ACCESS_WRITE: read_write storage, or a storage texture
    std::uint64_t size = 0;       // the fixed size in bytes; the fixed part before a runtime-sized array
    std::string name;             // the WGSL variable's name
    std::string storage_format;   // storage textures only
};

struct ReflectedEntry {
    std::string name;
    EntryStage stage = EntryStage::compute;
    std::array<std::uint32_t, 3> workgroup_size{};   // compute only; zeros otherwise
};

struct Reflection {
    std::vector<ReflectedBinding> bindings;   // in (group, binding) order
    std::vector<ReflectedEntry> entries;
};

// ---- verdict
// The outcome of one check: ok, or the first failure with its code, one plain
// sentence, where it is (a WGSL variable, a declaration key such as
// output.max_count, or empty) and, for a compiler message, the line and the
// column in the checked source, from 1 (0 when it has no place).
struct Verdict {
    std::string code;      // empty: ok
    std::string message;
    std::string where;
    std::uint32_t line = 0;
    std::uint32_t column = 0;

    bool ok() const { return code.empty(); }
};

// The result of a compile: a verdict, the reflection when it compiled, and
// the warnings of the compiler.
struct Compiled {
    Verdict verdict;
    Reflection reflection;
    std::vector<Verdict> warnings;
};

}  // namespace caliper_builder::programs
