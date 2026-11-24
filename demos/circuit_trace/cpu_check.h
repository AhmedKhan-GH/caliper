#pragma once
#include <cstdio>
#include <cstdlib>

// Keep CPU test checks active when NDEBUG is defined.
inline void circuit_cpu_check(bool passed, const char* expression, int line) {
    if (!passed) {
        std::fprintf(stderr, "CPU check failed at line %d: %s\n", line, expression);
        std::exit(EXIT_FAILURE);
    }
}
#define CIRCUIT_CHECK(expression) circuit_cpu_check(bool(expression), #expression, __LINE__)
