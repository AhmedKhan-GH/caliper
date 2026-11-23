/*
 * The producer runtime's C interface for the derive half of a piece
 * (design/programmable-graphics.md, Derive Half and Resources), beside
 * caliper_producer.h.
 *
 * A derive program runs several compute passes, in their order and in one
 * submit, at a pass of the shared cadence, under the same time watchdog,
 * memory budget and read-only rule as a one-pass program. It writes named,
 * typed resources that keep their contents between runs (a resource with
 * `clear` is zeroed before each run), and a counted buffer's counter is set
 * to zero before each run. The runtime composes the module itself with
 * compose_derive() of caliper_builder/programs/pieces.h from this
 * description and the files that PROGRAM carries with their imports
 * resolved, so that the script checks the same text as the app.
 *
 * caliper_producer_run() runs a derive program like any other, and
 * caliper_producer_read_status() reports it; caliper_producer_rebind() and
 * caliper_producer_set_params() work on it too. After a pass, the facade reads
 * each published resource with caliper_producer_read_resource() and sends it
 * as one RESOURCE message.
 *
 * The runtime holds the rules of one-pass programs for every derive pass:
 *   - Read-only: a tensor input is var<storage, read> in the header that the
 *     runtime composes; a binding of the reflection that is not a binding of
 *     that header at its place refuses the program
 *     ("program.binding_not_allowed"), a tensor input that writes refuses it
 *     ("program.writes_input"), and the layer's read_only flag refuses it again.
 *   - Bounds: the composed module is translated with every buffer index
 *     checked, as a one-pass program is.
 *   - Sizes: every count, side, stride and byte size is bounded before
 *     allocation, in 64-bit arithmetic that cannot wrap.
 *   - Time and memory: one run of all passes is one submit under the time
 *     watchdog and the timeout of caliper_producer_run(), and the resources,
 *     counters and bound tensors stay inside memory_limit_bytes.
 * On a platform without the Metal runtime, both functions answer
 * CALIPER_PRODUCER_UNSUPPORTED with the code "program.unavailable".
 */
#ifndef CALIPER_PRODUCER_DERIVE_H
#define CALIPER_PRODUCER_DERIVE_H

#include "producer/caliper_producer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum caliper_producer_resource_kind {
    CALIPER_PRODUCER_RESOURCE_BUFFER = 1,   /* array<T> of `count` elements                     */
    CALIPER_PRODUCER_RESOURCE_TEXTURE = 2   /* array<u32> of width*height packed rgba8unorm texels */
} caliper_producer_resource_kind;

/* One entry of `resources` (spec/piece.h, Resource). */
typedef struct caliper_producer_resource_desc {
    const char* id;
    uint32_t kind;                    /* caliper_producer_resource_kind                       */
    const char* element;              /* buffer: a plain element type, or NULL                */
    const char* struct_name;          /* buffer: the struct's name when `fields` is given     */
    const char* const* fields;        /* buffer: "name: type", in order                      */
    size_t field_count;               /* at most 16                                           */
    uint32_t count;                   /* buffer: elements, at most 4,194,304                  */
    uint32_t width;                   /* texture                                              */
    uint32_t height;
    uint32_t counted;                 /* 1: the counter <id>_count bounds the valid elements  */
    uint32_t publish;                 /* 1: read after a pass and sent; 0: private            */
    uint32_t clear;                   /* 1: zeroed before each run of the passes              */
} caliper_producer_resource_desc;

/* One compute pass of the derive half. */
typedef struct caliper_producer_derive_pass {
    const char* entry;
    uint32_t workgroups[3];
    const char* const* writes;        /* resources that the pass may change                   */
    size_t write_count;
} caliper_producer_derive_pass;

/* The derive half of a program, an instance of a piece, after the facade's
 * parse of its PROGRAM declaration. */
typedef struct caliper_producer_derive_desc {
    const char* id;                   /* the program's id                                     */
    const char* piece_hash;           /* "sha256:<hex>" of the pin; the runtime checks its    */
                                      /* form only, and the app compared the hash of the      */
                                      /* files with the pin before it sent the program        */
    const char* wgsl;                 /* derive.wgsl, imports resolved                        */
    size_t wgsl_size;
    const char* common;               /* the lib modules and common.wgsl, or NULL             */
    size_t common_size;
    const caliper_producer_input* inputs;   /* `var` is the piece's input name; an entry with   */
                                            /* tensor 0 holds the place of a buffer input, which */
                                            /* the derive half does not bind                     */
    size_t input_count;               /* at most 8                                            */
    const char* const* input_types;   /* "f32", "i32" or "u32" for each input; NULL: all f32  */
    const caliper_producer_resource_desc* resources;
    size_t resource_count;            /* at most 8                                            */
    const caliper_producer_derive_pass* passes;
    size_t pass_count;                /* 1 to 8                                               */
    const caliper_producer_param* params;
    const char* const* param_types;   /* "f32", "i32" or "u32" for each param, in order       */
    size_t param_count;               /* at most 16                                           */
    double time_limit_ms;             /* all passes of one run                                */
    uint64_t memory_limit_bytes;      /* resources and bound tensors                          */
} caliper_producer_derive_desc;

/* Checks, composes and compiles a derive program, allocates its resources
 * inside its budget, and gives its handle for caliper_producer_run(). */
caliper_producer_status caliper_producer_compile_derive(caliper_producer* runtime,
                                                        const caliper_producer_derive_desc* desc, uint32_t* program,
                                                        caliper_producer_error* error);

/* The valid bytes of one published resource after the last pass: `count` is
 * the valid elements (min(counter, count) for a counted buffer), `words` the
 * 32-bit words of one element. The buffer stays valid until the next
 * caliper_producer_run(), caliper_producer_remove() or caliper_producer_destroy(). */
caliper_producer_status caliper_producer_read_resource(caliper_producer* runtime, uint32_t program, const char* resource,
                                                       caliper_producer_output* out, caliper_producer_error* error);

#ifdef __cplusplus
}
#endif

#endif /* CALIPER_PRODUCER_DERIVE_H */
