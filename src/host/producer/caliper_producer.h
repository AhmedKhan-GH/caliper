/*
 * The producer runtime's C interface (design/gpu-programs.md, Producer Runtime
 * Interface; design/producer-runtime.md): the graphics layer without a window
 * in the training script's process, with the runtime WGSL compiler, which runs
 * GPU programs over the script's tensors in place.
 *
 * The native module caliper._producer calls this interface; it is the only
 * caller. The facade python/caliper/_programs.py decides when a pass runs (the
 * cadence) and gives each output to the publish path of caliper.scene or
 * caliper.show. The library caliper_producer (src/host/producer) holds the
 * implementation.
 *
 * Life of a runtime:
 *   caliper_producer_create()          on the device of the first bound tensor
 *   caliper_producer_import()          each input tensor, in place and read-only
 *   caliper_producer_compile()         a program, after the sandbox checks of
 *                                      caliper_builder/programs/sandbox.h
 *   caliper_producer_run()             at caliper.step(), after the framework's
 *                                      work is complete: one pass of the due
 *                                      programs, waited for before it returns
 *   caliper_producer_read_output()     each output buffer after the pass
 *   caliper_producer_read_status()     the state, times and memory of a program
 *   caliper_producer_remove() / caliper_producer_release() / caliper_producer_destroy()
 *
 * Rules:
 *   - Threads: one runtime is used from one thread, the training thread
 *     (decision 17 of src/host/gfx/gfx.h). The runtime adds no thread.
 *   - Read-only: an import never gets a write binding. The runtime refuses a
 *     program whose input binding is not var<storage, read> before any backend
 *     sees it, and the layer's read_only flag of the import refuses it again.
 *   - Copies: the runtime never copies a training tensor; an import that cannot
 *     be in place is refused (CALIPER_PRODUCER_IMPORT_REFUSED).
 *   - Errors: every function returns a status and, when it is not OK and
 *     `error` is not NULL, fills `error` with a stable code of sandbox.h
 *     (such as "program.writes_input"), one sentence and, for a compiler
 *     message, the line and column in the program's WGSL. Free it with
 *     caliper_producer_error_free(). No function throws or aborts.
 *   - Ownership: the runtime owns every pointer it returns. An output's buffer
 *     stays valid until the next caliper_producer_run(), caliper_producer_remove()
 *     of its program, or caliper_producer_destroy().
 */
#ifndef CALIPER_PRODUCER_H
#define CALIPER_PRODUCER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 2: the tensor output, the uniform `params` and caliper_producer_set_params(). */
#define CALIPER_PRODUCER_ABI_VERSION 2u

typedef struct caliper_producer caliper_producer;

typedef enum caliper_producer_status {
    CALIPER_PRODUCER_OK = 0,
    CALIPER_PRODUCER_ARGUMENT = 1,        /* a NULL pointer, a bad size, an unknown handle          */
    CALIPER_PRODUCER_UNSUPPORTED = 2,     /* no Metal or Vulkan device, or no WGSL compiler          */
    CALIPER_PRODUCER_REFUSED = 3,         /* a sandbox check or the compiler refused the program     */
    CALIPER_PRODUCER_IMPORT_REFUSED = 4,  /* another device, not contiguous, or a short buffer       */
    CALIPER_PRODUCER_OVER_BUDGET = 5,     /* the outputs and bound tensors are above memory_bytes    */
    CALIPER_PRODUCER_TIMEOUT = 6,         /* the wait of a pass timed out; every program stopped      */
    CALIPER_PRODUCER_DEVICE_LOST = 7,     /* the device was lost; every program stopped               */
    CALIPER_PRODUCER_INTERNAL = 8
} caliper_producer_status;

typedef struct caliper_producer_error {
    caliper_producer_status status;
    const char* code;   /* a static string of sandbox.h, such as "program.over_time"; never freed     */
    uint32_t line;      /* in the program's WGSL, from 1; 0 when the error has no place                */
    uint32_t column;
    char* message;      /* one plain sentence, owned; caliper_producer_error_free() frees it           */
} caliper_producer_error;

/* The element types of an imported tensor (DLPack's, as caliper._native maps them). */
typedef enum caliper_producer_dtype {
    CALIPER_PRODUCER_F32 = 0,
    CALIPER_PRODUCER_F16 = 1,
    CALIPER_PRODUCER_BF16 = 2,
    CALIPER_PRODUCER_I32 = 3,
    CALIPER_PRODUCER_U32 = 4,
    CALIPER_PRODUCER_I64 = 5,
    CALIPER_PRODUCER_U8 = 6
} caliper_producer_dtype;

#define CALIPER_PRODUCER_MAX_DIMS 8u

/* A tensor to import, from a DLPack capsule of device kDLMetal: the storage's
 * id<MTLBuffer> and the byte offset of the tensor in it. The tensor must be
 * contiguous. The caller keeps the capsule alive until caliper_producer_release(),
 * thus the framework cannot give the storage to another tensor meanwhile. */
typedef struct caliper_producer_tensor {
    const char* name;         /* the live name, such as "layers.0.weight"; copied           */
    void* buffer;             /* id<MTLBuffer> on Metal                                      */
    uint64_t byte_offset;
    uint64_t byte_size;       /* the tensor's bytes from byte_offset                        */
    uint32_t dtype;           /* caliper_producer_dtype                                     */
    uint32_t ndim;
    int64_t shape[CALIPER_PRODUCER_MAX_DIMS];
} caliper_producer_tensor;

typedef enum caliper_producer_output_kind {
    CALIPER_PRODUCER_GEOMETRY = 1,
    CALIPER_PRODUCER_PIXELS = 2,
    CALIPER_PRODUCER_TENSOR = 3      /* the f32 array `out`, published as the live view program.<id> */
} caliper_producer_output_kind;

typedef enum caliper_producer_part {
    CALIPER_PRODUCER_POINTS = 1,
    CALIPER_PRODUCER_LINES = 2
} caliper_producer_part;

/* One input: the WGSL variable and the handle of an imported tensor. */
typedef struct caliper_producer_input {
    const char* var;
    uint32_t tensor;
} caliper_producer_input;

/* One value of the uniform `params` (design/instrument-interface.md, Program
 * Parameters): a name of the program's struct Params and its number, which
 * the runtime converts to the field's type (f32, i32 or u32). */
typedef struct caliper_producer_param {
    const char* name;
    double value;
} caliper_producer_param;

/* A program after the editor's and the facade's parse of its declaration
 * (spec/program.h). The runtime runs check_declaration(), check_source(),
 * compile_and_reflect() and check_program() of sandbox.h on it again. */
typedef struct caliper_producer_program_desc {
    const char* id;
    const char* wgsl;
    size_t wgsl_size;
    const char* entry;                       /* NULL: the one compute entry point             */
    const caliper_producer_input* inputs;
    size_t input_count;                      /* at most 8                                     */
    uint32_t output_kind;                    /* caliper_producer_output_kind                  */
    /* geometry */
    uint32_t part;                           /* caliper_producer_part                         */
    uint32_t max_count;
    const char* const* arrays;               /* "positions", "colors", "values", "radii"      */
    size_t array_count;
    /* pixels */
    uint32_t width;
    uint32_t height;
    /* tensor: the shape of `out`, 1 to 4 axes and at most 1,000,000 elements */
    uint32_t tensor_ndim;
    int64_t tensor_shape[4];
    /* params: the start values of the uniform `params`, by name; the names
     * are those of the struct Params of the WGSL (program.params_mismatch) */
    const caliper_producer_param* params;
    size_t param_count;                      /* at most 16                                    */
    /* dispatch and limits */
    uint32_t workgroups[3];                  /* zeros: derived (sandbox.h dispatch_size())    */
    double time_limit_ms;
    uint64_t memory_limit_bytes;
} caliper_producer_program_desc;

/* What a pass of caliper_producer_run() reads about the step. */
typedef struct caliper_producer_frame {
    int64_t step;          /* the script's step                                    */
    double time_s;         /* seconds since the program was installed (the caller's clock) */
    uint32_t every;        /* the cadence N of this pass, for the uniform caliper   */
} caliper_producer_frame;

/* One output array after a pass. */
typedef struct caliper_producer_output {
    void* buffer;          /* id<MTLBuffer> in shared storage                       */
    uint64_t byte_offset;
    uint64_t byte_size;    /* of the valid items                                    */
    uint32_t count;        /* geometry: min(count, max_count); pixels: width*height;
                              tensor: the elements of its shape                    */
    uint32_t words;        /* 32-bit words of one item: 3 for positions, else 1     */
} caliper_producer_output;

typedef enum caliper_producer_state {
    CALIPER_PRODUCER_COMPILED = 1,
    CALIPER_PRODUCER_RUNNING = 2,
    CALIPER_PRODUCER_STOPPED = 3
} caliper_producer_state;

typedef struct caliper_producer_program_status {
    uint32_t state;                /* caliper_producer_state                                   */
    uint64_t runs;                 /* passes run                                               */
    int64_t last_step;             /* -1 before the first pass                                 */
    double last_pass_ms;           /* GPU time of its last dispatch; < 0 when unknown          */
    uint32_t over_time_passes;     /* passes over time_limit_ms in a row                       */
    uint64_t memory_bytes;         /* its outputs and its bound tensors                        */
    const char* stop_code;         /* the code that stopped it, or NULL; static                */
} caliper_producer_program_status;

/* The result of one pass. */
typedef struct caliper_producer_pass {
    double wall_ms;        /* from the submit to the end of the wait, on the calling thread */
    double gpu_ms;         /* the sum of the dispatches' GPU times; < 0 when unknown        */
    uint32_t ran;          /* programs that ran                                             */
    uint32_t stopped;      /* programs that this pass stopped (time limit)                  */
} caliper_producer_pass;

uint32_t caliper_producer_abi_version(void);

/* Makes a runtime on `device` (an id<MTLDevice>), with its own command queue.
 * NULL takes the device of the first imported tensor's buffer. */
caliper_producer_status caliper_producer_create(void* device, caliper_producer** out, caliper_producer_error* error);

/* Imports one tensor in place and read-only, and gives its handle. */
caliper_producer_status caliper_producer_import(caliper_producer* runtime, const caliper_producer_tensor* tensor,
                                                uint32_t* handle, caliper_producer_error* error);

/* Releases an import; a program that binds it stops. */
void caliper_producer_release(caliper_producer* runtime, uint32_t tensor);

/* Checks and compiles a program, allocates its outputs inside its budget, and
 * gives its handle. A refusal leaves nothing behind. */
caliper_producer_status caliper_producer_compile(caliper_producer* runtime, const caliper_producer_program_desc* desc,
                                                 uint32_t* program, caliper_producer_error* error);

/* Runs one pass of `programs` (one submit for each, in order) and waits for it,
 * at most `timeout_ms`. The caller calls it only after the framework's work on
 * the bound tensors is complete (TorchAdapter.framework_done()). A program over
 * its time limit on 3 passes in a row stops with "program.over_time". A timeout
 * stops every program and returns CALIPER_PRODUCER_TIMEOUT. */
caliper_producer_status caliper_producer_run(caliper_producer* runtime, const uint32_t* programs, size_t count,
                                             const caliper_producer_frame* frame, double timeout_ms,
                                             caliper_producer_pass* pass, caliper_producer_error* error);

/* One output of a program after its last pass: an array by name for geometry,
 * "pixels", or "out" for a tensor. */
caliper_producer_status caliper_producer_read_output(caliper_producer* runtime, uint32_t program, const char* name,
                                                caliper_producer_output* out, caliper_producer_error* error);

/* Sets values of the uniform `params` of a program by name; the next pass
 * reads them. A name that the struct Params lacks is
 * "program.params_mismatch" and changes nothing. No compile and no new hash. */
caliper_producer_status caliper_producer_set_params(caliper_producer* runtime, uint32_t program,
                                                    const caliper_producer_param* params, size_t count,
                                                    caliper_producer_error* error);
/* The state, the times and the memory of a program. */
caliper_producer_status caliper_producer_read_status(caliper_producer* runtime, uint32_t program,
                                                        caliper_producer_program_status* out);

/* Removes a program and frees its outputs. */
void caliper_producer_remove(caliper_producer* runtime, uint32_t program);

/* Stops every program, releases every import and frees the runtime. NULL is a no-op. */
void caliper_producer_destroy(caliper_producer* runtime);

void caliper_producer_error_free(caliper_producer_error* error);

#ifdef __cplusplus
}
#endif

#endif /* CALIPER_PRODUCER_H */
