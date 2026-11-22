/* ==========================================================================
 * caliper/wgsl.h: the runtime WGSL compiler, a C interface to Naga.
 *
 * The library validates a WGSL source and translates one entry point of it
 * to MSL for Metal and to SPIR-V for Vulkan. It is the same pinned Naga that
 * translates the built-in shaders at developer time (tools/shaders), with
 * the same options, thus a program that an agent writes and a built-in
 * shader follow one convention for bindings, versions and bounds checks.
 *
 * LIBRARY
 *   A Rust static library (tools/shaders/caliper-wgsl), prebuilt for each
 *   platform so that a build of Caliper needs no Rust. CMake makes it the
 *   imported target caliper::wgsl when it is present (cmake/CaliperWgsl.cmake).
 *
 * MODULE
 *   caliper_wgsl_module_create() parses and validates a source and keeps the
 *   result. The module is immutable: translations and binding queries only
 *   read it, thus one module can serve several threads at the same time.
 *   Unused declarations are dropped, as the developer-time driver does.
 *
 * ERRORS
 *   Every function that can fail returns a caliper_wgsl_status and fills an
 *   optional caliper_wgsl_error with the message and the 1-based line and
 *   column of the error in the source. No panic of Naga leaves the library:
 *   the library catches it and returns CALIPER_WGSL_ERROR_INTERNAL.
 *
 * MEMORY
 *   The library owns every pointer it returns. A buffer goes back through
 *   caliper_wgsl_buffer_free(), an error through caliper_wgsl_error_free()
 *   and a module through caliper_wgsl_module_destroy(). A name in a binding
 *   or an entry point stays valid until its module is destroyed.
 *
 * METAL BINDING MAP
 *   As in src/host/gfx/gfx.h: every binding that an entry point uses gets one
 *   Metal index, the same in every stage. Buffers, textures and samplers are
 *   numbered separately, each from 0, in (group, binding) order across the
 *   module. Buffer indices stay below metal_vertex_buffer_base, and the index
 *   just below it holds the sizes of runtime-sized arrays. SPIR-V keeps each
 *   group and binding as its descriptor set and binding.
 * ========================================================================== */
#ifndef CALIPER_WGSL_H
#define CALIPER_WGSL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** The interface version of this header. caliper_wgsl_abi_version() returns
 *  the version the library was built with, and a host refuses a mismatch. */
#define CALIPER_WGSL_ABI_VERSION 1u

/** The result of a call. */
typedef enum caliper_wgsl_status {
    CALIPER_WGSL_OK = 0,
    CALIPER_WGSL_ERROR_ARGUMENT = 1,     /**< A null pointer, an unknown stage or entry point. */
    CALIPER_WGSL_ERROR_TOO_LARGE = 2,    /**< The source is larger than max_source_bytes. */
    CALIPER_WGSL_ERROR_PARSE = 3,        /**< The source is not valid UTF-8 or not valid WGSL syntax. */
    CALIPER_WGSL_ERROR_VALIDATION = 4,   /**< The WGSL parses but breaks a rule of the language. */
    CALIPER_WGSL_ERROR_UNSUPPORTED = 5,  /**< Valid WGSL that Caliper's backends do not take. */
    CALIPER_WGSL_ERROR_TRANSLATION = 6,  /**< The MSL or SPIR-V writer refused the module. */
    CALIPER_WGSL_ERROR_INTERNAL = 7      /**< A panic inside Naga or the library, caught. */
} caliper_wgsl_status;

/** A shader stage. */
typedef enum caliper_wgsl_stage {
    CALIPER_WGSL_STAGE_VERTEX = 0,
    CALIPER_WGSL_STAGE_FRAGMENT = 1,
    CALIPER_WGSL_STAGE_COMPUTE = 2
} caliper_wgsl_stage;

/** The kind of a binding. The names follow BindingType of src/host/gfx/gfx.h. */
typedef enum caliper_wgsl_binding_kind {
    CALIPER_WGSL_BINDING_UNIFORM_BUFFER = 1,
    CALIPER_WGSL_BINDING_STORAGE_BUFFER = 2,
    CALIPER_WGSL_BINDING_TEXTURE = 3,          /**< A sampled 2-D float texture. */
    CALIPER_WGSL_BINDING_UINT_TEXTURE = 4,     /**< A sampled 2-D unsigned integer texture. */
    CALIPER_WGSL_BINDING_STORAGE_TEXTURE = 5,  /**< A 2-D storage texture. */
    CALIPER_WGSL_BINDING_SAMPLER = 6
} caliper_wgsl_binding_kind;

/** Bits of caliper_wgsl_binding.access. */
enum {
    CALIPER_WGSL_ACCESS_READ = 1u,
    CALIPER_WGSL_ACCESS_WRITE = 2u
};

/** The options of a module. caliper_wgsl_options_init() writes the values
 *  of the developer-time driver, which the built-in shaders use. */
typedef struct caliper_wgsl_options {
    /** The largest source in bytes that caliper_wgsl_module_create() takes.
     *  The default is 1 MiB. */
    size_t max_source_bytes;
    /** The Metal Shading Language version. The default is 2.4. */
    uint32_t msl_version_major;
    uint32_t msl_version_minor;
    /** The SPIR-V version. The default is 1.3 (Vulkan 1.1). */
    uint32_t spirv_version_major;
    uint32_t spirv_version_minor;
    /** The first Metal buffer index of the vertex buffers. Vertex buffer slot
     *  i is Metal buffer index metal_vertex_buffer_base + i. The default is
     *  16. It is between 2 and 31. */
    uint32_t metal_vertex_buffer_base;
} caliper_wgsl_options;

/** An error. The strings are NUL-terminated and belong to the library. */
typedef struct caliper_wgsl_error {
    /** The status of the failed call. */
    caliper_wgsl_status status;
    /** The 1-based line and column (in bytes) of the error in the source,
     *  or 0 when the error has no place in the source. */
    uint32_t line;
    uint32_t column;
    /** The message on one line, or NULL. */
    char* message;
    /** The full report with the lines of the source it concerns, or NULL. */
    char* report;
} caliper_wgsl_error;

/** Bytes that the library allocated: size bytes and one NUL byte after them
 *  that size does not count, thus MSL text is also a C string. SPIR-V is
 *  little-endian 32-bit words, and its size is a multiple of 4. */
typedef struct caliper_wgsl_buffer {
    uint8_t* data;
    size_t size;
} caliper_wgsl_buffer;

/** A binding that an entry point of the module uses. */
typedef struct caliper_wgsl_binding {
    uint32_t group;
    uint32_t binding;
    /** A caliper_wgsl_binding_kind. */
    uint32_t kind;
    /** CALIPER_WGSL_ACCESS_* bits: READ for a uniform buffer, a sampled
     *  texture or a sampler, READ and WRITE as the WGSL declares them for a
     *  storage buffer or a storage texture. */
    uint32_t access;
    /** Bit (1 << caliper_wgsl_stage) for each stage that uses the binding. */
    uint32_t stages;
    /** The Metal index of the binding, in its kind's numbering. */
    uint32_t metal_index;
    /** The size of a buffer's type in bytes (the fixed part when it ends in
     *  a runtime-sized array), 0 for a texture or a sampler. */
    uint64_t size;
    /** The WGSL name of the variable. */
    const char* name;
    /** The texel format of a storage texture in WGSL spelling, such as
     *  "rgba8unorm", or NULL. */
    const char* storage_format;
} caliper_wgsl_binding;

/** An entry point of the module. */
typedef struct caliper_wgsl_entry_point {
    const char* name;
    /** A caliper_wgsl_stage. */
    uint32_t stage;
    /** The workgroup size of a compute entry point, else 0, 0, 0. */
    uint32_t workgroup_size[3];
} caliper_wgsl_entry_point;

/** A validated WGSL module. */
typedef struct caliper_wgsl_module caliper_wgsl_module;

/** Returns CALIPER_WGSL_ABI_VERSION as the library was built. */
uint32_t caliper_wgsl_abi_version(void);

/** Returns the version of Naga in the library, such as "30.0.2". */
const char* caliper_wgsl_naga_version(void);

/** Writes the default options, the developer-time driver's values. */
void caliper_wgsl_options_init(caliper_wgsl_options* options);

/** Parses and validates source (size bytes of UTF-8, no NUL needed) and
 *  creates a module. options NULL means the defaults. On success *module is
 *  the new module, else NULL and error, if not NULL, describes the failure. */
caliper_wgsl_status caliper_wgsl_module_create(const char* source, size_t size,
                                               const caliper_wgsl_options* options,
                                               caliper_wgsl_module** module,
                                               caliper_wgsl_error* error);

/** Destroys a module. NULL is allowed. */
void caliper_wgsl_module_destroy(caliper_wgsl_module* module);

/** Returns the number of bindings that the entry points of the module use. */
size_t caliper_wgsl_module_binding_count(const caliper_wgsl_module* module);

/** Reads binding index, in (group, binding) order, into *binding. */
caliper_wgsl_status caliper_wgsl_module_binding(const caliper_wgsl_module* module, size_t index,
                                                caliper_wgsl_binding* binding);

/** Returns the number of entry points of the module. */
size_t caliper_wgsl_module_entry_point_count(const caliper_wgsl_module* module);

/** Reads entry point index, in source order, into *entry_point. */
caliper_wgsl_status caliper_wgsl_module_entry_point(const caliper_wgsl_module* module, size_t index,
                                                    caliper_wgsl_entry_point* entry_point);

/** Translates the entry point name of the given stage to MSL text in *msl. */
caliper_wgsl_status caliper_wgsl_translate_msl(const caliper_wgsl_module* module, const char* name,
                                               caliper_wgsl_stage stage, caliper_wgsl_buffer* msl,
                                               caliper_wgsl_error* error);

/** Translates the entry point name of the given stage to a SPIR-V module in
 *  *spirv, with that entry point only. */
caliper_wgsl_status caliper_wgsl_translate_spirv(const caliper_wgsl_module* module, const char* name,
                                                 caliper_wgsl_stage stage, caliper_wgsl_buffer* spirv,
                                                 caliper_wgsl_error* error);

/** Frees the bytes of a buffer and clears it. NULL is allowed. */
void caliper_wgsl_buffer_free(caliper_wgsl_buffer* buffer);

/** Frees the strings of an error and clears it. NULL is allowed. */
void caliper_wgsl_error_free(caliper_wgsl_error* error);

/** Panics inside the library's guard and returns what the guard returns,
 *  CALIPER_WGSL_ERROR_INTERNAL with the panic's message. It exists for the
 *  test of the guard; a host never calls it. */
caliper_wgsl_status caliper_wgsl_test_panic(caliper_wgsl_error* error);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* CALIPER_WGSL_H */
