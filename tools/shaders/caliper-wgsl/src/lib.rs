//! caliper-wgsl: Caliper's runtime WGSL compiler, the C interface of include/caliper/wgsl.h.
//!
//! It validates a WGSL source with Naga and translates one entry point to MSL or SPIR-V with
//! the options and the Metal binding map of the developer-time driver (tools/shaders/naga-driver),
//! so that a program written at run time follows the convention of the built-in shaders. Every
//! exported function runs inside `guard`, which turns a panic into CALIPER_WGSL_ERROR_INTERNAL:
//! an unwind must never cross the C boundary. The header is the reference for each function.

use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use naga::back::{msl, spv};
use naga::proc::{BoundsCheckPolicies, BoundsCheckPolicy, Layouter};
use naga::valid::{Capabilities, ModuleInfo, ValidationFlags, Validator};
use naga::{AddressSpace, Binding as IoBinding, ImageClass, Interpolation, Sampling, ScalarKind, ShaderStage, SourceLocation, TypeInner};

const ABI_VERSION: u32 = 3;
const NAGA_VERSION: &CStr = c"30.0.2";

const STATUS_OK: i32 = 0;
const STATUS_ARGUMENT: i32 = 1;
const STATUS_TOO_LARGE: i32 = 2;
const STATUS_PARSE: i32 = 3;
const STATUS_VALIDATION: i32 = 4;
const STATUS_UNSUPPORTED: i32 = 5;
const STATUS_TRANSLATION: i32 = 6;
const STATUS_INTERNAL: i32 = 7;

const KIND_UNIFORM_BUFFER: u32 = 1;
const KIND_STORAGE_BUFFER: u32 = 2;
const KIND_TEXTURE: u32 = 3;
const KIND_UINT_TEXTURE: u32 = 4;
const KIND_STORAGE_TEXTURE: u32 = 5;
const KIND_SAMPLER: u32 = 6;

const ACCESS_READ: u32 = 1;
const ACCESS_WRITE: u32 = 2;

const IO_INPUT: u32 = 0;
const IO_OUTPUT: u32 = 1;

const SCALAR_OTHER: u32 = 0;
const SCALAR_F32: u32 = 1;
const SCALAR_I32: u32 = 2;
const SCALAR_U32: u32 = 3;
const SCALAR_F16: u32 = 4;

const INTERPOLATION_PERSPECTIVE: u32 = 0;
const INTERPOLATION_LINEAR: u32 = 1;
const INTERPOLATION_FLAT: u32 = 2;
const INTERPOLATION_OTHER: u32 = 3;

const SAMPLING_CENTER: u32 = 0;
const SAMPLING_CENTROID: u32 = 1;
const SAMPLING_SAMPLE: u32 = 2;
const SAMPLING_FIRST: u32 = 3;
const SAMPLING_EITHER: u32 = 4;

const BOUNDS_CHECKED: u32 = 0;
const BOUNDS_UNCHECKED_BUFFERS: u32 = 1;

// The C structs of include/caliper/wgsl.h.

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Options {
    max_source_bytes: usize,
    msl_version_major: u32,
    msl_version_minor: u32,
    spirv_version_major: u32,
    spirv_version_minor: u32,
    metal_vertex_buffer_base: u32,
    bounds_checks: u32,
}

// The developer-time driver's values, except bounds_checks: checked unless a caller asks for the
// driver's policies (tools/shaders/naga-driver/src/main.rs).
// The largest max_source_bytes that check_options accepts, 16 MiB; the same value as
// CALIPER_WGSL_MAX_SOURCE_BYTES_LIMIT in include/caliper/wgsl.h.
const MAX_SOURCE_BYTES_LIMIT: usize = 16 << 20;

const DEFAULT_OPTIONS: Options = Options {
    max_source_bytes: 1 << 20,
    msl_version_major: 2,
    msl_version_minor: 4,
    spirv_version_major: 1,
    spirv_version_minor: 3,
    metal_vertex_buffer_base: 16,
    bounds_checks: BOUNDS_CHECKED,
};

#[repr(C)]
pub struct Error {
    status: i32,
    line: u32,
    column: u32,
    message: *mut c_char,
    report: *mut c_char,
}

#[repr(C)]
pub struct Buffer {
    data: *mut u8,
    size: usize,
}

#[repr(C)]
pub struct Binding {
    group: u32,
    binding: u32,
    kind: u32,
    access: u32,
    stages: u32,
    metal_index: u32,
    size: u64,
    name: *const c_char,
    storage_format: *const c_char,
}

#[repr(C)]
pub struct EntryPoint {
    name: *const c_char,
    stage: u32,
    workgroup_size: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Io {
    direction: u32,
    location: u32,
    scalar: u32,
    components: u32,
    interpolation: u32,
    sampling: u32,
}

// A failure before it is written into an Error.
struct Failure {
    status: i32,
    location: Option<SourceLocation>,
    message: String,
    report: String,
}

impl Failure {
    fn new(status: i32, message: impl Into<String>) -> Self {
        let message = message.into();
        Failure { status, location: None, report: message.clone(), message }
    }
}

struct Resource {
    global: usize,          // the handle index of the global variable, Naga's member size<global>
    runtime_sized: bool,    // ends in a runtime-sized array, thus Naga's MSL reads its byte size
    group: u32,
    binding: u32,
    kind: u32,
    access: u32,
    stages: u32,
    metal_index: u32,
    size: u64,
    name: CString,
    storage_format: Option<CString>,
}

struct Entry {
    name: CString,
    stage: ShaderStage,
    workgroup_size: [u32; 3],
    io: Vec<Io>,
}

pub struct Module {
    module: naga::Module,
    info: ModuleInfo,
    options: Options,
    resources: Vec<Resource>,
    entries: Vec<Entry>,
}

// The bounds checks of a translation (include/caliper/wgsl.h, caliper_wgsl_bounds_checks).
//
// Checked, the default and the only choice for a program or a stage from a spec: an index into
// a uniform or storage buffer and a texel load out of range read zero, and a store out of range
// is skipped (ReadZeroSkipWrite), because Metal has no robust buffer access and a store past a
// binding lands in other memory of the process, such as a training tensor. A runtime-sized array
// is checked against the byte size that the host binds in the sizes buffer; a fixed-size array
// against its length. Restrict would clamp such a store into the last element of the binding
// and change a valid result. Fixed-size arrays, vectors and matrices in function, private and
// workgroup memory keep Restrict, which WGSL itself specifies: the index is clamped into the
// value, thus no access leaves it. Texture stores have no policy in Naga: Metal and Vulkan
// discard a store out of range. Binding arrays are refused at the module's creation.
//
// Unchecked buffers are the developer-time driver's policies, which the built-in shaders were
// generated with; only a comparison with those generated bytes asks for them.
fn policies(bounds_checks: u32) -> BoundsCheckPolicies {
    if bounds_checks == BOUNDS_UNCHECKED_BUFFERS {
        return BoundsCheckPolicies {
            index: BoundsCheckPolicy::Restrict,
            buffer: BoundsCheckPolicy::Unchecked,
            image_load: BoundsCheckPolicy::Unchecked,
            binding_array: BoundsCheckPolicy::Unchecked,
        };
    }
    BoundsCheckPolicies {
        index: BoundsCheckPolicy::Restrict,
        buffer: BoundsCheckPolicy::ReadZeroSkipWrite,
        image_load: BoundsCheckPolicy::ReadZeroSkipWrite,
        binding_array: BoundsCheckPolicy::ReadZeroSkipWrite,
    }
}

fn stage_bit(s: ShaderStage) -> u32 {
    match s {
        ShaderStage::Vertex => 1 << 0,
        ShaderStage::Fragment => 1 << 1,
        ShaderStage::Compute => 1 << 2,
        _ => 0,
    }
}

fn stage_from_c(stage: i32) -> Option<ShaderStage> {
    match stage {
        0 => Some(ShaderStage::Vertex),
        1 => Some(ShaderStage::Fragment),
        2 => Some(ShaderStage::Compute),
        _ => None,
    }
}

fn stage_name(s: ShaderStage) -> &'static str {
    match s {
        ShaderStage::Vertex => "vertex",
        ShaderStage::Fragment => "fragment",
        ShaderStage::Compute => "compute",
        _ => "other",
    }
}

// The user-defined inputs or outputs of an entry point (include/caliper/wgsl.h, caliper_wgsl_io):
// each value with a @location attribute, as an argument or a result, or as a member of a struct
// that one of them has as its type. Built-in values are left out. An interpolation or a
// sampling that the source leaves out gets WGSL's default, thus `@interpolate(flat)` and
// `@interpolate(flat, first)` compare equal: perspective and center for a float, flat and
// first for an integer.
fn collect_io(module: &naga::Module, binding: Option<&IoBinding>, ty: naga::Handle<naga::Type>, direction: u32, out: &mut Vec<Io>) {
    match binding {
        Some(IoBinding::BuiltIn(_)) => {}
        Some(IoBinding::Location { location, interpolation, sampling, .. }) => {
            let (scalar, components) = match module.types[ty].inner {
                TypeInner::Scalar(s) => (s, 1u32),
                TypeInner::Vector { size, scalar } => (scalar, size as u32),
                _ => {
                    out.push(Io { direction, location: *location, scalar: SCALAR_OTHER, components: 0, interpolation: INTERPOLATION_OTHER, sampling: SAMPLING_CENTER });
                    return;
                }
            };
            let code = match (scalar.kind, scalar.width) {
                (ScalarKind::Float, 4) => SCALAR_F32,
                (ScalarKind::Float, 2) => SCALAR_F16,
                (ScalarKind::Sint, 4) => SCALAR_I32,
                (ScalarKind::Uint, 4) => SCALAR_U32,
                _ => SCALAR_OTHER,
            };
            let float = scalar.kind == ScalarKind::Float;
            let interpolation = match interpolation {
                Some(Interpolation::Perspective) => INTERPOLATION_PERSPECTIVE,
                Some(Interpolation::Linear) => INTERPOLATION_LINEAR,
                Some(Interpolation::Flat) => INTERPOLATION_FLAT,
                Some(_) => INTERPOLATION_OTHER,
                None if float => INTERPOLATION_PERSPECTIVE,
                None => INTERPOLATION_FLAT,
            };
            let sampling = match sampling {
                Some(Sampling::Center) => SAMPLING_CENTER,
                Some(Sampling::Centroid) => SAMPLING_CENTROID,
                Some(Sampling::Sample) => SAMPLING_SAMPLE,
                Some(Sampling::First) => SAMPLING_FIRST,
                Some(Sampling::Either) => SAMPLING_EITHER,
                None if interpolation == INTERPOLATION_FLAT => SAMPLING_FIRST,
                None => SAMPLING_CENTER,
            };
            out.push(Io { direction, location: *location, scalar: code, components, interpolation, sampling });
        }
        None => {
            if let TypeInner::Struct { ref members, .. } = module.types[ty].inner {
                for m in members {
                    collect_io(module, m.binding.as_ref(), m.ty, direction, out);
                }
            }
        }
    }
}

// A C string without interior NULs; Naga's names never hold one, but a panic here would only
// be caught, so strip them rather than fail.
fn c_string(s: &str) -> CString {
    CString::new(s.replace('\0', "")).unwrap_or_default()
}

// Line and column of the first byte that is not UTF-8.
fn utf8_location(bytes: &[u8], offset: usize) -> SourceLocation {
    let before = &bytes[..offset];
    let line_start = before.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    SourceLocation {
        line_number: before.iter().filter(|&&b| b == b'\n').count() as u32 + 1,
        line_position: (offset - line_start) as u32 + 1,
        offset: offset as u32,
        length: 1,
    }
}

fn check_options(o: &Options) -> Result<(), Failure> {
    if o.max_source_bytes > MAX_SOURCE_BYTES_LIMIT {
        return Err(Failure::new(
            STATUS_ARGUMENT,
            format!("max_source_bytes is more than {MAX_SOURCE_BYTES_LIMIT}"),
        ));
    }
    let msl_ok = o.msl_version_major == 1 || o.msl_version_major == 2 || o.msl_version_major == 3;
    let spv_ok = o.spirv_version_major == 1 && o.spirv_version_minor <= 6;
    if !msl_ok || o.msl_version_minor > 9 {
        return Err(Failure::new(STATUS_ARGUMENT, "msl_version is not a Metal Shading Language version"));
    }
    if !spv_ok {
        return Err(Failure::new(STATUS_ARGUMENT, "spirv_version is not a SPIR-V version from 1.0 to 1.6"));
    }
    if !(2..=31).contains(&o.metal_vertex_buffer_base) {
        return Err(Failure::new(STATUS_ARGUMENT, "metal_vertex_buffer_base is not between 2 and 31"));
    }
    if o.bounds_checks != BOUNDS_CHECKED && o.bounds_checks != BOUNDS_UNCHECKED_BUFFERS {
        return Err(Failure::new(STATUS_ARGUMENT, "bounds_checks is not a caliper_wgsl_bounds_checks value"));
    }
    Ok(())
}

// Refuses a source size above max_source_bytes. The size is the caller's claim, thus this runs
// before the source is read.
fn check_source_size(size: usize, options: &Options) -> Result<(), Failure> {
    if size > options.max_source_bytes {
        return Err(Failure::new(
            STATUS_TOO_LARGE,
            format!("the source has {size} bytes, more than the limit of {}", options.max_source_bytes),
        ));
    }
    Ok(())
}

// The options and the size of source are already checked.
fn create(source: &[u8], options: Options) -> Result<Module, Failure> {
    let text = std::str::from_utf8(source).map_err(|e| {
        let mut f = Failure::new(STATUS_PARSE, "the source is not valid UTF-8");
        f.location = Some(utf8_location(source, e.valid_up_to()));
        f
    })?;

    // Parse and validate, then drop what no entry point reaches, as the driver does.
    let mut module = naga::front::wgsl::parse_str(text).map_err(|e| Failure {
        status: STATUS_PARSE,
        location: e.location(text),
        message: e.message().to_string(),
        report: e.emit_to_string(text),
    })?;
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::default());
    let validation = |e: naga::WithSpan<naga::valid::ValidationError>| Failure {
        status: STATUS_VALIDATION,
        location: e.location(text),
        message: e.as_inner().to_string(),
        report: e.emit_to_string(text),
    };
    validator.validate(&module).map_err(validation)?;
    if !module.overrides.is_empty() {
        return Err(Failure::new(
            STATUS_UNSUPPORTED,
            "override declarations are not supported: Naga's MSL writer cannot declare them",
        ));
    }
    naga::compact::compact(&mut module, naga::compact::KeepUnused::No);
    let info = validator.validate(&module).map_err(validation)?;
    let mut layouter = Layouter::default();
    layouter
        .update(module.to_ctx())
        .map_err(|e| Failure::new(STATUS_UNSUPPORTED, format!("layout: {e}")))?;

    // The resources that entry points use, with the driver's Metal binding map.
    let mut resources = Vec::new();
    for (handle, var) in module.global_variables.iter() {
        let Some(rb) = var.binding else { continue };
        let mut stages = 0;
        for (i, ep) in module.entry_points.iter().enumerate() {
            if !info.get_entry_point(i)[handle].is_empty() {
                stages |= stage_bit(ep.stage);
            }
        }
        if stages == 0 {
            continue;
        }
        let name = var.name.clone().unwrap_or_default();
        let unsupported = |what: String| Failure::new(STATUS_UNSUPPORTED, format!("{name}: {what}"));
        let size = layouter[var.ty].size as u64;
        let (kind, access, size, storage_format) = match (var.space, &module.types[var.ty].inner) {
            (AddressSpace::Uniform, _) => (KIND_UNIFORM_BUFFER, ACCESS_READ, size, None),
            (AddressSpace::Storage { access }, _) => {
                let write = if access.contains(naga::StorageAccess::STORE) { ACCESS_WRITE } else { 0 };
                (KIND_STORAGE_BUFFER, ACCESS_READ | write, size, None)
            }
            (AddressSpace::Handle, TypeInner::Image { dim, arrayed, class }) => {
                if *dim != naga::ImageDimension::D2 || *arrayed {
                    return Err(unsupported("only 2-D, non-arrayed textures are supported".into()));
                }
                match class {
                    ImageClass::Sampled { kind: ScalarKind::Float, multi: false } => (KIND_TEXTURE, ACCESS_READ, 0, None),
                    ImageClass::Sampled { kind: ScalarKind::Uint, multi: false } => (KIND_UINT_TEXTURE, ACCESS_READ, 0, None),
                    ImageClass::Storage { format, access } => {
                        let mut a = 0;
                        if access.contains(naga::StorageAccess::LOAD) {
                            a |= ACCESS_READ;
                        }
                        if access.contains(naga::StorageAccess::STORE) {
                            a |= ACCESS_WRITE;
                        }
                        // Naga's variant names are WGSL's texel formats in another case.
                        let f = format!("{format:?}").to_ascii_lowercase();
                        (KIND_STORAGE_TEXTURE, a, 0, Some(c_string(&f)))
                    }
                    other => return Err(unsupported(format!("unsupported texture class {other:?}"))),
                }
            }
            (AddressSpace::Handle, TypeInner::Sampler { comparison: false }) => (KIND_SAMPLER, ACCESS_READ, 0, None),
            _ => return Err(unsupported("unsupported binding".into())),
        };
        resources.push(Resource {
            global: handle.index(),
            runtime_sized: module.types[var.ty].inner.needs_host_buffer_byte_size(&module.types),
            group: rb.group,
            binding: rb.binding,
            kind,
            access,
            stages,
            metal_index: 0,
            size,
            name: c_string(&name),
            storage_format,
        });
    }
    resources.sort_by_key(|r| (r.group, r.binding));
    let class = |k: u32| match k {
        KIND_UNIFORM_BUFFER | KIND_STORAGE_BUFFER => 0usize,
        KIND_SAMPLER => 2,
        _ => 1,
    };
    let sizes_buffer = options.metal_vertex_buffer_base - 1;
    let mut next = [0u32; 3];
    for r in resources.iter_mut() {
        let c = class(r.kind);
        r.metal_index = next[c];
        next[c] += 1;
        let limit = [sizes_buffer, 31, 16][c];
        if r.metal_index >= limit {
            return Err(Failure::new(
                STATUS_UNSUPPORTED,
                format!("{}: Metal index {} is past the limit {}", r.name.to_string_lossy(), r.metal_index, limit),
            ));
        }
    }
    let entries = module
        .entry_points
        .iter()
        .map(|ep| {
            let mut io = Vec::new();
            for arg in &ep.function.arguments {
                collect_io(&module, arg.binding.as_ref(), arg.ty, IO_INPUT, &mut io);
            }
            if let Some(result) = &ep.function.result {
                collect_io(&module, result.binding.as_ref(), result.ty, IO_OUTPUT, &mut io);
            }
            Entry {
                name: c_string(&ep.name),
                stage: ep.stage,
                workgroup_size: if ep.stage == ShaderStage::Compute { ep.workgroup_size } else { [0; 3] },
                io,
            }
        })
        .collect();
    Ok(Module { module, info, options, resources, entries })
}

fn find_entry(m: &Module, name: *const c_char, stage: i32) -> Result<(ShaderStage, String), Failure> {
    let stage = stage_from_c(stage).ok_or_else(|| Failure::new(STATUS_ARGUMENT, "the stage is not vertex, fragment or compute"))?;
    if name.is_null() {
        return Err(Failure::new(STATUS_ARGUMENT, "the entry point name is NULL"));
    }
    // SAFETY: the caller passes a NUL-terminated string, as the header requires.
    let name = unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned();
    if !m.entries.iter().any(|e| e.stage == stage && e.name.to_bytes() == name.as_bytes()) {
        return Err(Failure::new(
            STATUS_ARGUMENT,
            format!("the module has no {} entry point named {}", stage_name(stage), name),
        ));
    }
    Ok((stage, name))
}

fn msl_of(m: &Module, name: *const c_char, stage: i32) -> Result<Vec<u8>, Failure> {
    let (stage, name) = find_entry(m, name, stage)?;
    let mut map = msl::BindingMap::default();
    for r in &m.resources {
        let slot = r.metal_index as u8;
        let target = match r.kind {
            KIND_UNIFORM_BUFFER | KIND_STORAGE_BUFFER => msl::BindTarget { buffer: Some(slot), ..Default::default() },
            KIND_SAMPLER => msl::BindTarget { sampler: Some(msl::BindSamplerTarget::Resource(slot)), ..Default::default() },
            _ => msl::BindTarget { texture: Some(slot), ..Default::default() },
        };
        map.insert(naga::ResourceBinding { group: r.group, binding: r.binding }, target);
    }
    let mut per_entry_point_map = msl::EntryPointResourceMap::default();
    for ep in &m.module.entry_points {
        per_entry_point_map.insert(
            ep.name.clone(),
            msl::EntryPointResources {
                resources: map.clone(),
                immediates_buffer: None,
                sizes_buffer: Some((m.options.metal_vertex_buffer_base - 1) as u8),
            },
        );
    }
    let options = msl::Options {
        lang_version: (m.options.msl_version_major as u8, m.options.msl_version_minor as u8),
        per_entry_point_map,
        fake_missing_bindings: false,
        bounds_check_policies: policies(m.options.bounds_checks),
        ..Default::default()
    };
    let pipeline = msl::PipelineOptions { entry_point: Some((stage, name.clone())), ..Default::default() };
    let (text, tinfo) = msl::write_string(&m.module, &m.info, &options, &pipeline)
        .map_err(|e| Failure::new(STATUS_TRANSLATION, format!("MSL: {e}")))?;
    match tinfo.entry_point_names.first() {
        Some(Ok(n)) if *n == name => {}
        Some(Ok(n)) => return Err(Failure::new(STATUS_TRANSLATION, format!("MSL renamed the entry point {name} to {n}"))),
        Some(Err(e)) => return Err(Failure::new(STATUS_TRANSLATION, format!("MSL entry point {name}: {e}"))),
        None => return Err(Failure::new(STATUS_TRANSLATION, format!("MSL wrote no entry point {name}"))),
    }
    Ok(sizes_layout(text, &m.resources)?.into_bytes())
}

// The struct _mslBufferSizes laid out as the host binds the sizes buffer (src/host/gfx/gfx.h,
// decision 5): one uint for each storage buffer binding, in (group, binding) order. Naga gives
// a member, size<global>, only to the globals that end in a runtime-sized array, in the order of
// their declaration, thus a fixed-size storage binding before one of them, or a declaration out
// of binding order, would make a bounds check read the size of another binding. The members keep
// Naga's names, which the code reads; a storage binding without one gets an unused member. When
// the orders agree, the text is unchanged. A member of Naga's that the new layout lacks fails
// the translation rather than leaving a check on a wrong size.
fn sizes_layout(text: String, resources: &[Resource]) -> Result<String, Failure> {
    const HEAD: &str = "struct _mslBufferSizes {\n";
    let Some(start) = text.find(HEAD) else {
        return if resources.iter().any(|r| r.runtime_sized) {
            Err(Failure::new(STATUS_TRANSLATION, "MSL: the sizes buffer's struct is missing"))
        } else {
            Ok(text)
        };
    };
    let body = start + HEAD.len();
    let end = text[body..]
        .find("};\n")
        .map(|e| body + e)
        .ok_or_else(|| Failure::new(STATUS_TRANSLATION, "MSL: the sizes buffer's struct has no end"))?;
    let mut layout = String::new();
    for r in resources.iter().filter(|r| r.kind == KIND_STORAGE_BUFFER) {
        if r.runtime_sized {
            layout += &format!("    uint size{};\n", r.global);
        } else {
            layout += &format!("    uint unused_{}_{};\n", r.group, r.binding);
        }
    }
    for member in text[body..end].lines() {
        if !layout.lines().any(|l| l == member) {
            return Err(Failure::new(
                STATUS_TRANSLATION,
                format!("MSL: the sizes buffer's member {} has no storage binding", member.trim()),
            ));
        }
    }
    Ok(format!("{}{}{}", &text[..body], layout, &text[end..]))
}

fn spirv_of(m: &Module, name: *const c_char, stage: i32) -> Result<Vec<u8>, Failure> {
    let (stage, name) = find_entry(m, name, stage)?;
    // Groups and bindings become descriptor sets and bindings unchanged, stated explicitly.
    let mut binding_map = spv::BindingMap::default();
    for (_, var) in m.module.global_variables.iter() {
        if let Some(rb) = var.binding {
            binding_map.insert(rb, spv::BindingInfo { descriptor_set: rb.group, binding: rb.binding, binding_array_size: None });
        }
    }
    let options = spv::Options {
        lang_version: (m.options.spirv_version_major as u8, m.options.spirv_version_minor as u8),
        binding_map,
        flags: spv::WriterFlags::LABEL_VARYINGS | spv::WriterFlags::CLAMP_FRAG_DEPTH,
        fake_missing_bindings: false,
        bounds_check_policies: policies(m.options.bounds_checks),
        ..Default::default()
    };
    let pipeline = spv::PipelineOptions { shader_stage: stage, entry_point: name };
    let words = spv::write_vec(&m.module, &m.info, &options, Some(&pipeline))
        .map_err(|e| Failure::new(STATUS_TRANSLATION, format!("SPIR-V: {e}")))?;
    Ok(words.iter().flat_map(|w| w.to_le_bytes()).collect())
}

// Writing results across the boundary.

fn into_c(s: &str) -> *mut c_char {
    c_string(s).into_raw()
}

unsafe fn write_error(error: *mut Error, f: Failure) -> i32 {
    if !error.is_null() {
        let (line, column) = f.location.map_or((0, 0), |l| (l.line_number, l.line_position));
        // SAFETY: the caller passes a writable caliper_wgsl_error or NULL.
        unsafe {
            error.write(Error { status: f.status, line, column, message: into_c(&f.message), report: into_c(&f.report) });
        }
    }
    f.status
}

unsafe fn clear_error(error: *mut Error) {
    if !error.is_null() {
        // SAFETY: as in write_error.
        unsafe {
            error.write(Error { status: STATUS_OK, line: 0, column: 0, message: ptr::null_mut(), report: ptr::null_mut() });
        }
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    let what = payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown cause".into());
    format!("internal error, a panic in the compiler: {what}")
}

// Runs f, catches a panic, and writes a failure into error.
fn guard(error: *mut Error, f: impl FnOnce() -> Result<(), Failure>) -> i32 {
    // SAFETY: error is the caller's pointer, writable or NULL, as the header requires.
    unsafe { clear_error(error) };
    let result = catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|p| Err(Failure::new(STATUS_INTERNAL, panic_message(p))));
    match result {
        Ok(()) => STATUS_OK,
        // SAFETY: as above.
        Err(failure) => unsafe { write_error(error, failure) },
    }
}

// A value-returning entry point; a panic returns the fallback.
fn guard_value<T>(fallback: T, f: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(fallback)
}

// Every buffer is its bytes and one NUL that size does not count, so that MSL text is a C
// string and caliper_wgsl_buffer_free knows the allocation's length, size + 1.
unsafe fn write_buffer(out: *mut Buffer, mut bytes: Vec<u8>) {
    let size = bytes.len();
    bytes.push(0);
    let data = Box::into_raw(bytes.into_boxed_slice()) as *mut u8;
    // SAFETY: out was checked to be non-null by the caller of this function.
    unsafe { out.write(Buffer { data, size }) };
}

fn module_ref<'a>(module: *const Module) -> Result<&'a Module, Failure> {
    // SAFETY: a non-null module comes from caliper_wgsl_module_create and lives until destroy.
    unsafe { module.as_ref() }.ok_or_else(|| Failure::new(STATUS_ARGUMENT, "the module is NULL"))
}

// The exported functions. Each one's contract is in include/caliper/wgsl.h.

#[no_mangle]
pub extern "C" fn caliper_wgsl_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub extern "C" fn caliper_wgsl_naga_version() -> *const c_char {
    NAGA_VERSION.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn caliper_wgsl_options_init(options: *mut Options) {
    if !options.is_null() {
        // SAFETY: the caller passes a writable caliper_wgsl_options.
        unsafe { options.write(DEFAULT_OPTIONS) };
    }
}

#[no_mangle]
pub unsafe extern "C" fn caliper_wgsl_module_create(
    source: *const c_char,
    size: usize,
    options: *const Options,
    module: *mut *mut Module,
    error: *mut Error,
) -> i32 {
    guard(error, || {
        if module.is_null() {
            return Err(Failure::new(STATUS_ARGUMENT, "the module pointer is NULL"));
        }
        // SAFETY: module is non-null and writable, as the header requires.
        unsafe { module.write(ptr::null_mut()) };
        if source.is_null() && size != 0 {
            return Err(Failure::new(STATUS_ARGUMENT, "the source is NULL"));
        }
        // SAFETY: options is NULL or a readable caliper_wgsl_options.
        let options = unsafe { options.as_ref() }.copied().unwrap_or(DEFAULT_OPTIONS);
        check_options(&options)?;
        check_source_size(size, &options)?;
        let bytes: &[u8] = if size == 0 {
            &[]
        } else {
            // SAFETY: source holds size readable bytes, as the header requires.
            unsafe { std::slice::from_raw_parts(source as *const u8, size) }
        };
        let m = create(bytes, options)?;
        // SAFETY: as above.
        unsafe { module.write(Box::into_raw(Box::new(m))) };
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn caliper_wgsl_module_destroy(module: *mut Module) {
    if !module.is_null() {
        // SAFETY: module comes from caliper_wgsl_module_create and is destroyed once.
        guard_value((), || drop(unsafe { Box::from_raw(module) }));
    }
}

#[no_mangle]
pub unsafe extern "C" fn caliper_wgsl_module_binding_count(module: *const Module) -> usize {
    guard_value(0, || module_ref(module).map_or(0, |m| m.resources.len()))
}

#[no_mangle]
pub unsafe extern "C" fn caliper_wgsl_module_binding(module: *const Module, index: usize, binding: *mut Binding) -> i32 {
    guard(ptr::null_mut(), || {
        let m = module_ref(module)?;
        let r = m.resources.get(index).ok_or_else(|| Failure::new(STATUS_ARGUMENT, "the binding index is out of range"))?;
        if binding.is_null() {
            return Err(Failure::new(STATUS_ARGUMENT, "the binding pointer is NULL"));
        }
        // SAFETY: binding is non-null and writable, as the header requires.
        unsafe {
            binding.write(Binding {
                group: r.group,
                binding: r.binding,
                kind: r.kind,
                access: r.access,
                stages: r.stages,
                metal_index: r.metal_index,
                size: r.size,
                name: r.name.as_ptr(),
                storage_format: r.storage_format.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            })
        };
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn caliper_wgsl_module_entry_point_count(module: *const Module) -> usize {
    guard_value(0, || module_ref(module).map_or(0, |m| m.entries.len()))
}

#[no_mangle]
pub unsafe extern "C" fn caliper_wgsl_module_entry_point(module: *const Module, index: usize, entry_point: *mut EntryPoint) -> i32 {
    guard(ptr::null_mut(), || {
        let m = module_ref(module)?;
        let e = m.entries.get(index).ok_or_else(|| Failure::new(STATUS_ARGUMENT, "the entry point index is out of range"))?;
        if entry_point.is_null() {
            return Err(Failure::new(STATUS_ARGUMENT, "the entry point pointer is NULL"));
        }
        let stage = match e.stage {
            ShaderStage::Vertex => 0,
            ShaderStage::Fragment => 1,
            _ => 2,
        };
        // SAFETY: entry_point is non-null and writable, as the header requires.
        unsafe { entry_point.write(EntryPoint { name: e.name.as_ptr(), stage, workgroup_size: e.workgroup_size }) };
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn caliper_wgsl_module_entry_io_count(module: *const Module, entry: usize) -> usize {
    guard_value(0, || module_ref(module).ok().and_then(|m| m.entries.get(entry)).map_or(0, |e| e.io.len()))
}

#[no_mangle]
pub unsafe extern "C" fn caliper_wgsl_module_entry_io(module: *const Module, entry: usize, index: usize, io: *mut Io) -> i32 {
    guard(ptr::null_mut(), || {
        let m = module_ref(module)?;
        let e = m.entries.get(entry).ok_or_else(|| Failure::new(STATUS_ARGUMENT, "the entry point index is out of range"))?;
        let v = e.io.get(index).ok_or_else(|| Failure::new(STATUS_ARGUMENT, "the interface index is out of range"))?;
        if io.is_null() {
            return Err(Failure::new(STATUS_ARGUMENT, "the interface pointer is NULL"));
        }
        // SAFETY: io is non-null and writable, as the header requires.
        unsafe { io.write(*v) };
        Ok(())
    })
}

unsafe fn translate(
    module: *const Module,
    out: *mut Buffer,
    error: *mut Error,
    f: impl FnOnce(&Module) -> Result<Vec<u8>, Failure>,
) -> i32 {
    guard(error, || {
        if out.is_null() {
            return Err(Failure::new(STATUS_ARGUMENT, "the buffer pointer is NULL"));
        }
        // SAFETY: out is non-null and writable, as the header requires.
        unsafe { out.write(Buffer { data: ptr::null_mut(), size: 0 }) };
        let bytes = f(module_ref(module)?)?;
        // SAFETY: out is non-null and writable.
        unsafe { write_buffer(out, bytes) };
        Ok(())
    })
}

#[no_mangle]
pub unsafe extern "C" fn caliper_wgsl_translate_msl(
    module: *const Module,
    name: *const c_char,
    stage: i32,
    msl: *mut Buffer,
    error: *mut Error,
) -> i32 {
    // SAFETY: the pointers follow the header's contract.
    unsafe { translate(module, msl, error, |m| msl_of(m, name, stage)) }
}

#[no_mangle]
pub unsafe extern "C" fn caliper_wgsl_translate_spirv(
    module: *const Module,
    name: *const c_char,
    stage: i32,
    spirv: *mut Buffer,
    error: *mut Error,
) -> i32 {
    // SAFETY: the pointers follow the header's contract.
    unsafe { translate(module, spirv, error, |m| spirv_of(m, name, stage)) }
}

#[no_mangle]
pub unsafe extern "C" fn caliper_wgsl_buffer_free(buffer: *mut Buffer) {
    // SAFETY: buffer is NULL or a buffer this library filled, freed once.
    let Some(b) = (unsafe { buffer.as_mut() }) else { return };
    if !b.data.is_null() {
        let len = b.size + 1;
        // SAFETY: data and len describe the boxed slice that write_buffer leaked.
        guard_value((), || drop(unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(b.data, len)) }));
    }
    b.data = ptr::null_mut();
    b.size = 0;
}

#[no_mangle]
pub unsafe extern "C" fn caliper_wgsl_error_free(error: *mut Error) {
    // SAFETY: error is NULL or an error this library filled, freed once.
    let Some(e) = (unsafe { error.as_mut() }) else { return };
    for p in [&mut e.message, &mut e.report] {
        if !p.is_null() {
            // SAFETY: the string comes from CString::into_raw in write_error.
            drop(unsafe { CString::from_raw(*p) });
            *p = ptr::null_mut();
        }
    }
    e.line = 0;
    e.column = 0;
}

// Only in a build with the test-hooks feature; the shipped library has no such function.
#[cfg(feature = "test-hooks")]
#[no_mangle]
pub unsafe extern "C" fn caliper_wgsl_test_panic(error: *mut Error) -> i32 {
    guard(error, || -> Result<(), Failure> { panic!("caliper_wgsl_test_panic") })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_error() -> Error {
        Error { status: STATUS_OK, line: 0, column: 0, message: ptr::null_mut(), report: ptr::null_mut() }
    }

    fn message_of(error: &Error) -> String {
        assert!(!error.message.is_null());
        // SAFETY: message is a string that write_error made.
        unsafe { CStr::from_ptr(error.message) }.to_string_lossy().into_owned()
    }

    #[test]
    fn guard_turns_a_panic_into_the_internal_status() {
        let mut error = empty_error();
        let status = guard(&mut error, || -> Result<(), Failure> { panic!("boom") });
        assert_eq!(status, STATUS_INTERNAL);
        assert_eq!(error.status, STATUS_INTERNAL);
        assert!(message_of(&error).contains("boom"));
        // SAFETY: error was filled by this library.
        unsafe { caliper_wgsl_error_free(&mut error) };
    }

    #[test]
    fn max_source_bytes_above_the_limit_is_an_argument_error() {
        let mut options = DEFAULT_OPTIONS;
        options.max_source_bytes = MAX_SOURCE_BYTES_LIMIT + 1;
        let source = b"a";
        let mut module: *mut Module = ptr::null_mut();
        let mut error = empty_error();
        // SAFETY: the pointers are valid.
        let status = unsafe {
            caliper_wgsl_module_create(source.as_ptr() as *const c_char, source.len(), &options, &mut module, &mut error)
        };
        assert_eq!(status, STATUS_ARGUMENT);
        assert!(module.is_null());
        assert!(message_of(&error).contains("max_source_bytes is more than 16777216"));
        // SAFETY: error was filled by this library.
        unsafe { caliper_wgsl_error_free(&mut error) };
    }

    #[test]
    fn a_size_of_usize_max_is_too_large_and_reads_nothing() {
        let byte = b'a';
        let mut module: *mut Module = ptr::null_mut();
        let mut error = empty_error();
        // SAFETY: the pointers are valid; the library must reject the size before it reads the source.
        let status = unsafe {
            caliper_wgsl_module_create(&byte as *const u8 as *const c_char, usize::MAX, ptr::null(), &mut module, &mut error)
        };
        assert_eq!(status, STATUS_TOO_LARGE);
        assert!(module.is_null());
        // SAFETY: error was filled by this library.
        unsafe { caliper_wgsl_error_free(&mut error) };
    }

    // A geometry program whose fixed-size counter comes first and whose arrays are declared out
    // of binding order: Naga's sizes struct would hold w's size before the counter's place.
    const OUT_OF_ORDER: &str = "@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(0) var<storage, read_write> count: atomic<u32>;
@group(0) @binding(1) var<storage, read_write> values: array<f32>;
@compute @workgroup_size(64)
fn run(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = atomicAdd(&count, 1u);
    values[id.x * 1000u] = w[id.x + 7u];
    var local: array<f32, 4>;
    local[id.x] = 1.0;
    values[i] = local[id.y];
}
";

    fn module_of(source: &str, bounds_checks: u32) -> Module {
        let mut options = DEFAULT_OPTIONS;
        options.bounds_checks = bounds_checks;
        create(source.as_bytes(), options).unwrap_or_else(|f| panic!("{}", f.report))
    }

    fn msl_text(m: &Module) -> String {
        let name = CString::new("run").unwrap();
        String::from_utf8(msl_of(m, name.as_ptr(), 2).unwrap_or_else(|f| panic!("{}", f.message))).unwrap()
    }

    #[test]
    fn checked_translation_guards_buffer_stores_and_loads() {
        let checked = msl_text(&module_of(OUT_OF_ORDER, BOUNDS_CHECKED));
        // The store into values and the load from w run only inside their binding's size.
        assert!(checked.contains("_buffer_sizes.size"), "{checked}");
        assert!(checked.contains("DefaultConstructible"), "{checked}");
        // The local array keeps Restrict: its index is clamped, never checked against a size.
        assert!(checked.contains("metal::min(unsigned("), "{checked}");
        let unchecked = msl_text(&module_of(OUT_OF_ORDER, BOUNDS_UNCHECKED_BUFFERS));
        assert!(!unchecked.contains("DefaultConstructible"), "{unchecked}");
        assert_ne!(checked, unchecked);
    }

    #[test]
    fn sizes_struct_follows_the_binding_order_of_the_host() {
        let m = module_of(OUT_OF_ORDER, BOUNDS_CHECKED);
        let text = msl_text(&m);
        let w = m.resources.iter().find(|r| r.name.to_bytes() == b"w").unwrap().global;
        let values = m.resources.iter().find(|r| r.name.to_bytes() == b"values").unwrap().global;
        let expected = format!(
            "struct _mslBufferSizes {{\n    uint unused_0_0;\n    uint size{values};\n    uint size{w};\n}};\n"
        );
        assert!(text.contains(&expected), "{text}");
    }

    #[test]
    fn sizes_struct_is_unchanged_when_the_orders_agree() {
        let source = "@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(1)
fn run(@builtin(global_invocation_id) id: vec3<u32>) { b[id.x] = a[id.x]; }
";
        let m = module_of(source, BOUNDS_CHECKED);
        let name = CString::new("run").unwrap();
        let mut options = msl::Options::default();
        let mut per = msl::EntryPointResources::default();
        per.sizes_buffer = Some(15);
        for r in &m.resources {
            per.resources.insert(
                naga::ResourceBinding { group: r.group, binding: r.binding },
                msl::BindTarget { buffer: Some(r.metal_index as u8), ..Default::default() },
            );
        }
        options.per_entry_point_map.insert("run".into(), per);
        options.lang_version = (2, 4);
        options.bounds_check_policies = policies(BOUNDS_CHECKED);
        let pipeline = msl::PipelineOptions { entry_point: Some((ShaderStage::Compute, "run".into())), ..Default::default() };
        let (naga_text, _) = msl::write_string(&m.module, &m.info, &options, &pipeline).unwrap();
        let ours = String::from_utf8(msl_of(&m, name.as_ptr(), 2).unwrap_or_else(|f| panic!("{}", f.message))).unwrap();
        assert_eq!(ours, naga_text);
    }

    #[test]
    fn checked_spirv_reads_the_array_length() {
        let m = module_of(OUT_OF_ORDER, BOUNDS_CHECKED);
        let name = CString::new("run").unwrap();
        let words = |m: &Module| -> Vec<u32> {
            spirv_of(m, name.as_ptr(), 2)
                .unwrap_or_else(|f| panic!("{}", f.message))
                .chunks(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        // OpArrayLength (68) bounds the runtime-sized arrays; the driver's policies never ask for it.
        let has_array_length = |w: &[u32]| {
            let mut at = 5; // after the header, one instruction at a time
            while at < w.len() {
                if w[at] & 0xffff == 68 {
                    return true;
                }
                at += ((w[at] >> 16) as usize).max(1);
            }
            false
        };
        assert!(has_array_length(&words(&m)));
        assert!(!has_array_length(&words(&module_of(OUT_OF_ORDER, BOUNDS_UNCHECKED_BUFFERS))));
    }

    #[test]
    fn an_unknown_bounds_check_value_is_an_argument_error() {
        let mut options = DEFAULT_OPTIONS;
        options.bounds_checks = 2;
        assert_eq!(check_options(&options).err().map(|f| f.status), Some(STATUS_ARGUMENT));
    }

    const STAGES: &str = r#"
struct VsOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(2) @interpolate(flat) item: u32,
    @location(3) @interpolate(linear, centroid) t: f32,
}
struct Picked {
    @location(0) color: vec4<f32>,
    @location(1) id: vec2<u32>,
}
@vertex
fn vs(@builtin(vertex_index) v: u32) -> VsOut {
    var o: VsOut;
    o.position = vec4<f32>(f32(v), 0.0, 0.0, 1.0);
    o.item = v;
    return o;
}
@fragment
fn fs(in: VsOut, @location(5) @interpolate(flat, first) extra: vec2<i32>) -> Picked {
    var p: Picked;
    p.color = in.color * in.t + vec4<f32>(f32(extra.x));
    p.id = vec2<u32>(in.item, 0u);
    return p;
}
@fragment
fn plain() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
"#;

    #[test]
    fn entry_points_list_their_locations_with_defaults_and_without_builtins() {
        let m = module_of(STAGES, BOUNDS_CHECKED);
        let io = |name: &str| m.entries.iter().find(|e| e.name.to_str() == Ok(name)).map(|e| e.io.clone()).unwrap();
        let v = |direction, location, scalar, components, interpolation, sampling| Io {
            direction,
            location,
            scalar,
            components,
            interpolation,
            sampling,
        };
        assert_eq!(
            io("vs"),
            vec![
                v(IO_OUTPUT, 0, SCALAR_F32, 4, INTERPOLATION_PERSPECTIVE, SAMPLING_CENTER),
                v(IO_OUTPUT, 2, SCALAR_U32, 1, INTERPOLATION_FLAT, SAMPLING_FIRST),
                v(IO_OUTPUT, 3, SCALAR_F32, 1, INTERPOLATION_LINEAR, SAMPLING_CENTROID),
            ]
        );
        assert_eq!(
            io("fs"),
            vec![
                v(IO_INPUT, 0, SCALAR_F32, 4, INTERPOLATION_PERSPECTIVE, SAMPLING_CENTER),
                v(IO_INPUT, 2, SCALAR_U32, 1, INTERPOLATION_FLAT, SAMPLING_FIRST),
                v(IO_INPUT, 3, SCALAR_F32, 1, INTERPOLATION_LINEAR, SAMPLING_CENTROID),
                v(IO_INPUT, 5, SCALAR_I32, 2, INTERPOLATION_FLAT, SAMPLING_FIRST),
                v(IO_OUTPUT, 0, SCALAR_F32, 4, INTERPOLATION_PERSPECTIVE, SAMPLING_CENTER),
                v(IO_OUTPUT, 1, SCALAR_U32, 2, INTERPOLATION_FLAT, SAMPLING_FIRST),
            ]
        );
        assert_eq!(io("plain"), vec![v(IO_OUTPUT, 0, SCALAR_F32, 4, INTERPOLATION_PERSPECTIVE, SAMPLING_CENTER)]);
    }
}
