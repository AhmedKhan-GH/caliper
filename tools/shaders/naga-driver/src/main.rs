//! caliper-naga-driver: translates one assembled WGSL program with Naga and describes it.
//!
//!     caliper-naga-driver INPUT.wgsl [--msl OUT.metal] [--spv OUT.spv] [--reflect OUT.json]
//!
//! It exists because Naga's command line has no way to pass a Metal binding map: without one,
//! Naga's MSL writer gives every resource the placeholder attribute [[user(fake0)]], which
//! Metal refuses. The driver applies Caliper's rule (src/host/gfx/gfx.h, decision 5):
//!
//! - every binding that an entry point uses gets one Metal index, the same in every stage;
//! - buffers (uniform and storage), textures and samplers are numbered separately, each from
//!   0, in (group, binding) order across the whole program;
//! - buffer indices stay below METAL_VERTEX_BUFFER_BASE, where vertex buffer slot i is Metal
//!   buffer index METAL_VERTEX_BUFFER_BASE + i, read through [[stage_in]].
//!
//! The SPIR-V keeps WGSL's groups and bindings as descriptor sets and bindings, and WGSL's
//! clip space (no y flip): the Vulkan backend flips with a negative viewport height. The
//! description (--reflect) is JSON: the bindings with their Metal indices, the entry points,
//! their vertex inputs and fragment outputs, and the layout of every uniform block, which the
//! generator turns into C++. A parse or validation error is printed with its source location
//! and the exit status is 1. The output is a function of the input and the pinned Naga only.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::process::ExitCode;

use naga::back::{msl, spv};
use naga::proc::{BoundsCheckPolicies, BoundsCheckPolicy, Layouter};
use naga::valid::{Capabilities, ModuleInfo, ValidationFlags, Validator};
use naga::{AddressSpace, Binding, ImageClass, Module, ScalarKind, ShaderStage, TypeInner};

const METAL_VERTEX_BUFFER_BASE: u32 = 16;
// Naga's MSL gives every entry point that uses a runtime-sized array an extra argument, a
// buffer of the byte sizes of those arrays, whatever the bounds check policy. It takes the
// last buffer index below the vertex buffers.
const METAL_SIZES_BUFFER: u32 = METAL_VERTEX_BUFFER_BASE - 1;
const MSL_VERSION: (u8, u8) = (2, 4);
const SPV_VERSION: (u8, u8) = (1, 3);

// Index checks: fixed-size arrays, vectors and matrices are clamped (Restrict), which costs a
// min() and needs no extra binding; runtime-sized buffer arrays, images and binding arrays are
// left unchecked, so no buffer of sizes is bound. The programs keep their indices in range.
fn policies() -> BoundsCheckPolicies {
    BoundsCheckPolicies {
        index: BoundsCheckPolicy::Restrict,
        buffer: BoundsCheckPolicy::Unchecked,
        image_load: BoundsCheckPolicy::Unchecked,
        binding_array: BoundsCheckPolicy::Unchecked,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Buffer,
    Texture,
    Sampler,
}

struct Resource {
    handle: naga::Handle<naga::GlobalVariable>,
    group: u32,
    binding: u32,
    name: String,
    kind: Kind,
    ty: &'static str, // the BindingType name of gfx.h
    size: u32,
    storage_format: Option<String>,
    stages: Vec<&'static str>,
    msl_index: u32,
}

fn fail(msg: impl AsRef<str>) -> ExitCode {
    eprintln!("caliper-naga-driver: {}", msg.as_ref());
    ExitCode::from(1)
}

fn stage_name(s: ShaderStage) -> &'static str {
    match s {
        ShaderStage::Vertex => "vertex",
        ShaderStage::Fragment => "fragment",
        ShaderStage::Compute => "compute",
        _ => "other",
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => {
                let _ = write!(o, "\\u{:04x}", c as u32);
            }
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn scalar_name(s: naga::Scalar) -> String {
    match (s.kind, s.width) {
        (ScalarKind::Float, 4) => "f32".into(),
        (ScalarKind::Float, 2) => "f16".into(),
        (ScalarKind::Uint, 4) => "u32".into(),
        (ScalarKind::Sint, 4) => "i32".into(),
        (ScalarKind::Bool, _) => "bool".into(),
        (k, w) => format!("{:?}{}", k, w * 8),
    }
}

fn vsize(v: naga::VectorSize) -> u32 {
    v as u32
}

// The WGSL spelling of a type, for vertex inputs and fragment outputs.
fn type_name(module: &Module, ty: naga::Handle<naga::Type>) -> String {
    match &module.types[ty].inner {
        TypeInner::Scalar(s) => scalar_name(*s),
        TypeInner::Vector { size, scalar } => format!("vec{}<{}>", vsize(*size), scalar_name(*scalar)),
        TypeInner::Matrix { columns, rows, scalar } => {
            format!("mat{}x{}<{}>", vsize(*columns), vsize(*rows), scalar_name(*scalar))
        }
        other => format!("{:?}", other),
    }
}

// The layout of a host-shareable type, as JSON, for the C++ structs.
fn layout_json(module: &Module, layouter: &Layouter, ty: naga::Handle<naga::Type>) -> String {
    let t = &module.types[ty];
    let size = layouter[ty].size;
    match &t.inner {
        TypeInner::Scalar(s) => format!("{{\"kind\": \"scalar\", \"scalar\": \"{}\", \"size\": {}}}", scalar_name(*s), size),
        TypeInner::Vector { size: n, scalar } => format!(
            "{{\"kind\": \"vector\", \"scalar\": \"{}\", \"count\": {}, \"size\": {}}}",
            scalar_name(*scalar), vsize(*n), size
        ),
        TypeInner::Matrix { columns, rows, scalar } => format!(
            "{{\"kind\": \"matrix\", \"scalar\": \"{}\", \"columns\": {}, \"rows\": {}, \"size\": {}}}",
            scalar_name(*scalar), vsize(*columns), vsize(*rows), size
        ),
        TypeInner::Array { base, size: n, stride } => {
            let count = match n {
                naga::ArraySize::Constant(c) => c.get().to_string(),
                _ => "0".into(), // runtime-sized
            };
            format!(
                "{{\"kind\": \"array\", \"count\": {}, \"stride\": {}, \"size\": {}, \"element\": {}}}",
                count, stride, size, layout_json(module, layouter, *base)
            )
        }
        TypeInner::Struct { members, span } => {
            let mut o = format!(
                "{{\"kind\": \"struct\", \"name\": {}, \"size\": {}, \"members\": [",
                json_str(t.name.as_deref().unwrap_or("")),
                span
            );
            for (i, m) in members.iter().enumerate() {
                if i > 0 {
                    o.push_str(", ");
                }
                let _ = write!(
                    o,
                    "{{\"name\": {}, \"offset\": {}, \"type\": {}}}",
                    json_str(m.name.as_deref().unwrap_or("")),
                    m.offset,
                    layout_json(module, layouter, m.ty)
                );
            }
            o.push_str("]}");
            o
        }
        TypeInner::Atomic(s) => format!("{{\"kind\": \"atomic\", \"scalar\": \"{}\", \"size\": {}}}", scalar_name(*s), size),
        other => format!("{{\"kind\": \"unsupported\", \"what\": {}}}", json_str(&format!("{:?}", other))),
    }
}

// Locations an entry point reads (vertex inputs) or writes (fragment outputs).
fn locations(module: &Module, ty: naga::Handle<naga::Type>, binding: &Option<Binding>, out: &mut Vec<(u32, String)>) {
    match binding {
        Some(Binding::Location { location, .. }) => out.push((*location, type_name(module, ty))),
        Some(_) => {}
        None => {
            if let TypeInner::Struct { members, .. } = &module.types[ty].inner {
                for m in members {
                    locations(module, m.ty, &m.binding, out);
                }
            }
        }
    }
}

fn storage_format_name(f: naga::StorageFormat) -> String {
    let s = format!("{:?}", f); // R8Unorm, Rgba8Unorm, R32Uint, ...
    s
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut input = None;
    let (mut msl_out, mut spv_out, mut reflect_out) = (None, None, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--msl" => msl_out = it.next().cloned(),
            "--spv" => spv_out = it.next().cloned(),
            "--reflect" => reflect_out = it.next().cloned(),
            "--version" => {
                println!("caliper-naga-driver on naga {}", naga_version());
                return ExitCode::SUCCESS;
            }
            s if !s.starts_with("--") && input.is_none() => input = Some(s.to_string()),
            s => return fail(format!("unknown argument {}", s)),
        }
    }
    let Some(input) = input else {
        return fail("usage: caliper-naga-driver INPUT.wgsl [--msl OUT] [--spv OUT] [--reflect OUT]");
    };
    let source = match std::fs::read_to_string(&input) {
        Ok(s) => s,
        Err(e) => return fail(format!("{}: {}", input, e)),
    };

    // Parse and validate as WGSL, then drop what no entry point reaches.
    let mut module = match naga::front::wgsl::parse_str(&source) {
        Ok(m) => m,
        Err(e) => {
            eprint!("{}", e.emit_to_string_with_path(&source, &input));
            return ExitCode::from(1);
        }
    };
    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::default());
    if let Err(e) = validator.validate(&module) {
        eprint!("{}", e.emit_to_string_with_path(&source, &input));
        return ExitCode::from(1);
    }
    if !module.overrides.is_empty() {
        // Naga's MSL writer refuses a module with overrides until they are replaced by values.
        return fail("override declarations are not supported: Naga's MSL writer cannot declare them");
    }
    naga::compact::compact(&mut module, naga::compact::KeepUnused::No);
    let info: ModuleInfo = match validator.validate(&module) {
        Ok(i) => i,
        Err(e) => {
            eprint!("{}", e.emit_to_string_with_path(&source, &input));
            return ExitCode::from(1);
        }
    };
    let mut layouter = Layouter::default();
    if let Err(e) = layouter.update(module.to_ctx()) {
        return fail(format!("layout: {:?}", e));
    }

    // The resources entry points use, with Caliper's Metal binding map.
    let mut resources: Vec<Resource> = Vec::new();
    for (handle, var) in module.global_variables.iter() {
        let Some(rb) = var.binding else { continue };
        let mut stages = Vec::new();
        for (i, ep) in module.entry_points.iter().enumerate() {
            if !info.get_entry_point(i)[handle].is_empty() && !stages.contains(&stage_name(ep.stage)) {
                stages.push(stage_name(ep.stage));
            }
        }
        if stages.is_empty() {
            continue;
        }
        let name = var.name.clone().unwrap_or_default();
        let inner = &module.types[var.ty].inner;
        let (kind, ty, size, storage_format) = match (var.space, inner) {
            (AddressSpace::Uniform, _) => (Kind::Buffer, "Uniform", layouter[var.ty].size, None),
            (AddressSpace::Storage { access }, _) => {
                let rw = access.contains(naga::StorageAccess::STORE);
                (Kind::Buffer, if rw { "StorageReadWrite" } else { "Storage" }, layouter[var.ty].size, None)
            }
            (AddressSpace::Handle, TypeInner::Image { dim, arrayed, class }) => {
                if *dim != naga::ImageDimension::D2 || *arrayed {
                    return fail(format!("{}: only 2-D, non-arrayed textures are supported", name));
                }
                match class {
                    ImageClass::Sampled { kind: ScalarKind::Float, multi: false } => (Kind::Texture, "Texture", 0, None),
                    ImageClass::Sampled { kind: ScalarKind::Uint, multi: false } => (Kind::Texture, "UintTexture", 0, None),
                    ImageClass::Storage { format, .. } => {
                        (Kind::Texture, "StorageTexture", 0, Some(storage_format_name(*format)))
                    }
                    other => return fail(format!("{}: unsupported texture class {:?}", name, other)),
                }
            }
            (AddressSpace::Handle, TypeInner::Sampler { comparison: false }) => (Kind::Sampler, "Sampler", 0, None),
            _ => return fail(format!("{}: unsupported binding", name)),
        };
        resources.push(Resource {
            handle,
            group: rb.group,
            binding: rb.binding,
            name,
            kind,
            ty,
            size,
            storage_format,
            stages,
            msl_index: 0,
        });
    }
    resources.sort_by_key(|r| (r.group, r.binding));
    let mut next: BTreeMap<u8, u32> = BTreeMap::new();
    for r in resources.iter_mut() {
        let n = next.entry(r.kind as u8).or_insert(0);
        r.msl_index = *n;
        *n += 1;
    }
    for r in &resources {
        let limit = match r.kind {
            Kind::Buffer => METAL_SIZES_BUFFER,
            Kind::Texture => 31,
            Kind::Sampler => 16,
        };
        if r.msl_index >= limit {
            return fail(format!("{}: Metal index {} is past the limit {}", r.name, r.msl_index, limit));
        }
    }

    if let Some(path) = &msl_out {
        let mut map = msl::BindingMap::default();
        for r in &resources {
            let slot = r.msl_index as u8;
            let target = match r.kind {
                Kind::Buffer => msl::BindTarget { buffer: Some(slot), ..Default::default() },
                Kind::Texture => msl::BindTarget { texture: Some(slot), ..Default::default() },
                Kind::Sampler => msl::BindTarget { sampler: Some(msl::BindSamplerTarget::Resource(slot)), ..Default::default() },
            };
            map.insert(naga::ResourceBinding { group: r.group, binding: r.binding }, target);
        }
        let mut per_entry_point_map = msl::EntryPointResourceMap::default();
        for ep in &module.entry_points {
            per_entry_point_map.insert(
                ep.name.clone(),
                msl::EntryPointResources {
                    resources: map.clone(),
                    immediates_buffer: None,
                    sizes_buffer: Some(METAL_SIZES_BUFFER as u8),
                },
            );
        }
        let options = msl::Options {
            lang_version: MSL_VERSION,
            per_entry_point_map,
            fake_missing_bindings: false,
            bounds_check_policies: policies(),
            ..Default::default()
        };
        let (text, tinfo) = match msl::write_string(&module, &info, &options, &msl::PipelineOptions::default()) {
            Ok(r) => r,
            Err(e) => return fail(format!("MSL: {:?}", e)),
        };
        for (ep, name) in module.entry_points.iter().zip(tinfo.entry_point_names.iter()) {
            match name {
                Ok(n) if *n == ep.name => {}
                Ok(n) => return fail(format!("MSL renamed the entry point {} to {}", ep.name, n)),
                Err(e) => return fail(format!("MSL entry point {}: {:?}", ep.name, e)),
            }
        }
        if let Err(e) = std::fs::write(path, text) {
            return fail(format!("{}: {}", path, e));
        }
    }

    if let Some(path) = &spv_out {
        // Groups and bindings become descriptor sets and bindings unchanged, stated explicitly.
        let mut binding_map = spv::BindingMap::default();
        for (_, var) in module.global_variables.iter() {
            if let Some(rb) = var.binding {
                binding_map.insert(
                    rb,
                    spv::BindingInfo { descriptor_set: rb.group, binding: rb.binding, binding_array_size: None },
                );
            }
        }
        let options = spv::Options {
            lang_version: SPV_VERSION,
            binding_map,
            flags: spv::WriterFlags::LABEL_VARYINGS | spv::WriterFlags::CLAMP_FRAG_DEPTH,
            fake_missing_bindings: false,
            bounds_check_policies: policies(),
            ..Default::default()
        };
        let words = match spv::write_vec(&module, &info, &options, None) {
            Ok(w) => w,
            Err(e) => return fail(format!("SPIR-V: {:?}", e)),
        };
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        if let Err(e) = std::fs::write(path, bytes) {
            return fail(format!("{}: {}", path, e));
        }
    }

    if let Some(path) = &reflect_out {
        let mut o = String::new();
        let _ = writeln!(o, "{{");
        let _ = writeln!(o, "  \"naga\": {},", json_str(naga_version()));
        let _ = writeln!(o, "  \"msl_version\": \"{}.{}\",", MSL_VERSION.0, MSL_VERSION.1);
        let _ = writeln!(o, "  \"spirv_version\": \"{}.{}\",", SPV_VERSION.0, SPV_VERSION.1);
        let _ = writeln!(o, "  \"metal_vertex_buffer_base\": {},", METAL_VERTEX_BUFFER_BASE);
        let _ = writeln!(o, "  \"metal_sizes_buffer\": {},", METAL_SIZES_BUFFER);
        let _ = writeln!(o, "  \"bindings\": [");
        for (i, r) in resources.iter().enumerate() {
            let block = match module.global_variables[r.handle].space {
                AddressSpace::Uniform => layout_json(&module, &layouter, module.global_variables[r.handle].ty),
                _ => "null".into(),
            };
            let stages: Vec<String> = r.stages.iter().map(|s| json_str(s)).collect();
            let _ = writeln!(
                o,
                "    {{\"group\": {}, \"binding\": {}, \"name\": {}, \"type\": \"{}\", \"stages\": [{}], \"size\": {}, \"storage_format\": {}, \"msl_index\": {}, \"block\": {}}}{}",
                r.group, r.binding, json_str(&r.name), r.ty, stages.join(", "), r.size,
                r.storage_format.as_deref().map(json_str).unwrap_or_else(|| "null".into()),
                r.msl_index, block,
                if i + 1 < resources.len() { "," } else { "" }
            );
        }
        let _ = writeln!(o, "  ],");
        let _ = writeln!(o, "  \"entries\": [");
        for (i, ep) in module.entry_points.iter().enumerate() {
            let fi = info.get_entry_point(i);
            let uses: Vec<String> = resources
                .iter()
                .filter(|r| !fi[r.handle].is_empty())
                .map(|r| format!("[{}, {}]", r.group, r.binding))
                .collect();
            let mut inputs = Vec::new();
            if ep.stage == ShaderStage::Vertex {
                for a in &ep.function.arguments {
                    locations(&module, a.ty, &a.binding, &mut inputs);
                }
            }
            inputs.sort();
            let mut outputs = Vec::new();
            if ep.stage == ShaderStage::Fragment {
                if let Some(res) = &ep.function.result {
                    locations(&module, res.ty, &res.binding, &mut outputs);
                }
            }
            outputs.sort();
            let fmt = |v: &Vec<(u32, String)>| -> String {
                v.iter().map(|(l, t)| format!("{{\"location\": {}, \"type\": {}}}", l, json_str(t))).collect::<Vec<_>>().join(", ")
            };
            let _ = writeln!(
                o,
                "    {{\"name\": {}, \"stage\": \"{}\", \"workgroup_size\": [{}, {}, {}], \"uses\": [{}], \"inputs\": [{}], \"outputs\": [{}]}}{}",
                json_str(&ep.name), stage_name(ep.stage),
                ep.workgroup_size[0], ep.workgroup_size[1], ep.workgroup_size[2],
                uses.join(", "), fmt(&inputs), fmt(&outputs),
                if i + 1 < module.entry_points.len() { "," } else { "" }
            );
        }
        let _ = writeln!(o, "  ]");
        let _ = writeln!(o, "}}");
        if let Err(e) = std::fs::write(path, o) {
            return fail(format!("{}: {}", path, e));
        }
    }
    ExitCode::SUCCESS
}

// The pinned version, as Cargo.toml requires it exactly.
fn naga_version() -> &'static str {
    "30.0.2"
}
