// The page half of shaders.spec.ts: raw WebGPU, with no view code, so that the generated
// programs are checked before the graphics layer's WebGPU backend exists. Each program's bind
// group layouts are built from its generated description (programs.json), as the layer will.

const STAGE = { vertex: 1, fragment: 2, compute: 4 };
const VERTEX = { 'f32': ['float32', 4], 'vec2<f32>': ['float32x2', 8], 'vec3<f32>': ['float32x3', 12],
  'vec4<f32>': ['float32x4', 16], 'u32': ['uint32', 4], 'i32': ['sint32', 4], 'vec2<u32>': ['uint32x2', 8] };
const TARGET = { 'vec4<f32>': 'rgba8unorm', 'vec2<u32>': 'rg32uint' };

/** A device on the adapter the browser offers: the default level, or compatibility mode. */
async function device(level) {
  const adapter = await navigator.gpu.requestAdapter(level === 'compatibility' ? { featureLevel: 'compatibility' } : {});
  if (!adapter) throw new Error(`no ${level} adapter`);
  const dev = await adapter.requestDevice();
  const i = adapter.info || {};
  return { dev, adapter: `${i.vendor} ${i.architecture} ${i.description}`.trim(),
           core: dev.features.has('core-features-and-limits'), fallback: !!i.isFallbackAdapter };
}

function entryOf(binding) {
  const e = { binding: binding.binding, visibility: binding.stages.reduce((a, s) => a | STAGE[s], 0) };
  switch (binding.type) {
    case 'Uniform': e.buffer = { type: 'uniform', hasDynamicOffset: binding.dynamic, minBindingSize: binding.size }; break;
    case 'Storage': e.buffer = { type: 'read-only-storage', minBindingSize: binding.size }; break;
    case 'StorageReadWrite': e.buffer = { type: 'storage', minBindingSize: binding.size }; break;
    case 'Texture': e.texture = { sampleType: 'float' }; break;
    case 'UintTexture': e.texture = { sampleType: 'uint' }; break;
    case 'Sampler': e.sampler = { type: 'filtering' }; break;
    case 'StorageTexture': e.storageTexture = { access: 'write-only', format: binding.storage_format.toLowerCase() }; break;
    default: throw new Error(`binding type ${binding.type}`);
  }
  return e;
}

/** The program's bind group layouts, one per group up to the last it uses, and its pipeline layout. */
function layouts(dev, prog) {
  const last = Math.max(-1, ...prog.bindings.map((b) => b.group));
  const groups = [];
  for (let g = 0; g <= last; g++)
    groups.push(dev.createBindGroupLayout({ entries: prog.bindings.filter((b) => b.group === g).map(entryOf) }));
  return { groups, layout: dev.createPipelineLayout({ bindGroupLayouts: groups }) };
}

/**
 * Vertex buffer layouts for an entry: by default all its inputs in one buffer, one after the
 * other (enough to make a pipeline); or as given, [{ stride, step, attrs: [[location, offset]] }].
 */
function vertexBuffers(prog, entry, given) {
  const e = prog.entries.find((x) => x.name === entry);
  const fmt = Object.fromEntries(e.inputs.map((i) => [i.location, VERTEX[i.type]]));
  if (given) {
    return given.map((b) => ({ arrayStride: b.stride, stepMode: b.step || 'instance',
      attributes: b.attrs.map(([loc, offset]) => ({ shaderLocation: loc, offset, format: fmt[loc][0] })) }));
  }
  if (!e.inputs.length) return [];
  let off = 0;
  const attributes = e.inputs.map((i) => { const a = { shaderLocation: i.location, offset: off, format: VERTEX[i.type][0] }; off += VERTEX[i.type][1]; return a; });
  return [{ arrayStride: off, stepMode: 'vertex', attributes }];
}

function targetOf(prog, entry) {
  return TARGET[prog.entries.find((x) => x.name === entry).outputs[0].type];
}

/** Compiles one program and makes its pipelines; returns the messages and counts. */
async function compileProgram(dev, prog, code) {
  const messages = [];
  dev.pushErrorScope('validation');
  const module = dev.createShaderModule({ code, label: prog.name });
  for (const m of (await module.getCompilationInfo()).messages)
    messages.push(`${prog.name} ${m.type} ${m.lineNum}:${m.linePos} ${m.message}`);
  const { layout } = layouts(dev, prog);
  const made = [];
  for (const e of prog.entries.filter((x) => x.stage === 'compute'))
    made.push(dev.createComputePipelineAsync({ layout, compute: { module, entryPoint: e.name } }).then(() => `compute ${e.name}`));
  for (const [vs, fs] of prog.pairs) {
    for (const depth of [false, true]) {
      made.push(dev.createRenderPipelineAsync({
        layout,
        vertex: { module, entryPoint: vs, buffers: vertexBuffers(prog, vs) },
        fragment: { module, entryPoint: fs, targets: [{ format: targetOf(prog, fs) }] },
        primitive: { topology: 'triangle-list' },
        depthStencil: depth ? { format: 'depth32float', depthWriteEnabled: true, depthCompare: 'less-equal' } : undefined,
      }).then(() => `${vs}+${fs}${depth ? '+depth' : ''}`));
    }
  }
  const settled = await Promise.allSettled(made);
  const err = await dev.popErrorScope();
  if (err) messages.push(`${prog.name} validation: ${err.message}`);
  let pipelines = 0;
  for (const s of settled) {
    if (s.status === 'fulfilled') pipelines++;
    else messages.push(`${prog.name} pipeline: ${s.reason && s.reason.message}`);
  }
  return { messages, pipelines, expected: made.length };
}

function b64(bytes) {
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s);
}
function unb64(s) { const b = atob(s); const u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; }

async function readBuffer(dev, src, size) {
  const rb = dev.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = dev.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, rb, 0, size);
  dev.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Uint8Array(rb.getMappedRange().slice(0));
  rb.unmap();
  rb.destroy();
  return out;
}

/** Runs the colormap probe on inputs (base64 u32s) with a Cmap block (base64); returns base64 u32s. */
async function colormap(dev, prog, code, blockB64, inputsB64) {
  const block = unb64(blockB64), input = unb64(inputsB64);
  dev.pushErrorScope('validation');
  const module = dev.createShaderModule({ code });
  const { groups, layout } = layouts(dev, prog);
  const pipe = await dev.createComputePipelineAsync({ layout, compute: { module, entryPoint: 'cs_colormap_probe' } });
  const ub = dev.createBuffer({ size: block.length, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(ub, 0, block);
  const ib = dev.createBuffer({ size: input.length, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(ib, 0, input);
  const ob = dev.createBuffer({ size: input.length, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const g1 = dev.createBindGroup({ layout: groups[1], entries: [{ binding: 1, resource: { buffer: ub, offset: 0, size: block.length } }] });
  const g3 = dev.createBindGroup({ layout: groups[3], entries: [{ binding: 0, resource: { buffer: ib } }, { binding: 1, resource: { buffer: ob } }] });
  const empty = dev.createBindGroup({ layout: groups[0], entries: [] });
  const empty2 = dev.createBindGroup({ layout: groups[2], entries: [] });
  const enc = dev.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, empty);
  pass.setBindGroup(1, g1, [0]);
  pass.setBindGroup(2, empty2);
  pass.setBindGroup(3, g3);
  pass.dispatchWorkgroups(input.length / 4 / 64);
  pass.end();
  dev.queue.submit([enc.finish()]);
  const out = await readBuffer(dev, ob, input.length);
  const err = await dev.popErrorScope();
  if (err) throw new Error(err.message);
  for (const b of [ub, ib, ob]) b.destroy();
  return b64(out);
}

// ---- the capture: points, segments and strips, and a mesh with flat faces --------------------

const SIZE = 256;

function uniformBuffer(dev, data, size) {
  const b = dev.createBuffer({ size: Math.max(size, data.byteLength), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(b, 0, data);
  return b;
}
function vbuf(dev, typed) {
  const b = dev.createBuffer({ size: Math.max(16, typed.byteLength), usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  dev.queue.writeBuffer(b, 0, typed);
  return b;
}

/** PrimParams, 192 bytes: mvp, the normal matrix's columns, then the scalar tail at its offsets. */
async function readTexture(dev, tex, rows) {
  const rb = dev.createBuffer({ size: 256 * 4 * rows, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const enc = dev.createCommandEncoder();
  enc.copyTextureToBuffer({ texture: tex }, { buffer: rb, bytesPerRow: 256 * 4 }, [256, rows]);
  dev.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Uint8Array(rb.getMappedRange().slice(0));
  rb.unmap();
  rb.destroy();
  return b64(out);
}

/**
 * The texture path's compute form, colormap_texture: inputs (base64 u32s, whole rows of 256)
 * as a 256-wide image, stored to an rgba8unorm storage texture; returns its texels (base64).
 */
async function colormapTexture(dev, prog, code, blockB64, inputsB64) {
  const block = unb64(blockB64), input = unb64(inputsB64), rows = input.length / 4 / 256;
  dev.pushErrorScope('validation');
  const module = dev.createShaderModule({ code });
  const { groups, layout } = layouts(dev, prog);
  const pipe = await dev.createComputePipelineAsync({ layout, compute: { module, entryPoint: 'cs_colormap_texture' } });
  const tp = uniformBuffer(dev, new Uint32Array([256, rows, 0, 256]), 16);
  const ub = uniformBuffer(dev, block, block.length);
  const ib = dev.createBuffer({ size: input.length, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  dev.queue.writeBuffer(ib, 0, input);
  const tex = dev.createTexture({ size: [256, rows], format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
  const g0 = dev.createBindGroup({ layout: groups[0], entries: [
    { binding: 0, resource: { buffer: tp, offset: 0, size: 16 } }, { binding: 1, resource: { buffer: ib } }, { binding: 2, resource: tex.createView() }] });
  const g1 = dev.createBindGroup({ layout: groups[1], entries: [{ binding: 1, resource: { buffer: ub, offset: 0, size: block.length } }] });
  const enc = dev.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, g0, [0]);
  pass.setBindGroup(1, g1, [0]);
  const wg = prog.entries.find((e) => e.name === 'cs_colormap_texture').workgroup_size;
  pass.dispatchWorkgroups(Math.ceil(256 / wg[0]), Math.ceil(rows / wg[1]));
  pass.end();
  dev.queue.submit([enc.finish()]);
  const out = await readTexture(dev, tex, rows);
  const err = await dev.popErrorScope();
  if (err) throw new Error(err.message);
  for (const b of [tp, ub, ib]) b.destroy();
  tex.destroy();
  return out;
}

window.shaderCheck = { device, compileProgram, colormap, colormapTexture };
