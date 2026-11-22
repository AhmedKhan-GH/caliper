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
function prim(fields) {
  const buf = new ArrayBuffer(192), f = new Float32Array(buf), u = new Uint32Array(buf);
  f.set(fields.mvp, 0);
  f.set(fields.nmat0 || [1, 0, 0, 0], 16); f.set(fields.nmat1 || [0, 1, 0, 0], 20); f.set(fields.nmat2 || [0, 0, 1, 0], 24);
  const at = { pick_id: 28, item_base: 29, point_shape: 30, radius_mode: 31, vertex_count: 33, color_mode: 34, shade_mode: 35,
               flat_rgba: 36, use_instance: 41, line_mode: 42, use_instance_attr: 43, image_mode: 44 };
  for (const [k, i] of Object.entries(at)) u[i] = fields[k] || 0;
  f[37] = fields.vmin || 0; f[38] = fields.vmax || 1; f[39] = fields.size_px || 1; f[40] = fields.radius_scale || 1;
  return buf;
}

/**
 * Draws one fixture per kind with the generated programs on a SIZE x SIZE target and returns
 * the RGBA pixels of each (base64), plus the index clamp's readback check.
 */
async function capture(dev, progs, codes, blockB64, fixture) {
  const block = unb64(blockB64);
  const errors = [];
  dev.pushErrorScope('validation');
  const frame = new ArrayBuffer(48), ff = new Float32Array(frame), fu = new Uint32Array(frame);
  ff.set([SIZE, SIZE, 1, 1, 1, 1, 0, 0]); fu[8] = 65536; fu[9] = 0xff808080;
  const frameBuf = uniformBuffer(dev, frame, 256);
  const cmapBuf = uniformBuffer(dev, block, block.length);
  const classTable = dev.createTexture({ size: [256, 256], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING });
  const imgValue = dev.createTexture({ size: [1, 1], format: 'r32uint', usage: GPUTextureUsage.TEXTURE_BINDING });
  const imgRgba = dev.createTexture({ size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING });
  const sampler = dev.createSampler({ magFilter: 'linear', minFilter: 'linear' });

  function groupsFor(name, primBuf) {
    const prog = progs[name];
    const { groups, layout } = layouts(dev, prog);
    const res = {
      '0,0': { buffer: frameBuf, offset: 0, size: 48 }, '0,1': classTable.createView(),
      '1,0': { buffer: primBuf, offset: 0, size: 192 }, '1,1': { buffer: cmapBuf, offset: 0, size: block.length },
      '2,0': imgRgba.createView(), '2,1': sampler, '2,3': imgValue.createView(),
    };
    const bind = groups.map((l, g) => dev.createBindGroup({ layout: l,
      entries: prog.bindings.filter((b) => b.group === g).map((b) => ({ binding: b.binding, resource: res[`${g},${b.binding}`] })) }));
    const offsets = groups.map((_, g) => prog.bindings.filter((b) => b.group === g && b.dynamic).map(() => 0));
    return { bind, offsets, layout };
  }

  async function draw(name, vs, fs, opts, encodeDraws) {
    const target = dev.createTexture({ size: [SIZE, SIZE], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const depth = dev.createTexture({ size: [SIZE, SIZE], format: 'depth32float', usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const module = dev.createShaderModule({ code: codes[name] });
    const enc = dev.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0.094, g: 0.09, b: 0.082, a: 1 } }],
      depthStencilAttachment: { view: depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    for (const d of encodeDraws) {
      const { bind, offsets, layout } = groupsFor(name, d.prim);
      const pipe = await dev.createRenderPipelineAsync({
        layout,
        vertex: { module, entryPoint: vs, buffers: vertexBuffers(progs[name], vs, d.layout) },
        fragment: { module, entryPoint: fs, targets: [{ format: 'rgba8unorm' }] },
        primitive: { topology: 'triangle-list' },
        depthStencil: { format: 'depth32float', depthWriteEnabled: true, depthCompare: opts.depthTest ? 'less-equal' : 'always' },
      });
      pass.setPipeline(pipe);
      bind.forEach((g, i) => pass.setBindGroup(i, g, offsets[i]));
      d.buffers.forEach(([buf, off], slot) => pass.setVertexBuffer(slot, buf, off));
      if (d.index) { pass.setIndexBuffer(d.index, 'uint32'); pass.drawIndexed(d.count, d.instances || 1); }
      else pass.draw(d.count, d.instances);
    }
    pass.end();
    const rb = dev.createBuffer({ size: SIZE * SIZE * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyTextureToBuffer({ texture: target }, { buffer: rb, bytesPerRow: SIZE * 4 }, [SIZE, SIZE]);
    dev.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const px = new Uint8Array(rb.getMappedRange().slice(0));
    rb.unmap();
    return b64(px);
  }

  const one = (v) => vbuf(dev, new Uint32Array([v, v, v, v]));
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const out = {};

  // Points: a spiral of discs coloured by value through the colormap.
  const P = fixture.points;
  out.points = await draw('scene_points', 'vs_points', 'fs_points', {}, [{
    prim: uniformBuffer(dev, prim({ mvp: identity, point_shape: 1, color_mode: 1, size_px: P.size_px }), 256),
    buffers: [[vbuf(dev, new Float32Array(P.pos)), 0], [vbuf(dev, new Uint32Array(P.values)), 0], [one(0), 0]],
    layout: [{ stride: 12, attrs: [[0, 0]] }, { stride: 4, attrs: [[1, 0]] }, { stride: 0, attrs: [[2, 0]] }],
    count: 6, instances: P.values.length,
  }]);

  // Lines: a star as a strip with per-vertex colours, and spokes as segments coloured by value.
  const L = fixture.lines;
  // Both ends from one buffer at offsets 0 and one vertex (the second binding of the buffer
  // starts there), colour sources likewise, and the break mask constant.
  const LINE = (ps, cs) => [{ stride: ps, attrs: [[0, 0]] }, { stride: ps, attrs: [[1, 0]] },
                            { stride: cs, attrs: [[2, 0]] }, { stride: cs, attrs: [[3, 0]] }, { stride: 0, attrs: [[4, 0]] }];
  const strip = vbuf(dev, new Float32Array(L.strip)), stripRgba = vbuf(dev, new Uint32Array(L.strip_rgba));
  const spokes = vbuf(dev, new Float32Array(L.spokes)), spokeVals = vbuf(dev, new Uint32Array(L.spoke_values));
  out.lines = await draw('scene_lines', 'vs_lines', 'fs_color', {}, [
    { prim: uniformBuffer(dev, prim({ mvp: identity, color_mode: 2, size_px: 5 }), 256),
      buffers: [[strip, 0], [strip, 12], [stripRgba, 0], [stripRgba, 4], [one(1), 0]],
      layout: LINE(12, 4), count: 6, instances: L.strip.length / 3 - 1 },
    { prim: uniformBuffer(dev, prim({ mvp: identity, color_mode: 1, size_px: 2 }), 256),
      buffers: [[spokes, 0], [spokes, 12], [spokeVals, 0], [spokeVals, 4], [one(1), 0]],
      layout: LINE(24, 8), count: 6, instances: L.spokes.length / 6 },
  ]);

  // Mesh: a height field through the index clamp, flat faces, coloured by height.
  const M = fixture.mesh;
  const idxIn = vbuf(dev, new Uint32Array(M.indices)), idxOut = vbuf(dev, new Uint32Array(M.indices.length));
  {
    const prog = progs.index_clamp;
    const module = dev.createShaderModule({ code: codes.index_clamp });
    const { groups, layout } = layouts(dev, prog);
    const pipe = await dev.createComputePipelineAsync({ layout, compute: { module, entryPoint: 'cs_clamp_indices' } });
    const cp = uniformBuffer(dev, new Uint32Array([M.indices.length, M.vertex_count - 1, 0, 0]), 16);
    const g = dev.createBindGroup({ layout: groups[0], entries: [
      { binding: 0, resource: { buffer: cp, offset: 0, size: 16 } }, { binding: 1, resource: { buffer: idxIn } }, { binding: 2, resource: { buffer: idxOut } }] });
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe); pass.setBindGroup(0, g, [0]);
    pass.dispatchWorkgroups(2);   // fewer workgroups than indices / 64: the grid-stride loop covers the rest
    pass.end();
    dev.queue.submit([enc.finish()]);
    const clamped = new Uint32Array((await readBuffer(dev, idxOut, M.indices.length * 4)).buffer);
    out.clamp = { tail: Array.from(clamped.slice(M.drawn)), last: M.vertex_count - 1,
                  same: clamped.slice(0, M.drawn).every((v, i) => v === M.indices[i]) };
  }
  const meshPrim = uniformBuffer(dev, prim({ mvp: M.mvp, nmat0: M.nmat0, nmat1: M.nmat1, nmat2: M.nmat2, color_mode: 1 }), 256);
  out.mesh = await draw('scene_mesh', 'vs_mesh', 'fs_mesh_flat', { depthTest: true }, [{
    prim: meshPrim,
    // Six buffers: positions and colour sources per vertex; the normal, the UV, the identity
    // pose (four columns in one buffer) and the instance value constant, at stride 0.
    buffers: [[vbuf(dev, new Float32Array(M.pos)), 0], [vbuf(dev, new Float32Array([0, 0, 1, 0])), 0], [vbuf(dev, new Uint32Array(M.values)), 0],
              [vbuf(dev, new Float32Array(4)), 0], [vbuf(dev, new Float32Array(identity)), 0], [one(0), 0]],
    layout: [{ stride: 12, step: 'vertex', attrs: [[0, 0]] }, { stride: 0, attrs: [[1, 0]] }, { stride: 4, step: 'vertex', attrs: [[2, 0]] },
             { stride: 0, attrs: [[3, 0]] }, { stride: 0, attrs: [[4, 0], [5, 16], [6, 32], [7, 48]] }, { stride: 0, attrs: [[8, 0]] }],
    index: idxOut, count: M.drawn, instances: 1,
  }]);
  const err = await dev.popErrorScope();
  if (err) errors.push(err.message);
  out.errors = errors;
  return out;
}

// ---- the colormap through the texture path's compute form and the points' vertex stage --------

/** Reads back a 256-wide rgba8unorm texture of `rows` rows as packed u32s (base64). */
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

/**
 * The vertex-stage colormap of scene_points: input i is the value of a point a pixel wide on the
 * centre of pixel (i % 256, i / 256) of a target cleared to 0; returns its pixels (base64).
 */
async function colormapPoints(dev, prog, code, blockB64, inputsB64) {
  const block = unb64(blockB64), input = unb64(inputsB64), n = input.length / 4, rows = n / 256;
  dev.pushErrorScope('validation');
  const module = dev.createShaderModule({ code });
  const { groups, layout } = layouts(dev, prog);
  const pipe = await dev.createRenderPipelineAsync({
    layout,
    vertex: { module, entryPoint: 'vs_points', buffers: vertexBuffers(prog, 'vs_points',
      [{ stride: 12, attrs: [[0, 0]] }, { stride: 4, attrs: [[1, 0]] }, { stride: 0, attrs: [[2, 0]] }]) },
    fragment: { module, entryPoint: 'fs_points', targets: [{ format: 'rgba8unorm' }] },
    primitive: { topology: 'triangle-list' },
  });
  const frame = new ArrayBuffer(48);
  new Float32Array(frame).set([256, rows, 1, 1, 1, 1, 0, 0]);
  const frameBuf = uniformBuffer(dev, frame, 256);
  const primBuf = uniformBuffer(dev, prim({ mvp: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], color_mode: 1, size_px: 1 }), 256);
  const cmapBuf = uniformBuffer(dev, block, block.length);
  const classTable = dev.createTexture({ size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING });
  const imgValue = dev.createTexture({ size: [1, 1], format: 'r32uint', usage: GPUTextureUsage.TEXTURE_BINDING });
  const res = { '0,0': { buffer: frameBuf, offset: 0, size: 48 }, '0,1': classTable.createView(),
                '1,0': { buffer: primBuf, offset: 0, size: 192 }, '1,1': { buffer: cmapBuf, offset: 0, size: block.length },
                '2,3': imgValue.createView() };
  const bind = groups.map((l, g) => dev.createBindGroup({ layout: l,
    entries: prog.bindings.filter((b) => b.group === g).map((b) => ({ binding: b.binding, resource: res[`${g},${b.binding}`] })) }));
  const offsets = groups.map((_, g) => prog.bindings.filter((b) => b.group === g && b.dynamic).map(() => 0));
  const centres = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    centres[3 * i] = 2 * ((i % 256) + 0.5) / 256 - 1;
    centres[3 * i + 1] = 1 - 2 * (Math.floor(i / 256) + 0.5) / rows;
    centres[3 * i + 2] = 0.5;
  }
  const pos = vbuf(dev, centres), vals = vbuf(dev, new Uint32Array(input.buffer, input.byteOffset, n));
  const radius = vbuf(dev, new Float32Array([1, 0, 0, 0]));
  const target = dev.createTexture({ size: [256, rows], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  const enc = dev.createCommandEncoder();
  const pass = enc.beginRenderPass({ colorAttachments: [{ view: target.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }] });
  pass.setPipeline(pipe);
  bind.forEach((g, i) => pass.setBindGroup(i, g, offsets[i]));
  [pos, vals, radius].forEach((b, slot) => pass.setVertexBuffer(slot, b));
  pass.draw(6, n);
  pass.end();
  dev.queue.submit([enc.finish()]);
  const out = await readTexture(dev, target, rows);
  const err = await dev.popErrorScope();
  if (err) throw new Error(err.message);
  for (const b of [frameBuf, primBuf, cmapBuf, pos, vals, radius]) b.destroy();
  for (const t of [classTable, imgValue, target]) t.destroy();
  return out;
}

window.shaderCheck = { device, compileProgram, colormap, colormapTexture, colormapPoints, capture };
