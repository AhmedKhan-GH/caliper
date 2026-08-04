# Caliper Framework

Caliper is a native C++ host and SDK for building in-process machine-learning visualization applets. The host owns rendering and the desktop frame loop; applets provide UI and compute through ABI epoch 2 and named service tables.

On supported Metal/MPS and Vulkan/CUDA paths, accepted dense tensors and geometry can render without CPU staging. OpenGL is a CPU-staged fallback, and chart-style views may use small CPU arrays.

!!! warning "Development status"
    Current host: **0.6.0** · SDK: **0.1.0** · ABI epoch: **2**. Build Caliper from source. The repository does not currently publish packaged binaries, separate SDK releases, runtime packs, or an applet registry.

## Choose a path

- **[Development basics](tutorials/development-basics.md)** — understand the host, applet, and build model.
- **[Your first applet](tutorials/first-applet.md)** — build the smallest ABI-epoch-2 UI applet.
- **[Your first ML applet](tutorials/first-ml-applet.md)** — add jobs, metrics, device selection, and tensor views.
- **[Embedding Caliper](reference/embedding.md)** — inspect the embeddable host interface.
- **[ABI and services](reference/abi.md)** — read the contract and service reference.

## Backend behavior

| Backend | Current behavior |
|---|---|
| **Metal / MPS** | macOS default. Accepted MPS buffers are processed by Metal without host staging. |
| **Vulkan / CUDA** | Windows default. Device-local external-memory interop; some paths make one device-to-device copy inside VRAM. |
| **OpenGL 3.3** | Compatibility fallback. Tensor uploads are CPU-staged inside the host. |

See [Rendering](explanation/rendering.md) and [`caliper.tensor_bridge.v1`](reference/services/tensor-bridge-v1.md) for the exact backend and acceptance rules.

## Documentation map

- **Tutorials** — start-to-finish development paths.
- **How-to guides** — porting, debugging, and ML applet recipes.
- **Reference** — ABI, manifest, service tables, adapters, embedding, and C++ sugar.
- **Explanation** — rendering, compatibility, trust, and the planned architecture.
- **[Decisions](decisions/index.md)** — the project decision log.

## Source of truth

Reference pages embed current headers and examples with checked snippets, and `mkdocs build --strict` fails on missing sources, links, or anchors. `PLATFORM.md` is a planning and design specification; where it describes future repository topology or distribution, those parts are not current shipped behavior.
