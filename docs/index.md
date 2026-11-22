# Overview

Caliper is a desktop application that draws live graphics from tensors that stay on the GPU. A host program owns the window, the renderer and the frame loop. Applets are shared libraries that the host loads at run time. An applet supplies ImGui user interface and compute, and talks to the host through a frozen C ABI and a set of named service tables.

Tensors reach the screen without a trip through the CPU on the Metal and Vulkan paths. The OpenGL path is a fallback that stages tensor data through the CPU inside the host. A Python package can open a live session from a running training script and show it in the Caliper viewer.

Caliper is built from source. This version of the repository is host 0.6.0, SDK 0.1.0 and ABI epoch 2.

## Starting Points

- [Development basics](development/development-basics.md) explains the host, the applet and the build.
- [Your first applet](applets/first-applet.md) builds the smallest UI applet.
- [Your first ML applet](applets/first-ml-applet.md) adds jobs, metrics, device selection and tensor views.
- [Embedding Caliper](web/embedding-api.md) describes the embeddable host interface.
- [The ABI](applets/applet-contract.md) is the contract between host and applets.

## Backends

| Backend | Where it is used | How tensors reach the screen |
|---|---|---|
| Metal and MPS | macOS default | Metal processes accepted MPS buffers directly, without host staging. |
| Vulkan and CUDA | Windows default | A CUDA kernel writes a tensor update straight into the Vulkan image, with no copy. |
| OpenGL 3.3 | Fallback on every platform | The host stages tensor uploads through the CPU. |

[Rendering](applets/rendering.md) explains how the host chooses a backend, and [`caliper.tensor_bridge.v1`](applets/services/caliper-tensor-bridge-v1.md) lists what each backend accepts.

## Documentation Areas

The documentation is arranged in areas, one per tab: [Introduction](introduction/index.md), [Python](python/index.md), [Applets](applets/index.md), [Builder](builder/index.md), [Agents](agents/index.md), [Web and Embedding](web/index.md), [Reference](reference/index.md), [Development](development/index.md) and [Roadmap](roadmap/index.md).

Plans for work that is not built yet, with the roadmap, are kept in the repository's `design/` folder and are not part of this site.
