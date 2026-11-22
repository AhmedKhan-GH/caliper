# Scenes

This page plans what remains of scenes: the scene model's additions of additive blending, instanced meshes and a depth offset for lines, the encoded image drawn by the view, the ground grid and the axes mark, the starting camera applied by the view, the text log and the bar chart, stamps on the timelines, scenes through the shared-memory session with their data views, linked selection from Caliper LIB, and a scene service that gives applets the scene model in place of the geometry service.

The rest of scenes is built, and the docs describe it. Scene Calls (`docs/python/scene-calls.md`) describes `caliper.scene` with its primitives, cameras, entity transforms, class tables, clears, static items, limits and call cost, and Scene Windows in Viewer (`docs/python/viewer.md`) describes a scene in the Caliper app. Scene Rendering (`docs/applets/scene-rendering.md`) describes the scene core, the scene view with its up-axis switch, scale bar, labels and pick pass, and the points view. Wire Protocol 3.2 (`docs/reference/wire-protocol-3-2.md`) specifies the scene messages that carry a scene over the network.

## Scene Model

The scene model is built as Scene Calls (`docs/python/scene-calls.md`) describes it, apart from the additions below.

### Primitives

A mesh takes the optional `instance_poses` f32 (N, 4, 4), with `instance_colors` (N, 4) or `instance_values` (N,), and its limit is N ≤ 1,048,576. Points and lines take the option `blend`.

`blend` is `"normal"`, the default, or `"additive"`. Additive points and lines are drawn after the opaque batches with the depth test on and depth writes off, adding their colour, so that a dense cloud or a bundle of trajectories brightens where it overlaps and the order of its items does not matter. A mesh with `instance_poses` is drawn once per pose, each pose applied before the entity's world pose, and its normals are transformed by the cofactor of each pose's upper 3 × 3, which lights any invertible pose correctly. `instance_colors` or `instance_values` colour each instance and take the place of the mesh's own colours. Both are additions to the shared contracts, made with the scene service that replaces the geometry service, because each is a general feature of a scene model that the applets' draws needed. The three codecs of protocol 3.2 are to carry them as a `blend` key of the points and lines parts and as instance arrays of the mesh part, which they do not carry yet.

The encoded image form (JPEG or PNG bytes, one video frame each, at most 16 MiB) is accepted by Caliper LIB and carried by the wire, but the scene view does not draw it yet and answers with the note that encoded images are not drawn yet. It draws the form once the view decodes it.

Lines, arrows and box edges are drawn with a small constant depth offset toward the camera, so that an edge lying on a mesh, such as a wireframe, wins over the surface.

### Up-Axis Switch

The ground grid and the axes mark in the corner follow the up axis that the switch sets. The scene view has neither yet.

## Text Log

The text log is a live view of the new kind `text`, proposed id 7. It keeps lines of text. Each line has a text of at most 4,096 bytes of UTF-8 (a longer one is cut and marked), a level, `debug`, `info`, `warning` or `error`, and a stamp. Like a series, it is append-only: a viewer receives the lines it lacks, never a replacement. Lines are host data and cost no device work, so the producer retains the last 4,096 lines of each text log whether or not anything is attached, and a late viewer receives them first. The view scrolls with the newest line, filters by level and searches by substring. When a recording is open, a click on a line moves the cursor to its stamp. In the shared-memory session a text log is one segment with one slot that is never claimed, as for a series: a byte ring of the last 4,096 lines and an index of their offsets, levels and stamps.

## Bar Chart

The bar chart is a live view of the new kind `bars`, proposed id 8: one vector of at most 4,096 values with a label per bar, replaced whole by each snapshot, as a histogram is. The values are a tensor and stay on its device; the labels are host strings of at most 63 bytes. Bars rise from zero, so negative values point down, and a hover shows the label and the value. With named dimensions, the values offer data views as a histogram's do, with one kept dimension, the bar dimension. A pick reports `bar`, `row_id` and `value`, and a select a `[start, stop]` range of bars; an item space on the bar dimension links them to other views. In the shared-memory session it has three slots, each a (B,) f32 payload and a host region for the labels.

## Caliper LIB

`caliper.scene` and its calls are built, apart from the arguments that the sections above and below plan: `data_view`, `items`, `blend` and the instance arrays of a mesh.

### Starting Camera

`view_camera` sets the camera on which every 3-D view of the scene opens, so that a demo or an applet opens on a good angle without a person arranging it.

Caliper LIB checks it, the wire carries it as a key of the scene's entry in VIEWS and the scene core holds the starting state with its reset, but the scene view does not apply it yet, so a view opens on the fit and a reset returns to the fit.

`view_camera` is a starting value only, like `up`. It is where a view starts and where its reset ends. Once a person moves the camera, by orbiting, panning, zooming, fitting or looking through a camera, the view remembers its own camera with its other settings, and that camera wins when the view is shown again. Resetting the view returns to `view_camera`, or to the fit when there is none. A second `caliper.scene` call with a new `view_camera` replaces the starting camera: a view whose camera the person has not moved since it opened or was reset moves to the new one at once, and a moved camera stays where the person left it until its next reset, which goes to the new one. `view_camera=None` leaves the starting camera as it is, and `view_camera={}` returns it to the fit.

The angles are always read against the scene's `up`. In a view that the person has switched to another mode, the starting camera follows the rule of the up-axis switch that Scene Rendering (`docs/applets/scene-rendering.md`) describes: the eye position and the target stay, and only the picture rolls. The camera-entity form opens the view through that entity's pinhole, with its pose at the cursor. The up-axis switch has no effect there, as for any pinhole; leaving the pinhole goes to the orbit of the fit, and a reset looks through the camera again.

A spec panel sets the same starting camera through its option `view_camera`, in the same two forms.

A test of the view opens a scene with a `view_camera` and checks that a point logged at the target is drawn at the view's centre and picked there; that a camera the person moved survives a second call with another `view_camera` and a close and reopen; that a reset returns to `view_camera`, and to the fit without one; that a switched view keeps the eye position and the target within 10<sup>-6</sup>; and that the camera-entity form opens through the pinhole and falls back to the fit, with its note, when the entity holds no camera. A capture row draws one fixture from a `view_camera` against a reference, natively and in headless Chromium.

Preset views, an orthographic projection, following an entity, saved viewpoints and linked cameras are later viewport work, and a fly mode is left out.

### `text` and `bars`

```python
caliper.text(message, name="log", level="info")
caliper.bars(values, name="bars", labels=None, colors=None, horizontal=False, rate=None, *,
             dims=None, data_view=None)
```

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `message` | `str` | | One entry; a message with line breaks is one entry of several lines |
| `level` | `str` | `"info"` | `"debug"`, `"info"`, `"warning"` or `"error"` |
| `values` | tensor (B,) | | 1 ≤ B ≤ 4,096; any dtype, cast to f32 on its device |
| `labels` | list of `str` or `None` | `None` | One per bar; `None` labels the bars by index |
| `colors` | tensor (B, 3 or 4) or one colour | `None` | Bar colours |
| `horizontal` | `bool` | `False` | Draw the bars from left to right |
| `name`, `rate`, `dims`, `data_view` | | | As for `caliper.histogram` |

`text` raises `ValueError` for an unknown level, and `bars` for a shape or label count that does not fit. None of the names `scene`, `text` and `bars` is taken in the `caliper` package today.

### Stamps on the Timelines

Every call of a scene, and every `text` and `bars` call, is stamped at the call with the session's position on each timeline: `step`, which the script advances with `caliper.step`; `time`, the wall clock; and the timelines the script defines. The calls that define and advance timelines belong to the recordings plan, which git history keeps as `design/recordings.md`. A call with `static=True` has no stamp. A snapshot carries the stamp of the call that made it, so a batch dropped by its rate limit leaves no stamp. Each snapshot keeps its monotonic `stamp_ns` for latency display, as today.

## Scene Data on the Device

### Copies per Route

Caliper LIB offers each call to every open route, as it does for the other views. The device-side work of a snapshot is built once and shared by the routes that take it. A scene snapshot is one batch: the arrays of one primitive of one entity.

The network route is built, and Scene Calls (`docs/python/scene-calls.md`), Call Cost, gives what it copies.

| Route | Metal with MPS | Vulkan with CUDA |
|---|---|---|
| An applet in the host, drawing its own tensors through the scene service | Zero-copy: the imported allocation is drawn in place | Zero-copy for tensors in the exportable pool, once the Vulkan backend of the graphics layer lands |
| Caliper LIB's in-process window or canvas | One copy on torch's stream into a Caliper-made shared `MTLBuffer` slot; the graphics layer imports the slot and draws it with no further copy | Not built: Caliper LIB runs on macOS only. Planned as on Metal, the slot being exportable device memory |
| The shared-memory session to the Caliper app | One copy on torch's stream into a slot of the session's shared memory; the app imports the slot and draws it with no further copy | Not built. Planned with the Windows session of [Windows parity](windows-parity.md): the directory in named shared memory and the slots in exportable CUDA memory, imported in the app through Vulkan external memory |

A scene from Caliper LIB is therefore never zero-copy in the strict sense of Zero-Copy Tensors (`docs/python/zero-copy-tensors.md`). Each snapshot is one copy inside the GPU, on Apple Silicon inside the one unified memory, and that copy is what lets the training loop rewrite its tensors while a viewer draws. From the slot on, nothing is copied, nothing reaches host memory, and nothing waits: completion is polled through a `torch.mps.Event`, as for points. Zero-copy holds from the slot to the pixels, through the graphics layer's import. A CPU tensor is copied once into the slot, and on macOS a tensor on any other device is moved to the CPU first, one host copy, as for every view.

Four kinds of data travel through host-visible memory because the viewer needs them on its CPU: entity transforms, cameras, the class table and labels. They are small. On Metal the slot is the same memory the CPU reads; on Vulkan with CUDA a device-given transform is copied asynchronously into a pinned host slot, and its completion is polled.

An image becomes a texture by one pass on the viewer's GPU, as heatmaps do today.

### Shared-Memory Session Growth

Protocol v2 gives each view one segment of three slots, each slot one payload, an optional f32 aux region and a stats region. A scene needs several arrays per batch, many batches, and per-batch publishing, because entities change at different rates. Its proposed layout keeps the v2 slot protocol unchanged and applies it per batch.

| Piece | Name | Content |
|---|---|---|
| Control table row | | Kind 6, the scene's name, and in `segment` its directory segment; `slot_count` 0, `dtype` and `shape` unused |
| Directory segment | `/<session>-v<id>` | A header page with a `generation` seqlock and the scene's starting camera, then the entity table, the batch table and the class table, about 2.8 MiB |
| Arena segments | `/<session>-v<id>a<k>`, k from 0 to 15 | Page-aligned slots of batches, added as the scene grows; a batch's slots are reallocated when its size changes and freed once both viewer claims on them are released |

| Entity row (1,024, 512 bytes each) | Meaning |
|---|---|
| `path`, `parent` | The entity path, and the row of its parent |
| `flags` | Holds a transform, holds a camera, which parts are static, cleared |
| `transform` | A host-given entity transform, 16 f32, or the row of the batch that carries a device-given one |
| `camera` | `fx`, `fy`, `cx`, `cy`, width, height, convention, plane distance |
| `stamp`, `clear_stamp` | The stamps of its last change and its last clear |

| Batch row (2,048, 1 KiB each) | Meaning |
|---|---|
| `entity`, `primitive`, `form` | The entity's row; the primitive, 1 to 6 in the order of the primitives table, or 7 for a device-given entity transform; the image form, the line mode (segments, strips or edges), `fill` |
| `arrays` | Up to 8 descriptors: role, dtype, shape of up to 3 dimensions, and byte offset within the slot. A lines batch of edges adds the role `indices`, (E, 2) i32 |
| `slot_arena`, `slot_offset` | For each of the three slots, its arena segment and its offset |
| `slot_bytes`, `host_bytes` | The device-written part of a slot and its host-written part (labels) |
| Sync words | `seq`, `published_slot`, `viewer_reading_slot`, `viewer_prev_slot`, `dropped_client`, `viewer_seq`, `skipped_viewer`, with the meanings and ordering of a v2 view segment |
| Slot metadata | For each slot: `seq`, `stamp_ns`, the timeline stamp, `colormap`, `vmin`, `vmax`, flags, item count, and 256 bytes of stats written on the device: value range and per-axis bounds |
| `viewer_skip` | Written by the viewer: 1 while the person hides the entity, so that the producer takes no snapshot of the batch |

The protocol keeps version 2, under its own append-only rules: the kinds 6 to 8, a `features` word appended to the control segment, and `"features": ["scenes", "text", "bars"]` in the session file. A v2 viewer from before scenes does not know kind 6. It must report an error for that view and draw the others, and a test against a v2 viewer from before scenes has to confirm that before the additions ship. One rule has to relax for this: today a reader refuses a struct smaller than its own, and a reader of the additions must instead read a smaller v2 struct as one without them.

### Scene Data Views

A subscription to a scene takes only the default data view today, which samples points and line segments to 100,000 by stride, and `caliper.scene` refuses `data_view`.

A scene offers a data view, so that the producer still reduces on its device and sends only the result. A subscription of the network names one, as for any view; the window and the shared-memory session use the scene's default, which by default sends every batch whole.

| Key | Form | Meaning |
|---|---|---|
| `entities` | `{"include": [...], "exclude": [...]}`, up to 16 filters each | A filter is a path (the entity alone), `<path>/**` (the entity and its subtree) or `**`. A batch is sent when an include filter matches its entity and no exclude filter does. Default: include `**` |
| `budget` | The keys below | Taken key by key, as the budget of the other kinds |

| Budget key | Applies to | Default over the network | Limit | How the producer meets it |
|---|---|---|---|---|
| `max_points` | Points, and the point cloud of a depth image | 100,000 | 1,000,000 | Stride sampling, with every array of the batch sampled alike, by the rule of the points data view |
| `max_items` | Boxes, arrows, line segments, index-pair edges | 100,000 | 1,048,576 | Stride sampling; a strip keeps every k-th vertex and its two ends, and edges keep their per-edge colours and values |
| `max_vertices` | Meshes, and the positions of index-pair edges | 1,000,000 | 4,194,304 | Not reduced: a mesh or an edge batch over the budget is sent as its bounding box, with a note in its hover |
| `max_extent` | Images | 1,024 | 8,192 | Colour by mean pooling; depth and segmentation by stride, so that every value sent is a real depth or class id |

Transforms, cameras, class tables and clears always travel whole. Encoded frames are not reduced. An excluded or hidden entity costs the producer nothing: no snapshot, no copy. Points sampled under a budget report picks in source indices through the payload axes, as points do today.

## Picking, Hover and Selection

The pick pass, hover, click, the lasso and rectangle selection by CPU projection, the dimming and the pick, hover and select events are built, and Scene Rendering (`docs/applets/scene-rendering.md`) describes the pick pass. On a route whose slots are not CPU-visible, the planned Windows session, a rectangle selects the visible items from a readback of the rectangle, and the lasso is unavailable. Clicking an entity in the view's tree selects the whole entity.

Caliper LIB still refuses `items=` on points and boxes, so a script cannot log a linked batch yet. A batch logged with `items` maps its row ids to the items of that space. A select on it adds `space` and `items`, as in 3.1, and a SELECTION of that space highlights the matching items in every batch, and every other view, that maps the space.

## Interaction with the Script

The script can set the selection of an item space, and every scene batch that maps the space highlights it, which needs `items=`. The shared-memory session has no channel from the viewer back to the script apart from the heartbeat, so the Caliper app attached through it shows hovers and selections without the script receiving them. A script that needs them is attached through `caliper.serve()`. Interventions through a scene, such as dragging an entity to set a parameter, need typed arguments in the key that 3.1 reserves for them and are planned later.

## Scene Service for Applets

An applet draws in 3-D through the scene service, `caliper.scene.v1`, which offers the scene model of this page to the applet ABI: scenes with a space, an up axis and units; entity paths; entity transforms; points, lines as segments, strips and index-pair edges, arrows, boxes, meshes and images in their colour, depth and segmentation forms; pinhole cameras; clears and static items; the class table; the starting camera; and the scale bar. It replaces the geometry service by decision, and [Renderer Core](renderer-core.md#migration) orders the steps that build it, move the applets onto it and delete the geometry service. Nothing of it is built.

The model is the one Caliper LIB logs, with four differences that follow from where an applet runs. Arrays name memory the applet already has, an imported allocation or a `CaliperTensor`, instead of a tensor in a Python process. An applet's updates carry no stamps of a script's timelines, because an applet has none. There are no rate limits or data views, because the applet draws in its own process at its own cadence and decides what to set. Encoded JPEG and PNG frames are not offered, because an applet holds pixels, not files.

### Service and Version

The service id is `caliper.scene.v1`, declared in `sdk/include/caliper/services/scene_v1.h`. It is one table, as every graphics service is once the migration's step 13 lands ([One Version of Each Graphics Service](renderer-core.md#one-version-of-each-graphics-service)): within an ABI epoch it grows only by appending members, which its `struct_size` guards, and each record below grows only by appending fields, which the record's own `struct_size` guards. An incompatible change is an epoch change. Every member is called on the frame thread only, the thread that runs the applet's `frame`, as for the tensor bridge.

### Array Names

Every array of a batch is one `CaliperSceneArray`. It names where the bytes are, as the geometry service's sources did with an allocation and a byte offset, and how to read them:

| Field | Meaning |
|---|---|
| `source` | `CALIPER_SCENE_SRC_ALLOC`: an imported allocation, read in place. `CALIPER_SCENE_SRC_TENSOR`: a `CaliperTensor`. `CALIPER_SCENE_SRC_NONE`: the optional array is absent |
| `alloc`, `offset` | With `SRC_ALLOC`: a `CaliperAllocId` from the tensor bridge's `import_allocation`, the same id space the bridge's textures use, and the byte offset of row 0 inside it, a multiple of 4 |
| `count` | The number of rows: points, vertices, strips, edges, triangles, boxes, instances or pixels |
| `dtype`, `components` | The `CaliperDType` of one component, and the components per row: 1, 2, 3, 4 or 16 |
| `stride` | The bytes from one row to the next, at most 2,048; 0 means packed. A stride lets an applet bind one column of a wider array without a copy |
| `tensor` | With `SRC_TENSOR`: a CPU tensor, copied at the call so that the applet may free it on return, or a device tensor of the host's backend (`CALIPER_DEV_METAL` today), imported in place as the bridge imports it |

An `SRC_ALLOC` array is imported through the graphics layer's `import_buffer` once per allocation and bound as a vertex, instance or index source at its offset, so the scene view draws the applet's memory in place and the zero-copy route of the geometry service is kept: the info line reports the route that ran, and the copy counters of the migration's step 0 read 0 on it. Index arrays may sit at any 4-byte offset: the index clamp binds the 256-byte-aligned offset below and starts at the remainder, so nothing is copied. Each role takes the dtypes and component counts that [Primitives](#primitives) gives after Caliper LIB's conversion: f32 positions, radii, values and depths, u8 colours and colour images, i32 indices and class ids, and u16 class ids and depths. Caliper LIB casts other dtypes on the device; the service converts nothing, so an applet gives these.

### Service Table

```c
#define CALIPER_SCENE_V1 "caliper.scene.v1"

#define CALIPER_SCENE_CAP_DRAW     (1u << 0)  /* the host draws scenes: it has a backend of the graphics layer */
#define CALIPER_SCENE_CAP_IN_PLACE (1u << 1)  /* SRC_ALLOC arrays are read in place; without it, copied at each call */

typedef uint64_t CaliperSceneId;      /* 0 = none */
typedef uint64_t CaliperSceneToken;   /* one accepted update of one part; 0 = refused */

enum { CALIPER_SCENE_SRC_NONE = 0, CALIPER_SCENE_SRC_ALLOC = 1, CALIPER_SCENE_SRC_TENSOR = 2 };
typedef struct CaliperSceneArray {    /* 48 bytes */
    uint32_t source, dtype;
    CaliperAllocId alloc;  uint64_t offset;
    uint64_t count;
    uint32_t components, stride;
    const CaliperTensor* tensor;
} CaliperSceneArray;

enum { CALIPER_SCENE_COLOR_DEFAULT = 0, CALIPER_SCENE_COLOR_FLAT, CALIPER_SCENE_COLOR_RGBA,
       CALIPER_SCENE_COLOR_VALUES, CALIPER_SCENE_COLOR_CLASS };
enum { CALIPER_SCENE_PER_ITEM = 0, CALIPER_SCENE_PER_VERTEX, CALIPER_SCENE_PER_STRIP, CALIPER_SCENE_PER_EDGE };
typedef struct CaliperSceneColor {    /* 72 bytes */
    uint32_t kind;                    /* CALIPER_SCENE_COLOR_* */
    uint32_t per;                     /* lines: what the array belongs to; else PER_ITEM */
    uint32_t flat_rgba;               /* FLAT: r | g<<8 | b<<16 | a<<24 */
    int32_t  colormap;                /* VALUES: CaliperColormap */
    float    vmin, vmax;              /* VALUES: the range; vmin == vmax maps every value to level 0 */
    CaliperSceneArray array;          /* RGBA: u8 x4; VALUES: f32 x1; CLASS: i32 or u16 x1 */
} CaliperSceneColor;

typedef struct CaliperScenePixels {   /* an image held in an array */
    CaliperSceneArray data;           /* count = width * height; components = channels */
    uint32_t width, height;           /* at most 8,192 each */
    uint32_t row_pitch, reserved;     /* bytes per image row; 0 = packed */
} CaliperScenePixels;

/* Flags of the records below. */
#define CALIPER_SCENE_STATIC    (1u << 0)  /* setup: a clear of the scene's live data leaves it */
#define CALIPER_SCENE_ADDITIVE  (1u << 1)  /* points and lines: additive blending */
#define CALIPER_SCENE_FILL      (1u << 2)  /* boxes: translucent faces */
#define CALIPER_SCENE_CLOUD     (1u << 3)  /* depth images: also the back-projected cloud */
#define CALIPER_SCENE_MATRIX    (1u << 4)  /* transforms: use matrix, not the parts */
#define CALIPER_SCENE_RUB       (1u << 5)  /* cameras: x right, y up, looking along -z */
#define CALIPER_SCENE_RECURSIVE (1u << 6)  /* clear: the whole subtree */
#define CALIPER_SCENE_VIEW_NO_HEADER (1u << 0)  /* view: the header's controls in the context menu */

typedef struct CaliperScenePoints {
    uint32_t struct_size, flags;              /* STATIC, ADDITIVE */
    CaliperSceneArray positions;              /* f32 (N, D) */
    CaliperSceneColor color;                  /* per point */
    CaliperSceneArray radii;                  /* f32 (N, 1) in scene units, or SRC_NONE */
    float radius;                             /* one radius in scene units; 0: a pixel size */
    float size_px;                            /* disc diameter in logical pixels; 0: by count, as the points view */
    const char* const* labels;                /* label_count strings, or NULL */
    uint32_t label_count, reserved;           /* 0, 1 (one for all) or N */
} CaliperScenePoints;

enum { CALIPER_SCENE_SEGMENTS = 0, CALIPER_SCENE_STRIPS, CALIPER_SCENE_EDGES };
typedef struct CaliperSceneLines {
    uint32_t struct_size, flags;              /* STATIC, ADDITIVE */
    uint32_t mode;  float width_px;           /* width 0: 2 pixels */
    CaliperSceneArray positions;              /* f32 (V, D) */
    CaliperSceneArray strip_lengths;          /* STRIPS: i32 (S, 1) */
    CaliperSceneArray indices;                /* EDGES: i32 (E, 2) */
    CaliperSceneColor color;                  /* per vertex, strip or edge */
} CaliperSceneLines;

typedef struct CaliperSceneArrows {
    uint32_t struct_size, flags;
    float width_px;  uint32_t reserved;
    CaliperSceneArray origins, vectors;       /* f32 (N, D) each */
    CaliperSceneColor color;                  /* per arrow */
} CaliperSceneArrows;

typedef struct CaliperSceneBoxes {
    uint32_t struct_size, flags;              /* STATIC, FILL */
    float width_px;  uint32_t label_count;
    CaliperSceneArray centers, half_sizes;    /* f32 (N, D) each */
    CaliperSceneArray rotations;              /* f32 (N, 4) quaternions x y z w, 3-D only, or SRC_NONE */
    CaliperSceneColor color;                  /* per box, class colours included */
    const char* const* labels;
} CaliperSceneBoxes;

typedef struct CaliperSceneMesh {
    uint32_t struct_size, flags;
    CaliperSceneArray positions;              /* f32 (V, D) */
    CaliperSceneArray indices;                /* i32 (T, 3), or SRC_NONE: rows 3k to 3k + 2 */
    CaliperSceneArray normals;                /* f32 (V, 3): the headlight; SRC_NONE: flat faces */
    CaliperSceneColor color;                  /* per vertex */
    CaliperSceneArray uvs;                    /* f32 (V, 2), with texture */
    CaliperScenePixels texture;               /* u8 (H, W, 3 or 4), or data SRC_NONE */
    CaliperSceneArray instance_poses;         /* f32 (N, 16) column-major, or SRC_NONE */
    CaliperSceneColor instance_color;         /* per instance; DEFAULT: the mesh's own colour */
} CaliperSceneMesh;

enum { CALIPER_SCENE_IMAGE_COLOR = 0, CALIPER_SCENE_IMAGE_DEPTH, CALIPER_SCENE_IMAGE_SEGMENTATION };
typedef struct CaliperSceneImage {
    uint32_t struct_size, flags;              /* STATIC, CLOUD */
    uint32_t form;  float opacity;            /* opacity < 0: the form's default */
    CaliperScenePixels pixels;                /* COLOR u8 x1, 3 or 4; DEPTH f32 or u16; SEGMENTATION i32 or u16 */
    float depth_scale;  int32_t colormap;     /* DEPTH; depth_scale 0: 1 for f32, 0.001 for u16 */
    float vmin, vmax;                         /* DEPTH */
} CaliperSceneImage;

typedef struct CaliperSceneTransform {
    uint32_t struct_size, flags;              /* STATIC, MATRIX */
    float translation[3];
    float rotation[4];                        /* x y z w; in 2-D, rotation[0] is the angle in radians */
    float scale[3];
    float matrix[16];                         /* column-major, with MATRIX */
} CaliperSceneTransform;

typedef struct CaliperScenePinhole {
    uint32_t struct_size, flags;              /* STATIC, RUB */
    uint32_t width, height;                   /* the image size in pixels */
    float fx, fy, cx, cy;                     /* cx or cy NaN: the image centre */
    float plane_distance;                     /* 0: a twentieth of the scene's bounding radius */
} CaliperScenePinhole;

typedef struct CaliperSceneClass { uint32_t id, rgba; const char* label; } CaliperSceneClass;

#define CALIPER_SCENE_VIEW_TARGET    (1u << 0)
#define CALIPER_SCENE_VIEW_DISTANCE  (1u << 1)
#define CALIPER_SCENE_VIEW_AZIMUTH   (1u << 2)
#define CALIPER_SCENE_VIEW_ELEVATION (1u << 3)
typedef struct CaliperSceneViewCamera {
    uint32_t struct_size, given;              /* CALIPER_SCENE_VIEW_*; what is not given comes from the fit */
    float target[3], distance, azimuth_deg, elevation_deg;
    const char* entity;                       /* look through this entity's camera instead, or NULL */
} CaliperSceneViewCamera;

enum { CALIPER_SCENE_UP_DEFAULT = 0, CALIPER_SCENE_UP_POS_X, CALIPER_SCENE_UP_NEG_X, CALIPER_SCENE_UP_POS_Y,
       CALIPER_SCENE_UP_NEG_Y, CALIPER_SCENE_UP_POS_Z, CALIPER_SCENE_UP_NEG_Z };
typedef struct CaliperSceneDesc {
    uint32_t struct_size;
    uint32_t space;                           /* 0: 3-D, 1: 2-D */
    uint32_t up;                              /* CALIPER_SCENE_UP_*; DEFAULT is +y in 3-D and -y in 2-D */
    uint32_t reserved;
    const char* name;                         /* 1 to 63 bytes, unique within the applet */
    const char* units;                        /* at most 15 bytes; NULL: "m" */
} CaliperSceneDesc;

enum { CALIPER_SCENE_EVENT_HOVER = 1, CALIPER_SCENE_EVENT_HOVER_END, CALIPER_SCENE_EVENT_PICK,
       CALIPER_SCENE_EVENT_SELECT };
enum { CALIPER_SCENE_POINTS = 1, CALIPER_SCENE_LINES, CALIPER_SCENE_ARROWS, CALIPER_SCENE_BOXES,
       CALIPER_SCENE_MESH, CALIPER_SCENE_IMAGE };
typedef struct CaliperSceneEvent {
    uint32_t struct_size, kind;               /* CALIPER_SCENE_EVENT_* */
    const char* view;                         /* the view's key */
    const char* entity;                       /* the entity path; NULL in an empty SELECT, which clears */
    uint32_t primitive, has_position;         /* CALIPER_SCENE_POINTS ... */
    uint64_t item;                            /* HOVER, PICK: the row of the batch's arrays */
    float position[3];                        /* the hit's world position, when has_position */
    uint32_t pixel[2];                        /* images: the pixel */
    const uint64_t* items;                    /* SELECT: item_count rows of this entity and primitive */
    uint64_t item_count;
} CaliperSceneEvent;                          /* strings and items are valid until the next poll_event */

typedef struct CaliperSceneV1 {
    uint32_t struct_size;
    uint32_t (*caps)(void);
    CaliperSceneId (*create)(const CaliperSceneDesc* desc);
    void (*remove)(CaliperSceneId scene);
    CaliperSceneToken (*points)(CaliperSceneId scene, const char* path, const CaliperScenePoints* b);
    CaliperSceneToken (*lines)(CaliperSceneId scene, const char* path, const CaliperSceneLines* b);
    CaliperSceneToken (*arrows)(CaliperSceneId scene, const char* path, const CaliperSceneArrows* b);
    CaliperSceneToken (*boxes)(CaliperSceneId scene, const char* path, const CaliperSceneBoxes* b);
    CaliperSceneToken (*mesh)(CaliperSceneId scene, const char* path, const CaliperSceneMesh* b);
    CaliperSceneToken (*image)(CaliperSceneId scene, const char* path, const CaliperSceneImage* b);
    bool (*transform)(CaliperSceneId scene, const char* path, const CaliperSceneTransform* t);
    bool (*camera)(CaliperSceneId scene, const char* path, const CaliperScenePinhole* c);
    bool (*classes)(CaliperSceneId scene, const CaliperSceneClass* table, uint32_t count, uint32_t flags);
    bool (*clear)(CaliperSceneId scene, const char* path, uint32_t flags);   /* RECURSIVE, STATIC */
    bool (*view_camera)(CaliperSceneId scene, const CaliperSceneViewCamera* c);   /* NULL: the fit */
    bool (*retired)(CaliperSceneId scene, CaliperSceneToken token);
    bool (*view)(CaliperSceneId scene, const char* key, float width, float height, uint32_t flags);
    bool (*poll_event)(CaliperSceneId scene, CaliperSceneEvent* out);
    const char* (*last_error)(void);
} CaliperSceneV1;
```

Each record means what the Caliper LIB call of the same name means in Scene Calls (`docs/python/scene-calls.md`), with the defaults given there, and the rules of [Primitives](#primitives), of Scene Calls (`docs/python/scene-calls.md`) for entity transforms, cameras and their pixel space, the class table and clearing and static items, and of [Starting Camera](#starting-camera) apply unchanged. A colour's range is given by the applet, because the service computes nothing on the device. `remove` releases the scene and its imports once the last frame that read them completed.

### Frames, Atomicity and Memory Stability

- **When an applet may update.** Only on the frame thread, during its `frame` or its `draw_ui`. A worker thread hands its slots to the frame thread, as the geometry service's applets do today.
- **One frame, together.** Every call made between two frames takes effect together in the next picture: the host compiles each changed scene once, after the applet's frame returns and before the ImGui pass. A later call at the same path and primitive in the same frame replaces an earlier one.
- **Refusal per part.** A refused call returns 0 or false, changes nothing of that part, leaves the previous batch drawn, and states its reason in `last_error` and in one `caliper.log.v1` line. The other calls of the frame still apply. The geometry service refused a whole frame, because its frame was a clear and a list of draws, and a refusal after the clear would have left a blank picture; a scene keeps every part it holds, so a refused part cannot blank the picture, and refusing the frame's other parts would protect nothing.
- **The spatial half of memory stability.** An `SRC_ALLOC` array is read in place whenever the view redraws: when data, the camera or the size changes, and for each pick. Its bytes must therefore not be rewritten while its batch is current, nor after the batch is replaced or cleared until `retired` returns true for the token of the update that set it, which happens once the last frame that read it completed. `retired` never blocks. An applet's triple-buffered slots fit this: it writes a slot, sets the batch from it, keeps the token with the slot, and writes the slot again only once that token has retired.
- **The temporal half.** The producer must have finished writing an array before the call, by draining its device, as for the geometry service: the service orders nothing after a producer that is still writing.
- **Without in-place import.** Where `CALIPER_SCENE_CAP_IN_PLACE` is unset, on a backend that cannot import, such as Vulkan without a paired CUDA device, an `SRC_ALLOC` array is copied at the call, its info line says so, and its token retires at once.
- **A `SRC_TENSOR` array** on the CPU is copied at the call. A device tensor follows the rules of `SRC_ALLOC`.

### View in the Applet Window

`view(scene, key, width, height, flags)` draws the scene view as one ImGui item in the applet's current window, at the cursor and at the given logical size, or in the rest of the content region when a size is 0. It is the same scene view that the Caliper app shows for a script's scene: the picture, the header with the fit button, the Y up and Z up switch and the display menu, orbit, pan and zoom by mouse and keys, the hover tooltip, labels, the axes mark and the scale bar. The host remembers each view's camera, up mode and display options under the applet's id and the view's key, as for every view, and the key also lets one scene be shown in two views with two cameras; `NULL` takes the scene's name. It returns true when the picture drew. The flag `CALIPER_SCENE_VIEW_NO_HEADER` moves the header's controls into the picture's context menu, for a small view.

The service hands out no texture id: the picture exists only as this item and through export. This is the choice between three forms, and the reason is that the shared camera, the up-axis switch, picking and the overlays come for nothing only if the host owns the view's input, while an applet's layout stays the applet's. An offscreen texture that the applet draws with `ImGui::Image`, as the geometry service's views were, would leave every applet to write its own camera and picking again, as each one does today. A view that the host places in a window of its own would take the layout away from applets that put two views side by side or a view beside their controls.

### Events

`poll_event` returns the next event of the scene's views, oldest first, once per call, and false when none is left; the applet calls it in its frame. A hover reports the item under the pointer when it changes, and a hover end when the pointer leaves every item. A click reports a pick. A rectangle or lasso reports one select event for each entity and primitive it caught, with the rows, at most 32,768 in all, and an empty select clears. An item is the row of the batch's arrays: a point, a segment, strip segment or edge, an arrow, a box, a mesh instance (or the triangle's first vertex when the mesh has no instances), or an image's pixel. Picks and hovers come from the scene core's pick pass, exact for every primitive and a frame or two late. A lasso projects the points and box centres that the CPU can read, as for every scene; an `SRC_ALLOC` array in memory the CPU cannot read, such as an MPS tensor's private storage, is selected by a rectangle from a readback of the pick target, and the lasso is unavailable for it, as on the planned Windows session.

### Capabilities and Refusals

Two capability bits remain, because a backend can truly lack each: `CALIPER_SCENE_CAP_DRAW`, unset where the host has no backend of the graphics layer, as on Windows until the Vulkan backend lands, where `create` returns 0 with that reason; and `CALIPER_SCENE_CAP_IN_PLACE`, unset on a backend that cannot import, such as Vulkan without a paired CUDA device, which copies. Every primitive, instancing, textures, picking and the overlays are present wherever the first bit is set, so no bit marks them.

A call is refused for the reasons Caliper LIB raises `ValueError` for, read from host metadata only: a path outside the grammar; a part the entity cannot hold (a primitive at a camera entity, a camera at an entity with primitives, 3-D data under a camera, an image outside a camera in a 3-D scene); a limit of Scene Calls (`docs/python/scene-calls.md`), Primitives, exceeded; a dtype, component count or row count that does not fit the role or the batch; an allocation that is not live; an offset that is not a multiple of 4; a window that leaves its allocation, checked without overflow; a stride above 2,048; and a record whose `struct_size` is below the first version's. Index values are not checked, because that would read the device; the index-clamp contract holds as for every scene.

### Applet Scene Features

An applet that moves from the geometry service to the scene service deletes its camera code, its picking code and its fallback ladder, and gains, from the host and with no code of its own:

- the shared camera: orbit, pan, zoom, fit, reset to the starting camera, and looking through a pinhole;
- the up-axis switch;
- picking and hover through the pick pass, with events back to the applet, exact for every primitive and with no CPU copy of the data;
- depth testing, the headlight on meshes with normals and flat faces without, and the ground grid and axes mark that follow the up axis;
- the scale bar in the scene's units, labels, and hovers that name the entity, the item and its class;
- export through the scene, at any size ([Export](#export));
- recordings, scrubbing and drawing in a browser, where the scene is wire-capable ([Wire-Capable Scenes](#wire-capable-scenes)).

### Export

`caliper.export.v1` draws a scene instead of geometry records. `view_png(path, width, height, scene, camera)` renders the scene offscreen at the requested size through the scene core, with the given starting camera or, when it is `NULL`, the camera of the scene's first view, reads it back with the graphics layer's `read_texture`, and writes the PNG and its sidecar as today; `begin_sequence`, `frame` and `end_sequence` take the scene in the same way. A second export of one scene and camera is byte-identical on the same backend. The export capability bit follows `CALIPER_SCENE_CAP_DRAW`.

### Wire-Capable Scenes

An applet's scene is the same model that protocol 3.2 carries, so the host can send it where a script's scene goes. A scene is wire-capable when every array can be read by the host and fits the wire: the host can read it back (an `SRC_ALLOC` array on a backend that copies buffers back, which Metal does, or a CPU tensor), and each batch is within the limits of a frame and of the receiving viewer's data view. The host then reads the arrays back in a job, never on the frame thread, packs strided columns into rows, and encodes the scene's parts as protocol 3.2 messages for the app's recorder and for a hub that relays the app's views. A recorded applet scene opens in the scrubber like any recorded scene, in the app and in a browser, and a browser attached to that hub draws it under the parity rules of [Web Geometry](web-geometry.md). Native-only are zero-copy, since every array that leaves the process is copied; the applet's events, since a pick in a browser reaches no applet; and the applet's cadence, since the host sends at most at the view's rate. Applet scenes in recordings and through the hub are planned later.

### C++ Sugar

`caliper.hpp` wraps the table. `caliper::Scene` is created from the `Host` and a description, removes its scene when destroyed, and is falsy without `CALIPER_SCENE_CAP_DRAW`. `caliper::SceneArray::alloc(ref, dtype, count, components, stride)` takes the `(alloc, offset)` reference that the exportable pool's `to_bridge` returns, and `caliper::SceneArray::tensor(t)` takes a `CaliperTensor`. Each record has a builder with the defaults of Caliper LIB, such as `caliper::Points().positions(a).values(v, CALIPER_CMAP_MAGMA, lo, hi).size_px(3).additive()`. `scene.points(path, p)` and the other primitive calls return the token; `caliper::SlotTokens` keeps the tokens of a pool's slots and says when a slot may be written again; `scene.view(key, size)` draws the view item; and `scene.events()` iterates the queued events. The torch adapter gains `adapters::scene_array(pool, tensor)`, which imports an exportable tensor and returns its array.

### Tests

Headless tests of the service run on the graphics layer's recording backend in both CI lanes: each record compiles to the expected draw list, every refusal above is hit with its reason, tokens retire after the frames that read them, and a scene removed with live imports releases them. Capture rows draw one fixture applet scene on Metal against references under the parity rules. An applet built against the installed SDK draws a scene, logs its events and exports it, in the SDK's install test.

### Mapping from the Geometry Service

The geometry service drew a fixed menu of draws into an offscreen texture, with the camera, the state and the order chosen by the applet for every draw. The scene model describes what is drawn and leaves the drawing to the view. Each capability of the geometry service is covered by the scene model as it is, covered after an addition that is a general feature of a scene model in its own right, or dropped, with the applet adapted:

| Geometry service | In the scene model | What changes, and who loses what |
|---|---|---|
| Points of a pixel size (`draw_points`, the points topology) | Points with `size_px` | Covered. Discs with a hard edge instead of hardware points, so the pixels change |
| One-pixel hardware lines, line strips, indexed or not | Lines as segments, strips and index-pair edges | Covered. Quads one pixel wide or wider replace the hardware rule; the rows that depended on it, such as crossing lines with masked ends, are rewritten under the parity rules |
| Triangles, indexed or not | Meshes | Covered |
| Triangle strips | None | Dropped. No applet draws one, and a scene model describes surfaces as indexed triangles; an applet with a strip writes its indices on its device |
| Flat colour, per-vertex RGBA, per-vertex values through the shared colormaps | Colours, `colors` and `values` | Covered. The colormap decides by integer keys and matches the reference byte for byte |
| Textured meshes: UVs and a bridge texture, with the fixed linear clamped sampler | Meshes with `uvs` and a `texture` image | Covered, with one change: the texture is an image array, not a bridge texture id, so that the scene is wire-capable. `twin_scope` colormaps its field into an RGBA image on its device, one gather in torch, before it sets the mesh |
| The headlight, per vertex, from normals, `0.30 + 0.70 · max(n · l, 0)` | Meshes with `normals` | Covered as it is |
| Unlit triangles | Meshes without normals are shaded flat | Dropped. No applet draws unlit triangles, and flat faces are the scene model's rule |
| Opaque and alpha blending, chosen per draw | Opacity from the colours, translucent batches after opaque ones | Covered by the scene's rule instead of per-draw state |
| Additive blending | `blend="additive"` for points and lines | Covered after an addition: additive batches draw after the opaque ones with the depth test on and depth writes off, so dense clouds and trajectories brighten where they overlap and their order does not matter. It is a general feature of point clouds and trails, and five applets draw additive points |
| Views with or without depth; depth test and write chosen per draw | Every view has depth; opaque batches test and write, translucent and additive ones test only | Changed. Points that the geometry service drew without any depth test now test against the scene's other batches, which changes nothing when they are alone. Lines, arrows and box edges gain a small constant depth offset toward the camera, so that an edge lying on a mesh, such as a wireframe, wins over the surface: a general rule of the scene model, added because quads, unlike hardware lines, do not rasterise at exactly the surface's depth |
| A model matrix per draw | Entity transforms | Covered |
| Camera matrices owned by the applet | The view's camera, its starting camera and the shared controls | Dropped. Every applet loses its own orbit code and the freedom to set any projection each frame, and gains the shared camera |
| Instanced draws: (N, 16) poses and a per-instance value through the colormap | Meshes with `instance_poses` and `instance_colors` or `instance_values` | Covered after an addition: many copies of one shape is a general feature. Normals are transformed by the cofactor of each pose's upper 3 × 3, which is exact for any invertible pose |
| The rigidity gate on lit instanced draws, with its tolerance | None | Dropped, because nothing needs it: the cofactor lights every invertible pose correctly, so no pose is refused, and the host no longer reads the instance matrices back before a frame, which on Metal cost a blit and a wait |
| The index-clamp contract | The same contract | Covered as it is |
| One atomic gated frame: every gate before encoding, the whole frame refused with its pixels untouched | Refusal per part, with the previous part kept; every accepted call of a frame appears together | Changed, as [Frames, Atomicity and Memory Stability](#frames-atomicity-and-memory-stability) explains |
| The memory-stability contract, in place until the view's next draw, drained before the call | The same two halves, with tokens that say when bytes may be rewritten | Covered |
| An offscreen view as a bridge texture id, shown with `ImGui::Image` | The view item | Changed: no texture id is handed out |
| The refusal of a geometry view named as a texture | None | Not needed, because no view texture is handed out |
| Four caps bits, one per revision | `CALIPER_SCENE_CAP_DRAW` and `CALIPER_SCENE_CAP_IN_PLACE` | Changed |
| Export drawing geometry records | Export drawing a scene | Changed |
| Byte-exact rows against a CPU reference | Rows under the parity rules | Changed, as below |

**The change of contract.** The geometry service's rows compare every pixel with one CPU reference, byte for byte, with lit pixels within 2 levels, and they depend on hardware points and lines and on state the applet chose per draw. The rows of the graphics suite that replace them are written for the scene service and follow the parity rules of [Web Geometry](web-geometry.md#parity-with-the-native-renderers) natively as in a browser: a pixel whose colour a lookup decides away from edges (a flat colour, an RGBA colour, a class colour, a value through the colormap by keys, a texel sampled nearest) is byte-exact; a pixel within one pixel of an edge may take either side; lit pixels may differ by 2 levels, and blended and linearly sampled ones by 1. The rows keep the geometry rows' fixtures where they still apply. The colormap rows, the texture path's rows and the graphics layer's own rows stay byte-exact. Export is deterministic, a second export of one scene being byte-identical, but no longer byte-exact against a CPU reference for every pixel. This is a real change of the graphics contract, which was accepted in the choice of a scene service for applets.

## Changes to the Shared Contracts

- **Additive blending and instanced meshes.** Points and lines take `blend`, and a mesh takes `instance_poses` with `instance_colors` or `instance_values`, so that the scene model covers what applets drew through the geometry service ([Primitives](#primitives), [Mapping from the Geometry Service](#mapping-from-the-geometry-service)).

## Decisions

- **A starting camera from the script, and no other viewport work for now.** `view_camera` is a starting value like `up`: a person's own camera wins once it is moved and is remembered per view, and a reset returns to `view_camera`. A 2-D scene takes none, because its fit already frames it whole.
- **Device-given entity transforms travel through slots and are read on the CPU.** World matrices, bounds and frustums are composed on the CPU, and no read of the device is ever waited for.
- **A mesh over its budget becomes its bounding box.** Decimating on the device is costly and changes the topology a person is looking at.
- **Depth and segmentation are reduced by stride.** Mean pooling would invent depths and class ids.
- **The text log is retained whether or not anything is attached.** Lines are cheap host strings and are often read after the fact.
- **The names `caliper.text` and `caliper.bars`, beside the built `caliper.scene`.** Each view call of Caliper LIB is named after its kind, as `points`, `series` and `histogram` are, and none of the names collides with the package.
- **The shared-memory protocol keeps version 2.** Its append-only rules allow the additions, and a viewer from before refuses only the new views.
- **A scene service for applets, replacing the geometry service.** Applets get the scene model, its camera, picking, overlays and export through one service instead of a fixed menu of draws with their own cameras, at about 25.5 builder days of migration in place of about 8.5 for moving the geometry service onto the graphics layer and giving it one version, with eight applets rewritten, seven from the geometry service and one from ImPlot3D, the geometry service deleted and the graphics contract for applets changed from byte-exact rows to the parity rules; [Renderer Core](renderer-core.md#migration) gives the steps.
- **The view as an ImGui item that the host draws in the applet's window.** The host then owns the input, so the camera, the switch, picking and the overlays come for nothing, and the applet keeps its layout. No texture id is handed out.
- **Refusal per part, not per frame.** A scene keeps every part it holds, so a refused part cannot blank the picture, and the parts that were accepted still apply.
- **Tokens for memory stability.** An applet learns without waiting when the bytes of a replaced batch may be rewritten, which fits the triple-buffered slots the applets already use.
- **Arrays name allocations or tensors, with the dtypes the model draws.** The service then converts nothing on the device, and an allocation is drawn in place through the same ids as the tensor bridge's.
- **Additive blending and instanced meshes added to the scene model; triangle strips, unlit triangles, per-draw depth state, applet-owned camera matrices and the rigidity gate dropped.** The additions are general features of a scene model that applets need; what is dropped belongs to a fixed-function draw menu, not to a description of a scene, and no applet loses a picture it needs.
- **A texture is an image array, not a bridge texture id.** The scene then stays wire-capable, and an applet that colours a field colormaps it on its device.

## Later Scene Work

These scene features are planned after the work above:

- **More shapes:** ellipsoids, capsules, cylinders and planes, drawn from their parameters.
- **Mesh files** (glTF, OBJ, STL) through file loaders, and a URDF helper in Caliper LIB.
- **Events from the app back to the script over shared memory:** an event ring in the scene's directory segment. Until then a script that needs events is attached through `caliper.serve()`.
- **Named frames,** through the import of robot recordings, which translates them into the entity tree. A second transform system beside the entity paths stays out.
- **The viewport features:** preset views, an orthographic projection, following an entity, saved viewpoints and linked cameras.
- **Trails over a time window,** slice planes and isosurfaces of 3-D fields, and comparing runs.
- **The drawing of the text log and the bar chart,** natively, in DOM and in Canvas 2D, and applet scenes in recordings and through the hub. The codec of protocol 3.2 already carries the messages of both views.

Compressed video stays out entirely: Caliper streams data, not video, so only single JPEG and PNG frames travel, as [Primitives](#primitives) describes.
