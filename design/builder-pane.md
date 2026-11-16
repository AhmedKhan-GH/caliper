# Builder pane

The builder pane is the WYSIWYG way to build a spec by hand, item 7 of the integration order in [Builder overview](builder-overview.md). A person picks a source, picks a primitive, gets a panel, arranges the panels, edits their properties and sees the spec text follow. Every edit is a command from the `builder` origin through the session's [command layer](command-layer.md), so it is journaled and can be undone, and nothing reaches around the command layer. This page describes the pane as it is built: its regions, its gestures and the commands each one sends, how the text stays the source of truth, the pane on the app's spec page, a new dashboard and files dropped on the window, the capture tool and the tests. The reference page for a person is Builder pane (`docs/references/builder-pane.md`).

## What is built and what is left

Built in `src/host/builder/` as the library `caliper_builder_pane` (with `CALIPER_WITH_BUILDER` only), included from the root build by `builder_pane.cmake`:

- `builder_pane.h` and `builder_pane.cpp`: `BuilderPane`, the pane's state and gestures over a `BuilderPort`, with `SessionBuilderPort` over a builder `Session`;
- `builder_pane_edits.h` and `.cpp`: the pure helpers, which need no window and no session: the shapes of sources, the primitives with their slots and options, the bindings a dropped source takes, ids and renames, and the edits of the layout tree;
- `builder_pane_layout.h` and `.cpp`: the pane's regions and the canvas's drop targets as plain rectangles;
- `builder_pane_draw.cpp`: the ImGui side in the house style, which turns the mouse and the keys into the gestures;
- `builder_shot_main.cpp`: `caliper-builder-shot`, a development tool that is not installed, which builds the three-panel dashboard by hand through the pane's own input on an offscreen canvas;
- `spec_page_layout.h` and `.cpp` (the library `caliper_spec_page_layout`): `layout_spec_page()`, every rectangle of the spec page for each combination of build mode, the text strip and the side chat;
- in the app: the pane on the spec page (`spec_page.h`), the page bar's items, files dropped on the window (GLFW's drop callback in `src/main.cpp`), F3 New dashboard on the landing screen (`src/host/ui/landing_screen.h`), and the hidden input driver's builder steps (`spec_ui_driver.h`);
- the tests in `src/host/builder/tests/`, run as `caliper_builder_pane` (headless), `caliper_builder_pane_gpu` (the tool as a process) and `caliper_builder_app` (the hidden app).

In the spec runtime, `SpecLayout` gained `splits`, every split node on a shown path with its rectangle, axis, divider and ratio, so that the pane hits and drags a divider from the layout's own rectangles (`src/host/spec_runtime/spec_layout.h`). Nothing else of the runtime changed.

Left: the pane on the live page, and Windows, where the builder is off.

## The regions

Build mode adds two columns and a strip beside the page's own canvas, as the placement sketch of the overview places them. The regions are computed without ImGui by `compute_builder_pane_layout()` from the work area that the host's strips leave; on the spec page through `layout_spec_page()`, which also gives the page's rectangles outside build mode:

| Region | Place | Size |
|---|---|---|
| Sources | Left column, upper part | The rest of the column above Primitives |
| Primitives | Left column, lower part | Its header and seven rows |
| Canvas | Centre | The rest of the width; the largest region |
| Spec text | Under the canvas, while it is open | 30 % of the centre's height, at least 9 lines |
| Properties | Right column, upper part | The whole column, or the part above the side chat |
| Side chat | Right column, lower part, while ⌘L Chat is on | 55 % of the column |

The left column is 17 % of the width and at least 14 lines of the UI face; the right column is 21 % and at least 18 lines; the canvas is at least 24 by 16 lines. With the side chat open, the columns take the chat layout's shares and minimums (`ChatColumnInput`, `ChatLayoutMetrics` in `side_chat.h`): the right column 28 % and at least the chat's 300 points, Properties 45 % of it above the chat, and the left column 18 %, so the chat keeps the width it has outside build mode. Outside build mode the chat's own column is what `layout_chat_column()` gives, which `ChatPane::draw_column()` claims as a viewport side bar. Every minimum grows with the text scale. A 1 px rule separates the regions. At 2560 by 1440 points, with the page bar and the status bar taking 50, the left column is 435 px, the canvas 1585 and the right column 538.

On a small window the regions give way in a fixed order: the text strip shrinks to its minimum and is then hidden, the right column and then the left shrink to their minimums, then the right column is hidden, then the left. The canvas never goes below its minimum while a column is shown. `BuilderPaneLayout::collapsed` lists what gave way, in that order.

## Sources

The palette lists the open spec's own sources first, under Spec, and then the catalog's sources (`catalog.list`, read again whenever the catalog's revision grows) grouped as Tables, Tensors, Live, Feeds and Metrics. Each row has its name and, at its right in mono, its shape in the style of the panel headers: "4,000 × 3" for a table, "f32 4×8×48×48" for a tensor, and a spec source's shape after its transforms as the validator computes it. A tabular row opens to its columns, each with its DuckDB type. A filter line narrows the rows by name, id or column.

A press on a row selects it and focuses it in Properties; moving the pointer then drags it. A row's column is selected and dragged the same way. The other way without dragging is Add (⏎ in the list, the header's ⏎ Add, or a double click), which makes a panel of the selection's default primitive.

## Primitives

The seven primitives as nouns. A click adds a panel of that primitive bound to the selected source by the pane's rules (`pane_auto_bind()`): a series takes a table's first column as x and its other numeric columns as lines, a heatmap a rank-2 tensor, an image `[H, W]`, `[H, W, 3]` or `[H, W, 4]`, points a tensor `[n, 2]` or `[n, 3]`, a histogram a numeric column or any tensor, a table the whole source, and text a column or a one-element tensor. A primitive that the selection does not fit is drawn faint, with the validator's message as its tooltip, and a click on it sends nothing. Dragging a primitive onto the canvas places it.

A panel needs its required slots: the validator refuses one without them (`spec.missing_slot`), so the pane cannot make an unbound panel with its empty state, as the overview's item 7 first said. Only a text panel stands alone, with `options.text` set to "Text".

## The canvas

The canvas is the spec runtime's own drawing of the open spec. In build mode:

- **Focus.** A click inside a panel focuses it; the focused panel has the bronze frame of the spec page, and Properties shows it. ⌫ removes it (`panel.remove`).
- **Moving.** A panel's title strip drags it. Dropped on another panel's edge it goes left of, right of, above or below that panel; on its centre, into tabs with it; in the band along the canvas's border, at that edge of the whole layout.
- **Resizing.** A divider between the two children of a split drags its ratio, in hundredths from 0.05 to 0.95, with the ratio shown beside the pointer. A split of more than two children has no ratio in the format, so its dividers do not move.
- **Drops.** A source, a column or a primitive dropped on the canvas makes a panel there, by the same zones as a move. A source or a column held over a panel shows that panel's slots as a column of rows in its centre, the ones it fits in parchment; dropped on a slot, it binds it, and on `series.y` it adds a line.
- **Where it lands.** While something is dragged, the pane works out the spec the drop would make, lays it out with the runtime's own layout, and outlines the rectangle the panel would take; a drop that does not fit is outlined in secondary with the validator's message as a tooltip, and releasing it sends nothing.

The zones are rectangles (`hit_drop()`): a panel's edge band is its outer quarter, at most six lines; the border band is ten pixels at scale 1.

## Properties

The right column's upper part, under a tab strip that holds Properties and the focused item's name. With a panel focused:

- **Panel**: its id, its primitive (a choice among the seven, which drops the slots and options the new primitive does not take and binds its required slots by the pane's rules), and its title, whose placeholder is the title the runtime shows without one.
- **Bindings**: one row per slot of the primitive, required slots marked with an asterisk; `series.y` has one row per line and one to add a line. Each row is a choice among the spec's sources and their columns. Every choice is checked against the open spec as the session would check it, and one that does not fit is faint with the validator's message as its tooltip and is refused before anything is sent; so is unbinding a required slot.
- **Options**: exactly the options the primitive takes: labels as text, ranges as two numbers whose placeholders are the data's extent, dimmer than any value, the log flags, the colormap as four ramps, bins, the table's columns as one flag each, and the text panel's text.

With a source focused: its id, its catalog id, its shape, and for a tensor its transforms as an ordered list, each with its fields (slice: axis, index or start, stop and step; reduce: axis and the function; normalise: the mode and axis) and the shape after it, from the validator; Slice, Reduce and Normalise add one, and × removes one. A catalog source that the spec does not read yet takes an id before it is added; its first transform, or the first panel made from it, adds it with that id. For a table, its columns. With nothing focused: the spec's id, title, description, its numbers of panels and sources, and its file.

## The spec text

A strip under the canvas, opened with ⌘T Text in the page bar. It shows the open spec's canonical TOML, read-only, with line numbers. ⌘E Edit turns it into an editor, and esc Done back.

**Text is the source of truth, and the two can never disagree.** The pane keeps no spec of its own. The canvas, Properties and the strip all read one `OpenSpec` of the session, the spec and its canonical text read under the session's lock, which the pane reads again after every command it sends and whenever the port's version moves (the journal, in the app). Every change, from the canvas, Properties or the text, is a command on the session. A draft is the one thing that can differ from the open text: each change of it is validated with the session's resolver, and applied with `spec.set_text` the moment it validates. While it does not, the strip shows each diagnostic at its line, with its code at the line's end and its message as a tooltip, the canvas keeps drawing the last valid spec, and the pane refuses every other edit (`builder.draft`), so no edit is lost between the two. When the open spec changes under an applied draft, from an undo or an agent, the draft becomes the new canonical text.

## Gestures and commands

Every command is sent from the `builder` origin, but the import of dropped files, which is the person's `ui`. The commands of one gesture share a group of the pane's own, and a value that is typed or dragged sends each change with a merge key under the group of its gesture, which begins when the field or the divider is taken and ends when it is let go. The registry's groups join the top step whatever the time between two commands, so one gesture is one undo step even with pauses longer than the merge window.

| Gesture | Commands |
|---|---|
| Add, a click on a primitive, a drop on a panel's right, top or bottom edge or its centre | `source.add` when the source is not in the spec, then `panel.create` with `place` |
| A drop on a panel's left edge or at an edge of the whole layout | `source.add` when needed, `panel.create`, `layout.set` |
| A source or column dropped on a slot, a binding chosen | `source.add` when needed, `panel.patch` of `bind` |
| A title, an option | `panel.patch` |
| A primitive chosen | `panel.patch` of `primitive`, with the slots and options it does not take removed |
| A panel's or a source's id | `spec.set_text` of the renamed spec |
| A transform added, changed or removed | `source.add` for the first one of a catalog source, then `source.patch` of `transform` |
| A panel moved onto another's right, top, bottom or centre | `layout.place` |
| A panel moved onto another's left or an edge of the whole layout, a divider dragged | `layout.set` |
| ⌫ Remove | `panel.remove` or `source.remove` |
| The spec's title or description, an edit of the text | `spec.set_text` |
| ⌘Z, ⇧⌘Z | `history.undo`, `history.redo` |
| ⌘S Save | `applet.save`, with `overwrite` when the open spec already is that applet |
| Files dropped on the window | `data.import` with `run`, through the session's queue |

Before it sends anything, the pane applies every command of the gesture in order to a copy of the open spec with `apply_spec_edit()` and the session's resolver, as the session will apply them. A gesture that does not fit is refused there, with the validator's or the command's own code and message, and nothing is sent; in the rare case that a command of a group still fails, the pane undoes what of the group ran. While a long command holds the session (`ready()` false), every gesture is refused with `builder.busy`. The page bar shows the last refusal's code, with its message as a tooltip.

Three gestures have no command of their own, and use one that exists rather than a new one: a rename is `spec.set_text` of the spec with every reference renamed, since ids are immutable in `panel.patch` and `source.patch`; a place left of a panel or at an edge of the whole layout is `layout.set`, since `layout.place` has no left and no whole-layout target; and a new panel there is `panel.create` followed by `layout.set` in one group.

## Keys

| Key | Action |
|---|---|
| ⌘B | Build mode on or off, on the spec page |
| ⌘T | The spec text strip |
| ⌘E, esc | Edit the text, and back |
| ⌘Z, ⇧⌘Z | Undo and redo, while no text field has the keyboard (a field's own undo comes first) |
| ⌘S | Save |
| ⏎ | Add the selection, in the Sources list |
| ⌫ | Remove the focused panel |
| esc | Cancels a drag, leaves the text editor, gives up a source's focus; the page then gives up a panel's focus and goes Home |

Off macOS each ⌘ is Ctrl, and the key buttons say so.

## The file drop

`BuilderPane::drop_files(paths)` submits `data.import` of the paths with `run` from the `ui` origin, a file the person dropped, through the session's queue, which the app drains and runs on its worker, so the loaders run there through the command; the paths outside the workspace are granted, as for the person's origins. `update()` reads the answer: the sources the import made are selected in the palette (`fresh()`), the first focused in Properties, and each refused path is kept with its code and reason (`refused()`), which Sources lists under Refused until the next drop.

In the app, GLFW's drop callback (`glfwSetDropCallback` on the window) hands the paths to the app's `on_drop()`, which keeps them for the frame loop. On the spec page they go to `SpecPage::drop_files()`, which turns build mode on first; on the landing screen the open spec's page opens in build mode, or a new Untitled spec when none is open, and the files follow once the page shows; other pages keep their windows and log the drop in Activity. A hidden window receives no real drop, so the input driver's `drop` step calls `on_drop()`, the function the callback calls.

## A new dashboard

The landing screen's Projects pane has a third way in, F3 New dashboard, with the command `new <title>`: F3 or its key button takes the command line's text (without a leading `new `) as the title, and Enter on a command line that begins with `new ` does the same. The command line is cleared once its text is taken, and reloaded in the line while it still has the keyboard, since an active ImGui text field writes its own text back over its buffer. The app sends `spec.new` from the `ui` origin through the queue with the title and an applet id that `pane_new_spec_id()` derives from it (lower-case letters and digits, one `-` for every other run, `d-` before a leading digit, `-2` and on while the workspace's `applets/` folder has the id), and shows the page in build mode when it succeeds (`SpecPage::request_new()`, `take_new()`). The spec's own id, title and description are edited in Properties with nothing focused, through `spec.set_text`, and ⌘S saves it as `applets/<id>/`, which the landing lists at once.

## The capture tool

`caliper-builder-shot --workspace <dir> --applet-from <spec.toml> --files <file>... --out-dir <dir> [--size WxH] [--prefix NAME] [--suffix TEXT] [--text-scale S]` begins an empty spec with the applet of `--applet-from`, drops the files on the pane, and then builds the dashboard of `dashboard.spec.toml` through the pane's own input: every click, drag, key and typed text goes into ImGui's event queue as a platform backend feeds a person's input (⌘ as the Super key, which ImGui swaps on macOS), aimed at the pane's named rectangles of the last frame (`item_rect()`). It writes `<prefix>-<step><suffix>.png` at six steps: the empty spec with the two palettes and the two sources; the curves made from the metrics table as a series, renamed, titled and labelled, with Properties showing its bindings; the attention source with its three transforms and the heatmap dragged beside the curves, with Properties showing the transforms and their shapes; the table dropped under everything with its columns chosen and the root divider held at 0.65; the spec text open beside the finished dashboard; and the dashboard after two undos. It then redoes both and saves, and prints the texts, the undo steps and every command it sent. The stand-in page bar it draws is the app's: Home, the title, the pane's items, and ⌘K Command at the right end.

## The pane on the spec page

The page (`spec_page.h`) owns one `BuilderPane`, made when it binds the session, with a `SessionBuilderPort` whose `ready()` is false while the session's worker runs (`AppSession::worker_busy()`) and whose version is the journal's count (`AppSession::journaled()`), and with `spec_page_layout_rule()` as its layout rule. Each frame, on the frame thread:

1. The page bar: Home, the title, the command input while it is open, then `builder->page_bar(room)` (⌘B Build, and in build mode ⌘T Text, ⌘Z Undo, ⇧⌘Z Redo, ⌘S Save), where `room` is the width left before ⌘L Chat and ⌘K Command at the right end; with less room, only Build. A spec applet's file drawn from outside the workspace has no Build.
2. The keys: an open inline prompt takes Escape first, then the command input, then `builder->keys(escape)` (⌘B, and in build mode ⌘T, ⌘Z, ⇧⌘Z, ⌘S, ⌫, and Escape for a drag, the text editor or a source's focus), then the page's own order (a panel's focus, then Home). ⌘K stays the page's: with a panel focused it opens the inline prompt in that panel, in build mode too.
3. The strips: in build mode the side chat does not claim its viewport column; the pane draws it in the right column's lower part through `ChatPane::draw_in()` (`set_chat()`). The approval cards' strip, the jobs tray and the status bar keep their places, and the work area is what they leave.
4. The canvas: in build mode `builder->draw(area, *runtime, look, &focused)` in place of the runtime's draw and the page's focus, so one focused panel is shared: the pane takes the page's focus when the page changed it (the command input finding a panel, Escape) and gives its own back. The page's overlay still draws the focused panel's frame, the approval cards' REVIEW mark and the inline surface in each panel.

An approved agent edit, a ⌘K Accept and a hand edit go through the session's one history, so ⌘Z Undo in the page bar undoes whichever came last; a test checks it. The page leaves build mode when it is left, so a spec opened again from the landing shows out of build mode.

## Tests

- `caliper_builder_pane`, headless, over a real `Session` whose workspace imports a Parquet table and a safetensors tensor through `drop_files()` (a pickle and a missing file refused with their codes): the dashboard built gesture by gesture through the pane's functions, each exactly one undo step of commands from the `builder` origin under one group, the typed and dragged ones with one merge key, ending equal to `dashboard.spec.toml` and saved as an applet whose file equals it, with two undos and two redos between; a binding, a primitive, a drop, an id and a removal that do not fit refused with nothing sent; drops on a slot, on edges, in a centre and at the whole layout's edges; the text strip both ways, with an invalid draft's diagnostic at its line, the other edits held, and the draft following an undo; renames reaching every reference; a busy session; and the regions at 2560 by 1440 and 1440 by 900, at two text scales, with what gives way on a small window, the drop zones, the dividers and the strips of the dashboard, and the pure helpers; `layout_spec_page()` at both sizes for every combination of Build, Text and Chat, the chat's column as wide in build mode as outside it; an agent's edit approved through the queue, a ⌘K Accept through the inline surface with the scripted agent, and a hand edit undone in turn through the pane's Undo; and a new spec's id from its title.
- `caliper_builder_pane_gpu`: `caliper-builder-shot` as a process over a new workspace; the spec it ends with and the applet it saves parse equal to the fixture, as the other three ways of building it are checked; two undos take two steps; every command and every journal line is the `builder` origin's; the six captures have the size asked for.
- `caliper_builder_app`: `caliper --hidden` on an empty workspace with `CALIPER_TEST_SPEC_UI`: the two files dropped on the landing, which makes a new spec in build mode with both sources; the spec named; the dashboard built through the pane; the text strip and the side chat opened in build mode; ⌘S; Home; and the dashboard opened again from the landing's command line. The saved spec equals `dashboard.spec.toml`, the journal holds `spec.new` and `data.import` from `ui`, the build from `builder` and `applet.open` after `applet.save`, and the six captures have the window's size. A second case checks that the landing's command line is empty when the person comes back Home after New dashboard took its text, by F3 and by `new <title>` and Enter (the input driver's `command-line` step).
- `caliper_spec_runtime` checks the new split records on the dashboard.

## Decisions

- **A panel is made bound, or not at all.** The validator refuses a panel without its required slots, so Add needs a selection that fits, and a primitive that it does not fit is faint with the validator's message.
- **Renames are `spec.set_text`.** Ids are immutable in the patch commands and the format has no rename; the text of the renamed spec is a command that exists, journaled and undone as one step.
- **Places that `layout.place` cannot name are `layout.set`,** with the new panel made first in the same group.
- **One group per gesture, and a merge key for a typed or dragged value.** Groups join the top step at any interval, so a pause longer than the merge window still leaves one undo step; the merge key keeps one record for a long drag.
- **Every gesture is checked on a copy before it is sent,** with the session's own edit function and resolver, so a refusal sends nothing and its message is the validator's.
- **A ratio is dragged in hundredths,** so that the text holds values such as 0.65.
- **A catalog source takes its spec id before it is added,** in Properties, so that a source can be named once and never renamed.
- **A new dashboard starts from the landing's Projects pane,** as a third way in beside Open folder and Attach to process, titled from the command line, because a spec page needs an open spec; its id comes from the title and is renamed in Properties when it must be another.
- **A drop on the landing opens build mode,** on the open spec or on a new Untitled one, because the files land in Sources, which only the spec page has; other pages keep their windows.
- **Dropped files are the person's `ui` origin;** the pane's own gestures are `builder`.
- **In build mode the chat keeps its width** and Properties takes 45 % of the column above it, the chat layout's shares, so toggling Build beside an open chat does not move it.
- **The page keeps the focus and ⌘K;** in build mode the pane takes the page's focus when the page changed it and gives its own back after each draw, so the bronze frame, ⌘K and Escape mean the same in both modes.

## Limits

- A split of more than two children shares its space equally in the format, so its dividers do not move.
- A tab is not dragged by its tab; a panel in tabs moves through the spec text.
- The transforms of a source are added, changed and removed, not reordered.
- The pane offers no Selection tab of its own; the right column's tab strip has room for the live page's Selection when the page gives it.
- The binding choices list the spec's own sources; a catalog source joins the choices once it is in the spec, by a drop or by Add.
