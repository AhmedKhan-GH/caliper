# Builder pane

The builder pane is the WYSIWYG way to build a spec by hand, item 7 of the integration order in [Builder overview](builder-overview.md). A person picks a source, picks a primitive, gets a panel, arranges the panels, edits their properties and sees the spec text follow. Every edit is a command from the `builder` origin through the session's [command layer](command-layer.md), so it is journaled and can be undone, and nothing reaches around the command layer. This page describes the pane as it is built: its regions, its gestures and the commands each one sends, how the text stays the source of truth, the pane on the app's spec page, a new dashboard and files dropped on the window, the capture tool and the tests. The reference page for a person is Builder pane (`docs/builder/builder-pane.md`).

## What is built and what is left

Built in `src/host/builder/` as the library `caliper_builder_pane` (with `CALIPER_WITH_BUILDER` only), included from the root build by `builder_pane.cmake`:

- `builder_pane.h` and `builder_pane.cpp`: `BuilderPane`, the pane's state and gestures over a `BuilderPort`, with `SessionBuilderPort` over a builder `Session`;
- `builder_pane_edits.h` and `.cpp`: the pure helpers, which need no window and no session: the shapes of sources, the primitives with their slots and options, the bindings a dropped source takes, ids and renames, and the edits of the layout tree;
- `builder_pane_layout.h` and `.cpp`: the pane's regions and the canvas's drop targets as plain rectangles;
- `builder_pane_draw.cpp`: the ImGui side in the house style, which turns the mouse and the keys into the gestures;
- `builder_shot_main.cpp`: `caliper-builder-shot`, a development tool that is not installed, which builds the three-panel dashboard by hand through the pane's own input on an offscreen canvas;
- `spec_page_layout.h` and `.cpp` (the library `caliper_spec_page_layout`): `layout_spec_page()`, every rectangle of the spec page for each combination of build mode, the text strip and the side chat;
- in the app: the pane on the spec page (`spec_page.h`), the page bar's items, files dropped on the window (GLFW's drop callback in `src/main.cpp`), on the spec page, the landing screen and the live page, "Save changes to <name>?" (`save_prompt.h`), F3 New dashboard on the landing screen (`src/host/ui/landing_screen.h`), and the hidden input driver's builder steps (`spec_ui_driver.h`);
- the tests in `src/host/builder/tests/`, run as `caliper_builder_pane` (headless), `caliper_builder_pane_gpu` (the tool as a process) and `caliper_builder_app` (the hidden app).

In the spec runtime, `SpecLayout` gained `splits`, every split node on a shown path with its rectangle, axis, divider and ratio, so that the pane hits and drags a divider from the layout's own rectangles (`src/host/spec_runtime/spec_layout.h`), and `set_empty_hint()`, the line an empty spec shows under "No panels" in build mode. Nothing else of the runtime changed.

Built in sprint 3.1: Sources lists the catalog one kind at a time and groups metrics by run ([Sources](#sources)), and a panel is removed from the page with the × on its strip, Remove panel in its right-click menu and ⌫, in build mode and on the dashboard page. All three call `BuilderPane::remove_panel()`, one `panel.remove`: from the `builder` origin in build mode, as the pane's other gestures and Remove in Properties, and from `ui` on the dashboard page, where ⌘Z and ⇧⌘Z now undo and redo too. The decision for ⌘K: while the inline prompt is open or a preview is shown, no panel offers removal at all, since a preview's panels are only proposed and removing the panel a preview is on would leave the preview pointing at nothing; the person accepts or rejects first. The page's `remove:<panel>`, `panel:<panel>` and `menu:remove` are named rectangles for the input driver, whose `menu <item>` step right-clicks.

Built after the walk-through's dry run of building a dashboard by hand: the empty canvas's line, the axes of a slice and a reduce with their ranges and names, a slice's index kept in its range, ⌘O Open in browser in the page bar ([The pane on the spec page](#the-pane-on-the-spec-page)), and the landing's Projects listing the workspace's dashboards ([A new dashboard](#a-new-dashboard)).

Left: the pane on the live page, and Windows, where the builder is off.

Open items, found in the walk-through's dry run and left for later:

- Activity shows a row for every `layout.set` of one divider drag; they are to be grouped as the side chat groups repeated calls of one command.
- Nothing tells a person that F3 New dashboard takes the command line's text as the new dashboard's title.

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

Metrics are grouped by run, since a store of many training runs holds one metric per run and tag. Each run is one line, with its name (`Run <id>` when it has none), its number of metrics and its id, newest first: the metrics catalog's run id, which only grows and is never given again, orders the runs, so the store needs no start time. A click opens a run to its metrics, each a row like any other, named by its tag and drawn under the run, which selects, drags, opens to its columns and adds as before. Only the newest five runs are listed until Show all, after the last run, which lists every run and becomes Show fewer. The filter searches every run: a run is listed, open, when its name, its experiment or one of its metrics matches, with all its metrics when the run itself matches and the matching ones otherwise, and Show all is not offered while it filters. `catalog.list` keeps its flat ids and results; only the pane groups them. The list as drawn (`BuilderPane::list_items()`) is made again only when the rows, the filter, an opened run or Show all change, never every frame, and a closed row scrolled out of sight takes its space only, so hundreds of runs cost little.

The pane lists the catalog one kind at a time, with `catalog.list`'s largest limit, 1000, for each. One list of every kind, sorted by id, put `metric:` before `table:` and `tensor:`, so a store with more than about a thousand metrics cut every table and tensor out of Sources. A list of metrics that is cut at the limit is listed again by the next character of the metric's name, `<run id>/<tag>`: each digit, and the `/` that ends a run's id, which reaches every metric in a few calls per thousand. The other kinds keep the first 1000 of each.

A press on a row selects it and focuses it in Properties; moving the pointer then drags it. A row's column is selected and dragged the same way. The other way without dragging is Add (⏎ in the list, the header's ⏎ Add, or a double click), which makes a panel of the selection's default primitive.

## Primitives

The seven primitives as nouns. A click adds a panel of that primitive bound to the selected source by the pane's rules (`pane_auto_bind()`): a series takes a table's first column as x and its other numeric columns as lines, a heatmap a rank-2 tensor, an image `[H, W]`, `[H, W, 3]` or `[H, W, 4]`, points a tensor `[n, 2]` or `[n, 3]`, a histogram a numeric column or any tensor, a table the whole source, and text a column or a one-element tensor. A primitive that the selection does not fit is drawn faint, with the validator's message as its tooltip, and a click on it sends nothing. When the rules bind nothing, the message goes on to say why in the source's own terms (`pane_fit_hint()`): what the source is, what the primitive takes, and, for a tensor with more dimensions than the primitive draws, that Properties slices or reduces it, as in "attention_attn is f32 4×4×32×32, and a heatmap takes a tensor of two dimensions: slice or reduce it in Properties." Dragging a primitive onto the canvas places it.

A panel needs its required slots: the validator refuses one without them (`spec.missing_slot`), so the pane cannot make an unbound panel with its empty state, as the overview's item 7 first said. Only a text panel stands alone, with `options.text` set to "Text".

## The canvas

The canvas is the spec runtime's own drawing of the open spec. A spec with no panel says "No panels" and, in build mode, how to start under it: "Drop a data file, or pick a source in Sources" (`BuilderPane::canvas_hint()`, drawn by the runtime in the empty state's place of the applet id, in the house face). In build mode:

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

With a source focused: its id, its catalog id, its shape, and for a tensor its transforms as an ordered list, each with its fields (slice: axis, index or start, stop and step; reduce: axis and the function; normalise: the mode and axis) and the shape after it, from the validator; Slice, Reduce and Normalise add one, and × removes one. A slice's and a reduce's axis is a choice of every axis of the step's input, each written by `pane_axis_text()` with its index range, `axis 1 · 0–3`, and with its name where the source carries names, `axis 0 layer · 0–3`; the names follow the transforms (`pane_name_dims()`: a slice's index and a reduce without keepdims take their axis's name away). A live view's names are the stream's payload axes, which the live catalog keeps per view (`LiveCatalog::dim_names()`) and the app hands the pane's port (`BuilderPort::dim_names()`); a safetensors tensor has none, and while the input's shape is not known the axis is a number. A slice's index shows its axis's range at the right of its field, and `set_slice_index()` sets a value outside it to the nearer end (`pane_clamp_index()`), with the note `7 is outside 0–3` in the range's place until the next transform edit; a change of axis keeps the index in the new axis's range the same way. A reduce's axes are a list: a click adds the axis or takes it out (`set_transform_axis()`), never leaving it empty, and `transform_axes_chosen()` gives every axis in it, which Properties highlights. A catalog source that the spec does not read yet takes an id before it is added; its first transform, or the first panel made from it, adds it with that id. For a table, its columns. With nothing focused: the spec's id, title, description, its numbers of panels and sources, and its file.

## The spec text

A strip under the canvas, opened with ⌘T Text in the page bar. It shows the open spec's canonical TOML, read-only, with line numbers. ⌘E Edit turns it into an editor, and esc Done back.

**Text is the source of truth, and the two can never disagree.** The pane keeps no spec of its own. The canvas, Properties and the strip all read one `OpenSpec` of the session, the spec and its canonical text read under the session's lock, which the pane reads again after every command it sends and whenever the port's version moves (the journal, in the app). Every change, from the canvas, Properties or the text, is a command on the session. A draft is the one thing that can differ from the open text: each change of it is validated with the session's resolver, and applied with `spec.set_text` the moment it validates. While it does not, the strip shows each diagnostic at its line, with its code at the line's end and its message as a tooltip, the canvas keeps drawing the last valid spec, and the pane refuses every other edit (`builder.draft`), so no edit is lost between the two. When the open spec changes under an applied draft, from an undo or an agent, the draft becomes the new canonical text.

The read-only strip follows each change of the open spec, from any origin: the pane keeps the last line that changed (`pane_text_follow_line()`, the last line that differs once the lines the two texts share at their start and end are set aside, or the line in place of lines taken out), and the strip scrolls it into view at its next draw when it is out of view, with two lines after it. A new panel's block, which ends the text, and a title changed in the middle both come into view. Another spec opened, with another applet id, is not followed.

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

In the app, GLFW's drop callback (`glfwSetDropCallback` on the window) hands the paths to the app's `on_drop()`, which keeps them for the frame loop. On the spec page they go to `SpecPage::drop_files()`, which turns build mode on first; on the landing screen the open spec's page opens in build mode, or a new Untitled spec when none is open, and the files follow once the page shows; on the live page a new Untitled spec opens in build mode, unless the open spec has unsaved changes, which takes the files as on the landing screen (`SpecPage::has_unsaved_spec()`), and the attached session stays attached, so its `live:` sources are in Sources beside the files'; an applet's page keeps its windows and logs the drop in Activity. A drop asks for at most one new spec: when that `spec.new` fails, the drop is refused in Activity rather than asked again. A hidden window receives no real drop, so the input driver's `drop` step calls `on_drop()`, the function the callback calls.

## A new dashboard

The landing screen's Projects pane has a third way in, F3 New dashboard, with the command `new <title>`: F3 or its key button takes the command line's text (without a leading `new `) as the title, and Enter on a command line that begins with `new ` does the same. The command line is cleared once its text is taken, and reloaded in the line while it still has the keyboard, since an active ImGui text field writes its own text back over its buffer. The app sends `spec.new` from the `ui` origin through the queue with the title and an applet id that `pane_new_spec_id()` derives from it (lower-case letters and digits, one `-` for every other run, `d-` before a leading digit, `-2` and on while the workspace's `applets/` folder has the id), and shows the page in build mode when it succeeds (`SpecPage::request_new()`, `take_new()`). The spec's own id, title and description are edited in Properties with nothing focused, through `spec.set_text`, and ⌘S saves it as `applets/<id>/`, which the landing lists at once under Projects.

The landing's Projects pane lists the open workspace's dashboards, its spec applets (`AppletLoader::in_workspace()`), with name, id and status, above the ways in, and its header counts them (`1 dashboard`) where it said `none yet`; Instruments lists the installed applets only. The two lists are one selection and one order for the keys and the command line: the dashboards come first, so ↑↓ walk from the last dashboard into the instruments, and the command line's top match can be a dashboard, which Enter opens. When a dashboard is saved for the first time, the app marks it ACTIVE while the page shows it and selects it (`note_spec_applets()` in `src/main.cpp`), so Home lands on its row in Projects; Instruments keeps the keys, which walk both lists.

## The capture tool

`caliper-builder-shot --workspace <dir> --applet-from <spec.toml> --files <file>... --out-dir <dir> [--size WxH] [--prefix NAME] [--suffix TEXT] [--text-scale S]` begins an empty spec with the applet of `--applet-from`, drops the files on the pane, and then builds the dashboard of `dashboard.spec.toml` through the pane's own input: every click, drag, key and typed text goes into ImGui's event queue as a platform backend feeds a person's input (⌘ as the Super key, which ImGui swaps on macOS), aimed at the pane's named rectangles of the last frame (`item_rect()`). It writes `<prefix>-<step><suffix>.png` at six steps: the empty spec with the two palettes and the two sources; the curves made from the metrics table as a series, renamed, titled and labelled, with Properties showing its bindings; the attention source with its three transforms and the heatmap dragged beside the curves, with Properties showing the transforms and their shapes; the table dropped under everything with its columns chosen and the root divider held at 0.65; the spec text open beside the finished dashboard; and the dashboard after two undos. It then redoes both and saves, and prints the texts, the undo steps and every command it sent. The stand-in page bar it draws is the app's: Home, the title, the pane's items, and ⌘K Command at the right end.

## The pane on the spec page

The page (`spec_page.h`) owns one `BuilderPane`, made when it binds the session, with a `SessionBuilderPort` whose `ready()` is false while the session's worker runs (`AppSession::worker_busy()`) and whose version is the journal's count (`AppSession::journaled()`), and with `spec_page_layout_rule()` as its layout rule. Each frame, on the frame thread:

1. The page bar: Home, the title, the command input while it is open, then `builder->page_bar(room)` (⌘B Build, and in build mode ⌘T Text, ⌘Z Undo, ⇧⌘Z Redo, ⌘S Save), where `room` is the width left before ⌘O Open in browser, ⌘L Chat and ⌘K Command at the right end; with less room, only Build. A spec applet's file drawn from outside the workspace has no Build. ⌘O Open in browser, in both modes, hands the saved dashboard's workspace id to the app (`SpecPage::set_open_in_browser()`), which starts Caliper API on loopback when it does not run, as the side chat does, and opens the viewer's link with the viewer token in the default browser (`src/main.cpp`, `src/host/open_url.h`); a hidden app writes the link to `CALIPER_TEST_OPEN_URL` instead, or refuses. The button is disabled with the reason for a dashboard never saved and for a file from outside the workspace, and the input driver clicks it as `page:browser` (`SpecPage::item_rect()`).
2. The keys: an open inline prompt takes Escape first, then the command input, then `builder->keys(escape)` (⌘B, and in build mode ⌘T, ⌘Z, ⇧⌘Z, ⌘S, ⌫, and Escape for a drag, the text editor or a source's focus), then the page's own order (a panel's focus, then Home). ⌘K stays the page's: with a panel focused it opens the inline prompt in that panel, in build mode too.
3. The strips: in build mode the side chat does not claim its viewport column; the pane draws it in the right column's lower part through `ChatPane::draw_in()` (`set_chat()`). The approval cards' strip, the jobs tray and the status bar keep their places, and the work area is what they leave.
4. The canvas: in build mode `builder->draw(area, *runtime, look, &focused)` in place of the runtime's draw and the page's focus, so one focused panel is shared: the pane takes the page's focus when the page changed it (the command input finding a panel, Escape) and gives its own back. The page's overlay still draws the focused panel's frame, the approval cards' REVIEW mark and the inline surface in each panel.

An approved agent edit, a ⌘K Accept and a hand edit go through the session's one history, so ⌘Z Undo in the page bar undoes whichever came last; a test checks it. The page leaves build mode when it is left, so a spec opened again from the landing shows out of build mode.

## Tests

- `caliper_builder_pane`, headless, over a real `Session` whose workspace imports a Parquet table and a safetensors tensor through `drop_files()` (a pickle and a missing file refused with their codes): the dashboard built gesture by gesture through the pane's functions, each exactly one undo step of commands from the `builder` origin under one group, the typed and dragged ones with one merge key, ending equal to `dashboard.spec.toml` and saved as an applet whose file equals it, with two undos and two redos between; a binding, a primitive, a drop, an id and a removal that do not fit refused with nothing sent; the reason a tensor of four dimensions and a table give for a heatmap, and the heatmap once two slices fit; drops on a slot, on edges, in a centre and at the whole layout's edges; the text strip both ways, with an invalid draft's diagnostic at its line, the other edits held, and the draft following an undo; the line the read-only strip follows after a new panel, a title above another panel and an undo; renames reaching every reference; a busy session; and the regions at 2560 by 1440 and 1440 by 900, at two text scales, with what gives way on a small window, the drop zones, the dividers and the strips of the dashboard, and the pure helpers; `layout_spec_page()` at both sizes for every combination of Build, Text and Chat, the chat's column as wide in build mode as outside it; an agent's edit approved through the queue, a ⌘K Accept through the inline surface with the scripted agent, and a hand edit undone in turn through the pane's Undo; and a new spec's id from its title.
- `caliper_builder_pane_gpu`: `caliper-builder-shot` as a process over a new workspace; the spec it ends with and the applet it saves parse equal to the fixture, as the other three ways of building it are checked; two undos take two steps; every command and every journal line is the `builder` origin's; the six captures have the size asked for.
- `caliper_builder_app`: `caliper --hidden` on an empty workspace with `CALIPER_TEST_SPEC_UI`: the two files dropped on the landing, which makes a new spec in build mode with both sources; the spec named; the dashboard built through the pane; the text strip and the side chat opened in build mode; ⌘S; Home; and the dashboard opened again from the landing's command line. The saved spec equals `dashboard.spec.toml`, the journal holds `spec.new` and `data.import` from `ui`, the build from `builder` and `applet.open` after `applet.save`, and the six captures have the window's size. A second case checks that the landing's command line is empty when the person comes back Home after New dashboard took its text, by F3 and by `new <title>` and Enter (the input driver's `command-line` step). A third makes a dashboard with F3, saves it and goes Home: the landing selects it and Projects lists it (the `landing selected` and `landing projects` steps). A fourth clicks ⌘O Open in browser in an app started without `--api`: the link reaches `CALIPER_TEST_OPEN_URL`, opens the spec page and reads the dashboard's spec over loopback with its token, which `/v0/info` and `/v0/applets` refuse with 403, and is neither the person's session token nor `api.json`'s; without the hook the hidden app refuses and starts no API.
- `caliper_builder_pane` also checks the axes of a slice and a reduce with their ranges, the names of a source that carries them through a slice and a reduce, a slice's index kept in its range with the note, and the empty canvas's line; `caliper_catalog_providers_tests` checks a live view's dim names from its payload axes.
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
- **A drop on the landing opens build mode,** on the open spec or on a new Untitled one, because the files land in Sources, which only the spec page has; an applet's page keeps its windows.
- **A drop on the live page starts a new dashboard** beside the stream's live sources, since the person is watching a run and the files are new data for it; an open spec with unsaved changes takes the files instead, so the drop never replaces unsaved work. A saved open spec is replaced (it stays in the landing's list, and ⌘Z brings it back).
- **"Save changes to <name>?" before the person replaces or closes unsaved work.** F3 New dashboard, opening another dashboard from the landing and closing the window ask, with ⏎ Save, Don't Save and esc Cancel (`src/host/builder/save_prompt.h`), because the open spec lives only in the session until it is saved. Only these `ui` paths ask: an agent's replacement is reviewed on an approval card and must not meet a second question, and `app.launch` and the API's launch are an agent's or a program's requests, whose `applet.open` the app sends from the caller's own origin, so that an agent's replacement of unsaved work waits on its card (command-layer.md, the review of replacing an open spec). Opening the open spec's own dashboard shows it with its changes instead of reading its file again, and a drop never replaces unsaved work, so neither asks. A hidden app's SIGTERM, `CALIPER_EXIT_AFTER` and the input driver's `quit` close without asking, so that quitting a test never waits. While the session's worker runs a command the open spec cannot be read; the question then waits for it (showing so, with Cancel) and never takes "cannot tell" for "nothing to save", which once let ⌘Q quit over unsaved work while an agent's query ran.
- **Dropped files are the person's `ui` origin;** the pane's own gestures are `builder`.
- **In build mode the chat keeps its width** and Properties takes 45 % of the column above it, the chat layout's shares, so toggling Build beside an open chat does not move it.
- **The page keeps the focus and ⌘K;** in build mode the pane takes the page's focus when the page changed it and gives its own back after each draw, so the bronze frame, ⌘K and Escape mean the same in both modes.
- **The person's dashboards are listed under Projects, not Instruments.** The landing screen's own description (`src/host/ui/landing_screen.h`, and the commit that made it) defines Instruments as the table of installed applets and Projects as the person's way into their own work, where F3 New dashboard makes a dashboard; a dashboard saved in the workspace is the person's work, not an installed applet, so Projects saying `none yet` while listing it under Instruments was the wrong half. Spec applets found outside the workspace are installed ones and stay in Instruments. The two tables keep one selection and one order, so the command line and ↑↓ work as before.
- **A slice's index outside its axis is set to the nearer end, and said,** rather than refused, because the field applies as it is typed: refusing would leave the last valid digit in the spec while the field shows another, and the note says what was kept.
- **⌘O Open in browser starts the API as the side chat does and keeps it until the app quits,** since a page open in a browser keeps asking for its panels' data and nothing tells the app when it closes; the side chat's end no longer stops an API whose link was given out. The link carries the viewer token only, checked before it is opened. Everything that can refuse the link (no browser viewer in the build, an id that the workspace's id rule refuses, so never `.` or `..`) is asked before the API starts, and an API started for one request stops again when that request hands no link over. The opener gets only the standard descriptors, on `/dev/null`.
- **An agent never ends the person's running applet.** An agent's `spec.new`, `spec.open` or `applet.open` while a compiled applet's page shows opens the spec in the session and leaves the page (`SpecPage::opened_by_agent()`, from `AppSession::agent_specs_opened()`); Activity says `agent opened` with how to get to it, the applet page's status bar names it, and F6 Dashboard on the landing shows it. From the landing, the live page or a spec page the app switches as before, and the person's own opens are unchanged.
- **F6 Dashboard on the landing** is the way back to the builder session's open spec, as F5 Live page is to the attached session: a dashboard left with Home, saved or not, has no other way back when it was never saved.
- **⌘O says the browser shows the saved file** in its tooltip while the dashboard has unsaved changes, and in the status bar once its link is out, rather than saving first, since ⌘S is the person's choice.

## Limits

- A split of more than two children shares its space equally in the format, so its dividers do not move.
- A tab is not dragged by its tab; a panel in tabs moves through the spec text.
- The transforms of a source are added, changed and removed, not reordered.
- The pane offers no Selection tab of its own; the right column's tab strip has room for the live page's Selection when the page gives it.
- The binding choices list the spec's own sources; a catalog source joins the choices once it is in the spec, by a drop or by Add.
