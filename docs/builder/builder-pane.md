# Builder Pane

The builder pane is where you build a spec by hand on the app's spec page. You pick a source and a primitive, and you get a panel. Then you arrange the panels, edit their properties and see the spec text follow. Every change is a command of the builder session from the `builder` origin, so it is journaled. Each gesture is one undo step.

The pane is not on the live page, and the browser viewer shows a spec without it.

## Pane Access

There are three ways to the pane:

- **A new dashboard.** On the landing screen, F3 New dashboard makes a new spec and opens it in build mode. Type `new <title>` in the command line and press Enter to set the title. F3 also takes the command line's text as the title. A dashboard without a title is called Untitled. The applet id comes from the title: its letters and digits in lower case, with one `-` for every other run of characters. For example, "Run 42" gives `run-42`.
- **Files dropped on the window.** A drop imports the files into the workspace. On the spec page, the pane opens in build mode with the new sources selected. On the landing screen, the open spec opens in build mode, or a new Untitled dashboard opens when no spec is open. On the live page, a new Untitled dashboard opens, and Sources lists the files beside the live sources. If the open spec has unsaved changes, the files go into it, so nothing unsaved is replaced. An applet's page says in Activity that a spec must be open.
- **An open spec.** ⌘B Build in the page bar (Ctrl+B Build elsewhere) turns build mode on and off. A spec applet found outside the workspace is drawn from its file and has no Build.

⌘S Save saves a dashboard as a spec applet in `applets/<id>/`. The dashboard then appears under Projects on the landing screen. ⌘O Open in browser opens a saved dashboard in a browser ([Browser Preview](#browser-preview)).

The open spec stays open, with its unsaved changes, when you go Home. If the spec has unsaved changes, the app asks "Save changes to <name>?" before F3 New dashboard, before you open another dashboard from the landing screen, and before you close the window or quit with ⌘Q. The name is the spec's title. ⏎ Save (Enter) saves and goes on, Don't Save goes on without a save, and esc Cancel (Esc) leaves everything as it was.

While a long command runs on the app's worker, such as an agent's query or import, the app cannot read the open spec. The action waits and shows "Waiting for the running command" with esc Cancel. Then the app asks if the spec has unsaved changes. A drop, the launches of Caliper API and `app.launch`, and the open spec's own dashboard on the landing screen do not ask. An agent that replaces the open spec is reviewed on an approval card instead. After any replacement, ⌘Z brings the previous spec back with its unsaved changes.

## Layout Regions

Build mode keeps the page bar, the status bar, the jobs tray and the approval cards where they are. It divides the rest of the window:

| Region | Place | Size |
|---|---|---|
| Sources | Left column, upper part | Above Primitives |
| Primitives | Left column, lower part | Its seven rows |
| Canvas | Centre | The rest, always the largest region |
| Spec text | Under the canvas, while ⌘T Text is on | 30 % of the centre's height, at least 9 lines |
| Properties | Right column, upper part | The whole column, or 45 % of it above the side chat |
| Side chat | Right column, lower part, while ⌘L Chat is on | 55 % of the column |

The left column is 17 % of the width (18 % with the chat open). The right column is 21 % (28 % with the chat open). On a small window, the text strip goes first, then the right column, then the left. The canvas keeps its minimum.

The page bar holds, after Home and the spec's title, ⌘B Build. In build mode it also holds ⌘T Text, ⌘Z Undo, ⇧⌘Z Redo and ⌘S Save. ⌘O Open in browser, ⌘L Chat and ⌘K Command are at its right end. When the bar is too narrow, only Build stays, and the other buttons keep their keys. The code of the last refused or failed change follows the buttons, with its reason in a tooltip.

## Sources

Sources lists the open spec's own sources under Spec. Then it lists the workspace's catalog, grouped as Tables, Tensors, Live, Feeds and Metrics. Each row has the source's name and, in mono, its shape, such as `4,000 × 3` for a table or `f32 4×8×48×48` for a tensor. A spec source's shape is the one after its transforms. The triangle at the left of a table shows its columns and their types. The filter line narrows the list by name, id or column.

Metrics are grouped by run, newest first. Each run is one line with its name, its number of metrics and its id. A click opens the run to its metrics, each named by its tag. Sources lists the newest five runs, and Show all lists every run. The filter searches every run by name, experiment or metric tag, and opens the runs it finds.

A click selects a row or a column and shows it in Properties. A drag of a row or a column onto the canvas makes a panel there. The primitive suits the data:

- **Series:** a column, or a table with numeric columns.
- **Heatmap:** a rank-2 tensor.
- **Points:** `[n, 2]` and `[n, 3]`.
- **Image:** `[H, W, 3]` and `[H, W, 4]`.
- **Table:** other tables.
- **Histogram:** everything else.

A drop on a panel binds the source to one of that panel's slots instead. ⏎ Add in the header, Enter in the list or a double click makes the same panel without a drag.

Files that an import refused are listed under Refused, with the file's name and the loader's code, such as `pickle_refused` or `no_reader`. The reason is in the row's tooltip. The list stays until the next drop.

## Primitives

The primitives are Series, Heatmap, Points, Histogram, Table, Image and Text. A click adds a panel of that primitive, bound to the selected source. A series takes a table's first column as x and its other numeric columns as lines. A primitive that the selection does not fit is faint, with the reason in its tooltip, and a click on it changes nothing. For a tensor with too many dimensions, such as a four-dimensional tensor for a heatmap, the reason says to slice or reduce it under Transforms in Properties. Every primitive needs a fitting selection, except Text, which starts with the text "Text". A drag of a primitive onto the canvas places the new panel where you drop it.

## Canvas

The canvas draws the spec as the page does outside build mode. While the spec has no panel, the canvas shows "No panels" and "Drop a data file, or pick a source in Sources".

| Gesture | What it does |
|---|---|
| A click in a panel | Focuses it with a bronze frame and shows its properties in Properties. ⌘K then opens the inline prompt in that panel. |
| Dragging a panel's title strip | Moves the panel. A drop on another panel's edge puts it beside, above or below that panel. A drop on its centre puts the two in tabs. A drop on the band along the canvas's border puts it at that edge of the whole layout. |
| Dragging the rule between two panels | Changes their split's share, in hundredths, shown beside the pointer. A split of more than two panels shares its space equally and does not move. |
| ⌫ | Removes the focused panel. |

You can remove a panel in build mode and on the dashboard page in three ways: the × at the right end of the panel's title strip (at the top right corner for a panel in tabs), Remove panel in the right-click menu, or ⌫ or Delete on the focused panel. The keys do not work while a text field, the command input or the inline prompt has the keyboard. Each way is one undo step, and ⌘Z brings the panel back where it was. No way is offered while the inline prompt is open or a ⌘K preview is shown. Accept or Reject the preview first. A dashboard drawn from a file outside the workspace offers none of them.

During a drag, the rectangle where the item would land has a bronze outline. A drop that does not fit has a grey outline with the reason in a tooltip, and nothing changes on release. A source held over a panel shows the panel's slots as rows, and the slots it fits are in parchment.

## Properties

With a panel focused, Properties shows:

- **Panel:** its id, its primitive and its title. A new id renames the panel everywhere in the spec. A new primitive keeps the bindings and options that it takes.
- **Bindings:** one row for each slot of the primitive, with required slots marked by an asterisk. A series has one row per line and one more to add a line. A click opens the choice of the spec's sources and their columns. The ones that do not fit are faint, with the reason in their tooltip. × removes a binding, except from a required slot.
- **Options:** exactly the options of the primitive. These are labels, ranges, the log flags, the colormap as four swatches, bins, a table's columns as one flag each, and a text panel's text. A range is empty until set, and the data's extent shows dimmed as a placeholder.

With a source selected, Properties shows its id, its catalog id and its shape. For a tensor it also shows its transforms, each with its fields and the shape after it. Slice, Reduce and Normalise add a transform, and × removes one. A catalog source that the spec does not read yet is added by its first transform or by the first panel made from it. For a source of the open spec, ⌘K beside what it reads and beside its transforms opens the widget agent's prompt for that property ([Widget agent](../agents/widget-agent.md)).

A slice's or a reduce's axis is a choice of every axis of the step's input, each with its index range, such as `axis 1 · 0–3` for an axis of 4. Where the source carries names, the axis also has its name, as in `axis 0 layer · 0–3`. A tensor from a safetensors file has no names. While the shape is not known, as for a live view before its first snapshot, the axis is a number. A slice takes one axis. A reduce takes a list of axes, and a click on an axis adds it or takes it out. The last axis stays, because a reduce over no axis is not a reduce.

An index outside a slice's axis range is set to the nearer end, such as `3` for a `7` typed on an axis of 4. The field then says `7 is outside 0–3` until the next change of the transforms. A negative index counts from the end, as the spec format says.

With nothing focused, Properties shows the spec's id, title and description, its numbers of panels and sources, and its file. Text and numbers apply as you type. Ids apply on Enter or when you leave the field.

## Spec Text

⌘T Text opens a strip under the canvas with the open spec's text, in the canonical TOML that the app saves. ⌘E Edit turns the strip into an editor, and Esc Done turns it back. An edit applies as soon as the text is a valid spec, and the panels follow. While the text is not valid, each error's line is marked with its code and the message is in the line's tooltip. The panels keep the last valid spec, and other changes wait until you fix the text or leave the editor. A change made on the canvas or in Properties shows in the text at once, and the strip scrolls to the changed line.

## Browser Preview

⌘O Open in browser opens the saved dashboard that the page shows in your default browser. The browser uses the viewer of [Caliper API](../agents/http-api.md#get-v0spec), which draws the dashboard with the layout and the look of the spec page. The button works in build mode and outside it.

The browser shows the dashboard as it was last saved, so save first to show a change. While the dashboard has unsaved changes, the button's tooltip says "the browser shows the saved dashboard". The button is disabled, with the reason in its tooltip, for a dashboard that was never saved and for a spec applet found outside the workspace.

If you started the app without `--api`, the button starts Caliper API as the side chat does. It uses a free port of 127.0.0.1 only, with new tokens, and the status bar shows `api 127.0.0.1:<port>`. The API stays on until the app quits, because the page in the browser keeps reading the dashboard's data. [Caliper API](../agents/http-api.md#app-startup) describes what the link carries and who can use it.

## Keys

| Key | Action |
|---|---|
| ⌘B | Build mode on or off |
| ⌘T | The spec text |
| ⌘E, Esc | Edit the text, and back |
| ⌘Z, ⇧⌘Z | Undo and redo, unless a text field has the keyboard; on the dashboard page too |
| ⌘S | Save |
| ⌘O | Open the saved dashboard in a browser |
| ⏎ | Add the selection, in Sources |
| ⌫, Delete | Remove the focused panel; on the dashboard page too |
| ⌘K | The inline prompt in the focused panel ([Widget agent](../agents/widget-agent.md)) |
| ⌘L | The side chat ([Side chat](../agents/side-chat.md)) |
| Esc | Cancels a drag, leaves the editor, gives up a source's focus, then a panel's, then goes Home |

Off macOS, each ⌘ is Ctrl.

## Undo and Journal

The pane changes the workspace only through commands of the builder session, from the `builder` origin. The import of files you drop is from the `ui` origin. A panel removed on the dashboard page is from the `ui` origin too. The commands of one gesture share a group, so a gesture is one undo step and one group of journal lines.

Undo and Redo use the workspace's one history, so they take back whatever came last. This can be a change made by hand, an agent's change approved on the approval cards, or an accepted ⌘K proposal. The pane refuses a change that the spec would refuse before it sends anything, and it gives the validator's reason.

Renaming a panel or a source, and editing the spec's id, title or description, send `spec.set_text` of the whole spec. A panel placed on another's left or at an edge of the whole layout is placed with `layout.set`.
