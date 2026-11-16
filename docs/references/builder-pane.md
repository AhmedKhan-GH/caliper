# Builder pane

The builder pane is where you build a spec by hand on the app's spec page: you pick a source, pick a primitive, get a
panel, arrange the panels, edit their properties and see the spec text follow. Every change is a command of the
builder session from the `builder` origin, so it is journaled and can be undone, and each gesture is one undo step.
This page gives what the pane shows, what each of its controls does, and its keys.

!!! note "Status"
    The pane is on the app's spec page (`src/host/builder/builder_pane.h`, laid out with
    `src/host/builder/spec_page_layout.h`). It is tested headless over a builder session and, from an empty workspace
    to a saved dashboard opened again, in the hidden app. It is not on the live page, and the browser viewer shows a
    spec without it.

## Starting a dashboard

There are three ways to the pane:

- **A new dashboard.** On the landing screen, F3 New dashboard, the Projects way in, makes a new spec and opens it in
  build mode. Type `new <title>` in the command line and press Enter to give it a title; F3 takes the command line's
  text as the title too, and a dashboard without one is called Untitled. The command line is empty again afterwards.
  The spec's applet id is made from the title: its letters and digits in lower case, with one `-` for every other run
  of characters, such as `run-42` for "Run 42".
- **Files dropped on the window.** Dropping files from the Finder or a file manager imports them into the workspace.
  On the spec page the pane opens in build mode with the new sources selected; on the landing screen the open spec's
  page opens in build mode first, or a new Untitled dashboard when no spec is open. Other pages keep their windows and
  say in Activity that a spec must be open.
- **An open spec.** ⌘B Build in the page bar (Ctrl+B Build elsewhere) turns build mode on and off for the spec the
  page shows. A spec applet found outside the workspace is drawn from its file and has no Build.

A dashboard is saved with ⌘S Save as a spec applet of the workspace, `applets/<id>/`, and appears at once under
Projects on the landing screen, which selects it when you go Home from it. A saved dashboard opens in a browser with
⌘O Open in browser in the page bar ([Open in browser](#open-in-browser)).

## Where things are

Build mode keeps the page bar, the status bar, the jobs tray and the approval cards where they are, and divides the
rest of the window:

| Region | Place | Size |
|---|---|---|
| Sources | Left column, upper part | Above Primitives |
| Primitives | Left column, lower part | Its seven rows |
| Canvas | Centre | The rest; always the largest region |
| Spec text | Under the canvas, while ⌘T Text is on | 30 % of the centre's height, at least 9 lines |
| Properties | Right column, upper part | The whole column, or 45 % of it above the side chat |
| Side chat | Right column, lower part, while ⌘L Chat is on | 55 % of the column |

The left column is 17 % of the width (18 % with the chat open) and the right column 21 % (28 % with the chat open, as
wide as the chat is outside build mode); each has a minimum that grows with the text scale. On a small window the text
strip goes first, then the right column, then the left, and the canvas keeps its minimum.

The page bar holds, after Home and the spec's title, ⌘B Build and, in build mode, ⌘T Text, ⌘Z Undo, ⇧⌘Z Redo and ⌘S
Save, with ⌘O Open in browser, ⌘L Chat and ⌘K Command at its right end. When the bar is too narrow for them, only Build
stays; the others keep their keys. The code of the last refused or failed change follows the buttons, with its reason
in a tooltip.

## Sources

Sources lists the open spec's own sources under Spec, and then the workspace's catalog grouped as Tables, Tensors,
Live, Feeds and Metrics. Each row has the source's name and, in mono, its shape: rows and columns for a table, such as
`4,000 × 3`, or the dtype and dimensions for a tensor, such as `f32 4×8×48×48`. A spec source's shape is the one after
its transforms. A table opens with the triangle at its left to show its columns and their types. The filter line
narrows the list by name, id or column.

A click selects a row or a column and shows it in Properties. Dragging a row or a column onto the canvas makes a panel
there, of the primitive that suits the data: a series for a column or a table with numeric columns, a heatmap for a
rank-2 tensor, points for `[n, 2]` and `[n, 3]`, an image for `[H, W, 3]` and `[H, W, 4]`, a table for other tables,
a histogram otherwise. Dropping it on a panel binds it to one of that panel's slots instead. ⏎ Add in the header, Enter
in the list or a double click makes the same panel without dragging.

Files that an import refused are listed under Refused, with the file's name and the loader's code, such as
`pickle_refused` or `no_reader`; the reason is in the row's tooltip. The list stays until the next drop.

## Primitives

Series, Heatmap, Points, Histogram, Table, Image and Text. A click adds a panel of that primitive bound to the selected
source: a series takes a table's first column as x and its other numeric columns as lines. A primitive that the
selection does not fit is faint, with the reason in its tooltip, and a click on it changes nothing. The reason says what
the source is and what the primitive takes; for a tensor with more dimensions than the primitive draws, such as a
four-dimensional attention tensor for a heatmap, it says to slice or reduce it under Transforms in Properties. A panel
always has its required slots, so a primitive needs a selection that fits, except Text, which starts with the text
"Text".
Dragging a primitive onto the canvas places the new panel where it is dropped.

## The canvas

The canvas draws the spec as the page does outside build mode. While the spec has no panel, it says so, and how to
start: "No panels", and under it "Drop a data file, or pick a source in Sources".

| Gesture | What it does |
|---|---|
| A click in a panel | Focuses it: a bronze frame, and its properties in Properties. ⌘K then opens the inline prompt in that panel, as outside build mode. |
| Dragging a panel's title strip | Moves the panel: onto another panel's left, right, top or bottom edge, beside or above or below it; onto its centre, into tabs with it; onto the band along the canvas's border, at that edge of the whole layout. |
| Dragging the rule between two panels | Changes their split's share, in hundredths, shown beside the pointer while it moves. A split of more than two panels shares its space equally and does not move. |
| ⌫ | Removes the focused panel. |

While something is dragged, the rectangle where it would land is outlined in bronze; a drop that would not fit is
outlined in grey with the reason in a tooltip, and nothing changes when it is released. A source held over a panel
shows the panel's slots as rows in its centre, the ones it fits in parchment.

## Properties

With a panel focused, Properties shows:

- **Panel**: its id, its primitive and its title. A new id renames the panel everywhere in the spec. A new primitive
  keeps the bindings and options the new one takes.
- **Bindings**: one row for each slot of the primitive, required slots marked with an asterisk, and for a series one row
  per line and one more to add a line. A click opens the choice of the spec's sources and their columns; the ones that do
  not fit are faint, with the reason in their tooltip. × removes a binding, except from a required slot.
- **Options**: exactly the options of the primitive: labels, ranges (empty until set; the data's extent shows dimmed
  as a placeholder), the log flags, the colormap as four swatches, bins, a table's columns as one flag each, and a text
  panel's text.

With a source selected: its id, its catalog id, its shape, and for a tensor its transforms, each with its fields and
the shape after it. Slice, Reduce and Normalise add one; × removes one. A catalog source that the spec does not read
yet takes its id here before it is added; its first transform, or the first panel made from it, adds it. For a source
of the open spec, ⌘K beside what it reads and beside its transforms opens the widget agent's prompt for that property
([Widget agent](widget-agent.md)).

A slice's or a reduce's axis is a choice of every axis of the step's input, each with its index range, such as
`axis 1 · 0–3` for an axis of 4, and with its name where the source carries names: a live view's dims are named by the
stream, as in `axis 0 layer · 0–3`, and a step that takes an axis away takes its name with it; a tensor from a
safetensors file has none. While the shape is not known, as for a live view before its first snapshot, the axis is a
number. A slice's index shows its axis's range at the right of the field. An index outside it is set to the nearer
end, `3` for a `7` typed on an axis of 4, and the field says so in place of the range, `7 is outside 0–3`, until the
next change of the transforms. A negative index counts from the end, as the spec format says.

With nothing focused (Esc gives a source's focus up): the spec's id, title and description, its numbers of panels and
sources, and its file.

Text and numbers apply as you type; ids apply on Enter or when the field is left.

## The spec text

⌘T Text opens a strip under the canvas with the open spec's text, in the canonical TOML that is saved. ⌘E Edit turns it
into an editor, and Esc Done back. An edit applies as soon as the text is a valid spec, and the panels follow. While
it is not, each error's line is marked with its code, the message is in the line's tooltip, the panels keep the last
valid spec, and other changes wait until the text is fixed or left. A change made on the canvas or in Properties
shows in the text at once, and the strip scrolls to the last line that changed when that line is out of view, so a
new panel's block at the end of the text, or a title changed in the middle of it, comes into view.

## Open in browser

⌘O Open in browser in the page bar opens the saved dashboard that the page shows in your default browser, at the
browser viewer of [Caliper API](api.md#get-v0spec), which draws it with the layout and the look of the spec page. It
works in build mode and outside it. The browser shows the dashboard as it was last saved, so save first to show a
change; the button is disabled, with the reason in its tooltip, for a dashboard that was never saved and for a spec
applet found outside the workspace.

When the app was started without `--api`, the button starts Caliper API as the side chat does: on a free port of
127.0.0.1 only, with new tokens, said in Activity and shown in the status bar as `api 127.0.0.1:<port>`. It stays on
until the app quits, since the page in the browser keeps reading the dashboard's data. What the link carries and who
can use it is in [Caliper API](api.md#started-by-the-app).

## Keys

| Key | Action |
|---|---|
| ⌘B | Build mode on or off |
| ⌘T | The spec text |
| ⌘E, Esc | Edit the text, and back |
| ⌘Z, ⇧⌘Z | Undo and redo, unless a text field has the keyboard |
| ⌘S | Save |
| ⌘O | Open the saved dashboard in a browser |
| ⏎ | Add the selection, in Sources |
| ⌫ | Remove the focused panel |
| ⌘K | The inline prompt in the focused panel ([Widget agent](widget-agent.md)) |
| ⌘L | The side chat ([Side chat](side-chat.md)) |
| Esc | Cancels a drag, leaves the editor, gives up a source's focus, then a panel's, then goes Home |

Off macOS each ⌘ is Ctrl.

## Undo and the journal

The pane changes the workspace only through commands of the builder session, from the `builder` origin; the import of
files you drop is from the `ui` origin. The commands of one gesture share a group, and a value that is typed or a rule
that is dragged sends each change with one merge key, so a gesture is one undo step and one group of journal lines.
Undo and Redo use the workspace's one history, so they take back whatever came last: a change made by hand, an agent's
change approved on the approval cards, or a ⌘K proposal accepted. A change that the spec would refuse is refused
before anything is sent, with the validator's reason.

Renaming a panel or a source, and editing the spec's id, title or description, are sent as `spec.set_text` of the whole
spec; a panel placed on another's left or at an edge of the whole layout is placed with `layout.set`.
