# UI Principles

These rules apply to the host and to every instrument. They describe structure, not visual style. The visual direction is a dense technical workstation in the Sternbild palette, with bronze as the single accent, square corners and 1 px rules. Researchers can have denser panes than first-time users, but the rules hold at any density.

## Text

- **No taglines or explanatory sentences in the UI.** Arrangement, icons and labels carry the meaning. Thus the UI has no pane subtitles, sentences beside controls, help lines or view footers.
- **Labels are nouns or short noun phrases.** Examples are "Layers", "Learning rate", "step 1200" and "Validation loss".
- **Legends are a symbol plus a noun.** For example, "■ train  ◆ test".
- **Things have names, never pointers.** Do not use "this item", "current", "here" or "selected" as a label. The open item has the same name everywhere: the header, the list, the point in a view, the legend and the table titles.
- **Tooltips show names and values only.**
- **Every mark in a view has a label or a key entry.** A line, ring or arrow with no label or key entry is unreadable to anyone who did not build it.

## Explanation

- **One on-demand place.** Explanation lives in a tour or behind an info icon beside a title, never on the surface.
- **A tour is data.** Each step names the region it describes, and the overlay outlines that region. Step text is a title and at most two plain sentences. The content lives in a file beside the instrument, so a developer can revise it without a code change and reuse it elsewhere.

## Layout

- **The subject of a view gets most of the window.** The thing under study takes the largest region. Controls and secondary panels sit beside it.
- **Panes are shares of the window, not fixed pixel sizes.** Each pane has a minimum that grows with the text scale and has no maximum. A larger window thus gives every pane more room.
- **A view's content fills its view.** A 3-D or 2-D view fits its content when it opens, when the window resizes and after the data changes. The fit does not jump.
- **Different questions get different tabs** and not more panes.
- **The host's chrome takes its own space.** The page bar, the status bar and the jobs tray are strips at the window edges. The panes fill only the space the strips leave. Nothing the host shows covers a pane unless the user opens it, as with a popup. A strip that comes and goes waits a moment before it takes its space and before it gives the space back, so the panes do not jump.

## Behavior

- **Open quietly.** Nothing plays or animates until the user picks something to look at.
- **One place to change each thing.** Every setting has exactly one control. Other panes display the setting and never change it.
- **Selection is linked.** When the user selects an item in any view, every other view marks the same item.

## Color and Size

- **One color key across the whole app.** The same class or entity has the same color in every pane.
- **Annotation marks have their own color,** separate from the data classes.
- **Text scale is a setting.** It applies to fonts, metrics and pane minimums together.

## Layout Checks

- Lay out panes as plain rectangles, computed without ImGui, so tests can check the sizes without a window.
- Check every layout change with screenshots at two window sizes before you accept it, for example 2560 × 1440 and 1440 × 900 points.
- Treat ImGui runtime errors as failures in development builds.
