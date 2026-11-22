# UI Principles

These rules apply to the host and to every applet. They come from building and
showing a domain applet to people outside the project, and they describe
structure, not visual style. The visual direction stays as set: a dense
technical workstation in the Sternbild palette, with bronze as the single
accent, square corners and 1 px rules. Researchers can have denser panes than
first-time users; the rules below hold at any density.

## Text

- **No taglines or explanatory sentences in the UI.** Arrangement, icons and
  labels carry the meaning. That rules out pane subtitles, sentences beside
  controls, help lines and view footers.
- **Labels are nouns or short noun phrases.** "Layers", "Learning rate",
  "step 1200", "Validation loss".
- **Legends are a symbol plus a noun.** "■ train  ◆ test".
- **Things are called by their names, never by pointers.** No "this item",
  "current", "here" or "selected" as a label. The open item carries the same
  name everywhere it appears: the header, the list, the point in a view, the
  legend and table titles.
- **Tooltips show names and values only.**
- **Every mark in a view is labelled or keyed.** A line, ring or arrow with no
  label or key entry is unreadable to anyone who did not build it.

## Explanation

- **One on-demand place.** Explanation lives in a tour or behind an info icon
  beside a title, never on the surface.
- **A tour is data.** Each step names the region it is about, and the overlay
  outlines that region. Step text is a title and at most two plain sentences.
  The content lives in a file beside the applet, so it can be revised without
  touching code and reused elsewhere.

## Layout

- **The subject of a view gets most of the window.** The thing being studied
  takes the largest region; controls and secondary panels sit beside it.
- **Panes are shares of the window, not fixed pixel sizes.** Each pane has a
  minimum that grows with the text scale, and no maximum, so a larger window
  gives every pane more room.
- **A view's content fills its view.** A 3-D or 2-D view fits its content on
  opening, on resize and after the data change, without jumps.
- **Different questions get different tabs** rather than more panes.
- **The host's chrome takes its own space.** The page bar, the status bar and
  the jobs tray are strips at the window's edges, and the panes fill only what
  they leave, so nothing the host shows covers a pane unless the user opens it,
  as with a popup. A strip that comes and goes waits a moment before it takes
  its space and before it gives it back, so the panes do not jump.

## Behaviour

- **Open quietly.** Nothing plays or animates until the user picks something to
  look at.
- **One place to change each thing.** Every setting has exactly one control.
  Other panes display it and never change it.
- **Selection is linked.** Selecting an item in any view marks the same item
  everywhere else.

## Colour and size

- **One colour key across the whole app.** The same class or entity has the
  same colour in every pane.
- **Annotation marks get their own colour,** separate from the data classes.
- **Text scale is a setting,** applied to fonts, metrics and pane minimums
  together.

## Checking a layout

- Lay panes out as plain rectangles, computed without ImGui, so tests can check
  the sizes without a window.
- Check every layout change with screenshots at two window sizes, for example
  2560 × 1440 and 1440 × 900 points, before accepting it.
- Treat ImGui's runtime errors as failures in development builds.
