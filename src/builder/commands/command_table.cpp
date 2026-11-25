#include <caliper_builder/commands/command_table.h>
#include <caliper_builder/commands/registry.h>

#include <algorithm>
#include <exception>
#include <map>
#include <utility>

namespace caliper_builder {

namespace {

using nlohmann::json;

// Shared pieces of the schemas. A schema names one with {"$use": "<name>"};
// the other keys of that object (a description) are laid over the piece.
const std::pair<const char*, const char*> kFragments[] = {
    {"id", R"json({"type": "string", "pattern": "^[a-z][a-z0-9_-]{0,63}$", "minLength": 1, "maxLength": 64})json"},

    {"catalog_id", R"json({"type": "string", "pattern": "^(table|tensor|live|feed|metric|run):.+$", "minLength": 3,
                           "maxLength": 263})json"},

    {"diagnostic", R"json({
      "type": "object", "additionalProperties": false, "required": ["code", "message", "where"],
      "properties": {
        "code": {"type": "string", "description": "A stable code, such as spec.unknown_primitive or spec.not_portable."},
        "message": {"type": "string", "description": "One plain sentence that says what is wrong."},
        "where": {"type": "string", "description": "Where in the spec: a path such as panel[2].bind.y, or a line and column such as 12:7; empty when it is not located."}}})json"},

    {"column", R"json({
      "type": "object", "additionalProperties": false, "required": ["name", "type"],
      "properties": {
        "name": {"type": "string", "description": "The column's name."},
        "type": {"type": "string", "description": "The column's DuckDB type, such as BIGINT, DOUBLE or VARCHAR."}}})json"},

    {"origin", R"json({
      "type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "loader": {"type": "string", "description": "The id of the loader that made the source."},
        "file": {"type": "string", "description": "The file or glob the loader read, relative to the workspace."},
        "output": {"type": "string", "description": "The name of the loader output that made the source."},
        "tensor": {"type": "string", "description": "The tensor's name inside the file."},
        "database": {"type": "string", "description": "The database that holds the table, workspace for the workspace's own."},
        "schema": {"type": "string", "description": "The table's DuckDB schema."},
        "table": {"type": "string", "description": "The table's or view's name in DuckDB."},
        "session": {"type": "string", "description": "The live session that publishes the view."},
        "view": {"type": "string", "description": "The live view's name, as the training script published it with Caliper LIB."},
        "view_type": {"type": "string", "description": "The live view's type: heatmap, points, histogram or series."},
        "service": {"type": "string", "description": "The app service that holds the data: feed for live telemetry, metrics for logged metrics."},
        "channel": {"type": "string", "description": "The feed channel's id, such as sys.cpu.util."},
        "units": {"type": "string", "description": "The feed channel's units, such as % or W."},
        "experiment": {"type": "string", "description": "The metrics experiment."},
        "run": {"type": "integer", "description": "The metrics run's id."},
        "run_name": {"type": "string", "description": "The metrics run's name."},
        "tag": {"type": "string", "description": "The metrics tag, such as train/loss."}}})json"},

    {"catalog_entry", R"json({
      "type": "object", "additionalProperties": false,
      "required": ["id", "kind", "name", "title", "origin", "columns", "rows", "dtype", "shape", "rate_hz", "bytes",
                   "wire", "warnings", "updated_at"],
      "properties": {
        "id": {"type": "string", "description": "The source id, <kind>:<name>; a spec source names it in its catalog field."},
        "kind": {"type": "string", "enum": ["table", "tensor", "live", "feed", "metric", "run"],
                 "description": "table: a DuckDB table or view, such as a loader made; tensor: a tensor read from a file, an image or an audio clip; live: a live view of the attached live session; feed: a channel of live telemetry; metric: a run and tag of logged metrics; run: a series of a managed run of the model runner, run:<run id>/<view>."},
        "name": {"type": "string", "description": "The name part of the id. For a table it is the table's name in SQL."},
        "title": {"type": "string", "description": "A name for a person."},
        "origin": {"$use": "origin", "description": "Where the source comes from: a loader and its file, a database table, a live view, a feed channel, or a metrics run and tag."},
        "columns": {"type": ["array", "null"], "items": {"$use": "column"},
                    "description": "For a table, feed or metric, its columns with their DuckDB types; null for a tensor or live view."},
        "rows": {"type": ["integer", "null"], "description": "For a table, feed or metric, its number of rows, or null when it is not known; null for a tensor or live view."},
        "dtype": {"type": ["string", "null"], "enum": ["u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64", "f16", "f32", "f64", "bool", null],
                  "description": "For a tensor or live view, its element type; null for a tabular source, and for a live view before its first snapshot."},
        "shape": {"type": ["array", "null"], "items": {"type": "integer", "minimum": 0}, "maxItems": 8,
                  "description": "For a tensor or live view, the size of each axis, at most 8 axes; null otherwise."},
        "rate_hz": {"type": ["number", "null"], "description": "How often new values arrive, in hertz: snapshots per second for a live view, the nominal rate of a feed channel, or the sampling rate along axis 0 of an audio tensor; null when the source has no rate."},
        "bytes": {"type": ["integer", "null"], "description": "The size of the data in bytes, or null when it is not known."},
        "wire": {"type": "boolean", "description": "Whether the source can be streamed to the browser viewer, so that a spec that binds it also runs on the web."},
        "warnings": {"type": "array", "items": {"type": "string"}, "description": "Notes about the source, such as that it was loaded weights-only from a pickle file."},
        "updated_at": {"type": ["string", "null"], "description": "When the source's data last changed, in UTC ISO 8601, or null."}}})json"},

    {"cell", R"json({"type": ["string", "number", "boolean", "null"]})json"},

    {"catalog_sample", R"json({
      "type": "object", "additionalProperties": false,
      "required": ["id", "kind", "columns", "rows", "total_rows", "truncated", "cut_cells", "dtype", "shape", "stats", "slice"],
      "properties": {
        "id": {"type": "string", "description": "The source id."},
        "kind": {"type": "string", "enum": ["table", "tensor", "live", "feed", "metric", "run"], "description": "The source's kind."},
        "columns": {"type": ["array", "null"], "items": {"$use": "column"}, "description": "For a tabular source, its columns, in the order of each row's cells; null otherwise."},
        "rows": {"type": ["array", "null"], "items": {"type": "array", "items": {"$use": "cell"}},
                 "description": "For a tabular source, the rows given, each a list of cells in column order: a table's first rows, a feed's newest, a metric's last by step. NaN and infinities are the strings nan, inf and -inf; dates are ISO 8601 strings; nested values are their JSON text. Null otherwise."},
        "total_rows": {"type": ["integer", "null"], "description": "The source's row count when it is known, else null."},
        "truncated": {"type": "boolean", "description": "Whether the source has more rows than were given."},
        "cut_cells": {"type": "integer", "description": "How many string cells were cut to 1024 bytes."},
        "dtype": {"type": ["string", "null"], "description": "For a tensor-shaped source, its element type; null otherwise."},
        "shape": {"type": ["array", "null"], "items": {"type": "integer"}, "description": "For a tensor-shaped source, its shape; null otherwise."},
        "stats": {"type": ["object", "null"], "additionalProperties": false,
                  "required": ["count", "sampled", "exact", "nonfinite", "min", "max", "mean", "std"],
                  "description": "For a tensor-shaped source, statistics over its elements; null otherwise.",
                  "properties": {
                    "count": {"type": "integer", "description": "The tensor's number of elements."},
                    "sampled": {"type": "integer", "description": "How many elements were read: all of them up to 2^26, else an even stride."},
                    "exact": {"type": "boolean", "description": "Whether every element was read."},
                    "nonfinite": {"type": "integer", "description": "How many of the elements read are NaN or infinite."},
                    "min": {"type": ["number", "null"], "description": "The smallest finite value, or null when there is none."},
                    "max": {"type": ["number", "null"], "description": "The largest finite value, or null."},
                    "mean": {"type": ["number", "null"], "description": "The mean of the finite values, or null."},
                    "std": {"type": ["number", "null"], "description": "The population standard deviation of the finite values, or null."}}},
        "slice": {"type": ["object", "null"], "additionalProperties": false, "required": ["start", "shape", "values"],
                  "description": "For a tensor-shaped source, a box of its values; null otherwise.",
                  "properties": {
                    "start": {"type": "array", "items": {"type": "integer"}, "description": "The box's first index on each axis."},
                    "shape": {"type": "array", "items": {"type": "integer"}, "description": "The box's size on each axis."},
                    "values": {"type": "array", "items": {"type": ["number", "boolean", "string"]},
                               "description": "The box's values in row-major order; NaN and infinities are the strings nan, inf and -inf."}}}}})json"},

    {"png", R"json({
      "type": "object", "additionalProperties": false, "required": ["png_base64", "width", "height"],
      "properties": {
        "png_base64": {"type": "string", "description": "The PNG image, base64-encoded."},
        "width": {"type": "integer", "description": "The image's width in pixels."},
        "height": {"type": "integer", "description": "The image's height in pixels."}}})json"},

    // The pieces of a spec, kept in line with design/spec-format.md and the
    // spec library by the session module's test of the table. Extension tables
    // (`x`) are left out: a tool schema is closed, and spec.set_text carries them.
    {"binding", R"json({"type": "string", "minLength": 1, "maxLength": 320, "pattern": "^[a-z][a-z0-9_-]{0,63}([.].+)?$"})json"},

    {"label", R"json({"type": "string", "maxLength": 256})json"},

    {"range", R"json({"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}})json"},

    {"axis", R"json({"type": "integer", "minimum": -8, "maximum": 7})json"},

    {"safe_integer", R"json({"type": "integer", "minimum": -9007199254740991, "maximum": 9007199254740991})json"},

    {"reference", R"json({
      "type": "object", "additionalProperties": false, "required": [],
      "description": "A binding reference that the app evaluates in place of a literal: { control = <id> } reads a control, { selection = <panel id>, field = <pick field> } reads the last pick of a panel, and { state = <id> } reads a state variable.",
      "properties": {
        "control": {"type": "string", "minLength": 1, "maxLength": 64, "description": "The id of a control whose current value the place takes."},
        "selection": {"type": "string", "minLength": 1, "maxLength": 64, "description": "The id of a panel whose last pick the place reads."},
        "state": {"type": "string", "minLength": 1, "maxLength": 64, "description": "The id of a state variable whose current value the place takes."},
        "field": {"type": "string", "description": "The field to read: lo or hi of a range, a pick field of a selection, or a field of a state such as the cursor's first and latest."},
        "entity": {"type": "string", "description": "selection: the entity path; only a pick on that entity counts."},
        "default": {"description": "The literal the place takes while the input has no value."}}})json"},

    {"transform", R"json({
      "type": "object", "additionalProperties": false, "required": ["op"],
      "properties": {
        "op": {"type": "string", "enum": ["slice", "reduce", "normalise", "window", "surface", "path"], "description": "slice selects along one axis; reduce reduces over axes; normalise rescales values; window keeps a range of the source's recent steps as a new leading axis; surface turns a tensor of rank 2 into a height field of geometry; path turns a tensor of rows into a polyline of geometry."},
        "axis": {"type": ["integer", "array"], "minimum": -8, "maximum": 7, "minItems": 1, "maxItems": 8, "items": {"$use": "axis"},
                 "description": "The axis, from -8 to 7, counting from 0 (negative counts from the end). Required for slice and reduce; reduce also takes a list of 1 to 8 axes. normalise without an axis covers the whole tensor."},
        "index": {"$use": "reference", "type": ["integer", "object"], "minimum": -9007199254740991, "maximum": 9007199254740991, "description": "slice: the one index to keep, negative counting from the end; it drops the axis. Not with start, stop or step. A number or a reference."},
        "start": {"$use": "reference", "type": ["integer", "object"], "minimum": -9007199254740991, "maximum": 9007199254740991, "description": "slice: the first index, Python style, negative allowed; window: the first step number, included. A number or a reference."},
        "stop": {"$use": "reference", "type": ["integer", "object"], "minimum": -9007199254740991, "maximum": 9007199254740991, "description": "slice: the index to stop before, Python style; window: the last step number, included. A number or a reference."},
        "steps": {"$use": "reference", "type": ["integer", "object"], "minimum": 1, "maximum": 65536, "description": "window: the last N kept steps up to the cursor; not with start or stop. A number or a reference."},
        "step": {"type": "integer", "minimum": 1, "maximum": 9007199254740991, "description": "slice: the step, at least 1."},
        "fn": {"type": "string", "enum": ["mean", "sum", "min", "max", "std", "var", "norm", "argmax", "argmin", "entropy"],
               "description": "reduce: the function; entropy treats the axis as a probability distribution, in nats; argmax and argmin take one axis."},
        "keepdims": {"type": "boolean", "description": "reduce: keep the reduced axes with size 1; false by default."},
        "mode": {"type": "string", "enum": ["minmax", "zscore", "softmax", "l2"], "description": "normalise: the rescaling."},
        "max_cells": {"type": "integer", "minimum": 1, "maximum": 262144, "description": "surface: the most cells of the height field, from 1 to 262144; 32768 when absent."},
        "pool": {"type": "string", "enum": ["mean", "max", "min"], "description": "surface: how the cells merge when the tensor has more than max_cells; mean when absent."},
        "columns": {"type": "array", "minItems": 2, "maxItems": 3, "items": {"type": "integer", "minimum": 0}, "description": "path: the two or three column indices that give the coordinates; the first three columns when absent."},
        "max_points": {"type": "integer", "minimum": 2, "maximum": 1048576, "description": "path: the most points of the polyline, from 2 to 1048576; 65536 when absent."}}})json"},

    {"source", R"json({
      "type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {
        "id": {"$use": "id", "description": "The source's id in this spec, which panels bind."},
        "catalog": {"$use": "catalog_id", "maxLength": 256, "description": "A catalog source id, from catalog.list, such as live:attn.weights or table:run42_metrics."},
        "sql": {"type": "string", "minLength": 1, "maxLength": 65536, "description": "One read-only SELECT (DuckDB SQL) over the workspace's tables, named as in data.query."},
        "from": {"$use": "id", "description": "The id of another source of this spec that this one derives from, usually through transforms."},
        "transform": {"type": "array", "maxItems": 16, "items": {"$use": "transform"}, "description": "Up to 16 tensor transforms applied in order; only for tensor-shaped data."},
        "history": {"type": "integer", "minimum": 0, "maximum": 65536, "description": "The steps the app keeps for a source whose root is a live tensor, from 0 to 65536; 64 when absent for such a source without from."}}})json"},

    {"bind", R"json({
      "type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "x": {"$use": "reference", "type": ["string", "object"], "minLength": 1, "maxLength": 320, "pattern": "^[a-z][a-z0-9_-]{0,63}([.].+)?$", "description": "series: the x values, a binding such as loss.step; points: the x coordinates. A reference to a text state whose choices are source ids also works."},
        "y": {"$use": "reference", "type": ["string", "array", "object"], "minLength": 1, "maxLength": 320, "pattern": "^[a-z][a-z0-9_-]{0,63}([.].+)?$",
              "minItems": 1, "maxItems": 16, "items": {"$use": "binding"},
              "description": "series: one binding or a list of 1 to 16 bindings drawn as lines; points: the y coordinates. A reference to a text state whose choices are source ids also works."},
        "z": {"$use": "reference", "type": ["string", "object"], "minLength": 1, "maxLength": 320, "pattern": "^[a-z][a-z0-9_-]{0,63}([.].+)?$", "description": "points: the z coordinates, for a 3-D cloud. A reference to a text state whose choices are source ids also works."},
        "color": {"$use": "reference", "type": ["string", "object"], "minLength": 1, "maxLength": 320, "pattern": "^[a-z][a-z0-9_-]{0,63}([.].+)?$", "description": "points: a value per point that sets its colour through the colormap. A reference to a text state whose choices are source ids also works."},
        "size": {"$use": "reference", "type": ["string", "object"], "minLength": 1, "maxLength": 320, "pattern": "^[a-z][a-z0-9_-]{0,63}([.].+)?$", "description": "points: a value per point that sets its size. A reference to a text state whose choices are source ids also works."},
        "positions": {"$use": "reference", "type": ["string", "object"], "minLength": 1, "maxLength": 320, "pattern": "^[a-z][a-z0-9_-]{0,63}([.].+)?$", "description": "points: a tensor of shape [n, 2] or [n, 3], instead of x, y and z. A reference to a text state whose choices are source ids also works."},
        "values": {"$use": "reference", "type": ["string", "object"], "minLength": 1, "maxLength": 320, "pattern": "^[a-z][a-z0-9_-]{0,63}([.].+)?$", "description": "heatmap: a rank-2 tensor; histogram: the values to count; image: a tensor [H, W], [H, W, 3] or [H, W, 4]. A reference to a text state whose choices are source ids also works."},
        "rows": {"$use": "reference", "type": ["string", "object"], "minLength": 1, "maxLength": 64, "pattern": "^[a-z][a-z0-9_-]{0,63}$", "description": "table: the id of a tabular source whose rows are shown. A reference to a text state whose choices are source ids also works."},
        "value": {"$use": "reference", "type": ["string", "object"], "minLength": 1, "maxLength": 320, "pattern": "^[a-z][a-z0-9_-]{0,63}([.].+)?$", "description": "text: a column, shown as its last value, or a tensor of one element. A reference to a text state whose choices are source ids also works."},
        "scene": {"$use": "reference", "type": ["string", "object"], "minLength": 6, "maxLength": 256, "pattern": "^live:.+$", "description": "scene (spec version 1): the live scene of the attached script, as live:<scene name>, such as live:model; catalog.list names the scenes. A reference to a text state whose choices are source ids also works."}}})json"},

    {"options", R"json({
      "type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "x_label": {"$use": "label", "description": "series, heatmap, points and histogram: the label of the x axis."},
        "y_label": {"$use": "label", "description": "series, heatmap, points and histogram: the label of the y axis."},
        "z_label": {"$use": "label", "description": "points: the label of the z axis."},
        "x_range": {"$use": "reference", "type": ["array", "object"], "minItems": 2, "maxItems": 2, "items": {"type": "number"}, "description": "series: a fixed x range, [min, max] with min below max, or a reference to a range."},
        "y_range": {"$use": "reference", "type": ["array", "object"], "minItems": 2, "maxItems": 2, "items": {"type": "number"}, "description": "series: a fixed y range, [min, max] with min below max, or a reference to a range."},
        "range": {"$use": "reference", "type": ["array", "object"], "minItems": 2, "maxItems": 2, "items": {"type": "number"}, "description": "heatmap and image: the values at the ends of the colormap; histogram: the binned range; [min, max] with min below max, or a reference to a range."},
        "x_log": {"type": "boolean", "description": "series: a log scale on the x axis."},
        "y_log": {"type": "boolean", "description": "series: a log scale on the y axis; histogram: on the counts."},
        "colormap": {"type": "string", "enum": ["viridis", "magma", "rdbu", "greys", "cividis", "inferno", "plasma", "turbo", "twilight", "coolwarm", "batlow", "vik", "roma", "oslo", "lajolla", "berlin"], "description": "heatmap and points: the colormap; viridis for a heatmap when absent. A spec file may also give a reference to a text state whose choices are colormap names, as an entity's colormap does."},
        "bins": {"type": "integer", "minimum": 1, "maximum": 65536, "description": "histogram: the number of bins, from 1 to 65536; 64 by default."},
        "columns": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "string", "minLength": 1, "maxLength": 255}, "description": "table: the columns to show, in order, each once."},
        "text": {"type": "string", "maxLength": 16384, "description": "text: static text to show."},
        "up": {"type": "string", "enum": ["+x", "-x", "+y", "-y", "+z", "-z"], "description": "scene: the axis kept upright; the scene's own when absent."},
        "view_camera": {"type": ["object", "string"], "additionalProperties": false, "required": [], "minLength": 1, "maxLength": 255,
                        "description": "scene: the camera the panel opens on, an orbit object or the entity path of a camera in the scene; the scene's own when absent.",
                        "properties": {
                          "target": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}, "description": "The point the orbit looks at."},
                          "distance": {"type": "number", "minimum": 0, "description": "The distance from the target, above 0."},
                          "azimuth": {"type": "number", "description": "The orbit's azimuth in degrees, read against up."},
                          "elevation": {"type": "number", "minimum": -89, "maximum": 89, "description": "The orbit's elevation in degrees, from -89 to 89."}}},
        "include": {"type": "array", "minItems": 1, "maxItems": 64, "items": {"type": "string", "minLength": 1, "maxLength": 255}, "description": "scene: entity path patterns to draw, where * matches one path segment and ** any number; every entity when absent."},
        "exclude": {"type": "array", "minItems": 1, "maxItems": 64, "items": {"type": "string", "minLength": 1, "maxLength": 255}, "description": "scene: entity path patterns not to draw, applied after include."},
        "labels": {"type": "boolean", "description": "scene: whether the entities' labels are drawn; true by default."},
        "axes": {"type": "boolean", "description": "scene: whether the axes and the ground grid are drawn; true by default."},
        "scale_bar": {"type": "boolean", "description": "scene: whether a scale bar in the scene's units is drawn; false by default."},
        "legend": {"type": ["string", "boolean"], "enum": ["auto", true, false], "description": "Spec version 1: auto, true or false; auto shows a legend for two series or more, or a colormap."},
        "box": {"type": ["string", "array"], "minLength": 4, "maxLength": 4, "minItems": 3, "maxItems": 3, "items": {"type": "number", "minimum": 0}, "description": "Spec version 1, scene with entities: the data box, auto or three positive proportions of its x, y and z sides."},
        "views": {"type": "array", "maxItems": 16, "description": "Spec version 1, scene: the saved views, each with a unique name.",
                  "items": {"type": "object", "additionalProperties": false, "required": ["name"],
                            "properties": {
                              "name": {"type": "string", "minLength": 1, "maxLength": 64, "description": "The view's name, unique in the panel."},
                              "target": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}, "description": "The point the orbit looks at."},
                              "distance": {"type": "number", "minimum": 0, "description": "The distance from the target."},
                              "azimuth": {"type": "number", "description": "The orbit's azimuth in degrees."},
                              "elevation": {"type": "number", "minimum": -89, "maximum": 89, "description": "The orbit's elevation in degrees."}}}},
        "select": {"type": "object", "additionalProperties": false, "required": ["state", "field"], "description": "Spec version 1: a pick on the panel writes one of its fields into a state variable.",
                   "properties": {
                     "state": {"type": "string", "minLength": 1, "maxLength": 64, "description": "The state variable that the pick writes."},
                     "field": {"type": "string", "minLength": 1, "maxLength": 32, "description": "The pick field it writes, such as column, row, index or value."},
                     "entity": {"type": "string", "minLength": 1, "maxLength": 255, "description": "Only a pick on this entity writes."}}},
        "format": {"type": "string", "minLength": 1, "maxLength": 1024, "description": "Spec version 1, text: a bound text in the format language, in place of text."},
        "fields": {"description": "Spec version 1, text: a JSON object that gives each name of the format a reference or a source id."}}})json"},

    {"panel", R"json({
      "type": "object", "additionalProperties": false, "required": ["id", "primitive", "title", "bind", "options"],
      "properties": {
        "id": {"$use": "id", "description": "The panel's id."},
        "primitive": {"type": "string", "enum": ["series", "heatmap", "points", "histogram", "table", "image", "text", "scene"], "description": "What the panel draws."},
        "title": {"type": ["string", "null"], "description": "The panel's title, or null when it has none."},
        "bind": {"$use": "bind", "description": "The panel's bindings, by slot."},
        "options": {"$use": "options", "description": "The panel's options."}}})json"},

    {"layout", R"json({
      "type": "object", "additionalProperties": false, "required": ["root", "nodes"],
      "properties": {
        "root": {"$use": "id", "description": "The id of the top node, or of the one panel or control when the layout is a single leaf."},
        "nodes": {"type": "array", "maxItems": 256, "description": "The nodes of the layout, in any order. Each splits its space between its children or shows them as tabs.",
                  "items": {"type": "object", "additionalProperties": false, "required": ["id", "split", "children"],
                            "properties": {
                              "id": {"$use": "id", "description": "The node's id; it must not be a panel's or a control's id."},
                              "split": {"type": "string", "enum": ["horizontal", "vertical", "tabs"], "description": "horizontal puts the children side by side, vertical one above another, tabs in tabs."},
                              "ratio": {"type": ["number", "null"], "minimum": 0.05, "maximum": 0.95, "description": "For a split of exactly two children, the share of the first, from 0.05 to 0.95; null for an even split and for tabs."},
                              "shares": {"type": "array", "minItems": 2, "maxItems": 64, "items": {"type": "number", "minimum": 0}, "description": "Spec version 1: one positive share for each child of a split, in place of ratio; each child first gets its minimum and the rest goes by the shares."},
                              "mins": {"description": "Spec version 1: a JSON object that gives a panel or control child, by its id, its minimum size as [columns, lines] of the text scale, each an integer from 4 to 400."},
                              "title": {"type": ["string", "null"], "minLength": 1, "maxLength": 256, "description": "The node's tab label when it is a child of tabs, or null."},
                              "children": {"type": "array", "minItems": 1, "maxItems": 64, "items": {"$use": "id"}, "description": "Panel ids, control ids (spec version 1) and node ids, in order: 2 to 64 for a split, 1 to 64 for tabs; each node is used once."}}}}}})json"},

    // Spec version 1 controls (design/instruments.md, Control Table).
    {"control_bind", R"json({"type": "string", "minLength": 5, "maxLength": 320, "pattern": "^(param:.{1,63}|action:.{1,63}|run:pause|value|state:[a-z][a-z0-9_-]{0,63}|cursor|intervene:.{1,255})$",
      "description": "The one target of the control: param:<name> for a parameter the script registered with caliper.param, action:<name> for an action it registered with caliper.action, run:pause for the run's pause and resume, value for a value the spec holds, state:<id> for a state variable of the spec, cursor for the time cursor, or intervene:<point> for an intervention point that intervene.points lists. A slider or text binds param:, value or state:; a range, a picker and a segmented group bind value or state:; a button binds action: or run:pause; a toggle binds intervene:<point>, param:<name>, value or state:; a transport binds cursor. run.state lists the names of parameters and actions."})json"},

    {"control_options", R"json({
      "type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "min": {"$use": "reference", "type": ["number", "object"], "description": "slider and range: the low end of the range, a number or a reference to a control, a selection or a state such as the cursor's first; with param:, the parameter's own minimum when absent."},
        "max": {"$use": "reference", "type": ["number", "object"], "description": "slider and range: the high end of the range, above min, a number or a reference to a control, a selection or a state such as the cursor's latest; with param:, the parameter's own maximum when absent."},
        "step": {"type": "number", "minimum": 0, "description": "slider and range: the step between values, above 0; continuous when absent."},
        "log": {"type": "boolean", "description": "slider: a log scale, which needs min above 0; with param:, the parameter's own scale when absent."},
        "value": {"description": "slider bound to value: the number it starts with, min when absent; range: [lo, hi]; toggle: a boolean; picker, text and segmented: the value or the list of values it starts with."},
        "choices": {"type": "array", "minItems": 1, "maxItems": 1024, "items": {"description": "A value, or a table with value, label and key."}, "description": "picker and segmented: the choices, from 1 to 1024 for a picker and 2 to 16 for a segmented group."},
        "choices_from": {"type": "string", "minLength": 3, "maxLength": 320, "description": "picker: <source>.<column>, the column whose distinct values are the choices, at most 1024."},
        "multiple": {"type": "boolean", "description": "segmented: whether any number of buttons are on and the value is a list; false when absent."},
        "placeholder": {"type": "string", "maxLength": 256, "description": "text: the hint that shows while the field is empty."},
        "max_length": {"type": "integer", "minimum": 1, "maximum": 4096, "description": "Text byte limit. A param binding permits at most 4096 UTF-8 bytes. Value and state bindings permit at most 1024."},
        "min_span": {"type": "number", "minimum": 0, "description": "range: the least distance between lo and hi."},
        "follow": {"type": "boolean", "description": "range: whether the range moves with its upper bound while its upper end equals it; false when absent."},
        "speed": {"type": "number", "minimum": 0.1, "maximum": 1000, "description": "transport: the play speed at open, in steps per second; 10 when absent."},
        "keys": {"type": "boolean", "description": "transport: whether Space, the arrow keys, Home and End control the cursor; true when absent."},
        "format": {"type": "string", "maxLength": 1024, "description": "readout: the text to show, written in the format language with {name} places."},
        "fields": {"description": "readout: a JSON object that gives each name of the format a reference."}}})json"},

    {"control", R"json({
      "type": "object", "additionalProperties": false, "required": ["id", "kind", "title", "bind", "options"],
      "properties": {
        "id": {"$use": "id", "description": "The control's id."},
        "kind": {"type": "string", "enum": ["slider", "button", "range", "toggle", "picker", "text", "segmented", "readout", "transport"], "description": "What the control is."},
        "title": {"type": ["string", "null"], "description": "The control's title, or null when it has none."},
        "bind": {"type": "string", "description": "The control's target, such as param:lr."},
        "options": {"$use": "control_options", "description": "The control's options."}}})json"},

    {"run_param", R"json({
      "type": "object", "additionalProperties": false, "required": ["name", "type", "value", "pending"],
      "properties": {
        "name": {"type": "string", "description": "The parameter's name, as the script registered it with caliper.param."},
        "type": {"type": "string", "enum": ["number", "integer", "boolean", "choice", "text"], "description": "The parameter's type."},
        "value": {"type": ["number", "integer", "boolean", "string"], "description": "The value the script uses now."},
        "pending": {"type": ["number", "integer", "boolean", "string", "null"], "description": "A value that was sent and applies at the script's next step, or null."},
        "min": {"type": "number", "description": "The lowest value the script accepts; absent when it set none."},
        "max": {"type": "number", "description": "The highest value the script accepts; absent when it set none."},
        "log": {"type": "boolean", "description": "Whether the parameter is shown on a log scale."},
        "choices": {"type": "array", "items": {"type": "string"}, "description": "A choice parameter's values."},
        "help": {"type": "string", "description": "The script's one-line help for the parameter."}}})json"},

    {"run_action", R"json({
      "type": "object", "additionalProperties": false, "required": ["name", "running", "queued"],
      "properties": {
        "name": {"type": "string", "description": "The action's name, as the script registered it with caliper.action."},
        "running": {"type": "boolean", "description": "Whether the action runs now."},
        "queued": {"type": "boolean", "description": "Whether a run of the action waits for the script's next step."},
        "help": {"type": "string", "description": "The script's one-line help for the action."}}})json"},

    {"run_state_name", R"json({"type": "string", "enum": ["none", "running", "pausing", "paused", "stepping"], "description": "The run's state: none before the script's first step, then running, pausing (a pause waits for the next step), paused or stepping."})json"},
    {"run_id", R"json({"type": "string", "pattern": "^[a-z0-9-]{1,40}$", "minLength": 1, "maxLength": 40, "description": "A run's id, from run.list: a managed run's id, or user for the person's own attached script."})json"},
    {"run_params", R"json({"type": "array", "maxItems": 64, "description": "Parameter values by name, at most 64, whose names and string values have at most 1024 bytes in all, so that the review card shows them in full; the script applies each one when it registers that parameter with caliper.param.",
      "items": {"type": "object", "additionalProperties": false, "required": ["name", "value"],
                "properties": {
                  "name": {"type": "string", "minLength": 1, "maxLength": 63, "description": "The parameter's name, as the script registers it."},
                  "value": {"type": ["number", "integer", "boolean", "string"], "maxLength": 256, "description": "Its value: a number, true or false, or a string of at most 256 bytes."}}}})json"},
    {"run_entry", R"json({
      "type": "object", "additionalProperties": false,
      "required": ["id", "managed", "state", "script", "args", "params", "parent", "step", "exit_code", "attached", "stopping", "error"],
      "properties": {
        "id": {"type": "string", "description": "The run's id; user for the person's own attached script."},
        "managed": {"type": "boolean", "description": "Whether the model runner started the run."},
        "state": {"type": "string", "enum": ["starting", "running", "paused", "exited", "failed"], "description": "starting until the script's stream answers, then running or paused, then exited or failed when its process ended."},
        "script": {"type": "string", "description": "The script: its path in the workspace for a managed run, its name for the person's script."},
        "args": {"type": "array", "items": {"type": "string"}, "description": "The script's arguments."},
        "params": {"$use": "run_params", "description": "The parameter values the run started with, or for a fork the values it changed."},
        "parent": {"type": ["string", "null"], "description": "The run a fork started from, or null."},
        "step": {"type": ["integer", "null"], "description": "The step of the run's latest checkpoint, or null before its first."},
        "exit_code": {"type": ["integer", "null"], "description": "The process's exit code once it ended, a signal as its negative number; null while it runs."},
        "attached": {"type": "boolean", "description": "Whether the app shows this run now."},
        "stopping": {"type": "boolean", "description": "Whether run.stop asked the run to stop."},
        "error": {"type": ["string", "null"], "description": "Why the run failed, in one sentence; null otherwise."}}})json"},

    {"event", R"json({
      "type": "object", "additionalProperties": false, "required": ["seq", "time", "kind", "panel", "view", "origin"],
      "properties": {
        "seq": {"type": "integer", "minimum": 1, "description": "The event's number; it grows by one with each event the app records."},
        "time": {"type": "string", "description": "When the app recorded it, in UTC ISO 8601 with milliseconds."},
        "kind": {"type": "string", "enum": ["pick", "select", "control", "run", "spec", "approval", "program", "state", "cursor"], "description": "What happened."},
        "panel": {"type": ["string", "null"], "description": "The spec panel or control it happened in, or null."},
        "view": {"type": ["string", "null"], "description": "pick and select: the live view, such as the scene's name; null otherwise."},
        "origin": {"type": "string", "enum": ["ui", "builder", "agent", "mcp", "api", "script"], "description": "Who caused it: ui for the person in the app, script for the running script, else the origin of the command."},
        "entity": {"type": "string", "description": "pick: the entity path of the item picked."},
        "primitive": {"type": "string", "description": "pick: the scene part of the item, such as points or boxes."},
        "index": {"type": ["integer", "array"], "items": {"type": "integer"}, "description": "pick: the item's index in the batch the script sent; for a pick of a piece's item, its item map's first four integer fields as tensor indices, -1 where there are fewer."},
        "item": {"type": "integer", "minimum": 0, "description": "pick of a piece's item: the element index in the item map's resource."},
        "row_id": {"type": ["integer", "null"], "description": "pick: the item's row in the script's source tensor, or null when the scene does not say."},
        "position": {"type": ["array", "null"], "minItems": 3, "maxItems": 3, "items": {"type": "number"}, "description": "pick: the point hit, in scene coordinates, or null without depth."},
        "pixel": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer"}, "description": "pick on an image: the pixel, x then y."},
        "entities": {"type": "array", "maxItems": 256, "description": "select: the items selected by entity; empty when the selection was cleared.",
                     "items": {"type": "object", "additionalProperties": false, "required": ["entity", "primitive", "indices", "row_ids"],
                               "properties": {
                                 "entity": {"type": "string", "description": "The entity path."},
                                 "primitive": {"type": "string", "description": "The scene part."},
                                 "indices": {"type": "array", "items": {"type": "integer"}, "description": "The items' indices in the batch."},
                                 "row_ids": {"type": "array", "items": {"type": ["integer", "null"]}, "description": "The items' rows in the source tensor."}}}},
        "truncated": {"type": "boolean", "description": "select: whether the selection held more than 32768 items and was cut."},
        "fields": {"description": "pick: the pick fields as a JSON object; for a piece's item, the item map's fields of the element by name, as numbers."},
        "annotation": {"type": "string", "description": "pick: the id of the annotation that was picked."},
        "key": {"type": "string", "description": "control: the shortcut key that made the change, when a key did."},
        "control": {"type": "string", "description": "control: the control's id."},
        "bind": {"type": "string", "description": "control: the control's target, such as param:lr."},
        "change": {"type": "string", "enum": ["state", "param", "action", "attach", "runner", "intervene", "compiled", "refused", "stopped", "live", "held", "playing", "moved", "clamped"], "description": "cursor: what changed: live, held, playing, moved (by a command, a step or a key) or clamped (to the nearest kept step); run: what changed: the run's state, a parameter applied at a step, the end of an action, the attachment of a script, the state of a managed run (runner), or an intervention applied at a step (intervene); program: compiled, refused by a check, or stopped by a limit."},
        "state": {"type": "string", "description": "state: the state variable's id; run with change state: the run's new state; approval: pending, approved, denied or expired."},
        "step": {"type": ["integer", "null"], "description": "run: the script's step when it changed, or null before the first step; cursor: the step that the cursor shows; pick of a piece's item: the step of the resource version that the frame showed."},
        "first": {"type": "integer", "description": "cursor: the first kept step."},
        "latest": {"type": "integer", "description": "cursor: the newest kept step."},
        "playing": {"type": "boolean", "description": "cursor: whether the cursor plays."},
        "speed": {"type": "number", "description": "cursor: the play speed in steps per second."},
        "previous": {"description": "state: the value before the change."},
        "writer": {"type": "string", "description": "state: what wrote the value, such as a control id, a panel id or a command."},
        "name": {"type": "string", "description": "run with change param or action: the parameter's or the action's name."},
        "value": {"description": "control: the control's new value, which is a list for a range or a group with multiple; state: the state variable's new value; run with change param: the value the script applied."},
        "status": {"type": "string", "enum": ["ok", "error"], "description": "run with change action: how the action ended."},
        "text": {"type": "string", "description": "run with change action: the text the action returned, or its error message."},
        "attached": {"type": "boolean", "description": "run with change attach: whether a script is attached now."},
        "script": {"type": "string", "description": "run with change attach: the attached script's name."},
        "command": {"type": "string", "description": "spec and approval: the command, such as panel.create."},
        "event": {"type": "string", "enum": ["do", "undo", "redo"], "description": "spec: whether the command was done, undone or redone."},
        "applet": {"type": "string", "description": "spec: the open spec's applet id."},
        "approval": {"type": "string", "description": "approval: the approval's id."},
        "target": {"type": "string", "description": "approval of a run command: the parameter or the action it names."},
        "run": {"type": "string", "description": "run with change runner: the managed run's id."},
        "parent": {"type": ["string", "null"], "description": "run with change runner: the run a fork started from, or null."},
        "exit_code": {"type": ["integer", "null"], "description": "run with change runner: the exit code once the process ended, else null."},
        "error": {"type": ["string", "null"], "description": "run with change runner: why the run failed, else null."},
        "point": {"type": "string", "description": "run with change intervene: the intervention point."},
        "op": {"type": ["string", "null"], "description": "run with change intervene: the op applied, or null when the point's interventions were cleared."},
        "program": {"type": "string", "description": "program: the program's id; pick of a piece's item: the program that drew it."},
        "half": {"type": "string", "enum": ["derive", "draw"], "description": "program of an instance of a piece: the half that compiled, was refused or stopped."},
        "pass": {"type": ["integer", "null"], "minimum": 0, "description": "program of an instance of a piece: the index of the pass in its half, or null for the whole half."},
        "host": {"type": "string", "enum": ["script", "viewer"], "description": "program: where it compiled, was refused or stopped."},
        "code": {"type": "string", "description": "program with change refused or stopped: a stable code, such as program.writes_input."},
        "message": {"type": "string", "description": "program with change refused or stopped: one plain sentence that says why."},
        "line": {"type": "integer", "minimum": 0, "description": "program with change refused: the line in the source, or 0."},
        "column": {"type": "integer", "minimum": 0, "description": "program with change refused: the column in the source, or 0."}}})json"},

    // GPU programs and interventions (design/gpu-programs.md)
    {"entity_path", R"json({"type": "string", "minLength": 1, "maxLength": 255, "pattern": "^[A-Za-z0-9_.-]{1,63}(/[A-Za-z0-9_.-]{1,63}){0,15}$", "description": "An entity path of a scene, such as programs/ribbons: segments of letters, digits, _, . and - joined by /."})json"},

    {"compiler_message", R"json({
      "type": "object", "additionalProperties": false, "required": ["severity", "code", "message", "line", "column"],
      "properties": {
        "severity": {"type": "string", "enum": ["error", "warning"], "description": "Whether the message stops the source or only warns."},
        "code": {"type": "string", "description": "A stable code, such as program.compile_failed or program.writes_input."},
        "message": {"type": "string", "description": "The message of the compiler or the sandbox, in one line."},
        "line": {"type": "integer", "minimum": 0, "description": "The line in the given source, from 1; 0 when the message has no place."},
        "column": {"type": "integer", "minimum": 0, "description": "The column in bytes, from 1; 0 when the message has no place."},
        "file": {"type": "string", "description": "A piece: the file of the line, such as draw.wgsl, common.wgsl, lib/colormap@1 or piece.toml."},
        "where": {"type": "string", "description": "A piece: the manifest key or the pass of the message, such as draw.passes[1].target."}}})json"},

    {"program_input", R"json({
      "type": "object", "additionalProperties": false, "required": ["var", "source"],
      "properties": {
        "var": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]{0,62}$", "description": "The WGSL variable that the input binds, declared var<storage, read> in any group and binding."},
        "source": {"type": "string", "minLength": 6, "maxLength": 68, "pattern": "^live:.{1,63}$", "description": "The script's tensor as live:<name>: a parameter of model.named_parameters() such as live:layers.0.weight, its gradient as live:<parameter>.grad, or a module's last output as live:<module>.act. The program reads it in place and cannot write it."}}})json"},

    {"program_output", R"json({
      "type": "object", "additionalProperties": false, "required": ["kind"],
      "properties": {
        "kind": {"type": "string", "enum": ["geometry", "pixels", "tensor"], "description": "geometry writes arrays that a scene panel draws as one entity; pixels writes an RGBA8 image that the live view program.<id> carries; tensor writes an array of f32 values with the given shape that a source can read."},
        "panel": {"$use": "id", "description": "geometry: the scene panel whose scene gets the entity."},
        "entity": {"$use": "entity_path", "description": "geometry: the entity path; programs/<id> when absent."},
        "part": {"type": "string", "enum": ["points", "lines"], "description": "geometry: points draws one point for each item; lines draws one segment for each two items; points when absent."},
        "max_count": {"type": "integer", "minimum": 1, "maximum": 1000000, "description": "geometry: the most items; the program adds the valid count to count, a var<storage, read_write> atomic<u32>."},
        "arrays": {"type": "array", "minItems": 1, "maxItems": 4, "items": {"type": "string", "enum": ["positions", "colors", "values", "radii"]}, "description": "geometry: the arrays the program writes, each a var<storage, read_write> of that name: positions array<f32> with 3 for each item (necessary), colors array<u32> (RGBA8 from pack4x8unorm), values array<f32>, radii array<f32> (points only)."},
        "style": {"type": "object", "additionalProperties": false, "required": [], "description": "geometry: how values map to colors.",
                  "properties": {
                    "colormap": {"type": "string", "enum": ["viridis", "magma", "rdbu", "greys", "cividis", "inferno", "plasma", "turbo", "twilight", "coolwarm", "batlow", "vik", "roma", "oslo", "lajolla", "berlin"], "description": "The colormap of values; viridis when absent."},
                    "vmin": {"type": "number", "description": "The value at the low end; the batch's minimum when absent."},
                    "vmax": {"type": "number", "description": "The value at the high end; the batch's maximum when absent."}}},
        "width": {"type": "integer", "minimum": 1, "maximum": 2048, "description": "pixels: the image's width."},
        "height": {"type": "integer", "minimum": 1, "maximum": 2048, "description": "pixels: the image's height."},
        "shape": {"type": "array", "minItems": 1, "maxItems": 4, "items": {"type": "integer", "minimum": 1, "maximum": 1000000}, "description": "tensor: the output's shape, from 1 to 4 sizes, with at most 1,000,000 elements in all."},
        "format": {"type": "string", "enum": ["rgba8unorm"], "description": "pixels: the format; the program writes pixels, a var<storage, read_write> array<u32> of width times height items packed with pack4x8unorm, row by row."}}})json"},

    {"program_limits", R"json({
      "type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "time_ms": {"type": "number", "minimum": 0.01, "maximum": 50, "description": "The most GPU time of one pass in milliseconds; 4 when absent. A program over it on 3 passes in a row stops."},
        "memory_mb": {"type": "integer", "minimum": 1, "maximum": 512, "description": "The budget of the output buffers and the bound tensors in MiB; 64 when absent."},
        "every": {"type": "integer", "minimum": 1, "maximum": 10000, "description": "The fewest steps between two passes; the runtime may run the program less often to keep training at 95 percent of its speed, never more often."}}})json"},

    {"program", R"json({
      "type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {
        "id": {"$use": "id", "maxLength": 55, "description": "The program's id, unique among programs, at most 55 characters, because its pixels output is the live view program.<id>."},
        "title": {"type": "string", "minLength": 1, "maxLength": 256, "description": "The program's label."},
        "wgsl": {"type": "string", "minLength": 1, "maxLength": 65536, "description": "The WGSL source with a compute entry point, at most 64 KiB, with no enable, requires or diagnostic directive."},
        "entry": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]{0,62}$", "description": "The compute entry point; the one compute entry point when absent."},
        "inputs": {"type": "array", "maxItems": 8, "items": {"$use": "program_input"}, "description": "The tensors the program reads, at most 8."},
        "output": {"$use": "program_output", "description": "The program's one output."},
        "limits": {"$use": "program_limits", "description": "The program's limits."},
        "workgroups": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "integer", "minimum": 1, "maximum": 65535}, "description": "The workgroup counts x, y and z of the dispatch; from max_count or the image size and the entry's workgroup size when absent."},
        "params": {"description": "A JSON object that gives each of the program's named parameters, at most 16 names, a number or a binding reference to a control, a selection or a state; the WGSL reads them by name."},
        "piece": {"$use": "piece_pin", "description": "An instance of a piece: the pinned piece. Such a program has none of wgsl, entry, inputs, output, limits and workgroups; a mix is program.form_mixed."},
        "bind": {"$use": "piece_bind"},
        "panel": {"$use": "id", "description": "An instance: the scene panel that the piece's draw half writes; necessary when the piece draws."}}})json"},

    // Pieces (design/programmable-graphics.md, Piece Manifest and Program
    // Instances): the manifest, the pin and the bindings of an instance, which
    // the piece commands, program.create, program.patch and the spec take.
    {"piece_input", R"json({"type": "object", "additionalProperties": false, "required": ["name", "kind"], "properties": {"name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "The input's name, which the WGSL reads as a generated read-only array."}, "kind": {"type": "string", "enum": ["tensor", "buffer"], "description": "tensor: a tensor of the script, read by the derive half; buffer: another program's published buffer resource, read by the draw half."}, "dtype": {"type": "string", "pattern": "^(f32|i32|u32|u8|u16)(\\|(f32|i32|u32|u8|u16)){0,4}$", "description": "tensor: the element type, f32, i32, u32, u8 or u16, or two to five of them joined by |, each once; f32 when absent. The bound tensor holds one of them. A narrow type or several types bind the tensor's bytes as array<u32>, read through the checked accessor <name>_at()."}, "rank_min": {"type": "integer", "minimum": 0, "maximum": 8, "description": "tensor: the fewest axes; 1 when absent."}, "rank_max": {"type": "integer", "minimum": 0, "maximum": 8, "description": "tensor: the most axes; 8 when absent."}, "element": {"type": "string", "enum": ["f32", "i32", "u32", "vec2<f32>", "vec4<f32>", "vec2<u32>", "vec4<u32>", "u8", "u16"], "description": "buffer: a plain element type, or the narrow u8 or u16 packed four or two to a 32-bit word; give either element, or struct with fields."}, "struct": {"type": "string", "pattern": "^[A-Z][A-Za-z0-9_]{0,31}$", "description": "buffer: the name of the struct that the composer generates from fields."}, "fields": {"type": "array", "minItems": 1, "maxItems": 16, "items": {"type": "string", "pattern": "^[a-z_][a-z0-9_]{0,31}: (f32|i32|u32|vec2<f32>|vec4<f32>|vec2<u32>|vec4<u32>)$"}, "description": "buffer: the struct's fields in order, each written name: type."}, "title": {"type": "string", "minLength": 1, "maxLength": 256, "description": "The input's label in the builder."}, "optional": {"type": "boolean", "description": "true: a program may leave the input unbound; a tensor input then has the shape (0, 0, 0, 0) and reads zero, and a buffer input has the count 0. false when absent."}}})json"},

    {"piece_param", R"json({"type": "object", "additionalProperties": false, "required": ["name"], "properties": {"name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "The parameter's name, a field of the generated struct Params that both halves read as params.<name>."}, "type": {"type": "string", "enum": ["f32", "i32", "u32"], "description": "The field's type; f32 when absent."}, "default": {"type": "number", "description": "The value when an instance gives none; 0 when absent."}, "min": {"type": "number", "description": "The least value; with max, an instance without a value gets a range control."}, "max": {"type": "number", "description": "The greatest value."}, "title": {"type": "string", "minLength": 1, "maxLength": 256, "description": "The label of the control."}}})json"},

    {"piece_resource", R"json({"type": "object", "additionalProperties": false, "required": ["id", "kind"], "properties": {"id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "The resource's id, a WGSL name, unique among the piece's inputs, resources and draw buffers."}, "kind": {"type": "string", "enum": ["buffer", "texture"], "description": "buffer: count elements of a layout; texture: an image of a format and a size per version, which the derive half writes as a packed array<u32> with the size in <id>_size and the draw half samples as texture_2d<f32>, or texture_2d<u32> for r32uint."}, "element": {"type": "string", "enum": ["f32", "i32", "u32", "vec2<f32>", "vec4<f32>", "vec2<u32>", "vec4<u32>", "u8", "u16"], "description": "buffer: a plain element type, or the narrow u8 or u16 packed four or two to a 32-bit word; give either element, or struct with fields."}, "struct": {"type": "string", "pattern": "^[A-Z][A-Za-z0-9_]{0,31}$", "description": "buffer: the name of the struct that the composer generates from fields."}, "fields": {"type": "array", "minItems": 1, "maxItems": 16, "items": {"type": "string", "pattern": "^[a-z_][a-z0-9_]{0,31}: (f32|i32|u32|vec2<f32>|vec4<f32>|vec2<u32>|vec4<u32>)$"}, "description": "buffer: the struct's fields in order, each written name: type."}, "count": {"type": "integer", "minimum": 1, "maximum": 134217728, "description": "buffer: the elements, at most 33,554,432 words: 134,217,728 u8 or 67,108,864 u16 elements."}, "counted": {"type": "boolean", "description": "buffer: the derive half also gets the counter <id>_count, an atomic<u32> set to zero before each run, and the valid elements are the first min(counter, count); false when absent."}, "width": {"type": "integer", "minimum": 1, "maximum": 8192, "description": "texture: the largest width of a version in pixels."}, "height": {"type": "integer", "minimum": 1, "maximum": 8192, "description": "texture: the largest height of a version in pixels."}, "format": {"type": "string", "enum": ["rgba8unorm", "r8unorm", "r32uint"], "description": "texture: rgba8unorm when absent, r8unorm (one byte a texel) or r32uint (sampled with textureLoad)."}, "texels": {"type": "integer", "minimum": 1, "maximum": 67108864, "description": "texture: the most texels of a version, width times height when absent; the largest version holds at most 128 MiB when published."}, "publish": {"type": "boolean", "description": "false: a private resource of the derive half that keeps state between runs and is never sent; true when absent, and the resource is an output that viewers and other pieces read."}, "clear": {"type": "boolean", "description": "true: zeroed before each run of the derive passes; false when absent, and the contents stay from the run before."}, "keep": {"type": "integer", "minimum": 1, "maximum": 4096, "description": "The versions that the app's step history keeps, thinned evenly over the run; 256 when absent."}}})json"},

    {"piece_derive", R"json({"type": "object", "additionalProperties": false, "required": ["passes"], "properties": {"passes": {"type": "array", "minItems": 1, "maxItems": 8, "items": {"type": "object", "additionalProperties": false, "required": ["entry"], "properties": {"entry": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]{0,62}$", "description": "The compute entry point of derive.wgsl; never main."}, "workgroups": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "integer", "minimum": 1, "maximum": 65535}, "description": "The workgroup counts x, y and z; [1, 1, 1] when absent."}, "writes": {"type": "array", "maxItems": 8, "items": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$"}, "description": "The resources that the pass changes; a published resource that no pass names is sent after the first run only."}}}, "description": "The compute passes, run in this order in one submit at a pass of the shared cadence."}, "time_ms": {"type": "number", "minimum": 0.01, "maximum": 50, "description": "The most GPU time of one run of all passes in milliseconds; 4 when absent."}, "memory_mb": {"type": "integer", "minimum": 1, "maximum": 512, "description": "The budget of the resources and the bound tensors in MiB; 64 when absent."}, "every": {"type": "integer", "minimum": 1, "maximum": 10000, "description": "The fewest steps between two runs; the shared cadence may run it less often, never more often."}}})json"},

    {"piece_draw_pass", R"json({"type": "object", "additionalProperties": false, "required": ["kind"], "properties": {"kind": {"type": "string", "enum": ["compute", "render"], "description": "compute: a dispatch that may write draw buffers; render: a draw into the host target or an offscreen target."}, "entry": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]{0,62}$", "description": "compute: the compute entry point of draw.wgsl."}, "workgroups": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "integer", "minimum": 1, "maximum": 65535}, "description": "compute: the workgroup counts; [1, 1, 1] when absent."}, "writes": {"type": "array", "maxItems": 4, "items": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$"}, "description": "compute: the draw buffers that the pass changes."}, "vertex": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]{0,62}$", "description": "render: the vertex entry point; absent for a full-screen triangle drawn by the prelude's caliper_fullscreen."}, "fragment": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]{0,62}$", "description": "render: the fragment entry point."}, "target": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "render: host, the scene panel's color, ID and depth, or an offscreen target; host when absent."}, "reads": {"type": "array", "maxItems": 4, "items": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$"}, "description": "Offscreen targets that the pass samples; an earlier pass of the frame wrote them, and the pass never reads its own target."}, "instances": {"type": ["integer", "string"], "description": "render: the instance count from 1 to 4194304, or the id of a buffer resource or buffer input whose valid count at the cursor is the count; 1 when absent."}, "vertices": {"type": "integer", "minimum": 1, "maximum": 65536, "description": "render: the vertices of one instance; 3 for a full-screen triangle, necessary with vertex."}, "indices": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "render with a vertex entry and a list topology: a published buffer resource or buffer input of u32 elements, whose valid elements the pass draws as an index buffer; each index is the vertex_index of the vertex entry, whose reads are bounds-checked. vertices then counts nothing."}, "topology": {"type": "string", "enum": ["triangle_list", "triangle_strip", "line_list", "line_strip", "point_list"], "description": "render: the topology; triangle_list when absent."}, "blend": {"type": "string", "enum": ["opaque", "alpha", "additive"], "description": "render: the color blend; opaque into host and additive into an offscreen target when absent. The ID never blends."}, "depth": {"type": "string", "enum": ["none", "test", "write"], "description": "render into host: none, a less-or-equal test, or the test with a write; write with a vertex entry and none for a full-screen triangle when absent. Offscreen targets have no depth."}, "pick": {"type": "boolean", "description": "render into host: the fragment returns CaliperPicked and writes pick::id(item) at location 1, which the pick readback reads; false when absent."}}})json"},

    {"piece_draw", R"json({"type": "object", "additionalProperties": false, "required": ["passes"], "properties": {"camera": {"type": "string", "enum": ["orbit", "pan_zoom", "none"], "description": "The host camera of the panel: orbit in 3-D, pan_zoom in 2-D, or none for clip space; orbit when absent."}, "buffers": {"type": "array", "maxItems": 4, "items": {"type": "object", "additionalProperties": false, "required": ["id", "count"], "properties": {"id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "The buffer's id, unique among the piece's inputs, resources and draw buffers."}, "element": {"type": "string", "enum": ["f32", "i32", "u32", "vec2<f32>", "vec4<f32>", "vec2<u32>", "vec4<u32>"], "description": "a plain element type; give either element, or struct with fields."}, "struct": {"type": "string", "pattern": "^[A-Z][A-Za-z0-9_]{0,31}$", "description": "the name of the struct that the composer generates from fields."}, "fields": {"type": "array", "minItems": 1, "maxItems": 16, "items": {"type": "string", "pattern": "^[a-z_][a-z0-9_]{0,31}: (f32|i32|u32|vec2<f32>|vec4<f32>|vec2<u32>|vec4<u32>)$"}, "description": "the struct's fields in order, each written name: type."}, "count": {"type": "integer", "minimum": 1, "maximum": 33554432, "description": "The elements."}, "reset": {"type": "string", "enum": ["install", "cursor", "resource"], "description": "When the buffer returns to zero besides an install and a new run: install only, also when the cursor moves by a command, a key or the transport, or also when a resource changes version; install when absent."}}}, "description": "Buffers that the draw half keeps between frames."}, "targets": {"type": "array", "maxItems": 4, "items": {"type": "object", "additionalProperties": false, "required": ["id"], "properties": {"id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "The target's id, which the draw half samples as texture_2d<f32>."}, "scale": {"type": "number", "minimum": 0.25, "maximum": 1, "description": "The size against the host target: 1, 0.5 or 0.25; 1 when absent."}, "clear": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number", "minimum": 0, "maximum": 1}, "description": "The color at the start of each frame; transparent black when absent."}}}, "description": "Offscreen rgba8unorm targets, such as a glow for a bloom."}, "passes": {"type": "array", "minItems": 1, "maxItems": 16, "items": {"$use": "piece_draw_pass"}, "description": "The passes of one frame, in this order."}, "pick": {"type": "object", "additionalProperties": false, "required": ["resource", "fields"], "properties": {"resource": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "A buffer resource with a struct."}, "fields": {"type": "array", "minItems": 1, "maxItems": 8, "items": {"type": "string", "pattern": "^[a-z_][a-z0-9_]{0,31}$"}, "description": "The struct's fields that the pick event carries, in this order; its first four integer fields are the tensor indices of the linked selection."}}, "description": "The item map: a picked item is an element index of the resource."}, "text": {"type": "object", "additionalProperties": false, "required": ["anchors"], "properties": {"anchors": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "A buffer resource with a struct; one label at each valid element."}, "position": {"type": "string", "pattern": "^[a-z_][a-z0-9_]{0,31}$", "description": "The vec4<f32> field whose xyz is the label's world position; position when absent."}, "string": {"type": "string", "pattern": "^[a-z_][a-z0-9_]{0,31}$", "description": "The u32 field that indexes strings; string when absent."}, "priority": {"type": "string", "pattern": "^[a-z_][a-z0-9_]{0,31}$", "description": "An f32 field; a label of a higher priority wins an overlap."}, "strings": {"type": "array", "maxItems": 4096, "items": {"type": "string", "maxLength": 64}, "description": "The texts of the labels."}, "strings_from": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "A published buffer with at most 8224 strings, each with at most 64 bytes of u32 fields. Its stride or first zero byte ends each UTF-8 string. Invalid UTF-8 hides the label."}, "align": {"type": "string", "enum": ["placed", "left", "right", "center"], "description": "placed: the placer chooses one of eight places around the anchor; left, right and center pin each label at its anchor with the start, the end or the middle of its text there. placed when absent."}, "hover": {"type": "object", "additionalProperties": false, "required": ["anchor"], "properties": {"anchor": {"type": "integer", "minimum": 0, "maximum": 511, "description": "The anchor index for the local hover label."}, "string_offset": {"type": "integer", "minimum": 0, "maximum": 4095, "description": "The string index offset for a local item hit. Zero when absent."}}, "description": "A local item hit selects this label's text."}, "selected": {"type": "object", "additionalProperties": false, "required": ["anchor", "param"], "properties": {"anchor": {"type": "integer", "minimum": 0, "maximum": 511, "description": "The anchor index for the selected label."}, "param": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "An i32 or u32 draw parameter with the selected item index."}, "string_offset": {"type": "integer", "minimum": 0, "maximum": 4095, "description": "The string index offset for a selected item. Zero when absent."}}, "description": "The current draw parameter selects this label from the shown item map."}}, "description": "Labels that the host draws and places."}, "frame_ms": {"type": "number", "minimum": 0.1, "maximum": 16, "description": "The most GPU time of the draw half in one frame in milliseconds; 4 when absent. Over it on 30 frames in a row, the program stops."}, "memory_mb": {"type": "integer", "minimum": 1, "maximum": 512, "description": "The budget of the resources, buffer inputs, draw buffers and targets in the viewer, in MiB; 64 when absent."}}})json"},

    {"piece", R"json({"type": "object", "additionalProperties": false, "required": ["name"], "properties": {"name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "The piece's name, at most 32 characters; its workspace folder is pieces/<name>/<version>/."}, "version": {"type": "integer", "minimum": 1, "maximum": 1000000, "description": "The piece's version; 1 when absent. A saved version never changes."}, "description": {"type": "string", "maxLength": 256, "description": "One line that says what the piece draws or publishes."}, "inputs": {"type": "array", "maxItems": 8, "items": {"$use": "piece_input"}, "description": "The typed inputs that an instance binds."}, "params": {"type": "array", "maxItems": 16, "items": {"$use": "piece_param"}, "description": "The parameters, which become controls when an instance gives no value."}, "resources": {"type": "array", "maxItems": 8, "items": {"$use": "piece_resource"}, "description": "The resources that the derive half writes; the published ones are outputs."}, "imports": {"type": "array", "maxItems": 8, "items": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}@[1-9][0-9]{0,5}$"}, "description": "Lib modules by name and version, such as colormap@1, whose items the WGSL calls as colormap::magma."}, "depends": {"type": "array", "maxItems": 8, "items": {"type": "object", "additionalProperties": false, "required": ["name", "version"], "properties": {"name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "The piece's name."}, "version": {"type": "integer", "minimum": 1, "maximum": 1000000, "description": "Its version."}}}, "description": "Pieces whose outputs feed this piece's buffer inputs, or that are drawn with it."}, "derive": {"$use": "piece_derive", "description": "The derive half, which runs in the training script; absent for a piece that only draws."}, "draw": {"$use": "piece_draw", "description": "The draw half, which runs in the viewer every frame; absent for a piece that only publishes."}, "sample": {"type": "array", "maxItems": 8, "items": {"type": "object", "additionalProperties": false, "required": ["input", "shape"], "properties": {"input": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "A tensor or buffer input of the piece."}, "shape": {"type": "array", "minItems": 1, "maxItems": 8, "items": {"type": "integer", "minimum": 1}, "description": "The sample's shape; its file is sample/<input>.bin, raw little-endian elements."}, "data": {"type": "string", "maxLength": 1398104, "description": "The file's bytes in base64, when a command writes it; at most 1 MiB for all samples."}}}, "description": "A small input for the preview."}, "hash": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$", "description": "The piece's hash over its canonical manifest and its resolved WGSL; the host computes it, and a given value must match."}, "files": {"type": "object", "additionalProperties": false, "required": [], "properties": {"common": {"type": "string", "maxLength": 65536, "description": "common.wgsl: structs, constants and functions that both halves include."}, "derive": {"type": "string", "maxLength": 65536, "description": "derive.wgsl: the derive half's compute entry points."}, "draw": {"type": "string", "maxLength": 65536, "description": "draw.wgsl: the draw half's entry points."}}, "description": "The WGSL files, with no binding and no directive; the prelude, the composer and the imports declare the rest."}}})json"},

    {"piece_pin", R"json({"type": "object", "additionalProperties": false, "required": ["name"], "properties": {"name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "The piece's name."}, "version": {"type": "integer", "minimum": 1, "maximum": 1000000, "description": "The piece's version; 1 when absent."}, "hash": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$", "description": "The piece's hash. The editor writes it when absent; a piece whose hash differs refuses the program with piece.hash_mismatch."}, "from": {"type": "string", "enum": ["library", "workspace", "inline"], "description": "The piece's source: the built-in library (the default), the workspace's pieces folder, or the spec's inline pieces."}}})json"},

    {"piece_bind", R"json({"description": "A JSON object that binds each input of the piece by name, at most 8: live:<tensor> for a tensor input, program:<id>/<resource> for a buffer input that reads another program's published buffer. An input unbound, or bound to another dtype, rank or layout, is piece.bad_binding."})json"},

    {"piece_summary", R"json({"type": "object", "additionalProperties": false, "required": ["name", "version", "from", "description", "hash", "draws", "derives", "inputs", "params", "outputs", "pinned_by"], "properties": {"name": {"type": "string", "description": "The piece's name."}, "version": {"type": "integer", "description": "Its version."}, "from": {"type": "string", "enum": ["library", "workspace", "inline"], "description": "library, workspace or inline in the open spec."}, "description": {"type": "string", "description": "Its one line."}, "hash": {"type": "string", "description": "Its hash."}, "draws": {"type": "boolean", "description": "Whether it has a draw half."}, "derives": {"type": "boolean", "description": "Whether it has a derive half."}, "inputs": {"type": "array", "items": {"type": "string"}, "description": "Its inputs' names."}, "params": {"type": "array", "items": {"type": "string"}, "description": "Its parameters' names."}, "outputs": {"type": "array", "items": {"type": "string"}, "description": "Its published resources' ids."}, "pinned_by": {"type": "array", "items": {"type": "string"}, "description": "The programs of the open spec that pin this version."}}})json"},

    {"program_status", R"json({
      "type": "object", "additionalProperties": false,
      "required": ["id", "kind", "host", "state", "every", "every_min", "last_pass_ms", "last_cost_ms", "step_ms", "overhead_share", "passes_per_s", "last_step", "runs", "memory_bytes", "memory_limit_bytes", "error", "sha256"],
      "properties": {
        "id": {"type": "string", "description": "The program's id."},
        "kind": {"type": "string", "enum": ["geometry", "pixels", "tensor", "operator", "piece"], "description": "The program's output kind, piece for an instance of a piece, or operator for the first surface or path step of a source over a live tensor, which runs in the script and has the id operator.<source>.<step index>."},
        "host": {"type": "string", "enum": ["script", "viewer", "none"], "description": "Where it runs: script for a program in the attached script, viewer for the draw half of a piece, none when nothing runs it."},
        "state": {"type": "string", "enum": ["waiting", "review", "compiled", "running", "refused", "stopped"], "description": "waiting for a host, review in a foreign spec without a review, compiled before its first pass, running, refused by a check, or stopped by a limit."},
        "every": {"type": ["integer", "null"], "description": "The steps between two passes of the program: the cadence N that all programs share, or a multiple of N for a larger limits.every; null before the first pass."},
        "every_min": {"type": ["integer", "null"], "description": "The declared limits.every; null when absent."},
        "last_pass_ms": {"type": ["number", "null"], "description": "The GPU time of the last pass or frame in milliseconds, or null."},
        "last_cost_ms": {"type": ["number", "null"], "description": "The overhead D of a pass in milliseconds: the time that a pass adds to the run's wall clock, shared by all programs; null before it is measured."},
        "step_ms": {"type": ["number", "null"], "description": "The base step time b in milliseconds: the time of a training step far from a pass; null before it is measured."},
        "overhead_share": {"type": ["number", "null"], "minimum": 0, "description": "The measured share of the training time that the passes take, D / (N b), which the cadence aims to keep at or below 0.04; null before it is measured."},
        "passes_per_s": {"type": ["number", "null"], "minimum": 0, "description": "The passes of the program a second on the run's wall clock, over its last passes; null before its second pass and while it does not run."},
        "last_step": {"type": ["integer", "null"], "description": "The script's step of the last pass, or null."},
        "runs": {"type": "integer", "minimum": 0, "description": "The count of passes."},
        "memory_bytes": {"type": "integer", "minimum": 0, "description": "The bytes of its output buffers and bound tensors."},
        "memory_limit_bytes": {"type": "integer", "minimum": 0, "description": "The budget from limits.memory_mb."},
        "error": {"type": ["object", "null"], "additionalProperties": false, "required": ["code", "message", "line", "column"], "description": "The last error, or null.",
                  "properties": {
                    "code": {"type": "string", "description": "A stable code, such as program.over_time."},
                    "message": {"type": "string", "description": "One plain sentence that says what happened."},
                    "line": {"type": "integer", "minimum": 0, "description": "The line in the source, or 0."},
                    "column": {"type": "integer", "minimum": 0, "description": "The column in the source, or 0."}}},
        "sha256": {"type": "string", "description": "The SHA-256 of the source the host runs, in hex; empty when none runs."},
        "piece": {"$use": "piece_pin", "description": "An instance: its pin; absent for another program."},
        "draw": {"type": ["object", "null"], "additionalProperties": false, "description": "An instance whose piece draws: its draw half in the viewer; absent for another program.",
                 "required": ["state", "frame_ms", "frame_limit_ms", "memory_bytes", "memory_limit_bytes", "resource_step", "error"],
                 "properties": {
                   "state": {"type": "string", "enum": ["waiting", "review", "compiled", "running", "refused", "stopped"], "description": "The draw half's state in the viewer: review in a foreign spec without a review of the piece's hash."},
                   "frame_ms": {"type": ["number", "null"], "description": "The GPU time of its last frame in milliseconds, or null."},
                   "frame_limit_ms": {"type": "number", "description": "Its draw.frame_ms."},
                   "memory_bytes": {"type": "integer", "minimum": 0, "description": "The bytes of its resources, buffer inputs, draw buffers and targets in the viewer."},
                   "memory_limit_bytes": {"type": "integer", "minimum": 0, "description": "Its draw.memory_mb in bytes."},
                   "resource_step": {"type": ["integer", "null"], "description": "The step of the newest resource version that a frame drew, or null."},
                   "error": {"type": ["object", "null"], "additionalProperties": false, "required": ["code", "message", "where"], "description": "The draw half's last error, or null.",
                             "properties": {
                               "code": {"type": "string", "description": "A stable code, such as program.draw_over_time."},
                               "message": {"type": "string", "description": "One plain sentence that names the pass, its time or memory, and its limit."},
                               "where": {"type": "string", "description": "The pass, such as draw.passes[1], or empty."}}}}}}})json"},

    {"intervention_point", R"json({
      "type": "object", "additionalProperties": false, "required": ["point", "kind", "module", "head", "type", "shape"],
      "properties": {
        "point": {"type": "string", "description": "The point's name: a module's dotted name, <module>#<h> for a head, or <block>.<probe> for a semantic probe."},
        "kind": {"type": "string", "enum": ["module", "head"], "description": "module for a module or semantic probe, head for one attention head."},
        "module": {"type": "string", "description": "The module's dotted name."},
        "head": {"type": ["integer", "null"], "description": "The head's index, or null for a module."},
        "type": {"type": "string", "description": "The module's class name, such as Linear or MultiheadAttention."},
        "shape": {"type": ["array", "null"], "items": {"type": "integer"}, "description": "The shape of the module's last output, or null before the first step."},
        "probe": {"type": "string", "enum": ["resid_pre", "mlp_in", "mlp_out", "attn_pattern"], "description": "The optional semantic form. attn_pattern is read-only."},
        "layer": {"type": "integer", "minimum": 0, "description": "The optional decoder layer index, from zero."},
        "pattern": {"type": ["string", "null"], "enum": ["softmax", "signed_value_mixing", null], "description": "The optional attention pattern form, or null for another semantic probe."}}})json"},

    {"intervention_spec", R"json({"type": "object", "additionalProperties": false, "required": ["point"], "properties": {"point": {"type": "string", "description": "The point name from intervene.points.", "minLength": 1, "maxLength": 255}, "op": {"type": "string", "description": "The legacy discriminator. Supply op or kind. Steer stays reserved.", "enum": ["ablate", "add", "scale", "clamp", "patch", "steer"]}, "kind": {"type": "string", "description": "The operation discriminator. It must equal op when both are present.", "enum": ["ablate", "add", "scale", "clamp", "patch", "steer"]}, "from_step": {"type": ["integer", "null"], "description": "The first step, or null for the next step.", "minimum": 0, "maximum": 9007199254740991}, "until_step": {"type": ["integer", "null"], "description": "The exclusive end step, or null until cleared.", "minimum": 0, "maximum": 9007199254740991}, "positions": {"type": "array", "description": "Selected token positions. Omission selects all, and an empty list selects none.", "maxItems": 32, "items": {"type": "integer", "minimum": 0, "maximum": 31}}, "vector": {"type": "array", "description": "The add vector. Its length must match the point width.", "minItems": 1, "maxItems": 16384, "items": {"type": "number"}}, "coefficient": {"type": "number", "description": "The finite coefficient for add."}, "feature": {"type": "integer", "description": "The feature for scale or clamp.", "minimum": 0, "maximum": 2147483647}, "multiplier": {"type": "number", "description": "The finite multiplier for scale."}, "value": {"type": "number", "description": "The finite activation for clamp."}, "source_prompt": {"type": "string", "description": "The clean patch source, at most 4096 UTF-8 bytes and 32 model tokens.", "minLength": 1, "maxLength": 4096}, "source_position": {"type": "integer", "description": "The one source position, or matching positions when absent.", "minimum": 0, "maximum": 31}, "dictionary": {"type": "string", "description": "The configured dictionary identifier for scale or clamp.", "minLength": 1, "maxLength": 255}}})json"},

    {"intervention", R"json({"type": "object", "additionalProperties": false, "required": ["point", "op", "from_step", "until_step", "since_step", "state"], "properties": {"point": {"type": "string", "description": "The point name from intervene.points.", "minLength": 1, "maxLength": 255}, "op": {"type": "string", "description": "The legacy discriminator. Supply op or kind. Steer stays reserved.", "enum": ["ablate", "add", "scale", "clamp", "patch", "steer"]}, "kind": {"type": "string", "description": "The operation discriminator. It must equal op when both are present.", "enum": ["ablate", "add", "scale", "clamp", "patch", "steer"]}, "from_step": {"type": ["integer", "null"], "description": "The first step, or null for the next step.", "minimum": 0, "maximum": 9007199254740991}, "until_step": {"type": ["integer", "null"], "description": "The exclusive end step, or null until cleared.", "minimum": 0, "maximum": 9007199254740991}, "positions": {"type": "array", "description": "Selected token positions. Omission selects all, and an empty list selects none.", "maxItems": 32, "items": {"type": "integer", "minimum": 0, "maximum": 31}}, "vector": {"type": "array", "description": "The add vector. Its length must match the point width.", "minItems": 1, "maxItems": 16384, "items": {"type": "number"}}, "coefficient": {"type": "number", "description": "The finite coefficient for add."}, "feature": {"type": "integer", "description": "The feature for scale or clamp.", "minimum": 0, "maximum": 2147483647}, "multiplier": {"type": "number", "description": "The finite multiplier for scale."}, "value": {"type": "number", "description": "The finite activation for clamp."}, "source_prompt": {"type": "string", "description": "The clean patch source, at most 4096 UTF-8 bytes and 32 model tokens.", "minLength": 1, "maxLength": 4096}, "source_position": {"type": "integer", "description": "The one source position, or matching positions when absent.", "minimum": 0, "maximum": 31}, "dictionary": {"type": "string", "description": "The configured dictionary identifier for scale or clamp.", "minLength": 1, "maxLength": 255}, "since_step": {"type": ["integer", "null"], "description": "The step the script applied it at, or null while it is pending."}, "state": {"type": "string", "enum": ["pending", "active", "ended"], "description": "pending until the script's next step, active, or ended after until_step."}, "error": {"type": "string", "description": "The execution refusal, when present.", "maxLength": 1024}, "error_code": {"type": "string", "description": "The structured refusal code, when present.", "maxLength": 63}}})json"},

    // Pieces of the instrument commands (design/instrument-interface.md, Commands).
    {"orbit", R"json({
      "type": "object", "additionalProperties": false, "required": ["target", "distance", "azimuth", "elevation"],
      "properties": {
        "target": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}, "description": "The point the orbit looks at."},
        "distance": {"type": "number", "minimum": 0, "description": "The distance from the target, above 0."},
        "azimuth": {"type": "number", "description": "The orbit's azimuth in degrees, read against up."},
        "elevation": {"type": "number", "minimum": -89, "maximum": 89, "description": "The orbit's elevation in degrees, from -89 to 89."}}})json"},

    {"camera_view", R"json({
      "type": "object", "additionalProperties": false, "required": ["name", "target", "distance", "azimuth", "elevation"],
      "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 64, "description": "The saved view's name, unique in the panel."},
        "target": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}, "description": "The point the view looks at."},
        "distance": {"type": "number", "minimum": 0, "description": "The distance from the target, above 0."},
        "azimuth": {"type": "number", "description": "The view's azimuth in degrees, read against up."},
        "elevation": {"type": "number", "minimum": -89, "maximum": 89, "description": "The view's elevation in degrees, from -89 to 89."}}})json"},

    {"cursor_state", R"json({
      "type": "object", "additionalProperties": false, "required": ["step", "first", "latest", "live", "playing", "speed"],
      "properties": {
        "step": {"type": "integer", "description": "The step that the panels show."},
        "first": {"type": "integer", "description": "The first step that a source keeps."},
        "latest": {"type": "integer", "description": "The newest step that a source keeps."},
        "live": {"type": "boolean", "description": "Whether the cursor follows the newest step."},
        "playing": {"type": "boolean", "description": "Whether the cursor plays forward."},
        "speed": {"type": "number", "description": "The play speed in steps per second."}}})json"},

    {"cursor_source", R"json({
      "type": "object", "additionalProperties": false, "required": ["id", "first", "last", "steps", "requested", "bytes", "note"],
      "properties": {
        "id": {"type": "string", "description": "The source's id."},
        "first": {"type": "integer", "description": "The first step that the source keeps."},
        "last": {"type": "integer", "description": "The last step that the source keeps."},
        "steps": {"type": "integer", "description": "How many steps the source keeps."},
        "requested": {"type": "integer", "description": "How many steps the spec asks the source to keep."},
        "bytes": {"type": "integer", "description": "The size of the kept steps in bytes."},
        "note": {"type": ["string", "null"], "description": "Why the source keeps fewer steps than it asks for, or null."}}})json"},

    // What a spec edit returns besides its own fields.
    {"warnings", R"json({"type": "array", "items": {"$use": "diagnostic"}, "description": "What the validator warns about, such as spec.not_portable for a source that cannot run in the browser viewer."})json"},

    {"diff", R"json({"type": "string", "description": "The change as a unified diff of the spec's canonical TOML text, before and after; empty when the text does not change."})json"},

    {"applied", R"json({"type": "boolean", "description": "Whether the change was applied: to the open spec, or to the preview when the call names one; false for a dry run."})json"},

    {"preview_id", R"json({"type": "string", "pattern": "^[a-z][a-z0-9_-]{0,63}$", "minLength": 1, "maxLength": 64, "description": "The preview's id, from preview.begin."})json"},

    {"dry_run", R"json({"type": "boolean", "default": false, "description": "Whether to check the change and return its diff without applying it, so that it can be shown for approval first; false when absent."})json"},

    {"loader_outputs", R"json({"type": "array", "items": {"type": "string"}})json"},
};

struct Row {
    const char* name;
    bool mutates;
    bool destructive;
    bool frame_thread;
    const char* summary;
    const char* args;
    const char* result;
    const char* example;
};

constexpr const char* kNoArgs = R"json({"type": "object", "additionalProperties": false, "required": [], "properties": {}})json";

const Row kRows[] = {
    // app
    {"app.info", false, false, false,
     "Describes the running Caliper app: its version, the ABI epoch its applets must be built for, the services it "
     "offers, and the versions of the SDK, libtorch and its interface libraries. It changes nothing.",
     kNoArgs,
     R"json({"type": "object", "additionalProperties": false, "required": ["abi_epoch", "host", "libtorch", "sdk", "services", "ui_stack"],
      "properties": {
        "abi_epoch": {"type": "integer", "description": "The ABI epoch that compiled applets must be built for."},
        "host": {"type": "string", "description": "The app's version."},
        "libtorch": {"type": ["string", "null"], "description": "The libtorch version, or null when the app has none."},
        "sdk": {"type": ["string", "null"], "description": "The Caliper SDK version, or null when it is not known."},
        "services": {"type": "array", "items": {"type": "string"}, "description": "The service ids the app offers to applets, such as caliper.data.v1."},
        "ui_stack": {"type": ["string", "null"], "description": "A fingerprint of the interface libraries, or null."}}})json",
     "{}"},
    {"app.applets", false, false, true,
     "Lists the applets the app found, as caliper list shows them: each applet's manifest id, its version, its "
     "kind (library or spec), its folder, its status (READY, ACTIVE for the one open now, REFUSED, SHADOWED) and "
     "the reason when it cannot run. "
     "The id is what app.launch takes. It changes nothing.",
     kNoArgs,
     R"json({"type": "object", "additionalProperties": false, "required": ["applets"],
      "properties": {"applets": {"type": "array", "description": "The applets, in the app's search order.",
        "items": {"type": "object", "additionalProperties": false, "required": ["folder", "id", "reason", "status", "version"],
          "properties": {
            "folder": {"type": "string", "description": "The folder the applet was found in."},
            "kind": {"type": "string", "enum": ["library", "spec"], "description": "library for a compiled applet, spec for a spec applet."},
            "id": {"type": "string", "description": "The applet's manifest id."},
            "reason": {"type": "string", "description": "Why the applet cannot run; empty for a READY applet."},
            "status": {"type": "string", "description": "READY, ACTIVE, REFUSED or SHADOWED, as caliper list prints it."},
            "version": {"type": "string", "description": "The applet's version."}}}}}})json",
     "{}"},
    {"app.launch", false, false, true,
     "Opens the page of one applet, as a click on its row on the app's landing screen does; an attached live "
     "session stays attached. The id is an applet's manifest id from app.applets. A compiled applet's page replaces "
     "the page shown at once. A dashboard of the workspace opens with an applet.open from your origin, and the page "
     "changes only when that has run: when it would replace the open spec's unsaved changes it waits for the "
     "person's approval, the status is PENDING_APPROVAL rather than ACTIVE, its note says how to check whether it "
     "opened, and if the person denies it nothing changes.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {"id": {"type": "string", "pattern": "^[A-Za-z0-9._-]{1,128}$", "minLength": 1, "maxLength": 128,
                            "description": "The applet's manifest id, such as com.example.my-viewer, from app.applets."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["already_open", "id", "status", "version"],
      "properties": {
        "already_open": {"type": "boolean", "description": "Whether the applet's page was already open."},
        "id": {"type": "string", "description": "The applet's manifest id."},
        "status": {"type": "string", "description": "The applet's status after the launch: ACTIVE when it opened; PENDING_APPROVAL when its open waits for the person's approval, because it would replace the open spec's unsaved changes; PENDING when the app cannot tell yet whether it will."},
        "version": {"type": "string", "description": "The applet's version."},
        "note": {"type": "string", "description": "With PENDING_APPROVAL or PENDING: what the launch waits for, and that spec.get names the open spec once it has opened."}}})json",
     R"json({"id": "com.example.my-viewer"})json"},

    // catalog
    {"catalog.list", false, false, false,
     "Lists the sources in the catalog: every table, tensor, live view, feed channel and metric that a spec can "
     "bind, with its schema, shape and rate. A source id is <kind>:<name>, such as table:run42_metrics or "
     "live:attn.weights; a spec source names it in its catalog field. Filter by kind or by the start of the name. "
     "It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "kind": {"type": "string", "enum": ["table", "tensor", "live", "feed", "metric", "run"], "description": "Only sources of this kind."},
        "prefix": {"type": "string", "maxLength": 256, "description": "Only sources whose name starts with this text, case-sensitive."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 100, "description": "The most entries to return, from 1 to 1000; 100 when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["entries", "truncated", "warnings"],
      "properties": {
        "entries": {"type": "array", "items": {"$use": "catalog_entry"}, "description": "The matching sources, sorted by id."},
        "truncated": {"type": "boolean", "description": "Whether more sources matched than limit."},
        "warnings": {"type": "array", "items": {"type": "string"}, "description": "Notes about the catalog itself, such as an origin that failed to answer or an id offered twice."}}})json",
     R"json({"kind": "tensor", "limit": 50})json"},
    {"catalog.describe", false, false, false,
     "Gives one catalog source in full: its kind, where it comes from, its columns and row count for a table, feed "
     "or metric, its dtype and shape for a tensor or live view, its rate, its size, whether it can run in the "
     "browser viewer, and any warnings. The id comes from catalog.list.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {"id": {"$use": "catalog_id", "description": "The source id, <kind>:<name>, from catalog.list."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["entry"],
      "properties": {"entry": {"$use": "catalog_entry", "description": "The source."}}})json",
     R"json({"id": "live:attn.weights"})json"},
    {"catalog.sample", false, false, false,
     "Shows a little of a source's data, to learn what it holds before binding it. A table, feed or metric gives up "
     "to limit rows (20 by default). A tensor or live view gives its statistics (min, max, mean, standard deviation "
     "and the count of non-finite values) and a box of its values, at most 64 per axis and 4096 in all, placed with "
     "start and count. The id comes from catalog.list. It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {
        "id": {"$use": "catalog_id", "description": "The source id, <kind>:<name>, from catalog.list."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 20, "description": "For a tabular source, the most rows to give, from 1 to 1000; 20 when absent."},
        "start": {"type": "array", "maxItems": 8, "items": {"type": "integer", "minimum": 0}, "description": "For a tensor-shaped source, the box's first index on each axis, one per axis; zeros when absent."},
        "count": {"type": "array", "maxItems": 8, "items": {"type": "integer", "minimum": 0, "maximum": 64}, "description": "For a tensor-shaped source, the box's size on each axis, one per axis, each at most 64 and at most 4096 values in all; as many as fit when absent."}}})json",
     R"json({"$use": "catalog_sample"})json",
     R"json({"id": "tensor:attn_weights", "start": [0, 0, 0, 0], "count": [1, 1, 8, 8]})json"},

    // data
    {"data.query", false, false, false,
     "Runs one read-only SQL SELECT, in DuckDB's dialect, over the workspace's tables, which are the table: sources "
     "of the catalog named without the table: prefix. It returns at most limit rows (1000 by default) and stops "
     "after timeout_ms (5000 by default). Use it to explore data or to try the SQL of a spec source. It changes "
     "nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": ["sql"],
      "properties": {
        "sql": {"type": "string", "minLength": 1, "maxLength": 65536, "description": "One SELECT statement, such as SELECT step, loss FROM run42_metrics ORDER BY step."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100000, "default": 1000, "description": "The most rows to return, from 1 to 100000; 1000 when absent."},
        "timeout_ms": {"type": "integer", "minimum": 100, "maximum": 30000, "default": 5000, "description": "The time limit in milliseconds, from 100 to 30000; 5000 when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["columns", "rows", "truncated", "elapsed_ms"],
      "properties": {
        "columns": {"type": "array", "items": {"$use": "column"}, "description": "The result's columns, in order."},
        "rows": {"type": "array", "items": {"type": "array", "items": {"$use": "cell"}}, "description": "The rows, each a list of cells in column order. NaN and infinities are the strings nan, inf and -inf; dates are ISO 8601 strings; nested values are their JSON text."},
        "truncated": {"type": "boolean", "description": "Whether the query had more rows than limit, or more than 8 MiB of them."},
        "elapsed_ms": {"type": "number", "description": "How long the query took, in milliseconds."}}})json",
     R"json({"sql": "SELECT step, loss FROM run42_metrics ORDER BY step", "limit": 100})json"},
    {"data.import", true, false, false,
     "Creates a loader for each file path given, choosing the reader from the file's extension (csv, tsv, parquet, "
     "json, jsonl, arrow, npy, npz, safetensors, hdf5, png, jpg, wav), and runs the new loaders unless run is false, "
     "so that the files' contents appear in the catalog as table: and tensor: sources. Paths are relative to the "
     "workspace; a path outside it is accepted only from the person's own drop or file dialog. Pickle files are "
     "refused. Undo deletes the loaders it created and their outputs.",
     R"json({"type": "object", "additionalProperties": false, "required": ["paths"],
      "properties": {
        "paths": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"type": "string", "minLength": 1, "maxLength": 4096}, "description": "The files to import, relative to the workspace, such as data/run42/metrics.parquet."},
        "run": {"type": "boolean", "default": true, "description": "Whether to run the new loaders at once; true when absent. A Python loader is never run by an import."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["loaders", "refused"],
      "properties": {
        "loaders": {"type": "array", "description": "One entry for each loader created.",
          "items": {"type": "object", "additionalProperties": false, "required": ["id", "path", "reader", "outputs", "ran"],
            "properties": {
              "id": {"type": "string", "description": "The new loader's id, for the loader commands."},
              "path": {"type": "string", "description": "The file it reads."},
              "reader": {"type": "string", "description": "The reader chosen, such as parquet or safetensors."},
              "outputs": {"$use": "loader_outputs", "description": "The catalog ids of the sources it made, such as table:metrics."},
              "ran": {"type": "boolean", "description": "Whether it ran."}}}},
        "refused": {"type": "array", "description": "One entry for each path that was refused.",
          "items": {"type": "object", "additionalProperties": false, "required": ["path", "code", "message"],
            "properties": {
              "path": {"type": "string", "description": "The path refused."},
              "code": {"type": "string", "description": "Why, such as loader.pickle_refused or loader.path_refused."},
              "message": {"type": "string", "description": "One plain sentence, naming the safe route when there is one."}}}}}})json",
     R"json({"paths": ["data/run42/metrics.parquet", "checkpoints/model.safetensors"]})json"},

    // loader
    {"loader.list", false, false, false,
     "Lists the workspace's saved loaders. A loader is a saved recipe that turns files into catalog sources: a file "
     "or glob, a reader, and optional SQL or tensor transforms. Each has an id, which the other loader commands "
     "take. It changes nothing.",
     kNoArgs,
     R"json({"type": "object", "additionalProperties": false, "required": ["loaders"],
      "properties": {"loaders": {"type": "array", "description": "The loaders, sorted by id.",
        "items": {"type": "object", "additionalProperties": false, "required": ["id", "title", "reader", "path", "status", "outputs", "updated_at"],
          "properties": {
            "id": {"type": "string", "description": "The loader's id."},
            "title": {"type": "string", "description": "The loader's title."},
            "reader": {"type": "string", "description": "The reader it uses, such as csv or npz."},
            "path": {"type": "string", "description": "The file or glob it reads, relative to the workspace."},
            "status": {"type": "string", "enum": ["never_run", "ok", "failed"], "description": "How its last run went."},
            "outputs": {"$use": "loader_outputs", "description": "The catalog ids of the sources it made."},
            "updated_at": {"type": ["string", "null"], "description": "When its recipe last changed, in UTC ISO 8601."}}}}}})json",
     "{}"},
    {"loader.describe", false, false, false,
     "Gives one loader in full: its TOML recipe, its reader and input path, how its last run went and the catalog "
     "sources it made. The id comes from loader.list.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {"id": {"$use": "id", "description": "The loader's id, from loader.list."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "title", "text", "reader", "path", "status", "last_run", "outputs"],
      "properties": {
        "id": {"type": "string", "description": "The loader's id."},
        "title": {"type": "string", "description": "The loader's title."},
        "text": {"type": "string", "description": "The recipe, as the TOML text saved in loaders/<id>.loader.toml."},
        "reader": {"type": "string", "description": "The reader it uses."},
        "path": {"type": "string", "description": "The file or glob it reads."},
        "status": {"type": "string", "enum": ["never_run", "ok", "failed"], "description": "How its last run went."},
        "last_run": {"type": ["object", "null"], "additionalProperties": false, "required": ["at", "ok", "error", "elapsed_ms"],
                     "description": "Its last run, or null when it never ran.",
                     "properties": {
                       "at": {"type": "string", "description": "When the run ended, in UTC ISO 8601."},
                       "ok": {"type": "boolean", "description": "Whether it succeeded."},
                       "error": {"type": ["string", "null"], "description": "The error code of a failed run, or null."},
                       "elapsed_ms": {"type": ["number", "null"], "description": "How long it took, in milliseconds, or null when it is not known, as after the workspace was reopened."}}},
        "outputs": {"$use": "loader_outputs", "description": "The catalog ids of the sources it made."}}})json",
     R"json({"id": "run42-metrics"})json"},
    {"loader.create", true, false, false,
     "Saves a new loader from its TOML recipe, as loaders/<id>.loader.toml in the workspace, and runs it when run is "
     "true. The recipe holds caliper_loader = 0; a [loader] table with id and title; an [input] table with path "
     "(a file or glob relative to the workspace) and an optional reader; optional [reader] options; and one or more "
     "[[output]] tables, each with a name and an optional SQL SELECT over the relation input, or a tensor to "
     "select. It fails when a loader with that id exists. Undo deletes it.",
     R"json({"type": "object", "additionalProperties": false, "required": ["text"],
      "properties": {
        "text": {"type": "string", "minLength": 1, "maxLength": 65536, "description": "The recipe as TOML text."},
        "run": {"type": "boolean", "default": false, "description": "Whether to run the loader at once; false when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "outputs"],
      "properties": {
        "id": {"type": "string", "description": "The new loader's id, from its recipe."},
        "outputs": {"$use": "loader_outputs", "description": "The catalog ids of the sources it made; empty when it did not run."}}})json",
     R"json({"text": "caliper_loader = 0\n[loader]\nid = \"run42-metrics\"\ntitle = \"Run 42 metrics\"\n[input]\npath = \"data/run42/*.parquet\"\n[[output]]\nname = \"run42_metrics\"\n", "run": true})json"},
    {"loader.update", true, true, false,
     "Replaces the recipe of an existing loader with new TOML text, whose [loader] id must stay the same, and runs "
     "it when run is true. It overwrites the saved recipe, so an agent's call needs the person's approval. Undo "
     "restores the previous recipe; the tables the new recipe made stay until the loader runs again. The id comes "
     "from loader.list.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "text"],
      "properties": {
        "id": {"$use": "id", "description": "The loader's id, from loader.list."},
        "text": {"type": "string", "minLength": 1, "maxLength": 65536, "description": "The new recipe as TOML text."},
        "run": {"type": "boolean", "default": false, "description": "Whether to run the loader at once; false when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "outputs"],
      "properties": {
        "id": {"type": "string", "description": "The loader's id."},
        "outputs": {"$use": "loader_outputs", "description": "The catalog ids of the sources it made; empty when it did not run."}}})json",
     R"json({"id": "run42-metrics", "text": "caliper_loader = 0\n[loader]\nid = \"run42-metrics\"\ntitle = \"Run 42\"\n[input]\npath = \"data/run42/*.parquet\"\n[[output]]\nname = \"run42_metrics\"\nsql = \"SELECT step, loss FROM input WHERE step > 0\"\n"})json"},
    {"loader.run", true, true, false,
     "Runs a loader: reads its input files and rebuilds its outputs, which replaces the tables and tensors its "
     "previous run made. A Python loader runs as a separate process. This cannot be undone, and an agent's call "
     "needs the person's approval. The id comes from loader.list.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {"id": {"$use": "id", "description": "The loader's id, from loader.list."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "outputs", "elapsed_ms"],
      "properties": {
        "id": {"type": "string", "description": "The loader's id."},
        "outputs": {"type": "array", "description": "The sources the run made.",
          "items": {"type": "object", "additionalProperties": false, "required": ["id", "rows", "shape"],
            "properties": {
              "id": {"type": "string", "description": "The source's catalog id."},
              "rows": {"type": ["integer", "null"], "description": "For a table, its row count; null for a tensor."},
              "shape": {"type": ["array", "null"], "items": {"type": "integer"}, "description": "For a tensor, its shape; null for a table."}}}},
        "elapsed_ms": {"type": "number", "description": "How long the run took, in milliseconds."}}})json",
     R"json({"id": "run42-metrics"})json"},
    {"loader.delete", true, true, false,
     "Deletes a saved loader and, unless keep_outputs is true, the catalog sources it made. An agent's call needs "
     "the person's approval. Undo restores the recipe; its outputs come back when it runs again. The id comes from "
     "loader.list.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {
        "id": {"$use": "id", "description": "The loader's id, from loader.list."},
        "keep_outputs": {"type": "boolean", "default": false, "description": "Whether to keep the sources it made; false when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "removed"],
      "properties": {
        "id": {"type": "string", "description": "The deleted loader's id."},
        "removed": {"$use": "loader_outputs", "description": "The catalog ids of the sources removed with it."}}})json",
     R"json({"id": "run42-metrics"})json"},

    // spec
    {"spec.new", true, false, true,
     "Starts a new, empty spec and makes it the open spec: the one the builder shows and that the source, panel and "
     "layout commands edit. A spec is the text that describes an applet built from data: its sources, its panels "
     "and their layout. The id names the applet, in lower case with digits, - and _. Undo returns to the spec that "
     "was open before.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {
        "id": {"$use": "id", "description": "The applet's id, such as attention-entropy."},
        "title": {"type": "string", "minLength": 1, "maxLength": 256, "description": "The applet's title for a person."},
        "description": {"type": "string", "maxLength": 4096, "description": "What the applet shows, in a sentence or two."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "text"],
      "properties": {
        "id": {"type": "string", "description": "The applet's id."},
        "text": {"type": "string", "description": "The new spec's TOML text."}}})json",
     R"json({"id": "attention-entropy", "title": "Attention entropy", "description": "Entropy of each attention head while the model trains."})json"},
    {"spec.open", true, false, true,
     "Opens a spec file of the workspace as the open spec. The path is relative to the workspace and ends in "
     ".spec.toml, such as applets/attention-entropy/attention-entropy.spec.toml. To open a saved applet by its id, "
     "use applet.open. Undo returns to the spec that was open before, with its unsaved changes.",
     R"json({"type": "object", "additionalProperties": false, "required": ["path"],
      "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 1024, "pattern": "\\.spec\\.toml$", "description": "The spec file, relative to the workspace."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "path", "warnings"],
      "properties": {
        "id": {"type": "string", "description": "The applet id of the spec opened."},
        "path": {"type": "string", "description": "The file opened, relative to the workspace."},
        "warnings": {"type": "array", "items": {"$use": "diagnostic"}, "description": "What the validator warns about, such as a source with no wire form."}}})json",
     R"json({"path": "applets/attention-entropy/attention-entropy.spec.toml"})json"},
    {"spec.get", false, false, true,
     "Returns the open spec as its TOML text and as its canonical JSON form, with its applet id, the file it was "
     "saved to or opened from, and whether it has unsaved changes. With applet, it returns that saved spec applet's "
     "spec instead, read from its files without opening it. When no spec is open and no applet is named, it answers "
     "open false and nothing else, which is not an error: make one with spec.new or open one with spec.open or "
     "applet.open. It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": [],
      "properties": {"applet": {"$use": "id", "description": "A saved spec applet's id, from applet.list; the open spec when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["open"],
      "properties": {
        "open": {"type": "boolean", "description": "True with a spec in the result, the open one or the named applet's; false when no applet is named and no spec is open, and then no other field is given."},
        "id": {"type": "string", "description": "The spec's applet id."},
        "path": {"type": ["string", "null"], "description": "Its file, relative to the workspace, or null when it was never saved."},
        "dirty": {"type": "boolean", "description": "Whether it has changes that are not saved."},
        "open_in_app": {"type": "boolean", "description": "With applet: whether the app has that applet open, so that its live sources, values and kept steps reach panel.data; a saved applet that is not open shows no live data."},
        "text": {"type": "string", "description": "The spec as TOML text, in canonical form."},
        "json": {"type": "object", "additionalProperties": true, "description": "The spec's canonical JSON form: the TOML tables as JSON objects, as the spec format's JSON Schema describes them."}}})json",
     "{}"},
    {"spec.set_text", true, false, true,
     "Replaces the whole open spec with new TOML text after checking it; text with an error is refused, and the "
     "open spec stays as it was. Use spec.validate first to see every error. Undo restores the previous text. The result holds the change as a diff; with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["text"],
      "properties": {
        "text": {"type": "string", "minLength": 1, "maxLength": 262144, "description": "The whole spec as TOML text, with caliper_spec = 0, an [applet] table, [[source]] and [[panel]] tables and a [layout] table."},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "warnings", "diff", "applied"],
      "properties": {
        "id": {"type": "string", "description": "The spec's applet id."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"text": "caliper_spec = 0\n[applet]\nid = \"loss\"\ntitle = \"Loss\"\n[[source]]\nid = \"loss\"\nsql = \"SELECT step, loss FROM run42_metrics ORDER BY step\"\n[[panel]]\nid = \"loss\"\nprimitive = \"series\"\nbind = { x = \"loss.step\", y = [\"loss.loss\"] }\n[layout]\nchildren = [\"loss\"]\n"})json"},
    {"spec.validate", false, false, true,
     "Checks a spec and lists its errors and warnings, each with a code, a message and where it is. It checks the "
     "given text, or the open spec when no text is given. An invalid spec is reported in the result, not as an "
     "error. It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": [],
      "properties": {"text": {"type": "string", "minLength": 1, "maxLength": 262144, "description": "A spec as TOML text; the open spec when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["valid", "errors", "warnings"],
      "properties": {
        "valid": {"type": "boolean", "description": "Whether the spec has no errors."},
        "errors": {"type": "array", "items": {"$use": "diagnostic"}, "description": "The errors, in the order of the text."},
        "warnings": {"type": "array", "items": {"$use": "diagnostic"}, "description": "The warnings, such as spec.not_portable for a source that cannot run in the browser viewer."}}})json",
     "{}"},

    // source
    {"source.add", true, false, true,
     "Adds a source to the open spec. A spec source gives data a name that panels bind. It has exactly one of "
     "catalog (a catalog source id from catalog.list, such as live:attn.weights), sql (one read-only SELECT over the "
     "workspace's tables) or from (the id of another source of this spec), and an optional list of tensor "
     "transforms (slice, reduce, normalise) for tensor-shaped data. Undo removes it. The result holds the change as a diff; with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {
        "id": {"$use": "id", "description": "The source's id in this spec, which panels bind."},
        "catalog": {"$use": "catalog_id", "maxLength": 256, "description": "A catalog source id, from catalog.list, such as live:attn.weights or table:run42_metrics."},
        "sql": {"type": "string", "minLength": 1, "maxLength": 65536, "description": "One read-only SELECT (DuckDB SQL) over the workspace's tables, named as in data.query."},
        "from": {"$use": "id", "description": "The id of another source of this spec that this one derives from, usually through transforms."},
        "transform": {"type": "array", "maxItems": 16, "items": {"$use": "transform"}, "description": "Up to 16 tensor transforms applied in order; only for tensor-shaped data."},
        "history": {"type": "integer", "minimum": 0, "maximum": 65536, "description": "The steps the app keeps for a source whose root is a live tensor, from 0 to 65536; 64 when absent for such a source without from."},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "warnings", "diff", "applied"],
      "properties": {
        "id": {"type": "string", "description": "The source's id."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"id": "entropy", "from": "attn", "transform": [{"op": "reduce", "axis": 3, "fn": "entropy"}, {"op": "reduce", "axis": 2, "fn": "mean"}]})json"},
    {"source.patch", true, false, true,
     "Changes a source of the open spec with a JSON merge patch (RFC 7386): the keys given replace the source's "
     "own, and a key set to null is removed. The id cannot change. The id comes from spec.get. Undo restores the "
     "source. The result holds the change as a diff; with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "patch"],
      "properties": {
        "id": {"$use": "id", "description": "The source's id in the open spec."},
        "patch": {"$patch": "source", "description": "The keys to change: catalog, sql, from, transform or history; null removes one."},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["source", "warnings", "diff", "applied"],
      "properties": {
        "source": {"$use": "source", "description": "The source as it is now, or would be after a dry run."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"id": "entropy", "patch": {"transform": [{"op": "reduce", "axis": 3, "fn": "entropy"}]}})json"},
    {"source.remove", true, false, true,
     "Removes a source from the open spec. A source that a panel binds, or that another source derives from, cannot "
     "be removed until they change. The id comes from spec.get. Undo restores it. The result holds the change as a diff; with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {
        "id": {"$use": "id", "description": "The source's id in the open spec."},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "warnings", "diff", "applied"],
      "properties": {
        "id": {"type": "string", "description": "The removed source's id."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"id": "loss"})json"},

    // panel
    {"panel.create", true, false, true,
     "Adds a panel to the open spec. A panel draws one primitive from the sources it binds: series (lines), heatmap "
     "(a colour-mapped grid), points (a 2-D or 3-D cloud), histogram, table, image or text. bind maps each slot of "
     "the primitive to a spec source id, or to <source>.<column> for one column of a tabular source. place puts it "
     "beside, above, below or in tabs with another panel; without place, the panel goes to the right of the whole "
     "layout. Undo removes it. The result holds the change as a diff; with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "primitive"],
      "properties": {
        "id": {"$use": "id", "description": "The new panel's id."},
        "primitive": {"type": "string", "enum": ["series", "heatmap", "points", "histogram", "table", "image", "text", "scene"], "description": "What the panel draws. Required slots: series y; heatmap values; points x and y, or positions; histogram values; table rows; image values; text none; scene scene (spec version 1)."},
        "title": {"type": "string", "minLength": 1, "maxLength": 256, "description": "The panel's title; the id when absent."},
        "bind": {"$use": "bind", "description": "The bindings by slot: a spec source id, or <source>.<column> for one column of a tabular source."},
        "entity": {"type": "array", "maxItems": 256, "items": {"description": "An entity table, a JSON object."}, "description": "scene: the entities of the panel's scene, as a list of entity tables."},
        "annotation": {"type": "array", "maxItems": 256, "items": {"description": "An annotation table, a JSON object."}, "description": "The annotations of the panel, as a list of annotation tables."},
        "options": {"$use": "options", "description": "The primitive's options."},
        "place": {"type": "object", "additionalProperties": false, "required": ["target", "where"],
                  "description": "Where to put the panel, next to another panel of the layout.",
                  "properties": {
                    "target": {"$use": "id", "description": "The id of a panel already in the layout."},
                    "where": {"type": "string", "enum": ["beside", "above", "below", "tabbed"], "description": "beside puts it to the right of the target; tabbed shows both in tabs."},
                    "ratio": {"type": "number", "minimum": 0.05, "maximum": 0.95, "default": 0.5, "description": "The share of the space the new panel gets; 0.5 when absent."}}},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "warnings", "diff", "applied"],
      "properties": {
        "id": {"type": "string", "description": "The new panel's id."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"id": "heat", "primitive": "heatmap", "title": "Entropy per layer and head", "bind": {"values": "entropy"}, "options": {"colormap": "viridis", "x_label": "Head", "y_label": "Layer"}})json"},
    {"panel.patch", true, false, true,
     "Changes a panel of the open spec with a JSON merge patch (RFC 7386) of its title, primitive, bindings and "
     "options: the keys given replace the panel's own, a key set to null is removed, and bind and options merge key "
     "by key. The id cannot change. The id comes from spec.get. Undo restores the panel. The result holds the change as a diff; with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "patch"],
      "properties": {
        "id": {"$use": "id", "description": "The panel's id in the open spec."},
        "patch": {"type": "object", "additionalProperties": false, "required": [],
                  "description": "The keys to change; null removes one.",
                  "properties": {
                    "title": {"type": ["string", "null"], "minLength": 1, "maxLength": 256, "description": "The panel's title."},
                    "primitive": {"type": "string", "enum": ["series", "heatmap", "points", "histogram", "table", "image", "text", "scene"], "description": "What the panel draws."},
                    "bind": {"$patch": "bind", "description": "Slots to bind, or null to unbind one."},
                    "options": {"$patch": "options", "description": "Options to set, or null to clear one."},
                    "entity": {"type": ["array", "null"], "maxItems": 256, "items": {"description": "An entity table, a JSON object."}, "description": "scene: the entities of the scene; the list replaces the panel's own, and null removes them."},
                    "annotation": {"type": ["array", "null"], "maxItems": 256, "items": {"description": "An annotation table, a JSON object."}, "description": "The annotations of the panel; the list replaces the panel's own, and null removes them."}}},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["panel", "warnings", "diff", "applied"],
      "properties": {
        "panel": {"$use": "panel", "description": "The panel as it is now, or would be after a dry run."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"id": "heat", "patch": {"title": "Entropy", "options": {"colormap": "magma", "y_label": null}}})json"},
    {"panel.remove", true, false, true,
     "Removes a panel from the open spec and from its layout. The id comes from spec.get. Undo restores it in its "
     "place. The result holds the change as a diff; with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {
        "id": {"$use": "id", "description": "The panel's id in the open spec."},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "warnings", "diff", "applied"],
      "properties": {
        "id": {"type": "string", "description": "The removed panel's id."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"id": "heat"})json"},
    {"panel.render", false, false, true,
     "Draws one panel of the open spec off screen and returns it as a PNG image, so that you can check what it "
     "looks like. width and height are in pixels, 800 by 500 by default. The id comes from spec.get. It changes "
     "nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {
        "id": {"$use": "id", "description": "The panel's id in the open spec."},
        "width": {"type": "integer", "minimum": 64, "maximum": 4096, "default": 800, "description": "The image's width in pixels; 800 when absent."},
        "height": {"type": "integer", "minimum": 64, "maximum": 4096, "default": 500, "description": "The image's height in pixels; 500 when absent."}}})json",
     R"json({"$use": "png"})json",
     R"json({"id": "heat", "width": 640, "height": 400})json"},
    {"panel.data", false, false, true,
     "Returns the data of one panel as the browser viewer draws it: the panel's data after its sources' transforms, "
     "within the bounds of what a panel can show (long lines as a min/max envelope, large grids and clouds sampled "
     "at a stride, a histogram's bin counts, the first 1,000 rows of a table), with a version. Pass the version you "
     "have as known, and the answer is unchanged, with no data, until the panel's sources, the spec or the catalog "
     "change. It reads a saved spec applet with applet, else the open spec. It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": ["panel"],
      "properties": {
        "panel": {"$use": "id", "description": "The panel's id in the spec."},
        "applet": {"$use": "id", "description": "A saved spec applet's id, from applet.list; the open spec when absent."},
        "known": {"type": "string", "maxLength": 64, "description": "The version of the data you have, from an earlier answer."}}})json",
     R"json({"type": "object", "additionalProperties": false,
      "required": ["panel", "primitive", "version", "unchanged", "error", "state", "bounds", "data"],
      "properties": {
        "panel": {"type": "string", "description": "The panel's id."},
        "primitive": {"type": "string", "description": "The panel's primitive, such as series or table."},
        "version": {"type": "string", "description": "The version of this data: 16 hex digits that change when the panel's sources, the spec or the catalog change."},
        "unchanged": {"type": "boolean", "description": "True when known was this version: the data is null and the caller keeps what it has."},
        "error": {"type": ["object", "null"], "additionalProperties": false, "required": ["code", "message", "where"],
          "description": "Why the data could not be read, such as resolve.unavailable for a live source that is not attached; null otherwise.",
          "properties": {
            "code": {"type": "string", "description": "A stable code, such as resolve.unavailable or resolve.unknown_column."},
            "message": {"type": "string", "description": "One plain sentence that says what is wrong."},
            "where": {"type": "string", "description": "The spec source or binding it concerns; empty when none."}}},
        "state": {"type": ["object", "null"], "additionalProperties": false, "required": ["kind", "detail"],
          "description": "A state of the data in place of the drawing; null when it draws.",
          "properties": {
            "kind": {"type": "string", "enum": ["empty", "mismatch"], "description": "empty: no values to draw; mismatch: the data does not fit the slot."},
            "detail": {"type": "string", "description": "The binding or panel it concerns."}}},
        "bounds": {"type": "array", "items": {"type": "string"}, "description": "The bounds that applied, in plain words, such as a line decimated or a table cut."},
        "data": {"type": ["object", "null"], "additionalProperties": true,
          "description": "The primitive's data, null with an error, a state or unchanged. Arrays of numbers are objects of dtype (f32 or f64), shape and base64, the little-endian bytes; tables, cells and counts are JSON. series: x_index, steps and lines of binding, row, count, last, x and y. heatmap: source_rows, source_cols, stride_rows, stride_cols and values. image: source_rows, source_cols, stride, channels, float and values. points: three_d, total, stride, x, y, z, color, color_integer and size. histogram: lo, hi, total, outside and counts. table: columns, rows, total_rows and truncated. text: value."}}})json",
     R"json({"panel": "curves", "applet": "run42-dashboard"})json"},

    // layout
    {"layout.place", true, false, true,
     "Places a panel of the open spec next to another one, the target: beside puts it to the right, above and "
     "below above or below, tabbed in tabs with the target. ratio is the share of the space the placed panel gets, "
     "from 0.05 to 0.95, half by default. A panel already in the layout is moved. Undo restores the previous layout."
     " The result holds the change as a diff; with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["panel", "target", "where"],
      "properties": {
        "panel": {"$use": "id", "description": "The id of the panel, or of the control (spec version 1), to place."},
        "target": {"$use": "id", "description": "The id of the panel or control to place it next to."},
        "where": {"type": "string", "enum": ["beside", "above", "below", "tabbed"], "description": "Where to place it, relative to the target."},
        "ratio": {"type": "number", "minimum": 0.05, "maximum": 0.95, "default": 0.5, "description": "The share of the space the placed panel gets; 0.5 when absent."},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["layout", "warnings", "diff", "applied"],
      "properties": {
        "layout": {"$use": "layout", "description": "The whole layout as it is now, or would be after a dry run."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"panel": "loss", "target": "heat", "where": "below", "ratio": 0.4})json"},
    {"layout.set", true, false, true,
     "Replaces the whole layout of the open spec. The layout is a flat list of nodes: each node splits its space "
     "horizontally or vertically between its children, with ratio the share of the first child or shares one share "
     "for each child, or shows them in tabs; mins gives a panel or control child its minimum in columns and lines. "
     "A child is a panel id or another node's id, and root names the top node, or the one panel when there "
     "is only one. Every panel of the spec should appear once; a panel left out is not drawn. Undo restores the "
     "previous layout. The result holds the change as a diff; with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["layout"],
      "properties": {
        "layout": {"$use": "layout", "description": "The new layout."},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["layout", "warnings", "diff", "applied"],
      "properties": {
        "layout": {"$use": "layout", "description": "The layout as it is now, or would be after a dry run."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"layout": {"root": "main", "nodes": [{"id": "main", "split": "horizontal", "ratio": 0.6, "children": ["heat", "right"]}, {"id": "right", "split": "vertical", "ratio": null, "children": ["mean", "loss"]}]}})json"},

    // preview
    {"preview.begin", false, false, true,
     "Starts a preview of the open spec: a scratch copy on which edits of the open spec stack, each checked like a "
     "real edit, while the open spec, the undo history and the journal stay as they are. An edit reaches the preview "
     "when its call names the preview's id, which Caliper API takes in the X-Caliper-Preview header; spec.get, "
     "spec.validate and panel.render read the preview the same way. With panel or source the preview is scoped to "
     "that target: its edits may change only the target, the sources a target panel binds and the items the preview "
     "adds. preview.diff shows the stacked change, and only the person commits it, with Accept in Caliper. A preview "
     "ends when it is committed or discarded, or after 30 minutes without use.",
     R"json({"type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "id": {"$use": "preview_id", "description": "The new preview's id; one is made when absent."},
        "panel": {"$use": "id", "description": "A panel of the open spec that the preview is about, from spec.get."},
        "source": {"$use": "id", "description": "A source of the open spec that the preview is about, from spec.get; give panel or source, not both."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "panel", "source"],
      "properties": {
        "id": {"type": "string", "description": "The preview's id."},
        "panel": {"type": ["string", "null"], "description": "The panel the preview is scoped to, or null."},
        "source": {"type": ["string", "null"], "description": "The source the preview is scoped to, or null."}}})json",
     R"json({"id": "widget-1", "panel": "heat"})json"},
    {"preview.diff", false, false, true,
     "Shows what a preview would change: the diff of the open spec's canonical TOML against the preview with every "
     "edit stacked on it, how many edits it holds and what the validator warns about. When the open spec changed "
     "since the edits were made, they are applied to it again first, and when one no longer applies the call fails "
     "with history.conflict. It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {"id": {"$use": "preview_id"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "edits", "diff", "warnings", "panel", "source"],
      "properties": {
        "id": {"type": "string", "description": "The preview's id."},
        "edits": {"type": "integer", "description": "How many edits the preview holds."},
        "diff": {"type": "string", "description": "The stacked change as a unified diff of the spec's canonical TOML text, from the open spec to the preview; empty when they are the same."},
        "warnings": {"$use": "warnings"},
        "panel": {"type": ["string", "null"], "description": "The panel the preview is scoped to, or null."},
        "source": {"type": ["string", "null"], "description": "The source the preview is scoped to, or null."}}})json",
     R"json({"id": "widget-1"})json"},
    {"preview.commit", true, false, true,
     "Applies every edit of a preview to the open spec, in order, as one step that one undo reverts, and ends the "
     "preview. Only the person commits a preview, with Accept in Caliper: a call from an agent, from caliper-mcp or "
     "from a program is refused with preview.not_allowed. When the open spec changed since the edits were made, they "
     "are applied to it again, and when one no longer applies the commit is refused with history.conflict and "
     "nothing changes.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {"id": {"$use": "preview_id"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "edits", "diff", "warnings"],
      "properties": {
        "id": {"type": "string", "description": "The preview's id."},
        "edits": {"type": "integer", "description": "How many edits were applied."},
        "diff": {"$use": "diff"},
        "warnings": {"$use": "warnings"}}})json",
     R"json({"id": "widget-1"})json"},
    {"preview.discard", false, false, true,
     "Ends a preview and drops its edits. The open spec, the undo history and the journal stay as they were, as if "
     "the preview had never been made.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {"id": {"$use": "preview_id"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "edits"],
      "properties": {
        "id": {"type": "string", "description": "The preview's id."},
        "edits": {"type": "integer", "description": "How many edits were dropped."}}})json",
     R"json({"id": "widget-1"})json"},

    // applet
    {"applet.save", true, true, true,
     "Saves the open spec as a spec applet of the workspace: the folder applets/<id>/ with a manifest "
     "<id>.caliper.toml and the spec <id>.spec.toml. The id is the spec's applet id unless one is given. Saving over "
     "an existing applet needs overwrite set to true. It writes files, so an agent's call needs the person's "
     "approval, and it cannot be undone.",
     R"json({"type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "id": {"$use": "id", "description": "The applet's id; the spec's own applet id when absent."},
        "overwrite": {"type": "boolean", "default": false, "description": "Whether to replace a saved applet of the same id; false when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "manifest", "spec"],
      "properties": {
        "id": {"type": "string", "description": "The saved applet's id."},
        "manifest": {"type": "string", "description": "The manifest's path, relative to the workspace."},
        "spec": {"type": "string", "description": "The spec's path, relative to the workspace."}}})json",
     R"json({"id": "attention-entropy"})json"},
    {"applet.open", true, false, true,
     "Opens a saved spec applet of the workspace by its id, from applet.list, as the open spec, so that it can be "
     "edited. Undo returns to the spec that was open before.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {"id": {"$use": "id", "description": "The applet's id, from applet.list."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "title", "warnings"],
      "properties": {
        "id": {"type": "string", "description": "The applet's id."},
        "title": {"type": "string", "description": "The applet's title."},
        "warnings": {"type": "array", "items": {"$use": "diagnostic"}, "description": "What the validator warns about."}}})json",
     R"json({"id": "attention-entropy"})json"},
    {"applet.list", false, false, false,
     "Lists the spec applets saved in the workspace, with each one's id, title, folder and the time it was last "
     "saved. The id is what applet.open takes. It changes nothing.",
     kNoArgs,
     R"json({"type": "object", "additionalProperties": false, "required": ["applets"],
      "properties": {"applets": {"type": "array", "description": "The saved spec applets, sorted by id.",
        "items": {"type": "object", "additionalProperties": false, "required": ["id", "title", "path", "updated_at"],
          "properties": {
            "id": {"type": "string", "description": "The applet's id."},
            "title": {"type": "string", "description": "The applet's title."},
            "path": {"type": "string", "description": "Its folder, relative to the workspace."},
            "updated_at": {"type": ["string", "null"], "description": "When it was last saved, in UTC ISO 8601."}}}}}})json",
     "{}"},

    // review
    {"review.status", false, false, false,
     "Says which of your commands wait for the person's approval in Caliper before they run: destructive commands, "
     "and edits of the open spec when the person reviews spec changes from your origin (for agents, on by default). "
     "A call that waits is held until the person decides, for up to 120 seconds, or answered with "
     "command.needs_approval; a dry run, an edit of a preview and an edit that changes nothing never wait. An agent "
     "that asks the person itself before a change can leave out what Caliper already asks. It changes nothing.",
     kNoArgs,
     R"json({"type": "object", "additionalProperties": false, "required": ["origin", "spec_edits", "destructive"],
      "properties": {
        "origin": {"type": "string", "enum": ["ui", "builder", "agent", "mcp", "api"], "description": "Your origin, as Caliper sees it."},
        "spec_edits": {"type": "boolean", "description": "Whether your edits of the open spec wait for the person's review, with their diff."},
        "destructive": {"type": "boolean", "description": "Whether your destructive commands, which delete or overwrite saved data or run a Python loader, wait for the person's approval."}}})json",
     "{}"},

    // capture
    {"capture.frame", false, false, true,
     "Returns the frame the app draws, the whole window, as a PNG image, so that you can see the app as the person "
     "does. It changes nothing.",
     kNoArgs, R"json({"$use": "png"})json", "{}"},

    // control (spec version 1; design/instruments.md)
    {"control.create", true, false, true,
     "Adds a control to the open spec: a slider, a button or a toggle that the person and you can use. bind names its "
     "one target: param:<name> for a parameter of the running script, action:<name> for an action of it, run:pause "
     "for pausing and resuming the run, value for a value the spec holds, or intervene:<point> for an intervention "
     "point, which a toggle ablates while it is on. run.state lists the parameters and actions, and intervene.points "
     "the points. "
     "A control is a leaf of the layout, like a panel: place puts it next to a panel or control, and without place it "
     "goes under the whole layout. Undo removes it. The result holds the change as a diff; with dry_run true the "
     "change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "kind"],
      "properties": {
        "id": {"$use": "id", "description": "The new control's id; panels and controls share one set of ids."},
        "kind": {"type": "string", "enum": ["slider", "button", "range", "toggle", "picker", "text", "segmented", "readout", "transport"], "description": "slider sets a number in a range; button acts once when pressed; toggle is on or off, and ablates intervene:<point> or sets a boolean param:<name>. range sets [lo, hi]; picker chooses one value from choices or choices_from; text sets a text; segmented is a row of 2 to 16 buttons, one on or, with multiple, any number; readout shows bound text and binds nothing; transport binds cursor and draws the step controls."},
        "title": {"type": "string", "minLength": 1, "maxLength": 256, "description": "The control's label; the id when absent."},
        "bind": {"$use": "control_bind"},
        "options": {"$use": "control_options", "description": "The control's options."},
        "place": {"type": "object", "additionalProperties": false, "required": ["target", "where"],
                  "description": "Where to put the control, next to a panel or control of the layout.",
                  "properties": {
                    "target": {"$use": "id", "description": "The id of a panel or control already in the layout."},
                    "where": {"type": "string", "enum": ["beside", "above", "below", "tabbed"], "description": "beside puts it to the right of the target; tabbed shows both in tabs."},
                    "ratio": {"type": "number", "minimum": 0.05, "maximum": 0.95, "default": 0.5, "description": "The share of the space the new control gets; 0.5 when absent."}}},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "warnings", "diff", "applied"],
      "properties": {
        "id": {"type": "string", "description": "The new control's id."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"id": "lr", "kind": "slider", "title": "Learning rate", "bind": "param:lr", "options": {"log": true}, "place": {"target": "world", "where": "below", "ratio": 0.15}})json"},
    {"control.patch", true, false, true,
     "Changes a control of the open spec with a JSON merge patch (RFC 7386) of its kind, title, bind and options: the "
     "keys given replace the control's own, a key set to null is removed, and options merge key by key. The id "
     "cannot change. The id comes from spec.get. Undo restores the control. The result holds the change as a diff; "
     "with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "patch"],
      "properties": {
        "id": {"$use": "id", "description": "The control's id in the open spec."},
        "patch": {"type": "object", "additionalProperties": false, "required": [],
                  "description": "The keys to change; null removes one.",
                  "properties": {
                    "kind": {"type": "string", "enum": ["slider", "button", "range", "toggle", "picker", "text", "segmented", "readout", "transport"], "description": "What the control is."},
                    "title": {"type": ["string", "null"], "minLength": 1, "maxLength": 256, "description": "The control's label."},
                    "bind": {"$use": "control_bind"},
                    "options": {"$patch": "control_options", "description": "Options to set, or null to clear one."}}},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["control", "warnings", "diff", "applied"],
      "properties": {
        "control": {"$use": "control", "description": "The control as it is now, or would be after a dry run."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"id": "lr", "patch": {"title": "Learning rate", "options": {"min": 0.0001, "max": 0.1}}})json"},
    {"control.remove", true, false, true,
     "Removes a control from the open spec and from its layout. The id comes from spec.get. Undo restores it in its "
     "place. The result holds the change as a diff; with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {
        "id": {"$use": "id", "description": "The control's id in the open spec."},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "warnings", "diff", "applied"],
      "properties": {
        "id": {"type": "string", "description": "The removed control's id."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"id": "lr"})json"},

    // state, selection, entity, annotation, cursor and camera (design/instrument-interface.md, Commands)
    {"state.create", true, false, true,
     "Adds a state variable to the open spec: a named value that controls and picks write and that panels read "
     "through references. type is number, integer, boolean, text, range or list. default, min, max and choices limit "
     "the value, and x holds extension keys. The id cursor is reserved. Undo removes it. The result holds the change "
     "as a diff; with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "type"],
      "properties": {
        "id": {"$use": "id", "description": "The new state variable's id; cursor is reserved."},
        "type": {"type": "string", "enum": ["number", "integer", "boolean", "text", "range", "list"], "description": "The type of the value. A list holds texts."},
        "title": {"type": "string", "minLength": 1, "maxLength": 256, "description": "The label that a control or a legend shows; the id when absent."},
        "default": {"description": "The starting value, a JSON value of the type; 0, false, the first choice, an empty text, [min, max] or an empty list when absent."},
        "min": {"type": "number", "description": "The low bound of a number, an integer or a range."},
        "max": {"type": "number", "description": "The high bound of a number, an integer or a range."},
        "choices": {"type": "array", "minItems": 1, "maxItems": 1024, "items": {"description": "One allowed value."}, "description": "The allowed values of a text or an integer, or the allowed items of a list, from 1 to 1024."},
        "x": {"description": "An extension table, a JSON object."},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "warnings", "diff", "applied"],
      "properties": {
        "id": {"type": "string", "description": "The new state variable's id."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"id": "token", "type": "integer", "title": "Token", "default": 0, "min": 0, "max": 63})json"},
    {"state.patch", true, false, true,
     "Changes a state variable of the open spec with the keys of set: the keys given replace the variable's own. "
     "The id cannot change. state.get lists the ids. Undo restores the variable. The result holds the change as a "
     "diff; with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "set"],
      "properties": {
        "id": {"$use": "id", "description": "The state variable's id in the open spec."},
        "set": {"description": "The keys to change, a JSON object with keys of a state variable such as title, default, min, max and choices."},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "warnings", "diff", "applied"],
      "properties": {
        "id": {"type": "string", "description": "The state variable's id."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"id": "token", "set": {"max": 127}})json"},
    {"state.remove", true, false, true,
     "Removes a state variable from the open spec. A reference, a control or a pick that reads or writes the "
     "variable keeps it in use, and the command then fails with state.in_use. state.get lists the ids. Undo restores "
     "the variable. The result holds the change as a diff; with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {
        "id": {"$use": "id", "description": "The state variable's id in the open spec."},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "warnings", "diff", "applied"],
      "properties": {
        "id": {"type": "string", "description": "The removed state variable's id."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"id": "token"})json"},
    {"state.set", false, false, true,
     "Sets the current value of a state variable, as a person's edit of a control does, and returns the value and the "
     "one before it. The value must fit the type, the bounds and the choices, or the command fails with "
     "state.bad_value. It adds a state event with the origin of the caller. It changes the session and not the "
     "workspace, so it has no undo step.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "value"],
      "properties": {
        "id": {"$use": "id", "description": "The state variable's id."},
        "value": {"description": "The new value, a JSON value of the variable's type."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["value", "previous"],
      "properties": {
        "value": {"description": "The value that the variable holds now."},
        "previous": {"description": "The value that the variable held before."}}})json",
     R"json({"id": "token", "value": 5})json"},
    {"state.get", false, false, true,
     "Returns the state variables of the open spec: each one's id, type, current value, default, the writers that "
     "set it and the readers that use it. ids limits the answer to some variables. It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "ids": {"type": "array", "minItems": 1, "maxItems": 256, "items": {"$use": "id"}, "description": "Only these state variables; every variable when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["states"],
      "properties": {
        "states": {"type": "array", "maxItems": 256, "description": "The state variables.",
          "items": {"type": "object", "additionalProperties": false, "required": ["id", "type", "value", "default", "writers", "readers"],
            "properties": {
              "id": {"type": "string", "description": "The state variable's id."},
              "type": {"type": "string", "description": "The type of the value: number, integer, boolean, text, range or list."},
              "value": {"description": "The current value."},
              "default": {"description": "The default value."},
              "writers": {"type": "array", "items": {"type": "string"}, "description": "What writes the variable, such as a control or a pick."},
              "readers": {"type": "array", "items": {"type": "string"}, "description": "What reads the variable, such as a panel option or an annotation."}}}}}})json",
     R"json({"ids": ["token"]})json"},
    {"control.set", false, false, true,
     "Sets the value of a control that binds value or a state, as a person's edit does, and returns the value. It "
     "adds a control event with the origin of the caller. A control that binds a parameter takes run.set_param, and "
     "this command refuses it with control.not_settable. It changes the session and not the workspace, so it has no "
     "undo step.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "value"],
      "properties": {
        "id": {"$use": "id", "description": "The control's id in the open spec."},
        "value": {"description": "The new value, a JSON value of the control's kind: a number, a boolean, a text, [lo, hi] for a range or a list for a group with multiple."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["value"],
      "properties": {
        "value": {"description": "The value that the control holds now."}}})json",
     R"json({"id": "measure", "value": "mean"})json"},
    {"selection.set", false, false, true,
     "Makes a pick in a panel: the entity and the item at index, a program that draws in the panel and the item "
     "of its item map, or clear to remove the selection. It makes the "
     "same pick event, field values and state writes as a person's pick, with the origin of the caller. It returns "
     "the pick fields. It changes the session and not the workspace, so it has no undo step.",
     R"json({"type": "object", "additionalProperties": false, "required": ["panel"],
      "properties": {
        "panel": {"$use": "id", "description": "The panel's id in the open spec."},
        "entity": {"$use": "entity_path", "description": "The entity of the item to pick."},
        "index": {"type": "integer", "minimum": 0, "description": "The item's index in the entity."},
        "program": {"$use": "id", "description": "A program of the panel whose item to pick, in place of an entity."},
        "item": {"type": "integer", "minimum": 0, "maximum": 4294967295, "description": "The element of the program's item map to pick, at the cursor."},
        "clear": {"type": "boolean", "description": "Whether to clear the selection of the panel; false when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["fields"],
      "properties": {
        "fields": {"description": "The pick fields as an object, or null when the panel has no selection."}}})json",
     R"json({"panel": "world", "entity": "terrain", "index": 3})json"},
    {"selection.get", false, false, true,
     "Returns the pick fields of the last pick in a panel, or null when the panel has no selection. It changes "
     "nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": ["panel"],
      "properties": {
        "panel": {"$use": "id", "description": "The panel's id in the open spec."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["fields"],
      "properties": {
        "fields": {"description": "The pick fields as an object, or null when the panel has no selection."}}})json",
     R"json({"panel": "world"})json"},
    {"entity.set", true, false, true,
     "Adds an entity to the scene of a scene panel, or replaces the entity with the same path. The entity is a JSON "
     "object with the keys of an entity table. The panel must be a scene panel. Undo restores the scene. The result "
     "holds the change as a diff; with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["panel", "entity"],
      "properties": {
        "panel": {"$use": "id", "description": "The scene panel's id in the open spec."},
        "entity": {"description": "The entity table, a JSON object with a path and the other keys of an entity."},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["panel", "warnings", "diff", "applied"],
      "properties": {
        "panel": {"type": "string", "description": "The panel's id."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"panel": "world", "entity": {"path": "markers/origin", "draw": "points"}})json"},
    {"entity.remove", true, false, true,
     "Removes the entity with a path from the scene of a scene panel. The panel must be a scene panel, and it must "
     "have the entity. Undo restores it. The result holds the change as a diff; with dry_run true the change is "
     "only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["panel", "path"],
      "properties": {
        "panel": {"$use": "id", "description": "The scene panel's id in the open spec."},
        "path": {"$use": "entity_path", "description": "The entity's path."},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["panel", "warnings", "diff", "applied"],
      "properties": {
        "panel": {"type": "string", "description": "The panel's id."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"panel": "world", "path": "markers/origin"})json"},
    {"annotation.set", true, false, true,
     "Adds an annotation to a panel, or replaces the annotation with the same id. The annotation is a JSON object "
     "with an id, a kind (label, threshold, band or marker) and the keys of its kind. Undo restores the panel. The "
     "result holds the change as a diff; with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["panel", "annotation"],
      "properties": {
        "panel": {"$use": "id", "description": "The panel's id in the open spec."},
        "annotation": {"description": "The annotation table, a JSON object with an id, a kind and the keys of that kind."},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["panel", "warnings", "diff", "applied"],
      "properties": {
        "panel": {"type": "string", "description": "The panel's id."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"panel": "loss", "annotation": {"id": "floor", "kind": "threshold", "axis": "y", "value": 0.1, "text": "Floor"}})json"},
    {"annotation.remove", true, false, true,
     "Removes the annotation with an id from a panel. The panel must have the annotation, or the command fails "
     "with annotation.not_found. Undo restores it. The result holds the change as a diff; with dry_run true the "
     "change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["panel", "id"],
      "properties": {
        "panel": {"$use": "id", "description": "The panel's id in the open spec."},
        "id": {"$use": "id", "description": "The annotation's id in the panel."},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["panel", "warnings", "diff", "applied"],
      "properties": {
        "panel": {"type": "string", "description": "The panel's id."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"panel": "loss", "id": "floor"})json"},
    {"cursor.get", false, false, true,
     "Returns the cursor, which chooses the step that the panels show, and each source's first and last kept step, "
     "the number of kept steps, the number of requested steps and the bytes. It changes nothing.",
     kNoArgs,
     R"json({"type": "object", "additionalProperties": false, "required": ["cursor", "sources"],
      "properties": {
        "cursor": {"$use": "cursor_state", "description": "The cursor."},
        "sources": {"type": "array", "maxItems": 256, "items": {"$use": "cursor_source"}, "description": "The sources that keep steps."}}})json",
     "{}"},
    {"cursor.set", false, false, true,
     "Moves the cursor, sets whether it follows the newest step (live), whether it plays and how fast. A step "
     "outside the kept steps goes to the nearest kept step, and the result gives the cursor as it is. When no source "
     "keeps steps, the command fails with cursor.no_history. It changes the session and not the workspace, so it has "
     "no undo step.",
     R"json({"type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "step": {"type": "integer", "description": "The step to show."},
        "live": {"type": "boolean", "description": "Whether the cursor follows the newest step."},
        "playing": {"type": "boolean", "description": "Whether the cursor plays forward."},
        "speed": {"type": "number", "minimum": 0.1, "maximum": 1000, "description": "The play speed in steps per second, from 0.1 to 1000."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["cursor"],
      "properties": {
        "cursor": {"$use": "cursor_state", "description": "The cursor after the move."}}})json",
     R"json({"step": 120, "live": false})json"},
    {"cursor.step", false, false, true,
     "Moves the cursor by count kept steps, forward when count is above 0 and back when it is below 0. A count of 0 "
     "is refused. The cursor stops at the nearest kept step, and the result gives the cursor as it is. It changes the "
     "session and not the workspace, so it has no undo step.",
     R"json({"type": "object", "additionalProperties": false, "required": ["count"],
      "properties": {
        "count": {"type": "integer", "minimum": -1000, "maximum": 1000, "description": "How many kept steps to move, from -1000 to 1000 and not 0."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["cursor"],
      "properties": {
        "cursor": {"$use": "cursor_state", "description": "The cursor after the move."}}})json",
     R"json({"count": -1})json"},
    {"camera.get", false, false, true,
     "Returns the orbit camera of a scene panel and the views that the panel saved. The panel must be a scene panel. "
     "It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": ["panel"],
      "properties": {
        "panel": {"$use": "id", "description": "The scene panel's id in the open spec."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["orbit", "views"],
      "properties": {
        "orbit": {"$use": "orbit", "description": "The camera now."},
        "views": {"type": "array", "maxItems": 16, "items": {"$use": "camera_view"}, "description": "The saved views, at most 16."}}})json",
     R"json({"panel": "world"})json"},
    {"camera.set", false, false, true,
     "Moves the orbit camera of a scene panel to a saved view by name, or to the orbit that you give, and returns "
     "the orbit. A name that the panel did not save fails with camera.unknown_view. It changes the session and not the "
     "workspace, so it has no undo step.",
     R"json({"type": "object", "additionalProperties": false, "required": ["panel"],
      "properties": {
        "panel": {"$use": "id", "description": "The scene panel's id in the open spec."},
        "view": {"type": "string", "minLength": 1, "maxLength": 64, "description": "The name of a saved view; not with orbit."},
        "orbit": {"$use": "orbit", "description": "The orbit to move to; not with view."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["orbit"],
      "properties": {
        "orbit": {"$use": "orbit", "description": "The camera after the move."}}})json",
     R"json({"panel": "world", "view": "overview"})json"},
    {"camera.save", true, false, true,
     "Saves the current camera of a scene panel as a view with a name, in the option views of the panel, and replaces "
     "the view with the same name. A panel holds at most 16 views. Undo restores the views. The result holds the "
     "change as a diff; with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["panel", "name"],
      "properties": {
        "panel": {"$use": "id", "description": "The scene panel's id in the open spec."},
        "name": {"type": "string", "minLength": 1, "maxLength": 64, "description": "The view's name."},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["panel", "warnings", "diff", "applied"],
      "properties": {
        "panel": {"type": "string", "description": "The panel's id."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"panel": "world", "name": "overview"})json"},

    // run (design/instruments.md)
    {"run.state", false, false, true,
     "Returns the state of the script that the app is attached to: whether one is attached, whether the app may "
     "control it and why not, the run's state and step, its parameters with their values, ranges and pending "
     "values, and its actions. It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "run": {"$use": "run_id", "description": "The run to act on: a managed run's id from run.list, or user for the person's own script; the attached script when absent. The app switches to a run it does not show."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["attached", "control", "state", "params", "actions"],
      "properties": {
        "attached": {"type": "boolean", "description": "Whether a script is attached to the app."},
        "script": {"type": "string", "description": "The attached script's name; absent when none is attached."},
        "control": {"type": "boolean", "description": "Whether the app may pause the run, set its parameters and run its actions."},
        "why_not": {"type": "string", "description": "When control is false, the reason in one sentence, such as a stream started without control."},
        "state": {"$use": "run_state_name"},
        "step": {"type": ["integer", "null"], "description": "The script's step, the count of its caliper.step() calls, or null before the first."},
        "steps_left": {"type": ["integer", "null"], "description": "While stepping, the steps left before the run pauses again; null otherwise."},
        "params": {"type": "array", "maxItems": 64, "items": {"$use": "run_param"}, "description": "The parameters the script registered, in its order."},
        "actions": {"type": "array", "maxItems": 64, "items": {"$use": "run_action"}, "description": "The actions the script registered, in its order."}}})json",
     "{}"},
    {"run.set_param", false, false, true,
     "Sets one parameter of the running script, such as its learning rate. The value applies at the script's next "
     "step (caliper.step(), or each optimizer step under caliper.watch(model, optimizer)), and the script clamps it "
     "to the parameter's range. The result says the value is pending; a run event of change param reports the value "
     "the script applied, through events.wait. run.state lists the parameters.",
     R"json({"type": "object", "additionalProperties": false, "required": ["name", "value"],
      "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 63, "description": "The parameter's name, from run.state."},
        "value": {"type": ["number", "integer", "boolean", "string"], "description": "The value must match the parameter type. Text accepts at most 4096 UTF-8 bytes without NUL. A choice accepts one listed value."},
        "run": {"$use": "run_id", "description": "The run to act on: a managed run's id from run.list, or user for the person's own script; the attached script when absent. The app switches to a run it does not show."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["name", "value", "pending"],
      "properties": {
        "name": {"type": "string", "description": "The parameter's name."},
        "value": {"type": ["number", "integer", "boolean", "string"], "description": "The value sent."},
        "pending": {"type": "boolean", "description": "True: the value waits for the script's next step."}}})json",
     R"json({"name": "lr", "value": 0.001})json"},
    {"run.action", false, false, true,
     "Runs one action that the running script registered with caliper.action, such as saving a checkpoint. The "
     "script runs it at its next step; the result says it is queued, and a run event of change action brings the "
     "text it returned, or its error, through events.wait. run.state lists the actions.",
     R"json({"type": "object", "additionalProperties": false, "required": ["name"],
      "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 63, "description": "The action's name, from run.state."},
        "run": {"$use": "run_id", "description": "The run to act on: a managed run's id from run.list, or user for the person's own script; the attached script when absent. The app switches to a run it does not show."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["name", "queued"],
      "properties": {
        "name": {"type": "string", "description": "The action's name."},
        "queued": {"type": "boolean", "description": "True: the action waits for the script's next step."}}})json",
     R"json({"name": "save_checkpoint"})json"},
    {"run.pause", false, false, true,
     "Pauses the running script at its next step, resumes it, or lets it run a number of steps and pause again. "
     "A script pauses only at a step, so a pause before its first step is refused. The result gives the state the "
     "request was sent in; run.state and the run events of change state follow it.",
     R"json({"type": "object", "additionalProperties": false, "required": ["op"],
      "properties": {
        "op": {"type": "string", "enum": ["pause", "resume", "step"], "description": "pause stops at the next step; resume runs on; step runs count steps, then pauses."},
        "count": {"type": "integer", "minimum": 1, "maximum": 1000000, "default": 1, "description": "step: the number of steps to run; 1 when absent."},
        "run": {"$use": "run_id", "description": "The run to act on: a managed run's id from run.list, or user for the person's own script; the attached script when absent. The app switches to a run it does not show."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["op", "state"],
      "properties": {
        "op": {"type": "string", "description": "The request sent."},
        "state": {"$use": "run_state_name"}}})json",
     R"json({"op": "pause"})json"},

    // model runner (design/hosted-runs.md)
    {"run.list", false, false, true,
     "Lists the runs: the managed runs that the model runner started, in the order of their start, and the person's "
     "own script when the app is attached to one, with the id user. Each run has its state, script, arguments, "
     "parameters, parent, step and exit code. It changes nothing.",
     kNoArgs,
     R"json({"type": "object", "additionalProperties": false, "required": ["runs"],
      "properties": {
        "runs": {"type": "array", "items": {"$use": "run_entry"}, "description": "The runs, managed runs first in the order of their start."}}})json",
     "{}"},
    {"run.start", false, false, true,
     "Starts a managed workspace script or local inference through the fixed Caliper module. "
     "Script runs take script, args, and params, with optional kind:script. "
     "Inference runs require kind:inference, model, dictionary, and prompt. The two forms cannot mix fields. "
     "Replication requires both replication_profile and private_output_directory. "
     "Both forms wait for approval on a review card. The result has state starting when the process starts. "
     "A runner event through events.wait, or run.list, reports later state changes.",
     R"json({"type": "object", "additionalProperties": false, "required": [], "properties": {"script": {"type": "string", "minLength": 1, "maxLength": 1024, "description": "The script's path relative to the workspace, with / between folders and no ., .. or link in it, such as scripts/train.py, so that the review card names the file that runs; another path to a file of the workspace fails with run.script_refused, whose message gives the path to use."}, "args": {"type": "array", "maxItems": 64, "items": {"type": "string", "maxLength": 256}, "description": "The script's command-line arguments, each at most 256 bytes and at most 1024 bytes in all, so that the review card shows them in full; none when absent."}, "params": {"$use": "run_params"}, "id": {"$use": "run_id", "description": "The new run's id, unique in the workspace; the script's file name and a number, such as train-1, when absent."}, "kind": {"type": "string", "description": "The run kind. Script is the default.", "enum": ["script", "inference"]}, "model": {"type": "string", "description": "The local inference model resource, at most 1024 UTF-8 bytes.", "minLength": 1, "maxLength": 1024}, "dictionary": {"type": "string", "description": "The local inference dictionary resource, at most 1024 UTF-8 bytes.", "minLength": 1, "maxLength": 1024}, "prompt": {"type": "string", "description": "The initial inference prompt, at most 4096 UTF-8 bytes.", "minLength": 1, "maxLength": 4096}, "replication_profile": {"type": "string", "description": "The optional inference replication profile. It requires private_output_directory. Script runs reject this field.", "enum": ["published_dallas_v1"]}, "private_output_directory": {"type": "string", "description": "An existing local directory for private replication output, at most 1024 UTF-8 bytes. Create a unique private directory before run.start. It requires replication_profile. Script runs reject this field.", "minLength": 1, "maxLength": 1024}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["run"],
      "properties": {
        "run": {"$use": "run_entry", "description": "The run as run.list gives it."}}})json",
     R"json({"script": "train.py", "args": ["--epochs", "5"], "params": [{"name": "lr", "value": 0.01}]})json"},
    {"run.fork", false, false, true,
     "Starts a new managed run from the next checkpoint of a running run, with changed parameters, beside its "
     "parent: the parent saves a checkpoint through its caliper.checkpoint action, and the fork runs the same script "
     "and arguments from it. It always waits for the person's approval on a review card. The result comes at once, "
     "in the state starting; a run event of change runner or run.list says when the fork runs.",
     R"json({"type": "object", "additionalProperties": false, "required": ["run"], "properties": {"run": {"$use": "run_id", "description": "The parent: a managed run that runs or is paused."}, "params": {"$use": "run_params", "description": "The parameters that change in the fork; the checkpoint holds the others."}, "id": {"$use": "run_id", "description": "The fork's id, unique in the workspace; the parent's script name and a number when absent."}, "step": {"type": "integer", "minimum": 0, "description": "Start from the parent's newest kept checkpoint at or before this step, in place of its next checkpoint."}, "interventions": {"type": "array", "maxItems": 16, "description": "Interventions that the fork applies before its first step, each as intervene.set takes it.", "items": {"$use": "intervention_spec"}}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["run"],
      "properties": {
        "run": {"$use": "run_entry", "description": "The run as run.list gives it."}}})json",
     R"json({"run": "train-1", "params": [{"name": "lr", "value": 0.001}]})json"},
    {"run.stop", false, false, true,
     "Stops a managed run at its next checkpoint through its caliper.stop action, and ends its process by its id "
     "when it has not ended after kill_after_s seconds. The result comes at once, with stopping true; run.list or a "
     "run event of change runner says when the run exited.",
     R"json({"type": "object", "additionalProperties": false, "required": ["run"],
      "properties": {
        "run": {"$use": "run_id", "description": "The managed run to stop."},
        "kill_after_s": {"type": "number", "minimum": 0, "maximum": 600, "default": 10, "description": "How long the run has to end by itself, in seconds; 10 when absent, 0 to end the process at once."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["run"],
      "properties": {
        "run": {"$use": "run_entry", "description": "The run as run.list gives it."}}})json",
     R"json({"run": "train-1", "kill_after_s": 30})json"},
    {"run.logs", false, false, true,
     "Returns the last lines that a managed run wrote to its standard output and standard error, which the runner "
     "keeps in the run's output.log. It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": ["run"],
      "properties": {
        "run": {"$use": "run_id", "description": "The managed run."},
        "lines": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50, "description": "How many of the last lines; 50 when absent, at most 200."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["run", "lines"],
      "properties": {
        "run": {"type": "string", "description": "The run's id."},
        "lines": {"type": "array", "maxItems": 200, "items": {"type": "string"}, "description": "The last lines, oldest first."}}})json",
     R"json({"run": "train-1", "lines": 20})json"},

    // events (design/instruments.md)
    {"events.wait", false, false, false,
     "Returns the events that came after a cursor: picks and selections in the app's views and scene panels, "
     "changes of controls, changes of the running script (state, parameters, actions, attachment), edits of the open "
     "spec, approvals, the compile, refusal and stop of GPU programs, writes of state variables and "
     "moves of the cursor. It answers at once when events "
     "wait, and otherwise holds the call until one comes or "
     "timeout_s ends. Pass the cursor of each result to the next call. The app keeps the last 1024 events; lost "
     "counts those that were dropped before you read them. Hovers are not events. It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "cursor": {"type": "integer", "minimum": 0, "default": 0, "description": "The cursor of the previous result; 0, the default, returns the oldest events the app still keeps."},
        "kinds": {"type": "array", "minItems": 1, "maxItems": 9, "items": {"type": "string", "enum": ["pick", "select", "control", "run", "spec", "approval", "program", "state", "cursor"]}, "description": "Only these kinds; every kind when absent."},
        "panels": {"type": "array", "minItems": 1, "maxItems": 64, "items": {"$use": "id"}, "description": "Only events of these panels or controls; events of every panel and of none when absent."},
        "timeout_s": {"type": "number", "minimum": 0, "maximum": 60, "default": 10, "description": "How long to wait for a first event, in seconds, at most 60; 10 when absent, 0 to answer at once."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 256, "default": 64, "description": "The most events returned; 64 when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["events", "cursor", "lost", "more"],
      "properties": {
        "events": {"type": "array", "maxItems": 256, "items": {"$use": "event"}, "description": "The events that match, oldest first."},
        "cursor": {"type": "integer", "minimum": 0, "description": "The cursor to pass to the next call: past every event this call looked at, matching or not."},
        "lost": {"type": "integer", "minimum": 0, "description": "How many events after the given cursor the app dropped before this call, because the ring keeps 1024; 0 when none."},
        "more": {"type": "boolean", "description": "Whether more matching events wait after cursor, beyond limit."}}})json",
     R"json({"cursor": 0, "kinds": ["pick"], "panels": ["world"], "timeout_s": 30})json"},

    // programs (design/gpu-programs.md)
    {"program.create", true, false, true,
     "Adds a GPU program to the open spec: a WGSL compute pass that reads live tensors of the running script in place "
     "and read-only, and writes one output. A geometry output becomes an entity of a scene panel. A pixels output "
     "becomes the live view program.<id>, which an image panel shows through a source of catalog live:program.<id>. "
     "The program runs in the script's process every N steps, with N set so that training keeps 95 percent of its "
     "speed, and a program over its time or memory limit stops. The sandbox checks the declaration and the source "
     "first; a refused program is not added, and messages give the compiler's line and column. program.status reports "
     "its state. Undo removes it. With dry_run true the change is only checked and not applied. An instance of a "
     "piece takes piece, bind, panel and params in place of the WGSL keys, and inline adds a whole piece to the "
     "spec's inline pieces in the same step; a missing or changed piece is piece.missing or piece.hash_mismatch.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {
        "id": {"$use": "id", "description": "The new program's id, unique among programs."},
        "title": {"type": "string", "minLength": 1, "maxLength": 256, "description": "The program's label; the id when absent."},
        "wgsl": {"type": "string", "minLength": 1, "maxLength": 65536, "description": "The WGSL source with a compute entry point, at most 64 KiB, with no enable, requires or diagnostic directive. It may declare var<uniform> caliper: CaliperFrame, a struct of 8 u32 and f32 fields (step, pass_index, max_count, width, height, time_s, every, pad)."},
        "entry": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]{0,62}$", "description": "The compute entry point; the one compute entry point when absent."},
        "inputs": {"type": "array", "maxItems": 8, "items": {"$use": "program_input"}, "description": "The tensors the program reads, at most 8."},
        "output": {"$use": "program_output", "description": "The program's one output."},
        "limits": {"$use": "program_limits", "description": "The program's limits; the defaults when absent."},
        "workgroups": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "integer", "minimum": 1, "maximum": 65535}, "description": "The workgroup counts x, y and z of the dispatch; from max_count or the image size and the entry's workgroup size when absent."},
        "params": {"description": "A JSON object that gives each of the program's named parameters, at most 16 names, a number or a binding reference to a control, a selection or a state; the WGSL reads them by name."},
        "dry_run": {"$use": "dry_run"},
        "piece": {"$use": "piece_pin", "description": "An instance of a piece: the pinned piece, in place of wgsl, inputs and output."},
        "bind": {"$use": "piece_bind"},
        "panel": {"$use": "id", "description": "An instance: the scene panel that the piece's draw half writes; necessary when the piece draws."},
        "inline": {"$use": "piece", "description": "A whole piece that joins the spec's inline pieces in the same step; without piece, the program pins it from inline."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "messages", "warnings", "diff", "applied"],
      "properties": {
        "id": {"type": "string", "description": "The new program's id."},
        "messages": {"type": "array", "items": {"$use": "compiler_message"}, "description": "The compiler's and the sandbox's warnings; an error refuses the edit instead."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"id": "rows", "title": "Rows of the first layer", "wgsl": "@group(0) @binding(0) var<storage, read> w: array<f32>;\n@group(0) @binding(1) var<storage, read_write> positions: array<f32>;\n@group(0) @binding(2) var<storage, read_write> count: atomic<u32>;\n@compute @workgroup_size(64) fn run(@builtin(global_invocation_id) id: vec3<u32>) {\n  let i = id.x;\n  if (i >= 256u) { return; }\n  positions[3u * i] = w[3u * i];\n  positions[3u * i + 1u] = w[3u * i + 1u];\n  positions[3u * i + 2u] = w[3u * i + 2u];\n  atomicAdd(&count, 1u);\n}\n", "inputs": [{"var": "w", "source": "live:layers.0.weight"}], "output": {"kind": "geometry", "panel": "world", "part": "points", "max_count": 256, "arrays": ["positions"]}, "limits": {"every": 10}})json"},
    {"program.patch", true, false, true,
     "Changes a GPU program of the open spec with a JSON merge patch (RFC 7386) of its title, wgsl, entry, inputs, "
     "output, limits and workgroups: the keys given replace the program's own, a key set to null is removed, and "
     "output and limits merge key by key. The id cannot change. The sandbox checks the program again, and the script "
     "runs the new version from its next pass. Undo restores the program. With dry_run true the change is only "
     "checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "patch"],
      "properties": {
        "id": {"$use": "id", "description": "The program's id in the open spec."},
        "patch": {"$patch": "program", "description": "The keys to change; null removes one."},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["program", "messages", "warnings", "diff", "applied"],
      "properties": {
        "program": {"$use": "program", "description": "The program as it is now, or would be after a dry run."},
        "messages": {"type": "array", "items": {"$use": "compiler_message"}, "description": "The compiler's and the sandbox's warnings; an error refuses the edit instead."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"id": "rows", "patch": {"limits": {"every": 20}, "output": {"max_count": 512}}})json"},
    {"program.remove", true, false, true,
     "Removes a GPU program from the open spec; the script stops it and frees its buffers, and its output leaves the "
     "scene or the live view. The id comes from spec.get or program.status. Undo restores it. The result holds the "
     "change as a diff; with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {
        "id": {"$use": "id", "description": "The program's id in the open spec."},
        "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "warnings", "diff", "applied"],
      "properties": {
        "id": {"type": "string", "description": "The removed program's id."},
        "warnings": {"$use": "warnings"},
        "diff": {"$use": "diff"},
        "applied": {"$use": "applied"}}})json",
     R"json({"id": "rows"})json"},
    {"program.status", false, false, true,
     "Returns the state of each GPU program of the open spec: where it runs, whether it compiled, runs, "
     "was refused or stopped, its cadence N, the GPU time of its last pass, the overhead of a pass and the base step "
     "time on the run's wall clock, the measured share of the training time, its passes a second, its memory, and "
     "its last error with line and column. It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "id": {"type": "string", "minLength": 1, "maxLength": 320, "description": "Only this program; every program when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["programs"],
      "properties": {
        "programs": {"type": "array", "maxItems": 16, "items": {"$use": "program_status"}, "description": "The programs, in the spec's order."}}})json",
     R"json({"id": "rows"})json"},

    // pieces and programmable graphics (design/programmable-graphics.md, Commands)
    {"piece.check", false, false, false,
     "Checks a piece on its own without saving it: its name, its manifest, its imports, and each pass of the derive "
     "half and the draw half composed with the prelude and the generated header and compiled as the script and the "
     "viewer compile it. Returns the messages with the file, line and column, the piece's hash, and the generated "
     "header of each half, which names every binding the files may use. It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": ["piece"], "properties": {"piece": {"$use": "piece", "description": "The piece, as piece.update takes it."}, "headers": {"type": "boolean", "description": "Whether the result carries the generated headers; true when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["ok", "hash", "messages", "passes", "headers"], "properties": {"ok": {"type": "boolean", "description": "Whether the manifest and every pass pass the checks."}, "hash": {"type": ["string", "null"], "description": "The piece's hash, or null when the manifest or an import is refused."}, "messages": {"type": "array", "items": {"$use": "compiler_message"}, "description": "The messages, errors first; each names the file in its message, and line and column are in that file."}, "passes": {"type": "array", "items": {"type": "object", "additionalProperties": false, "required": ["half", "index", "ok"], "properties": {"half": {"type": "string", "enum": ["derive", "draw"], "description": "The half of the pass."}, "index": {"type": "integer", "minimum": 0, "description": "The pass's index in its half."}, "ok": {"type": "boolean", "description": "Whether the pass compiled and passed its checks."}}}, "description": "One item for each pass, the derive passes first."}, "headers": {"type": "object", "additionalProperties": false, "required": ["derive", "draw"], "properties": {"derive": {"type": "string", "description": "The derive half's generated header."}, "draw": {"type": "string", "description": "The draw half's generated header for its first pass."}}, "description": "The generated headers, or empty texts when headers is false."}}})json",
     R"json({"piece": {"name": "uv_field", "draw": {"camera": "none", "passes": [{"kind": "render", "fragment": "fs"}]}, "files": {"draw": "@fragment fn fs(in: CaliperFullscreen) -> @location(0) vec4<f32> { return vec4<f32>(in.uv, 0.0, 1.0); }"}}})json"},
    {"piece.list", false, false, true,
     "Lists the pieces that an instrument can use: the built-in library, the workspace's pieces/ folder and the "
     "inline pieces of the open spec, each version with its inputs, parameters, outputs, hash and the programs of "
     "the open spec that pin it. It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": [], "properties": {"from": {"type": "string", "enum": ["library", "workspace", "inline"], "description": "Only the pieces of this source; every source when absent."}, "match": {"type": "string", "minLength": 1, "maxLength": 64, "description": "Only the pieces whose names match this glob, such as glow_*; every piece when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["pieces", "modules"], "properties": {"modules": {"type": "array", "items": {"type": "string"}, "description": "The lib modules that a piece imports, as <name>@<version>, such as colormap@1."}, "pieces": {"type": "array", "items": {"$use": "piece_summary"}, "description": "The pieces, library first, then workspace and inline, by name and version."}}})json",
     R"json({"from": "library"})json"},
    {"piece.show", false, false, true,
     "Returns one piece with its manifest and its WGSL files, its source, its folder and the programs of the open "
     "spec that pin it. The newest version of the first source that has the name when version and from are absent. "
     "It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": ["name"], "properties": {"name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "The piece's name."}, "version": {"type": "integer", "minimum": 1, "maximum": 1000000, "description": "The version; the newest when absent."}, "from": {"type": "string", "enum": ["library", "workspace", "inline"], "description": "The source; library, then workspace, then inline when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["piece", "from", "path", "pinned_by"], "properties": {"piece": {"$use": "piece", "description": "The piece, with its files and its hash."}, "from": {"type": "string", "enum": ["library", "workspace", "inline"], "description": "Its source."}, "path": {"type": ["string", "null"], "description": "The piece's folder relative to the workspace, pieces/<name>/<version>; null for a library or inline piece."}, "pinned_by": {"type": "array", "items": {"type": "string"}, "description": "The programs of the open spec that pin this version."}}})json",
     R"json({"name": "points"})json"},
    {"piece.save", true, false, true,
     "Extracts an inline piece of the open spec into the workspace as pieces/<name>/<version>/ and pins the "
     "programs that used it to the workspace piece, in one undo step of the spec. A version that exists is "
     "piece.exists. It checks the piece first.",
     R"json({"type": "object", "additionalProperties": false, "required": ["name"], "properties": {"name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "The inline piece's name."}, "version": {"type": "integer", "minimum": 1, "maximum": 1000000, "description": "The version to write; the inline piece's own when absent."}, "dry_run": {"$use": "dry_run"}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["name", "version", "hash", "path", "programs", "messages", "applied"], "properties": {"name": {"type": "string", "description": "The piece's name."}, "version": {"type": "integer", "description": "The version written."}, "hash": {"type": "string", "description": "Its hash."}, "path": {"type": ["string", "null"], "description": "The piece's folder relative to the workspace, pieces/<name>/<version>; null for a library or inline piece."}, "programs": {"type": "array", "items": {"type": "string"}, "description": "The programs that now pin the workspace piece."}, "messages": {"type": "array", "items": {"$use": "compiler_message"}, "description": "The messages, errors first; each names the file in its message, and line and column are in that file."}, "applied": {"$use": "applied"}}})json",
     R"json({"name": "attention_arcs"})json"},
    {"piece.copy", true, false, false,
     "Forks a library piece into the workspace as version 1 of a new name, or of its own name, so that the person "
     "or the agent can change it. A library piece never changes, and an instrument uses the workspace copy only "
     "when its program says from workspace.",
     R"json({"type": "object", "additionalProperties": false, "required": ["name"], "properties": {"name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "The library piece's name."}, "version": {"type": "integer", "minimum": 1, "maximum": 1000000, "description": "The library version; the newest when absent."}, "as": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "The workspace name; the library name when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["name", "version", "hash", "path"], "properties": {"name": {"type": "string", "description": "The workspace piece's name."}, "version": {"type": "integer", "description": "Its version, 1."}, "hash": {"type": "string", "description": "Its hash."}, "path": {"type": ["string", "null"], "description": "The piece's folder relative to the workspace, pieces/<name>/<version>; null for a library or inline piece."}}})json",
     R"json({"name": "points", "as": "glow_points"})json"},
    {"piece.update", true, false, false,
     "Writes a new version of a workspace piece after it passes the checks: the version must be above every saved "
     "one, and the next one when absent. Saved versions never change, and the programs that pin them keep them "
     "until program.patch pins the new one.",
     R"json({"type": "object", "additionalProperties": false, "required": ["piece"], "properties": {"piece": {"$use": "piece", "description": "The piece, with its manifest and files."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["name", "version", "hash", "path", "messages"], "properties": {"name": {"type": "string", "description": "The piece's name."}, "version": {"type": "integer", "description": "The version written."}, "hash": {"type": "string", "description": "Its hash."}, "path": {"type": ["string", "null"], "description": "The piece's folder relative to the workspace, pieces/<name>/<version>; null for a library or inline piece."}, "messages": {"type": "array", "items": {"$use": "compiler_message"}, "description": "The messages, errors first; each names the file in its message, and line and column are in that file."}}})json",
     R"json({"piece": {"name": "uv_field", "draw": {"camera": "none", "passes": [{"kind": "render", "fragment": "fs"}]}, "files": {"draw": "@fragment fn fs(in: CaliperFullscreen) -> @location(0) vec4<f32> { return vec4<f32>(in.uv, 0.0, 1.0); }"}}})json"},
    {"piece.remove", true, true, false,
     "Removes one version of a workspace piece and its folder. A version that a program of the open spec pins is "
     "piece.in_use, and a library piece is piece.read_only.",
     R"json({"type": "object", "additionalProperties": false, "required": ["name", "version"], "properties": {"name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "The piece's name."}, "version": {"type": "integer", "minimum": 1, "maximum": 1000000, "description": "The version to remove."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["name", "version", "removed"], "properties": {"name": {"type": "string", "description": "The piece's name."}, "version": {"type": "integer", "description": "The version."}, "removed": {"type": "boolean", "description": "Whether the folder is gone."}}})json",
     R"json({"name": "glow_points", "version": 1})json"},
    {"program.read", false, false, true,
     "Reads elements of a published resource of a program at a step, as the app's step history keeps it: the step "
     "of the version, its valid count, and each element's fields as numbers. It lets an agent check what its derive "
     "half wrote. It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "resource"], "properties": {"id": {"$use": "id", "maxLength": 55, "description": "The program's id."}, "resource": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,31}$", "description": "The resource's id."}, "step": {"type": "integer", "minimum": 0, "description": "The newest version at or before this step; the cursor's step when absent."}, "first": {"type": "integer", "minimum": 0, "description": "The first element; 0 when absent."}, "count": {"type": "integer", "minimum": 1, "maximum": 1024, "description": "The elements to read; 16 when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "resource", "step", "count", "elements", "kept"], "properties": {"id": {"type": "string", "description": "The program's id."}, "resource": {"type": "string", "description": "The resource's id."}, "step": {"type": ["integer", "null"], "description": "The step of the version, or null when no version is kept."}, "count": {"type": "integer", "minimum": 0, "description": "The valid elements of the version."}, "elements": {"type": "array", "maxItems": 1024, "items": {"type": "array", "items": {"type": "number"}}, "description": "Each element read, as the numbers of its fields in their order; a vector field gives one number for each component."}, "kept": {"type": "object", "additionalProperties": false, "required": ["versions", "first", "last"], "properties": {"versions": {"type": "integer", "minimum": 0, "description": "The count of kept versions."}, "first": {"type": ["integer", "null"], "description": "The step of the oldest kept version."}, "last": {"type": ["integer", "null"], "description": "The step of the newest kept version."}}, "description": "The versions that the step history keeps."}}})json",
     R"json({"id": "tower_arcs", "resource": "arcs", "count": 4})json"},

    // Interpretation commands use bounded CONTROL metadata.
    {"feature.dictionary", false, false, true,
     "Reads the registered dictionary metadata at a run boundary.",
     R"json({"type": "object", "additionalProperties": false, "required": [], "properties": {"request_id": {"type": "integer", "minimum": 1, "maximum": 4294967295, "description": "Poll a prior request of this command without another model operation."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["request_id", "pending", "result"], "properties": {"request_id": {"type": "integer", "minimum": 1, "maximum": 4294967295, "description": "The transport request ID."}, "pending": {"type": "boolean", "description": "True until the deferred reply arrives."}, "result": {"type": ["object", "null"], "additionalProperties": true, "description": "The bounded structured result, or null while pending."}}})json",
     R"json({})json"},
    {"feature.show", false, false, true,
     "Reads the real feature label, count, maximum, and promoted tokens.",
     R"json({"type": "object", "additionalProperties": false, "required": [], "properties": {"layer": {"type": "integer", "minimum": 0, "maximum": 4095, "description": "The dictionary layer."}, "feature": {"type": "integer", "minimum": 0, "maximum": 2147483647, "description": "The feature index."}, "node": {"type": "integer", "minimum": 0, "maximum": 8191, "description": "The selected graph row. Supply node or layer and feature."}, "request_id": {"type": "integer", "minimum": 1, "maximum": 4294967295, "description": "Poll a prior request of this command without another model operation."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["request_id", "pending", "result"], "properties": {"request_id": {"type": "integer", "minimum": 1, "maximum": 4294967295, "description": "The transport request ID."}, "pending": {"type": "boolean", "description": "True until the deferred reply arrives."}, "result": {"type": ["object", "null"], "additionalProperties": true, "description": "The bounded structured result, or null while pending."}}})json",
     R"json({"layer": 0, "feature": 0})json"},
    {"feature.examples", false, false, true,
     "Reads up to sixteen 33-token windows and publishes real rows and UTF-8 labels for token_strip.",
     R"json({"type": "object", "additionalProperties": false, "required": [], "properties": {"layer": {"type": "integer", "minimum": 0, "maximum": 4095, "description": "The dictionary layer."}, "feature": {"type": "integer", "minimum": 0, "maximum": 2147483647, "description": "The feature index."}, "node": {"type": "integer", "minimum": 0, "maximum": 8191, "description": "The selected graph row. Supply node or layer and feature."}, "limit": {"type": "integer", "minimum": 0, "maximum": 16, "description": "The maximum example count. The default is 16."}, "request_id": {"type": "integer", "minimum": 1, "maximum": 4294967295, "description": "Poll a prior request of this command without another model operation."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["request_id", "pending", "result"], "properties": {"request_id": {"type": "integer", "minimum": 1, "maximum": 4294967295, "description": "The transport request ID."}, "pending": {"type": "boolean", "description": "True until the deferred reply arrives."}, "result": {"type": ["object", "null"], "additionalProperties": true, "description": "The bounded structured result, or null while pending."}}})json",
     R"json({"node": 1})json"},
    {"trace.run", false, false, true,
     "Queues a trace for the next run boundary. Poll its request_id or use trace.status to read progress.",
     R"json({"type": "object", "additionalProperties": false, "required": [], "properties": {"prompt": {"type": "string", "minLength": 1, "maxLength": 4096, "description": "The prompt, at most 4096 UTF-8 bytes and 32 model tokens."}, "max_targets": {"type": "integer", "minimum": 0, "maximum": 5000, "description": "The feature target cap. The caliper limit is 512. The reference profile limit is 5000."}, "max_nodes": {"type": "integer", "minimum": 1, "maximum": 8192, "description": "The node cap. The caliper limit is 512. The reference profile limit is 8192."}, "max_edges": {"type": "integer", "minimum": 0, "maximum": 262144, "description": "The edge cap. The caliper limit is 8192. The reference profile limit is 262144."}, "request_id": {"type": "integer", "minimum": 1, "maximum": 4294967295, "description": "Poll a prior request of this command without another model operation."}, "profile": {"type": "string", "enum": ["caliper", "circuit_tracer_82996f14"], "default": "caliper", "description": "The trace semantics. Reference ceilings require circuit_tracer_82996f14."}, "max_n_logits": {"type": "integer", "minimum": 1, "maximum": 10, "default": 10, "description": "The maximum selected logit count."}, "desired_logit_prob": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.95, "description": "The cumulative probability target for selected logits."}, "node_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.8, "description": "The cumulative node influence threshold."}, "edge_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.98, "description": "The cumulative edge influence threshold."}, "batch_size": {"type": ["integer", "null"], "minimum": 1, "maximum": 512, "default": null, "description": "The logical target batch size, at most 64 for caliper or 512 for the reference profile. Null uses the profile default."}, "update_interval": {"type": "integer", "minimum": 1, "maximum": 128, "default": 4, "description": "The reference selection update interval."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["request_id", "pending", "result"], "properties": {"request_id": {"type": "integer", "minimum": 1, "maximum": 4294967295, "description": "The transport request ID."}, "pending": {"type": "boolean", "description": "True until the deferred reply arrives."}, "result": {"type": ["object", "null"], "additionalProperties": true, "description": "The bounded structured result, or null while pending."}}})json",
     R"json({"prompt": "Fact: the capital of the state containing Dallas is"})json"},
    {"trace.status", false, false, true,
     "Reads the last trace state without a model forward or a wait on the model.",
     R"json({"type": "object", "additionalProperties": false, "required": [], "properties": {"request_id": {"type": "integer", "minimum": 1, "maximum": 4294967295, "description": "Poll a prior request of this command without another model operation."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["request_id", "pending", "result"], "properties": {"request_id": {"type": "integer", "minimum": 1, "maximum": 4294967295, "description": "The transport request ID."}, "pending": {"type": "boolean", "description": "True until the deferred reply arrives."}, "result": {"type": ["object", "null"], "additionalProperties": true, "description": "The bounded structured result, or null while pending."}}})json",
     R"json({})json"},
    {"trace.read", false, false, true,
     "Reads a bounded page from the completed graph.",
     R"json({"type": "object", "additionalProperties": false, "required": [], "properties": {"node_offset": {"type": "integer", "minimum": 0, "maximum": 8192, "description": "The first node."}, "node_limit": {"type": "integer", "minimum": 0, "maximum": 256, "description": "The node page size. The default is 128."}, "edge_offset": {"type": "integer", "minimum": 0, "maximum": 262144, "description": "The first edge."}, "edge_limit": {"type": "integer", "minimum": 0, "maximum": 256, "description": "The edge page size. The default is 256."}, "request_id": {"type": "integer", "minimum": 1, "maximum": 4294967295, "description": "Poll a prior request of this command without another model operation."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["request_id", "pending", "result"], "properties": {"request_id": {"type": "integer", "minimum": 1, "maximum": 4294967295, "description": "The transport request ID."}, "pending": {"type": "boolean", "description": "True until the deferred reply arrives."}, "result": {"type": ["object", "null"], "additionalProperties": true, "description": "The bounded structured result, or null while pending."}}})json",
     R"json({"node_limit": 128})json"},

    {"patch.sweep", false, false, true,
     "Queues a patch sweep at the next run boundary.",
     R"json({"type": "object", "additionalProperties": false, "required": [], "properties": {"clean_prompt": {"type": "string", "description": "The clean prompt, at most 4096 UTF-8 bytes and 32 model tokens.", "minLength": 1, "maxLength": 4096}, "corrupted_prompt": {"type": "string", "description": "The corrupted prompt with the same token count.", "minLength": 1, "maxLength": 4096}, "metric": {"type": "object", "description": "Distinct output token IDs for the logit difference.", "additionalProperties": false, "required": ["positive_token", "negative_token"], "properties": {"positive_token": {"type": "integer", "description": "The output token ID.", "minimum": 0, "maximum": 2147483647}, "negative_token": {"type": "integer", "description": "The output token ID.", "minimum": 0, "maximum": 2147483647}}}, "request_id": {"type": "integer", "minimum": 1, "maximum": 4294967295, "description": "Poll a prior request without another model operation."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["request_id", "pending", "result"], "properties": {"request_id": {"type": "integer", "minimum": 1, "maximum": 4294967295, "description": "Poll a prior request without another model operation."}, "pending": {"type": "boolean", "description": "True until the final reply arrives."}, "result": {"type": ["object", "null"], "description": "The bounded result, or null while pending.", "additionalProperties": true}}})json",
     R"json({"clean_prompt": "AB", "corrupted_prompt": "AC", "metric": {"positive_token": 65, "negative_token": 66}})json"},
    {"trace.validate", false, false, true,
     "Queues validation of the latest trace and intervention snapshot.",
     R"json({"type": "object", "additionalProperties": false, "required": [], "properties": {"request_id": {"type": "integer", "minimum": 1, "maximum": 4294967295, "description": "Poll a prior request without another model operation."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["request_id", "pending", "result"], "properties": {"request_id": {"type": "integer", "minimum": 1, "maximum": 4294967295, "description": "Poll a prior request without another model operation."}, "pending": {"type": "boolean", "description": "True until the final reply arrives."}, "result": {"type": ["object", "null"], "description": "The bounded result, or null while pending.", "additionalProperties": true}}})json",
     R"json({})json"},

    // interventions (design/gpu-programs.md)
    {"intervene.points", false, false, true,
     "Lists the intervention points of the model that the attached script watches: each module's output by its dotted "
     "name, and each head of an attention module as <module>#<h>, with the module's class and the shape of its last "
     "output. Semantic probes retain kind module and add probe, layer and pattern metadata. It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "match": {"type": "string", "minLength": 1, "maxLength": 255, "description": "Only the points whose names match this glob, such as blocks.*.attn*; every point when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["points", "step"],
      "properties": {
        "points": {"type": "array", "items": {"$use": "intervention_point"}, "description": "The points, in the model's order."},
        "step": {"type": ["integer", "null"], "description": "The script's step, or null before the first."}}})json",
     R"json({"match": "blocks.0.*"})json"},
    {"intervene.set", false, false, true,
     "Sets one intervention on a point of the attached model, applied at the script's next step through the module "
     "hooks. Kinds are ablate, add, scale, clamp, and patch. Steer stays reserved. "
     "It lasts until intervene.clear, or from from_step until until_step. The result says it is "
     "pending; a run event of change intervene reports the step the script applied it at. intervene.points lists the "
     "points.",
     R"json({"$use": "intervention_spec"})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["point", "op", "pending"],
      "properties": {
        "point": {"type": "string", "description": "The point."},
        "op": {"type": "string", "description": "The op sent."},
        "pending": {"type": "boolean", "description": "True: the intervention waits for the script's next step."}}})json",
     R"json({"point": "blocks.0.attn#3", "op": "ablate"})json"},
    {"intervene.clear", false, false, true,
     "Clears the interventions on one point of the attached model, or every intervention with all true, at the "
     "script's next step. The result names the points cleared and says the change is pending; a run event of change "
     "intervene reports the step the script applied it at.",
     R"json({"type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "point": {"type": "string", "minLength": 1, "maxLength": 255, "description": "The point whose interventions to clear, from intervene.list."},
        "all": {"type": "boolean", "description": "true clears every intervention; point is then not given."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["cleared", "pending"],
      "properties": {
        "cleared": {"type": "array", "items": {"type": "string"}, "description": "The points whose interventions are cleared."},
        "pending": {"type": "boolean", "description": "True: the change waits for the script's next step."}}})json",
     R"json({"point": "blocks.0.attn#3"})json"},
    {"intervene.list", false, false, true,
     "Lists the interventions on the attached model with their points, ops, step ranges and states: pending until the "
     "script's next step, active, or ended after their last step. It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": [], "properties": {}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["interventions"],
      "properties": {
        "interventions": {"type": "array", "items": {"$use": "intervention"}, "description": "The interventions, oldest first."}}})json",
     "{}"},
};

// Turns an object schema into the schema of its merge patch: nothing is
// required, the id is left out (a patch cannot change it), and every other
// property may also be null (enums gain null too); nested objects become
// patches of their own.
json patch_of(const json& schema) {
    json out = schema;
    out["required"] = json::array();
    if (!out.contains("properties")) return out;
    out["properties"].erase("id");
    for (auto& item : out["properties"].items()) {
        json& p = item.value();
        if (p.contains("properties")) p = patch_of(p);
        // A property of any JSON value (a description and nothing else)
        // already admits null; any other property without a type is a string.
        bool any_value = !p.contains("type");
        for (auto k = p.begin(); any_value && k != p.end(); ++k) any_value = k.key() == "description";
        if (any_value) continue;
        json type = p.contains("type") ? p["type"] : json("string");
        if (type.is_string()) type = json::array({type});
        if (std::find(type.begin(), type.end(), json("null")) == type.end()) type.push_back("null");
        p["type"] = type;
        if (p.contains("enum") && std::find(p["enum"].begin(), p["enum"].end(), json(nullptr)) == p["enum"].end())
            p["enum"].push_back(nullptr);
    }
    return out;
}

json expand(const json& v, const std::map<std::string, json>& fragments, std::vector<std::string>& errors,
            int depth = 0) {
    if (depth > 32) {
        errors.push_back("A fragment nests too deeply.");
        return v;
    }
    if (v.is_object()) {
        for (const char* key : {"$use", "$patch"}) {
            if (!v.contains(key)) continue;
            const std::string name = v[key].is_string() ? v[key].get<std::string>() : std::string();
            const auto it = fragments.find(name);
            if (it == fragments.end()) {
                errors.push_back("No fragment is named " + name + ".");
                return v;
            }
            json out = expand(it->second, fragments, errors, depth + 1);
            if (std::string(key) == "$patch") out = patch_of(out);
            for (const auto& item : v.items())
                if (item.key() != key) out[item.key()] = expand(item.value(), fragments, errors, depth + 1);
            return out;
        }
        json out = json::object();
        for (const auto& item : v.items()) out[item.key()] = expand(item.value(), fragments, errors, depth + 1);
        return out;
    }
    if (v.is_array()) {
        json out = json::array();
        for (const auto& item : v) out.push_back(expand(item, fragments, errors, depth + 1));
        return out;
    }
    return v;
}

// What each command does (CommandEffect, design/command-layer.md, Agent
// permissions), by which an agent's mode decides it. Every command of the
// table is listed here; one that is not is a fault of the build, so a new
// command cannot go without a class. When in doubt a command is `change`,
// which asks.
using E = CommandEffect;
const std::pair<const char*, CommandEffect> kEffects[] = {
    {"app.info", E::read},
    {"app.applets", E::read},
    {"app.launch", E::change},  // changes the page the app shows, and may replace the open spec
    {"catalog.list", E::read},
    {"catalog.describe", E::read},
    {"catalog.sample", E::read},
    {"data.query", E::read},
    {"data.import", E::change},
    {"loader.list", E::read},
    {"loader.describe", E::read},
    {"loader.create", E::change},
    {"loader.update", E::change},
    {"loader.run", E::change},
    {"loader.delete", E::change},
    {"spec.new", E::change},  // replaces the open spec
    {"spec.open", E::change},
    {"spec.get", E::read},
    {"spec.set_text", E::edit},
    {"spec.validate", E::read},
    {"source.add", E::edit},
    {"source.patch", E::edit},
    {"source.remove", E::edit},
    {"panel.create", E::edit},
    {"panel.patch", E::edit},
    {"panel.remove", E::edit},
    {"panel.render", E::read},
    {"panel.data", E::read},
    {"layout.place", E::edit},
    {"layout.set", E::edit},
    {"preview.begin", E::read},  // a proposal: the workspace does not change
    {"preview.diff", E::read},
    {"preview.commit", E::edit},
    {"preview.discard", E::read},
    {"applet.save", E::change},
    {"applet.open", E::change},
    {"applet.list", E::read},
    {"review.status", E::read},
    {"capture.frame", E::read},
    {"control.create", E::edit},
    {"control.patch", E::edit},
    {"control.remove", E::edit},
    {"state.create", E::edit},
    {"state.patch", E::edit},
    {"state.remove", E::edit},
    {"state.set", E::change},  // session state: no undo step and no journal entry
    {"state.get", E::read},
    {"control.set", E::change},
    {"selection.set", E::change},
    {"selection.get", E::read},
    {"entity.set", E::edit},
    {"entity.remove", E::edit},
    {"annotation.set", E::edit},
    {"annotation.remove", E::edit},
    {"cursor.get", E::read},
    {"cursor.set", E::change},
    {"cursor.step", E::change},
    {"camera.get", E::read},
    {"camera.set", E::change},
    {"camera.save", E::edit},
    {"run.state", E::read},
    {"run.set_param", E::run},
    {"run.action", E::run},
    {"run.pause", E::run},
    {"run.list", E::read},
    {"run.start", E::run},
    {"run.fork", E::run},
    {"run.stop", E::run},
    {"run.logs", E::read},
    {"events.wait", E::read},
    {"program.create", E::edit},
    {"program.patch", E::edit},
    {"program.remove", E::edit},
    {"program.status", E::read},
    {"piece.check", E::read},
    {"piece.list", E::read},
    {"piece.show", E::read},
    {"piece.save", E::change},
    {"piece.copy", E::change},
    {"piece.update", E::change},
    {"piece.remove", E::change},
    {"program.read", E::read},
    {"intervene.points", E::read},
    {"feature.dictionary", E::read},
    {"feature.show", E::read},
    {"feature.examples", E::read},
    {"trace.run", E::run},
    {"trace.status", E::read},
    {"trace.read", E::read},
    {"patch.sweep", E::run},
    {"trace.validate", E::run},
    {"intervene.set", E::run},  // acts on the running script at its next step
    {"intervene.clear", E::run},
    {"intervene.list", E::read},
    {"history.undo", E::change},
    {"history.redo", E::change},
    {"history.list", E::read},
};

struct Table {
    std::vector<CommandDescriptor> commands;
    std::vector<std::string> errors;
};

json parse_or_note(const char* text, const std::string& where, std::vector<std::string>& errors) {
    json v = json::parse(text, nullptr, false);
    if (v.is_discarded()) errors.push_back(where + " is not valid JSON.");
    return v;
}

Table build() {
    Table t;
    try {
        std::map<std::string, json> fragments;
        for (const auto& f : kFragments) fragments[f.first] = parse_or_note(f.second, std::string("fragment ") + f.first, t.errors);
        for (const Row& row : kRows) {
            CommandDescriptor d;
            d.name = row.name;
            d.summary = row.summary;
            d.mutates = row.mutates;
            d.destructive = row.destructive;
            d.frame_thread = row.frame_thread;
            d.args = expand(parse_or_note(row.args, d.name + " args", t.errors), fragments, t.errors);
            d.result = expand(parse_or_note(row.result, d.name + " result", t.errors), fragments, t.errors);
            d.examples.push_back(parse_or_note(row.example, d.name + " example", t.errors));
            t.commands.push_back(std::move(d));
        }
        for (auto& d : history_command_descriptors()) t.commands.push_back(std::move(d));
        for (auto& d : t.commands) {
            const auto* e = std::find_if(std::begin(kEffects), std::end(kEffects),
                                         [&](const auto& x) { return d.name == x.first; });
            if (e == std::end(kEffects))
                t.errors.push_back(d.name + " has no effect: say whether it reads, edits the open spec or changes "
                                            "something else (kEffects).");
            else
                d.effect = e->second;
        }
        for (const auto& e : kEffects)
            if (std::none_of(t.commands.begin(), t.commands.end(), [&](const CommandDescriptor& d) { return d.name == e.first; }))
                t.errors.push_back(std::string(e.first) + " has an effect but is not in the table.");
    } catch (const std::exception& e) {
        t.errors.push_back(std::string("The table could not be built: ") + e.what());
    }
    return t;
}

const Table& table() {
    static const Table t = build();
    return t;
}

}  // namespace

const std::vector<CommandDescriptor>& command_table() { return table().commands; }

const std::vector<std::string>& command_table_build_errors() { return table().errors; }

const CommandDescriptor* find_command_descriptor(std::string_view name) {
    for (const auto& d : command_table())
        if (d.name == name) return &d;
    return nullptr;
}

json command_table_to_json() {
    json commands = json::array();
    for (const auto& d : command_table()) commands.push_back(command_descriptor_to_json(d));
    return json{{"version", 0}, {"commands", std::move(commands)}};
}

std::string command_table_json_text() {
    return command_table_to_json().dump(2, ' ', false, json::error_handler_t::replace) + "\n";
}

std::vector<std::pair<std::string, std::string>> command_table_open_paths() {
    return {{"spec.get", "result.json"}, {"panel.data", "result.data"},
            {"feature.dictionary", "result.result"}, {"feature.show", "result.result"},
            {"feature.examples", "result.result"}, {"trace.run", "result.result"},
            {"trace.status", "result.result"}, {"trace.read", "result.result"},
            {"patch.sweep", "result.result"}, {"trace.validate", "result.result"}};
}

}  // namespace caliper_builder
