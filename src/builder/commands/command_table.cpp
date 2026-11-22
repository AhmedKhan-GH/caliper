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

    {"transform", R"json({
      "type": "object", "additionalProperties": false, "required": ["op"],
      "properties": {
        "op": {"type": "string", "enum": ["slice", "reduce", "normalise"], "description": "slice selects along one axis; reduce reduces over axes; normalise rescales values."},
        "axis": {"type": ["integer", "array"], "minimum": -8, "maximum": 7, "minItems": 1, "maxItems": 8, "items": {"$use": "axis"},
                 "description": "The axis, from -8 to 7, counting from 0 (negative counts from the end). Required for slice and reduce; reduce also takes a list of 1 to 8 axes. normalise without an axis covers the whole tensor."},
        "index": {"$use": "safe_integer", "description": "slice: the one index to keep, negative counting from the end; it drops the axis. Not with start, stop or step."},
        "start": {"$use": "safe_integer", "description": "slice: the first index, Python style, negative allowed."},
        "stop": {"$use": "safe_integer", "description": "slice: the index to stop before, Python style."},
        "step": {"type": "integer", "minimum": 1, "maximum": 9007199254740991, "description": "slice: the step, at least 1."},
        "fn": {"type": "string", "enum": ["mean", "sum", "min", "max", "std", "var", "norm", "argmax", "argmin", "entropy"],
               "description": "reduce: the function; entropy treats the axis as a probability distribution, in nats; argmax and argmin take one axis."},
        "keepdims": {"type": "boolean", "description": "reduce: keep the reduced axes with size 1; false by default."},
        "mode": {"type": "string", "enum": ["minmax", "zscore", "softmax", "l2"], "description": "normalise: the rescaling."}}})json"},

    {"source", R"json({
      "type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {
        "id": {"$use": "id", "description": "The source's id in this spec, which panels bind."},
        "catalog": {"$use": "catalog_id", "maxLength": 256, "description": "A catalog source id, from catalog.list, such as live:attn.weights or table:run42_metrics."},
        "sql": {"type": "string", "minLength": 1, "maxLength": 65536, "description": "One read-only SELECT (DuckDB SQL) over the workspace's tables, named as in data.query."},
        "from": {"$use": "id", "description": "The id of another source of this spec that this one derives from, usually through transforms."},
        "transform": {"type": "array", "maxItems": 16, "items": {"$use": "transform"}, "description": "Up to 16 tensor transforms applied in order; only for tensor-shaped data."}}})json"},

    {"bind", R"json({
      "type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "x": {"$use": "binding", "description": "series: the x values, a binding such as loss.step; points: the x coordinates."},
        "y": {"type": ["string", "array"], "minLength": 1, "maxLength": 320, "pattern": "^[a-z][a-z0-9_-]{0,63}([.].+)?$",
              "minItems": 1, "maxItems": 16, "items": {"$use": "binding"},
              "description": "series: one binding or a list of 1 to 16 bindings drawn as lines; points: the y coordinates."},
        "z": {"$use": "binding", "description": "points: the z coordinates, for a 3-D cloud."},
        "color": {"$use": "binding", "description": "points: a value per point that sets its colour through the colormap."},
        "size": {"$use": "binding", "description": "points: a value per point that sets its size."},
        "positions": {"$use": "binding", "description": "points: a tensor of shape [n, 2] or [n, 3], instead of x, y and z."},
        "values": {"$use": "binding", "description": "heatmap: a rank-2 tensor; histogram: the values to count; image: a tensor [H, W], [H, W, 3] or [H, W, 4]."},
        "rows": {"$use": "id", "description": "table: the id of a tabular source whose rows are shown."},
        "value": {"$use": "binding", "description": "text: a column, shown as its last value, or a tensor of one element."},
        "scene": {"type": "string", "minLength": 6, "maxLength": 256, "pattern": "^live:.+$", "description": "scene (spec version 1): the live scene of the attached script, as live:<scene name>, such as live:model; catalog.list names the scenes."}}})json"},

    {"options", R"json({
      "type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "x_label": {"$use": "label", "description": "series, heatmap, points and histogram: the label of the x axis."},
        "y_label": {"$use": "label", "description": "series, heatmap, points and histogram: the label of the y axis."},
        "z_label": {"$use": "label", "description": "points: the label of the z axis."},
        "x_range": {"$use": "range", "description": "series: a fixed x range, [min, max] with min below max."},
        "y_range": {"$use": "range", "description": "series: a fixed y range, [min, max] with min below max."},
        "range": {"$use": "range", "description": "heatmap and image: the values at the ends of the colormap; histogram: the binned range; [min, max] with min below max."},
        "x_log": {"type": "boolean", "description": "series: a log scale on the x axis."},
        "y_log": {"type": "boolean", "description": "series: a log scale on the y axis; histogram: on the counts."},
        "colormap": {"type": "string", "enum": ["viridis", "magma", "rdbu", "greys"], "description": "heatmap, points and image: the colormap; viridis by default, greys for a one-channel image."},
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
        "scale_bar": {"type": "boolean", "description": "scene: whether a scale bar in the scene's units is drawn; false by default."}}})json"},

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
                              "title": {"type": ["string", "null"], "minLength": 1, "maxLength": 256, "description": "The node's tab label when it is a child of tabs, or null."},
                              "children": {"type": "array", "minItems": 1, "maxItems": 64, "items": {"$use": "id"}, "description": "Panel ids, control ids (spec version 1) and node ids, in order: 2 to 64 for a split, 1 to 64 for tabs; each node is used once."}}}}}})json"},

    // Spec version 1 controls (design/instruments.md, Control Table).
    {"control_bind", R"json({"type": "string", "minLength": 5, "maxLength": 320, "pattern": "^(param:.{1,63}|action:.{1,63}|run:pause|value)$",
      "description": "The one target of the control: param:<name> for a parameter the script registered with caliper.param, action:<name> for an action it registered with caliper.action, run:pause for the run's pause and resume, or value for a value the spec holds. A slider binds param: or value; a button binds action: or run:pause. run.state lists the names."})json"},

    {"control_options", R"json({
      "type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "min": {"type": "number", "description": "slider: the low end of the range; with param:, the parameter's own minimum when absent."},
        "max": {"type": "number", "description": "slider: the high end of the range, above min; with param:, the parameter's own maximum when absent."},
        "step": {"type": "number", "minimum": 0, "description": "slider: the step between values, above 0; continuous when absent."},
        "log": {"type": "boolean", "description": "slider: a log scale, which needs min above 0; with param:, the parameter's own scale when absent."},
        "value": {"type": "number", "description": "slider bound to value: the value it starts with; min when absent."}}})json"},

    {"control", R"json({
      "type": "object", "additionalProperties": false, "required": ["id", "kind", "title", "bind", "options"],
      "properties": {
        "id": {"$use": "id", "description": "The control's id."},
        "kind": {"type": "string", "enum": ["slider", "button", "range", "toggle", "picker", "text"], "description": "What the control is."},
        "title": {"type": ["string", "null"], "description": "The control's title, or null when it has none."},
        "bind": {"type": "string", "description": "The control's target, such as param:lr."},
        "options": {"$use": "control_options", "description": "The control's options."}}})json"},

    {"run_param", R"json({
      "type": "object", "additionalProperties": false, "required": ["name", "type", "value", "pending"],
      "properties": {
        "name": {"type": "string", "description": "The parameter's name, as the script registered it with caliper.param."},
        "type": {"type": "string", "enum": ["number", "integer", "boolean", "choice"], "description": "The parameter's type."},
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
        "kind": {"type": "string", "enum": ["pick", "select", "control", "run", "spec", "approval"], "description": "What happened."},
        "panel": {"type": ["string", "null"], "description": "The spec panel or control it happened in, or null."},
        "view": {"type": ["string", "null"], "description": "pick and select: the live view, such as the scene's name; null otherwise."},
        "origin": {"type": "string", "enum": ["ui", "builder", "agent", "mcp", "api", "script"], "description": "Who caused it: ui for the person in the app, script for the running script, else the origin of the command."},
        "entity": {"type": "string", "description": "pick: the entity path of the item picked."},
        "primitive": {"type": "string", "description": "pick: the scene part of the item, such as points or boxes."},
        "index": {"type": "integer", "minimum": 0, "description": "pick: the item's index in the batch the script sent."},
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
        "control": {"type": "string", "description": "control: the control's id."},
        "bind": {"type": "string", "description": "control: the control's target, such as param:lr."},
        "change": {"type": "string", "enum": ["state", "param", "action", "attach", "runner"], "description": "run: what changed: the run's state, a parameter applied at a step, the end of an action, the attachment of a script, or the state of a managed run (runner)."},
        "state": {"type": "string", "description": "run with change state: the run's new state; approval: pending, approved, denied or expired."},
        "step": {"type": ["integer", "null"], "description": "run: the script's step when it changed, or null before the first step."},
        "name": {"type": "string", "description": "run with change param or action: the parameter's or the action's name."},
        "value": {"type": ["number", "integer", "boolean", "string", "null"], "description": "control: the control's new value; run with change param: the value the script applied."},
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
        "error": {"type": ["string", "null"], "description": "run with change runner: why the run failed, else null."}}})json"},

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
        "patch": {"$patch": "source", "description": "The keys to change: catalog, sql, from or transform; null removes one."},
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
                    "options": {"$patch": "options", "description": "Options to set, or null to clear one."}}},
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
     "horizontally or vertically between its children, with ratio the share of the first child, or shows them in "
     "tabs. A child is a panel id or another node's id, and root names the top node, or the one panel when there "
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
     "Adds a control to the open spec: a slider or a button that the person and you can use. bind names its one "
     "target: param:<name> for a parameter of the running script, action:<name> for an action of it, run:pause for "
     "pausing and resuming the run, or value for a value the spec holds. run.state lists the parameters and actions. "
     "A control is a leaf of the layout, like a panel: place puts it next to a panel or control, and without place it "
     "goes under the whole layout. Undo removes it. The result holds the change as a diff; with dry_run true the "
     "change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "kind", "bind"],
      "properties": {
        "id": {"$use": "id", "description": "The new control's id; panels and controls share one set of ids."},
        "kind": {"type": "string", "enum": ["slider", "button", "range", "toggle", "picker", "text"], "description": "slider sets a number in a range; button acts once when pressed. range, toggle, picker and text are reserved and refused with spec.unsupported_control until they are built."},
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
                    "kind": {"type": "string", "enum": ["slider", "button", "range", "toggle", "picker", "text"], "description": "What the control is."},
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
        "value": {"type": ["number", "integer", "boolean", "string"], "description": "The new value: a number for a number or integer parameter, true or false for a boolean, one of the choices for a choice."},
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
     "Starts a Python script of the workspace as a managed run, for training or for inference, with arguments and "
     "parameter values, and attaches the app to it. It always waits for the person's approval on a review card, "
     "because the script runs as Python code. The result comes when the process starts, in the state starting; a run "
     "event of change runner, through events.wait, or run.list says when it runs. The script's caliper.watch() line "
     "writes the stream link that the app attaches to.",
     R"json({"type": "object", "additionalProperties": false, "required": ["script"],
      "properties": {
        "script": {"type": "string", "minLength": 1, "maxLength": 1024, "description": "The script's path relative to the workspace, with / between folders and no ., .. or link in it, such as scripts/train.py, so that the review card names the file that runs; another path to a file of the workspace fails with run.script_refused, whose message gives the path to use."},
        "args": {"type": "array", "maxItems": 64, "items": {"type": "string", "maxLength": 256}, "description": "The script's command-line arguments, each at most 256 bytes and at most 1024 bytes in all, so that the review card shows them in full; none when absent."},
        "params": {"$use": "run_params"},
        "id": {"$use": "run_id", "description": "The new run's id, unique in the workspace; the script's file name and a number, such as train-1, when absent."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["run"],
      "properties": {
        "run": {"$use": "run_entry", "description": "The run as run.list gives it."}}})json",
     R"json({"script": "train.py", "args": ["--epochs", "5"], "params": [{"name": "lr", "value": 0.01}]})json"},
    {"run.fork", false, false, true,
     "Starts a new managed run from the next checkpoint of a running run, with changed parameters, beside its "
     "parent: the parent saves a checkpoint through its caliper.checkpoint action, and the fork runs the same script "
     "and arguments from it. It always waits for the person's approval on a review card. The result comes at once, "
     "in the state starting; a run event of change runner or run.list says when the fork runs.",
     R"json({"type": "object", "additionalProperties": false, "required": ["run"],
      "properties": {
        "run": {"$use": "run_id", "description": "The parent: a managed run that runs or is paused."},
        "params": {"$use": "run_params", "description": "The parameters that change in the fork; the checkpoint holds the others."},
        "id": {"$use": "run_id", "description": "The fork's id, unique in the workspace; the parent's script name and a number when absent."}}})json",
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
     "spec and approvals. It answers at once when events wait, and otherwise holds the call until one comes or "
     "timeout_s ends. Pass the cursor of each result to the next call. The app keeps the last 1024 events; lost "
     "counts those that were dropped before you read them. Hovers are not events. It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "cursor": {"type": "integer", "minimum": 0, "default": 0, "description": "The cursor of the previous result; 0, the default, returns the oldest events the app still keeps."},
        "kinds": {"type": "array", "minItems": 1, "maxItems": 6, "items": {"type": "string", "enum": ["pick", "select", "control", "run", "spec", "approval"]}, "description": "Only these kinds; every kind when absent."},
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
    return {{"spec.get", "result.json"}, {"panel.data", "result.data"}};
}

}  // namespace caliper_builder
