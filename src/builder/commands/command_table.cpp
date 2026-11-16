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

    {"catalog_id", R"json({"type": "string", "pattern": "^(table|tensor|live|feed|metric):.+$", "minLength": 3,
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
        "kind": {"type": "string", "enum": ["table", "tensor", "live", "feed", "metric"],
                 "description": "table: a DuckDB table or view, such as a loader made; tensor: a tensor read from a file, an image or an audio clip; live: a live view of the attached live session; feed: a channel of live telemetry; metric: a run and tag of logged metrics."},
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
        "kind": {"type": "string", "enum": ["table", "tensor", "live", "feed", "metric"], "description": "The source's kind."},
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
        "value": {"$use": "binding", "description": "text: a column, shown as its last value, or a tensor of one element."}}})json"},

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
        "text": {"type": "string", "maxLength": 16384, "description": "text: static text to show."}}})json"},

    {"panel", R"json({
      "type": "object", "additionalProperties": false, "required": ["id", "primitive", "title", "bind", "options"],
      "properties": {
        "id": {"$use": "id", "description": "The panel's id."},
        "primitive": {"type": "string", "enum": ["series", "heatmap", "points", "histogram", "table", "image", "text"], "description": "What the panel draws."},
        "title": {"type": ["string", "null"], "description": "The panel's title, or null when it has none."},
        "bind": {"$use": "bind", "description": "The panel's bindings, by slot."},
        "options": {"$use": "options", "description": "The panel's options."}}})json"},

    {"layout", R"json({
      "type": "object", "additionalProperties": false, "required": ["root", "nodes"],
      "properties": {
        "root": {"$use": "id", "description": "The id of the top node, or of the one panel when the layout is a single panel."},
        "nodes": {"type": "array", "maxItems": 256, "description": "The nodes of the layout, in any order. Each splits its space between its children or shows them as tabs.",
                  "items": {"type": "object", "additionalProperties": false, "required": ["id", "split", "children"],
                            "properties": {
                              "id": {"$use": "id", "description": "The node's id; it must not be a panel's id."},
                              "split": {"type": "string", "enum": ["horizontal", "vertical", "tabs"], "description": "horizontal puts the children side by side, vertical one above another, tabs in tabs."},
                              "ratio": {"type": ["number", "null"], "minimum": 0.05, "maximum": 0.95, "description": "For a split of exactly two children, the share of the first, from 0.05 to 0.95; null for an even split and for tabs."},
                              "title": {"type": ["string", "null"], "minLength": 1, "maxLength": 256, "description": "The node's tab label when it is a child of tabs, or null."},
                              "children": {"type": "array", "minItems": 1, "maxItems": 64, "items": {"$use": "id"}, "description": "Panel ids and node ids, in order: 2 to 64 for a split, 1 to 64 for tabs; each node is used once."}}}}}})json"},

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
     "folder, its status (READY, ACTIVE for the one open now, REFUSED, SHADOWED) and the reason when it cannot run. "
     "The id is what app.launch takes. It changes nothing.",
     kNoArgs,
     R"json({"type": "object", "additionalProperties": false, "required": ["applets"],
      "properties": {"applets": {"type": "array", "description": "The applets, in the app's search order.",
        "items": {"type": "object", "additionalProperties": false, "required": ["folder", "id", "reason", "status", "version"],
          "properties": {
            "folder": {"type": "string", "description": "The folder the applet was found in."},
            "id": {"type": "string", "description": "The applet's manifest id."},
            "reason": {"type": "string", "description": "Why the applet cannot run; empty for a READY applet."},
            "status": {"type": "string", "description": "READY, ACTIVE, REFUSED or SHADOWED, as caliper list prints it."},
            "version": {"type": "string", "description": "The applet's version."}}}}}})json",
     "{}"},
    {"app.launch", false, false, true,
     "Opens the page of one applet, as a click on its row on the app's landing screen does. The page shown before "
     "is closed first, and an attached live session is detached. The id is an applet's manifest id from "
     "app.applets.",
     R"json({"type": "object", "additionalProperties": false, "required": ["id"],
      "properties": {"id": {"type": "string", "pattern": "^[A-Za-z0-9._-]{1,128}$", "minLength": 1, "maxLength": 128,
                            "description": "The applet's manifest id, such as com.example.my-viewer, from app.applets."}}})json",
     R"json({"type": "object", "additionalProperties": false, "required": ["already_open", "id", "status", "version"],
      "properties": {
        "already_open": {"type": "boolean", "description": "Whether the applet's page was already open."},
        "id": {"type": "string", "description": "The applet's manifest id."},
        "status": {"type": "string", "description": "The applet's status after the launch, ACTIVE when it opened."},
        "version": {"type": "string", "description": "The applet's version."}}})json",
     R"json({"id": "com.example.my-viewer"})json"},

    // catalog
    {"catalog.list", false, false, false,
     "Lists the sources in the catalog: every table, tensor, live view, feed channel and metric that a spec can "
     "bind, with its schema, shape and rate. A source id is <kind>:<name>, such as table:run42_metrics or "
     "live:attn.weights; a spec source names it in its catalog field. Filter by kind or by the start of the name. "
     "It changes nothing.",
     R"json({"type": "object", "additionalProperties": false, "required": [],
      "properties": {
        "kind": {"type": "string", "enum": ["table", "tensor", "live", "feed", "metric"], "description": "Only sources of this kind."},
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
     "saved to or opened from, and whether it has unsaved changes. It changes nothing.",
     kNoArgs,
     R"json({"type": "object", "additionalProperties": false, "required": ["id", "path", "dirty", "text", "json"],
      "properties": {
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
        "primitive": {"type": "string", "enum": ["series", "heatmap", "points", "histogram", "table", "image", "text"], "description": "What the panel draws. Required slots: series y; heatmap values; points x and y, or positions; histogram values; table rows; image values; text none."},
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
                    "primitive": {"type": "string", "enum": ["series", "heatmap", "points", "histogram", "table", "image", "text"], "description": "What the panel draws."},
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

    // layout
    {"layout.place", true, false, true,
     "Places a panel of the open spec next to another one, the target: beside puts it to the right, above and "
     "below above or below, tabbed in tabs with the target. ratio is the share of the space the placed panel gets, "
     "from 0.05 to 0.95, half by default. A panel already in the layout is moved. Undo restores the previous layout."
     " The result holds the change as a diff; with dry_run true the change is only checked and not applied.",
     R"json({"type": "object", "additionalProperties": false, "required": ["panel", "target", "where"],
      "properties": {
        "panel": {"$use": "id", "description": "The id of the panel to place."},
        "target": {"$use": "id", "description": "The id of the panel to place it next to."},
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

    // capture
    {"capture.frame", false, false, true,
     "Returns the frame the app draws, the whole window, as a PNG image, so that you can see the app as the person "
     "does. It changes nothing.",
     kNoArgs, R"json({"$use": "png"})json", "{}"},
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

std::vector<std::pair<std::string, std::string>> command_table_open_paths() { return {{"spec.get", "result.json"}}; }

}  // namespace caliper_builder
