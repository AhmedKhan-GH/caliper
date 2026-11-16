// The tabular readers through DuckDB.
#include "tabular.h"

#include <algorithm>
#include <cctype>
#include <filesystem>

namespace caliper_builder {
namespace detail {

namespace {

std::string lower_extension(const std::string& path) {
    std::string ext = std::filesystem::u8path(path).extension().u8string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return ext;
}

// Cuts a single statement at its terminating semicolon, so that it can sit
// inside parentheses. The statement already parsed as one SELECT, so the first
// semicolon outside strings, quoted identifiers and comments is its end. A
// misjudged cut can only give SQL that fails the CREATE check, never more SQL.
std::string cut_terminator(const std::string& sql) {
    const std::size_t n = sql.size();
    std::size_t i = 0;
    auto skip_quoted = [&](char q) {
        ++i;
        while (i < n) {
            if (sql[i] == q) {
                if (i + 1 < n && sql[i + 1] == q) {
                    i += 2;
                    continue;
                }
                ++i;
                return;
            }
            ++i;
        }
    };
    while (i < n) {
        const char c = sql[i];
        if (c == '\'' || c == '"') {
            skip_quoted(c);
        } else if (c == '-' && i + 1 < n && sql[i + 1] == '-') {
            while (i < n && sql[i] != '\n') ++i;
        } else if (c == '/' && i + 1 < n && sql[i + 1] == '*') {
            int depth = 1;
            i += 2;
            while (i < n && depth > 0) {
                if (sql[i] == '/' && i + 1 < n && sql[i + 1] == '*') {
                    ++depth;
                    i += 2;
                } else if (sql[i] == '*' && i + 1 < n && sql[i + 1] == '/') {
                    --depth;
                    i += 2;
                } else {
                    ++i;
                }
            }
        } else if (c == '$') {
            std::size_t j = i + 1;
            while (j < n && (std::isalnum(static_cast<unsigned char>(sql[j])) || sql[j] == '_')) ++j;
            if (j < n && sql[j] == '$') {
                const std::string tag = sql.substr(i, j - i + 1);
                const std::size_t end = sql.find(tag, j + 1);
                i = end == std::string::npos ? n : end + tag.size();
            } else {
                ++i;
            }
        } else if (c == ';') {
            return sql.substr(0, i);
        } else {
            ++i;
        }
    }
    return sql;
}

}  // namespace

Error tabular_relation(Db& db, const TabularRead& read, std::string* sql) {
    if (read.files.empty()) return Error("loader.no_match", "The reader has no files to read.");
    if (Error e = validate_reader_options(read.reader, read.options)) return e;
    std::string files = "[";
    for (std::size_t i = 0; i < read.files.size(); ++i) {
        if (!sql_text_ok(read.files[i], false)) {
            return Error("loader.bad_value", "A file path holds a control character.", read.files[i]);
        }
        if (i) files += ", ";
        files += sql_literal(read.files[i]);
    }
    files += "]";
    const nlohmann::json& o = read.options;
    std::string params;
    auto add = [&](const std::string& p) { params += ", " + p; };
    auto boolean = [](const nlohmann::json& v) { return v.get<bool>() ? std::string("true") : std::string("false"); };
    if (o.contains("union_by_name")) add("union_by_name = " + boolean(o["union_by_name"]));
    std::string function;
    switch (read.reader) {
        case ReaderKind::csv:
            function = "read_csv";
            if (o.contains("header")) add("header = " + boolean(o["header"]));
            if (o.contains("delimiter")) add("delim = " + sql_literal(o["delimiter"].get<std::string>()));
            else if (lower_extension(read.files[0]) == ".tsv") add("delim = " + sql_literal("\t"));
            if (o.contains("quote")) add("quote = " + sql_literal(o["quote"].get<std::string>()));
            if (o.contains("escape")) add("escape = " + sql_literal(o["escape"].get<std::string>()));
            if (o.contains("skip")) add("skip = " + std::to_string(o["skip"].get<int64_t>()));
            if (o.contains("null")) add("nullstr = " + sql_literal(o["null"].get<std::string>()));
            break;
        case ReaderKind::parquet: function = "read_parquet"; break;
        case ReaderKind::json:
        case ReaderKind::jsonl:
            if (!db.has_table_function("read_json")) {
                return Error("loader.json_not_available",
                             "This build's DuckDB has no JSON extension, so it cannot read JSON files; convert the "
                             "file to CSV or Parquet.");
            }
            function = "read_json";
            if (read.reader == ReaderKind::jsonl) add("format = 'newline_delimited'");
            else if (o.contains("format")) add("format = " + sql_literal(o["format"].get<std::string>()));
            break;
        case ReaderKind::arrow:
            return Error("loader.arrow_not_available",
                         "Arrow IPC files are not read natively yet; convert the file to Parquet, or use the Python "
                         "loader.");
        default:
            return Error("loader.bad_value", std::string("The ") + reader_name(read.reader) + " reader is not tabular.");
    }
    *sql = "SELECT * FROM " + function + "(" + files + params + ")";
    return {};
}

Error make_table_output(Db& db, const TabularRead& read, const TableOutput& output, TableMade* made) {
    std::string relation;
    if (Error e = tabular_relation(db, read, &relation)) return e;
    std::string body = relation;
    if (output.sql) {
        if (Error e = db.check_select(*output.sql)) {
            e.where = "sql";
            return e;
        }
        body = "WITH input AS (" + relation + ")\nSELECT * FROM (\n" + cut_terminator(*output.sql) + "\n)";
    }
    const std::string name = sql_identifier(output.name);
    const std::string statement = std::string("CREATE ") + (output.copy ? "TABLE " : "VIEW ") + name + " AS " + body;
    if (Error e = db.check_create(statement)) {
        return Error("loader.sql", "The output's SQL does not fit inside a view; write it as one plain SELECT.", "sql");
    }
    if (Error e = db.drop_relation(output.name)) return e;
    if (Error e = db.execute(statement)) return e;
    if (Error e = db.describe("SELECT * FROM " + name, &made->columns)) return e;
    return db.scalar("SELECT count(*) FROM " + name, &made->rows);
}

}  // namespace detail
}  // namespace caliper_builder
