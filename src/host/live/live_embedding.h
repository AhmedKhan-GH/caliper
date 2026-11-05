#pragma once
// The Embedding pane: every row of the model's embedding table as a point in 3-D, laid out by the host (UMAP of the
// normalised rows, or their first principal components) and coloured by the kind of token it is. All the rows are
// drawn from one imported GPU buffer, grouped by kind, each point opaque in its kind's colour and growing as the
// camera closes in. The legend hides or shows a kind, and hovering a point names its token.
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace caliper_host { class HostRenderer; }

namespace caliper_live {

struct EmbeddingMap {
    std::string method;                  // "umap" or "pca"
    std::vector<std::string> kind_names;
    std::vector<float> xyz;              // 3 per row, within [-1, 1]
    std::vector<uint8_t> kinds;          // per row, an index into kind_names
    size_t rows() const { return kinds.size(); }
};

// The host's reply payloads: int16 positions over [-1, 1] (3 per row) and one kind byte per row. False if the
// sizes disagree with count or a kind is out of range.
bool read_embedding_map(int count, const std::vector<uint8_t>& positions, const std::vector<uint8_t>& kinds,
                        size_t kind_count, EmbeddingMap& out);

class EmbeddingView {
public:
    EmbeddingView();
    ~EmbeddingView();
    void set(EmbeddingMap map);
    void status(const std::string& text);    // progress while the host lays the map out
    void failed(const std::string& error);
    bool loaded() const;
    std::string subtitle() const;
    // Draws into the rest of the current window. words holds token texts by id; returns a hovered token whose text
    // is not there yet, for the inspector to ask the host for (-1 for none).
    int draw(caliper_host::HostRenderer* renderer, const std::map<int32_t, std::string>& words);
    bool requested = false;                  // the inspector asked the host for the map

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace caliper_live
