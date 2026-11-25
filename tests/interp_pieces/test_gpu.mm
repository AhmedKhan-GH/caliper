#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include "../gpu_skip.h"
#include "../boxes_match/match_rig.h"
#include "../draw_parity/parity_fixture.h"
#include "pieces.h"
#include "draw/draw_host.h"
#include "live/live_scene.h"
#include "live/live_views_scene.h"
#include "gfx/shaders/generated/programs_metal.h"
#include "spec_runtime/spec_offscreen.h"
#include <filesystem>
#include <fstream>
#include <set>

namespace ip = interp_pieces;
namespace mr = match_rig;
namespace dp = draw_parity;
namespace cd = caliper_draw;
namespace cp = caliper_builder::programs;
namespace cs = caliper_builder::spec;
namespace wire = caliper_net::wire;
using caliper_net::Json;
using mr::HostTensor;
using mr::Published;
namespace {
constexpr unsigned size = 512;
const Published& resource(const std::vector<Published>& resources, const std::string& name) {
    for (const auto& r : resources) if (r.id == name) return r;
    throw std::runtime_error("Missing resource: " + name);
}
template<class T> T scalar(const Published& r, unsigned row, unsigned offset, unsigned stride) {
    REQUIRE((row + 1) * stride <= r.bytes.size());
    T value;
    std::memcpy(&value, r.bytes.data() + row * stride + offset, sizeof(value));
    return value;
}
HostTensor text_tensor(unsigned rows, const std::string& prefix) {
    std::vector<std::uint32_t> words(rows * 16, 0);
    for (unsigned i = 0; i < rows; ++i) {
        const std::string text = prefix + std::to_string(i);
        std::memcpy(words.data() + 16 * i, text.data(), std::min<std::size_t>(text.size(), 64));
    }
    return HostTensor::of("label_text", "u32", {rows, 16}, words);
}
std::vector<HostTensor> graph_tensors(bool dense = false) {
    const unsigned count = dense ? 512 : 16;
    std::vector<std::int32_t> ni(512*4), ei(8192*2), counts{int(count), dense ? 0 : 18, 4, 4};
    std::vector<float> nf(512*4), ef(8192*2);
    for (unsigned i = 0; i < count; ++i) {
        unsigned kind = i < 4 ? 0 : i < 12 ? 1 : i < 14 ? 2 : 3;
        if (dense) kind = 1;
        ni[4*i] = kind;
        ni[4*i+1] = dense ? 1 : kind == 0 ? -1 : kind == 3 ? 4 : kind == 2 ? 2 : int((i-4)/4);
        ni[4*i+2] = dense ? 2 : kind == 3 ? 3 : i % 4;
        ni[4*i+3] = 100+i;
        nf[4*i] = 1;
        nf[4*i+1] = float((i*7)%11); // Equal influence exercises the row-index tie break.
    }
    for (unsigned i = 0; i < unsigned(counts[1]); ++i) {
        ei[2*i] = i < 12 ? int(i) : int(i-8);
        ei[2*i+1] = i < 12 ? int(std::min(i+4,15u)) : 15;
        ef[2*i] = i % 2 ? -0.5f : 1.0f;
        ef[2*i+1] = ef[2*i];
    }
    return {HostTensor::of("graph_counts", "i32", {4}, counts), HostTensor::of("node_i", "i32", {512,4}, ni),
            HostTensor::of("node_f", "f32", {512,4}, nf), HostTensor::of("edge_i", "i32", {8192,2}, ei),
            HostTensor::of("edge_f", "f32", {8192,2}, ef), text_tensor(544,"label ")};
}
std::vector<HostTensor> narrow_graph_tensors() {
    auto tensors=graph_tensors(true);
    std::vector<std::int32_t> ni(512*4), counts{16,0,32,24};
    for (unsigned i=0;i<16;++i) {
        ni[4*i]=1; ni[4*i+1]=12; ni[4*i+2]=16; ni[4*i+3]=100+i;
    }
    tensors[0]=HostTensor::of("graph_counts","i32",{4},counts);
    tensors[1]=HostTensor::of("node_i","i32",{512,4},ni);
    return tensors;
}
cd::FrameInputs frame(const cs::Piece& piece, const Json& params = Json::object(), unsigned side = size) {
    cd::FrameInputs in;
    in.clip_from_world = in.world_from_clip = in.view_from_world = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    in.viewport = {0,0,float(side),float(side)};
    in.pixel_ratio = 1;
    in.cursor_step = in.latest_step = 1;
    in.live = true;
    in.params = caliper_web::pack_params(piece,params);
    return in;
}
void set_resources(mr::PieceDraw& draw, const cs::Piece& piece, const std::vector<Published>& data) {
    draw.install(piece,piece.name);
    for (const auto& r : data) draw.set(r.id,r.bytes,r.count,&r);
}
std::pair<int,int> pixel(float x, float y) { return {int((x+1)*size*0.5f),int((1-y)*size*0.5f)}; }
void check_pick(const mr::PieceDraw& draw, float x, float y, unsigned item) {
    const auto [px,py] = pixel(x,y);
    REQUIRE(px >= 0); REQUIRE(py >= 0); REQUIRE(px < int(size)); REQUIRE(py < int(size));
    CHECK(mr::identity_of(draw.ids,size,px,py) == item+1);
    const auto pick = draw.rt.pick(item);
    REQUIRE(pick);
    CHECK(pick->item == item);
}
void no_overlap(const Published& nodes) {
    struct Placed { float x, y, radius, influence; unsigned index; };
    std::map<std::pair<int,int>,std::vector<Placed>> cells;
    for (unsigned i=0;i<nodes.count;++i) {
        const float x=scalar<float>(nodes,i,0,48), y=scalar<float>(nodes,i,4,48), r=scalar<float>(nodes,i,12,48);
        REQUIRE(std::isfinite(x)); REQUIRE(std::isfinite(y)); REQUIRE(r>0);
        const auto key=std::make_pair(scalar<int>(nodes,i,24,48),scalar<int>(nodes,i,28,48));
        cells[key].push_back({x,y,r,scalar<float>(nodes,i,36,48),i});
    }
    for (auto& [key,placed]:cells) {
        std::sort(placed.begin(),placed.end(),[](const Placed& a,const Placed& b){
            return a.y == b.y ? a.x < b.x : a.y > b.y;
        });
        for (unsigned i=1;i<placed.size();++i) {
            const auto& a=placed[i-1]; const auto& b=placed[i];
            CHECK(a.influence>=b.influence);
            if (a.influence==b.influence) CHECK(a.index<b.index);
        }
        for (unsigned i=0;i<placed.size();++i) for (unsigned j=0;j<i;++j) {
            const auto& a=placed[i]; const auto& b=placed[j];
            // Disjoint bounds also cover the square, diamond, and triangle silhouettes.
            CHECK(std::max(std::abs(b.x-a.x),std::abs(b.y-a.y))>a.radius+b.radius);
        }
    }
}
void write(const std::filesystem::path& path, const void* bytes, std::size_t n) {
    std::ofstream f(path,std::ios::binary);
    f.write(static_cast<const char*>(bytes),std::streamsize(n));
    REQUIRE(f.good());
}
struct Export {
    dp::Plan plan;
    std::vector<std::uint8_t> frames;
    Json expected = Json::object();
    std::filesystem::path out;
    explicit Export(unsigned side = size, const std::string& subfolder = "") {
        const char* folder = std::getenv("CALIPER_INTERP_PIECES_OUT");
        out = folder ? folder : CALIPER_INTERP_PIECES_DEFAULT_OUT;
        if (!subfolder.empty()) out /= subfolder;
        std::filesystem::create_directories(out / "app");
        plan.size = side; plan.frames = 1; plan.step = 1;
    }
    void add_frame(const wire::Message& message) {
        std::vector<std::uint8_t> bytes;
        std::string why;
        REQUIRE_MESSAGE(wire::encode(message,&bytes,&why),why);
        wire::Message decoded;
        const auto result = wire::decode(bytes.data(), bytes.size(), &decoded);
        REQUIRE_MESSAGE(result.ok(), result.reason, " ", result.message);
        dp::append_frame(&frames,bytes);
    }
    void add(const std::string& id, const cs::Piece& piece, const std::vector<Published>& data,
             const mr::PieceDraw& draw, const Json& values, const Json& picks) {
        wire::Program message;
        std::string why;
        REQUIRE(dp::instance_message(piece,id,id,values,piece.hash,&message,&why));
        if (values.size() == 0) message.params = Json();
        message.bind = Json::object();
        for (const auto& input : piece.inputs) message.bind.set(input.name,"live:"+input.name);
        cs::Piece decoded;
        cd::Instance instance;
        const auto verdict = caliper_web::decode_program(message, &decoded, &instance);
        REQUIRE_MESSAGE(verdict.ok(), verdict.code, " ", verdict.message);
        add_frame(message);
        for (const auto& r : data) {
            auto m = dp::resource_message(id,r.id,r.bytes,1);
            m.count = r.count;
            cp::ElementLayout layout;
            for (const auto& desc : piece.resources) if (desc.id == r.id) REQUIRE(cp::element_layout(desc.layout,&layout).ok());
            m.stride = layout.stride;
            add_frame(m);
        }
        {
            // The actual browser spec page consumes these same files and resource bytes.
            nlohmann::json viewer = {{"id", id}, {"title", "Attribution Graph"}, {"state", "ready"}, {"error", nullptr},
                {"piece", {{"name", piece.name}, {"version", piece.version}, {"from", "library"}}},
                {"params", nlohmann::json::parse(values.dump())}, {"resources", nlohmann::json::array()}};
            viewer["instance"] = {{"op", "set"}, {"id", id}, {"panel", id},
                {"piece", nlohmann::json::parse(message.piece.dump())},
                {"manifest", nlohmann::json::parse(message.manifest.dump())},
                {"bind", nlohmann::json::parse(message.bind.dump())}, {"params", viewer["params"]},
                {"segments", nlohmann::json::parse(message.segments.dump())},
                {"common", message.common.value_or("")}, {"derive", message.derive.value_or("")}, {"draw", message.draw.value_or("")}};
            for (const auto& r : data) {
                cp::ElementLayout layout;
                for (const auto& desc : piece.resources)
                    if (desc.id == r.id) REQUIRE(cp::element_layout(desc.layout, &layout).ok());
                viewer["resources"].push_back({{"resource", r.id}, {"step", 1}, {"version", "1"},
                    {"count", r.count}, {"stride", layout.stride}, {"width", 0}, {"height", 0}, {"bytes", r.bytes}});
            }
            const auto json = viewer.dump();
            write(out / (id + ".viewer.json"), json.data(), json.size());
            if (id == "graph_selected") write(out / "hover.json", json.data(), json.size());
        }
        dp::ProgramPlan pp;
        pp.id = id; pp.pick_id = mr::kPiecePick; pp.inputs = frame(piece,values,plan.size);
        plan.programs.push_back(pp);
        expected.set(id,picks);
        write(out / "app" / (id+".color"),draw.color.data(),draw.color.size());
        write(out / "app" / (id+".id"),draw.ids.data(),draw.ids.size());
        REQUIRE(stbi_write_png((out / "app" / (id+".png")).string().c_str(),plan.size,plan.size,4,draw.color.data(),plan.size*4));
    }
    void finish() {
        const auto json = dp::plan_json(plan), picks = expected.dump();
        write(out/"frames.bin",frames.data(),frames.size());
        write(out/"plan.json",json.data(),json.size());
        write(out/"picks.json",picks.data(),picks.size());
    }
};
Export& exporter() { static Export output; return output; }
Json pick_at(float x, float y, unsigned item) {
    const auto [px,py] = pixel(x,y);
    Json row=Json::array(); row.push(px); row.push(py); row.push(std::uint64_t(item)); return row;
}
}

TEST_CASE("reference graph capacities retain nodes and edges beyond the legacy limits") {
    @autoreleasepool {
        mr::Metal metal;
        if (!metal.backend) CALIPER_SKIP_NO_GPU(metal.why);
        const auto piece = ip::load("attribution_graph");
        constexpr unsigned nodes = 513, edges = 8193;
        std::vector<std::int32_t> ni(8192 * 4), ei(262144 * 2), counts{nodes, edges, 32, 24};
        std::vector<float> nf(8192 * 4), ef(262144 * 2);
        for (unsigned i = 0; i < nodes; ++i) {
            ni[4 * i] = 1;
            ni[4 * i + 1] = i % 24;
            ni[4 * i + 2] = i % 32;
            ni[4 * i + 3] = i;
            nf[4 * i + 1] = 1.0f / (i + 1);
        }
        for (unsigned i = 0; i < edges; ++i) {
            ei[2 * i] = i % nodes;
            ei[2 * i + 1] = (i + 1) % nodes;
            ef[2 * i] = -0.25f;
            ef[2 * i + 1] = -1.0f;
        }
        std::vector<HostTensor> tensors{
            HostTensor::of("graph_counts", "i32", {4}, counts),
            HostTensor::of("node_i", "i32", {8192, 4}, ni),
            HostTensor::of("node_f", "f32", {8192, 4}, nf),
            HostTensor::of("edge_i", "i32", {262144, 2}, ei),
            HostTensor::of("edge_f", "f32", {262144, 2}, ef), text_tensor(8224, "reference ")};
        const auto data = mr::run_derive(metal.device, piece, tensors, {});
        REQUIRE(resource(data, "nodes").count == nodes);
        REQUIRE(resource(data, "curves").count == edges);
        CHECK(scalar<std::uint32_t>(resource(data, "nodes"), nodes - 1, 16, 48) == nodes - 1);
        CHECK(scalar<float>(resource(data, "nodes"), nodes - 1, 12, 48) > 0.0f);
        CHECK(scalar<float>(resource(data, "curves"), edges - 1, 32, 48) == -0.25f);
        CHECK(scalar<std::uint32_t>(resource(data, "curves"), edges - 1, 40, 48) == (edges - 1) % nodes);
        CHECK(scalar<std::uint32_t>(resource(data, "strings"), 32 + nodes - 1, 0, 64) == 0x65666572u);
        counts[0] = 8193;
        tensors[0] = HostTensor::of("graph_counts", "i32", {4}, counts);
        const auto refused = mr::run_derive(metal.device, piece, tensors, {});
        CHECK(resource(refused, "nodes").count == 0);
        CHECK(resource(refused, "curves").count == 0);
    }
}

TEST_CASE("small graph density markers retain distinct center pixels for every kind") {
    @autoreleasepool {
        mr::Metal metal;
        if (!metal.backend) CALIPER_SKIP_NO_GPU(metal.why);
        const auto piece=ip::load("attribution_graph");
        auto data=mr::run_derive(metal.device,piece,graph_tensors(),{});
        for (auto& resource:data) if (resource.id=="nodes") {
            for (unsigned i=0;i<resource.count;++i) {
                const float radius=0.00001f;
                std::memcpy(resource.bytes.data()+i*48+12,&radius,4);
            }
        }
        mr::PieceDraw draw(*metal.backend,size);
        set_resources(draw,piece,data);
        REQUIRE(draw.draw(frame(piece)));
        const auto& nodes=resource(data,"nodes");
        std::set<unsigned> visible;
        for (unsigned y=0;y<size;++y) for (unsigned x=0;x<size;++x) {
            const unsigned item=mr::identity_of(draw.ids,size,x,y);
            if (item) visible.insert(item-1);
        }
        REQUIRE(visible.size()==nodes.count);
        for (unsigned i=0;i<nodes.count;++i) {
            CHECK(visible.count(i)==1);
            check_pick(draw,scalar<float>(nodes,i,0,48),scalar<float>(nodes,i,4,48),i);
        }
    }
}

TEST_CASE("dense graph raster retains every ID and center pick in square and narrow panels") {
    @autoreleasepool {
        mr::Metal metal;
        if (!metal.backend) CALIPER_SKIP_NO_GPU(metal.why);
        const auto piece=ip::load("attribution_graph");
        const auto data=mr::run_derive(metal.device,piece,graph_tensors(true),{});
        const auto narrow=mr::run_derive(metal.device,piece,narrow_graph_tensors(),{});
        no_overlap(resource(data,"nodes"));
        no_overlap(resource(narrow,"nodes"));
        struct RasterCase {
            const char* folder;
            unsigned side, count;
            const std::vector<Published>* data;
        };
        for (const auto& test : {RasterCase{"density",512,512,&data},
                                 RasterCase{"density_narrow",256,512,&data},
                                 RasterCase{"narrow_cells",256,16,&narrow}}) {
            @autoreleasepool {
                CAPTURE(test.folder);
                const auto side=test.side;
                const auto& nodes=resource(*test.data,"nodes");
                REQUIRE(nodes.count==test.count);
                mr::PieceDraw draw(*metal.backend,side);
                set_resources(draw,piece,*test.data);
                REQUIRE(draw.draw(frame(piece,Json::object(),side)));
                std::map<unsigned,unsigned> visible;
                for (unsigned y=0;y<side;++y) for (unsigned x=0;x<side;++x) {
                    const auto item=mr::identity_of(draw.ids,side,x,y);
                    if (item) ++visible[unsigned(item-1)];
                }
                REQUIRE(visible.size()==nodes.count);
                std::set<std::pair<int,int>> centers;
                Json picks=Json::array();
                for (unsigned i=0;i<nodes.count;++i) {
                    const float x=scalar<float>(nodes,i,0,48), y=scalar<float>(nodes,i,4,48);
                    const int px=int((x+1)*side*0.5f), py=int((1-y)*side*0.5f);
                    REQUIRE(px>=0); REQUIRE(py>=0); REQUIRE(px<int(side)); REQUIRE(py<int(side));
                    CHECK(centers.emplace(px,py).second);
                    CHECK(visible.count(i)==1);
                    CHECK(visible[i]==1); // No sample lies outside the node's center pixel.
                    CHECK(mr::identity_of(draw.ids,side,px,py)==i+1);
                    const auto picked=draw.rt.pick(i);
                    REQUIRE(picked);
                    CHECK(picked->item==i);
                    Json row=Json::array(); row.push(px); row.push(py); row.push(std::uint64_t(i));
                    picks.push(row);
                }
                Export output(side,test.folder);
                output.add("graph_dense",piece,*test.data,draw,Json::object(),picks);
                output.finish();
            }
        }
    }
}

TEST_CASE("graph layout has disjoint cells, stable bytes, labels, and sixteen center picks") {
    @autoreleasepool {
        mr::Metal metal;
        if (!metal.backend) CALIPER_SKIP_NO_GPU(metal.why);
        const auto piece=ip::load("attribution_graph");
        const auto tensors=graph_tensors();
        const auto data=mr::run_derive(metal.device,piece,tensors,{{"selected_node",4}});
        const auto again=mr::run_derive(metal.device,piece,tensors,{{"selected_node",4}});
        for (const auto& r:data) CHECK(r.bytes==resource(again,r.id).bytes);
        const auto& nodes=resource(data,"nodes");
        const auto& curves=resource(data,"curves");
        REQUIRE(nodes.count==16); REQUIRE(curves.count==18);
        no_overlap(nodes);
        for (unsigned i=0;i<curves.count;++i) {
            const auto s=scalar<unsigned>(curves,i,40,48), t=scalar<unsigned>(curves,i,44,48);
            CHECK(scalar<float>(curves,i,0,48)==scalar<float>(nodes,s,0,48));
            CHECK(scalar<float>(curves,i,4,48)==scalar<float>(nodes,s,4,48));
            CHECK(scalar<float>(curves,i,24,48)==scalar<float>(nodes,t,0,48));
            CHECK(scalar<float>(curves,i,28,48)==scalar<float>(nodes,t,4,48));
            CHECK(scalar<float>(curves,i,32,48)==(i%2 ? -0.5f : 1.0f));
        }
        mr::PieceDraw draw(*metal.backend,size);
        set_resources(draw,piece,data);
        Json params=Json::object(); params.set("selected_node",4);
        REQUIRE(draw.draw(frame(piece,params)));
        Json picks=Json::array();
        for (unsigned i=0;i<16;++i) {
            const auto x=scalar<float>(nodes,i,0,48), y=scalar<float>(nodes,i,4,48);
            check_pick(draw,x,y,i); picks.push(pick_at(x,y,i));
        }
        const auto labels=draw.rt.labels();
        REQUIRE(labels.size()==5); // Four token labels and one selected label, without a pointer hit.
        CHECK(labels[0].text=="label 0"); CHECK(labels[4].text=="label 36");
        CHECK(scalar<unsigned>(resource(data,"labels"),32,16,32)==0xffffffffu);
        auto& output=exporter();
        output.add("graph_selected",piece,data,draw,params,picks);
        const auto selected=draw.color;
        const auto shown=draw.rt.shown_items();
        Json changed=Json::object(); changed.set("selected_node",5);
        REQUIRE(draw.draw(frame(piece,changed)));
        REQUIRE(draw.rt.labels().size()==5);
        CHECK(draw.rt.labels()[4].text=="label 37");
        CHECK(draw.rt.shown_items().version==shown.version);
        REQUIRE(draw.draw(frame(piece)));
        REQUIRE(draw.rt.labels().size()==4);
        std::size_t dimmed=0;
        for (unsigned y=0;y<size;++y) for (unsigned x=0;x<size;++x) {
            if (mr::identity_of(draw.ids,size,x,y)) continue;
            const auto p=(y*size+x)*4;
            dimmed += draw.color[p]+draw.color[p+1]+draw.color[p+2] > selected[p]+selected[p+1]+selected[p+2]+5;
        }
        CHECK(dimmed>100);
        for (unsigned i=0;i<2;++i) {
            const float t=0.37f, u=1-t;
            const float x=u*u*u*scalar<float>(curves,i,0,48)+3*u*u*t*scalar<float>(curves,i,8,48)+
                3*u*t*t*scalar<float>(curves,i,16,48)+t*t*t*scalar<float>(curves,i,24,48);
            const float y=u*u*u*scalar<float>(curves,i,4,48)+3*u*u*t*scalar<float>(curves,i,12,48)+
                3*u*t*t*scalar<float>(curves,i,20,48)+t*t*t*scalar<float>(curves,i,28,48);
            const auto [px,py]=pixel(x,y);
            const auto at=(py*size+px)*4;
            CHECK(mr::identity_of(draw.ids,size,px,py)==0);
            if (i==0) {
                CHECK(draw.color[at+2]>draw.color[at]);
                CHECK(draw.color[at+2]==selected[at+2]);
            } else {
                CHECK(draw.color[at]>draw.color[at+2]);
                CHECK(draw.color[at]>selected[at]+20);
            }
        }
        output.add("graph",piece,data,draw,Json::object(),picks);
        output.finish();
    }
}

TEST_CASE("graph count and shape guards bound the layout and dense cells remain separate") {
    @autoreleasepool {
        mr::Metal metal;
        if (!metal.backend) CALIPER_SKIP_NO_GPU(metal.why);
        const auto piece=ip::load("attribution_graph");
        auto tensors=graph_tensors(true);
        const auto data=mr::run_derive(metal.device,piece,tensors,{});
        REQUIRE(resource(data,"nodes").count==512);
        no_overlap(resource(data,"nodes"));
        std::int32_t zero=0;
        std::memcpy(tensors[0].bytes.data(),&zero,4);
        const auto empty=mr::run_derive(metal.device,piece,tensors,{});
        CHECK(resource(empty,"nodes").count==0); CHECK(resource(empty,"curves").count==0);
        const std::int32_t nonempty=512;
        std::memcpy(tensors[0].bytes.data(),&nonempty,4);
        tensors[1].shape={256,8};
        const auto wrong=mr::run_derive(metal.device,piece,tensors,{});
        CHECK(resource(wrong,"nodes").count==0);
    }
}

TEST_CASE("example pages retain eight row picks and two hundred dynamic token labels") {
    @autoreleasepool {
        mr::Metal metal;
        if (!metal.backend) CALIPER_SKIP_NO_GPU(metal.why);
        const auto piece=ip::load("token_strip");
        std::vector<std::int32_t> ids(16), counts(16,25), tokens(400);
        std::vector<float> activations(400);
        for (unsigned i=0;i<400;++i) { tokens[i]=int(i+100); activations[i]=float(i%25)/24; }
        for (unsigned i=0;i<16;++i) ids[i]=int(700+i);
        auto text=text_tensor(400,"word ");
        std::vector<HostTensor> inputs={HostTensor::of("tokens","i32",{16,25},tokens),
            HostTensor::of("activations","f32",{16,25},activations),HostTensor::of("example_ids","i32",{16},ids),
            HostTensor::of("valid_tokens","i32",{16},counts),text};
        std::set<int> seen;
        mr::PieceDraw draw(*metal.backend,size);
        draw.install(piece,piece.name);
        auto& output=exporter();
        for (unsigned page=0;page<2;++page) {
            @autoreleasepool {
                const auto data=mr::run_derive(metal.device,piece,inputs,{{"page",page}});
                for (const auto& r:data) draw.set(r.id,r.bytes,r.count,&r);
                Json params=Json::object(); params.set("page",std::uint64_t(page));
                REQUIRE(draw.draw(frame(piece,params)));
                const auto labels=draw.rt.labels(); REQUIRE(labels.size()==200);
                CHECK(labels.front().text=="word "+std::to_string(page*200));
                CHECK(labels.back().text=="word "+std::to_string(page*200+199));
                const auto& rows=resource(data,"rows"); const auto& cells=resource(data,"cells");
                REQUIRE(rows.count==8); REQUIRE(cells.count==200);
                for (unsigned i=0;i<200;++i) {
                    const unsigned source=page*200+i;
                    CHECK(scalar<float>(cells,i,16,32)==activations[source]);
                    CHECK(scalar<int>(cells,i,20,32)==tokens[source]);
                    CHECK(scalar<unsigned>(cells,i,24,32)==i/25);
                    CHECK(scalar<unsigned>(cells,i,28,32)==1);
                    CHECK(labels[i].text=="word "+std::to_string(source));
                }
                Json picks=Json::array();
                for (unsigned row=0;row<8;++row) {
                    CHECK(scalar<int>(rows,row,0,16)==int(700+page*8+row));
                    CHECK(scalar<unsigned>(rows,row,4,16)==page*8+row);
                    CHECK(scalar<unsigned>(rows,row,8,16)==row);
                    CHECK(scalar<unsigned>(rows,row,12,16)==25);
                    seen.insert(scalar<int>(rows,row,0,16));
                    const float y=0.81f-float(row)*1.8f/8;
                    check_pick(draw,0,y,row); picks.push(pick_at(0,y,row));
                    const auto picked=draw.rt.pick(row);
                    REQUIRE(picked);
                    const std::map<std::string,double> fields(picked->fields.begin(),picked->fields.end());
                    CHECK(fields.at("example_id")==700+page*8+row);
                    CHECK(fields.at("example")==page*8+row);
                    CHECK(fields.at("visible_row")==row);
                    CHECK(fields.at("token_count")==25);
                }
                output.add("examples_"+std::to_string(page),piece,data,draw,params,picks);
            }
        }
        CHECK(seen.size()==16);
        output.finish();
    }
}

TEST_CASE("example pages clear invalid cells before any input read") {
    @autoreleasepool {
        mr::Metal metal;
        if (!metal.backend) CALIPER_SKIP_NO_GPU(metal.why);
        const auto piece=ip::load("token_strip");
        std::vector<HostTensor> inputs={
            HostTensor::of("tokens","i32",{16,25},std::vector<std::int32_t>(400,123)),
            HostTensor::of("activations","f32",{16,25},std::vector<float>(400,0.75f)),
            HostTensor::of("example_ids","i32",{16},std::vector<std::int32_t>(16,701)),
            HostTensor::of("valid_tokens","i32",{16},std::vector<std::int32_t>(16,25)),text_tensor(400,"word ")};
        unsigned input=0;
        bool short_buffer=false;
        SUBCASE("short token shape") { input=0; }
        SUBCASE("short activation shape") { input=1; }
        SUBCASE("short example ID shape") { input=2; }
        SUBCASE("short token count shape") { input=3; }
        SUBCASE("full token shape with a short buffer") { input=0; short_buffer=true; }
        SUBCASE("full activation shape with a short buffer") { input=1; short_buffer=true; }
        SUBCASE("full example ID shape with a short buffer") { input=2; short_buffer=true; }
        SUBCASE("full token count shape with a short buffer") { input=3; short_buffer=true; }
        auto& malformed=inputs[input];
        if (short_buffer) malformed.bytes.resize(sizeof(std::uint32_t));
        else for (auto& axis:malformed.shape) axis=1;
        const auto data=mr::run_derive(metal.device,piece,inputs,{{"page",1}});
        const auto& rows=resource(data,"rows");
        const auto& cells=resource(data,"cells");
        const auto& labels=resource(data,"labels");
        const auto& strings=resource(data,"strings");
        REQUIRE(rows.count==8); REQUIRE(cells.count==200); REQUIRE(labels.count==200); REQUIRE(strings.count==200);
        for (unsigned row=0;row<8;++row) {
            CHECK(scalar<int>(rows,row,0,16)==(input<2 ? 701 : -1));
            CHECK(scalar<unsigned>(rows,row,4,16)==8+row);
            CHECK(scalar<unsigned>(rows,row,8,16)==row);
            CHECK(scalar<unsigned>(rows,row,12,16)==(input<2 ? 25u : 0u));
        }
        for (unsigned i=0;i<200;++i) {
            CHECK(scalar<float>(cells,i,16,32)==0.0f);
            CHECK(scalar<int>(cells,i,20,32)==-1);
            CHECK(scalar<unsigned>(cells,i,24,32)==i/25);
            CHECK(scalar<unsigned>(cells,i,28,32)==0);
            CHECK(scalar<unsigned>(labels,i,16,32)==0xffffffffu);
        }
        CHECK(std::all_of(strings.bytes.begin(),strings.bytes.end(),[](std::uint8_t b){ return b==0; }));
        mr::PieceDraw draw(*metal.backend,size);
        set_resources(draw,piece,data);
        REQUIRE(draw.draw(frame(piece)));
        CHECK(draw.rt.labels().empty());
        CHECK(std::all_of(draw.ids.begin(),draw.ids.end(),[](std::uint8_t id){ return id==0; }));
    }
}

TEST_CASE("example labels reject incomplete packed text without changing row picks") {
    @autoreleasepool {
        mr::Metal metal;
        if (!metal.backend) CALIPER_SKIP_NO_GPU(metal.why);
        const auto piece=ip::load("token_strip");
        auto text=text_tensor(400,"word ");
        SUBCASE("short text shape") { text.shape={1,16}; }
        SUBCASE("full text shape with a short buffer") { text.bytes.resize(64); }
        std::vector<HostTensor> inputs={
            HostTensor::of("tokens","i32",{16,25},std::vector<std::int32_t>(400,123)),
            HostTensor::of("activations","f32",{16,25},std::vector<float>(400,0.75f)),
            HostTensor::of("example_ids","i32",{16},std::vector<std::int32_t>(16,701)),
            HostTensor::of("valid_tokens","i32",{16},std::vector<std::int32_t>(16,25)),text};
        const auto data=mr::run_derive(metal.device,piece,inputs,{{"page",1}});
        const auto& strings=resource(data,"strings");
        CHECK(std::all_of(strings.bytes.begin(),strings.bytes.end(),[](std::uint8_t b){ return b==0; }));
        mr::PieceDraw draw(*metal.backend,size);
        set_resources(draw,piece,data);
        REQUIRE(draw.draw(frame(piece)));
        CHECK(draw.rt.labels().empty());
        for (unsigned row=0;row<8;++row) check_pick(draw,0,0.81f-float(row)*1.8f/8,row);
    }
}

TEST_CASE("example padding keeps valid token values and page-local row identities") {
    @autoreleasepool {
        mr::Metal metal;
        if (!metal.backend) CALIPER_SKIP_NO_GPU(metal.why);
        const auto piece=ip::load("token_strip");
        std::vector<std::int32_t> counts(16,25), tokens(400,123), ids(16);
        const std::array<int,8> page_counts={0,1,25,-3,26,2,25,1};
        for (unsigned row=0;row<8;++row) counts[8+row]=page_counts[row];
        for (unsigned row=0;row<16;++row) ids[row]=700+row;
        tokens[225]=0;
        const std::vector<HostTensor> inputs={HostTensor::of("tokens","i32",{16,25},tokens),
            HostTensor::of("activations","f32",{16,25},std::vector<float>(400,0.75f)),
            HostTensor::of("example_ids","i32",{16},ids),HostTensor::of("valid_tokens","i32",{16},counts),
            text_tensor(400,"word ")};
        const auto data=mr::run_derive(metal.device,piece,inputs,{{"page",1}});
        const auto& rows=resource(data,"rows"); const auto& cells=resource(data,"cells");
        const auto& labels=resource(data,"labels"); const auto& strings=resource(data,"strings");
        REQUIRE(rows.count==8); REQUIRE(cells.count==200); REQUIRE(labels.count==200); REQUIRE(strings.count==200);
        REQUIRE(strings.bytes.size()==200*64);
        unsigned visible=0;
        for (unsigned row=0;row<8;++row) {
            const unsigned count=unsigned(std::clamp(page_counts[row],0,25));
            visible+=count;
            CHECK(scalar<unsigned>(rows,row,12,16)==count);
            if (count) {
                CHECK(scalar<float>(cells,row*25,0,32)==doctest::Approx(-0.95f));
                CHECK(scalar<float>(cells,row*25+count-1,8,32)==doctest::Approx(0.946f));
            }
            for (unsigned col=0;col<25;++col) {
                const unsigned i=row*25+col;
                const bool valid=col<count;
                CHECK(scalar<float>(cells,i,16,32)==(valid ? 0.75f : 0.0f));
                CHECK(scalar<int>(cells,i,20,32)==(valid ? tokens[200+i] : -1));
                CHECK(scalar<unsigned>(cells,i,28,32)==unsigned(valid));
                CHECK(scalar<unsigned>(labels,i,16,32)==(valid ? i : 0xffffffffu));
                if (!valid) CHECK(std::all_of(strings.bytes.begin()+i*64,strings.bytes.begin()+(i+1)*64,
                    [](std::uint8_t b){ return b==0; }));
            }
        }
        CHECK(visible==79);
        mr::PieceDraw draw(*metal.backend,size);
        set_resources(draw,piece,data);
        REQUIRE(draw.draw(frame(piece)));
        CHECK(draw.rt.labels().size()==visible);
        for (unsigned row=0;row<8;++row) {
            const float y=0.81f-float(row)*1.8f/8;
            if (page_counts[row]>0) check_pick(draw,-0.912f,y,row);
            else {
                const auto [x,py]=pixel(-0.912f,y);
                CHECK(mr::identity_of(draw.ids,size,x,py)==0);
            }
        }
    }
}

TEST_CASE("output bars retain baseline probabilities and recovery cells retain their indices") {
    @autoreleasepool {
        mr::Metal metal;
        if (!metal.backend) CALIPER_SKIP_NO_GPU(metal.why);
        auto& output=exporter();
        {
            const auto piece=ip::load("logit_bars");
            std::vector<float> probs(20); std::vector<std::int32_t> tokens(10);
            for (unsigned i=0;i<10;++i) { probs[i]=float(10-i)*0.08f; probs[10+i]=float(i+1)*0.06f; tokens[i]=100+i; }
            const auto data=mr::run_derive(metal.device,piece,{HostTensor::of("probs","f32",{2,10},probs),
                HostTensor::of("tokens","i32",{10},tokens),text_tensor(10,"token ")},{});
            const auto& bars=resource(data,"cells"); REQUIRE(bars.count==10);
            mr::PieceDraw draw(*metal.backend,size); set_resources(draw,piece,data); REQUIRE(draw.draw(frame(piece)));
            REQUIRE(draw.rt.labels().size()==10);
            Json picks=Json::array();
            for (unsigned i=0;i<10;++i) {
                CHECK(scalar<float>(bars,i,16,32)==probs[i]); CHECK(scalar<float>(bars,i,20,32)==probs[10+i]);
                const float x=-0.81f+float(i)*0.18f;
                check_pick(draw,x,-0.7f,i); picks.push(pick_at(x,-0.7f,i));
                // The tallest filled pixel at the center tracks max(baseline, current) within two pixels.
                const auto [px,expected_y]=pixel(x,-0.72f+1.44f*std::max(probs[i],probs[10+i]));
                int top=int(size);
                for (unsigned y=0;y<size;++y) if (mr::identity_of(draw.ids,size,px,y)==i+1) { top=int(y); break; }
                CHECK(std::abs(top-expected_y)<=2);
                const auto [ghost_x,baseline_y]=pixel(x-0.065f,-0.72f+1.44f*probs[i]);
                int ghost_top=int(size), current_top=int(size);
                for (unsigned y=0;y<size;++y) {
                    if (ghost_top==int(size) && mr::identity_of(draw.ids,size,ghost_x,y)==i+1) ghost_top=int(y);
                    const auto at=(y*size+px)*4;
                    if (current_top==int(size) && draw.color[at+1]>150 && draw.color[at]<90) current_top=int(y);
                }
                CHECK(std::abs(ghost_top-baseline_y)<=2);
                const auto current_y=pixel(x,-0.72f+1.44f*probs[10+i]).second;
                CHECK(std::abs(current_top-current_y)<=2);
            }
            output.add("logits",piece,data,draw,Json::object(),picks);
        }
        {
            const auto piece=ip::load("heat_grid"); std::vector<float> values(24);
            for (unsigned i=0;i<24;++i) values[i]=(float(i)-12)/12;
            const auto data=mr::run_derive(metal.device,piece,{HostTensor::of("grid","f32",{4,6},values)},{});
            const auto& cells=resource(data,"cells"); REQUIRE(cells.count==24);
            mr::PieceDraw draw(*metal.backend,size); set_resources(draw,piece,data); REQUIRE(draw.draw(frame(piece)));
            Json picks=Json::array();
            for (unsigned i=0;i<24;++i) {
                CHECK(scalar<unsigned>(cells,i,20,32)==i/6); CHECK(scalar<unsigned>(cells,i,24,32)==i%6);
                CHECK(scalar<float>(cells,i,16,32)==values[i]);
                const float x=-0.9f+(float(i%6)+0.5f)*1.8f/6, y=-0.9f+(float(i/6)+0.5f)*1.8f/4;
                check_pick(draw,x,y,i); picks.push(pick_at(x,y,i));
            }
            output.add("recovery",piece,data,draw,Json::object(),picks);
        }
        output.finish();
    }
}

TEST_CASE("dynamic labels retain complete UTF-8 characters at the byte limit") {
    @autoreleasepool {
        mr::Metal metal;
        if (!metal.backend) CALIPER_SKIP_NO_GPU(metal.why);
        const std::vector<std::string> raw={std::string(63,'a')+"\xc3",std::string(62,'b')+"\xc3\xa9",
            std::string(61,'c')+"\xf0\x9f\x8c",std::string(60,'d')+"\xf0\x9f\x8c\x90"};
        const std::vector<std::string> expected={std::string(63,'a'),raw[1],std::string(61,'c'),raw[3]};
        for (const std::string name:{"attribution_graph","token_strip","logit_bars"}) {
            @autoreleasepool {
                CAPTURE(name);
                const auto piece=ip::load(name);
                std::vector<HostTensor> inputs;
                if (name=="attribution_graph") inputs=graph_tensors();
                else if (name=="token_strip") inputs={
                    HostTensor::of("tokens","i32",{16,25},std::vector<std::int32_t>(400,1)),
                    HostTensor::of("activations","f32",{16,25},std::vector<float>(400,0.5f)),
                    HostTensor::of("example_ids","i32",{16},std::vector<std::int32_t>(16,1)),
                    HostTensor::of("valid_tokens","i32",{16},std::vector<std::int32_t>(16,25)),text_tensor(400,"word ")};
                else inputs={HostTensor::of("probs","f32",{2,10},std::vector<float>(20,0.5f)),
                    HostTensor::of("tokens","i32",{10},std::vector<std::int32_t>(10,1)),text_tensor(10,"word ")};
                for (auto& input:inputs) if (input.var=="label_text")
                    for (unsigned i=0;i<raw.size();++i) std::memcpy(input.bytes.data()+64*i,raw[i].data(),64);
                const auto data=mr::run_derive(metal.device,piece,inputs,{});
                mr::PieceDraw draw(*metal.backend,size); set_resources(draw,piece,data); REQUIRE(draw.draw(frame(piece)));
                const auto labels=draw.rt.labels(); REQUIRE(labels.size()>=4);
                for (unsigned i=0;i<4;++i) CHECK(labels[i].text==expected[i]);
            }
        }
    }
}
TEST_CASE("native scene pointer hover shows labels without a click or derive update") {
    @autoreleasepool {
        std::string why;
        auto canvas = caliper_host::SpecOffscreen::create(size, size, 1, &why);
        if (!canvas) CALIPER_SKIP_NO_GPU(why);
        auto backend = caliper_gfx::make_metal_backend(canvas->native_device(), canvas->native_queue());
        REQUIRE(backend);
        struct Feed : cd::DrawFeed {
            cs::Piece piece = ip::load("attribution_graph");
            std::vector<Published> data;
            std::uint64_t resource_revision = 1, sent = 0;
            bool unmeasured = false;
            double reported_ms = -1;
            std::uint64_t revision() const override { return 1; }
            std::vector<cd::PanelProgram> programs() const override {
                return {{piece, {"graph", {}, "graph"}, mr::kPiecePick, 1, {}}};
            }
            void frame(cd::DrawRuntime& rt, cd::FrameInputs* in) override {
                if (sent != resource_revision) {
                    for (const auto& r : data) {
                        cd::ResourceVersion v;
                        v.resource = r.id; v.step = 1; v.count = r.count;
                        v.bytes = r.bytes.data(); v.size = r.bytes.size();
                        REQUIRE(rt.set_resource(v).ok());
                    }
                    sent = resource_revision;
                }
                Json params = Json::object(); params.set("selected_node", 4);
                in->params = caliper_web::pack_params(piece, params);
            }
            void status(const cd::DrawRuntime& rt) override {
                const auto status = rt.status();
                CHECK_MESSAGE(status.error.ok(), status.error.message);
                if (status.resource_step == 1) {
                    unmeasured = unmeasured || status.frame_ms < 0;
                    reported_ms = status.frame_ms;
                }
            }
        } feed;
        feed.data = mr::run_derive((__bridge id<MTLDevice>)canvas->native_device(), feed.piece, graph_tensors(), {{"selected_node",4}});
        caliper_live_views::SceneView view(*backend, caliper_gfx::shaders::kMetalPrograms);
        REQUIRE_MESSAGE(view.init(&why), why);
        wire::SceneView description; description.space = CALIPER_NET_SPACE_3D; description.up = "+z";
        view.describe(description);
        caliper_live_views::SceneRectOptions options;
        options.draw_feed = &feed;
        caliper_live_views::SceneRectState state;
        std::vector<std::pair<std::string, std::string>> events;
        const auto sink = [&](const char* type, const std::string& detail) { events.emplace_back(type, detail); };
        const auto scene = [&](const ImRect& area) {
            caliper_live_views::draw_scene_view_in_rect(view, 1, "##hover", area.Min, area.GetSize(), options, state, sink);
        };
        const auto draw = [&] {
            @autoreleasepool {
                REQUIRE_MESSAGE(canvas->render(1, scene, nullptr, &why), why);
            }
        };
        const auto capture = [&](const char* name) {
            std::vector<std::uint8_t> rgba;
            REQUIRE_MESSAGE(canvas->render(1, scene, &rgba, &why), why);
            REQUIRE(stbi_write_png((exporter().out / "app" / name).string().c_str(), size, size, 4, rgba.data(), size * 4));
        };
        const auto has = [&](const std::string& text) {
            return std::any_of(state.labels.begin(), state.labels.end(), [&](const auto& l) { return l.text == text; });
        };
        const auto until = [&](const std::function<bool()>& ready) {
            for (unsigned frame = 0; frame < 120 && !ready(); ++frame) draw();
            REQUIRE(ready());
        };
        const auto move = [&](float x, float y) { canvas->with_context([&] { ImGui::GetIO().AddMousePosEvent(x, y); }); };
        const auto at_node = [&](unsigned item) {
            const auto& nodes = resource(feed.data, "nodes");
            double p[4];
            caliper_scene::transform_h(view.program_clip(),
                {scalar<float>(nodes,item,0,48), scalar<float>(nodes,item,4,48), 0}, p);
            return ImVec2(float((1 + p[0]/p[3]) * size/2), float((1 - p[1]/p[3]) * size/2));
        };
        move(-100, -100);
        until([&] { return has("label 36"); });
        until([&] { return feed.reported_ms >= 0; });
        CHECK(feed.unmeasured);
        CHECK(feed.sent == 1);
        REQUIRE(std::holds_alternative<caliper_scene::PanZoom>(view.camera()));
        CHECK_FALSE(std::get<caliper_scene::PanZoom>(view.camera()).y_down);
        const auto clip = view.program_clip();
        CHECK(clip.m[0] == doctest::Approx(1));
        CHECK(clip.m[5] == doctest::Approx(1));
        CHECK(clip.m[3] == 0);
        CHECK(clip.m[7] == 0);
        CHECK(clip.m[15] == 1);
        REQUIRE(view.render(1024, 256, 1, 0xff101010, &why));
        CHECK(view.program_clip().m[0] == doctest::Approx(1));
        CHECK(view.program_clip().m[5] == doctest::Approx(1));
        view.pan(64, 16, 256);
        CHECK(std::get<caliper_scene::PanZoom>(view.camera()).center[0] == doctest::Approx(-0.125));
        CHECK(std::get<caliper_scene::PanZoom>(view.camera()).center[1] == doctest::Approx(0.125));
        view.fit();
        view.zoom(1, 0.4f, -0.2f, 1024, 256);
        const auto zoomed = std::get<caliper_scene::PanZoom>(view.camera());
        CHECK(zoomed.center[0] + 0.4 * zoomed.half_height == doctest::Approx(0.4));
        CHECK(zoomed.center[1] - 0.2 * zoomed.half_height == doctest::Approx(-0.2));
        view.reset();
        draw();
        view.pan(32, 16, size);
        CHECK(view.moved());
        const auto moved = view.camera();
        draw();
        CHECK(std::get<caliper_scene::PanZoom>(view.camera()).center[0] ==
              std::get<caliper_scene::PanZoom>(moved).center[0]);
        view.fit();
        draw();
        CHECK(view.program_clip().m[0] == doctest::Approx(1));
        CHECK(view.program_clip().m[12] == doctest::Approx(0));
        view.reset();
        draw();
        CHECK(view.program_clip().m[5] == doctest::Approx(1));
        auto point = at_node(5); move(point.x, point.y);
        until([&] { return has("label 37"); });
        CHECK(has("label 36")); CHECK(events.empty()); CHECK(feed.sent == 1);
        capture("hover-node.png");
        point = at_node(6); move(point.x, point.y);
        until([&] { return has("label 38") && !has("label 37"); });
        CHECK(events.empty());
        move(-100, -100); draw(); CHECK_FALSE(has("label 38"));
        capture("hover-exit.png");
        move(point.x, point.y);
        until([&] { return has("label 38"); });
        // A same-step replacement moves the node away from a still pointer.
        for (auto& r : feed.data) if (r.id == "nodes") {
            const float outside = 20;
            std::memcpy(r.bytes.data() + 6 * 48, &outside, 4);
        }
        ++feed.resource_revision; draw(); CHECK_FALSE(has("label 38"));
        point = at_node(5); move(point.x, point.y);
        until([&] { return has("label 37"); });
        canvas->with_context([] { ImGui::GetIO().AddMouseButtonEvent(0, true); });
        draw(); CHECK_FALSE(has("label 37"));
        canvas->with_context([] { ImGui::GetIO().AddMouseButtonEvent(0, false); });
        until([&] { return !events.empty(); });
        REQUIRE(events.size() == 1); CHECK(events[0].first == "pick");
        const auto picked = nlohmann::json::parse(events[0].second);
        CHECK(picked.at("item") == 5); CHECK(picked.at("fields").at("index") == 5);
    }
}

TEST_CASE("panel layouts retain graph coverage and disjoint example and probability labels") {
    for (const std::string name : {"attribution_graph", "token_strip", "logit_bars"}) {
        @autoreleasepool {
            CAPTURE(name);
            const unsigned width = name == "attribution_graph" ? 1100 : name == "token_strip" ? 1440 : 760;
            const unsigned height = name == "attribution_graph" ? 760 : 360;
            const unsigned label_count = name == "attribution_graph" ? 4 : name == "token_strip" ? 200 : 10;
            std::string why;
            auto canvas = caliper_host::SpecOffscreen::create(width, height, 1, &why);
            if (!canvas) CALIPER_SKIP_NO_GPU(why);
            auto backend = caliper_gfx::make_metal_backend(canvas->native_device(), canvas->native_queue());
            REQUIRE(backend);
            struct Feed : cd::DrawFeed {
                cs::Piece piece;
                std::vector<Published> data;
                bool sent = false;
                std::uint64_t revision() const override { return 1; }
                std::vector<cd::PanelProgram> programs() const override {
                    return {{piece, {"panel", {}, "panel"}, mr::kPiecePick, 1, {}}};
                }
                void frame(cd::DrawRuntime& rt, cd::FrameInputs*) override {
                    if (sent) return;
                    for (const auto& r : data) {
                        cd::ResourceVersion v;
                        v.resource = r.id; v.step = 1; v.count = r.count;
                        v.bytes = r.bytes.data(); v.size = r.bytes.size();
                        REQUIRE(rt.set_resource(v).ok());
                    }
                    sent = true;
                }
                void status(const cd::DrawRuntime& rt) override {
                    CHECK_MESSAGE(rt.status().error.ok(), rt.status().error.message);
                }
            } feed;
            feed.piece = ip::load(name);
            std::vector<HostTensor> tensors;
            if (name == "attribution_graph") tensors = graph_tensors();
            else if (name == "token_strip") {
                std::vector<float> activations(400);
                for (unsigned i = 0; i < 400; ++i) activations[i] = float(i % 25) / 24;
                tensors = {HostTensor::of("tokens", "i32", {16,25}, std::vector<int>(400, 17)),
                    HostTensor::of("activations", "f32", {16,25}, activations),
                    HostTensor::of("example_ids", "i32", {16}, std::vector<int>(16, 27)),
                    HostTensor::of("valid_tokens", "i32", {16}, std::vector<int>(16, 25)), text_tensor(400, "")};
                const std::vector<std::string> words = {"The", "small", "model", "reads", "each", "token", "and", "shows",
                    "a", "clear", "pattern", "in", "this", "short", "example", "while", "the", "color", "marks", "its",
                    "feature", "value", "for", "every", "position"};
                for (unsigned i = 0; i < 400; ++i) {
                    auto* bytes = tensors.back().bytes.data() + 64*i;
                    std::memset(bytes, 0, 64);
                    std::memcpy(bytes, words[i%25].data(), words[i%25].size());
                }
            } else {
                std::vector<float> probs(20);
                for (unsigned i = 0; i < 10; ++i) { probs[i] = float(10-i)*0.08f; probs[10+i] = float(i+1)*0.06f; }
                tensors = {HostTensor::of("probs", "f32", {2,10}, probs),
                    HostTensor::of("tokens", "i32", {10}, std::vector<int>(10, 37)), text_tensor(10, "Output Token ")};
            }
            feed.data = mr::run_derive((__bridge id<MTLDevice>)canvas->native_device(), feed.piece, tensors, {});
            caliper_live_views::SceneView view(*backend, caliper_gfx::shaders::kMetalPrograms);
            REQUIRE_MESSAGE(view.init(&why), why);
            wire::SceneView description; description.space = CALIPER_NET_SPACE_3D; description.up = "+z";
            view.describe(description);
            caliper_live_views::SceneRectOptions options; options.draw_feed = &feed;
            caliper_live_views::SceneRectState state;
            const auto scene = [&](const ImRect& area) {
                caliper_live_views::draw_scene_view_in_rect(view, 1, "##layout", area.Min, area.GetSize(), options, state, {});
            };
            for (unsigned frame = 0; frame < 120 && state.labels.size() != label_count; ++frame) {
                @autoreleasepool { REQUIRE_MESSAGE(canvas->render(1, scene, nullptr, &why), why); }
            }
            REQUIRE(state.labels.size() == label_count);
            REQUIRE(std::holds_alternative<caliper_scene::PanZoom>(view.camera()));
            CHECK(view.program_clip().m[0] == doctest::Approx(1));
            CHECK(view.program_clip().m[5] == doctest::Approx(1));
            nlohmann::json report = {{"width", width}, {"height", height}, {"labels", nlohmann::json::array()}};
            for (std::size_t i = 0; i < state.labels.size(); ++i) {
                const auto& a = state.labels[i];
                CHECK(a.x0 >= 0); CHECK(a.y0 >= 0); CHECK(a.x1 <= width); CHECK(a.y1 <= height);
                CHECK(a.x1 > a.x0); CHECK(a.y1 > a.y0);
                report["labels"].push_back({a.text, a.x0, a.y0, a.x1, a.y1});
                for (std::size_t j = i+1; j < state.labels.size(); ++j) {
                    const auto& b = state.labels[j];
                    CHECK_FALSE((a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1));
                }
            }
            if (name == "attribution_graph") {
                const auto& nodes = resource(feed.data, "nodes");
                float x0 = 1, y0 = 1, x1 = -1, y1 = -1;
                for (unsigned i = 0; i < nodes.count; ++i) {
                    const float x = scalar<float>(nodes, i, 0, 48), y = scalar<float>(nodes, i, 4, 48);
                    x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y);
                    CHECK(x > -1); CHECK(x < 1); CHECK(y > -1); CHECK(y < 1);
                    if (i && i < 4) CHECK(x > scalar<float>(nodes, i-1, 0, 48));
                    if (i >= 14) CHECK(y > scalar<float>(nodes, 12, 4, 48));
                    if (i >= 4 && i < 12) CHECK(y > scalar<float>(nodes, i-4, 4, 48));
                }
                CHECK((x1-x0)/2 >= 0.7f); CHECK((y1-y0)/2 >= 0.7f);
                CHECK((x1-x0)*(y1-y0)/4 >= 0.7f);
                report["coverage"] = {(x1-x0)/2, (y1-y0)/2};
                report["coverage_area"] = (x1-x0)*(y1-y0)/4;
            }
            std::vector<std::uint8_t> rgba;
            REQUIRE_MESSAGE(canvas->render(1, scene, &rgba, &why), why);
            if (name == "attribution_graph") {
                const auto& nodes = resource(feed.data, "nodes");
                const int cx = int((scalar<float>(nodes, 5, 0, 48)+1)*width/2);
                const int cy = int((1-scalar<float>(nodes, 5, 4, 48))*height/2);
                const auto green = [&](int x, int y) {
                    const auto at = 4*(y*width+x);
                    return rgba[at+1] > rgba[at]+45 && rgba[at+1] > rgba[at+2]+20;
                };
                REQUIRE(green(cx, cy));
                int left = cx, right = cx, top = cy, bottom = cy;
                while (left > 0 && green(left-1, cy)) --left;
                while (right+1 < int(width) && green(right+1, cy)) ++right;
                while (top > 0 && green(cx, top-1)) --top;
                while (bottom+1 < int(height) && green(cx, bottom+1)) ++bottom;
                CHECK(std::abs((right-left)-(bottom-top)) <= 2);
                report["feature_pixels"] = {right-left+1, bottom-top+1};
            }
            const auto out = exporter().out / "app";
            REQUIRE(stbi_write_png((out / (name + "-panel.png")).string().c_str(), width, height, 4, rgba.data(), width*4));
            const auto json = report.dump(2);
            write(out / (name + "-layout.json"), json.data(), json.size());
        }
    }
}
CALIPER_GPU_TEST_MAIN()
