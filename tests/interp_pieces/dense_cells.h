#pragma once

namespace dense_cells {
struct RectTarget {
    caliper_gfx::Backend& g;
    caliper_draw::HostTarget host;
    bool ok = false;

    RectTarget(caliper_gfx::Backend& backend, std::uint32_t width, std::uint32_t height) : g(backend) {
        caliper_gfx::TextureDesc d;
        d.width = width;
        d.height = height;
        d.format = caliper_gfx::Format::RGBA8Unorm;
        d.usage = caliper_gfx::kRenderTarget | caliper_gfx::kSampled | caliper_gfx::kCopySrcTex;
        d.label = "parity color";
        host.color = g.create_texture(d);
        d.format = caliper_gfx::Format::RG32Uint;
        d.usage = caliper_gfx::kRenderTarget | caliper_gfx::kCopySrcTex;
        d.label = "parity id";
        host.id = g.create_texture(d);
        d.format = caliper_gfx::Format::Depth32Float;
        d.usage = caliper_gfx::kRenderTarget;
        d.label = "parity depth";
        host.depth = g.create_texture(d);
        host.width = width;
        host.height = height;
        caliper_gfx::TextureDesc c;
        c.width = 256;
        c.height = caliper_gfx::cmap::kBuiltinColormaps;
        c.format = caliper_gfx::Format::RGBA8Unorm;
        c.usage = caliper_gfx::kSampled | caliper_gfx::kCopyDstTex;
        c.label = "parity colormaps";
        host.colormaps = g.create_texture(c);
        std::vector<std::uint32_t> rows(256u * caliper_gfx::cmap::kBuiltinColormaps);
        for (int id = 0; id < caliper_gfx::cmap::kBuiltinColormaps; ++id)
            std::memcpy(rows.data() + 256u * std::uint32_t(id), caliper_gfx::cmap::builtin_lut(id), 256 * 4);
        ok = host.color && host.id && host.depth && host.colormaps && g.write_texture(host.colormaps, rows.data(), 256 * 4);
    }
    ~RectTarget() {
        for (caliper_gfx::Texture t : {host.color, host.id, host.depth, host.colormaps})
            if (t) g.release(t);
    }
    RectTarget(const RectTarget&) = delete;
    RectTarget& operator=(const RectTarget&) = delete;

    // One frame: the scene's clear of the host target in a submit, then the program's frames, each
    // in a submit of its own, as the scene view submits them. The serials go to *serials.
    bool frame(caliper_draw::DrawRuntime& rt, const caliper_draw::FrameInputs& in,
               std::vector<caliper_gfx::Serial>* serials, std::string* why) {
        caliper_gfx::RenderTarget clear;
        clear.color = host.color;
        clear.id = host.id;
        clear.depth = host.depth;
        clear.clear_rgba = draw_parity::kBackground;
        caliper_gfx::Frame f;
        f.render(caliper_gfx::DrawList(clear));
        const caliper_gfx::Serial s = g.submit(std::move(f));
        if (s == 0) return (*why = "the clear was refused: " + g.last_error(), false);
        serials->push_back(s);
        std::vector<caliper_gfx::Frame> frames;
        if (!rt.encode(&frames, in, host)) {
            const caliper_draw::DrawStatus st = rt.status();
            *why = "the program drew nothing: " + st.error.code + " " + st.error.message;
            return false;
        }
        for (caliper_gfx::Frame& pf : frames) {
            const caliper_gfx::Serial ps = g.submit(std::move(pf));
            if (ps == 0) return (*why = "a frame of the program was refused: " + g.last_error(), false);
            serials->push_back(ps);
        }
        return true;
    }
};

}

namespace dense_cells {
using Document = nlohmann::json;

std::vector<HostTensor> connected_cell() {
    std::vector<int> ni(512*4), ei(8192*2);
    std::vector<float> nf(512*4), ef(8192*2);
    for (unsigned i=0; i<93; ++i) {
        ni[4*i]=i<91 ? 1 : i==91 ? 0 : 3;
        ni[4*i+1]=i<91 ? 3 : i==91 ? -1 : 26;
        ni[4*i+2]=16; ni[4*i+3]=100+i;
        nf[4*i]=1; nf[4*i+1]=93-i;
    }
    for (unsigned i=0; i<91; ++i) {
        ei[4*i]=91; ei[4*i+1]=i;
        ei[4*i+2]=i; ei[4*i+3]=92;
        ef[4*i]=ef[4*i+1]=1;
        ef[4*i+2]=ef[4*i+3]=-0.5f;
    }
    return {HostTensor::of("graph_counts","i32",{4},std::vector<int>{93,182,32,26}),
        HostTensor::of("node_i","i32",{512,4},ni), HostTensor::of("node_f","f32",{512,4},nf),
        HostTensor::of("edge_i","i32",{8192,2},ei), HostTensor::of("edge_f","f32",{8192,2},ef),
        text_tensor(544,"connected ")};
}

struct Image {
    std::vector<std::uint8_t> color, ids;
    std::map<unsigned,std::vector<std::pair<int,int>>> samples;
};

Image render(mr::Metal& metal, const cs::Piece& piece, const std::vector<Published>& data,
             unsigned width, unsigned height, const std::string& name, bool nodes_visible=true, bool selected=true) {
    @autoreleasepool {
        RectTarget target(*metal.backend,width,height);
        REQUIRE(target.ok);
        cd::DrawRuntime runtime(*metal.backend,false);
        cd::Instance instance; instance.program="graph"; instance.panel="graph";
        REQUIRE(runtime.install(piece,instance,mr::kPiecePick).ok());
        for (const auto& r:data) {
            cd::ResourceVersion v;
            v.resource=r.id; v.step=1; v.count=r.count; v.bytes=r.bytes.data(); v.size=r.bytes.size();
            REQUIRE(runtime.set_resource(v).ok());
        }
        Json params=Json::object(); params.set("selected_node",selected ? 10 : -1);
        auto in=frame(piece,params,width); in.viewport[3]=height;
        std::vector<caliper_gfx::Serial> serials;
        std::string why;
        REQUIRE_MESSAGE(target.frame(runtime,in,&serials,&why),why);
        for (const auto serial:serials) REQUIRE(metal.backend->wait(serial));
        Image image;
        image.color=mr::read_back(*metal.backend,target.host.color,width,height);
        image.ids=mr::read_back(*metal.backend,target.host.id,width,height);
        for (unsigned y=0; y<height; ++y) for (unsigned x=0; x<width; ++x) {
            const auto item=mr::identity_of(image.ids,width,x,y);
            if (item) image.samples[unsigned(item-1)].push_back({int(x),int(y)});
        }
        const auto& nodes=resource(data,"nodes");
        const auto out=exporter().out/"dense_cells";
        std::filesystem::create_directories(out);
        REQUIRE(stbi_write_png((out/(name+".png")).string().c_str(),width,height,4,image.color.data(),width*4));
        write(out/(name+".rgba"),image.color.data(),image.color.size());
        write(out/(name+".ids"),image.ids.data(),image.ids.size());
        for (const auto& r:data) if (r.id=="nodes" || r.id=="curves" || r.id=="cells")
            write(out/(name+"-"+r.id+".bin"),r.bytes.data(),r.bytes.size());
        Document report={{"width",width},{"height",height},{"node_count",nodes.count},
            {"visible_ids",image.samples.size()},{"piece_hash",piece.hash}};
        for (const auto& [item,samples]:image.samples) report["samples"][std::to_string(item)]=samples;
        std::ofstream(out/(name+".json"))<<report.dump(2)<<'\n';
        if (nodes_visible) {
            REQUIRE(image.samples.size()==nodes.count);
            for (unsigned i=0; i<nodes.count; ++i) {
                REQUIRE(image.samples.count(i)==1);
                const auto picked=runtime.pick(i);
                REQUIRE(picked);
                CHECK(picked->item==i);
                REQUIRE(picked->fields.size()==6);
                CHECK(picked->fields[0].second==i);
                CHECK(picked->fields[1].second==scalar<unsigned>(nodes,i,20,48));
                CHECK(picked->fields[2].second==scalar<int>(nodes,i,24,48));
                CHECK(picked->fields[3].second==scalar<int>(nodes,i,28,48));
                CHECK(picked->fields[4].second==scalar<int>(nodes,i,32,48));
            }
            const auto hovered=runtime.pick(10);
            REQUIRE(hovered);
            const auto labels=runtime.labels(&*hovered);
            REQUIRE(labels.size()>=2);
            const auto& strings=resource(data,"strings");
            const std::string label(reinterpret_cast<const char*>(strings.bytes.data()+42*64));
            CHECK(labels[labels.size()-1].text==label);
            if (selected) CHECK(labels[labels.size()-2].text==label);
        }
        return image;
    }
}
}

TEST_CASE("Connected Cell Raster Coverage") {
    @autoreleasepool {
        mr::Metal metal;
        if (!metal.backend) CALIPER_SKIP_NO_GPU(metal.why);
        const auto piece=ip::load("attribution_graph");
        const auto data=mr::run_derive(metal.device,piece,dense_cells::connected_cell(),{});
        const auto& nodes=resource(data,"nodes");
        REQUIRE(nodes.count==93);
        REQUIRE(resource(data,"curves").count==182);
        const auto small=dense_cells::render(metal,piece,data,1355,400,"connected-400");
        // Every dense feature occupies one pixel, inside its original layer and token cell.
        std::vector<std::pair<int,int>> centers;
        for (unsigned i=0; i<91; ++i) {
            REQUIRE(small.samples.at(i).size()==1);
            const auto [x,y]=small.samples.at(i).front();
            CHECK(x>=681); CHECK(x<=711); CHECK(y>=311); CHECK(y<=319);
            centers.push_back({y,x});
        }
        CHECK(std::is_sorted(centers.begin(),centers.end()));
        // The same producer resources support a second view size.
        const auto taller=dense_cells::render(metal,piece,data,1355,520,"connected-520");
        REQUIRE(taller.samples.size()==93);
        // Export the connected fallback case for the existing browser parity checks.
        {
            mr::PieceDraw draw(*metal.backend,size);
            set_resources(draw,piece,data);
            Json params=Json::object(); params.set("selected_node",10);
            REQUIRE(draw.draw(frame(piece,params)));
            std::map<unsigned,std::pair<unsigned,unsigned>> samples;
            std::map<unsigned,unsigned> counts;
            for (unsigned y=0; y<size; ++y) for (unsigned x=0; x<size; ++x) {
                const auto item=mr::identity_of(draw.ids,size,x,y);
                if (item) { samples.emplace(unsigned(item-1),std::make_pair(x,y)); ++counts[unsigned(item-1)]; }
            }
            REQUIRE(samples.size()==93);
            Json picks=Json::array();
            bool relocated=false;
            for (unsigned i=0; i<93; ++i) {
                REQUIRE(samples.count(i)==1);
                const auto [x,y]=samples.at(i);
                if (i<91) {
                    CHECK(counts.at(i)==1);
                    const auto original=pixel(scalar<float>(nodes,i,0,48),scalar<float>(nodes,i,4,48));
                    relocated=relocated || int(x)!=original.first || int(y)!=original.second;
                }
                Json row=Json::array(); row.push(x); row.push(y); row.push(std::uint64_t(i));
                picks.push(row);
            }
            CHECK(relocated);
            Export output(size,"connected_cells");
            output.add("graph_dense",piece,data,draw,params,picks);
            output.finish();
        }
        // One isolated edge must reach the moved feature, not its old producer position.
        auto edge_piece=piece;
        edge_piece.draw->passes.pop_back();
        edge_piece.hash=cp::piece_hash(edge_piece);
        auto edge_data=data;
        for (auto& r:edge_data) if (r.id=="curves") {
            const auto start=r.bytes.begin()+20*48;
            r.bytes=std::vector<std::uint8_t>(start,start+48); r.count=1;
        }
        const auto edge=dense_cells::render(metal,edge_piece,edge_data,1355,400,"connected-edge",false);
        const auto [x,y]=small.samples.at(10).front();
        const auto p=4*((y+1)*1355+x);
        CHECK(edge.color[p+2]>edge.color[p]+60);
        CHECK(edge.samples.empty());
    }
}

TEST_CASE("Saved Graph Raster Coverage") {
    const char* input=std::getenv("CALIPER_DENSE_GRAPH_INPUT");
    if (!input) CALIPER_SKIP_NOT_APPLICABLE("Saved graph input is available only in the bounded local check");
    @autoreleasepool {
        dense_cells::Document graph;
        std::ifstream stream(input);
        stream>>graph;
        REQUIRE(graph.at("layers")==26);
        REQUIRE(graph.at("tokens")==11);
        REQUIRE(graph.at("nodes").size()==848);
        REQUIRE(graph.at("edges").size()==34098);
        std::vector<int> ni(8192*4),ei(262144*2);
        std::vector<float> nf(8192*4),ef(262144*2);
        auto text=text_tensor(8224,"Token ");
        for (unsigned i=0; i<848; ++i) {
            for (unsigned j=0; j<4; ++j) ni[4*i+j]=graph["nodes"][i][j];
            for (unsigned j=0; j<3; ++j) nf[4*i+j]=graph["nodes"][i][4+j];
            const std::string label=graph["nodes"][i][8];
            REQUIRE(label.size()<64);
            std::memset(text.bytes.data()+(32+i)*64,0,64);
            std::memcpy(text.bytes.data()+(32+i)*64,label.data(),label.size());
        }
        float largest=0;
        for (const auto& e:graph["edges"]) largest=std::max(largest,std::abs(e[2].get<float>()));
        for (unsigned i=0; i<34098; ++i) {
            ei[2*i]=graph["edges"][i][0]; ei[2*i+1]=graph["edges"][i][1];
            ef[2*i]=graph["edges"][i][2]; ef[2*i+1]=ef[2*i]/largest;
        }
        const std::vector<HostTensor> tensors{
            HostTensor::of("graph_counts","i32",{4},std::vector<int>{848,34098,11,26}),
            HostTensor::of("node_i","i32",{8192,4},ni), HostTensor::of("node_f","f32",{8192,4},nf),
            HostTensor::of("edge_i","i32",{262144,2},ei), HostTensor::of("edge_f","f32",{262144,2},ef),text};
        mr::Metal metal;
        if (!metal.backend) CALIPER_SKIP_NO_GPU(metal.why);
        const auto piece=ip::load("attribution_graph");
        const auto data=mr::run_derive(metal.device,piece,tensors,{});
        const auto& nodes=resource(data,"nodes");
        const auto& curves=resource(data,"curves");
        REQUIRE(curves.count==34098);
        for (unsigned i=0; i<curves.count; ++i) {
            const auto s=scalar<unsigned>(curves,i,40,48),t=scalar<unsigned>(curves,i,44,48);
            CHECK(s==unsigned(ei[2*i])); CHECK(t==unsigned(ei[2*i+1]));
            CHECK(scalar<float>(curves,i,0,48)==scalar<float>(nodes,s,0,48));
            CHECK(scalar<float>(curves,i,4,48)==scalar<float>(nodes,s,4,48));
            CHECK(scalar<float>(curves,i,24,48)==scalar<float>(nodes,t,0,48));
            CHECK(scalar<float>(curves,i,28,48)==scalar<float>(nodes,t,4,48));
            CHECK(scalar<float>(curves,i,32,48)==ef[2*i]);
            CHECK(scalar<float>(curves,i,36,48)==ef[2*i+1]);
        }
        const auto image=dense_cells::render(metal,piece,data,1355,520,"saved-520",true,false);
        for (unsigned i=0; i<nodes.count; ++i) {
            const int x=int((scalar<float>(nodes,i,0,48)+1)*1355*0.5f);
            const int y=int((1-scalar<float>(nodes,i,4,48))*520*0.5f);
            CHECK(mr::identity_of(image.ids,1355,x,y)==i+1);
        }
        dense_cells::render(metal,piece,data,1355,520,"saved-selected-520",true,true);
    }
}
