#pragma once

TEST_CASE("Incident Exposure Budget") {
    @autoreleasepool {
        mr::Metal metal;
        if (!metal.backend) CALIPER_SKIP_NO_GPU(metal.why);
        const auto piece=ip::load("attribution_graph");
        // Seven separate tracks: signed magnitudes, two coincident fans, and zero.
        // The fans have equal total display exposure but different absolute mass.
        std::vector<int> ni(512*4),ei(8192*2);
        std::vector<float> nf(512*4),ef(8192*2);
        unsigned edges=0;
        for (unsigned track=0; track<7; ++track) {
            for (unsigned end=0; end<2; ++end) {
                const auto i=2*track+end;
                ni[4*i]=end ? 3 : 0; ni[4*i+1]=end ? 1 : -1;
                ni[4*i+2]=track; ni[4*i+3]=100+i; nf[4*i+1]=1;
            }
            const unsigned count=track==4 || track==5 ? 32 : 1;
            for (unsigned j=0; j<count; ++j,++edges) {
                const float magnitude=track==1 || track==3 ? 0.5f : track==4 ? 1.0f/32 : track==6 ? 0 : 1;
                const float sign=track==2 || track==3 || (track>=4 && j%2) ? -1 : 1;
                ei[2*edges]=2*track; ei[2*edges+1]=2*track+1;
                ef[2*edges]=ef[2*edges+1]=sign*magnitude;
            }
        }
        const std::vector<HostTensor> tensors{
            HostTensor::of("graph_counts","i32",{4},std::vector<int>{14,int(edges),7,1}),
            HostTensor::of("node_i","i32",{512,4},ni), HostTensor::of("node_f","f32",{512,4},nf),
            HostTensor::of("edge_i","i32",{8192,2},ei), HostTensor::of("edge_f","f32",{8192,2},ef),
            text_tensor(544,"exposure ")};
        const auto data=mr::run_derive(metal.device,piece,tensors,{});
        const auto& nodes=resource(data,"nodes");
        const auto& curves=resource(data,"curves");
        REQUIRE(nodes.count==14); REQUIRE(curves.count==edges);
        for (unsigned i=0; i<14; ++i) {
            const unsigned track=i/2;
            const float expected=track==1 || track==3 ? 0.5f : track==5 ? 32 : track==6 ? 0 : 1;
            CHECK(scalar<float>(resource(data,"cells"),i,36,40)==expected);
        }
        for (unsigned i=0; i<edges; ++i) {
            CHECK(scalar<unsigned>(curves,i,40,48)==unsigned(ei[2*i]));
            CHECK(scalar<unsigned>(curves,i,44,48)==unsigned(ei[2*i+1]));
            CHECK(scalar<float>(curves,i,32,48)==ef[2*i]);
            CHECK(scalar<float>(curves,i,36,48)==ef[2*i+1]);
        }
        mr::PieceDraw draw(*metal.backend,size);
        set_resources(draw,piece,data);
        Export output(size,"exposure");
        Json picks=Json::array();
        for (int selected : {-1,0,2}) {
            Json params=Json::object(); params.set("selected_node",selected);
            REQUIRE(draw.draw(frame(piece,params)));
            for (unsigned i=0; i<14; ++i) {
                const float x=scalar<float>(nodes,i,0,48),y=scalar<float>(nodes,i,4,48);
                check_pick(draw,x,y,i);
                if (selected<0) picks.push(pick_at(x,y,i));
            }
            const auto name=selected<0 ? "graph" : selected==0 ? "graph_selected" : "graph_selected_weak";
            output.add(name,piece,data,draw,params,picks);
            for (unsigned track=0; track<7; ++track) {
                const auto [x,y]=pixel(scalar<float>(nodes,2*track,0,48),0.1f);
                const auto p=4*(y*size+x);
                const std::array<float,3> positive{0.15f,0.55f,0.95f},negative{0.95f,0.25f,0.12f};
                const bool incident=selected==int(2*track);
                const float alpha=incident ? 0.7f : track==1 || track==3 ? 0.35f : track==6 ? 0 : 0.7f;
                const float dim=selected>=0 && !incident ? 0.12f : 1;
                for (unsigned c=0; c<3; ++c) {
                    const float color=track<2 ? positive[c] : track<4 ? negative[c] : (positive[c]+negative[c])*0.5f;
                    const auto background=(dp::kBackground>>(8*c))&255;
                    const float expected=background+255*alpha*dim*color;
                    // Each RGBA8 addition has at most half a code of conversion error.
                    const unsigned additions=track==4 || track==5 ? 32 : 1;
                    CAPTURE(track); CAPTURE(selected); CAPTURE(c);
                    CHECK(std::abs(float(draw.color[p+c])-expected)<=0.5f*additions+1);
                }
                CHECK(mr::identity_of(draw.ids,size,x,y)==0);
            }
        }
        output.finish();
    }
}
