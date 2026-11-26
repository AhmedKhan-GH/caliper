#pragma once

TEST_CASE("Published Viewer Edge Weighting and Selection") {
    @autoreleasepool {
        mr::Metal metal;
        if (!metal.backend) CALIPER_SKIP_NO_GPU(metal.why);
        const auto piece=ip::load("attribution_graph");
        // Seven tracks exercise target-side pctInput normalization, PRGn sign colors,
        // dense incident fans, zero weights, and unselected overview hiding.
        std::vector<int> ni(512*4),ei(8192*2);
        std::vector<float> nf(512*4),ef(8192*2);
        unsigned edges=0;
        for (unsigned track=0; track<7; ++track) {
            for (unsigned end=0; end<2; ++end) {
                const auto i=2*track+end;
                ni[4*i]=end ? 3 : 0; ni[4*i+1]=end ? 1 : -1;
                ni[4*i+2]=track; ni[4*i+3]=100+i; nf[4*i+1]=1;
            }
            if (track==0) {
                ei[2*edges]=0; ei[2*edges+1]=1; ef[2*edges]=ef[2*edges+1]=1.0f; ++edges;
            } else if (track==1) {
                // Node 3 receives +0.2 from node 2 and -0.8 from node 4 => pctInput(2->3) = +0.2.
                ei[2*edges]=2; ei[2*edges+1]=3; ef[2*edges]=ef[2*edges+1]=0.2f; ++edges;
                ei[2*edges]=4; ei[2*edges+1]=3; ef[2*edges]=ef[2*edges+1]=-0.8f; ++edges;
            } else if (track==2) {
                ei[2*edges]=4; ei[2*edges+1]=5; ef[2*edges]=ef[2*edges+1]=-1.0f; ++edges;
            } else if (track==3) {
                // Node 7 receives -0.2 from node 6 and +0.8 from node 8 => pctInput(6->7) = -0.2.
                ei[2*edges]=6; ei[2*edges+1]=7; ef[2*edges]=ef[2*edges+1]=-0.2f; ++edges;
                ei[2*edges]=8; ei[2*edges+1]=7; ef[2*edges]=ef[2*edges+1]=0.8f; ++edges;
            } else if (track==4) {
                // Dense 32-edge fan: 31 weak edges (+0.01) and 1 dominant edge (+0.69), inputAbsSum = 1.0.
                for (unsigned j=0; j<32; ++j,++edges) {
                    const float w = j==31 ? 0.69f : 0.01f;
                    ei[2*edges]=8; ei[2*edges+1]=9; ef[2*edges]=ef[2*edges+1]=w;
                }
            } else if (track==5) {
                // Dense 32-edge uniform fan: inputAbsSum = 32, |pctInput| = 1/32, visible subpixel width.
                for (unsigned j=0; j<32; ++j,++edges) {
                    const float w = (j%2) ? -1.0f : 1.0f;
                    ei[2*edges]=10; ei[2*edges+1]=11; ef[2*edges]=ef[2*edges+1]=w;
                }
            } else {
                ei[2*edges]=12; ei[2*edges+1]=13; ef[2*edges]=ef[2*edges+1]=0.0f; ++edges;
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
        const auto& cells=resource(data,"cells");
        REQUIRE(nodes.count==14); REQUIRE(curves.count==edges);
        const std::array<float,14> expected_in{0,1.0f, 0,1.0f, 0,1.0f, 0,1.0f, 0,1.0f, 0,32.0f, 0,0.0f};
        for (unsigned i=0; i<14; ++i) {
            CHECK(scalar<float>(cells,i,36,40)==doctest::Approx(expected_in[i]));
        }
        // Verify Curve.normalized stores target-normalized pctInput (util-cg.js:295-296).
        CHECK(scalar<float>(curves,0,36,48)==doctest::Approx(1.0f));
        CHECK(scalar<float>(curves,1,36,48)==doctest::Approx(0.2f));
        CHECK(scalar<float>(curves,2,36,48)==doctest::Approx(-0.8f));
        CHECK(scalar<float>(curves,3,36,48)==doctest::Approx(-1.0f));
        CHECK(scalar<float>(curves,4,36,48)==doctest::Approx(-0.2f));
        CHECK(scalar<float>(curves,5,36,48)==doctest::Approx(0.8f));
        CHECK(scalar<float>(curves,edges-1,36,48)==0.0f);
        mr::PieceDraw draw(*metal.backend,size);
        set_resources(draw,piece,data);
        Export output(size,"exposure");
        Json picks=Json::array();
        const auto curve_mid = [&](unsigned idx) {
            const float x=0.5f*(scalar<float>(curves,idx,0,48)+scalar<float>(curves,idx,24,48));
            const float y=0.5f*(scalar<float>(curves,idx,4,48)+scalar<float>(curves,idx,28,48));
            return pixel(x,y);
        };
        const auto check_rgb = [&](int x, int y, int r, int g, int b, int tol = 2) {
            const auto p=4*(y*size+x);
            CHECK(std::abs(int(draw.color[p])-r)<=tol);
            CHECK(std::abs(int(draw.color[p+1])-g)<=tol);
            CHECK(std::abs(int(draw.color[p+2])-b)<=tol);
            CHECK(mr::identity_of(draw.ids,size,x,y)==0);
        };
        const int bg_r=int(dp::kBackground&255), bg_g=int((dp::kBackground>>8)&255), bg_b=int((dp::kBackground>>16)&255);
        for (int selected : {-1,0,2,4,6,8,12}) {
            Json params=Json::object(); params.set("selected_node",selected);
            REQUIRE(draw.draw(frame(piece,params)));
            for (unsigned i=0; i<14; ++i) {
                const float x=scalar<float>(nodes,i,0,48),y=scalar<float>(nodes,i,4,48);
                check_pick(draw,x,y,i);
                if (selected<0) picks.push(pick_at(x,y,i));
            }
            if (selected<0) output.add("graph",piece,data,draw,params,picks);
            else if (selected==0) output.add("graph_selected",piece,data,draw,params,picks);
            else if (selected==2) output.add("graph_selected_weak",piece,data,draw,params,picks);
            const auto [x0,y0]=curve_mid(0);
            const auto [x1,y1]=curve_mid(1);
            const auto [x2,y2]=curve_mid(3);
            const auto [x3,y3]=curve_mid(4);
            const auto [x4,y4]=curve_mid(6);
            const auto [x6,y6]=curve_mid(edges-1);
            if (selected<0) {
                // Overview with no clicked or pinned node draws zero edges (init-cg-link-graph.js:209-238).
                check_rgb(x0,y0,bg_r,bg_g,bg_b,0);
                check_rgb(x1,y1,bg_r,bg_g,bg_b,0);
                check_rgb(x2,y2,bg_r,bg_g,bg_b,0);
                check_rgb(x3,y3,bg_r,bg_g,bg_b,0);
                check_rgb(x4,y4,bg_r,bg_g,bg_b,0);
            } else if (selected==0) {
                // pctInput = +1.0 => PRGn #00441b = (0, 68, 27) at full opacity.
                check_rgb(x0,y0,0,68,27);
                check_rgb(x1,y1,bg_r,bg_g,bg_b,0);
                check_rgb(x2,y2,bg_r,bg_g,bg_b,0);
            } else if (selected==2) {
                // pctInput = +0.2 => PRGn (127, 196, 128) at full opacity; non-incident edges hidden.
                check_rgb(x1,y1,127,196,128);
                check_rgb(x0,y0,bg_r,bg_g,bg_b,0);
                check_rgb(x2,y2,bg_r,bg_g,bg_b,0);
            } else if (selected==4) {
                // pctInput = -1.0 => PRGn #40004b = (64, 0, 75) at full opacity.
                check_rgb(x2,y2,64,0,75);
                check_rgb(x0,y0,bg_r,bg_g,bg_b,0);
            } else if (selected==6) {
                // pctInput = -0.2 => PRGn (173, 138, 189) at full opacity.
                check_rgb(x3,y3,173,138,189);
                check_rgb(x0,y0,bg_r,bg_g,bg_b,0);
            } else if (selected==8) {
                // Dominant +0.69 edge in 32-edge fan paints on top with #00441b = (0, 68, 27).
                check_rgb(x4,y4,0,68,27);
            } else if (selected==12) {
                // Zero-weight edge never draws.
                check_rgb(x6,y6,bg_r,bg_g,bg_b,0);
            }
        }
        output.finish();
    }
}
