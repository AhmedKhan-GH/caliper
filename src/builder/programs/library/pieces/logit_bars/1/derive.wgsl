fn text_row(row: u32) -> Text {
    let b = row*16u;
    var words: array<u32,16>;
    for (var i=0u; i<16u; i++) { words[i]=label_text[b+i]; }
    return bounded_text(words);
}

@compute @workgroup_size(16)
fn output_bars(@builtin(global_invocation_id) id: vec3<u32>) {
    let i=id.x;
    let valid=shapes.probs.z==2u && shapes.probs.w==10u && shapes.tokens.w==10u;
    if (i==0u) { atomicStore(&cells_count,select(0u,10u,valid)); }
    if (i>=10u) { return; }
    labels[i]=Label(vec4<f32>(-0.81+f32(i)*0.18,-0.86,0.0,1.0),select(0xffffffffu,i,valid));
    var text: Text;
    if (shapes.label_text.z==10u && shapes.label_text.w==16u) { text=text_row(i); }
    strings[i]=text;
    if (!valid) { return; }
    var baseline=probs[i];
    var probability=probs[10u+i];
    if (!(abs(baseline)<=3.402823e38)) { baseline=0.0; }
    if (!(abs(probability)<=3.402823e38)) { probability=0.0; }
    baseline=clamp(baseline,0.0,1.0);
    probability=clamp(probability,0.0,1.0);
    let x=-0.88+f32(i)*0.18;
    cells[i]=Bar(vec4<f32>(x,-0.72,x+0.14,0.72),baseline,probability,tokens[i],i);
}
