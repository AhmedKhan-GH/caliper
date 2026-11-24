fn text_row(row: u32) -> Text {
    let b = row*16u;
    var words: array<u32,16>;
    for (var i=0u; i<16u; i++) { words[i]=label_text[b+i]; }
    return bounded_text(words);
}

@compute @workgroup_size(8)
fn example_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    if (row>=8u) { return; }
    let example = min(params.page,1u)*8u+row;
    rows[row] = Row(-1,example,row,0u);
    if (shapes.example_ids.w!=16u || shapes.valid_tokens.w!=16u ||
        arrayLength(&example_ids)<16u || arrayLength(&valid_tokens)<16u) { return; }
    rows[row] = Row(example_ids[example],example,row,u32(clamp(valid_tokens[example],0,25)));
}
@compute @workgroup_size(64)
fn token_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i>=200u) { return; }
    let row = rows[i/25u];
    let col = i%25u;
    let source = row.example*25u+col;
    let valid = col<row.token_count && shapes.tokens.z==16u && shapes.tokens.w==25u &&
                shapes.activations.z==16u && shapes.activations.w==25u &&
                arrayLength(&tokens)>=400u && arrayLength(&activations)>=400u;
    let cell_width = 1.9/f32(max(row.token_count,1u));
    let x = -0.95+f32(col)*cell_width;
    let y = 0.9-f32(i/25u)*1.8/8.0;
    let rect = vec4<f32>(x,y-0.18,x+cell_width-0.004,y);
    cells[i] = Cell(rect,0.0,-1,i/25u,0u);
    labels[i] = Label(vec4<f32>((rect.xy+rect.zw)*0.5,0.0,1.0),0xffffffffu);
    if (!valid) { return; }
    var value = activations[source];
    if (!(abs(value)<=3.402823e38)) { value=0.0; }
    cells[i] = Cell(rect,value,tokens[source],i/25u,1u);
    labels[i].string = i;
}
@compute @workgroup_size(64)
fn token_text(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i>=200u) { return; }
    var text: Text;
    if (cells[i].valid!=0u && shapes.label_text.z==400u && shapes.label_text.w==16u &&
        arrayLength(&label_text)>=6400u) { text=text_row(min(params.page,1u)*200u+i); }
    strings[i]=text;
}
