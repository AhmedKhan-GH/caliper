fn text_row(row: u32) -> Text {
    let b = row*16u;
    return Text(vec4<u32>(label_text[b],label_text[b+1u],label_text[b+2u],label_text[b+3u]),
                vec4<u32>(label_text[b+4u],label_text[b+5u],label_text[b+6u],label_text[b+7u]),
                vec4<u32>(label_text[b+8u],label_text[b+9u],label_text[b+10u],label_text[b+11u]),
                vec4<u32>(label_text[b+12u],label_text[b+13u],label_text[b+14u],label_text[b+15u]));
}

@compute @workgroup_size(8)
fn example_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let row = id.x;
    let example = min(params.page,1u)*8u+row;
    var count = 0u;
    if (shapes.example_ids.w == 16u && shapes.valid_tokens.w == 16u) { count = u32(clamp(valid_tokens[example],0,25)); }
    rows[row] = Row(example_ids[example],example,row,count);
}
@compute @workgroup_size(64)
fn token_cells(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i>=200u) { return; }
    let row = rows[i/25u];
    let col = i%25u;
    let source = row.example*25u+col;
    let valid = col<row.token_count && shapes.tokens.z==16u && shapes.tokens.w==25u &&
                shapes.activations.z==16u && shapes.activations.w==25u;
    let x = -0.95+f32(col)*1.9/25.0;
    let y = 0.9-f32(i/25u)*1.8/8.0;
    let rect = vec4<f32>(x,y-0.18,x+1.9/25.0-0.004,y);
    var value = activations[source];
    if (!(abs(value)<=3.402823e38)) { value=0.0; }
    cells[i] = Cell(rect,value,tokens[source],i/25u,select(0u,1u,valid));
    labels[i] = Label(vec4<f32>((rect.xy+rect.zw)*0.5,0.0,1.0),select(0xffffffffu,i,valid));
}
@compute @workgroup_size(64)
fn token_text(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i>=200u) { return; }
    var text: Text;
    if (shapes.label_text.z==400u && shapes.label_text.w==16u) { text=text_row(min(params.page,1u)*200u+i); }
    strings[i]=text;
}
