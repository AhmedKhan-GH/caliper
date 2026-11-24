// A packed text row has 64 bytes. Remove an incomplete final UTF-8 sequence.
// The host still validates every other byte before it places a label.
fn text_byte(words: array<u32,16>, at: u32) -> u32 {
    return (words[at/4u] >> (8u*(at%4u))) & 255u;
}
fn bounded_text(packed: array<u32,16>) -> Text {
    var words=packed;
    var end=0u;
    loop {
        if (end==64u) { break; }
        if (text_byte(words,end)==0u) { break; }
        end++;
    }
    if (end>0u) {
        var start=end-1u;
        for (var back=0u; back<3u; back++) {
            if (start==0u || (text_byte(words,start)&192u)!=128u) { break; }
            start--;
        }
        let lead=text_byte(words,start);
        var length=1u;
        if (lead>=194u && lead<=223u) { length=2u; }
        if (lead>=224u && lead<=239u) { length=3u; }
        if (lead>=240u && lead<=244u) { length=4u; }
        if (start+length>end) {
            for (var at=start; at<end; at++) { words[at/4u] &= ~(255u << (8u*(at%4u))); }
        }
    }
    return Text(vec4<u32>(words[0],words[1],words[2],words[3]), vec4<u32>(words[4],words[5],words[6],words[7]),
                vec4<u32>(words[8],words[9],words[10],words[11]), vec4<u32>(words[12],words[13],words[14],words[15]));
}
