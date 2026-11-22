---
kind: reference
---

# Wire Frames

This page defines the binary frame that carries every message of [Wire Protocol](wire-protocol.md) v3.

## Frames

Every WebSocket message is one binary **frame**. All integers in it are little-endian.

### Frame Layout

| Offset | Size | Type | Field | Rule |
|---|---|---|---|---|
| 0 | 4 | u32 | `magic` | `0x574C4143`, the bytes `C A L W` |
| 4 | 2 | u16 | `major` | 3 |
| 6 | 2 | u16 | `minor` | The minor version of the sender: 1 in 3.1, 0 in 3.0 |
| 8 | 2 | u16 | `type` | The [message type](wire-protocol.md#message-types) |
| 10 | 2 | u16 | `flags` | Frame flags. Version 3.1 defines none, so senders write 0 |
| 12 | 4 | u32 | `frame_len` | Bytes of the whole frame, this header included |
| 16 | 4 | u32 | `meta_len` | Bytes of the metadata |
| 20 | 2 | u16 | `section_count` | Payload sections, 0 to 8 |
| 22 | 2 | u16 | `reserved` | 0 when sent, ignored when received |
| 24 | 4 × `section_count` | u32 each | `section_len[i]` | Bytes of section i |
| 24 + 4 × `section_count` | `meta_len` | UTF-8 | metadata | One JSON object |
| See below | `section_len[i]` | bytes | section i | A raw array |

Section i starts at the first offset that is a multiple of 8, counted from the first byte of the frame, at or after the end of the part before it. The part before section 0 is the metadata. The sender writes zero bytes in the gaps, and the receiver ignores them. `frame_len` is the end of the last section, or the end of the metadata when there is no section. No bytes follow. Because every section starts at a multiple of 8, a browser that receives the frame as an `ArrayBuffer` can wrap a section in a `Float32Array` or `Uint8Array` without a copy.

### Metadata

The metadata is one JSON object (RFC 8259) in UTF-8. It is at least 2 bytes long, has no byte order mark, and nests objects and arrays at most 16 deep. It holds no `NaN` or `Infinity`, which JSON cannot express. A value that may be non-finite is written as `null`. A sender never repeats a key in an object. An optional key with the value `null` means the same as an absent key.

Integers are exact in every implementation up to 2<sup>53</sup>. The only field that can exceed that is `stamp_ns`, a monotonic clock in nanoseconds. A JavaScript reader then loses a few nanoseconds of it, which only affects the display of latency.

A receiver accepts keys in any order and any whitespace. The Python encoder and the golden frames write the canonical form: no whitespace, keys sorted, and non-ASCII characters as raw UTF-8. Encoders in other languages need not produce the same bytes.

### Sections

When `section_count` is not 0, the metadata key `sections` is an array with one descriptor per section, in order. The key `sections` is reserved in every message.

| Key | Type | Meaning |
|---|---|---|
| `name` | string | The name of the section, unique in the frame. Each message names its sections |
| `dtype` | integer | A dtype id from the table below |
| `shape` | array of integers | The shape of the array, row-major |
| `encoding` | integer, optional | 0, a raw array (the default). 1, Arrow IPC, is reserved for a future table kind and not used in 3.0 |

A raw section holds the product of `shape` times the item size bytes, little-endian, row-major.

| Id | Name | Item size | numpy | TypedArray |
|---|---|---|---|---|
| 0 | `f32` | 4 | `<f4` | `Float32Array` |
| 1 | `f16` | 2 | `<f2` | `Float16Array` where available |
| 2 | `bf16` | 2 | none | `Uint16Array` (raw bits) |
| 3 | `i64` | 8 | `<i8` | `BigInt64Array` |
| 4 | `i32` | 4 | `<i4` | `Int32Array` |
| 5 | `u8` | 1 | `u1` | `Uint8Array` |

Producers of 3.1 send only `f32` and `u8` payloads. The other ids exist so that a receiver can name what it refuses.

### Frame Decoding

A receiver applies these checks in this order, so every implementation rejects a bad frame with the same close code:

| Step | Check | Close code |
|---|---|---|
| 1 | Fewer than 8 bytes | 4000 |
| 2 | `magic` is not `0x574C4143` | 4000 |
| 3 | `major` is not 3 | 4001 |
| 4 | Fewer than 24 bytes | 4000 |
| 5 | `frame_len` over 64 MiB, `meta_len` over 1 MiB, or `section_count` over 8 | 1009 |
| 6 | The message length is not `frame_len` | 4000 |
| 7 | The layout does not end exactly at `frame_len` | 4000 |
| 8 | The metadata is not a JSON object as described above | 4000 |
| 9 | `sections` is missing or has another count, a descriptor is malformed, or a raw section of a known dtype has a length that disagrees with its shape | 4000 |

`minor`, `flags`, `reserved` and the padding bytes are never grounds to reject a frame. A frame that passes every check is **well formed**. If a well-formed frame holds an invalid message, such as one with a missing field, an unknown view or a bad data view, the receiver answers with an ERROR message and the connection stays open. If a section has an encoding or dtype the receiver does not support, the receiver ignores the message and answers ERROR `unsupported`.

## Worked Example

This is the golden frame `tests/fixtures/wire_v3/series.bin`. It holds three values of the series `loss` for subscription 3: 2.5, 1.25 and 0.75. They are the first three values ever appended. The frame comes from a 3.0 sender, so its minor version is 0. A 3.1 sender writes 1 there.

```text
0000  43 41 4c 57 03 00 00 00 31 00 00 00 a4 00 00 00
0010  7a 00 00 00 01 00 00 00 0c 00 00 00 7b 22 73 65
0020  63 74 69 6f 6e 73 22 3a 5b 7b 22 64 74 79 70 65
0030  22 3a 30 2c 22 6e 61 6d 65 22 3a 22 76 61 6c 75
0040  65 73 22 2c 22 73 68 61 70 65 22 3a 5b 33 5d 7d
0050  5d 2c 22 73 74 61 6d 70 5f 6e 73 22 3a 38 31 32
0060  33 34 35 36 37 38 39 31 31 32 33 2c 22 73 74 61
0070  72 74 22 3a 30 2c 22 73 75 62 22 3a 33 2c 22 74
0080  6f 74 61 6c 22 3a 33 2c 22 76 69 65 77 22 3a 22
0090  6c 6f 73 73 22 7d 00 00 00 00 20 40 00 00 a0 3f
00a0  00 00 40 3f
```

| Offset | Bytes | Field | Value |
|---|---|---|---|
| 0 | `43 41 4c 57` | `magic` | `CALW` |
| 4 | `03 00` | `major` | 3 |
| 6 | `00 00` | `minor` | 0 |
| 8 | `31 00` | `type` | `0x0031`, series |
| 10 | `00 00` | `flags` | 0 |
| 12 | `a4 00 00 00` | `frame_len` | 164 |
| 16 | `7a 00 00 00` | `meta_len` | 122 |
| 20 | `01 00` | `section_count` | 1 |
| 22 | `00 00` | `reserved` | 0 |
| 24 | `0c 00 00 00` | `section_len[0]` | 12 |
| 28 | `7b 22` to `22 7d` | metadata | 122 bytes, ending at offset 150 |
| 150 | `00 00` | padding | up to 152, the next multiple of 8 |
| 152 | `00 00 20 40` | `values[0]` | 2.5 |
| 156 | `00 00 a0 3f` | `values[1]` | 1.25 |
| 160 | `00 00 40 3f` | `values[2]` | 0.75 |

The metadata, in canonical form:

```json
{"sections":[{"dtype":0,"name":"values","shape":[3]}],"stamp_ns":81234567891123,"start":0,"sub":3,"total":3,"view":"loss"}
```

The values section starts at offset 152, so a browser reads it as `new Float32Array(buffer, 152, 3)`.
