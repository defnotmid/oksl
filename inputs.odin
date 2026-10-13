package oksl

// The host inputs: built-in uniforms at fixed offsets at the start of the
// uniform block (std140), and the four channels (samplers at bindings
// 0..3). One flat table, indexed by `Input`, drives everything: SkSL binds
// a declared `uniform <type> <name>;` of a row's name and type to its slot,
// the GLSL dialect declares every row before the source (Shadertoy's
// inputs), the backends lay out the block from it, and hosts fill it by
// index (`write_input`, `write_builtins`).

Input :: enum u8 {
	Resolution,
	Time,
	Time_Delta,
	Frame,
	Frame_Rate,
	Sample_Rate,
	Mouse,
	Date,
	Channel_Resolution,
	Channel_Time,
	Channel0,
	Channel1,
	Channel2,
	Channel3,
}
Inputs :: bit_set[Input;u16]

// The uniform rows (in block order) and the channel rows.
UNIFORM_INPUTS :: Inputs{.Resolution, .Time, .Time_Delta, .Frame, .Frame_Rate, .Sample_Rate, .Mouse, .Date, .Channel_Resolution, .Channel_Time}
CHANNEL_INPUTS :: Inputs{.Channel0, .Channel1, .Channel2, .Channel3}

Input_Info :: struct {
	name:   string,
	type:   Type,
	// Uniforms: byte offset in the block. Channels: the sampler binding
	// (the channel index).
	offset: int,
}

CHANNEL_COUNT :: 4

OFFSET_RESOLUTION :: 0 // vec3
OFFSET_TIME :: 12 // float
OFFSET_TIME_DELTA :: 16 // float
OFFSET_FRAME :: 20 // int
OFFSET_FRAME_RATE :: 24 // float
OFFSET_SAMPLE_RATE :: 28 // float
OFFSET_MOUSE :: 32 // vec4
OFFSET_DATE :: 48 // vec4
// Channel n's size in pixels (what `.eval` and texelFetch divide by),
// vec3[4], stride 16.
OFFSET_CHANNEL_RESOLUTION :: 64
// Channel n's playback time in seconds, float[4], stride 16.
OFFSET_CHANNEL_TIME :: 128
// User uniforms start here.
OFFSET_USER :: 192

@(rodata)
INPUTS := [Input]Input_Info {
	.Resolution         = {"iResolution", {base = .Float, rows = 3}, OFFSET_RESOLUTION},
	.Time               = {"iTime", T_FLOAT, OFFSET_TIME},
	.Time_Delta         = {"iTimeDelta", T_FLOAT, OFFSET_TIME_DELTA},
	.Frame              = {"iFrame", T_INT, OFFSET_FRAME},
	.Frame_Rate         = {"iFrameRate", T_FLOAT, OFFSET_FRAME_RATE},
	.Sample_Rate        = {"iSampleRate", T_FLOAT, OFFSET_SAMPLE_RATE},
	.Mouse              = {"iMouse", {base = .Float, rows = 4}, OFFSET_MOUSE},
	.Date               = {"iDate", {base = .Float, rows = 4}, OFFSET_DATE},
	.Channel_Resolution = {"iChannelResolution", {base = .Float, rows = 3, array = CHANNEL_COUNT}, OFFSET_CHANNEL_RESOLUTION},
	.Channel_Time       = {"iChannelTime", {base = .Float, rows = 1, array = CHANNEL_COUNT}, OFFSET_CHANNEL_TIME},
	.Channel0           = {"iChannel0", T_SAMPLER, 0},
	.Channel1           = {"iChannel1", T_SAMPLER, 1},
	.Channel2           = {"iChannel2", T_SAMPLER, 2},
	.Channel3           = {"iChannel3", T_SAMPLER, 3},
}

// The uniform input at block offset `offset`.
input_at_offset :: proc "contextless" (offset: int) -> (Input, bool) {
	for in_ in UNIFORM_INPUTS {
		if INPUTS[in_].offset == offset {
			return in_, true
		}
	}
	return {}, false
}

// Writes float components of uniform input `in_` (element `index` of the
// arrays) into a block of at least `Module.block_size` bytes; iFrame takes
// its value converted to int.
write_input :: proc(block: []u8, in_: Input, values: []f32, index := 0) {
	if in_ not_in UNIFORM_INPUTS {
		return
	}
	info := INPUTS[in_]
	stride := std140_array_stride(info.type) if info.type.array > 0 else 0
	write_values(block, info.offset + index * stride, elem_type(info.type), values)
}

// Built-in uniform values, written by `write_builtins`.
Builtins :: struct {
	resolution:         [3]f32,
	time:               f32,
	time_delta:         f32,
	frame:              i32,
	frame_rate:         f32,
	sample_rate:        f32,
	mouse:              [4]f32,
	date:               [4]f32,
	channel_resolution: [CHANNEL_COUNT][3]f32,
	channel_time:       [CHANNEL_COUNT]f32,
}

// Writes every built-in uniform into the start of a uniform block of at
// least `m.block_size` bytes.
write_builtins :: proc(block: []u8, b: Builtins) {
	b := b
	write_input(block, .Resolution, b.resolution[:])
	write_input(block, .Time, {b.time})
	write_input(block, .Time_Delta, {b.time_delta})
	write_input(block, .Frame, {f32(b.frame)})
	write_input(block, .Frame_Rate, {b.frame_rate})
	write_input(block, .Sample_Rate, {b.sample_rate})
	write_input(block, .Mouse, b.mouse[:])
	write_input(block, .Date, b.date[:])
	for ch in 0 ..< CHANNEL_COUNT {
		write_input(block, .Channel_Resolution, b.channel_resolution[ch][:], ch)
		write_input(block, .Channel_Time, {b.channel_time[ch]}, ch)
	}
}

// Float components of a scalar, vector or matrix (columns in order) at
// `offset`, std140: matrix columns are 16 bytes apart. Ints and bools are
// converted.
@(private)
write_values :: proc(block: []u8, offset: int, t: Type, values: []f32) {
	cols := max(int(t.cols), 1)
	k := 0
	for col in 0 ..< cols {
		for row in 0 ..< int(t.rows) {
			if k >= len(values) {
				return
			}
			off := offset + col * 16 + row * 4
			if off + 4 > len(block) {
				return
			}
			v := values[k]
			k += 1
			switch t.base {
			case .Float:
				copy(block[off:off + 4], ([^]u8)(&v)[:4])
			case .Int:
				iv := i32(v)
				copy(block[off:off + 4], ([^]u8)(&iv)[:4])
			case .Uint, .Bool:
				uv := u32(v) if t.base == .Uint else u32(v != 0)
				copy(block[off:off + 4], ([^]u8)(&uv)[:4])
			case .Void, .Sampler, .Struct:
			}
		}
	}
}
