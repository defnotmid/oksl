// Renders a shader with the CPU interpreter to a PNG.
//
//     odin run tools/cpu_render -- shader.sksl|.glsl|.frag out.png [width height time]
//
// The extension says what the source is (oksl.SOURCE_EXTENSIONS): `.glsl`
// Shadertoy code, `.frag` a GLSL ES 3.0 fragment shader (its location 0
// input gets the layer's uv), otherwise SkSL.
package cpu_render

import "core:fmt"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:time"
import oksl "../.."
import stbi "vendor:stb/image"

// A 64x64 colour checker for the channels, sampled bilinear with repeat.
TEX :: 64

texel :: proc(x, y: int) -> [4]f32 {
	xx := ((x % TEX) + TEX) % TEX
	yy := ((y % TEX) + TEX) % TEX
	on := ((xx / 8) + (yy / 8)) % 2 == 0
	return {f32(xx) / TEX, f32(yy) / TEX, 0.9 if on else 0.2, 1}
}

sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32) -> [4]f32 {
	fx := uv.x * TEX - 0.5
	fy := uv.y * TEX - 0.5
	x0 := int(fx) if fx >= 0 else int(fx) - 1
	y0 := int(fy) if fy >= 0 else int(fy) - 1
	tx := fx - f32(x0)
	ty := fy - f32(y0)
	a := texel(x0, y0)
	b := texel(x0 + 1, y0)
	c := texel(x0, y0 + 1)
	d := texel(x0 + 1, y0 + 1)
	return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty
}

main :: proc() {
	if len(os.args) < 3 {
		fmt.eprintln("usage: cpu_render shader.sksl|shader.glsl out.png [width height time]")
		os.exit(1)
	}
	w, h := 256, 144
	t: f32 = 1.5
	if len(os.args) > 5 {
		w = strconv.parse_int(os.args[3]) or_else w
		h = strconv.parse_int(os.args[4]) or_else h
		t = f32(strconv.parse_f64(os.args[5]) or_else 1.5)
	}
	data, err := os.read_entire_file(os.args[1], context.allocator)
	if err != nil {
		fmt.eprintln("can not read", os.args[1])
		os.exit(1)
	}
	m: ^oksl.Module
	errors: []oksl.Error
	ok: bool
	options, _ := oksl.options_of_path(os.args[1])
	m, errors, ok = oksl.compile(string(data), options)
	defer oksl.destroy(m)
	if !ok {
		fmt.eprint(oksl.format_errors(errors, context.temp_allocator))
		os.exit(1)
	}
	block := make([]u8, m.block_size)
	oksl.write_builtins(block, {resolution = {f32(w), f32(h), 1}, time = t, frame = 90, frame_rate = 60, channel_resolution = {{TEX, TEX, 1}, {TEX, TEX, 1}, {TEX, TEX, 1}, {TEX, TEX, 1}}})
	oksl.set_uniform(m, block, "uSpeed", {0.5})
	oksl.set_uniform(m, block, "uScale", {1.5})
	oksl.set_uniform(m, block, "uTint", {1, 0.9, 0.8})
	oksl.set_uniform(m, block, "uPalette", {1, 0.2, 0.2, 1}, 0)
	oksl.set_uniform(m, block, "uPalette", {0.2, 1, 0.2, 1}, 1)
	oksl.set_uniform(m, block, "uPalette", {0.2, 0.2, 1, 1}, 2)
	oksl.set_uniform(m, block, "uRot", {0.8, 0.6, -0.6, 0.8})
	oksl.set_uniform(m, block, "uMode", {1})
	oksl.set_uniform(m, block, "uInvert", {0})
	pixels := make([]u8, w * h * 4)
	for i in 0 ..< w * h {
		pixels[i * 4 + 3] = 255
	}
	start := time.tick_now()
	oksl.run(&m.program, block, w, h, pixels, w * 4, {sampler = sampler})
	fmt.printfln("%s: %d instructions, %d slots, %.1f ms", os.args[1], len(m.program.code), m.program.num_slots, time.duration_milliseconds(time.tick_since(start)))
	stbi.write_png(strings.clone_to_cstring(os.args[2]), i32(w), i32(h), 4, raw_data(pixels), i32(w * 4))
}
