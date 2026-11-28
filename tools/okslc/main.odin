// Compiles shaders and prints errors, or writes the outputs. The extension
// says what the source is (oksl.SOURCE_EXTENSIONS): `.sksl` SkSL, `.glsl`
// Shadertoy code, `.frag`/`.vert`/`.comp` GLSL ES 3.0 / 3.1; anything else
// is SkSL.
//
//     odin run tools/okslc -- [--oplat] [--odin-asserts] [-o out_prefix] shader.sksl ...
//
// `--oplat` selects Layout.Oplat (Vulkan 1.3 SPIR-V 1.6 + Metal 3.0 MSL; no
// DXBC). `--odin-asserts` implies `--oplat` and prints `#assert(offset_of(...))`
// and `#assert(size_of(...))` lines for the push block and its structs.
// `-o` writes out_prefix.spv, out_prefix.metal and out_prefix.dxbc (one
// shader). Under SDL_GPU, DXBC is on, so what Direct3D 12 lacks
// (gl_NumWorkGroups) is an error here.
package okslc

import "core:fmt"
import "core:os"
import oksl "../.."

main :: proc() {
	args := os.args[1:]
	prefix := ""
	oplat := false
	odin_asserts := false
	files := make([dynamic]string, context.temp_allocator)
	for i := 0; i < len(args); i += 1 {
		a := args[i]
		switch {
		case a == "--oplat":
			oplat = true
		case a == "--odin-asserts":
			odin_asserts = true
			oplat = true
		case a == "-o":
			if i + 1 >= len(args) {
				fmt.eprintln("-o requires an output prefix")
				os.exit(1)
			}
			i += 1
			prefix = args[i]
		case:
			append(&files, a)
		}
	}
	if len(files) == 0 {
		fmt.eprintln("usage: okslc [--oplat] [--odin-asserts] [-o out_prefix] shader.sksl|shader.glsl|shader.frag ...")
		os.exit(1)
	}
	failed := 0
	for path in files {
		data, err := os.read_entire_file(path, context.allocator)
		if err != nil {
			fmt.eprintln("can not read", path)
			os.exit(1)
		}
		m: ^oksl.Module
		errors: []oksl.Error
		ok: bool
		options, _ := oksl.options_of_path(path)
		if oplat {
			options.layout = .Oplat
		} else {
			options.dxbc = true
		}
		m, errors, ok = oksl.compile(string(data), options)
		if !ok {
			failed += 1
			fmt.printfln("%s: FAILED", path)
			for e in errors {
				fmt.printfln("%s:%d:%d: %s", path, e.pos.line, e.pos.col, oksl.error_message(e, context.temp_allocator))
			}
		} else {
			if odin_asserts {
				fmt.print(oksl.format_odin_asserts(m, context.temp_allocator))
			} else {
				fmt.printfln("%s: ok, %d SPIR-V words, %d MSL bytes, %d DXBC bytes, block %d bytes, %d children", path, len(m.spirv), len(m.msl), len(m.dxbc), m.block_size, m.num_samplers)
			}
			if prefix != "" {
				_ = os.write_entire_file(fmt.tprintf("%s.spv", prefix), ([^]u8)(raw_data(m.spirv))[:len(m.spirv) * 4])
				_ = os.write_entire_file(fmt.tprintf("%s.metal", prefix), transmute([]u8)m.msl)
				if len(m.dxbc) > 0 {
					_ = os.write_entire_file(fmt.tprintf("%s.dxbc", prefix), m.dxbc)
				}
			}
		}
		oksl.destroy(m)
		delete(data)
	}
	if failed > 0 {
		os.exit(1)
	}
}
