// Compiles shaders and prints errors, or writes the outputs. The extension
// says what the source is (oksl.SOURCE_EXTENSIONS): `.sksl` SkSL, `.glsl`
// Shadertoy code, `.frag` a GLSL ES 3.0 fragment shader; anything else is
// SkSL.
//
//     odin run tools/okslc -- shader.sksl [more.sksl tunnel.glsl ...]
//     odin run tools/okslc -- -o out_prefix shader.sksl
//
// `-o` writes out_prefix.spv and out_prefix.metal (one shader).
package okslc

import "core:fmt"
import "core:os"
import oksl "../.."

main :: proc() {
	args := os.args[1:]
	prefix := ""
	if len(args) >= 2 && args[0] == "-o" {
		prefix = args[1]
		args = args[2:]
	}
	if len(args) == 0 {
		fmt.eprintln("usage: okslc [-o out_prefix] shader.sksl|shader.glsl ...")
		os.exit(1)
	}
	failed := 0
	for path in args {
		data, err := os.read_entire_file(path, context.allocator)
		if err != nil {
			fmt.eprintln("can not read", path)
			os.exit(1)
		}
		m: ^oksl.Module
		errors: []oksl.Error
		ok: bool
		options, _ := oksl.options_of_path(path)
		m, errors, ok = oksl.compile(string(data), options)
		if !ok {
			failed += 1
			fmt.printfln("%s: FAILED", path)
			for e in errors {
				fmt.printfln("%s:%d:%d: %s", path, e.pos.line, e.pos.col, oksl.error_message(e, context.temp_allocator))
			}
		} else {
			fmt.printfln("%s: ok, %d SPIR-V words, %d MSL bytes, block %d bytes, %d children", path, len(m.spirv), len(m.msl), m.block_size, m.num_samplers)
			if prefix != "" {
				_ = os.write_entire_file(fmt.tprintf("%s.spv", prefix), ([^]u8)(raw_data(m.spirv))[:len(m.spirv) * 4])
				_ = os.write_entire_file(fmt.tprintf("%s.metal", prefix), transmute([]u8)m.msl)
			}
		}
		oksl.destroy(m)
		delete(data)
	}
	if failed > 0 {
		os.exit(1)
	}
}
