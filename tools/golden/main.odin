// Golden snapshots of what oksl produces, for refactors that should not
// change it (or should change only what they say).
//
//     odin run tools/golden -- snapshot out_dir
//     odin run tools/golden -- compare ref_dir out_dir
//
// `snapshot` writes out_dir/corpus.tsv, one line per generated case (every
// built-in over every combination of argument types, every operator over
// every pair of types, compound assignments, increments, constructors and
// ?:): the diagnostics, or the result type, hashes of the SPIR-V, MSL and
// interpreter code, and a hash of the CPU result with the value folded into
// the output colour. Per shader in testdata/ and demo/media/ it writes the
// .spv, the .metal, a float CPU render (.f32) and the reflection the host
// sees (.reflect), or the diagnostics (.errors).
//
// `compare` reports, per kind of output, which cases and files differ;
// it exits 1 when anything does.
package golden

import "core:fmt"
import "core:hash"
import "core:os"
import "core:path/filepath"
import "core:slice"
import "core:strings"
import oksl "../.."

Arg :: struct {
	name, decl: string,
}

// Locals computed from the coordinate (not constant-foldable), and two
// literals.
POOL := []Arg {
	{"f1", "float f1 = p.x * 0.37 - 1.1;"},
	{"f2", "float2 f2 = p * 0.29 - 0.6;"},
	{"f3", "float3 f3 = float3(p.x * 0.21 - 0.4, p.y * 0.7 - 0.9, 0.3 - p.x * 0.05);"},
	{"f4", "float4 f4 = float4(p.x * 0.1, -p.y, 0.75, p.x * p.y * 0.03 - 0.2);"},
	{"i1", "int i1 = int(p.x) - 3;"},
	{"i3", "int3 i3 = int3(int(p.x) - 4, int(p.y) * 7 - 5, 9 - int(p.x) * 3);"},
	{"u1", "uint u1 = uint(p.x) * 3u + 1u;"},
	{"u2", "uint2 u2 = uint2(uint(p.x) + 2u, uint(p.y) * 5u);"},
	{"b1", "bool b1 = p.x > 3.5;"},
	{"b3", "bool3 b3 = bool3(p.x > 2.0, p.y > 1.0, p.x < 6.0);"},
	{"m2", "float2x2 m2 = float2x2(p.x * 0.2 - 0.3, 0.5, p.y - 1.2, 0.25 + p.x * 0.1);"},
	{"m3", "float3x3 m3 = float3x3(p.x * 0.2, 0.5, -0.3, p.y, 1.0, 0.25, 0.7, p.x * 0.05 - 0.1, 0.4);"},
	{"m23", "float2x3 m23 = float2x3(p.x * 0.1, 0.2, 0.3, -0.4, p.y, 0.6);"},
	{"2", ""},
	{"0.5", ""},
}

// The variables of POOL (assignable), and the subset for three arguments.
VARIABLES :: 13
POOL3 := []int{0, 1, 2, 4, 5, 6, 8, 9, 10, 14}

BUILTINS := []string {
	"radians", "degrees", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
	"pow", "exp", "log", "exp2", "log2", "sqrt", "inversesqrt", "abs", "sign", "floor", "ceil", "fract", "trunc", "round",
	"roundEven", "mod", "min", "max", "clamp", "mix", "step", "smoothstep", "isnan", "isinf", "length", "distance", "dot",
	"cross", "normalize", "faceforward", "reflect", "refract", "matrixCompMult", "outerProduct", "transpose", "determinant",
	"inverse", "lessThan", "lessThanEqual", "greaterThan", "greaterThanEqual", "equal", "notEqual", "any", "all", "not",
	"dFdx", "dFdy", "fwidth", "floatBitsToInt", "floatBitsToUint", "intBitsToFloat", "uintBitsToFloat", "saturate",
}

BINOPS := []string{"+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "&&", "||", "^^", "==", "!=", "<", ">", "<=", ">="}
COMPOUND := []string{"+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}
UNARY := []string{"-", "!", "~", "+"}
CTORS := []string {
	"float", "int", "uint", "bool", "float2", "float3", "float4", "int2", "int3", "uint2", "uint3", "bool2", "bool3",
	"float2x2", "float3x3", "float2x3", "float3x2",
}

// Corpus columns after the case: "ok" type spv msl vm cpu, or "error" text.
COLUMNS :: [?]string{"result", "type", "spv", "msl", "vm", "cpu"}

out: strings.Builder

main :: proc() {
	args := os.args[1:]
	switch {
	case len(args) == 2 && args[0] == "snapshot":
		snapshot(args[1])
	case len(args) == 3 && args[0] == "compare":
		if !compare(args[1], args[2]) {
			os.exit(1)
		}
	case:
		fmt.eprintln("usage: golden snapshot out_dir | golden compare ref_dir out_dir")
		os.exit(1)
	}
}

snapshot :: proc(dir: string) {
	_ = os.make_directory(dir)
	out = strings.builder_make()
	files(dir)
	corpus()
	_ = os.write_entire_file(fmt.tprintf("%s/corpus.tsv", dir), out.buf[:])
	fmt.printfln("%s: %d cases", dir, strings.count(strings.to_string(out), "\n"))
}

// ---- generated cases --------------------------------------------------------------------

decls :: proc(sb: ^strings.Builder) {
	for a in POOL {
		if a.decl != "" {
			fmt.sbprintf(sb, "\t%s\n", a.decl)
		}
	}
}

decl_type :: proc(a: Arg) -> string {
	return a.decl[:strings.index_byte(a.decl, ' ')]
}

// A float4 made from `r`, a value of the SkSL type named `t`.
fold :: proc(t: string) -> string {
	switch t {
	case "float":
		return "float4(r, r * 2.0, -r, 1.0)"
	case "int", "uint":
		return "float4(float(r), 0.0, 0.0, 1.0)"
	case "bool":
		return "float4(r ? 1.0 : 0.0, 0.0, 0.0, 1.0)"
	case "float2":
		return "float4(r, r.yx)"
	case "float3":
		return "float4(r, 1.0)"
	case "float4":
		return "r"
	}
	if strings.has_prefix(t, "float") && strings.contains(t, "x") {
		// float<C>x<R>
		cols := int(t[5] - '0')
		return fmt.tprintf("float4(r[0].xy, r[%d].xy) + float4(r[0][1])", cols - 1)
	}
	n := int(t[len(t) - 1] - '0')
	ft := fmt.tprintf("float%d", n)
	conv := fmt.tprintf("%s(r)", ft)
	if strings.has_prefix(t, "bool") {
		conv = fmt.tprintf("mix(%s(0.0), %s(1.0), r)", ft, ft)
	}
	switch n {
	case 2:
		return fmt.tprintf("float4(%s, 0.5, 1.0)", conv)
	case 3:
		return fmt.tprintf("float4(%s, 1.0)", conv)
	}
	return conv
}

code_hashes :: proc(m: ^oksl.Module) -> string {
	spv := hash.fnv64a(slice.to_bytes(m.spirv))
	msl := hash.fnv64a(transmute([]u8)m.msl)
	vm: u64 = 0xcbf29ce484222325
	for in_ in m.program.code {
		fields := [8]i64{i64(in_.op), i64(in_.n), i64(in_.k), i64(in_.dst), i64(in_.a), i64(in_.b), i64(in_.c), i64(in_.d)}
		vm = hash.fnv64a(slice.to_bytes(fields[:]), vm)
	}
	return fmt.tprintf("%x\t%x\t%x", spv, msl, vm)
}

// Runs an 8x2 layer (two rows of four quads) and hashes the colours.
cpu_hash :: proc(m: ^oksl.Module) -> string {
	block := make([]u8, m.block_size, context.temp_allocator)
	oksl.write_builtins(block, {resolution = {8, 2, 1}, time = 1.25})
	px := make([][4]f32, 16, context.temp_allocator)
	scratch: oksl.Scratch
	defer oksl.scratch_destroy(&scratch)
	oksl.run_f32(&m.program, block, 8, 2, px, &scratch)
	return fmt.tprintf("%x", hash.fnv64a(slice.to_bytes(px)))
}

one_line :: proc(errs: []oksl.Error) -> string {
	text := strings.trim_space(oksl.format_errors(errs, context.temp_allocator))
	return strings.join(strings.split_lines(text, context.temp_allocator), " | ", context.temp_allocator)
}

// Compiles `expr` (after `prefix` statements) in a shader declaring the
// pool. A valid expression's type comes from the diagnostic for assigning
// it to a struct; it is then compiled again with its value folded into the
// output and run.
expr_case :: proc(key, expr: string, prefix := "") {
	sb := strings.builder_make(context.temp_allocator)
	fmt.sbprint(&sb, "struct Zz { float z; };\nhalf4 main(float2 p) {\n")
	decls(&sb)
	fmt.sbprintf(&sb, "\t%s\n\tZz r = %s;\n\treturn half4(0);\n}}\n", prefix, expr)
	m, errs, ok := oksl.compile(strings.to_string(sb))
	t := ""
	if !ok {
		e0 := errs[0]
		if len(errs) == 1 && e0.code == .Init_Mismatch && e0.name == "r" && e0.a == "Zz" {
			t = strings.clone(e0.b, context.temp_allocator)
		} else {
			fmt.sbprintf(&out, "%s\terror\t%s\n", key, one_line(errs))
			oksl.destroy(m)
			return
		}
	}
	oksl.destroy(m)
	sb2 := strings.builder_make(context.temp_allocator)
	fmt.sbprint(&sb2, "half4 main(float2 p) {\n")
	decls(&sb2)
	fmt.sbprintf(&sb2, "\t%s\n\t%s r = %s;\n\treturn %s;\n}}\n", prefix, t, expr, fold(t))
	m2, errs2, ok2 := oksl.compile(strings.to_string(sb2))
	if ok2 {
		fmt.sbprintf(&out, "%s\tok\t%s\t%s\t%s\n", key, t, code_hashes(m2), cpu_hash(m2))
	} else {
		fmt.sbprintf(&out, "%s\terror\tfolding %s failed: %s\n", key, t, one_line(errs2))
	}
	oksl.destroy(m2)
}

corpus :: proc() {
	for b in BUILTINS {
		expr_case(fmt.tprintf("%s()", b), fmt.tprintf("%s()", b))
		for a in POOL {
			e := fmt.tprintf("%s(%s)", b, a.name)
			expr_case(e, e)
		}
		for a in POOL {
			for c in POOL {
				e := fmt.tprintf("%s(%s, %s)", b, a.name, c.name)
				expr_case(e, e)
				free_all(context.temp_allocator)
			}
		}
		for i in POOL3 {
			for j in POOL3 {
				for k in POOL3 {
					e := fmt.tprintf("%s(%s, %s, %s)", b, POOL[i].name, POOL[j].name, POOL[k].name)
					expr_case(e, e)
					free_all(context.temp_allocator)
				}
			}
		}
	}
	for op in BINOPS {
		for a in POOL {
			for c in POOL {
				e := fmt.tprintf("%s %s %s", a.name, op, c.name)
				expr_case(e, e)
				free_all(context.temp_allocator)
			}
		}
	}
	// x is a copy of a, modified in place.
	for op in COMPOUND {
		for a in POOL[:VARIABLES] {
			for c in POOL {
				key := fmt.tprintf("%s %s %s", a.name, op, c.name)
				expr_case(key, "x", fmt.tprintf("%s x = %s; x %s %s;", decl_type(a), a.name, op, c.name))
				free_all(context.temp_allocator)
			}
		}
	}
	for op in UNARY {
		for a in POOL {
			e := fmt.tprintf("%s%s", op, a.name)
			expr_case(e, e)
		}
	}
	for a in POOL[:VARIABLES] {
		for op in ([]string{"++", "--"}) {
			expr_case(fmt.tprintf("%s%s", a.name, op), "x", fmt.tprintf("%s x = %s; x%s;", decl_type(a), a.name, op))
			expr_case(fmt.tprintf("%s%s", op, a.name), "x", fmt.tprintf("%s x = %s; %sx;", decl_type(a), a.name, op))
		}
	}
	for t in CTORS {
		for a in POOL {
			e := fmt.tprintf("%s(%s)", t, a.name)
			expr_case(e, e)
		}
		for a in POOL {
			for c in POOL {
				e := fmt.tprintf("%s(%s, %s)", t, a.name, c.name)
				expr_case(e, e)
				free_all(context.temp_allocator)
			}
		}
	}
	for a in POOL {
		for c in POOL {
			e := fmt.tprintf("b1 ? %s : %s", a.name, c.name)
			expr_case(e, e)
			free_all(context.temp_allocator)
		}
	}
}

// ---- shader files -----------------------------------------------------------------------

// The channels: uv and lod as colour, so sampling shows in the render.
sampler :: proc(user: rawptr, channel: int, uv: [2]f32, lod: f32) -> [4]f32 {
	return {uv.x, uv.y, f32(channel) * 0.25 + lod, 1}
}

files :: proc(dir: string) {
	paths := make([dynamic]string)
	for sub in ([]string{"testdata", "demo/media"}) {
		fis, _ := os.read_all_directory_by_path(sub, context.allocator)
		for fi in fis {
			if o, known := oksl.options_of_path(fi.name); known && o.stage == .Fragment {
				append(&paths, fi.fullpath)
			}
		}
	}
	for path in paths {
		data, _ := os.read_entire_file(path, context.allocator)
		m: ^oksl.Module
		errs: []oksl.Error
		ok: bool
		options, _ := oksl.options_of_path(path)
		m, errs, ok = oksl.compile(string(data), options)
		base := fmt.tprintf("%s/%s", dir, filepath.base(path))
		if !ok {
			_ = os.write_entire_file(fmt.tprintf("%s.errors", base), transmute([]u8)oksl.format_errors(errs))
			continue
		}
		refl := strings.builder_make()
		fmt.sbprintf(&refl, "inputs_used=%v\nchannels_used=%v\nnum_samplers=%d\nchannel_names=%v\nchannel_pos=%v\nblock_size=%d\n", m.inputs_used, m.channels_used, m.num_samplers, m.channel_names, m.channel_pos, m.block_size)
		for un in m.uniforms {
			fmt.sbprintf(&refl, "uniform %v\n", un)
		}
		_ = os.write_entire_file(fmt.tprintf("%s.reflect", base), refl.buf[:])
		_ = os.write_entire_file(fmt.tprintf("%s.spv", base), slice.to_bytes(m.spirv))
		_ = os.write_entire_file(fmt.tprintf("%s.metal", base), transmute([]u8)m.msl)
		W, H :: 96, 54
		block := make([]u8, m.block_size)
		oksl.write_builtins(block, {resolution = {W, H, 1}, time = 1.5, frame = 90, frame_rate = 60, mouse = {30, 20, 1, 0}, channel_resolution = {{64, 64, 1}, {32, 32, 1}, {64, 64, 1}, {64, 64, 1}}})
		oksl.set_uniform(m, block, "uSpeed", {0.5})
		oksl.set_uniform(m, block, "uScale", {1.5})
		oksl.set_uniform(m, block, "uTint", {1, 0.9, 0.8})
		oksl.set_uniform(m, block, "uPalette", {1, 0.2, 0.2, 1}, 0)
		oksl.set_uniform(m, block, "uPalette", {0.2, 1, 0.2, 1}, 1)
		oksl.set_uniform(m, block, "uRot", {0.8, 0.6, -0.6, 0.8})
		oksl.set_uniform(m, block, "uMode", {1})
		// Discarded pixels keep this.
		px := make([][4]f32, W * H)
		for &p in px {
			p = {-7, -7, -7, -7}
		}
		scratch: oksl.Scratch
		oksl.run_f32(&m.program, block, W, H, px, &scratch, {sampler = sampler})
		oksl.scratch_destroy(&scratch)
		_ = os.write_entire_file(fmt.tprintf("%s.f32", base), slice.to_bytes(px))
		oksl.destroy(m)
	}
}

// ---- compare ----------------------------------------------------------------------------

read_corpus :: proc(dir: string) -> (keys: [dynamic]string, rows: map[string][]string) {
	data, err := os.read_entire_file(fmt.tprintf("%s/corpus.tsv", dir), context.allocator)
	if err != nil {
		fmt.eprintfln("can not read %s/corpus.tsv", dir)
		os.exit(1)
	}
	rows = make(map[string][]string)
	for line in strings.split_lines(string(data)) {
		if line == "" {
			continue
		}
		cols := strings.split(line, "\t")
		append(&keys, cols[0])
		rows[cols[0]] = cols[1:]
	}
	return
}

compare :: proc(ref, dir: string) -> bool {
	same := true
	ref_keys, ref_rows := read_corpus(ref)
	_, rows := read_corpus(dir)
	COUNT :: len(COLUMNS)
	diffs: [COUNT][dynamic]string
	missing := make([dynamic]string)
	for key in ref_keys {
		a := ref_rows[key]
		b, found := rows[key]
		if !found {
			append(&missing, key)
			continue
		}
		if a[0] != b[0] || a[0] == "error" {
			// The outcome, or the diagnostics.
			if !slice.equal(a, b) {
				append(&diffs[0], key)
			}
			continue
		}
		for col in 1 ..< COUNT {
			if a[col] != b[col] {
				append(&diffs[col], key)
			}
		}
	}
	columns := COLUMNS
	fmt.printfln("corpus: %d cases", len(ref_keys))
	for col in 0 ..< COUNT {
		if len(diffs[col]) == 0 {
			continue
		}
		same = false
		fmt.printfln("  %s differs in %d: %s", columns[col], len(diffs[col]), sample(diffs[col][:]))
	}
	if len(missing) > 0 {
		same = false
		fmt.printfln("  missing %d: %s", len(missing), sample(missing[:]))
	}

	fis, _ := os.read_all_directory_by_path(ref, context.allocator)
	by_ext := make(map[string][dynamic]string)
	for fi in fis {
		if fi.name == "corpus.tsv" {
			continue
		}
		a, _ := os.read_entire_file(fi.fullpath, context.allocator)
		b, err := os.read_entire_file(fmt.tprintf("%s/%s", dir, fi.name), context.allocator)
		if err == nil && slice.equal(a, b) {
			continue
		}
		ext := filepath.ext(fi.name)
		list := by_ext[ext]
		append(&list, strings.trim_suffix(fi.name, ext))
		by_ext[ext] = list
	}
	fmt.printfln("files: %d", len(fis) - 1)
	for ext, names in by_ext {
		same = false
		slice.sort(names[:])
		fmt.printfln("  %s differs in %d: %s", ext, len(names), sample(names[:]))
	}
	if same {
		fmt.println("identical")
	}
	return same
}

sample :: proc(keys: []string) -> string {
	N :: 8
	s := strings.join(keys[:min(len(keys), N)], ", ")
	return s if len(keys) <= N else fmt.tprintf("%s, ...", s)
}
