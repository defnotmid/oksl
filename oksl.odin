// oksl compiles shaders to SPIR-V and MSL for SDL_GPU, and to a program the
// CPU interpreter runs, with no external compiler. One front end (lexer,
// preprocessor, parser, checker) takes three kinds of source: SkSL (Skia
// runtime effects, `half4 main(float2 coord)`), Shadertoy's GLSL ES 3
// (`mainImage`, through `compile_shadertoy`) and GLSL ES 3.0 fragment
// shaders (`void main()` with in and out variables, `compile_glsl`). See
// README.md.
package oksl

import "core:mem/virtual"
import "core:strings"

Options :: struct {
	// The kind of source (dialect.odin): Skia's SkSL effects, Shadertoy's
	// GLSL or a GLSL ES 3.0 fragment shader.
	profile:  Profile,
	// The pipeline stage (the profile's PROFILE_RULES stages).
	stage:    Stage,
	// Source preprocessed and checked ahead of the shader's own, whose
	// macros apply to it: a host's built-in declarations and #defines.
	// Its errors are the Prelude's; uniforms and inputs it declares that
	// the shader never uses are left out (GLSL profile).
	prelude:  string,
	// Shadertoy: return mainImage's colour as it is (unclamped, with its
	// alpha) rather than clamped and opaque, for a pass that renders into a
	// float target another pass reads (compile_program's buffers).
	unclamped: bool,
	// Skip outputs the host does not need.
	no_spirv: bool,
	no_msl:   bool,
	no_cpu:   bool,
	// Vertex shaders: runs of main for other environments, each giving the
	// vertex's position there (a motion blur's shutter opening). In
	// compile_pipeline, the vertex stage's.
	taps:     []Tap,
}

// One more run of what a vertex shader's main does toward gl_Position,
// before main itself, as a copy of the statements the position depends on
// (in main, the functions it reaches and the globals' initializers): in tap
// N each uniform reads at its offset plus (N + 1) * `Module.block_stride`,
// the attributes listed read their twins, gl_Position goes to `position`,
// and the shader's own outputs and globals are private to the copy.
Tap :: struct {
	// An `out vec4` (declared in the prelude) the run's gl_Position goes to.
	position:   string,
	// Attributes the run reads from twins, `in` variables of the same type:
	// {"position", "o3dOpenPosition"}. A twin the copy never reads is no
	// input.
	attributes: [][2]string,
}

// A user uniform in the block (after the built-ins at OFFSET_USER).
Uniform_Info :: struct {
	name:   string,
	type:   Type,
	// Byte offset in the uniform block, std140.
	offset: int,
	// Bytes the uniform occupies (arrays: count * stride).
	size:   int,
	// Array element count, 0 for non-arrays.
	count:  int,
	// Byte stride between array elements (16 under std140).
	stride: int,
}

// A stage input or output: what the vertex shader writes at `location`,
// or the colour target `location`. The layer profiles have one of each
// (uv and the colour, at 0).
Stage_Var_Info :: struct {
	name:     string,
	type:     Type,
	location: int,
	// Inputs: not interpolated.
	flat:     bool,
	// Varyings (vertex outputs, fragment inputs): the float offset in the
	// CPU interpreter's record of `Module.varying_floats` per vertex
	// (run_vertex writes it, run_fragment reads it, interpolated).
	offset:   int,
}

Module :: struct {
	// SPIR-V 1.0 fragment module; entry point "main".
	spirv:         []u32,
	// Metal source; entry point MSL_ENTRY.
	msl:           string,
	// The uniforms of the block's first environment.
	uniforms:      []Uniform_Info,
	// Size of the uniform block in bytes (a multiple of 16): (1 + taps)
	// environments of `block_stride` bytes, each laid out as `uniforms`.
	block_size:    int,
	block_stride:  int,
	taps:          int,
	// The inputs (INPUTS rows) the shader reads.
	inputs_used:   Inputs,
	// Channels the shader samples; bind at least `num_samplers` samplers.
	channels_used: bit_set[0 ..< CHANNEL_COUNT],
	num_samplers:  int,
	// The name each channel binds by: a `uniform shader` child, "iChannelN"
	// or a `uniform sampler2D` ("" when unused).
	channel_names: [CHANNEL_COUNT]string,
	// Where the source first names each channel (for diagnostics).
	channel_pos:   [CHANNEL_COUNT]Pos,
	profile:       Profile,
	stage:         Stage,
	// Floats per vertex in the varyings' record (Stage_Var_Info.offset).
	varying_floats: int,
	stage_inputs:  []Stage_Var_Info,
	stage_outputs: []Stage_Var_Info,
	program:       Program,
	arena:         virtual.Arena,
}

SPIRV_ENTRY :: "main"
MSL_ENTRY :: "oksl_main"

// Compiles `source` as `options.profile` (SkSL by default). On failure
// `errors` lists every problem found (with line:column) and the module is
// empty. `errors` and the module live until `destroy`.
compile :: proc(source: string, options := Options{}) -> (m: ^Module, errors: []Error, ok: bool) {
	m = module_new() or_return
	context.allocator = virtual.arena_allocator(&m.arena)
	u, toks, front_ok := module_front(source, options, .Main, .Prelude)
	if !front_ok {
		return m, u.errors[:], false
	}
	ok = module_build(m, u, toks, options)
	return m, u.errors[:], ok
}

// A unit for `options` and the preprocessed tokens of its prelude and
// source (positioned in sources `src` and `prelude_src`), in the context's
// allocator (the module's arena: names in the module point into the
// sources).
@(private)
module_front :: proc(source: string, options: Options, src, prelude_src: Source) -> (u: ^Unit, toks: []Token, ok: bool) {
	u = new(Unit)
	unit_init(u, options.profile, context.allocator, buffer_pass = options.unclamped, stage = options.stage, taps = options.taps)
	if options.stage not_in PROFILE_RULES[options.profile].stages {
		append(&u.errors, Error{pos = {line = 1, col = 1, src = src}, code = .Stage_Unsupported, a = PROFILE_NAMES[options.profile], b = STAGE_NAMES[options.stage]})
		return u, nil, false
	}
	if len(options.taps) > 0 && options.stage != .Vertex {
		append(&u.errors, Error{pos = {line = 1, col = 1, src = src}, code = .Tap_Stage, a = STAGE_NAMES[options.stage]})
		return u, nil, false
	}
	pp: Preprocessor
	preprocess_init(&pp, options.profile, &u.errors, context.allocator)
	defer preprocess_destroy(&pp)
	if options.prelude != "" {
		preprocess_unit(&pp, tokenize_source(strings.clone(options.prelude), prelude_src))
	}
	main_toks := tokenize_source(strings.clone(source), src)
	preprocess_unit(&pp, main_toks)
	toks = preprocess_end(&pp, main_toks[len(main_toks) - 1].pos)
	return u, toks, tokens_valid(toks, &u.errors)
}

@(private)
module_new :: proc(allocator := context.allocator) -> (m: ^Module, ok: bool) {
	m = new(Module, allocator)
	if virtual.arena_init_growing(&m.arena) != nil {
		free(m)
		return nil, false
	}
	return m, true
}

// Tokens of `src`, positioned in source `which`.
@(private)
tokenize_source :: proc(src: string, which: Source) -> []Token {
	toks := tokenize(src)
	for &t in toks {
		t.pos.src = which
	}
	return toks[:]
}

// Reports malformed tokens; false if there were errors already or now.
@(private)
tokens_valid :: proc(toks: []Token, errors: ^[dynamic]Error) -> bool {
	for t in toks {
		if t.kind == .Invalid {
			append(errors, Error{pos = t.pos, code = .Unexpected_Character, a = t.text})
		}
	}
	return len(errors) == 0
}

// Parses preprocessed tokens into `u` (which may hold syntax already),
// checks it and fills `m` with its reflection and code.
@(private)
module_build :: proc(m: ^Module, u: ^Unit, toks: []Token, options: Options) -> bool {
	checked := module_check(u, toks) or_return
	module_emit(m, u, &checked, options)
	return true
}

// Parses and checks a unit.
@(private)
module_check :: proc(u: ^Unit, toks: []Token) -> (checked: Checked, ok: bool) {
	context.allocator = u.allocator
	parse(u, toks)
	if len(u.errors) > 0 {
		return {}, false
	}
	return check(u)
}

// Fills a module from a checked unit: reflection and every output.
@(private)
module_emit :: proc(m: ^Module, u: ^Unit, c: ^Checked, options: Options) {
	context.allocator = u.allocator

	m.block_size = c.block_size
	m.block_stride = c.block_stride
	m.taps = c.taps
	m.inputs_used = c.inputs_used
	// The taps' copies follow the first environment's uniforms.
	uniforms := make([dynamic]Uniform_Info)
	for sid in c.uniforms {
		s := u.syms[sid]
		if s.index >= c.block_stride {
			break
		}
		t := ty(u, s.type)
		_, size := std140(t)
		info := Uniform_Info{name = s.name, type = t, offset = s.index, size = size}
		if t.array > 0 {
			info.count = int(t.array)
			info.stride = std140_array_stride(t)
		}
		append(&uniforms, info)
	}
	m.uniforms = uniforms[:]
	for ch, i in c.channels {
		if ch == 0 {
			continue
		}
		m.channels_used += {i}
		m.num_samplers = i + 1
		m.channel_names[i] = strings.clone(u.syms[ch].name)
		// Where the source first names the channel.
		m.channel_pos[i] = u.syms[ch].pos
		for e in u.exprs {
			if e.kind == .Ident && e.sym == ch {
				m.channel_pos[i] = e.pos
				break
			}
		}
	}
	inputs := make([dynamic]Stage_Var_Info)
	outputs := make([dynamic]Stage_Var_Info)
	for row in c.interface {
		if row.kind == .Stage_In || row.kind == .Stage_Out {
			s := u.syms[row.sym]
			info := Stage_Var_Info{strings.clone(s.name), ty(u, s.type), int(row.binding), row.flat, int(row.offset)}
			append(&inputs if row.kind == .Stage_In else &outputs, info)
		}
	}
	m.stage_inputs = inputs[:]
	m.stage_outputs = outputs[:]
	if !options.no_spirv {
		m.spirv = emit_spirv(u, c)
	}
	if !options.no_msl {
		m.msl = emit_msl(u, c)
	}
	if !options.no_cpu {
		m.program = build_program(u, c)
	}
	m.profile = u.profile
	m.stage = u.stage
	m.varying_floats = c.varying_floats
}

destroy :: proc(m: ^Module) {
	if m == nil {
		return
	}
	virtual.arena_destroy(&m.arena)
	free(m)
}

// Finds a user uniform by name.
find_uniform :: proc(m: ^Module, name: string) -> (Uniform_Info, bool) {
	for u in m.uniforms {
		if u.name == name {
			return u, true
		}
	}
	return {}, false
}

// Writes float components of a user uniform (element `index` for arrays).
// Matrices take their columns in order; ints and bools are converted.
set_uniform :: proc(m: ^Module, block: []u8, name: string, values: []f32, index := 0) -> bool {
	u, ok := find_uniform(m, name)
	if !ok {
		return false
	}
	write_values(block, u.offset + index * u.stride, elem_type(u.type), values)
	return true
}

// The vertex shader hosts draw oksl layers with (shaders/layer.slang): a
// quad in target pixels (location 0, vec2) with uvs (location 1, vec2) and
// a vertex uniform buffer holding the target size (vec4, xy used).
LAYER_VERTEX_SPIRV :: #load("shaders/layer.vert.spv")
LAYER_VERTEX_SPIRV_ENTRY :: "main"
LAYER_VERTEX_MSL :: #load("shaders/layer.vert.msl", string)
LAYER_VERTEX_MSL_ENTRY :: "vs_main"
