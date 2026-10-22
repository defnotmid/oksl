package oksl

import "core:fmt"
import "core:mem/virtual"
import "core:strings"

// Shadertoy multipass: a program of an optional Common source, up to four
// buffer passes (Buffer A-D) and the Image pass, compiled into a flat pass
// table and a flat edge array.
//
// - Common is one shared unit: preprocessed and parsed once; its macros
//   (a mark of the macro table, `preprocess_mark`) and its syntax (copied
//   flat arrays, `unit_copy_syntax`) start every pass, so its declarations
//   are visible in each pass the way Shadertoy prepends it, and errors in
//   it are reported once, at its own lines.
// - Each pass is one module, checked with Common's declarations and its
//   own. Buffer passes return mainImage's colour as it is (unclamped, with
//   alpha) for the float textures they render into; the Image pass is
//   clamped and opaque, like a single-pass compile.
// - Passes run in table order: the buffers (A..D, those present), then the
//   Image pass, last. An edge says what a pass's channel reads: this
//   frame's output of an earlier pass (`Buffer`), the previous frame's
//   output of itself or of a later pass (`Previous`, Shadertoy's
//   feedback), or one of the host's inputs (`External`). Edges are only
//   made for the channels a pass samples, sorted by pass, then channel.
//
// Positions carry their source (`Pos.src`: Common, Buffer A..D, Image), so
// `format_errors` writes "Buffer A:12:7: ...".

Pass_Kind :: enum u8 {
	Buffer_A,
	Buffer_B,
	Buffer_C,
	Buffer_D,
	Image,
}

BUFFER_COUNT :: 4

// The source a position is in. Main is a single compile's.
Source :: enum u8 {
	Main,
	Common,
	Buffer_A,
	Buffer_B,
	Buffer_C,
	Buffer_D,
	Image,
	// Options.prelude, and a pipeline's stages and their preludes.
	Prelude,
	Vertex,
	Fragment,
	Vertex_Prelude,
	Fragment_Prelude,
}

@(rodata)
SOURCE_NAMES := [Source]string {
	.Main     = "",
	.Common   = "Common",
	.Buffer_A = "Buffer A",
	.Buffer_B = "Buffer B",
	.Buffer_C = "Buffer C",
	.Buffer_D = "Buffer D",
	.Image    = "Image",
	.Prelude  = "Prelude",
	.Vertex   = "Vertex",
	.Fragment = "Fragment",
	.Vertex_Prelude = "Vertex prelude",
	.Fragment_Prelude = "Fragment prelude",
}

pass_source :: #force_inline proc "contextless" (k: Pass_Kind) -> Source {
	return Source(int(k) + int(Source.Buffer_A))
}

// What a pass's channel reads, and what an edge says it reads.
Edge_Source :: enum u8 {
	// Unbound (channel references only): reads transparent black.
	None,
	// The host's input `id`.
	External,
	// This frame's output of pass `id`, which runs earlier.
	Buffer,
	// The previous frame's output of pass `id` (any pass, itself included);
	// transparent black on the first frame.
	Previous,
}

// The name Channel_Ref's source had before graphs were dialect-neutral.
Channel_Kind :: Edge_Source

// What a channel of a pass reads. In a Program_Desc, `id` of .Buffer and
// .Previous is a Pass_Kind, and .Buffer reads the previous frame when the
// buffer does not run earlier (Shadertoy's rule); in a Pass_Spec it is a
// pass index.
Channel_Ref :: struct {
	source: Edge_Source,
	id:     u8,
}

Program_Desc :: struct {
	// Shared declarations and macros; may be empty.
	common:   string,
	// Each pass's Shadertoy GLSL; "" for a buffer the program does not
	// have. The Image pass is required.
	sources:  [Pass_Kind]string,
	channels: [Pass_Kind][CHANNEL_COUNT]Channel_Ref,
}

Edge :: struct {
	pass:    u8,
	channel: u8,
	source:  Edge_Source,
	id:      u8,
}

// The most passes a graph has (Shadertoy's five fit).
PASS_MAX :: 8

Pass :: struct {
	// Shadertoy's label (compile_program); .Image in a make_program graph.
	kind:   Pass_Kind,
	name:   string,
	module: ^Module,
	// This pass's edges, into `Pass_Graph.edges`.
	edges:  Range,
}

Pass_Graph :: struct {
	// In running order (compile_program: buffers in A..D order, then the
	// Image pass).
	passes: []Pass,
	edges:  []Edge,
	// The pass whose output is the result.
	output: int,
	// The pass index of each Shadertoy kind, -1 when absent (all -1 in a
	// make_program graph).
	index:  [Pass_Kind]i8,
	arena:  virtual.Arena,
}

// A pass of make_program: a compiled module of any profile, which the
// graph takes, and what each of its channels reads (pass ids are indices
// into the pass list).
Pass_Spec :: struct {
	module:   ^Module,
	name:     string,
	channels: [CHANNEL_COUNT]Channel_Ref,
}

// Builds a pass graph from compiled modules, in running order: a channel
// may read the host's inputs, this frame's output of an earlier pass
// (.Buffer) or the previous frame of any pass (.Previous). `output` is the
// result's pass (-1: the last). The graph owns every module passed, also
// on failure, so destroy_program is the one cleanup.
make_program :: proc(passes: []Pass_Spec, output := -1) -> (g: ^Pass_Graph, errors: []Error, ok: bool) {
	g = new(Pass_Graph)
	if virtual.arena_init_growing(&g.arena) != nil {
		for p in passes {
			destroy(p.module)
		}
		free(g)
		return nil, nil, false
	}
	context.allocator = virtual.arena_allocator(&g.arena)
	errs := make([dynamic]Error)
	g.passes = make([]Pass, len(passes))
	for p, i in passes {
		g.passes[i] = {kind = .Image, name = strings.clone(p.name), module = p.module}
	}
	for &i in g.index {
		i = -1
	}
	g.output = output if output >= 0 else len(passes) - 1
	switch {
	case len(passes) == 0 || len(passes) > PASS_MAX:
		append(&errs, Error{pos = {line = 1, col = 1}, code = .Pass_Count, n = PASS_MAX, m = i64(len(passes))})
	case g.output >= len(passes):
		append(&errs, Error{pos = {line = 1, col = 1}, code = .Pass_Output, n = i64(g.output), m = i64(len(passes))})
	}
	refs := make([][CHANNEL_COUNT]Channel_Ref, len(passes))
	for p, pi in passes {
		refs[pi] = p.channels
		for ref, ch in p.channels {
			code := Error_Code.None
			switch {
			case (ref.source == .Buffer || ref.source == .Previous) && int(ref.id) >= len(passes):
				code = .Pass_Missing
			case ref.source == .Buffer && int(ref.id) >= pi:
				code = .Pass_Not_Before
			}
			if code != .None {
				// An unnamed pass by its index.
				name := p.name if p.name != "" else fmt.tprintf("pass %d", pi)
				append(&errs, Error{pos = {line = 1, col = 1}, code = code, name = strings.clone(name), n = i64(ch), m = i64(ref.id)})
			}
		}
	}
	if len(errs) > 0 {
		return g, errs[:], false
	}
	program_edges(g, refs)
	return g, errs[:], true
}

// The edges of the graph's passes from what each channel reads (pass ids
// as indices), for the channels each module samples.
@(private)
program_edges :: proc(g: ^Pass_Graph, refs: [][CHANNEL_COUNT]Channel_Ref) {
	edges := make([dynamic]Edge)
	for &p, pi in g.passes {
		start := u32(len(edges))
		for ref, ch in refs[pi] {
			if ref.source != .None && ch in p.module.channels_used {
				append(&edges, Edge{u8(pi), u8(ch), ref.source, ref.id})
			}
		}
		p.edges = {start, u32(len(edges)) - start}
	}
	g.edges = edges[:]
}

// Compiles a Shadertoy program (GLSL dialect). On failure `errors` lists
// every problem, positioned in its source, and the graph holds no passes.
// The graph, its modules and `errors` live until `destroy_program`.
compile_program :: proc(desc: Program_Desc, options := Options{}) -> (g: ^Pass_Graph, errors: []Error, ok: bool) {
	g = new(Pass_Graph)
	outer := context.allocator
	if virtual.arena_init_growing(&g.arena) != nil {
		free(g)
		return nil, nil, false
	}
	allocator := virtual.arena_allocator(&g.arena)
	context.allocator = allocator
	o := options
	o.profile = .Shadertoy
	errs := make([dynamic]Error)
	if desc.sources[.Image] == "" {
		append(&errs, Error{pos = {line = 1, col = 1, src = .Image}, code = .No_Image_Pass})
		return g, errs[:], false
	}
	n := 0
	for kind in Pass_Kind {
		g.index[kind] = -1
		if desc.sources[kind] != "" {
			g.index[kind] = i8(n)
			n += 1
		}
	}

	// Common, once. Names in the modules point into the sources, which
	// live in the graph's arena.
	pp: Preprocessor
	preprocess_init(&pp, .Shadertoy, &errs, allocator)
	defer preprocess_destroy(&pp)
	common := new(Unit)
	unit_init(common, .Shadertoy, allocator)
	if desc.common != "" {
		toks := tokenize_source(strings.clone(desc.common), .Common)
		preprocess_unit(&pp, toks)
		common_toks := preprocess_end(&pp, toks[len(toks) - 1].pos)
		if tokens_valid(common_toks, &errs) {
			parse(common, common_toks)
			append(&errs, ..common.errors[:])
		}
		if len(errs) > 0 {
			return g, errs[:], false
		}
	}
	mark := preprocess_mark(&pp)

	passes := make([]Pass, n)
	refs := make([][CHANNEL_COUNT]Channel_Ref, n)
	for kind in Pass_Kind {
		pi := int(g.index[kind])
		if pi < 0 {
			continue
		}
		passes[pi].kind = kind
		passes[pi].name = SOURCE_NAMES[pass_source(kind)]
		preprocess_reset(&pp, mark)
		toks := tokenize_source(strings.clone(desc.sources[kind]), pass_source(kind))
		before := len(errs)
		preprocess_unit(&pp, toks)
		pass_toks := preprocess_end(&pp, toks[len(toks) - 1].pos)
		tokens_valid(pass_toks, &errs)
		if len(errs) > before {
			continue
		}
		m, made := module_new(outer)
		if !made {
			return g, errs[:], false
		}
		passes[pi].module = m
		u := new(Unit, virtual.arena_allocator(&m.arena))
		// Buffers keep mainImage's colour as it is, for float targets.
		unit_init(u, .Shadertoy, virtual.arena_allocator(&m.arena), buffer_pass = kind != .Image)
		unit_copy_syntax(u, common)
		built := module_build(m, u, pass_toks, o)
		// Common's errors show up in every pass; keep the first. The text
		// moves to the graph's arena, as the module may go.
		next: for e in u.errors {
			if e.pos.src == .Common {
				for seen in errs {
					if seen.pos == e.pos && seen.code == e.code && seen.name == e.name {
						continue next
					}
				}
			}
			kept := e
			if kept.pos.src == .Main {
				// Positionless (no mainImage): the pass's.
				kept.pos.src = pass_source(kind)
			}
			kept.name, kept.a, kept.b, kept.c = strings.clone(e.name), strings.clone(e.a), strings.clone(e.b), strings.clone(e.c)
			append(&errs, kept)
		}
		if !built {
			continue
		}
		// Pass indices for what the channels read; a buffer that does not
		// run earlier is read as its previous frame.
		for ch in 0 ..< CHANNEL_COUNT {
			if ch not_in m.channels_used {
				continue
			}
			ref := desc.channels[kind][ch]
			switch ref.source {
			case .None:
			case .External:
				refs[pi][ch] = ref
			case .Buffer, .Previous:
				target: i8 = -1
				what := "a pass that is not a buffer"
				if ref.id < BUFFER_COUNT {
					target = g.index[Pass_Kind(ref.id)]
					what = SOURCE_NAMES[pass_source(Pass_Kind(ref.id))]
				}
				if target < 0 {
					append(&errs, Error{pos = m.channel_pos[ch], code = .Channel_No_Buffer, name = strings.clone(m.channel_names[ch]), a = what})
					continue
				}
				src := Edge_Source.Buffer if ref.source == .Buffer && int(target) < pi else .Previous
				refs[pi][ch] = {src, u8(target)}
			}
		}
	}
	if len(errs) > 0 {
		context.allocator = outer
		for p in passes {
			destroy(p.module)
		}
		return g, errs[:], false
	}
	g.passes = passes
	g.output = n - 1
	program_edges(g, refs)
	return g, errs[:], true
}

destroy_program :: proc(g: ^Pass_Graph) {
	if g == nil {
		return
	}
	for p in g.passes {
		destroy(p.module)
	}
	virtual.arena_destroy(&g.arena)
	free(g)
}
