package oksl

import "core:mem/virtual"

// A vertex and a fragment shader compiled together (GLSL ES 3.0, each with
// an optional prelude), so what passes between them agrees: varyings are
// matched by name, as GLSL ES 3.00 links them, and each name gets one
// location in both stages. Each stage is an ordinary Module (its own
// uniform block, reflection and outputs).
//
//     p, errors, ok := oksl.compile_pipeline({vertex = vs, fragment = fs, vertex_prelude = vp, fragment_prelude = fp})
//     defer oksl.destroy_pipeline(p)
//     // p.vertex.spirv / .msl at set 0 (samplers) and 1 (uniforms),
//     // p.fragment's at sets 2 and 3; p.vertex.stage_inputs are the
//     // attributes it reads.

Pipeline_Desc :: struct {
	vertex, fragment:                 string,
	// Declarations ahead of each stage's source (Options.prelude).
	vertex_prelude, fragment_prelude: string,
}

Pipeline :: struct {
	vertex, fragment: ^Module,
	errors:           [dynamic]Error,
}

// Compiles and links a vertex and a fragment shader. Errors name their
// source ("Fragment:12:7: ...", "Vertex prelude:3:1: ..."); a fragment
// input the vertex shader does not write, or writes with another type or
// interpolation, is an error. Everything lives until `destroy_pipeline`.
compile_pipeline :: proc(desc: Pipeline_Desc, options := Options{}) -> (p: ^Pipeline, errors: []Error, ok: bool) {
	p = new(Pipeline)
	p.errors = make([dynamic]Error)
	Stage_Source :: struct {
		stage:        Stage,
		source:       string,
		prelude:      string,
		src, pre_src: Source,
	}
	stages := [2]Stage_Source {
		{.Vertex, desc.vertex, desc.vertex_prelude, .Vertex, .Vertex_Prelude},
		{.Fragment, desc.fragment, desc.fragment_prelude, .Fragment, .Fragment_Prelude},
	}
	units: [2]^Unit
	checked: [2]Checked
	modules: [2]^Module
	ok = true
	for st, i in stages {
		m, made := module_new()
		if !made {
			return p, p.errors[:], false
		}
		modules[i] = m
		o := options
		o.profile, o.stage, o.prelude = .Glsl, st.stage, st.prelude
		{
			context.allocator = virtual.arena_allocator(&m.arena)
			u, toks, front_ok := module_front(st.source, o, st.src, st.pre_src)
			units[i] = u
			stage_ok := front_ok
			if stage_ok {
				checked[i], stage_ok = module_check(u, toks)
			}
			ok &&= stage_ok
		}
		for e in units[i].errors {
			kept := e
			if kept.pos.src == .Main {
				// Positionless (no main): the stage's.
				kept.pos.src = st.src
			}
			append(&p.errors, kept)
		}
	}
	p.vertex, p.fragment = modules[0], modules[1]
	if !ok {
		return p, p.errors[:], false
	}
	link_stages(units[0], &checked[0], units[1], &checked[1], &p.errors)
	if len(p.errors) > 0 {
		return p, p.errors[:], false
	}
	for st, i in stages {
		o := options
		o.profile, o.stage = .Glsl, st.stage
		module_emit(modules[i], units[i], &checked[i], o)
	}
	return p, p.errors[:], true
}

destroy_pipeline :: proc(p: ^Pipeline) {
	if p == nil {
		return
	}
	destroy(p.vertex)
	destroy(p.fragment)
	delete(p.errors)
	free(p)
}

// Gives each fragment input the location of the vertex output of the same
// name (the vertex shader's own: explicit, or the next free one in its
// declaration order).
@(private)
link_stages :: proc(vu: ^Unit, vc: ^Checked, fu: ^Unit, fc: ^Checked, errors: ^[dynamic]Error) {
	report :: proc(errors: ^[dynamic]Error, s: Symbol, code: Error_Code, a := "", b := "", n: i64 = 0, m: i64 = 0) {
		append(errors, Error{pos = s.pos, code = code, name = s.name, a = a, b = b, n = n, m = m})
	}
	next: for &row in fc.interface {
		if row.kind != .Stage_In {
			continue
		}
		fs := fu.syms[row.sym]
		for out in vc.interface {
			vs := vu.syms[out.sym]
			if out.kind != .Stage_Out || vs.name != fs.name {
				continue
			}
			ft, vt := ty(fu, fs.type), ty(vu, vs.type)
			explicit := fu.vars[fs.decl].location
			switch {
			case ft != vt:
				report(errors, fs, .Varying_Type, a = type_name(vu, vt), b = type_name(fu, ft))
			case out.flat != row.flat:
				report(errors, fs, .Varying_Flat)
			case explicit >= 0 && explicit != out.binding:
				report(errors, fs, .Varying_Location, n = i64(explicit), m = i64(out.binding))
			}
			row.binding, row.offset = out.binding, out.offset
			fu.syms[row.sym].index = int(out.binding)
			continue next
		}
		report(errors, fs, .Varying_Unwritten)
	}
	// The fragment reads the vertex shader's record.
	fc.varying_floats = vc.varying_floats
}

// Where uniform `name` is in the module's block, and its type: for hosts
// that look the offsets up once and write the block themselves. Not found
// (undeclared, or unused in a GLSL shader): ok is false.
uniform_offset :: proc(m: ^Module, name: string) -> (offset: int, type: Type, ok: bool) {
	info := find_uniform(m, name) or_return
	return info.offset, info.type, true
}
