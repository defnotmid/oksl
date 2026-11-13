package oksl

import "core:fmt"
import "core:strings"

// Metal Shading Language emission for SDL_GPU on macOS. The fragment entry
// MSL_ENTRY declares what Checked.interface lists: stage inputs at location
// N as [[user(locnN)]] (layer.vert writes uv as locn0), gl_FragCoord
// as [[position]], outputs as [[color(N)]] of the returned Oksl_Out, the
// uniform block at [[buffer(0)]] and channel n at [[texture(n)]] /
// [[sampler(n)]]. User names get an `ok_` prefix so they never collide with
// Metal names; globals (and the stage variables, in_* and out_*) live in a
// struct passed to every function together with the uniforms and channels.

@(private)
Msl :: struct {
	u:       ^Unit,
	c:       ^Checked,
	b:       strings.Builder,
	helpers: strings.Builder,
	// The current function's temporaries, declared ahead of its body.
	fn_vars: strings.Builder,
	// Constructors that need a helper function, by signature.
	ctor_helpers: map[string]string,
	indent:  int,
	temp_id: int,
	// Extra parameters/arguments every user function takes.
	params:  string,
	args:    string,
	// The sampler2DArray helpers written to `helpers`.
	array_helpers: bit_set[Builtin],
}

// A sampler's Metal texture type.
@(private)
msl_texture_type :: proc "contextless" (t: Type) -> string {
	return "texture2d_array<float>" if t == T_SAMPLER_ARRAY else "texture2d<float>"
}

// A storage image's Metal texture type.
@(private)
msl_image_type :: proc(t: Type) -> string {
	acc := "access::read" if image_readonly(t) else ("access::write" if image_writeonly(t) else "access::read_write")
	return fmt.aprintf("texture2d<%s, %s>", msl_scalar(image_texel_base(t)), acc)
}

// A sampler2DArray sample (b: Texture_Lod, Texture or Texture_Grad) as a
// helper taking the coordinate once: Metal takes the layer apart, as a
// uint, which GPUs otherwise round to nearest even and clamp.
@(private)
msl_array_helper :: proc(m: ^Msl, b: Builtin) -> string {
	if b not_in m.array_helpers {
		if m.array_helpers == {} {
			strings.write_string(&m.helpers, "inline uint oksl_layer(texture2d_array<float> t, float z) { return uint(clamp(rint(z), 0.0f, float(t.get_array_size()) - 1.0f)); }\n")
		}
		m.array_helpers += {b}
		#partial switch b {
		case .Texture_Lod:
			strings.write_string(&m.helpers, "inline float4 oksl_sample_level(texture2d_array<float> t, sampler s, float3 c, float l) { return t.sample(s, c.xy, oksl_layer(t, c.z), level(l)); }\n")
		case .Texture:
			strings.write_string(&m.helpers, "inline float4 oksl_sample_bias(texture2d_array<float> t, sampler s, float3 c, float b) { return t.sample(s, c.xy, oksl_layer(t, c.z), bias(b)); }\n")
		case .Texture_Grad:
			strings.write_string(&m.helpers, "inline float4 oksl_sample_grad(texture2d_array<float> t, sampler s, float3 c, float2 dx, float2 dy) { return t.sample(s, c.xy, oksl_layer(t, c.z), gradient2d(dx, dy)); }\n")
		}
	}
	#partial switch b {
	case .Texture_Lod:
		return "oksl_sample_level"
	case .Texture:
		return "oksl_sample_bias"
	}
	return "oksl_sample_grad"
}

emit_msl :: proc(u: ^Unit, c: ^Checked) -> string {
	m := Msl {
		u            = u,
		c            = c,
		b            = strings.builder_make(),
		helpers      = strings.builder_make(),
		fn_vars      = strings.builder_make(),
		ctor_helpers = make(map[string]string),
	}
	// Shared parameters: uniforms, globals and the channels in use.
	pb := strings.builder_make()
	ab := strings.builder_make()
	// Without a uniform block (a GLSL shader using no uniforms) there is no
	// `u` to pass.
	has_block := false
	for row in c.interface {
		has_block ||= row.kind == .Uniform_Block
	}
	strings.write_string(&pb, "constant Oksl_Uniforms& u, thread Oksl_Globals& g" if has_block else "thread Oksl_Globals& g")
	strings.write_string(&ab, "u, g" if has_block else "g")
	for row in c.interface {
		#partial switch row.kind {
		case .Channel:
			n := u.syms[row.sym].name
			fmt.sbprintf(&pb, ", %s ok_%s_tex, sampler ok_%s_smp", msl_texture_type(sym_type(u, row.sym)), n, n)
			fmt.sbprintf(&ab, ", ok_%s_tex, ok_%s_smp", n, n)
		case .Storage_Image:
			n := u.syms[row.sym].name
			fmt.sbprintf(&pb, ", %s ok_%s_img", msl_image_type(sym_type(u, row.sym)), n)
			fmt.sbprintf(&ab, ", ok_%s_img", n)
		case .Buffer:
			sym := u.syms[row.sym]
			bd := u.buffer_decls[c.buffer_decls[sym.index]]
			qual := "const device uint*" if bd.readonly else "device uint*"
			fmt.sbprintf(&pb, ", %s oksl_buf_%s", qual, sym.name)
			fmt.sbprintf(&ab, ", oksl_buf_%s", sym.name)
		}
	}
	m.params = strings.to_string(pb)
	m.args = strings.to_string(ab)
	if len(c.images) > 0 {
		strings.write_string(&m.helpers, "template<typename T> inline void oksl_fence(texture2d<T, access::read_write> t) { t.fence(); }\n")
		strings.write_string(&m.helpers, "template<typename T> inline void oksl_fence(texture2d<T, access::write> t) {}\n")
	}

	body := strings.builder_make()
	m.b = body
	for t in u.structs {
		w(&m, "struct ok_%s {{\n", t.name)
		for f in u.fields[t.fields.start:][:t.fields.count] {
			w(&m, "\t%s ok_%s;\n", msl_type(&m, ty(u, f.type)), f.name)
		}
		w(&m, "};\n\n")
	}
	w(&m, "struct Oksl_Globals {{\n")
	for g in c.globals {
		w(&m, "\t%s ok_%s;\n", msl_type(&m, sym_type(u, g)), u.syms[g].name)
	}
	for row in c.interface {
		#partial switch row.kind {
		case .Stage_In, .Frag_Coord, .Front_Facing, .Vertex_ID,
		     .Global_Invocation_ID, .Local_Invocation_ID, .Work_Group_ID, .Num_Work_Groups, .Local_Invocation_Index:
			w(&m, "\t%s in_%s;\n", msl_type(&m, sym_type(u, row.sym)), u.syms[row.sym].name)
		case .Stage_Out, .Position, .Sample_Mask:
			w(&m, "\t%s out_%s;\n", msl_type(&m, sym_type(u, row.sym)), u.syms[row.sym].name)
		}
	}
	w(&m, "\tint _unused;\n};\n\n")
	for b, bi in c.buffers {
		if !u.buffer_decls[c.buffer_decls[bi]].writeonly {
			emit_buffer_load(&m, b)
		}
	}
	for f in c.funcs {
		if f != c.entry {
			emit_msl_function(&m, f)
		}
	}
	emit_msl_entry(&m)

	out := strings.builder_make()
	strings.write_string(&out, MSL_PRELUDE)
	emit_uniform_struct(&m, &out)
	strings.write_string(&out, strings.to_string(m.helpers))
	strings.write_string(&out, strings.to_string(m.b))
	return strings.to_string(out)
}

@(private)
w :: proc(m: ^Msl, format: string, args: ..any) {
	fmt.sbprintf(&m.b, format, ..args)
}

@(private)
tabs :: proc(m: ^Msl) {
	for _ in 0 ..< m.indent {
		strings.write_byte(&m.b, '\t')
	}
}

@(private)
MSL_PRELUDE :: `#include <metal_stdlib>
using namespace metal;

// GLSL semantics: x - y * floor(x / y) (Metal's fmod truncates).
template<typename T, typename U> inline T oksl_mod(T x, U y) { return x - y * floor(x / y); }

template<int C, int R> inline matrix<float, C, R> oksl_comp_mult(matrix<float, C, R> a, matrix<float, C, R> b) {
	matrix<float, C, R> m;
	for (int i = 0; i < C; i++) m[i] = a[i] * b[i];
	return m;
}
template<int C, int R> inline matrix<float, C, R> oksl_comp_div(matrix<float, C, R> a, matrix<float, C, R> b) {
	matrix<float, C, R> m;
	for (int i = 0; i < C; i++) m[i] = a[i] / b[i];
	return m;
}
template<int C, int R> inline matrix<float, C, R> oksl_add_s(matrix<float, C, R> a, float s) {
	matrix<float, C, R> m;
	for (int i = 0; i < C; i++) m[i] = a[i] + s;
	return m;
}
template<int C, int R> inline matrix<float, C, R> oksl_sub_s(matrix<float, C, R> a, float s) {
	matrix<float, C, R> m;
	for (int i = 0; i < C; i++) m[i] = a[i] - s;
	return m;
}
template<int C, int R> inline matrix<float, C, R> oksl_s_sub(float s, matrix<float, C, R> a) {
	matrix<float, C, R> m;
	for (int i = 0; i < C; i++) m[i] = s - a[i];
	return m;
}
template<int C, int R> inline matrix<float, C, R> oksl_div_s(matrix<float, C, R> a, float s) {
	matrix<float, C, R> m;
	for (int i = 0; i < C; i++) m[i] = a[i] / s;
	return m;
}
template<int C, int R> inline matrix<float, C, R> oksl_s_div(float s, matrix<float, C, R> a) {
	matrix<float, C, R> m;
	for (int i = 0; i < C; i++) m[i] = s / a[i];
	return m;
}
template<int C, int R> inline bool oksl_meq(matrix<float, C, R> a, matrix<float, C, R> b) {
	for (int i = 0; i < C; i++) if (any(a[i] != b[i])) return false;
	return true;
}
template<int C, int R> inline matrix<float, C, R> oksl_outer(vec<float, R> c, vec<float, C> r) {
	matrix<float, C, R> m;
	for (int i = 0; i < C; i++) m[i] = c * r[i];
	return m;
}
template<int C, int R> inline matrix<float, C, R> oksl_diag(float s) {
	matrix<float, C, R> m;
	for (int i = 0; i < C; i++) for (int j = 0; j < R; j++) m[i][j] = i == j ? s : 0.0f;
	return m;
}
template<int C, int R, int C2, int R2> inline matrix<float, C, R> oksl_resize(matrix<float, C2, R2> a) {
	matrix<float, C, R> m;
	for (int i = 0; i < C; i++) for (int j = 0; j < R; j++) m[i][j] = (i < C2 && j < R2) ? a[i][j] : (i == j ? 1.0f : 0.0f);
	return m;
}
inline float2x2 oksl_inverse(float2x2 m) {
	float d = 1.0f / determinant(m);
	return float2x2(float2(m[1][1], -m[0][1]) * d, float2(-m[1][0], m[0][0]) * d);
}
inline float3x3 oksl_inverse(float3x3 m) {
	float3 a = m[0], b = m[1], c = m[2];
	float3 r0 = cross(b, c), r1 = cross(c, a), r2 = cross(a, b);
	float d = 1.0f / dot(a, r0);
	return transpose(float3x3(r0, r1, r2)) * d;
}
inline float4x4 oksl_inverse(float4x4 m) {
	float4x4 r;
	float d = 0.0f;
	for (int c = 0; c < 4; c++) for (int k = 0; k < 4; k++) {
		float s[9]; int n = 0;
		for (int cc = 0; cc < 4; cc++) { if (cc == k) continue; for (int rr = 0; rr < 4; rr++) { if (rr == c) continue; s[n++] = m[cc][rr]; } }
		float det3 = s[0] * (s[4] * s[8] - s[7] * s[5]) - s[3] * (s[1] * s[8] - s[7] * s[2]) + s[6] * (s[1] * s[5] - s[4] * s[2]);
		r[c][k] = ((c + k) % 2 == 0 ? 1.0f : -1.0f) * det3;
	}
	for (int i = 0; i < 4; i++) d += m[0][i] * r[i][0];
	return r * (1.0f / d);
}

`

// ---- types ------------------------------------------------------------------------------

@(private)
msl_scalar :: proc(b: Base) -> string {
	#partial switch b {
	case .Bool:
		return "bool"
	case .Int:
		return "int"
	case .Uint:
		return "uint"
	case .Void:
		return "void"
	}
	return "float"
}

@(private)
msl_type :: proc(m: ^Msl, t: Type) -> string {
	if t.array > 0 {
		return fmt.aprintf("array<%s, %d>", msl_type(m, elem_type(t)), t.array)
	}
	if t.base == .Struct {
		return fmt.aprintf("ok_%s", m.u.structs[t.strukt].name)
	}
	if is_matrix(t) {
		return fmt.aprintf("float%dx%d", t.cols, t.rows)
	}
	if t.rows > 1 {
		return fmt.aprintf("%s%d", msl_scalar(t.base), t.rows)
	}
	return msl_scalar(t.base)
}

// ---- uniform block ----------------------------------------------------------------------

@(private)
Msl_Member :: struct {
	name:   string,
	type:   Type,
	offset: int,
}

// The uniform struct, padded to the std140 offsets the SPIR-V uses: vec3s
// are packed_float3, matrices and array elements are float4 rows.
@(private)
emit_uniform_struct :: proc(m: ^Msl, out: ^strings.Builder) {
	members := make([dynamic]Msl_Member, context.temp_allocator)
	if m.c.header {
		for b in builtin_members() {
			append(&members, Msl_Member{b.name, b.type, b.offset})
		}
	}
	for s in m.c.uniforms {
		sym := m.u.syms[s]
		append(&members, Msl_Member{fmt.aprintf("ok_%s", sym.name), ty(m.u, sym.type), sym.index})
	}
	strings.write_string(out, "struct Oksl_Uniforms {\n")
	cur := 0
	pad := 0
	for mem in members {
		if mem.offset > cur {
			fmt.sbprintf(out, "\tchar _pad%d[%d];\n", pad, mem.offset - cur)
			pad += 1
		}
		t := mem.type
		sb := msl_scalar(.Uint if t.base == .Bool else t.base)
		_, size := std140(t)
		switch {
		case t.array > 0:
			e := elem_type(t)
			n := int(t.array) * max(int(e.cols), 1)
			fmt.sbprintf(out, "\t%s4 %s[%d];\n", sb, mem.name, n)
		case t.cols > 0:
			fmt.sbprintf(out, "\tfloat4 %s[%d];\n", mem.name, t.cols)
		case t.rows == 3:
			fmt.sbprintf(out, "\tpacked_%s3 %s;\n", sb, mem.name)
		case t.rows > 1:
			fmt.sbprintf(out, "\t%s%d %s;\n", sb, t.rows, mem.name)
		case:
			fmt.sbprintf(out, "\t%s %s;\n", sb, mem.name)
		}
		cur = mem.offset + size
	}
	if cur < m.c.block_size {
		fmt.sbprintf(out, "\tchar _pad%d[%d];\n", pad, m.c.block_size - cur)
	}
	strings.write_string(out, "};\n\n")
}

// The uniform struct member `sym` reads. Built-ins go by their block
// offset: the checker's internal alias ($channel_resolution, used by
// child.eval) names the iChannelResolution member.
@(private)
uniform_member_name :: proc(sym: Symbol) -> string {
	if sym.storage == .Builtin_Uniform {
		return builtin_members()[builtin_member_index(sym.index)].name
	}
	return fmt.aprintf("ok_%s", sym.name)
}

// How to read uniform `sym` as a value of its GLSL type, and element `i`
// of a uniform array (`idx` is the index expression text).
@(private)
uniform_read :: proc(m: ^Msl, sym: Symbol, idx: string = "") -> string {
	t := ty(m.u, sym.type)
	name := uniform_member_name(sym)
	ref := fmt.aprintf("u.%s", name)
	elem := t
	if t.array > 0 {
		elem = elem_type(t)
		if idx == "" {
			// The whole array: one helper per uniform.
			return fmt.aprintf("oksl_array_%s(u)", name)
		}
		if elem.cols > 0 {
			return fmt.aprintf("oksl_load_%s(u, %s)", name, idx)
		}
		ref = fmt.aprintf("u.%s[%s]", name, idx)
		return component_view(ref, elem, true)
	}
	if t.cols > 0 {
		cols := make([dynamic]string, context.temp_allocator)
		for c in 0 ..< int(t.cols) {
			append(&cols, component_view(fmt.aprintf("%s[%d]", ref, c), column_type(t), true))
		}
		return fmt.aprintf("float%dx%d(%s)", t.cols, t.rows, strings.join(cols[:], ", "))
	}
	return component_view(ref, t, false)
}

// A value of type `t` from a stored member: `wide` members are 4-wide.
@(private)
component_view :: proc(ref: string, t: Type, wide: bool) -> string {
	if t.base == .Bool {
		inner := component_view(ref, with_base(t, .Uint), wide)
		if t.rows == 1 {
			return fmt.aprintf("(%s != 0u)", inner)
		}
		return fmt.aprintf("(%s != uint%d(0u))", inner, t.rows)
	}
	sb := msl_scalar(t.base)
	switch t.rows {
	case 1:
		return fmt.aprintf("%s.x", ref) if wide else ref
	case 2:
		return fmt.aprintf("%s.xy", ref) if wide else ref
	case 3:
		return fmt.aprintf("%s.xyz", ref) if wide else fmt.aprintf("%s3(%s)", sb, ref)
	}
	return ref
}

// Helpers for uniform arrays of matrices and whole uniform arrays.
@(private)
uniform_helpers :: proc(m: ^Msl) {
	syms := make([dynamic]Sym_Id, context.temp_allocator)
	for s, i in m.u.syms {
		if i > 0 && s.index >= 0 && (s.storage == .Uniform || s.storage == .Builtin_Uniform) && ty(m.u, s.type).array > 0 {
			append(&syms, Sym_Id(i))
		}
	}
	done := make(map[string]bool, context.temp_allocator)
	for id in syms {
		sym := m.u.syms[id]
		t := ty(m.u, sym.type)
		e := elem_type(t)
		name := uniform_member_name(sym)
		// Built-ins can have two symbols ($channel_resolution and a
		// declared iChannelResolution).
		if done[name] {
			continue
		}
		done[name] = true
		if e.cols > 0 {
			cols := make([dynamic]string, context.temp_allocator)
			for c in 0 ..< int(e.cols) {
				append(&cols, component_view(fmt.aprintf("u.%s[i * %d + %d]", name, e.cols, c), column_type(e), true))
			}
			fmt.sbprintf(&m.helpers, "inline %s oksl_load_%s(constant Oksl_Uniforms& u, int i) {{ return %s(%s); }}\n", msl_type(m, e), name, msl_type(m, e), strings.join(cols[:], ", "))
		}
		elems := make([dynamic]string, context.temp_allocator)
		for i in 0 ..< int(t.array) {
			append(&elems, uniform_read(m, sym, fmt.tprintf("%d", i)))
		}
		fmt.sbprintf(&m.helpers, "inline %s oksl_array_%s(constant Oksl_Uniforms& u) {{ return %s{{%s}}; }}\n", msl_type(m, t), name, msl_type(m, t), strings.join(elems[:], ", "))
	}
	strings.write_string(&m.helpers, "\n")
}

// ---- functions --------------------------------------------------------------------------

@(private)
emit_msl_function :: proc(m: ^Msl, f: Func_Id) {
	u := m.u
	fn := u.funcs[f]
	ps := fparams(u, f)
	m.temp_id = 0
	strings.builder_reset(&m.fn_vars)
	img_params := 0
	for p in ps {
		if ty(u, p.type).base == .Image {
			img_params += 1
		}
	}
	if img_params > 0 {
		w(m, "template<")
		for k in 0 ..< img_params {
			if k > 0 {
				w(m, ", ")
			}
			w(m, "access oksl_a%d", k)
		}
		w(m, ">\n")
	}
	w(m, "%s ok_%s(", msl_type(m, ty(u, fn.ret)), fn.name)
	img_idx := 0
	for p, i in ps {
		if i > 0 {
			w(m, ", ")
		}
		pt := ty(u, p.type)
		name := p.name if p.name != "" else fmt.tprintf("_unnamed%d", i)
		switch {
		case pt.base == .Sampler:
			pfx := "okp" if msl_shadows_channel(m, name) else "ok"
			w(m, "%s %s_%s_tex, sampler %s_%s_smp", msl_texture_type(pt), pfx, name, pfx, name)
		case pt.base == .Image:
			w(m, "texture2d<%s, oksl_a%d> okp_%s_img", msl_scalar(image_texel_base(pt)), img_idx, name)
			img_idx += 1
		case p.qual != .In:
			w(m, "thread %s& ok_%s", msl_type(m, pt), name)
		case:
			w(m, "%s ok_%s", msl_type(m, pt), name)
		}
	}
	if len(ps) > 0 {
		w(m, ", ")
	}
	w(m, "%s) {{\n", m.params)
	outer := m.b
	m.b = strings.builder_make()
	m.indent = 1
	body := u.func_decls[fn.decl].body
	for st in list(u, body != 0 ? u.stmts[body].body : {}) {
		msl_stmt(m, Stmt_Id(st))
	}
	if ty(u, fn.ret) != T_VOID && !always_returns(u, body) {
		w(m, "\treturn {{}};\n")
	}
	m.indent = 0
	fn_body := strings.to_string(m.b)
	m.b = outer
	strings.write_string(&m.b, strings.to_string(m.fn_vars))
	strings.write_string(&m.b, fn_body)
	w(m, "}\n\n")
}

// Element i of a storage buffer, from its words at their std430 offsets
// (oksl_buffer_<name>(words, i)).
@(private)
emit_buffer_load :: proc(m: ^Msl, b: Sym_Id) {
	u := m.u
	t := sym_type(u, b)
	stride := std430_stride(u, t)
	w(m, "static %s oksl_buffer_%s(const device uint* w, uint i) {{\n", msl_type(m, t), u.syms[b].name)
	w(m, "\tuint b = i * %du;\n\treturn ", stride / 4)
	value :: proc(m: ^Msl, t: Type, word: int) {
		u := m.u
		if t.base == .Struct {
			w(m, "%s{{", msl_type(m, t))
			for f, i in struct_fields(u, t.strukt) {
				ft := ty(u, f.type)
				off := std430_field_offset(u, t.strukt, i32(i))
				w(m, ", " if i > 0 else "")
				value(m, ft, word + off / 4)
			}
			w(m, "}}")
			return
		}
		if t.rows > 1 {
			w(m, "%s(", msl_type(m, t))
		}
		for i in 0 ..< int(t.rows) {
			w(m, ", " if i > 0 else "")
			#partial switch t.base {
			case .Float:
				w(m, "as_type<float>(w[b + %du])", word + i)
			case .Int:
				w(m, "as_type<int>(w[b + %du])", word + i)
			case:
				w(m, "w[b + %du]", word + i)
			}
		}
		if t.rows > 1 {
			w(m, ")")
		}
	}
	value(m, t, 0)
	w(m, ";\n}}\n\n")
}

@(private)
emit_msl_entry :: proc(m: ^Msl) {
	u := m.u
	c := m.c
	vertex := u.stage == .Vertex
	compute := u.stage == .Compute
	has_block := false
	num_samplers := 0
	for ch, i in c.channels {
		if ch != 0 {
			num_samplers = i + 1
		}
	}
	for row in c.interface {
		has_block ||= row.kind == .Uniform_Block
	}
	uniform_helpers(m)
	inputs := 0
	if !compute {
		// Vertex inputs are [[attribute(N)]]; what passes between the stages is
		// [[user(locnN)]], as SPIRV-Cross (and so SDL_shadercross) names it.
		w(m, "struct Oksl_In {{\n")
		for row in c.interface {
			sym := u.syms[row.sym]
			#partial switch row.kind {
			case .Stage_In:
				if vertex {
					w(m, "\t%s in_%s [[attribute(%d)]];\n", msl_type(m, ty(u, sym.type)), sym.name, row.binding)
				} else {
					w(m, "\t%s in_%s [[user(locn%d)%s]];\n", msl_type(m, ty(u, sym.type)), sym.name, row.binding, ", flat" if row.flat else "")
				}
			case .Frag_Coord:
				w(m, "\tfloat4 in_%s [[position]];\n", sym.name)
			case .Front_Facing:
				w(m, "\tbool in_%s [[front_facing]];\n", sym.name)
			case:
				continue
			}
			inputs += 1
		}
		w(m, "}};\n\nstruct Oksl_Out {{\n")
		for row in c.interface {
			sym := u.syms[row.sym]
			#partial switch row.kind {
			case .Stage_Out:
				if vertex {
					w(m, "\t%s out_%s [[user(locn%d)]];\n", msl_type(m, ty(u, sym.type)), sym.name, row.binding)
				} else {
					w(m, "\t%s out_%s [[color(%d)]];\n", msl_type(m, ty(u, sym.type)), sym.name, row.binding)
				}
			case .Position:
				w(m, "\tfloat4 out_%s [[position]];\n", sym.name)
			case .Sample_Mask:
				w(m, "\tuint out_%s [[sample_mask]];\n", sym.name)
			}
		}
		w(m, "}};\n\n")
	}
	stage := STAGE_RULES[u.stage]
	w(m, "%s %s %s(", stage.msl_qualifier, "void" if compute else "Oksl_Out", MSL_ENTRY)
	sep := ""
	if inputs > 0 {
		w(m, "Oksl_In oksl_in [[stage_in]]")
		sep = ", "
	}
	for row in c.interface {
		#partial switch row.kind {
		case .Vertex_ID:
			// Built-ins of a vertex function are its parameters.
			w(m, "%suint oksl_vertex_id [[vertex_id]]", sep)
			sep = ", "
		case .Global_Invocation_ID:
			w(m, "%suint3 oksl_global_id [[thread_position_in_grid]]", sep)
			sep = ", "
		case .Local_Invocation_ID:
			w(m, "%suint3 oksl_local_id [[thread_position_in_threadgroup]]", sep)
			sep = ", "
		case .Work_Group_ID:
			w(m, "%suint3 oksl_group_id [[threadgroup_position_in_grid]]", sep)
			sep = ", "
		case .Num_Work_Groups:
			w(m, "%suint3 oksl_num_groups [[threadgroups_per_grid]]", sep)
			sep = ", "
		case .Local_Invocation_Index:
			w(m, "%suint oksl_local_index [[thread_index_in_threadgroup]]", sep)
			sep = ", "
		case .Uniform_Block:
			w(m, "%sconstant Oksl_Uniforms& u [[buffer(%d)]]", sep, row.binding)
			sep = ", "
		case .Channel:
			n := u.syms[row.sym].name
			w(m, "%s%s ok_%s_tex [[texture(%d)]], sampler ok_%s_smp [[sampler(%d)]]", sep, msl_texture_type(sym_type(u, row.sym)), n, row.binding, n, row.binding)
			sep = ", "
		case .Storage_Image:
			sym := u.syms[row.sym]
			w(m, "%s%s ok_%s_img [[texture(%d)]]", sep, msl_image_type(ty(u, sym.type)), sym.name, num_samplers + sym.index)
			sep = ", "
		case .Buffer:
			// SDL_GPU's storage buffers follow the uniform buffers.
			sym := u.syms[row.sym]
			bd := u.buffer_decls[c.buffer_decls[sym.index]]
			qual := "const device uint*" if bd.readonly else "device uint*"
			w(m, "%s%s oksl_buf_%s [[buffer(%d)]]", sep, qual, sym.name, sym.index + (1 if has_block else 0))
			sep = ", "
		}
	}
	w(m, ") {{\n")
	m.indent = 1
	// Stage variables live with the globals (g.in_*, g.out_*), so every
	// function reaches them; the entry copies them across.
	w(m, "\tOksl_Globals g = {{}};\n")
	for row in c.interface {
		#partial switch row.kind {
		case .Stage_In, .Frag_Coord, .Front_Facing:
			n := u.syms[row.sym].name
			w(m, "\tg.in_%s = oksl_in.in_%s;\n", n, n)
		case .Vertex_ID:
			w(m, "\tg.in_%s = int(oksl_vertex_id);\n", u.syms[row.sym].name)
		case .Global_Invocation_ID:
			w(m, "\tg.in_%s = oksl_global_id;\n", u.syms[row.sym].name)
		case .Local_Invocation_ID:
			w(m, "\tg.in_%s = oksl_local_id;\n", u.syms[row.sym].name)
		case .Work_Group_ID:
			w(m, "\tg.in_%s = oksl_group_id;\n", u.syms[row.sym].name)
		case .Num_Work_Groups:
			w(m, "\tg.in_%s = oksl_num_groups;\n", u.syms[row.sym].name)
		case .Local_Invocation_Index:
			w(m, "\tg.in_%s = oksl_local_index;\n", u.syms[row.sym].name)
		}
	}
	m.temp_id = 0
	strings.builder_reset(&m.fn_vars)
	outer := m.b
	m.b = strings.builder_make()
	for st in list(u, u.stmts[u.func_decls[u.funcs[c.entry].decl].body].body) {
		msl_stmt(m, Stmt_Id(st))
	}
	entry_body := strings.to_string(m.b)
	m.b = outer
	strings.write_string(&m.b, strings.to_string(m.fn_vars))
	strings.write_string(&m.b, entry_body)
	if !compute {
		w(m, "\tOksl_Out oksl_out;\n")
		for row in c.interface {
			#partial switch row.kind {
			case .Stage_Out, .Position:
				n := u.syms[row.sym].name
				w(m, "\toksl_out.out_%s = g.out_%s;\n", n, n)
			case .Sample_Mask:
				n := u.syms[row.sym].name
				w(m, "\toksl_out.out_%s = uint(g.out_%s[0]);\n", n, n)
			}
		}
		w(m, "\treturn oksl_out;\n")
	}
	w(m, "}\n")
	m.indent = 0
}

// ---- statements -------------------------------------------------------------------------

@(private)
msl_stmt :: proc(m: ^Msl, id: Stmt_Id) {
	u := m.u
	if id == 0 {
		return
	}
	st := u.stmts[id]
	switch st.kind {
	case .Empty:
	case .Block:
		tabs(m)
		w(m, "{{\n")
		m.indent += 1
		for b in list(u, st.body) {
			msl_stmt(m, Stmt_Id(b))
		}
		m.indent -= 1
		tabs(m)
		w(m, "}\n")
	case .Decl:
		msl_decl(m, st, false)
	case .Expr:
		tabs(m)
		w(m, "%s;\n", msl_expr(m, st.expr))
	case .If:
		tabs(m)
		w(m, "if (%s)\n", msl_expr(m, st.expr))
		msl_sub(m, st.then_)
		if st.else_ != 0 {
			tabs(m)
			w(m, "else\n")
			msl_sub(m, st.else_)
		}
	case .For:
		tabs(m)
		w(m, "{{\n")
		m.indent += 1
		msl_stmt(m, st.init)
		tabs(m)
		w(m, "for (; %s; %s)\n", msl_expr(m, st.expr) if st.expr != 0 else "", msl_expr(m, st.step) if st.step != 0 else "")
		msl_sub(m, st.then_)
		m.indent -= 1
		tabs(m)
		w(m, "}\n")
	case .While:
		tabs(m)
		w(m, "while (%s)\n", msl_expr(m, st.expr))
		msl_sub(m, st.then_)
	case .Do_While:
		tabs(m)
		w(m, "do\n")
		msl_sub(m, st.then_)
		tabs(m)
		w(m, "while (%s);\n", msl_expr(m, st.expr))
	case .Break:
		tabs(m)
		w(m, "break;\n")
	case .Continue:
		tabs(m)
		w(m, "continue;\n")
	case .Discard:
		tabs(m)
		w(m, "discard_fragment();\n")
	case .Switch:
		tabs(m)
		w(m, "switch (%s) {{\n", msl_expr(m, st.expr))
		for b in list(u, st.body) {
			sub := u.stmts[b]
			#partial switch sub.kind {
			case .Case:
				tabs(m)
				if expr_type(u, sub.expr) == T_UINT {
					w(m, "case %du:\n", u32(sub.value))
				} else if i32(sub.value) == min(i32) {
					w(m, "case (-2147483647 - 1):\n")
				} else {
					w(m, "case %d:\n", i32(sub.value))
				}
			case .Default:
				tabs(m)
				w(m, "default:\n")
			case .Decl:
				m.indent += 1
				msl_decl(m, sub, true)
				m.indent -= 1
			case:
				m.indent += 1
				msl_stmt(m, Stmt_Id(b))
				m.indent -= 1
			}
		}
		tabs(m)
		w(m, "}\n")
	case .Case, .Default:
		// Emitted by their switch.
	case .Return:
		tabs(m)
		if st.expr != 0 {
			w(m, "return %s;\n", msl_expr(m, st.expr))
		} else {
			w(m, "return;\n")
		}
	}
}

// Local declarations. Directly in a switch body (`split`), C++ forbids
// jumping past an initialization to a later label, so the variable is
// declared bare and assigned.
@(private)
msl_decl :: proc(m: ^Msl, st: Stmt, split: bool) {
	u := m.u
	for d in list(u, st.decls) {
		v := u.vars[d]
		tabs(m)
		t := msl_type(m, sym_type(u, v.sym))
		init := msl_expr(m, v.init) if v.init != 0 else "{}"
		if split {
			w(m, "%s ok_%s; ok_%s = %s;\n", t, v.name, v.name, init)
		} else {
			w(m, "%s ok_%s = %s;\n", t, v.name, init)
		}
	}
}

// A sub-statement, always braced.
@(private)
msl_sub :: proc(m: ^Msl, id: Stmt_Id) {
	if m.u.stmts[id].kind == .Block {
		msl_stmt(m, id)
		return
	}
	tabs(m)
	w(m, "{{\n")
	m.indent += 1
	msl_stmt(m, id)
	m.indent -= 1
	tabs(m)
	w(m, "}\n")
}

// ---- expressions ------------------------------------------------------------------------

@(private)
msl_float :: proc(v: f64) -> string {
	s := fmt.aprintf("%.9g", f32(v))
	if !strings.contains_any(s, ".en") {
		s = strings.concatenate({s, ".0"})
	}
	return strings.concatenate({s, "f"})
}

@(private)
msl_shadows_channel :: proc(m: ^Msl, name: string) -> bool {
	for row in m.c.interface {
		if row.kind == .Channel && m.u.syms[row.sym].name == name {
			return true
		}
	}
	return false
}

// Text of a sampler argument: the channel's texture and sampler.
@(private)
msl_sampler :: proc(m: ^Msl, id: Expr_Id) -> (tex, smp: string) {
	e := m.u.exprs[id]
	name := m.u.syms[e.sym].name if e.sym != 0 else e.name
	pfx := "okp" if e.sym != 0 && m.u.syms[e.sym].storage == .Param && msl_shadows_channel(m, name) else "ok"
	return fmt.aprintf("%s_%s_tex", pfx, name), fmt.aprintf("%s_%s_smp", pfx, name)
}

// Text of a storage image argument.
@(private)
msl_image :: proc(m: ^Msl, id: Expr_Id) -> string {
	e := m.u.exprs[id]
	name := m.u.syms[e.sym].name if e.sym != 0 else e.name
	pfx := "okp" if e.sym != 0 && m.u.syms[e.sym].storage == .Param else "ok"
	return fmt.aprintf("%s_%s_img", pfx, name)
}

@(private)
msl_buf_prepare :: proc(m: ^Msl, p: Buf_Path, pre: ^[dynamic]string) -> (buf_ptr, base_var: string) {
	u := m.u
	buf_ptr = fmt.aprintf("oksl_buf_%s", u.syms[p.sym].name)
	base_var = msl_temp(m, T_UINT)
	base_str := fmt.aprintf("uint(%s) * %du", msl_expr(m, p.elem_idx), p.stride_words)
	if p.word_off != 0 {
		base_str = fmt.aprintf("%s + %du", base_str, p.word_off)
	}
	append(pre, fmt.aprintf("(%s = %s)", base_var, base_str))
	if p.dyn_comp != 0 {
		clamped := fmt.aprintf("uint(clamp(int(%s), 0, %d))", msl_expr(m, p.dyn_comp), p.dyn_rows - 1)
		if p.dyn_swz_len > 0 {
			ctmp := msl_temp(m, T_UINT)
			append(pre, fmt.aprintf("(%s = %s)", ctmp, clamped))
			mapped := fmt.aprintf("%du", p.dyn_swz[0])
			for k in 1 ..< p.dyn_rows {
				mapped = fmt.aprintf("(%s == %du ? %du : %s)", ctmp, k, p.dyn_swz[k], mapped)
			}
			clamped = mapped
		}
		append(pre, fmt.aprintf("(%s += %s)", base_var, clamped))
	}
	return
}

@(private)
msl_buf_read_at :: proc(m: ^Msl, buf_ptr, base_var: string, t: Type, word: int) -> string {
	u := m.u
	if t.base == .Struct {
		parts := make([dynamic]string, context.temp_allocator)
		for f, i in struct_fields(u, t.strukt) {
			ft := ty(u, f.type)
			off := std430_field_offset(u, t.strukt, i32(i))
			append(&parts, msl_buf_read_at(m, buf_ptr, base_var, ft, word + off / 4))
		}
		return fmt.aprintf("%s{{%s}}", msl_type(m, t), strings.join(parts[:], ", "))
	}
	comps := make([dynamic]string, context.temp_allocator)
	for i in 0 ..< int(t.rows) {
		ref := fmt.aprintf("%s[%s + %du]", buf_ptr, base_var, word + i)
		#partial switch t.base {
		case .Float:
			append(&comps, fmt.aprintf("as_type<float>(%s)", ref))
		case .Int:
			append(&comps, fmt.aprintf("as_type<int>(%s)", ref))
		case:
			append(&comps, ref)
		}
	}
	if t.rows == 1 {
		return comps[0]
	}
	return fmt.aprintf("%s(%s)", msl_type(m, t), strings.join(comps[:], ", "))
}

@(private)
msl_buf_read :: proc(m: ^Msl, p: Buf_Path, buf_ptr, base_var: string) -> string {
	if p.swz_len == 0 {
		return msl_buf_read_at(m, buf_ptr, base_var, p.type, 0)
	}
	sc := scalar_of(p.type)
	comps := make([dynamic]string, context.temp_allocator)
	for k in 0 ..< p.swz_len {
		append(&comps, msl_buf_read_at(m, buf_ptr, base_var, sc, int(p.swz[k])))
	}
	return fmt.aprintf("%s(%s)", msl_type(m, p.type), strings.join(comps[:], ", "))
}

@(private)
msl_buf_write_at :: proc(m: ^Msl, buf_ptr, base_var: string, t: Type, word: int, val_ref: string, post: ^[dynamic]string) {
	u := m.u
	if t.base == .Struct {
		for f, i in struct_fields(u, t.strukt) {
			ft := ty(u, f.type)
			off := std430_field_offset(u, t.strukt, i32(i))
			msl_buf_write_at(m, buf_ptr, base_var, ft, word + off / 4, fmt.aprintf("%s.ok_%s", val_ref, f.name), post)
		}
		return
	}
	for i in 0 ..< int(t.rows) {
		comp := val_ref if t.rows == 1 else fmt.aprintf("%s[%d]", val_ref, i)
		bits := comp if t.base == .Uint else fmt.aprintf("as_type<uint>(%s)", comp)
		append(post, fmt.aprintf("(%s[%s + %du] = %s)", buf_ptr, base_var, word + i, bits))
	}
}

@(private)
msl_buf_write :: proc(m: ^Msl, p: Buf_Path, buf_ptr, base_var, val_var: string, post: ^[dynamic]string) {
	if p.swz_len == 0 {
		msl_buf_write_at(m, buf_ptr, base_var, p.type, 0, val_var, post)
		return
	}
	sc := scalar_of(p.type)
	for k in 0 ..< p.swz_len {
		msl_buf_write_at(m, buf_ptr, base_var, sc, int(p.swz[k]), fmt.aprintf("%s[%d]", val_var, k), post)
	}
}

// Broadcasts scalar expression text `s` to vector type `t`.
@(private)
splat_text :: proc(m: ^Msl, s: string, from, to: Type) -> string {
	if is_scalar(from) && is_vector(to) {
		return fmt.aprintf("%s(%s)", msl_type(m, with_base(to, from.base)), s)
	}
	return s
}

@(private)
wrap_pre :: proc(pre: []string, body: string) -> string {
	if len(pre) == 0 {
		return body
	}
	return fmt.aprintf("(%s, %s)", strings.join(pre, ", "), body)
}

@(private)
msl_expr :: proc(m: ^Msl, id: Expr_Id) -> string {
	u := m.u
	e := u.exprs[id]
	t := ty(u, e.type)
	switch e.kind {
	case .None:
		return ""
	case .Buffer_Load:
		n := u.syms[e.sym].name
		return fmt.aprintf("oksl_buffer_%s(oksl_buf_%s, uint(%s))", n, n, msl_expr(m, e.b))
	case .Int_Lit:
		return "(-2147483647 - 1)" if i32(e.value) == min(i32) else fmt.aprintf("%d", i32(e.value))
	case .Uint_Lit:
		return fmt.aprintf("%du", u32(e.value))
	case .Float_Lit:
		return msl_float(lit_f64(e))
	case .Bool_Lit:
		return "true" if e.value != 0 else "false"
	case .Ident:
		sym := u.syms[e.sym]
		switch sym.storage {
		case .Uniform, .Builtin_Uniform:
			return uniform_read(m, sym)
		case .Global:
			return fmt.aprintf("g.ok_%s", sym.name)
		case .Stage_In:
			return fmt.aprintf("g.in_%s", sym.name)
		case .Stage_Out:
			return fmt.aprintf("g.out_%s", sym.name)
		case .Work_Group_Size:
			ls := m.c.local_size
			return fmt.aprintf("uint3(%du, %du, %du)", ls[0], ls[1], ls[2])
		case .Local, .Param, .Channel, .Buffer, .Storage_Image:
		}
		return fmt.aprintf("ok_%s", sym.name)
	case .Convert:
		return fmt.aprintf("%s(%s)", msl_type(m, t), msl_expr(m, e.a))
	case .Comma:
		return fmt.aprintf("(%s, %s)", msl_expr(m, e.a), msl_expr(m, e.b))
	case .Length:
		at := expr_type(u, e.a)
		n := at.array if is_array(at) else (i32(at.cols) if is_matrix(at) else i32(at.rows))
		return fmt.aprintf("%d", n)
	case .Unary:
		a := msl_expr(m, e.a)
		#partial switch e.op {
		case .Minus:
			if is_matrix(t) {
				return fmt.aprintf("(%s * -1.0f)", a)
			}
			return fmt.aprintf("(-%s)", a)
		case .Bang:
			return fmt.aprintf("(!%s)", a)
		case .Tilde:
			return fmt.aprintf("(~%s)", a)
		}
		return a
	case .Pre_Inc, .Pre_Dec, .Post_Inc, .Post_Dec:
		if bp, is_buf := buf_path(u, e.a); is_buf {
			parts := make([dynamic]string, context.temp_allocator)
			buf_ptr, base_var := msl_buf_prepare(m, bp, &parts)
			old_var := msl_temp(m, t)
			new_var := msl_temp(m, t)
			append(&parts, fmt.aprintf("(%s = %s)", old_var, msl_buf_read(m, bp, buf_ptr, base_var)))
			inc := e.kind == .Pre_Inc || e.kind == .Post_Inc
			one := "1.0f" if t.base == .Float else ("1u" if t.base == .Uint else "1")
			append(&parts, fmt.aprintf("(%s = %s %s %s)", new_var, old_var, "+" if inc else "-", one))
			msl_buf_write(m, bp, buf_ptr, base_var, new_var, &parts)
			append(&parts, new_var if e.kind == .Pre_Inc || e.kind == .Pre_Dec else old_var)
			return fmt.aprintf("(%s)", strings.join(parts[:], ", "))
		}
		#partial switch e.kind {
		case .Pre_Inc:
			return fmt.aprintf("(++%s)", msl_expr(m, e.a))
		case .Pre_Dec:
			return fmt.aprintf("(--%s)", msl_expr(m, e.a))
		case .Post_Inc:
			return fmt.aprintf("(%s++)", msl_expr(m, e.a))
		}
		return fmt.aprintf("(%s--)", msl_expr(m, e.a))
	case .Binary:
		if e.op != .And_And && e.op != .Or_Or && (may_write(u, e.a) || may_write(u, e.b)) {
			ta, tb := expr_type(u, e.a), expr_type(u, e.b)
			la := msl_temp(m, ta)
			lb := msl_temp(m, tb)
			sa := fmt.aprintf("(%s = %s)", la, msl_expr(m, e.a))
			sb := fmt.aprintf("(%s = %s)", lb, msl_expr(m, e.b))
			return fmt.aprintf("(%s, %s, %s)", sa, sb, msl_binary(m, e.op, la, ta, lb, tb))
		}
		return msl_binary(m, e.op, msl_expr(m, e.a), expr_type(u, e.a), msl_expr(m, e.b), expr_type(u, e.b))
	case .Assign:
		if bp, is_buf := buf_path(u, e.a); is_buf {
			parts := make([dynamic]string, context.temp_allocator)
			buf_ptr, base_var := msl_buf_prepare(m, bp, &parts)
			rhs := msl_expr(m, e.b)
			val_var := msl_temp(m, t)
			if e.op == .Eq {
				append(&parts, fmt.aprintf("(%s = %s)", val_var, rhs))
			} else {
				ta, tb := expr_type(u, e.a), expr_type(u, e.b)
				rhs_var := msl_temp(m, tb)
				append(&parts, fmt.aprintf("(%s = %s)", rhs_var, rhs))
				old_expr := msl_buf_read(m, bp, buf_ptr, base_var)
				append(&parts, fmt.aprintf("(%s = %s)", val_var, msl_binary(m, compound_op(e.op), old_expr, ta, rhs_var, tb)))
			}
			msl_buf_write(m, bp, buf_ptr, base_var, val_var, &parts)
			append(&parts, val_var)
			return fmt.aprintf("(%s)", strings.join(parts[:], ", "))
		}
		if may_write(u, e.b) || may_write(u, e.a) {
			parts := make([dynamic]string, context.temp_allocator)
			lv := msl_lvalue(m, e.a, &parts)
			ta, tb := expr_type(u, e.a), expr_type(u, e.b)
			rhs_var := msl_temp(m, tb)
			append(&parts, fmt.aprintf("(%s = %s)", rhs_var, msl_expr(m, e.b)))
			if e.op == .Eq {
				append(&parts, fmt.aprintf("(%s = %s)", lv, rhs_var))
				append(&parts, rhs_var)
			} else {
				val_var := msl_temp(m, t)
				append(&parts, fmt.aprintf("(%s = %s)", val_var, msl_binary(m, compound_op(e.op), lv, ta, rhs_var, tb)))
				append(&parts, fmt.aprintf("(%s = %s)", lv, val_var))
				append(&parts, val_var)
			}
			return fmt.aprintf("(%s)", strings.join(parts[:], ", "))
		}
		lhs := msl_expr(m, e.a)
		rhs := msl_expr(m, e.b)
		if e.op == .Eq {
			return fmt.aprintf("(%s = %s)", lhs, rhs)
		}
		ta, tb := expr_type(u, e.a), expr_type(u, e.b)
		op := compound_op(e.op)
		if is_matrix(ta) && (op != .Star || is_scalar(tb)) && !(op == .Star && is_scalar(tb)) || (is_vector(ta) && is_matrix(tb)) {
			// No compound form in Metal: spell out the operation.
			return fmt.aprintf("(%s = %s)", lhs, msl_binary(m, op, lhs, ta, rhs, tb))
		}
		return fmt.aprintf("(%s %s= %s)", lhs, token_text(op), rhs)
	case .Ternary:
		if t == T_VOID {
			return fmt.aprintf("(%s ? (void)%s : (void)%s)", msl_expr(m, e.a), msl_expr(m, e.b), msl_expr(m, e.c))
		}
		return fmt.aprintf("(%s ? %s : %s)", msl_expr(m, e.a), msl_expr(m, e.b), msl_expr(m, e.c))
	case .Index:
		a := u.exprs[e.a]
		if a.kind == .Ident && a.sym != 0 {
			sym := u.syms[a.sym]
			if (sym.storage == .Uniform || sym.storage == .Builtin_Uniform) && ty(u, sym.type).array > 0 {
				return uniform_read(m, sym, msl_expr(m, e.b))
			}
		}
		if may_write(u, e.a) || may_write(u, e.b) {
			if addressable(u, e.a) {
				pre := make([dynamic]string, context.temp_allocator)
				lv := msl_lvalue(m, id, &pre)
				return wrap_pre(pre[:], lv)
			}
			ta, tb := expr_type(u, e.a), expr_type(u, e.b)
			la, lb := msl_temp(m, ta), msl_temp(m, tb)
			sa := fmt.aprintf("(%s = %s)", la, msl_expr(m, e.a))
			sb := fmt.aprintf("(%s = %s)", lb, msl_expr(m, e.b))
			return fmt.aprintf("(%s, %s, %s[%s])", sa, sb, la, lb)
		}
		return fmt.aprintf("%s[%s]", msl_expr(m, e.a), msl_expr(m, e.b))
	case .Field:
		return msl_field(m, id, msl_expr(m, e.a))
	case .Call:
		switch e.call {
		case .User:
			return msl_user_call(m, id)
		case .Builtin:
			return msl_builtin(m, id)
		case .Constructor:
			return msl_constructor(m, id)
		case .None:
		}
	}
	return ""
}

@(private)
msl_binary :: proc(m: ^Msl, op: Token_Kind, a: string, ta: Type, b: string, tb: Type) -> string {
	#partial switch op {
	case .Xor_Xor:
		return fmt.aprintf("(%s != %s)", a, b)
	case .Eq_Eq, .Not_Eq:
		eq := op == .Eq_Eq
		if is_matrix(ta) {
			return fmt.aprintf("(%soksl_meq(%s, %s))", "" if eq else "!", a, b)
		}
		if is_vector(ta) {
			return fmt.aprintf("%s(%s %s %s)", "all" if eq else "any", a, "==" if eq else "!=", b)
		}
	case .Plus, .Minus, .Slash:
		if is_matrix(ta) || is_matrix(tb) {
			switch {
			case is_matrix(ta) && is_matrix(tb):
				if op == .Slash {
					return fmt.aprintf("oksl_comp_div(%s, %s)", a, b)
				}
			case is_matrix(ta):
				name := "oksl_add_s" if op == .Plus else ("oksl_sub_s" if op == .Minus else "oksl_div_s")
				return fmt.aprintf("%s(%s, %s)", name, a, b)
			case:
				#partial switch op {
				case .Plus:
					return fmt.aprintf("oksl_add_s(%s, %s)", b, a)
				case .Minus:
					return fmt.aprintf("oksl_s_sub(%s, %s)", a, b)
				case:
					return fmt.aprintf("oksl_s_div(%s, %s)", a, b)
				}
			}
		}
	}
	return fmt.aprintf("(%s %s %s)", a, token_text(op), b)
}

@(private)
msl_temp :: proc(m: ^Msl, t: Type) -> string {
	tmp := fmt.aprintf("_t%d", m.temp_id)
	m.temp_id += 1
	fmt.sbprintf(&m.fn_vars, "\t%s %s = {{}};\n", msl_type(m, t), tmp)
	return tmp
}

// Field or swizzle `id` applied to `base`, its operand already emitted.
@(private)
msl_field :: proc(m: ^Msl, id: Expr_Id, base: string) -> string {
	u := m.u
	e := u.exprs[id]
	at := expr_type(u, e.a)
	if e.swizzle_len == 0 {
		return fmt.aprintf("%s.ok_%s", base, struct_fields(u, at.strukt)[e.field].name)
	}
	if is_scalar(at) {
		if e.swizzle_len == 1 {
			return base
		}
		return fmt.aprintf("%s(%s)", msl_type(m, ty(u, e.type)), base)
	}
	comps := "xyzw"
	b := strings.builder_make(context.temp_allocator)
	for i in 0 ..< int(e.swizzle_len) {
		strings.write_byte(&b, comps[e.swizzle[i]])
	}
	return fmt.aprintf("%s.%s", base, strings.to_string(b))
}

// An out/inout argument with its dynamic indices captured into `pre`, so
// copying it in and back evaluates them once, before the call.
@(private)
msl_lvalue :: proc(m: ^Msl, id: Expr_Id, pre: ^[dynamic]string) -> string {
	u := m.u
	e := u.exprs[id]
	#partial switch e.kind {
	case .Index:
		a := u.exprs[e.a]
		if a.kind == .Ident && a.sym != 0 {
			sym := u.syms[a.sym]
			if (sym.storage == .Uniform || sym.storage == .Builtin_Uniform) && ty(u, sym.type).array > 0 {
				ix := u.exprs[e.b]
				if ix.kind == .Int_Lit || ix.kind == .Uint_Lit {
					return uniform_read(m, sym, msl_expr(m, e.b))
				}
				idx := msl_temp(m, expr_type(u, e.b))
				append(pre, fmt.aprintf("(%s = %s)", idx, msl_expr(m, e.b)))
				return uniform_read(m, sym, idx)
			}
		}
		base := msl_lvalue(m, e.a, pre)
		ix := u.exprs[e.b]
		if ix.kind == .Int_Lit || ix.kind == .Uint_Lit {
			return fmt.aprintf("%s[%s]", base, msl_expr(m, e.b))
		}
		idx := msl_temp(m, expr_type(u, e.b))
		append(pre, fmt.aprintf("(%s = %s)", idx, msl_expr(m, e.b)))
		return fmt.aprintf("%s[%s]", base, idx)
	case .Field:
		return msl_field(m, id, msl_lvalue(m, e.a, pre))
	}
	return msl_expr(m, id)
}

// Out and inout arguments pass through temporaries, as in SPIR-V: zeroed
// (out) or copied in (inout) before the call and copied back after it, so
// Metal never binds a reference to a swizzle or to storage the callee also
// reaches.
@(private)
msl_user_call :: proc(m: ^Msl, id: Expr_Id) -> string {
	u := m.u
	f := u.exprs[id].func
	ps := fparams(u, f)
	args := args_of(u, id)
	pre := make([dynamic]string, context.temp_allocator)
	post := make([dynamic]string, context.temp_allocator)
	stage_in := false
	for a in args {
		if may_write(u, a) {
			stage_in = true
			break
		}
	}
	b := strings.builder_make()
	fmt.sbprintf(&b, "ok_%s(", u.funcs[f].name)
	for a, i in args {
		if i > 0 {
			strings.write_string(&b, ", ")
		}
		if expr_type(u, a).base == .Sampler {
			tex, smp := msl_sampler(m, a)
			fmt.sbprintf(&b, "%s, %s", tex, smp)
			continue
		}
		if expr_type(u, a).base == .Image {
			strings.write_string(&b, msl_image(m, a))
			continue
		}
		if ps[i].qual == .In {
			if stage_in {
				tmp := msl_temp(m, expr_type(u, a))
				append(&pre, fmt.aprintf("(%s = %s)", tmp, msl_expr(m, a)))
				strings.write_string(&b, tmp)
			} else {
				strings.write_string(&b, msl_expr(m, a))
			}
			continue
		}
		pt := ty(u, ps[i].type)
		if bp, is_buf := buf_path(u, a); is_buf {
			buf_ptr, base_var := msl_buf_prepare(m, bp, &pre)
			tmp := msl_temp(m, pt)
			if ps[i].qual == .Inout {
				append(&pre, fmt.aprintf("(%s = %s)", tmp, msl_buf_read(m, bp, buf_ptr, base_var)))
			} else {
				append(&pre, fmt.aprintf("(%s = %s{{}})", tmp, msl_type(m, pt)))
			}
			msl_buf_write(m, bp, buf_ptr, base_var, tmp, &post)
			strings.write_string(&b, tmp)
			continue
		}
		lv := msl_lvalue(m, a, &pre)
		tmp := msl_temp(m, pt)
		if ps[i].qual == .Inout {
			append(&pre, fmt.aprintf("(%s = %s)", tmp, lv))
		} else {
			// Review: zeroed on every call, as the CPU and SPIR-V do; without
			// it a call in a loop sees the previous call's value wherever the
			// callee leaves its out parameter unwritten.
			append(&pre, fmt.aprintf("(%s = %s{{}})", tmp, msl_type(m, pt)))
		}
		append(&post, fmt.aprintf("(%s = %s)", lv, tmp))
		strings.write_string(&b, tmp)
	}
	if len(args) > 0 {
		strings.write_string(&b, ", ")
	}
	fmt.sbprintf(&b, "%s)", m.args)
	call := strings.to_string(b)
	if len(pre) == 0 && len(post) == 0 {
		return call
	}
	parts := make([dynamic]string, context.temp_allocator)
	append(&parts, ..pre[:])
	ret_t := ty(u, u.funcs[f].ret)
	if ret_t != T_VOID {
		rt := msl_temp(m, ret_t)
		append(&parts, fmt.aprintf("(%s = %s)", rt, call))
		append(&parts, ..post[:])
		append(&parts, rt)
	} else {
		append(&parts, call)
		append(&parts, ..post[:])
	}
	return fmt.aprintf("(%s)", strings.join(parts[:], ", "))
}

@(private)
msl_constructor :: proc(m: ^Msl, id: Expr_Id) -> string {
	u := m.u
	t := expr_type(u, id)
	args := args_of(u, id)
	pre := make([dynamic]string, context.temp_allocator)
	need_seq := false
	if len(args) > 1 {
		for a in args {
			if may_write(u, a) {
				need_seq = true
				break
			}
		}
	}
	texts := make([]string, len(args), context.temp_allocator)
	for a, i in args {
		s := msl_expr(m, a)
		if need_seq {
			tmp := msl_temp(m, expr_type(u, a))
			append(&pre, fmt.aprintf("(%s = %s)", tmp, s))
			s = tmp
		}
		texts[i] = s
	}
	if t.array > 0 || t.base == .Struct {
		return wrap_pre(pre[:], fmt.aprintf("%s{{%s}}", msl_type(m, t), strings.join(texts, ", ")))
	}
	if len(args) == 1 {
		at := expr_type(u, args[0])
		switch {
		case is_matrix(t) && is_matrix(at):
			if t == at {
				return wrap_pre(pre[:], texts[0])
			}
			return wrap_pre(pre[:], fmt.aprintf("oksl_resize<%d, %d>(%s)", t.cols, t.rows, texts[0]))
		case is_matrix(t) && is_scalar(at):
			return wrap_pre(pre[:], fmt.aprintf("oksl_diag<%d, %d>(float(%s))", t.cols, t.rows, texts[0]))
		case is_scalar(at) || is_scalar(t):
			if is_scalar(t) && !is_scalar(at) {
				// The first component.
				first := fmt.aprintf("%s[0]", texts[0]) if is_vector(at) else fmt.aprintf("%s[0][0]", texts[0])
				return wrap_pre(pre[:], fmt.aprintf("%s(%s)", msl_type(m, t), first))
			}
			return wrap_pre(pre[:], fmt.aprintf("%s(%s)", msl_type(m, t), texts[0]))
		case is_vector(t) && is_vector(at):
			if at.rows > t.rows {
				comps := "xyzw"
				return wrap_pre(pre[:], fmt.aprintf("%s(%s.%s)", msl_type(m, t), texts[0], comps[:t.rows]))
			}
			return wrap_pre(pre[:], fmt.aprintf("%s(%s)", msl_type(m, t), texts[0]))
		}
	}
	// Vector of scalars and vectors: Metal flattens like GLSL once each
	// argument has the target's base. Anything with a matrix, or a matrix
	// built from components, goes through a helper.
	native := is_vector(t)
	for a in args {
		if is_matrix(expr_type(u, a)) {
			native = false
		}
	}
	if native {
		b := strings.builder_make()
		fmt.sbprintf(&b, "%s(", msl_type(m, t))
		for a, i in args {
			if i > 0 {
				strings.write_string(&b, ", ")
			}
			at := expr_type(u, a)
			if at.base != t.base {
				fmt.sbprintf(&b, "%s(%s)", msl_type(m, with_base(at, t.base)), texts[i])
			} else {
				strings.write_string(&b, texts[i])
			}
		}
		strings.write_string(&b, ")")
		return wrap_pre(pre[:], strings.to_string(b))
	}
	return wrap_pre(pre[:], fmt.aprintf("%s(%s)", ctor_helper(m, t, args), strings.join(texts, ", ")))
}

// A helper building `t` from the given argument types component by
// component.
@(private)
ctor_helper :: proc(m: ^Msl, t: Type, args: []Expr_Id) -> string {
	u := m.u
	key_b := strings.builder_make(context.temp_allocator)
	strings.write_string(&key_b, msl_type(m, t))
	for a in args {
		fmt.sbprintf(&key_b, "|%s", msl_type(m, expr_type(u, a)))
	}
	key := strings.to_string(key_b)
	if name, ok := m.ctor_helpers[key]; ok {
		return name
	}
	name := fmt.aprintf("oksl_ctor%d", len(m.ctor_helpers))
	m.ctor_helpers[strings.clone(key)] = name
	comps := make([dynamic]string, context.temp_allocator)
	sig := make([dynamic]string, context.temp_allocator)
	sb := msl_scalar(t.base)
	for a, i in args {
		at := expr_type(u, a)
		append(&sig, fmt.aprintf("%s a%d", msl_type(m, at), i))
		switch {
		case is_scalar(at):
			append(&comps, fmt.aprintf("%s(a%d)", sb, i))
		case is_vector(at):
			for k in 0 ..< int(at.rows) {
				append(&comps, fmt.aprintf("%s(a%d[%d])", sb, i, k))
			}
		case:
			for c in 0 ..< int(at.cols) {
				for r in 0 ..< int(at.rows) {
					append(&comps, fmt.aprintf("%s(a%d[%d][%d])", sb, i, c, r))
				}
			}
		}
	}
	body: string
	if is_vector(t) {
		body = fmt.aprintf("%s(%s)", msl_type(m, t), strings.join(comps[:int(t.rows)], ", "))
	} else {
		cols := make([dynamic]string, context.temp_allocator)
		for c in 0 ..< int(t.cols) {
			append(&cols, fmt.aprintf("float%d(%s)", t.rows, strings.join(comps[c * int(t.rows):(c + 1) * int(t.rows)], ", ")))
		}
		body = fmt.aprintf("%s(%s)", msl_type(m, t), strings.join(cols[:], ", "))
	}
	fmt.sbprintf(&m.helpers, "inline %s %s(%s) {{ return %s; }}\n", msl_type(m, t), name, strings.join(sig[:], ", "), body)
	return name
}

@(private)
msl_builtin :: proc(m: ^Msl, id: Expr_Id) -> string {
	u := m.u
	e := u.exprs[id]
	b := e.builtin
	t := ty(u, e.type)
	args := args_of(u, id)
	pre := make([dynamic]string, context.temp_allocator)
	need_seq := b == .Image_Store
	if !need_seq && len(args) > 1 {
		for x in args {
			if may_write(u, x) {
				need_seq = true
				break
			}
		}
	}
	a := make([]string, len(args), context.temp_allocator)
	for x, i in args {
		at := expr_type(u, x)
		if at.base != .Sampler && at.base != .Image {
			s := msl_expr(m, x)
			if need_seq {
				tmp := msl_temp(m, at)
				append(&pre, fmt.aprintf("(%s = %s)", tmp, s))
				s = tmp
			}
			a[i] = s
		}
		if i in BUILTINS[b].broadcast {
			a[i] = splat_text(m, a[i], at, t)
		}
	}
	if form := MSL_BUILTINS[b]; form != "" {
		if strings.contains_rune(form, '%') {
			vals := make([]any, len(a), context.temp_allocator)
			for &v, i in vals {
				v = a[i]
			}
			return wrap_pre(pre[:], fmt.aprintf(form, ..vals))
		}
		return wrap_pre(pre[:], fmt.aprintf("%s(%s)", form, strings.join(a, ", ")))
	}
	#partial switch b {
	case .Dot:
		if is_scalar(expr_type(u, args[0])) {
			return wrap_pre(pre[:], fmt.aprintf("(%s * %s)", a[0], a[1]))
		}
		return wrap_pre(pre[:], fmt.aprintf("dot(%s, %s)", a[0], a[1]))
	case .Float_Bits_To_Int, .Float_Bits_To_Uint, .Int_Bits_To_Float, .Uint_Bits_To_Float:
		return wrap_pre(pre[:], fmt.aprintf("as_type<%s>(%s)", msl_type(m, t), a[0]))
	case .Texture_Lod, .Texture, .Texture_Grad:
		tex, smp := msl_sampler(m, args[0])
		if expr_type(u, args[0]) == T_SAMPLER_ARRAY {
			extra := fmt.aprintf(", %s, %s", a[2], a[3]) if b == .Texture_Grad else fmt.aprintf(", %s", a[2])
			return wrap_pre(pre[:], fmt.aprintf("%s(%s, %s, %s%s)", msl_array_helper(m, b), tex, smp, a[1], extra))
		}
		#partial switch b {
		case .Texture_Lod:
			return wrap_pre(pre[:], fmt.aprintf("%s.sample(%s, %s, level(%s))", tex, smp, a[1], a[2]))
		case .Texture:
			return wrap_pre(pre[:], fmt.aprintf("%s.sample(%s, %s, bias(%s))", tex, smp, a[1], a[2]))
		}
		return wrap_pre(pre[:], fmt.aprintf("%s.sample(%s, %s, gradient2d(%s, %s))", tex, smp, a[1], a[2], a[3]))
	case .Texel_Fetch:
		rules := PROFILE_RULES[u.profile]
		tex, smp := msl_sampler(m, args[0])
		sz := a[3] if rules.header else fmt.aprintf("float2(int2(%s.get_width(uint(%s)), %s.get_height(uint(%s))))", tex, a[2], tex, a[2])
		uv := fmt.aprintf("((float2(%s) + float2(0.5f)) / %s)", a[1], sz)
		if rules.flip_y {
			uv = fmt.aprintf("((%s * float2(1.0f, -1.0f)) + float2(0.0f, 1.0f))", uv)
		}
		return wrap_pre(pre[:], fmt.aprintf("%s.sample(%s, %s, level(float(%s)))", tex, smp, uv, a[2]))
	case .Texture_Size:
		tex, _ := msl_sampler(m, args[0])
		if expr_type(u, args[0]) == T_SAMPLER_ARRAY {
			return wrap_pre(pre[:], fmt.aprintf("int3(%s.get_width(uint(%s)), %s.get_height(uint(%s)), %s.get_array_size())", tex, a[1], tex, a[1], tex))
		}
		return wrap_pre(pre[:], fmt.aprintf("int2(%s.get_width(uint(%s)), %s.get_height(uint(%s)))", tex, a[1], tex, a[1]))
	case .Image_Load:
		return wrap_pre(pre[:], fmt.aprintf("%s.read(uint2(%s))", msl_image(m, args[0]), a[1]))
	case .Image_Store:
		img := msl_image(m, args[0])
		return wrap_pre(pre[:], fmt.aprintf("(oksl_fence(%s), %s.write(%s, uint2(%s)), oksl_fence(%s))", img, img, a[2], a[1], img))
	case .Image_Size:
		img := msl_image(m, args[0])
		return wrap_pre(pre[:], fmt.aprintf("int2(%s.get_width(), %s.get_height())", img, img))
	}
	return ""
}

// Each built-in in Metal: a function called with the arguments (those in
// BUILTINS' broadcast sets splatted), or an expression whose %s take them.
// The rest are written by hand above.
@(rodata)
MSL_BUILTINS := #partial [Builtin]string {
	.Radians            = "(%s * 0.0174532925f)",
	.Degrees            = "(%s * 57.2957795f)",
	.Sin                = "sin",
	.Cos                = "cos",
	.Tan                = "tan",
	.Asin               = "asin",
	.Acos               = "acos",
	.Atan               = "atan",
	.Atan2              = "atan2",
	.Sinh               = "sinh",
	.Cosh               = "cosh",
	.Tanh               = "tanh",
	.Asinh              = "asinh",
	.Acosh              = "acosh",
	.Atanh              = "atanh",
	.Pow                = "pow",
	.Exp                = "exp",
	.Log                = "log",
	.Exp2               = "exp2",
	.Log2               = "log2",
	.Sqrt               = "sqrt",
	.Inversesqrt        = "rsqrt",
	.Floor              = "floor",
	.Ceil               = "ceil",
	.Fract              = "fract",
	.Trunc              = "trunc",
	.Round              = "round",
	.Round_Even         = "rint",
	.Abs                = "abs",
	.Sign               = "sign",
	// GLSL's mod (x - y * floor(x / y)), not Metal's fmod.
	.Mod                = "oksl_mod",
	.Min                = "min",
	.Max                = "max",
	.Clamp              = "clamp",
	.Mix                = "mix",
	.Mix_Bool           = "select",
	.Step               = "step",
	.Smoothstep         = "smoothstep",
	.Normalize          = "normalize",
	.Length             = "length",
	.Distance           = "distance",
	.Cross              = "cross",
	.Faceforward        = "faceforward",
	.Reflect            = "reflect",
	.Refract            = "refract",
	.Transpose          = "transpose",
	.Determinant        = "determinant",
	.Inverse            = "oksl_inverse",
	.Isnan              = "isnan",
	.Isinf              = "isinf",
	.Less_Than          = "(%s < %s)",
	.Less_Than_Equal    = "(%s <= %s)",
	.Greater_Than       = "(%s > %s)",
	.Greater_Than_Equal = "(%s >= %s)",
	.Equal              = "(%s == %s)",
	.Not_Equal          = "(%s != %s)",
	.Any                = "any",
	.All                = "all",
	.Not                = "(!%s)",
	.Dfdx               = "dfdx",
	.Dfdy               = "dfdy",
	.Fwidth             = "fwidth",
	.Matrix_Comp_Mult   = "oksl_comp_mult",
	.Outer_Product      = "oksl_outer",
}
