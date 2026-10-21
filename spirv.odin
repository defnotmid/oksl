package oksl

import "core:fmt"
import "core:strings"

// SPIR-V 1.0 emission for Vulkan through SDL_GPU: one fragment entry
// "main" reading `uv` (location 0) and writing a vec4 (location 0); the
// uniform block at set 3 binding 0 and channels at set 2 binding n.

@(private)
Op :: enum u32 {
	Undef                    = 1,
	Name                     = 5,
	Member_Name              = 6,
	Ext_Inst_Import          = 11,
	Ext_Inst                 = 12,
	Memory_Model             = 14,
	Entry_Point              = 15,
	Execution_Mode           = 16,
	Capability               = 17,
	Type_Void                = 19,
	Type_Bool                = 20,
	Type_Int                 = 21,
	Type_Float               = 22,
	Type_Vector              = 23,
	Type_Matrix              = 24,
	Type_Image               = 25,
	Type_Sampled_Image       = 27,
	Type_Array               = 28,
	Type_Struct              = 30,
	Type_Pointer             = 32,
	Type_Function            = 33,
	Constant_True            = 41,
	Constant_False           = 42,
	Constant                 = 43,
	Constant_Composite       = 44,
	Constant_Null            = 46,
	Function                 = 54,
	Function_Parameter       = 55,
	Function_End             = 56,
	Function_Call            = 57,
	Variable                 = 59,
	Load                     = 61,
	Store                    = 62,
	Access_Chain             = 65,
	Decorate                 = 71,
	Member_Decorate          = 72,
	Vector_Extract_Dynamic   = 77,
	Vector_Shuffle           = 79,
	Composite_Construct      = 80,
	Composite_Extract        = 81,
	Composite_Insert         = 82,
	Transpose                = 84,
	Image_Sample_Implicit_Lod = 87,
	Image_Sample_Explicit_Lod = 88,
	Image                    = 100,
	Image_Query_Size_Lod     = 103,
	Convert_F_To_U           = 109,
	Convert_F_To_S           = 110,
	Convert_S_To_F           = 111,
	Convert_U_To_F           = 112,
	Bitcast                  = 124,
	S_Negate                 = 126,
	F_Negate                 = 127,
	I_Add                    = 128,
	F_Add                    = 129,
	I_Sub                    = 130,
	F_Sub                    = 131,
	I_Mul                    = 132,
	F_Mul                    = 133,
	U_Div                    = 134,
	S_Div                    = 135,
	F_Div                    = 136,
	U_Mod                    = 137,
	S_Rem                    = 138,
	Vector_Times_Scalar      = 142,
	Matrix_Times_Scalar      = 143,
	Vector_Times_Matrix      = 144,
	Matrix_Times_Vector      = 145,
	Matrix_Times_Matrix      = 146,
	Outer_Product            = 147,
	Dot                      = 148,
	Any                      = 154,
	All                      = 155,
	Is_Nan                   = 156,
	Is_Inf                   = 157,
	Logical_Equal            = 164,
	Logical_Not_Equal        = 165,
	Logical_Or               = 166,
	Logical_And              = 167,
	Logical_Not              = 168,
	Select                   = 169,
	I_Equal                  = 170,
	I_Not_Equal              = 171,
	U_Greater_Than           = 172,
	S_Greater_Than           = 173,
	U_Greater_Than_Equal     = 174,
	S_Greater_Than_Equal     = 175,
	U_Less_Than              = 176,
	S_Less_Than              = 177,
	U_Less_Than_Equal        = 178,
	S_Less_Than_Equal        = 179,
	F_Ord_Equal              = 180,
	F_Unord_Not_Equal        = 183,
	F_Ord_Less_Than          = 184,
	F_Ord_Greater_Than       = 186,
	F_Ord_Less_Than_Equal    = 188,
	F_Ord_Greater_Than_Equal = 190,
	Shift_Right_Logical      = 194,
	Shift_Right_Arithmetic   = 195,
	Shift_Left_Logical       = 196,
	Bitwise_Or               = 197,
	Bitwise_Xor              = 198,
	Bitwise_And              = 199,
	Not                      = 200,
	D_Pdx                    = 207,
	D_Pdy                    = 208,
	Fwidth                   = 209,
	Phi                      = 245,
	Loop_Merge               = 246,
	Selection_Merge          = 247,
	Label                    = 248,
	Branch                   = 249,
	Branch_Conditional       = 250,
	Switch                   = 251,
	Kill                     = 252,
	Return                   = 253,
	Return_Value             = 254,
	Unreachable              = 255,
}

// GLSL.std.450 extended instructions.
@(private)
Glsl :: enum u32 {
	Round        = 1,
	Round_Even   = 2,
	Trunc        = 3,
	F_Abs        = 4,
	S_Abs        = 5,
	F_Sign       = 6,
	S_Sign       = 7,
	Floor        = 8,
	Ceil         = 9,
	Fract        = 10,
	Radians      = 11,
	Degrees      = 12,
	Sin          = 13,
	Cos          = 14,
	Tan          = 15,
	Asin         = 16,
	Acos         = 17,
	Atan         = 18,
	Sinh         = 19,
	Cosh         = 20,
	Tanh         = 21,
	Asinh        = 22,
	Acosh        = 23,
	Atanh        = 24,
	Atan2        = 25,
	Pow          = 26,
	Exp          = 27,
	Log          = 28,
	Exp2         = 29,
	Log2         = 30,
	Sqrt         = 31,
	Inverse_Sqrt = 32,
	Determinant  = 33,
	Inverse      = 34,
	F_Min        = 37,
	U_Min        = 38,
	S_Min        = 39,
	F_Max        = 40,
	U_Max        = 41,
	S_Max        = 42,
	F_Clamp      = 43,
	U_Clamp      = 44,
	S_Clamp      = 45,
	F_Mix        = 46,
	Step         = 48,
	Smooth_Step  = 49,
	Length       = 66,
	Distance     = 67,
	Cross        = 68,
	Normalize    = 69,
	Face_Forward = 70,
	Reflect      = 71,
	Refract      = 72,
}

@(private)
SC_UNIFORM_CONSTANT :: 0
@(private)
SC_INPUT :: 1
@(private)
SC_UNIFORM :: 2
@(private)
SC_OUTPUT :: 3
@(private)
SC_PRIVATE :: 6
@(private)
SC_FUNCTION :: 7

@(private)
DEC_BLOCK :: 2
@(private)
DEC_COL_MAJOR :: 5
@(private)
DEC_ARRAY_STRIDE :: 6
@(private)
DEC_MATRIX_STRIDE :: 7
@(private)
DEC_BUILTIN :: 11
@(private)
DEC_FLAT :: 14
@(private)
DEC_LOCATION :: 30
@(private)
DEC_BINDING :: 33
@(private)
DEC_DESCRIPTOR_SET :: 34
@(private)
DEC_OFFSET :: 35
@(private)
BUILTIN_POSITION :: 0
@(private)
BUILTIN_FRAG_COORD :: 15
@(private)
BUILTIN_FRONT_FACING :: 17

// Where break and continue go: a loop's merge and continue blocks, or a
// switch's merge and the continue block of the loop around it.
@(private)
Loop_Target :: struct {
	merge, cont: u32,
	broke:       bool,
}

// A place values can be loaded from or stored to.
@(private)
LValue :: struct {
	ptr:     u32,
	type:    Type,
	storage: u32,
	// Bools in the uniform block are stored as uints.
	as_uint: bool,
	swz:     [4]u8,
	swz_len: int,
}

@(private)
Spv :: struct {
	u:           ^Unit,
	checked:     ^Checked,
	next_id:     u32,
	caps:        [dynamic]u32,
	imports:     [dynamic]u32,
	header:      [dynamic]u32,
	debug:       [dynamic]u32,
	annotations: [dynamic]u32,
	globals:     [dynamic]u32,
	functions:   [dynamic]u32,
	types:       map[Type]u32,
	ptr_types:   map[[2]u32]u32,
	fn_types:    map[string]u32,
	consts:      map[[3]u32]u32,
	nulls:       map[u32]u32,
	glsl:        u32,
	t_image:     u32,
	t_sampled:   u32,
	// The ImageQuery capability is declared (textureSize).
	image_query: bool,
	block_var:   u32,
	block_type:  u32,
	// Member index of each user uniform symbol in the block.
	members:     map[Sym_Id]u32,
	// SPIR-V type id of each block member.
	member_types: []u32,
	// SPIR-V id per symbol (variable pointer) and per function.
	sym_ids:     []u32,
	func_ids:    []u32,
	// Per function.
	vars:        [dynamic]u32,
	body:        [dynamic]u32,
	cur_block:   u32,
	terminated:  bool,
	loops:       [dynamic]Loop_Target,
}

emit_spirv :: proc(u: ^Unit, c: ^Checked) -> []u32 {
	s := Spv {
		u       = u,
		checked = c,
		next_id = 1,
	}
	s.sym_ids = make([]u32, len(u.syms))
	s.func_ids = make([]u32, len(u.funcs))
	s.caps = make([dynamic]u32)
	s.imports = make([dynamic]u32)
	s.header = make([dynamic]u32)
	s.debug = make([dynamic]u32)
	s.annotations = make([dynamic]u32)
	s.globals = make([dynamic]u32)
	s.functions = make([dynamic]u32)
	s.types = make(map[Type]u32)
	s.ptr_types = make(map[[2]u32]u32)
	s.fn_types = make(map[string]u32)
	s.consts = make(map[[3]u32]u32)
	s.nulls = make(map[u32]u32)
	s.members = make(map[Sym_Id]u32)
	s.loops = make([dynamic]Loop_Target)

	inst(&s.caps, .Capability, 1) // Shader
	s.glsl = new_id(&s)
	inst_str(&s.imports, .Ext_Inst_Import, {s.glsl}, "GLSL.std.450")

	declare_interface(&s)
	for f in c.funcs {
		s.func_ids[f] = new_id(&s)
	}
	_ = s.u
	for f in c.funcs {
		emit_function(&s, f)
	}
	entry := s.func_ids[c.entry]

	header := make([dynamic]u32)
	inst(&header, .Memory_Model, 0, 1) // Logical, GLSL450
	ep := make([dynamic]u32, context.temp_allocator)
	stage := STAGE_RULES[u.stage]
	append(&ep, stage.execution_model, entry)
	append_str(&ep, SPIRV_ENTRY)
	append(&ep, ..s.header[:])
	inst_words(&header, .Entry_Point, ep[:])
	for mode in stage.execution_modes {
		inst(&header, .Execution_Mode, entry, mode)
	}

	out := make([dynamic]u32)
	append(&out, 0x07230203, 0x00010000, 0, s.next_id, 0)
	append(&out, ..s.caps[:])
	append(&out, ..s.imports[:])
	append(&out, ..header[:])
	append(&out, ..s.debug[:])
	append(&out, ..s.annotations[:])
	append(&out, ..s.globals[:])
	append(&out, ..s.functions[:])
	return out[:]
}

// ---- encoding -------------------------------------------------------------------------

@(private)
new_id :: proc(s: ^Spv) -> u32 {
	id := s.next_id
	s.next_id += 1
	return id
}

@(private)
inst :: proc(buf: ^[dynamic]u32, op: Op, operands: ..u32) {
	append(buf, u32(len(operands) + 1) << 16 | u32(op))
	append(buf, ..operands)
}

@(private)
inst_words :: proc(buf: ^[dynamic]u32, op: Op, operands: []u32) {
	append(buf, u32(len(operands) + 1) << 16 | u32(op))
	append(buf, ..operands)
}

@(private)
append_str :: proc(buf: ^[dynamic]u32, str: string) {
	n := len(str) + 1
	words := (n + 3) / 4
	for w in 0 ..< words {
		v: u32
		for k in 0 ..< 4 {
			i := w * 4 + k
			if i < len(str) {
				v |= u32(str[i]) << u32(8 * k)
			}
		}
		append(buf, v)
	}
}

@(private)
inst_str :: proc(buf: ^[dynamic]u32, op: Op, before: []u32, str: string, after: ..u32) {
	words := make([dynamic]u32, context.temp_allocator)
	append(&words, ..before)
	append_str(&words, str)
	append(&words, ..after)
	inst_words(buf, op, words[:])
}

@(private)
name :: proc(s: ^Spv, id: u32, str: string) {
	inst_str(&s.debug, .Name, {id}, str)
}

// ---- types and constants ----------------------------------------------------------------

@(private)
type_id :: proc(s: ^Spv, t: Type) -> u32 {
	if id, ok := s.types[t]; ok {
		return id
	}
	id: u32
	switch {
	case t.array > 0:
		elem := type_id(s, elem_type(t))
		n := const_u32(s, u32(t.array))
		id = new_id(s)
		inst(&s.globals, .Type_Array, id, elem, n)
	case t.base == .Struct:
		members := make([dynamic]u32, context.temp_allocator)
		for f in struct_fields(s.u, t.strukt) {
			append(&members, type_id(s, ty(s.u, f.type)))
		}
		id = new_id(s)
		words := make([dynamic]u32, context.temp_allocator)
		append(&words, id)
		append(&words, ..members[:])
		inst_words(&s.globals, .Type_Struct, words[:])
		name(s, id, s.u.structs[t.strukt].name)
	case t.cols > 0:
		col := type_id(s, column_type(t))
		id = new_id(s)
		inst(&s.globals, .Type_Matrix, id, col, u32(t.cols))
	case t.rows > 1:
		comp := type_id(s, scalar_of(t))
		id = new_id(s)
		inst(&s.globals, .Type_Vector, id, comp, u32(t.rows))
	case:
		id = new_id(s)
		switch t.base {
		case .Void:
			inst(&s.globals, .Type_Void, id)
		case .Bool:
			inst(&s.globals, .Type_Bool, id)
		case .Int:
			inst(&s.globals, .Type_Int, id, 32, 1)
		case .Uint:
			inst(&s.globals, .Type_Int, id, 32, 0)
		case .Float:
			inst(&s.globals, .Type_Float, id, 32)
		case .Sampler:
			s.next_id -= 1
			return sampled_image_type(s)
		case .Struct:
		}
	}
	s.types[t] = id
	return id
}

@(private)
sampled_image_type :: proc(s: ^Spv) -> u32 {
	if s.t_sampled != 0 {
		return s.t_sampled
	}
	f := type_id(s, T_FLOAT)
	s.t_image = new_id(s)
	inst(&s.globals, .Type_Image, s.t_image, f, 1, 0, 0, 0, 1, 0) // 2D, sampled, unknown format
	s.t_sampled = new_id(s)
	inst(&s.globals, .Type_Sampled_Image, s.t_sampled, s.t_image)
	return s.t_sampled
}

@(private)
ptr_type :: proc(s: ^Spv, storage, pointee: u32) -> u32 {
	key := [2]u32{storage, pointee}
	if id, ok := s.ptr_types[key]; ok {
		return id
	}
	id := new_id(s)
	inst(&s.globals, .Type_Pointer, id, storage, pointee)
	s.ptr_types[key] = id
	return id
}

@(private)
fn_type :: proc(s: ^Spv, ret: u32, params: []u32) -> u32 {
	key := fmt.tprint(ret, params)
	if id, ok := s.fn_types[key]; ok {
		return id
	}
	id := new_id(s)
	words := make([dynamic]u32, context.temp_allocator)
	append(&words, id, ret)
	append(&words, ..params)
	inst_words(&s.globals, .Type_Function, words[:])
	s.fn_types[strings.clone(key)] = id
	return id
}

@(private)
const_bits :: proc(s: ^Spv, base: Base, bits: u32) -> u32 {
	key := [3]u32{u32(base), bits, 0}
	if id, ok := s.consts[key]; ok {
		return id
	}
	id := new_id(s)
	switch base {
	case .Bool:
		inst(&s.globals, .Constant_True if bits != 0 else .Constant_False, type_id(s, T_BOOL), id)
	case .Int, .Uint, .Float:
		inst(&s.globals, .Constant, type_id(s, Type{base = base, rows = 1}), id, bits)
	case .Void, .Sampler, .Struct:
	}
	s.consts[key] = id
	return id
}

@(private)
const_f32 :: proc(s: ^Spv, v: f32) -> u32 {
	return const_bits(s, .Float, transmute(u32)v)
}

@(private)
const_i32 :: proc(s: ^Spv, v: i32) -> u32 {
	return const_bits(s, .Int, u32(v))
}

@(private)
const_u32 :: proc(s: ^Spv, v: u32) -> u32 {
	return const_bits(s, .Uint, v)
}

@(private)
const_null :: proc(s: ^Spv, t: u32) -> u32 {
	if id, ok := s.nulls[t]; ok {
		return id
	}
	id := new_id(s)
	inst(&s.globals, .Constant_Null, t, id)
	s.nulls[t] = id
	return id
}

// A scalar constant of `base` with value v (0 or 1 usually).
@(private)
const_of :: proc(s: ^Spv, base: Base, v: f64) -> u32 {
	switch base {
	case .Float:
		return const_f32(s, f32(v))
	case .Int:
		return const_i32(s, i32(v))
	case .Uint:
		return const_u32(s, u32(v))
	case .Bool:
		return const_bits(s, .Bool, u32(v != 0))
	case .Void, .Sampler, .Struct:
	}
	return 0
}

// ---- module interface ------------------------------------------------------------------

@(private)
Block_Member :: struct {
	name:   string,
	type:   Type,
	offset: int,
}

// The uniform rows come first in Input.
BUILTIN_MEMBER_COUNT :: int(Input.Channel_Time) + 1

// The block's first members: every uniform input, in offset order.
@(private)
builtin_members :: proc() -> (out: [BUILTIN_MEMBER_COUNT]Block_Member) {
	i := 0
	for in_ in UNIFORM_INPUTS {
		out[i] = {INPUTS[in_].name, INPUTS[in_].type, INPUTS[in_].offset}
		i += 1
	}
	return
}

@(private)
builtin_member_index :: proc(offset: int) -> u32 {
	for m, i in builtin_members() {
		if m.offset == offset {
			return u32(i)
		}
	}
	return 0
}

// The type a uniform member has in the block: bools become uints and
// arrays get their std140 stride.
@(private)
block_member_type :: proc(s: ^Spv, t: Type) -> u32 {
	st := t
	if st.base == .Bool {
		st.base = .Uint
	}
	if st.array > 0 {
		elem := type_id(s, elem_type(st))
		id := new_id(s)
		inst(&s.globals, .Type_Array, id, elem, const_u32(s, u32(st.array)))
		inst(&s.annotations, .Decorate, id, DEC_ARRAY_STRIDE, u32(std140_array_stride(st)))
		return id
	}
	return type_id(s, st)
}

@(private)
declare_interface :: proc(s: ^Spv) {
	c := s.checked
	for row in c.interface {
		switch row.kind {
		case .Stage_In, .Frag_Coord, .Front_Facing, .Stage_Out, .Position:
			sym := s.u.syms[row.sym]
			class := u32(SC_OUTPUT if row.kind == .Stage_Out || row.kind == .Position else SC_INPUT)
			v := new_id(s)
			inst(&s.globals, .Variable, ptr_type(s, class, type_id(s, ty(s.u, sym.type))), v, class)
			#partial switch row.kind {
			case .Frag_Coord:
				inst(&s.annotations, .Decorate, v, DEC_BUILTIN, BUILTIN_FRAG_COORD)
			case .Front_Facing:
				inst(&s.annotations, .Decorate, v, DEC_BUILTIN, BUILTIN_FRONT_FACING)
			case .Position:
				inst(&s.annotations, .Decorate, v, DEC_BUILTIN, BUILTIN_POSITION)
			case:
				inst(&s.annotations, .Decorate, v, DEC_LOCATION, u32(row.binding))
			}
			if row.flat {
				inst(&s.annotations, .Decorate, v, DEC_FLAT)
			}
			name(s, v, sym.name)
			append(&s.header, v)
			s.sym_ids[row.sym] = v
		case .Uniform_Block:
			declare_uniform_block(s, row)
		case .Channel:
			v := new_id(s)
			inst(&s.globals, .Variable, ptr_type(s, SC_UNIFORM_CONSTANT, sampled_image_type(s)), v, SC_UNIFORM_CONSTANT)
			inst(&s.annotations, .Decorate, v, DEC_DESCRIPTOR_SET, u32(row.set))
			inst(&s.annotations, .Decorate, v, DEC_BINDING, u32(row.binding))
			name(s, v, s.u.syms[row.sym].name)
			s.sym_ids[row.sym] = v
		}
	}

	// Private globals, zero-initialized; the entry runs their initializers.
	for g in c.globals {
		t := type_id(s, sym_type(s.u, g))
		v := new_id(s)
		inst(&s.globals, .Variable, ptr_type(s, SC_PRIVATE, t), v, SC_PRIVATE, const_null(s, t))
		name(s, v, s.u.syms[g].name)
		s.sym_ids[g] = v
	}
}

// The uniform block: the inputs' members, then the user uniforms.
@(private)
declare_uniform_block :: proc(s: ^Spv, row: Interface_Var) {
	c := s.checked
	members := make([dynamic]Block_Member, context.temp_allocator)
	if c.header {
		for m in builtin_members() {
			append(&members, m)
		}
	}
	first_user := len(members)
	for id in c.uniforms {
		sym := s.u.syms[id]
		append(&members, Block_Member{sym.name, ty(s.u, sym.type), sym.index})
	}
	member_types := make([dynamic]u32)
	for m in members {
		append(&member_types, block_member_type(s, m.type))
	}
	s.member_types = member_types[:]
	s.block_type = new_id(s)
	words := make([dynamic]u32, context.temp_allocator)
	append(&words, s.block_type)
	append(&words, ..member_types[:])
	inst_words(&s.globals, .Type_Struct, words[:])
	inst(&s.annotations, .Decorate, s.block_type, DEC_BLOCK)
	name(s, s.block_type, "Oksl_Uniforms")
	for m, i in members {
		inst(&s.annotations, .Member_Decorate, s.block_type, u32(i), DEC_OFFSET, u32(m.offset))
		if m.type.cols > 0 {
			inst(&s.annotations, .Member_Decorate, s.block_type, u32(i), DEC_COL_MAJOR)
			inst(&s.annotations, .Member_Decorate, s.block_type, u32(i), DEC_MATRIX_STRIDE, 16)
		}
		inst_str(&s.debug, .Member_Name, {s.block_type, u32(i)}, m.name)
	}
	for id, i in c.uniforms {
		s.members[id] = u32(first_user + i)
	}
	s.block_var = new_id(s)
	inst(&s.globals, .Variable, ptr_type(s, SC_UNIFORM, s.block_type), s.block_var, SC_UNIFORM)
	inst(&s.annotations, .Decorate, s.block_var, DEC_DESCRIPTOR_SET, u32(row.set))
	inst(&s.annotations, .Decorate, s.block_var, DEC_BINDING, u32(row.binding))
	name(s, s.block_var, "u")
}

// ---- functions -------------------------------------------------------------------------

@(private)
begin_function :: proc(s: ^Spv) {
	s.vars = make([dynamic]u32, context.temp_allocator)
	s.body = make([dynamic]u32, context.temp_allocator)
	s.terminated = false
	clear(&s.loops)
}

// Appends the function's first block (variables first) and body.
@(private)
end_function :: proc(s: ^Spv, header: []u32, first_label: u32) {
	append(&s.functions, ..header)
	inst(&s.functions, .Label, first_label)
	append(&s.functions, ..s.vars[:])
	append(&s.functions, ..s.body[:])
	inst(&s.functions, .Function_End)
}

@(private)
local_var :: proc(s: ^Spv, t: Type, debug_name := "") -> u32 {
	v := new_id(s)
	inst(&s.vars, .Variable, ptr_type(s, SC_FUNCTION, type_id(s, t)), v, SC_FUNCTION)
	if debug_name != "" {
		name(s, v, debug_name)
	}
	return v
}

@(private)
label :: proc(s: ^Spv, id: u32) {
	inst(&s.body, .Label, id)
	s.cur_block = id
	s.terminated = false
}

@(private)
terminate :: proc(s: ^Spv, op: Op, operands: ..u32) {
	inst_words(&s.body, op, operands)
	s.terminated = true
}

@(private)
emit_function :: proc(s: ^Spv, fid: Func_Id) {
	u := s.u
	begin_function(s)
	f := u.funcs[fid]
	ps := fparams(u, fid)
	header := make([dynamic]u32, context.temp_allocator)
	ret_t := ty(u, f.ret)
	ret := type_id(s, ret_t)
	param_types := make([dynamic]u32, context.temp_allocator)
	for p in ps {
		pt := ty(u, p.type)
		switch {
		case pt.base == .Sampler:
			append(&param_types, ptr_type(s, SC_UNIFORM_CONSTANT, sampled_image_type(s)))
		case p.qual != .In:
			append(&param_types, ptr_type(s, SC_FUNCTION, type_id(s, pt)))
		case:
			append(&param_types, type_id(s, pt))
		}
	}
	id := s.func_ids[fid]
	inst(&header, .Function, ret, id, 0, fn_type(s, ret, param_types[:]))
	name(s, id, f.name)
	param_ids := make([]u32, len(ps), context.temp_allocator)
	for &pid, i in param_ids {
		pid = new_id(s)
		inst(&header, .Function_Parameter, param_types[i], pid)
	}
	first := new_id(s)
	s.cur_block = first
	for p, i in ps {
		if p.sym == 0 {
			continue
		}
		pt := ty(u, p.type)
		if pt.base == .Sampler || p.qual != .In {
			s.sym_ids[p.sym] = param_ids[i]
			continue
		}
		// `in` parameters are writable copies.
		v := local_var(s, pt, p.name)
		inst(&s.body, .Store, v, param_ids[i])
		s.sym_ids[p.sym] = v
	}
	emit_block(s, u.func_decls[f.decl].body)
	if !s.terminated {
		if ret_t == T_VOID {
			inst(&s.body, .Return)
		} else {
			inst(&s.body, .Return_Value, const_null(s, ret))
		}
	}
	end_function(s, header[:], first)
}

// ---- expression helpers ----------------------------------------------------------------

@(private)
unop :: proc(s: ^Spv, op: Op, t: u32, a: u32) -> u32 {
	id := new_id(s)
	inst(&s.body, op, t, id, a)
	return id
}

@(private)
binop_raw :: proc(s: ^Spv, op: Op, t: u32, a, b: u32) -> u32 {
	id := new_id(s)
	inst(&s.body, op, t, id, a, b)
	return id
}

@(private)
ext :: proc(s: ^Spv, t: u32, op: Glsl, args: ..u32) -> u32 {
	id := new_id(s)
	words := make([dynamic]u32, context.temp_allocator)
	append(&words, t, id, s.glsl, u32(op))
	append(&words, ..args)
	inst_words(&s.body, .Ext_Inst, words[:])
	return id
}

@(private)
extract :: proc(s: ^Spv, t: u32, v: u32, indices: ..u32) -> u32 {
	id := new_id(s)
	words := make([dynamic]u32, context.temp_allocator)
	append(&words, t, id, v)
	append(&words, ..indices)
	inst_words(&s.body, .Composite_Extract, words[:])
	return id
}

@(private)
construct :: proc(s: ^Spv, t: u32, parts: ..u32) -> u32 {
	id := new_id(s)
	words := make([dynamic]u32, context.temp_allocator)
	append(&words, t, id)
	append(&words, ..parts)
	inst_words(&s.body, .Composite_Construct, words[:])
	return id
}

// Broadcasts a scalar value to `t` (a vector) when needed.
@(private)
splat :: proc(s: ^Spv, v: u32, from, to: Type) -> u32 {
	if from == to || !is_vector(to) || !is_scalar(from) {
		return v
	}
	parts := make([]u32, int(to.rows), context.temp_allocator)
	for &p in parts {
		p = v
	}
	return construct(s, type_id(s, to), ..parts)
}

// Converts a scalar or vector between bases.
@(private)
convert :: proc(s: ^Spv, v: u32, from, to: Type) -> u32 {
	if from.base == to.base {
		return v
	}
	tt := type_id(s, to)
	if to.base == .Bool {
		zero := splat_const(s, from, 0)
		return binop_raw(s, SPV_BINOPS[.Not_Eq][from.base], tt, v, zero)
	}
	if from.base == .Bool {
		return select_consts(s, v, to, 1, 0)
	}
	if op := SPV_CONVERT[to.base][from.base]; op != Op(0) {
		return unop(s, op, tt, v)
	}
	return v
}

// Numeric conversions, [to][from]; bools compare with zero or select.
@(rodata)
SPV_CONVERT := #partial [Base][Base]Op {
	.Float = #partial {.Int = .Convert_S_To_F, .Uint = .Convert_U_To_F},
	.Int   = #partial {.Float = .Convert_F_To_S, .Uint = .Bitcast},
	.Uint  = #partial {.Float = .Convert_F_To_U, .Int = .Bitcast},
}

// A constant of type `t` (scalar or vector) with every component `v`.
@(private)
splat_const :: proc(s: ^Spv, t: Type, v: f64) -> u32 {
	c := const_of(s, t.base, v)
	if is_scalar(t) {
		return c
	}
	key := [3]u32{u32(t.base) | 0x100, transmute(u32)f32(v), u32(t.rows)}
	if id, ok := s.consts[key]; ok {
		return id
	}
	id := new_id(s)
	words := make([dynamic]u32, context.temp_allocator)
	append(&words, type_id(s, t), id)
	for _ in 0 ..< t.rows {
		append(&words, c)
	}
	inst_words(&s.globals, .Constant_Composite, words[:])
	s.consts[key] = id
	return id
}

@(private)
select_consts :: proc(s: ^Spv, cond: u32, t: Type, a, b: f64) -> u32 {
	id := new_id(s)
	inst(&s.body, .Select, type_id(s, t), id, cond, splat_const(s, t, a), splat_const(s, t, b))
	return id
}


@(private)
per_column :: proc(s: ^Spv, t: Type, a, b: u32, f: proc(s: ^Spv, ct: u32, a, b: u32) -> u32) -> u32 {
	ct := type_id(s, column_type(t))
	cols := make([]u32, int(t.cols), context.temp_allocator)
	for &col, i in cols {
		ca := extract(s, ct, a, u32(i))
		cb: u32 = 0
		if b != 0 {
			cb = extract(s, ct, b, u32(i))
		}
		col = f(s, ct, ca, cb)
	}
	return construct(s, type_id(s, t), ..cols)
}

// `a op b` with operands already of a common base; scalars broadcast.
@(private)
binop :: proc(s: ^Spv, op: Token_Kind, a: u32, ta: Type, b: u32, tb: Type, result: Type) -> u32 {
	rt := type_id(s, result)
	// Matrices.
	if is_matrix(ta) || is_matrix(tb) {
		#partial switch op {
		case .Star:
			switch {
			case is_matrix(ta) && is_matrix(tb):
				return binop_raw(s, .Matrix_Times_Matrix, rt, a, b)
			case is_matrix(ta) && is_vector(tb):
				return binop_raw(s, .Matrix_Times_Vector, rt, a, b)
			case is_vector(ta) && is_matrix(tb):
				return binop_raw(s, .Vector_Times_Matrix, rt, a, b)
			case is_matrix(ta) && is_scalar(tb):
				return binop_raw(s, .Matrix_Times_Scalar, rt, a, b)
			case is_scalar(ta) && is_matrix(tb):
				return binop_raw(s, .Matrix_Times_Scalar, rt, b, a)
			}
		}
		mt := ta if is_matrix(ta) else tb
		ct := column_type(mt)
		ctid := type_id(s, ct)
		if op == .Eq_Eq || op == .Not_Eq {
			// Column by column: == holds when every column's does, != when
			// any column's does.
			eq := op == .Eq_Eq
			bt := type_id(s, with_base(ct, .Bool))
			r: u32
			for i in 0 ..< u32(mt.cols) {
				cmp := binop_raw(s, SPV_BINOPS[op][.Float], bt, extract(s, ctid, a, i), extract(s, ctid, b, i))
				col := unop(s, .All if eq else .Any, rt, cmp)
				r = col if i == 0 else binop_raw(s, .Logical_And if eq else .Logical_Or, rt, r, col)
			}
			return r
		}
		// Component-wise: each column with a column or a broadcast scalar.
		cols := make([]u32, int(mt.cols), context.temp_allocator)
		for &col, i in cols {
			ca := extract(s, ctid, a, u32(i)) if is_matrix(ta) else splat(s, a, ta, ct)
			cb := extract(s, ctid, b, u32(i)) if is_matrix(tb) else splat(s, b, tb, ct)
			col = binop_raw(s, SPV_BINOPS[op][.Float], ctid, ca, cb)
		}
		return construct(s, rt, ..cols)
	}
	opc := SPV_BINOPS[op][ta.base]
	#partial switch op {
	case .Eq_Eq, .Not_Eq:
		if is_vector(ta) {
			bv := binop_raw(s, opc, type_id(s, with_base(ta, .Bool)), a, b)
			return unop(s, .All if op == .Eq_Eq else .Any, rt, bv)
		}
		return binop_raw(s, opc, rt, a, b)
	case .Shl, .Shr:
		return binop_raw(s, opc, rt, a, splat(s, b, tb, with_base(result, tb.base)))
	}
	if ta.base == .Float && op == .Star && is_vector(ta) != is_vector(tb) {
		if is_vector(ta) {
			return binop_raw(s, .Vector_Times_Scalar, rt, a, b)
		}
		return binop_raw(s, .Vector_Times_Scalar, rt, b, a)
	}
	return binop_raw(s, opc, rt, splat(s, a, ta, result), splat(s, b, tb, result))
}

// The instruction for a binary operator, by its left operand's base (the
// checker gives both the same one, except a shift's amount).
@(rodata)
SPV_BINOPS := #partial [Token_Kind][Base]Op {
	.Plus    = #partial {.Float = .F_Add, .Int = .I_Add, .Uint = .I_Add},
	.Minus   = #partial {.Float = .F_Sub, .Int = .I_Sub, .Uint = .I_Sub},
	.Star    = #partial {.Float = .F_Mul, .Int = .I_Mul, .Uint = .I_Mul},
	.Slash   = #partial {.Float = .F_Div, .Int = .S_Div, .Uint = .U_Div},
	.Percent = #partial {.Int = .S_Rem, .Uint = .U_Mod},
	.Amp     = #partial {.Int = .Bitwise_And, .Uint = .Bitwise_And},
	.Pipe    = #partial {.Int = .Bitwise_Or, .Uint = .Bitwise_Or},
	.Caret   = #partial {.Int = .Bitwise_Xor, .Uint = .Bitwise_Xor},
	.Shl     = #partial {.Int = .Shift_Left_Logical, .Uint = .Shift_Left_Logical},
	.Shr     = #partial {.Int = .Shift_Right_Arithmetic, .Uint = .Shift_Right_Logical},
	.Lt      = #partial {.Float = .F_Ord_Less_Than, .Int = .S_Less_Than, .Uint = .U_Less_Than},
	.Gt      = #partial {.Float = .F_Ord_Greater_Than, .Int = .S_Greater_Than, .Uint = .U_Greater_Than},
	.Lt_Eq   = #partial {.Float = .F_Ord_Less_Than_Equal, .Int = .S_Less_Than_Equal, .Uint = .U_Less_Than_Equal},
	.Gt_Eq   = #partial {.Float = .F_Ord_Greater_Than_Equal, .Int = .S_Greater_Than_Equal, .Uint = .U_Greater_Than_Equal},
	.Eq_Eq   = #partial {.Float = .F_Ord_Equal, .Int = .I_Equal, .Uint = .I_Equal, .Bool = .Logical_Equal},
	.Not_Eq  = #partial {.Float = .F_Unord_Not_Equal, .Int = .I_Not_Equal, .Uint = .I_Not_Equal, .Bool = .Logical_Not_Equal},
	.Xor_Xor = #partial {.Bool = .Logical_Not_Equal},
}

