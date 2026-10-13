package oksl

import "base:runtime"
import "core:fmt"
import "core:strings"

Allocator :: runtime.Allocator

Base :: enum u8 {
	Void,
	Bool,
	Int,
	Uint,
	Float,
	Sampler,
	Struct,
}

// A type. Scalars have `rows == 1`, vectors `rows` 2..4; matrices
// are float with `cols` 2..4 columns of `rows` components. `array` > 0 is
// an array of that many elements of the rest of the type.
Type :: struct {
	base:   Base,
	rows:   u8,
	cols:   u8,
	array:  i32,
	strukt: i32,
}

T_VOID :: Type{base = .Void}
T_BOOL :: Type{base = .Bool, rows = 1}
T_INT :: Type{base = .Int, rows = 1}
T_UINT :: Type{base = .Uint, rows = 1}
T_FLOAT :: Type{base = .Float, rows = 1}
T_SAMPLER :: Type{base = .Sampler, rows = 1}
T_ERROR :: Type{base = .Void, rows = 255}

vec_type :: proc "contextless" (base: Base, n: int) -> Type {
	return {base = base, rows = u8(n)}
}

mat_type :: proc "contextless" (cols, rows: int) -> Type {
	return {base = .Float, rows = u8(rows), cols = u8(cols)}
}

is_error :: #force_inline proc "contextless" (t: Type) -> bool {
	return t.rows == 255
}

is_array :: #force_inline proc "contextless" (t: Type) -> bool {
	return t.array > 0
}

is_matrix :: #force_inline proc "contextless" (t: Type) -> bool {
	return t.cols > 0 && t.array == 0
}

is_vector :: #force_inline proc "contextless" (t: Type) -> bool {
	return t.rows > 1 && t.cols == 0 && t.array == 0 && t.base != .Struct
}

is_scalar :: #force_inline proc "contextless" (t: Type) -> bool {
	return t.rows == 1 && t.cols == 0 && t.array == 0 && t.base >= .Bool && t.base <= .Float
}

// Scalar or vector of a numeric or bool base (no matrices, arrays, structs).
is_sv :: #force_inline proc "contextless" (t: Type) -> bool {
	return t.cols == 0 && t.array == 0 && t.base >= .Bool && t.base <= .Float
}

is_numeric_base :: #force_inline proc "contextless" (b: Base) -> bool {
	return b == .Int || b == .Uint || b == .Float
}

is_integer_base :: #force_inline proc "contextless" (b: Base) -> bool {
	return b == .Int || b == .Uint
}

elem_type :: proc "contextless" (t: Type) -> Type {
	e := t
	e.array = 0
	return e
}

column_type :: proc "contextless" (t: Type) -> Type {
	return {base = .Float, rows = t.rows}
}

scalar_of :: proc "contextless" (t: Type) -> Type {
	return {base = t.base, rows = 1}
}

with_base :: proc "contextless" (t: Type, b: Base) -> Type {
	r := t
	r.base = b
	return r
}

// Number of scalar components (flattened, arrays and structs included).
component_count :: proc(u: ^Unit, t: Type) -> int {
	n := 0
	switch t.base {
	case .Void, .Sampler:
		n = 0 if t.base == .Void else 1
	case .Struct:
		for f in struct_fields(u, t.strukt) {
			n += component_count(u, u.types[f.type])
		}
	case .Bool, .Int, .Uint, .Float:
		n = int(t.rows) * max(int(t.cols), 1)
	}
	if t.array > 0 {
		n *= int(t.array)
	}
	return n
}

type_name :: proc(u: ^Unit, t: Type, allocator := context.temp_allocator) -> string {
	if is_error(t) {
		return "<error>"
	}
	b := strings.builder_make(allocator)
	switch t.base {
	case .Void:
		strings.write_string(&b, "void")
	case .Sampler:
		strings.write_string(&b, "sampler2D" if u.dialect == .GLSL else "shader")
	case .Struct:
		strings.write_string(&b, u.structs[t.strukt].name if int(t.strukt) < len(u.structs) else "struct")
	case .Bool, .Int, .Uint, .Float:
		if u.dialect == .GLSL && (t.cols > 0 || t.rows > 1) {
			// vec3, ivec2, mat2x4.
			if t.cols > 0 {
				fmt.sbprintf(&b, "mat%dx%d", t.cols, t.rows)
			} else {
				prefixes := [4]string{"bvec", "ivec", "uvec", "vec"}
				fmt.sbprintf(&b, "%s%d", prefixes[int(t.base) - 1], t.rows)
			}
			break
		}
		names := [4]string{"bool", "int", "uint", "float"}
		strings.write_string(&b, names[int(t.base) - 1])
		if t.cols > 0 {
			fmt.sbprintf(&b, "%dx%d", t.cols, t.rows)
		} else if t.rows > 1 {
			fmt.sbprintf(&b, "%d", t.rows)
		}
	}
	if t.array > 0 {
		fmt.sbprintf(&b, "[%d]", t.array)
	}
	return strings.to_string(b)
}

// The type a built-in type name of `dialect` denotes: the GLSL spellings
// (`vec3`, `ivec2`, `mat2x4`, ...) in both, and in SkSL also `float3`,
// `half4`, `float2x3`, `shader`, ... (`half` is `float`: oksl computes
// everything in 32 bits); GLSL has `sampler2D`.
builtin_type :: proc(name: string, dialect := Dialect.SkSL) -> (Type, bool) {
	has := DIALECT_RULES[dialect].has
	switch name {
	case "void":
		return T_VOID, true
	case "bool":
		return T_BOOL, true
	case "int":
		return T_INT, true
	case "uint":
		return T_UINT, true
	case "float":
		return T_FLOAT, true
	case "short":
		return T_INT, .Sksl_Types in has
	case "ushort":
		return T_UINT, .Sksl_Types in has
	case "half":
		return T_FLOAT, .Sksl_Types in has
	case "shader":
		return T_SAMPLER, .Children in has
	case "sampler2D":
		return T_SAMPLER, .Samplers in has
	}
	if len(name) < 4 {
		return {}, false
	}
	digit :: proc(c: u8) -> int {
		n := int(c) - '0'
		return n if n >= 2 && n <= 4 else 0
	}
	// SkSL: floatN, halfN, intN, uintN, boolN, shortN, ushortN; floatNxM, halfNxM.
	sksl := .Sksl_Types in has
	prefixes := [?]struct {
		p: string,
		b: Base,
	}{{"float", .Float}, {"half", .Float}, {"int", .Int}, {"uint", .Uint}, {"bool", .Bool}, {"short", .Int}, {"ushort", .Uint}}
	for p in prefixes {
		if !sksl || !strings.has_prefix(name, p.p) {
			continue
		}
		rest := name[len(p.p):]
		if len(rest) == 1 && digit(rest[0]) != 0 {
			return vec_type(p.b, digit(rest[0])), true
		}
		if p.b == .Float && len(rest) == 3 && rest[1] == 'x' && digit(rest[0]) != 0 && digit(rest[2]) != 0 {
			return mat_type(digit(rest[0]), digit(rest[2])), true
		}
	}
	// GLSL spellings: vecN, ivecN, uvecN, bvecN, matN, matNxM.
	glsl := [?]struct {
		p: string,
		b: Base,
	}{{"bvec", .Bool}, {"ivec", .Int}, {"uvec", .Uint}, {"vec", .Float}}
	for p in glsl {
		if len(name) == len(p.p) + 1 && strings.has_prefix(name, p.p) && digit(name[len(name) - 1]) != 0 {
			return vec_type(p.b, digit(name[len(name) - 1])), true
		}
	}
	if strings.has_prefix(name, "mat") {
		rest := name[3:]
		if len(rest) == 1 && digit(rest[0]) != 0 {
			return mat_type(digit(rest[0]), digit(rest[0])), true
		} else if len(rest) == 3 && rest[1] == 'x' && digit(rest[0]) != 0 && digit(rest[2]) != 0 {
			return mat_type(digit(rest[0]), digit(rest[2])), true
		}
	}
	return {}, false
}

// ---- std140 layout ----------------------------------------------------------------

// Base alignment and size of a uniform under std140.
std140 :: proc(t: Type) -> (align, size: int) {
	if t.array > 0 {
		e := elem_type(t)
		_, es := std140(e)
		stride := round_up(es, 16)
		return 16, stride * int(t.array)
	}
	if t.cols > 0 {
		return 16, 16 * int(t.cols)
	}
	switch t.rows {
	case 1:
		return 4, 4
	case 2:
		return 8, 8
	case 3:
		return 16, 12
	}
	return 16, 16
}

// Byte stride between elements of a std140 uniform array.
std140_array_stride :: proc(t: Type) -> int {
	_, es := std140(elem_type(t))
	return round_up(es, 16)
}

round_up :: proc "contextless" (v, a: int) -> int {
	return (v + a - 1) / a * a
}
