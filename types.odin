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
	Image,
}

// Storage image texel formats (GLSL ES 3.10 layout qualifiers).
Image_Format :: enum u8 {
	None,
	Rgba8,
	Rgba16f,
	Rgba32f,
	R32f,
	R32ui,
	R32i,
}

@(rodata)
IMAGE_FORMAT_NAMES := [Image_Format]string {
	.None    = "",
	.Rgba8   = "rgba8",
	.Rgba16f = "rgba16f",
	.Rgba32f = "rgba32f",
	.R32f    = "r32f",
	.R32ui   = "r32ui",
	.R32i    = "r32i",
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
// A sampler's rows: 1 for sampler2D (and SkSL's shader), 2 for
// sampler2DArray.
T_SAMPLER :: Type{base = .Sampler, rows = 1}
T_SAMPLER_ARRAY :: Type{base = .Sampler, rows = 2}
// An image's rows hold its texel Base (.Float, .Uint, .Int); `strukt` packs
// Image_Format in bits 0..7 and readonly/writeonly in bits 8..9.
T_IMAGE_2D :: Type{base = .Image, rows = u8(Base.Float)}
T_UIMAGE_2D :: Type{base = .Image, rows = u8(Base.Uint)}
T_IIMAGE_2D :: Type{base = .Image, rows = u8(Base.Int)}
T_ERROR :: Type{base = .Void, rows = 255}

image_texel_base :: #force_inline proc "contextless" (t: Type) -> Base {
	return Base(t.rows)
}

image_format_of :: #force_inline proc "contextless" (t: Type) -> Image_Format {
	return Image_Format(t.strukt & 0xff)
}

image_readonly :: #force_inline proc "contextless" (t: Type) -> bool {
	return (t.strukt & 0x100) != 0
}

image_writeonly :: #force_inline proc "contextless" (t: Type) -> bool {
	return (t.strukt & 0x200) != 0
}

with_image_qual :: proc "contextless" (t: Type, format: Image_Format, readonly, writeonly: bool) -> Type {
	r := t
	r.strukt = i32(format) | (0x100 if readonly else 0) | (0x200 if writeonly else 0)
	return r
}

image_format_from_name :: proc "contextless" (name: string) -> (Image_Format, bool) {
	switch name {
	case "rgba8":
		return .Rgba8, true
	case "rgba16f":
		return .Rgba16f, true
	case "rgba32f":
		return .Rgba32f, true
	case "r32f":
		return .R32f, true
	case "r32ui":
		return .R32ui, true
	case "r32i":
		return .R32i, true
	}
	return .None, false
}

image_format_base :: proc "contextless" (f: Image_Format) -> Base {
	#partial switch f {
	case .R32ui:
		return .Uint
	case .R32i:
		return .Int
	}
	return .Float
}

image_base_name :: proc "contextless" (t: Type) -> string {
	#partial switch Base(t.rows) {
	case .Uint:
		return "uimage2D"
	case .Int:
		return "iimage2D"
	}
	return "image2D"
}

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
	return t.base == .Float && t.cols > 0 && t.array == 0
}

is_vector :: #force_inline proc "contextless" (t: Type) -> bool {
	return t.rows > 1 && t.cols == 0 && t.array == 0 && t.base >= .Bool && t.base <= .Float
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
	case .Void, .Sampler, .Image:
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
		strings.write_string(&b, "sampler2DArray" if t.rows == 2 else ("sampler2D" if u.dialect == .GLSL else "shader"))
	case .Image:
		if f := image_format_of(t); f != .None {
			fmt.sbprintf(&b, "layout(%s) ", IMAGE_FORMAT_NAMES[f])
		}
		if image_readonly(t) {
			strings.write_string(&b, "readonly ")
		}
		if image_writeonly(t) {
			strings.write_string(&b, "writeonly ")
		}
		strings.write_string(&b, image_base_name(t))
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
	case "sampler2DArray":
		return T_SAMPLER_ARRAY, .Samplers in has
	case "image2D":
		return T_IMAGE_2D, .Storage_Images in has
	case "uimage2D":
		return T_UIMAGE_2D, .Storage_Images in has
	case "iimage2D":
		return T_IIMAGE_2D, .Storage_Images in has
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

// A buffer element's std430 size and alignment in bytes, when it can be
// one: float, int and uint, their vectors, and structs of those (a field
// at the next multiple of its alignment, a struct's size a multiple of
// its largest). An array of `t` has element stride `std430_stride(u, t)`
// (`round_up(size, align)`, so vec3[] steps by 16 bytes while a struct's
// vec3 field takes 12).
std430 :: proc(u: ^Unit, t: Type) -> (size, align: int, ok: bool) {
	switch {
	case (t.base == .Float || t.base == .Int || t.base == .Uint) && t.cols == 0 && t.array == 0:
		n := int(t.rows)
		align = 4 if n == 1 else (8 if n == 2 else 16)
		return 4 * n, align, true
	case t.base == .Struct && t.array == 0:
		for f in struct_fields(u, t.strukt) {
			fs, fa := std430(u, ty(u, f.type)) or_return
			size = round_up(size, fa) + fs
			align = max(align, fa)
		}
		return round_up(size, align), align, align > 0
	}
	return
}

// Byte stride between elements of a std430 runtime-sized array `T[]`.
std430_stride :: proc(u: ^Unit, t: Type) -> int {
	size, align, _ := std430(u, t)
	return round_up(size, align) if align > 0 else 0
}

// Byte offset of field `field` inside struct `st` under std430 layout
// (0 when `st` is not a valid std430 buffer struct).
std430_field_offset :: proc(u: ^Unit, st: i32, field: i32) -> int {
	off := 0
	for f, i in struct_fields(u, st) {
		size, align, ok := std430(u, ty(u, f.type))
		if !ok {
			return 0
		}
		off = round_up(off, align)
		if i32(i) == field {
			return off
		}
		off += size
	}
	return off
}
