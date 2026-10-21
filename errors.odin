package oksl

import "core:fmt"
import "core:strings"

// Diagnostics are plain values: a code and its operands. Text is made only
// at the edge (`error_message`, `format_errors`) from ERROR_TEXT, where
// `{name}`, `{a}`, `{b}`, `{c}`, `{n}` and `{m}` stand for the fields and
// `{q:name}`, `{q:a}`, `{q:b}` for quoted ones.

Error_Code :: enum u8 {
	None,
	// Tokens and the preprocessor.
	Unexpected_Character,
	No_Preprocessor,
	Unknown_Directive,
	Macro_Name_Expected,
	Macro_Params,
	Macro_Arg_Count,
	Macro_Unterminated,
	Paste_Invalid,
	If_Without_If,
	Else_After_Else,
	If_Unterminated,
	If_Expression,
	If_Undefined,
	If_Divide,
	Line_Number,
	Error_Directive,
	// Syntax.
	Expected,
	Expected_Name,
	Expected_Type,
	Expected_Expression,
	Expected_Body,
	Expected_While,
	Global_In_Out,
	Varying_Keyword,
	Const_Function,
	Anonymous_Struct,
	Local_Struct,
	Unknown_Type,
	// Dialect rules (DIALECT_RULES says which applies).
	Sksl_Sampler2D,
	Sksl_Child_Kind,
	Sksl_Texture_Call,
	Sksl_Discard,
	Sksl_Main_Image,
	Glsl_Shader_Type,
	Glsl_Eval,
	// Entry points.
	Entry_Signature,
	No_Entry,
	Entry_No_Body,
	No_Main_Image,
	Main_Image_Signature,
	Main_Reserved,
	No_Void_Main,
	Stage_Unsupported,
	Fragment_Only,
	Void_Main_Signature,
	// Declarations.
	Already_Declared,
	Already_Declared_At,
	Already_Defined_At,
	Input_Redeclared,
	Array_Of_Arrays,
	Array_Size_Constant,
	Array_Size_Positive,
	Struct_Redeclared,
	Struct_Sampler,
	Init_Mismatch,
	Uniform_Initializer,
	Uniform_Kind,
	Sampler_Array,
	Sampler_Not_Uniform,
	Sampler_Param,
	Too_Many_Channels,
	Const_Needs_Init,
	Unsized_Needs_Init,
	Return_Type_Redeclared,
	// Stage variables (in, out).
	Stage_Var_Type,
	Stage_Var_Initializer,
	Flat_Required,
	Location_Required,
	Location_Taken,
	Location_Range,
	// Linking a vertex and a fragment shader.
	Varying_Unwritten,
	Varying_Type,
	Varying_Flat,
	Varying_Location,
	Recursion,
	Never_Defined,
	// Statements.
	Break_Outside,
	Continue_Outside,
	Must_Return,
	Returns_Void,
	Return_Mismatch,
	Condition_Type,
	Switch_Type,
	Case_Constant,
	Case_Duplicate,
	Default_Duplicate,
	Case_Outside,
	Statement_Before_Case,
	// Expressions.
	Undeclared,
	Is_Function,
	Ternary_Mismatch,
	Unary_Number,
	Not_Bool,
	Tilde_Integer,
	Inc_Number,
	Assign_Mismatch,
	Compound_Changes_Left,
	Compound_Result,
	Index_Type,
	Not_Indexable,
	Index_Range,
	Length_Type,
	No_Field,
	Binary_Undefined,
	Compare_Unsupported,
	Compare_Vectors,
	Bad_Swizzle,
	Swizzle_Mixed,
	Swizzle_Range,
	Swizzle_Repeat,
	Read_Only,
	Is_Const,
	Not_Assignable,
	// Calls and constructors.
	Unknown_Function,
	Arg_Count,
	Ambiguous_Call,
	No_Overload,
	Out_Arg_Type,
	Eval_Child,
	Eval_Args,
	Eval_Coord,
	Texture_Sampler,
	Texture_Channel,
	Not_Constructible,
	Array_Ctor_Count,
	Array_Element_Type,
	Struct_Ctor_Count,
	Struct_Field_Type,
	Ctor_No_Args,
	Ctor_Arg_Type,
	Ctor_Too_Few,
	Ctor_Matrix_Args,
	Ctor_Too_Many,
	Ctor_Components,
	// Programs (Common, Buffer A-D, Image).
	No_Image_Pass,
	Channel_No_Buffer,
	Pass_Count,
	Pass_Output,
	Pass_Reference,
}

// A diagnostic at `pos` (the user's line and column: macro expansions keep
// the position of the macro's use). The operands depend on `code`: names
// and type names, then numbers.
Error :: struct {
	pos:     Pos,
	code:    Error_Code,
	name:    string,
	a, b, c: string,
	n, m:    i64,
}

@(rodata)
ERROR_TEXT := [Error_Code]string {
	.None                  = "",
	.Unexpected_Character  = "unexpected character {q:a}",
	.No_Preprocessor       = "SkSL has no preprocessor: #{name} is not supported",
	.Unknown_Directive     = "unknown directive #{name}",
	.Macro_Name_Expected   = "#{name} needs a macro name",
	.Macro_Params          = "bad parameter list in the definition of {name}",
	.Macro_Arg_Count       = "macro {name} takes {n} arguments, got {m}",
	.Macro_Unterminated    = "unterminated argument list for macro {name}",
	.Paste_Invalid         = "pasting {q:a} and {q:b} does not give a valid token",
	.If_Without_If         = "#{name} without #if",
	.Else_After_Else       = "#{name} after #else",
	.If_Unterminated       = "#{name} without #endif",
	.If_Expression         = "invalid #{name} expression near {q:a}",
	.If_Undefined          = "{q:a} is not a macro (#{name} expressions see only macros)",
	.If_Divide             = "division by zero in #{name}",
	.Line_Number           = "#line needs a line number",
	.Error_Directive       = "#error {a}",
	.Expected              = "expected {a}, found {q:b}",
	.Expected_Name         = "expected a name after {q:a}",
	.Expected_Type         = "expected a type, found {q:a}",
	.Expected_Expression   = "expected an expression, found {q:a}",
	.Expected_Body         = "expected '{' or ';' after the parameters of {name}",
	.Expected_While        = "expected 'while' after do-block",
	.Global_In_Out         = "global '{name}' variables are not supported; the entry point takes the coordinate and returns the colour",
	.Varying_Keyword       = "{name} variables are GLSL ES 1.0; declare in and out variables",
	.Const_Function        = "a function can not be const, uniform, in or out",
	.Anonymous_Struct      = "anonymous structs are not supported",
	.Local_Struct          = "declare structs at file scope",
	.Unknown_Type          = "unknown type {q:name}",
	.Sksl_Sampler2D        = "sampler2D is not SkSL; declare `uniform shader name;` and sample it with name.eval(coord) in its pixels",
	.Sksl_Child_Kind       = "only `uniform shader` children are supported, not {name}",
	.Sksl_Texture_Call     = "{name}() is not SkSL; declare `uniform shader name;` and sample it with name.eval(coord) in its pixels",
	.Sksl_Discard          = "discard is not allowed in SkSL shaders; return half4(0) instead",
	.Sksl_Main_Image       = "mainImage is Shadertoy's entry point; compile Shadertoy code through compile_shadertoy, or define half4 main(float2 coord)",
	.Glsl_Shader_Type      = "`shader` children are SkSL; Shadertoy code samples iChannel0..3 (or a `uniform sampler2D`) with texture(channel, uv)",
	.Glsl_Eval             = ".eval is SkSL; Shadertoy code samples with texture(channel, uv), uv in 0..1",
	.Entry_Signature       = "the entry point must be half4 main(float2 coord)",
	.No_Entry              = "no entry point: define half4 main(float2 coord)",
	.Entry_No_Body         = "the entry point {name} has no body",
	.No_Main_Image         = "no entry point: define void mainImage(out vec4 fragColor, in vec2 fragCoord)",
	.Main_Image_Signature  = "mainImage must be void mainImage(out vec4 fragColor, in vec2 fragCoord)",
	.Main_Reserved         = "main is reserved in Shadertoy code; the entry point is mainImage",
	.Stage_Unsupported     = "{a} source can not be a {b} shader",
	.Fragment_Only         = "{name} is only available in fragment shaders",
	.No_Void_Main          = "no entry point: define void main()",
	.Void_Main_Signature   = "the entry point must be void main()",
	.Already_Declared      = "{name} is already declared",
	.Already_Declared_At   = "{name} is already declared at {n}:{m}",
	.Already_Defined_At    = "{name} is already defined at {n}:{m}",
	.Input_Redeclared      = "{name} is a Shadertoy input of type {a}",
	.Array_Of_Arrays       = "arrays of arrays are not supported",
	.Array_Size_Constant   = "array size must be a constant integer expression",
	.Array_Size_Positive   = "array size must be positive",
	.Struct_Redeclared     = "struct {name} is already declared",
	.Struct_Sampler        = "structs can not hold {a} values",
	.Init_Mismatch         = "can not initialize {a} {name} with {b}",
	.Uniform_Initializer   = "uniforms can not have initializers",
	.Uniform_Kind          = "uniform {name}: struct and unsized uniforms are not supported",
	.Sampler_Array         = "{a} arrays are not supported",
	.Sampler_Not_Uniform   = "a {a} must be a uniform",
	.Sampler_Param         = "shaders can not be parameters; call .eval on the `uniform shader` directly",
	.Too_Many_Channels     = "at most {n} {a} uniforms are supported",
	.Const_Needs_Init      = "const {name} needs an initializer",
	.Unsized_Needs_Init    = "unsized array {name} needs an initializer",
	.Return_Type_Redeclared = "{name} is redeclared with a different return type",
	.Stage_Var_Type        = "{a} variable {name} can not be {b}; use a float, int or uint scalar or vector",
	.Stage_Var_Initializer = "{a} variable {name} can not have an initializer",
	.Flat_Required         = "integer {a} {name} needs the flat qualifier",
	.Location_Required     = "with several outputs, {name} needs layout(location = N)",
	.Location_Taken        = "{a} location {n} is already used by {name}",
	.Location_Range        = "{a} location {n} is out of range (0..{m})",
	.Varying_Unwritten     = "fragment input {name} is not an output of the vertex shader",
	.Varying_Type          = "{name} is {a} in the vertex shader and {b} in the fragment shader",
	.Varying_Flat          = "{name} must be flat in both shaders or in neither",
	.Varying_Location      = "{name} is at location {n} in the fragment shader and {m} in the vertex shader",
	.Recursion             = "recursion is not supported: {a}",
	.Never_Defined         = "{name} is declared but never defined",
	.Break_Outside         = "break outside a loop or switch",
	.Continue_Outside      = "continue outside a loop",
	.Must_Return           = "{name} must return a {a}",
	.Returns_Void          = "{name} returns void",
	.Return_Mismatch       = "{name} returns {a}, not {b}",
	.Condition_Type        = "condition must be a bool, not {a}",
	.Switch_Type           = "switch needs an int or uint, not {a}",
	.Case_Constant         = "case label must be a constant integer expression",
	.Case_Duplicate        = "case {n} is already handled",
	.Default_Duplicate     = "a switch has only one default label",
	.Case_Outside          = "{name} label outside a switch",
	.Statement_Before_Case = "statements in a switch must come after a case or default label",
	.Undeclared            = "undeclared identifier {q:name}",
	.Is_Function           = "{name} is a function; call it with ()",
	.Ternary_Mismatch      = "the branches of ?: have different types ({a} and {b})",
	.Unary_Number          = "unary {name} needs a number, not {a}",
	.Not_Bool              = "! needs a bool, not {a} (use not() for vectors)",
	.Tilde_Integer         = "~ needs an integer, not {a}",
	.Inc_Number            = "++/-- needs a number, not {a}",
	.Assign_Mismatch       = "can not assign {a} to {b}",
	.Compound_Changes_Left = "{a} {name} {b} would change the type of the left side",
	.Compound_Result       = "{a} {name} {b} gives {c}, which can not be stored in {a}",
	.Index_Type            = "index must be an int, not {a}",
	.Not_Indexable         = "{a} can not be indexed",
	.Index_Range           = "index {n} is out of range for {a}",
	.Length_Type           = "length() needs an array, vector or matrix, not {a}",
	.No_Field              = "{a} has no field {q:name}",
	.Binary_Undefined      = "{a} {name} {b} is not defined",
	.Compare_Unsupported   = "comparing {a} values is not supported",
	.Compare_Vectors       = "{name} compares scalars; use lessThan() and friends for vectors",
	.Bad_Swizzle           = "bad swizzle {q:name}",
	.Swizzle_Mixed         = "bad swizzle {q:name} (mixed component sets)",
	.Swizzle_Range         = "swizzle {q:name} reads past the {n} components of {a}",
	.Swizzle_Repeat        = "swizzle {q:name} repeats a component and can not be assigned",
	.Read_Only             = "{name} is read-only",
	.Is_Const              = "{name} is const",
	.Not_Assignable        = "this expression can not be assigned to",
	.Unknown_Function      = "unknown function {q:name}",
	.Arg_Count             = "{name} takes {n} arguments, got {m}",
	.Ambiguous_Call        = "call to {name}({a}) is ambiguous",
	.No_Overload           = "no overload of {name} takes ({a})",
	.Out_Arg_Type          = "out argument must be exactly {a}",
	.Eval_Child            = "eval needs a `uniform shader` child, not {a}",
	.Eval_Args             = "{name}.eval takes one float2 coordinate",
	.Eval_Coord            = "{name}.eval takes a float2 coordinate, not {a}",
	.Texture_Sampler       = "{name} needs a sampler2D, not {a}",
	.Texture_Channel       = "{name} needs iChannel0..3 or a uniform sampler2D itself, not a parameter",
	.Not_Constructible     = "{name} can not be constructed",
	.Array_Ctor_Count      = "{a} needs {n} elements, got {m}",
	.Array_Element_Type    = "array element must be {a}, not {b}",
	.Struct_Ctor_Count     = "{a} has {n} fields, got {m} values",
	.Struct_Field_Type     = "field {name} of {a} is {b}, not {c}",
	.Ctor_No_Args          = "{a} needs arguments",
	.Ctor_Arg_Type         = "{a} can not build a {b}",
	.Ctor_Too_Few          = "{a} has too few components for {b}",
	.Ctor_Matrix_Args      = "a {a} needs a scalar, a matrix or all of its components",
	.Ctor_Too_Many         = "too many arguments for {a}",
	.Ctor_Components       = "{a} needs {n} components, got {m}",
	.No_Image_Pass         = "a program needs an Image pass",
	.Channel_No_Buffer     = "{name} reads {a}, which the program does not have",
	.Pass_Count            = "a program has 1 to {n} passes, not {m}",
	.Pass_Output           = "the output pass {n} is not one of the {m} passes",
	.Pass_Reference        = "{name}'s channel {n} reads pass {m}, which {a}",
}

// The text of a diagnostic (without its position).
error_message :: proc(e: Error, allocator := context.allocator) -> string {
	b := strings.builder_make(allocator)
	write_error_message(&b, e)
	return strings.to_string(b)
}

// "line:col: message" lines for printing ("Buffer A:line:col: message" in
// a program's sources).
format_errors :: proc(errors: []Error, allocator := context.allocator) -> string {
	b := strings.builder_make(allocator)
	for e in errors {
		if e.pos.src != .Main {
			fmt.sbprintf(&b, "%s:", SOURCE_NAMES[e.pos.src])
		}
		fmt.sbprintf(&b, "%d:%d: ", e.pos.line, e.pos.col)
		write_error_message(&b, e)
		strings.write_byte(&b, '\n')
	}
	return strings.to_string(b)
}

@(private)
write_error_message :: proc(b: ^strings.Builder, e: Error) {
	text := ERROR_TEXT[e.code]
	for i := 0; i < len(text); {
		if text[i] != '{' {
			strings.write_byte(b, text[i])
			i += 1
			continue
		}
		end := strings.index_byte(text[i:], '}')
		if end < 0 {
			strings.write_string(b, text[i:])
			return
		}
		field := text[i + 1:i + end]
		quoted := strings.has_prefix(field, "q:")
		if quoted {
			field = field[2:]
		}
		value: string
		switch field {
		case "name":
			value = e.name
		case "a":
			value = e.a
		case "b":
			value = e.b
		case "c":
			value = e.c
		case "n":
			fmt.sbprintf(b, "%d", e.n)
			i += end + 1
			continue
		case "m":
			fmt.sbprintf(b, "%d", e.m)
			i += end + 1
			continue
		case:
			// Not a field (`'{'` in a message).
			strings.write_byte(b, '{')
			i += 1
			continue
		}
		if quoted {
			fmt.sbprintf(b, "%q", value)
		} else {
			strings.write_string(b, value)
		}
		i += end + 1
	}
}
