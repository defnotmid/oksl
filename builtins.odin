package oksl

Builtin :: enum u8 {
	None,
	// float -> float, component-wise
	Radians,
	Degrees,
	Sin,
	Cos,
	Tan,
	Asin,
	Acos,
	Atan,
	Sinh,
	Cosh,
	Tanh,
	Asinh,
	Acosh,
	Atanh,
	Exp,
	Log,
	Exp2,
	Log2,
	Sqrt,
	Inversesqrt,
	Floor,
	Ceil,
	Fract,
	Trunc,
	Round,
	Round_Even,
	Dfdx,
	Dfdy,
	Fwidth,
	Normalize,
	// float or integer, component-wise
	Abs,
	Sign,
	// two float operands
	Atan2,
	Pow,
	Mod,
	Step,
	// numeric
	Min,
	Max,
	Clamp,
	Mix,
	// mix(x, y, bvec)
	Mix_Bool,
	Smoothstep,
	Isnan,
	Isinf,
	Length,
	Distance,
	Dot,
	Cross,
	Faceforward,
	Reflect,
	Refract,
	Matrix_Comp_Mult,
	Outer_Product,
	Transpose,
	Determinant,
	Inverse,
	Less_Than,
	Less_Than_Equal,
	Greater_Than,
	Greater_Than_Equal,
	Equal,
	Not_Equal,
	Any,
	All,
	Not,
	// child.eval(coord), lowered by the checker (not callable by name)
	Texture_Lod,
	Float_Bits_To_Int,
	Float_Bits_To_Uint,
	Int_Bits_To_Float,
	Uint_Bits_To_Float,
}

// How a built-in's arguments are checked (and converted). Each rule runs
// after the argument count is checked; on failure the call is reported
// with the argument types as converted so far.
Builtin_Args :: enum u8 {
	None,
	// genFType: int arguments convert to float.
	Float,
	// Exactly a float scalar or vector (no conversion).
	Float_Exact,
	// float or int scalar or vector.
	Float_Or_Int,
	// The argument's scalar or vector of `Builtin_Info.base`.
	Exact_Base,
	// genFType arguments of one type; those in `broadcast` may instead be
	// scalars.
	Float_Same,
	// numeric (not bool) scalar or vector, the rest the same or scalar; an
	// integer first argument becomes float when another is float.
	Numeric_Same,
	// The last argument is genFType, the others (edges) equal each other
	// and are that type or scalar: step, smoothstep.
	Edges,
	Mix_Bool,
	Vec3,
	// refract(I, N, eta): genFType, genFType, float.
	Refract,
	Float_Vectors,
	Matrix,
	Matrix_Same,
	Square_Matrix,
	// lessThan and friends: numeric vectors of one type (unified).
	Vector_Ordered,
	// equal, notEqual: vectors of one type (unified).
	Vector_Equal,
	Bool_Vector,
	Texture_Lod,
}

// A built-in's result type, from its (checked) arguments.
Builtin_Result :: enum u8 {
	First,
	Last,
	Float,
	Bool,
	Vec4,
	// The first argument's shape with another base.
	Bool_Shape,
	Int_Shape,
	Uint_Shape,
	Float_Shape,
	// outerProduct(c, r): columns of c, one per component of r.
	Outer,
	Transposed,
}

Builtin_Info :: struct {
	// "" for the ones not callable by name.
	name:      string,
	argc:      u8,
	args:      Builtin_Args,
	result:    Builtin_Result,
	// Arguments that may be scalars where the result is a vector; every
	// backend broadcasts them to the result's shape.
	broadcast: bit_set[0 ..< 3;u8],
	// Exact_Base: the base the argument must have.
	base:      Base,
	// Features the stage must have (derivatives: fragment shaders).
	needs:     Features,
}

// What each built-in takes and returns. The backends' tables (VM_BUILTINS,
// SPV_EXT_BUILTINS / SPV_OP_BUILTINS, MSL_BUILTINS) say how each lowers.
@(rodata)
BUILTINS := [Builtin]Builtin_Info {
	.None               = {},
	.Radians            = {"radians", 1, .Float, .First, {}, {}, {}},
	.Degrees            = {"degrees", 1, .Float, .First, {}, {}, {}},
	.Sin                = {"sin", 1, .Float, .First, {}, {}, {}},
	.Cos                = {"cos", 1, .Float, .First, {}, {}, {}},
	.Tan                = {"tan", 1, .Float, .First, {}, {}, {}},
	.Asin               = {"asin", 1, .Float, .First, {}, {}, {}},
	.Acos               = {"acos", 1, .Float, .First, {}, {}, {}},
	// atan(y, x) is Atan2.
	.Atan               = {"atan", 1, .Float, .First, {}, {}, {}},
	.Sinh               = {"sinh", 1, .Float, .First, {}, {}, {}},
	.Cosh               = {"cosh", 1, .Float, .First, {}, {}, {}},
	.Tanh               = {"tanh", 1, .Float, .First, {}, {}, {}},
	.Asinh              = {"asinh", 1, .Float, .First, {}, {}, {}},
	.Acosh              = {"acosh", 1, .Float, .First, {}, {}, {}},
	.Atanh              = {"atanh", 1, .Float, .First, {}, {}, {}},
	.Exp                = {"exp", 1, .Float, .First, {}, {}, {}},
	.Log                = {"log", 1, .Float, .First, {}, {}, {}},
	.Exp2               = {"exp2", 1, .Float, .First, {}, {}, {}},
	.Log2               = {"log2", 1, .Float, .First, {}, {}, {}},
	.Sqrt               = {"sqrt", 1, .Float, .First, {}, {}, {}},
	.Inversesqrt        = {"inversesqrt", 1, .Float, .First, {}, {}, {}},
	.Floor              = {"floor", 1, .Float, .First, {}, {}, {}},
	.Ceil               = {"ceil", 1, .Float, .First, {}, {}, {}},
	.Fract              = {"fract", 1, .Float, .First, {}, {}, {}},
	.Trunc              = {"trunc", 1, .Float, .First, {}, {}, {}},
	.Round              = {"round", 1, .Float, .First, {}, {}, {}},
	.Round_Even         = {"roundEven", 1, .Float, .First, {}, {}, {}},
	.Dfdx               = {"dFdx", 1, .Float, .First, {}, {}, {.Derivatives}},
	.Dfdy               = {"dFdy", 1, .Float, .First, {}, {}, {.Derivatives}},
	.Fwidth             = {"fwidth", 1, .Float, .First, {}, {}, {.Derivatives}},
	.Normalize          = {"normalize", 1, .Float, .First, {}, {}, {}},
	.Abs                = {"abs", 1, .Float_Or_Int, .First, {}, {}, {}},
	.Sign               = {"sign", 1, .Float_Or_Int, .First, {}, {}, {}},
	.Atan2              = {"", 2, .Float_Same, .First, {}, {}, {}},
	.Pow                = {"pow", 2, .Float_Same, .First, {}, {}, {}},
	.Mod                = {"mod", 2, .Float_Same, .First, {1}, {}, {}},
	.Step               = {"step", 2, .Edges, .Last, {0}, {}, {}},
	.Min                = {"min", 2, .Numeric_Same, .First, {1}, {}, {}},
	.Max                = {"max", 2, .Numeric_Same, .First, {1}, {}, {}},
	.Clamp              = {"clamp", 3, .Numeric_Same, .First, {1, 2}, {}, {}},
	// mix(x, y, bvec) is Mix_Bool.
	.Mix                = {"mix", 3, .Float_Same, .First, {2}, {}, {}},
	.Mix_Bool           = {"", 3, .Mix_Bool, .First, {}, {}, {}},
	.Smoothstep         = {"smoothstep", 3, .Edges, .Last, {0, 1}, {}, {}},
	.Isnan              = {"isnan", 1, .Float_Exact, .Bool_Shape, {}, {}, {}},
	.Isinf              = {"isinf", 1, .Float_Exact, .Bool_Shape, {}, {}, {}},
	.Length             = {"length", 1, .Float, .Float, {}, {}, {}},
	.Distance           = {"distance", 2, .Float_Same, .Float, {}, {}, {}},
	.Dot                = {"dot", 2, .Float_Same, .Float, {}, {}, {}},
	.Cross              = {"cross", 2, .Vec3, .First, {}, {}, {}},
	.Faceforward        = {"faceforward", 3, .Float_Same, .First, {}, {}, {}},
	.Reflect            = {"reflect", 2, .Float_Same, .First, {}, {}, {}},
	.Refract            = {"refract", 3, .Refract, .First, {}, {}, {}},
	.Matrix_Comp_Mult   = {"matrixCompMult", 2, .Matrix_Same, .First, {}, {}, {}},
	.Outer_Product      = {"outerProduct", 2, .Float_Vectors, .Outer, {}, {}, {}},
	.Transpose          = {"transpose", 1, .Matrix, .Transposed, {}, {}, {}},
	.Determinant        = {"determinant", 1, .Square_Matrix, .Float, {}, {}, {}},
	.Inverse            = {"inverse", 1, .Square_Matrix, .First, {}, {}, {}},
	.Less_Than          = {"lessThan", 2, .Vector_Ordered, .Bool_Shape, {}, {}, {}},
	.Less_Than_Equal    = {"lessThanEqual", 2, .Vector_Ordered, .Bool_Shape, {}, {}, {}},
	.Greater_Than       = {"greaterThan", 2, .Vector_Ordered, .Bool_Shape, {}, {}, {}},
	.Greater_Than_Equal = {"greaterThanEqual", 2, .Vector_Ordered, .Bool_Shape, {}, {}, {}},
	.Equal              = {"equal", 2, .Vector_Equal, .Bool_Shape, {}, {}, {}},
	.Not_Equal          = {"notEqual", 2, .Vector_Equal, .Bool_Shape, {}, {}, {}},
	.Any                = {"any", 1, .Bool_Vector, .Bool, {}, {}, {}},
	.All                = {"all", 1, .Bool_Vector, .Bool, {}, {}, {}},
	.Not                = {"not", 1, .Bool_Vector, .First, {}, {}, {}},
	.Texture_Lod        = {"", 3, .Texture_Lod, .Vec4, {}, {}, {}},
	.Float_Bits_To_Int  = {"floatBitsToInt", 1, .Float_Exact, .Int_Shape, {}, {}, {}},
	.Float_Bits_To_Uint = {"floatBitsToUint", 1, .Float_Exact, .Uint_Shape, {}, {}, {}},
	.Int_Bits_To_Float  = {"intBitsToFloat", 1, .Exact_Base, .Float_Shape, {}, .Int, {}},
	.Uint_Bits_To_Float = {"uintBitsToFloat", 1, .Exact_Base, .Float_Shape, {}, .Uint, {}},
}

@(private)
builtin_from_name :: proc(name: string) -> Builtin {
	for info, b in BUILTINS {
		if info.name == name && name != "" {
			return b
		}
	}
	return .None
}
