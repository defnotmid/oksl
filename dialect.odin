package oksl

// One front end, three kinds of source. What differs is plain data the
// shared passes consult, along three axes:
//
// - Dialect (DIALECT_RULES): the language's spelling. SkSL (Skia's) has
//   floatN types, `uniform shader` children and no preprocessor; GLSL ES 3
//   has sampler2D, texture() and a token-level preprocessor.
// - Profile (PROFILE_RULES): a host's conventions on top of a dialect. The
//   entry point, the inputs declared before the source, how sampling
//   coordinates run.
//     - Sksl: Skia's runtime effects, `half4 main(float2 coord)` over a
//       layer.
//     - Shadertoy: GLSL with `mainImage`, the Shadertoy inputs and uv from
//       the bottom left.
//     - Glsl: a GLSL ES 3.0 fragment shader, `void main()` with
//       `layout(location = N)` in and out variables.
// - Stage (STAGE_RULES, stage.odin): the pipeline stage the profile
//   compiles to.
//
// A feature is available when the dialect has it and neither the profile
// nor the stage lacks it.

Dialect :: enum u8 {
	SkSL,
	GLSL,
}

Feature :: enum u8 {
	// #define, #if and friends (without it only #version is skipped).
	Preprocessor,
	// The SkSL type spellings: floatN, halfN, intN, uintN, boolN, shortN,
	// floatNxM, half, short, ushort.
	Sksl_Types,
	// `shader` children and `child.eval(coord)`.
	Children,
	// `sampler2D` and texture, textureLod, texelFetch, textureSize.
	Samplers,
	Discard,
	// saturate(x).
	Saturate,
	// sk_FragCoord: the entry's coordinate.
	Sk_Frag_Coord,
	// gl_FragCoord: Shadertoy's fragCoord, set by the entry adapter, or the
	// fragment's window position.
	Gl_Frag_Coord,
	// Function parameters of sampler type.
	Sampler_Params,
	// Global `in` and `out` variables: the stage's interface.
	Stage_Variables,
	// dFdx, dFdy, fwidth.
	Derivatives,
	// gl_Position: a vertex shader's clip-space output.
	Gl_Position,
	// gl_FrontFacing: whether the fragment's triangle faces the camera.
	Gl_Front_Facing,
}
Features :: bit_set[Feature;u16]

Dialect_Rules :: struct {
	has:        Features,
	// The diagnostic for a feature the dialect lacks; None: report it like
	// any unknown name.
	missing:    [Feature]Error_Code,
	// Macros defined before the source.
	predefined: []Predefined_Macro,
}

Predefined_Macro :: struct {
	name, value: string,
}

@(rodata)
DIALECT_RULES := [Dialect]Dialect_Rules {
	.SkSL = {
		has = {.Sksl_Types, .Children, .Saturate, .Sk_Frag_Coord, .Derivatives},
		missing = #partial {.Preprocessor = .No_Preprocessor, .Samplers = .Sksl_Texture_Call, .Discard = .Sksl_Discard, .Sampler_Params = .Sampler_Param, .Stage_Variables = .Global_In_Out},
	},
	.GLSL = {
		has = {.Preprocessor, .Samplers, .Discard, .Gl_Frag_Coord, .Sampler_Params, .Stage_Variables, .Derivatives, .Gl_Position, .Gl_Front_Facing},
		missing = #partial {.Children = .Glsl_Eval},
		predefined = {{"GL_ES", "1"}, {"__VERSION__", "300"}},
	},
}

Profile :: enum u8 {
	Sksl,
	Shadertoy,
	Glsl,
}

@(rodata)
PROFILE_NAMES := [Profile]string {
	.Sksl      = "SkSL",
	.Shadertoy = "Shadertoy",
	.Glsl      = "GLSL",
}

Entry_Kind :: enum u8 {
	// `half4 main(float2 coord)` over a layer (stage.odin).
	Main,
	// Shadertoy's `void mainImage(out vec4, in vec2)`, wrapped in a
	// synthesized `main` (shadertoy.odin).
	Main_Image,
	// `void main()`, reading and writing the stage's variables.
	Void_Main,
}

Profile_Rules :: struct {
	dialect:      Dialect,
	// The stages its source can be (Options.stage).
	stages:       bit_set[Stage],
	entry:        Entry_Kind,
	// Features the profile takes away from its dialect, and what using
	// one reports.
	lacks:        Features,
	missing:      [Feature]Error_Code,
	// Inputs declared before the source.
	inputs:       Inputs,
	// The uniform block starts with the inputs (at INPUTS' offsets, the
	// source's uniforms from OFFSET_USER). Without it the block holds only
	// the uniforms the source uses, from offset 0.
	header:       bool,
	// A `uniform` declared with an input's name and type is that input.
	binds_inputs: bool,
	// texture() and texelFetch() coordinates start at the bottom left.
	flip_y:       bool,
	// Macros defined before the source, after the dialect's.
	predefined:   []Predefined_Macro,
}

@(rodata)
PROFILE_RULES := [Profile]Profile_Rules {
	.Sksl = {dialect = .SkSL, stages = {.Fragment}, entry = .Main, header = true, binds_inputs = true},
	.Shadertoy = {
		dialect = .GLSL,
		stages = {.Fragment},
		entry = .Main_Image,
		header = true,
		lacks = {.Stage_Variables, .Gl_Front_Facing},
		missing = #partial {.Stage_Variables = .Global_In_Out},
		inputs = UNIFORM_INPUTS | CHANNEL_INPUTS,
		flip_y = true,
		predefined = {{"HW_PERFORMANCE", "1"}},
	},
	.Glsl = {dialect = .GLSL, stages = {.Fragment, .Vertex}, entry = .Void_Main},
}

// What a source file's extension says it is, for tools.
Source_Extension :: struct {
	ext:     string,
	profile: Profile,
	stage:   Stage,
}

@(rodata)
SOURCE_EXTENSIONS := [?]Source_Extension {
	{".sksl", .Sksl, .Fragment},
	{".glsl", .Shadertoy, .Fragment},
	{".frag", .Glsl, .Fragment},
	{".vert", .Glsl, .Vertex},
}

// The options (profile and stage) a source file's extension names; SkSL
// for an unknown one.
options_of_path :: proc(path: string) -> (Options, bool) {
	for e in SOURCE_EXTENSIONS {
		if len(path) >= len(e.ext) && path[len(path) - len(e.ext):] == e.ext {
			return {profile = e.profile, stage = e.stage}, true
		}
	}
	return {}, false
}

// The features a profile's source can use in a stage.
profile_features :: proc "contextless" (p: Profile, s: Stage) -> Features {
	r := PROFILE_RULES[p]
	return DIALECT_RULES[r.dialect].has - r.lacks - STAGE_RULES[s].lacks
}

// What using a feature the profile and stage do not have reports (None:
// report it like any unknown name).
missing_feature :: proc "contextless" (p: Profile, s: Stage, f: Feature) -> Error_Code {
	r := PROFILE_RULES[p]
	switch {
	case f not_in DIALECT_RULES[r.dialect].has:
		return DIALECT_RULES[r.dialect].missing[f]
	case f in r.lacks:
		return r.missing[f]
	}
	return STAGE_RULES[s].missing[f]
}
