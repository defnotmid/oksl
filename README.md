# oksl

<img src="assets/logo.png" width="96" alt="oksl's mascot: a pixel-art firefly with a glowing lantern">

**A shader compiler you embed in your app.** oksl compiles a small, safe
shading language at runtime, through one front end, from SkSL (the
language of Skia's runtime effects), GLSL ES 3 as Shadertoy writes it, or
plain GLSL ES 3.0 / 3.1 vertex, fragment and compute shaders, into:

- **SPIR-V** for Vulkan,
- **MSL** for Metal,
- a **CPU program** that renders or dispatches the same shader with no GPU at all.

It's plain Odin with no external toolchain. Runtime shaders usually mean
shipping glslang, shaderc, SPIRV-Cross, DXC or Slang; with oksl the compiler
is part of your binary, so nobody installs anything. The library depends on
`core:` only; the tools also use `vendor:sdl3` and `vendor:stb`.

## Who it's for

- **SDL_GPU apps** that want shaders compiled at runtime on every platform,
  instead of shipping pre-built blobs made offline with SDL_shadercross.
- **Games and engines:** hot-reloaded shaders while developing, modding and
  player-made effects. The language has no pointers or memory access beyond
  its inputs, which is why Skia uses SkSL for runtime effects (see
  Limitations about loops before running untrusted code).
- **Creative coding and VJ tools:** a Shadertoy-style playground in your
  own app.
- **UI frameworks:** custom effects (ripples, blurs, gradients, glows) as
  small shaders, the way Flutter uses Skia's runtime effects.
- **Headless rendering and tests:** the CPU interpreter renders the same
  effect on machines without a GPU (servers, CI) and gives a reference for
  golden-image tests. `tools/gpu_parity` checks GPU against CPU.
- **Learning:** a small, readable compiler from source to real GPU code.

Relatives: Skia's SkSL runtime effects, naga (wgpu's shader translator),
SDL_shadercross. oksl's niche is Odin-native, tiny, runtime, with a CPU
fallback. It started as the shader engine of
[onimite](https://github.com/defnotmid/onimite) (video from Lua scripts), where it compiles
fframes' SkSL shaders unchanged; [demo/](demo) is an onimite video about it.

## Use

```odin
m, errors, ok := oksl.compile(source)            // SkSL
// m, errors, ok := oksl.compile_shadertoy(glsl) // Shadertoy GLSL (Options.profile = .Shadertoy)
// m, errors, ok := oksl.compile_glsl(frag)      // GLSL ES 3.0 fragment shader (Options.profile = .Glsl)
// m, errors, ok := oksl.compile_glsl(comp, {stage = .Compute}) // GLSL ES 3.1 compute shader
defer oksl.destroy(m)
if !ok {
	// Errors are values ({pos, code, operands}); text is made here.
	fmt.eprint(oksl.format_errors(errors))   // "12:7: undeclared identifier \"foo\""
	return
}
// GPU: SDL_CreateGPUShader with m.spirv (entry oksl.SPIRV_ENTRY) or m.msl
// (entry oksl.MSL_ENTRY), num_samplers = m.num_samplers, num_uniform_buffers =
// m.num_uniform_buffers, num_storage_buffers = len(m.storage_buffers);
// Vertex shader: oksl.compile(oksl.LAYER_VERTEX_GLSL, {profile = .Glsl, stage = .Vertex}), once.
// CPU:
block := make([]u8, m.block_size)
oksl.write_builtins(block, {resolution = {f32(w), f32(h), 1}, time = t, channel_resolution = {{tw, th, 1}, {}, {}, {}}})
// or one input at a time: oksl.write_input(block, .Channel_Time, {clip_seconds}, 0)
oksl.set_uniform(m, block, "uSpeed", {0.5})
scratch: oksl.Scratch                      // working memory, one per thread
defer oksl.scratch_destroy(&scratch)
oksl.run(&m.program, block, w, h, pixels, w * 4, &scratch, {sampler = my_sampler})
```

A Shadertoy multipass program (Common, Buffer A-D, Image) compiles into a
pass table and an edge array:

```odin
desc := oksl.Program_Desc{common = common_src}
desc.sources[.Buffer_A] = trail_src
desc.sources[.Image] = image_src
desc.channels[.Buffer_A][0] = {.Buffer, u8(oksl.Pass_Kind.Buffer_A)} // itself: the previous frame
desc.channels[.Image][0] = {.Buffer, u8(oksl.Pass_Kind.Buffer_A)}    // this frame's
desc.channels[.Image][1] = {.External, 7}                            // the host's input 7
g, errors, ok := oksl.compile_program(desc)  // errors: "Buffer A:3:9: ..."
defer oksl.destroy_program(g)
for pass, i in g.passes {                    // buffers (A..D), then Image
	for e in g.edges[pass.edges.start:][:pass.edges.count] {
		// e.channel reads: e.source = .Buffer (pass e.id, this frame),
		// .Previous (pass e.id, last frame) or .External (input e.id)
	}
	// run pass.module like any module, into a float target for buffers
}
```

## Language

### SkSL

SkSL as Skia's runtime effects accept it (with ES3-style extensions where
they cost nothing):

- **Entry point:** `half4 main(float2 coord)` (`float4`/`vec4` and
  `vec2` spellings too). `coord` is the position in layer units (pixel
  centres, origin top-left; the host's `iResolution` scales uv to it). The
  result is premultiplied RGBA.
- **Types:** `bool int uint float half` (`half` is `float`: everything is
  32-bit), vectors `float2..4`, `half2..4`, `int2..4`, `uint2..4`,
  `bool2..4`, matrices `float2x2..float4x4` / `halfNxM` (column-major, N
  columns of M rows), the GLSL spellings `vec3`, `ivec2`, `mat3`, `mat2x4`
  ..., structs, and constant-size arrays (dynamic indexing allowed, no
  arrays of arrays).
- **Uniforms:** scalars, vectors, matrices and arrays of them;
  `layout(...)` qualifiers (`layout(color)`) are accepted and ignored. A
  uniform declared with the name and type of an input row (`float3
  iResolution`, `float iTime`, `int iFrame`, ...: the table below) binds
  to that slot; declared with another type it is an ordinary uniform the
  host fills by name.
- **Children:** `uniform shader name;` (up to four, channels in
  declaration order). `name.eval(p)` samples the child at `p` in the
  child's own pixels: it lowers to an explicit-lod sample at `p /
  childSize` (the size comes from the block), bilinear and clamped by the
  host's sampler, like Skia's image shaders.
- **Expressions and statements:** the usual operators, swizzles (`xyzw`,
  `rgba`, `stpq`) for reading and writing, constructors with flattening,
  `?:`, `,`, `.length()`, `if/else`, `for`, `while`, `do-while`, `switch`
  (int or uint, constant labels, fallthrough), `break`, `continue`,
  `return`, `const`, `in`/`out`/`inout` parameters, overloads, prototypes
  and calls before definitions. Implicit `int -> uint -> float`
  conversions are accepted. Recursion is an error.
- **Built-ins:** SkSL's intrinsics (the GLSL ES 3 math, geometric, matrix,
  vector-relational and bit-cast functions, `saturate`), `sk_FragCoord`
  (`float4(coord, 0, 1)`: the layer is its own device), and the extensions
  `dFdx`, `dFdy`, `fwidth`.
- **Diagnosed:** SkSL has no preprocessor (`#version` lines are skipped,
  any other directive is an error), no `discard`, no `sampler2D` /
  `texture()` (use `uniform shader` and `.eval`), no `colorFilter` /
  `blender` children, no shader parameters, and `mainImage` is not an entry
  point (compile Shadertoy code through `compile_shadertoy`). Errors are
  `line:column: message`; several are collected per compile.

### GLSL (Shadertoy)

`compile_shadertoy` (or `Options.profile = .Shadertoy`) takes Shadertoy code,
GLSL ES 3, through the same parser, checker and backends as SkSL:

- **Entry point:** `void mainImage(out vec4 fragColor, in vec2 fragCoord)`.
  The checker synthesizes the `vec4 main(vec2 coord)` the backends call,
  as IR (not text): `fragCoord` has the bottom-left origin, `gl_FragCoord`
  is the same point, and the colour comes out clamped with alpha 1. `main`
  itself is reserved.
- **Inputs:** every Shadertoy input is declared before the source, from
  the input table: `iResolution`, `iTime`, `iTimeDelta`, `iFrame`,
  `iFrameRate`, `iMouse`, `iDate`, `iSampleRate`, `iChannelTime[4]`,
  `iChannelResolution[4]`, `iChannel0..3`. Redeclaring one with its own
  type (`uniform vec3 iResolution;`, common in code from other hosts) is
  allowed. A shader uses (and the host binds) only the ones it reads.
- **Sampling:** `texture(iChannelN, uv)` takes 0..1 uv with the origin at
  the bottom left, like Shadertoy; `textureLod`, `texelFetch` (integer
  texels from the bottom left) and `textureSize` work too. `texture`
  samples at the level its uv derivatives pick (plus its bias), the others
  at an explicit level, as SkSL's `.eval` does; on the CPU the host
  sampler sees the level (see the CPU interpreter). A `uniform sampler2D name;` takes a
  channel the source doesn't name as `iChannelN`, and samplers can be
  function parameters (`texelFetch` and `textureSize` need the channel
  itself, for its size).
- **Preprocessor:** token level: object-like and function-like `#define`
  (with `##`), `#undef`, `#if` / `#ifdef` / `#ifndef` / `#elif` / `#else`
  / `#endif` with `defined()` and integer expressions, `#line`, `#error`;
  `#version`, `#extension` and `#pragma` are skipped. `GL_ES`,
  `__VERSION__` (300) and Shadertoy's `HW_PERFORMANCE` (1) are defined.
  Tokens keep their source positions (a macro's tokens take the position
  of its use), so errors point at the user's line and column.
- **Language:** the SkSL list above with the GLSL spellings only (`vec3`,
  `ivec2`, `mat2x3`; no `float3`, `half`, `saturate`, `sk_FragCoord`),
  plus `discard`, `precision` statements and `highp` / `mediump` / `lowp`
  (accepted and ignored). `uniform shader` and `.eval` are SkSL's and
  diagnosed.

### GLSL vertex and fragment shaders

`compile_glsl` (or `Options.profile = .Glsl`) takes a GLSL ES 3.0
fragment shader, or a vertex shader with `Options.stage = .Vertex`: the
Shadertoy language and preprocessor above, without Shadertoy's
conventions. `compile_pipeline` compiles a vertex and a fragment shader
together and links them:

```odin
p, errors, ok := oksl.compile_pipeline({vertex = vs, fragment = fs, vertex_prelude = vp, fragment_prelude = fp})
defer oksl.destroy_pipeline(p)
// p.vertex and p.fragment are Modules; errors read "Fragment:12:7: ...".
offset, type, found := oksl.uniform_offset(p.vertex, "modelViewMatrix")
```

- **Entry point:** `void main()`.
- **Vertex shaders:** `in` attributes (`layout(location = N)` or the next
  free location), `out` varyings and `gl_Position`, written as it is (no
  depth or y remapping: give it SDL_GPU's clip space, depth 0..1, y up).
  Interpolated integer outputs need `flat`. `Module.stage_inputs` are the
  attributes the shader reads.
- **Linking:** a pipeline matches varyings by name, as GLSL ES 3.00 does
  (which allows no locations on them): each fragment input takes the
  location of the vertex output of its name. One the vertex shader does
  not write, or writes with another type or interpolation, is an error.
- **Preludes:** `Options.prelude` (per stage in a pipeline) is source
  preprocessed and checked ahead of the shader's own: a host's built-in
  declarations, and `#define`s that apply to the shader, including ones
  that rename keywords and built-ins (`#define varying out`,
  `#define gl_FragColor pc_fragColor`), as three.js's ShaderMaterial
  prefixes do. Errors in it are the prelude's (`Vertex prelude:3:1: ...`).
- **Stage variables:** `in` and `out` globals, with
  `layout(location = N)` (other layout qualifiers are ignored). Inputs
  without one take the next free location in declaration order; one output
  may omit it, several may not. Inputs are float, int or uint scalars or
  vectors, integer ones `flat` (`smooth` and `centroid` are accepted);
  outputs are float, int or uint scalars or vectors, at locations 0..7.
  `Module.stage_inputs` / `stage_outputs` reflect them (name, type,
  location, flat).
- **gl_FragCoord:** the fragment's window position, origin top-left
  (Vulkan's and Metal's), pixel centres at .5. **gl_FrontFacing:** whether
  the triangle faces the camera. **gl_VertexID** (vertex shaders): the
  vertex's index in the draw, its first vertex included (Vulkan's
  VertexIndex, Metal's `vertex_id`); on the CPU, `Run_Options.first_vertex`
  plus the vertex's index in `run_vertex`. **gl_SampleMask[0]** (fragment
  shaders): the samples of the fragment's coverage it keeps (SPIR-V's
  SampleMask, Metal's `sample_mask`; `run_fragment`'s `sample_masks`).
  `dFdx`, `dFdy` and `fwidth` are fine derivatives, per row and column of
  the quad, as the interpreter computes them (SPIR-V's `OpDPdxFine` and
  friends: a driver may take plain `OpDPdx` coarse).
- **Uniforms and samplers:** nothing is declared up front. The uniform
  block holds only the uniforms the shader uses, std140 from offset 0, in
  declaration order (none: no block); `uniform_offset` and `set_uniform`
  find them by name, and unused or undeclared names are not found.
  Up to 16 `uniform sampler2D`s and `sampler2DArray`s take channels in
  declaration order (`Module.channel_arrays` marks the arrays). `texture`
  uv starts at the top left, as in Vulkan (no flip). `textureSize` queries
  the image (on the CPU, the size the host gives; an array's z is its
  layers) and `texelFetch` samples the texel's centre at its level, so
  both take sampler parameters too. A `sampler2DArray` takes `vec3(uv,
  layer)` in `texture`, `textureLod` and `textureGrad` (the layer rounded
  to nearest even and clamped, as GPUs take it); `texelFetch` takes 2D
  samplers only. `textureGrad(s, uv, dPdx, dPdy)` samples at the level
  those derivatives pick, in any stage.
- **Storage buffers:** in vertex and fragment shaders, read-only, GLSL ES
  3.1's block with one runtime-sized array: `layout(std430) readonly buffer
  Rows { Row rows[]; } rowBuf;` read as `rowBuf.rows[i]` (or `rows[i]`
  without an instance name), at any int or uint index, in any function and
  either stage (`row_major` and `column_major` layout qualifiers are
  accepted as no-ops; compute shaders also allow read-write and `writeonly`
  storage buffers, bound in SDL_GPU's compute order below). Elements are
  float, int and uint scalars and vectors and structs of them, laid out
  std430 (`layout(std430)` is optional; other packings, writes in graphics
  stages, sized arrays, bools and matrices are diagnosed).
  `Module.storage_buffers` gives each block's name, element type and
  stride (`vec3[]` steps by 16 bytes, while a struct's `vec3` field takes
  12) in SDL_GPU binding order—which in graphics stages equals declaration
  order since every buffer is `readonly`. Word addresses are 32-bit (they
  wrap) on every backend; a read past the buffer gives 0 on the CPU and is
  undefined on the GPU (Metal reads past it).
- **Taps:** `Options.taps` (in a pipeline, the vertex stage's) runs what
  a vertex shader's `main` does toward `gl_Position` again for other
  environments, such as a motion blur's shutter opening, once per tap and
  before `main` itself. Tap N copies the statements the position depends
  on (in `main`, the functions it reaches and the globals' initializers):
  each uniform reads at its offset plus (N + 1) × `Module.block_stride`,
  the attributes the tap lists read their twins
  (`{"position", "o3dOpenPosition"}`), `gl_Position` goes to the tap's
  `position`, and the shader's own outputs and globals are private to the
  copy. Twins and positions are ordinary `in` and `out vec4` declarations,
  usually the prelude's, so locations, reflection and linking treat them
  like any other; a twin the copy never reads is no input, and a position
  takes a location like any output, so a prelude declares it only for a
  pipeline with taps. The block holds `1 + taps` environments of
  `block_stride` bytes, each laid out as `Module.uniforms`
  (`set_uniform(m, block[n * m.block_stride:], ...)` fills environment n).
  `run_vertex` runs the taps too, their positions ordinary varyings; each
  tap costs another run of the arithmetic the position depends on.
- **Diagnosed:** `varying` / `attribute` (unless a prelude defines them),
  duplicate or out-of-range locations, integer interpolated variables
  without `flat`, initializers on stage variables, writes to inputs,
  gl_FragCoord and gl_FrontFacing, fragment-only features (`discard`,
  derivatives, gl_FragCoord, gl_SampleMask) in vertex shaders, gl_VertexID
  in fragment shaders, `sampler2DArray` in the Shadertoy profile (whose channels are
  2D), more than 16 samplers, taps outside a vertex
  shader, a tap position, attribute or twin not declared as one, a
  position the shader uses itself, and two taps writing one position.

### GLSL compute shaders

`compile_glsl(src, {stage = .Compute})` (or `Options{profile = .Glsl,
stage = .Compute}`, `.comp` in the tools; rejected by the SkSL and
Shadertoy profiles) compiles a GLSL ES 3.1 compute shader:

- **Workgroup size:** `layout(local_size_x = X, local_size_y = Y,
  local_size_z = Z) in;` is required (omitted axes default to 1, positive
  integer literals up to 1024 per axis and `X * Y * Z <= 1024` in total;
  duplicate declarations must specify the same set of axes with the same
  values) and is reflected in `Module.threadcount`. Vulkan's guaranteed
  minimums (`maxComputeWorkGroupSize = {128, 128, 64}`,
  `maxComputeWorkGroupInvocations = 128`) are lower, so `8x8`, `16x8` and
  `64x1` stay portable across drivers.
- **Built-ins:** `gl_GlobalInvocationID`, `gl_LocalInvocationID`,
  `gl_WorkGroupID`, `gl_NumWorkGroups` (`uvec3`),
  `gl_LocalInvocationIndex` (`uint`), and the compile-time constant
  `gl_WorkGroupSize` (`const uvec3`, usable in constant expressions and
  array sizes).
- **Storage buffers and binding order:** `std430` storage buffers may be
  `readonly`, read-write (unqualified) or `writeonly` (`readonly` and
  `writeonly` together is diagnosed; reads from `writeonly` and writes to
  `readonly`, including compound assignment, `++`/`--` and `out`/`inout`
  arguments, are diagnosed). Because SDL_GPU binds read-only storage
  buffers before read-write / `writeonly` ones, `Module.storage_buffers`
  and `Run_Options.buffers` use that same order—read-only blocks in
  declaration order, then read-write and `writeonly` blocks in declaration
  order—and each `Storage_Buffer_Info` carries `readonly` and `writeonly`:

  ```glsl
  buffer A { uint out_words[]; };
  readonly buffer B { float weights[]; };
  // Binding / reflection / Run_Options.buffers order: [B, A].
  ```

  Writes are 32-bit word-addressed across SPIR-V, MSL and the CPU VM
  (masked per lane; out-of-bounds writes are dropped on the CPU and
  undefined on the GPU). Graphics stages still allow only `readonly`
  buffers, so their binding order equals declaration order.
- **Storage images:** up to 16 (`IMAGE_COUNT`) 2D storage images
  (`image2D`, `uimage2D`, `iimage2D`), each with a required
  `layout(<format>)` from `rgba8`, `rgba16f`, `rgba32f`, `r32f`
  (`image2D`), `r32ui` (`uimage2D`) or `r32i` (`iimage2D`), qualified
  `readonly` or `writeonly`; unqualified read-write is allowed on the
  single-channel formats `r32f`, `r32ui` and `r32i` (GLSL ES 3.1 / Vulkan
  baseline without `shaderStorageImageReadWithoutFormat`). Built-ins are
  `imageLoad(img, ivec2)`, `imageStore(img, ivec2, val)` and
  `imageSize(img)`. User functions may take `in` image parameters without
  `layout(...)` qualifiers: the format is inferred from call-site arguments
  and conflicting call-site formats are diagnosed (a read-write image may be
  passed to a `readonly` or `writeonly` parameter). Portable GLSL ES 3.1
  shaders pass images to `writeonly` parameters (`glslang` requires
  `GL_EXT_shader_image_load_formatted` on `readonly` and read-write image
  parameters, which oksl accepts as an extension). `Module.storage_images`
  and `Run_Options.images` follow SDL_GPU's order: read-only images first,
  then read-write / `writeonly`, each in declaration order. When creating
  SDL_GPU textures, use `COMPUTE_STORAGE_READ` for `readonly`,
  `COMPUTE_STORAGE_WRITE` for `writeonly`, and
  `COMPUTE_STORAGE_SIMULTANEOUS_READ_WRITE` (not `READ | WRITE`) for
  read-write `r32*` images.
- **Sampling:** `sampler2D` and `sampler2DArray` work with `textureLod`,
  `texelFetch`, `textureGrad` and `textureSize`. Implicit-level `texture`,
  `textureProj` and derivatives (`dFdx`, `dFdy`, `fwidth`) need fragment
  quads and are diagnosed in compute, along with stage `in`/`out`
  variables, `discard` and vertex/fragment built-ins.
- **Phase 1 scope:** `shared` workgroup memory, `barrier()` /
  `memoryBarrier*()` and atomic operations are not implemented yet.

### Pass graphs and programs (multipass)

A `Pass_Graph` is passes (modules, in running order) and edges saying
what each pass's channels read. Nothing in it depends on the dialect:
`make_program` builds one from compiled modules of any profile, and
`compile_program` builds Shadertoy's (below) on top of it.

```odin
blur, _, _ := oksl.compile(blur_sksl)                // any profile
smooth, _, _ := oksl.compile_glsl(smooth_frag)
g, errors, ok := oksl.make_program({
	{module = blur, name = "blur", channels = {0 = {.External, 7}}},
	{module = smooth, name = "smooth", channels = {0 = {.Buffer, 0}, 1 = {.Previous, 1}}},
})
defer oksl.destroy_program(g)                         // frees the modules too, also on failure
for p in g.passes {
	block := make([]u8, p.module.block_size)
	oksl.write_builtins(p.module, block, builtins)  // the inputs only where the module has them
	// ... render p.module into its target, binding channels from g.edges[p.edges]
}
```

- A channel reads the host's input (`.External`, its id), this frame's
  output of an earlier pass (`.Buffer`, pass index) or the previous frame
  of any pass, itself included (`.Previous`; transparent black on the
  first frame). `g.output` is the result's pass (the last by default);
  at most `PASS_MAX` (8) passes.
- Inputs appear per dialect: SkSL `uniform shader` children, Shadertoy
  `iChannel0..3`, GLSL `uniform sampler2D`s, channels in declaration
  order. A GLSL pass on the layer quad reads `in vec2 uv` at location 0.
- A Shadertoy module compiled with `Options.unclamped` returns
  mainImage's colour as it is, for a pass rendering into a float target.

`compile_program` takes a `Program_Desc`: an optional Common source, up
to four buffer passes (Buffer A-D, `""` for absent ones), the Image pass
(required) and, per pass, what each channel reads (`Channel_Ref`: a
buffer by its `Pass_Kind`, read as this frame's output when it runs
earlier and as the previous frame otherwise (or explicitly `.Previous`),
one of the host's inputs by id, or nothing).

- **Common** is one shared unit, preprocessed and parsed once: its macros
  (a mark of the macro table) and its declarations (copies of its flat
  syntax arrays) start every pass, as Shadertoy prepends it. A pass's own
  macros stay in that pass. Errors in Common are reported once.
- **Passes** are modules, in `Pass_Graph.passes`: the buffers in A..D
  order, then the Image pass, which runs last. Buffer passes return
  mainImage's colour unclamped with its alpha (render them into float
  targets); the Image pass is clamped and opaque like `compile_shadertoy`.
- **Edges** are one flat array, for the channels each pass samples, sorted
  by pass then channel (`Pass.edges` is a pass's range): `Buffer` reads
  this frame's output of an earlier pass, `Previous` the previous frame's
  output of itself or a later pass (Shadertoy's feedback; the host binds
  transparent black on the first frame), `External` the host's input `id`.
- **Errors** carry their source in `Pos.src` (`Source`: Common, Buffer A..D,
  Image; `SOURCE_NAMES` spells them) and `format_errors` writes
  `Buffer A:12:7: ...`. A channel bound to a buffer the program lacks is
  an error at the channel's first use in the pass.

### The uniform block (std140)

The inputs come first, at fixed offsets, whether or not they are used:

| offset | uniform |
| --- | --- |
| 0 | `iResolution` vec3 |
| 12 | `iTime` |
| 16 | `iTimeDelta` |
| 20 | `iFrame` (int) |
| 24 | `iFrameRate` |
| 28 | `iSampleRate` |
| 32 | `iMouse` vec4 |
| 48 | `iDate` vec4 |
| 64 | `iChannelResolution`: channel sizes in pixels, vec3[4] (stride 16; `.eval` and `texelFetch` divide by xy) |
| 128 | `iChannelTime` float[4] (stride 16) |
| 192 | user uniforms, std140, in declaration order |

The rows are one table, `INPUTS` (indexed by `Input`: name, type, offset;
the channel rows give the sampler binding). `Module.uniforms` reflects each
user uniform (offset, size, count, stride) and `Module.block_size` gives
the total size (with vertex taps, `block_stride` is one environment's).
Bools are stored as 4-byte uints. `write_builtins`, `write_input` (one
row, by index) and `set_uniform` fill a block.
`inputs_used` and `channels_used`/`num_samplers` say what the shader
reads, and `channel_names[n]` names what channel n binds (a `uniform
shader`, `iChannelN` or a `uniform sampler2D`). SkSL and Shadertoy have
four channels (the rows above); GLSL shaders have 16 (`SAMPLER_COUNT`).

### Bindings (SDL_GPU)

- **SPIR-V 1.0:** fragment entry `main`; stage inputs and outputs at their
  locations (the layer profiles: `vec2 uv` at location 0, 0..1 across the
  layer, origin top-left, and the `vec4` colour at location 0);
  gl_FragCoord as the `FragCoord` built-in; integer inputs `Flat`; channel
  n as a combined image sampler at set 2, binding n (an arrayed image for a
  `sampler2DArray`); storage buffer i at set 2, binding `num_samplers + i`
  (a `BufferBlock` of `uint` words, read-only); the uniform block at set 3,
  binding 0. `OriginUpperLeft`.
- **MSL:** fragment entry `oksl_main`, returning `Oksl_Out` (output N as
  `[[color(N)]]`); input N read as `[[user(locnN)]]` (SPIRV-Cross's names,
  so SDL_shadercross vertex shaders match), gl_FragCoord as
  `[[position]]`; channel n at `[[texture(n)]]` / `[[sampler(n)]]`
  (`texture2d_array` for a `sampler2DArray`); the block at
  `[[buffer(0)]]`, storage buffer i at `[[buffer(num_uniform_buffers +
  i)]]` as `const device uint*`. Its struct is padded to the same std140
  offsets.
- **Vertex shaders:** SPIR-V entry `main` (Vertex), attributes at their
  locations, `gl_Position` as the `Position` built-in, varyings at their
  locations, `gl_VertexID` as `VertexIndex`; samplers then storage buffers
  at set 0, the uniform block at set 1, binding 0. MSL: `vertex Oksl_Out
  oksl_main` with attributes as `[[attribute(N)]]` of a `[[stage_in]]`
  struct, `gl_VertexID` as a `[[vertex_id]]` parameter, varyings as
  `[[user(locnN)]]` and the position as `[[position]]`; the block at
  `[[buffer(0)]]`, storage buffers after it. One vertex uniform block
  (`Module.num_uniform_buffers` 1, or 0 when there is none).
- **Compute shaders:** `Module.threadcount` and the six SDL_GPU counts
  (`num_samplers`, `num_readonly_storage_textures`,
  `num_readonly_storage_buffers`, `num_readwrite_storage_textures`,
  `num_readwrite_storage_buffers`, `num_uniform_buffers`) populate
  `SDL_GPUComputePipelineCreateInfo` directly. SPIR-V (`GLCompute`,
  `LocalSize`, `NonWritable` on `readonly`, `NonReadable` on `writeonly`):
  set 0 binds combined samplers (`0..num_samplers - 1`), read-only storage
  images, then read-only storage buffers; set 1 binds read-write /
  `writeonly` storage images, then read-write / `writeonly` storage
  buffers; set 2, binding 0 binds the uniform block. MSL (`kernel void
  oksl_main`): flat `[[texture(n)]]` (`0..num_samplers - 1` sampled
  textures paired with `[[sampler(n)]]`, then read-only storage images as
  `access::read`, then read-write / `writeonly` storage images as
  `access::read_write` / `access::write`), the uniform block at
  `[[buffer(0)]]` when present, then read-only storage buffers (`const
  device uint*`) followed by read-write / `writeonly` storage buffers
  (`device uint*`) starting at `[[buffer(num_uniform_buffers)]]`.
- **Layer vertex shader** (`LAYER_VERTEX_GLSL`, the source of
  `testdata/layer.vert`, which a host compiles once like any vertex
  shader: `compile(LAYER_VERTEX_GLSL, {profile = .Glsl, stage =
  .Vertex})`): position (vec2, target pixels, y down) at location 0, uv
  (vec2) at location 1, and its uniform block holding the target size
  (`size`, `vec4`, xy used). Positions map to NDC like ovg's SDL_GPU
  backend. It writes uv at location 0 (in MSL as `[[user(locn0)]]`). A
  GLSL fragment shader reading `layout(location = 0) in vec2` can run on
  it too.

The generated fragment code derives `coord` from `uv * iResolution.xy`, so
set `iResolution` to the layer's size in the units `coord` should run in
(pixels, or the box's own units as fframes does).

## CPU interpreter

`compile` also builds `Module.program`. Run it with
`run(&m.program, block, width, height, out, stride, &scratch, opts)` for
RGBA8, or `run_f32` for float RGBA, from the output at location 0;
`run_targets` writes every output, one float target per location. A
`Scratch` (registers and masks) grows to the largest program it has run,
so runs that reuse one allocate nothing. `Run_Options` takes a
sampler callback (`proc(user, channel, uv, lod, layer) -> [4]f32`; `.eval`
passes lod 0; `layer` is a `sampler2DArray`'s, rounded to nearest even,
for the host to clamp to its layers), the channels' sizes in texels (and
layers), optional bound 2D float images (`channel_images`, below), the
storage buffers' words (`buffers`, one `[]u32` per buffer in
`Module.storage_buffers` order; a read past one gives 0 and a write past
one is dropped), the compute storage images (`images`, one `Storage_Image`
per image in `Module.storage_images` order), an optional sub-rectangle and
the stage inputs by location
(`Varying`: per component `a + dx * x + dy * y` at pixel centres, what one
triangle's rasterization gives). An input without one is zero, except
location 0, which gets the layer's uv as the layer vertex shader writes it.

For hosts that rasterize themselves (a software 3D renderer keeping its
own clipping, coverage and depth), two calls run a pipeline's stages:

```odin
scratch: oksl.Scratch                      // one per thread; a Program is only read
defer oksl.scratch_destroy(&scratch)
// Vertices: attribute n from (pointer, byte stride, format: 1 to 4 f32, i32
// or u32, Ubyte4_Norm or Ushort4, as SDL_GPU's formats); out come clip
// positions and each vertex's varyings, `varying_floats` per vertex at the
// offsets Stage_Var_Info.offset gives (flat ones flagged; integers as their
// bits, so a fragment's flat integer input is its provoking vertex's bits).
oksl.run_vertex(&p.vertex.program, vblock, attrs, count, positions, varyings, &scratch)
// Fragments: the host's 2x2 quads (top-left pixel, facing, helper lanes,
// optionally each lane's depth for gl_FragCoord.z) with their interpolated
// varyings, one float of the record at a time across every lane; out come
// every output per lane (targets[n] for location n, nil to skip), the
// lanes shaded and, when given, each lane's kept samples (gl_SampleMask[0],
// all ones when unwritten: AND it with coverage).
oksl.run_fragment(&p.fragment.program, fblock, {count = n, origin = origins, front_facing = facing, helpers = helpers, varyings = soa, depth = z}, targets, shaded, &scratch, {}, sample_masks)
```

`run_fragment` shades `LANES / 4` (16) quads at a time, each quad's
derivatives from its own lanes as in the layer runner, so its results
match the layer runner's bit for bit (a test checks).

Compute shaders run through `run_compute`:

```odin
img := oksl.Storage_Image{data = rgba_bytes, width = w, height = h, stride = w * 4, format = .Rgba8}
// Run all workgroups, or split [first_group ..< first_group + group_count) across a worker pool:
oksl.run_compute(&m.program, block, {groups_x, groups_y, 1}, &scratch, {images = {img}, buffers = {out_words}}, first_group, group_count)
```

`run_compute` executes one workgroup at a time in `ceil(local_size / 64)`
64-lane chunks (`group_count = -1` runs to the end of `groups.x * groups.y
* groups.z`). Local sizes below 64 leave CPU lanes idle (`8x8` or `64x1`
fill a batch), and lane layout inside a workgroup does not depend on how
the workgroup range is split across caller threads (give each thread its
own `Scratch`). `run_compute` validates `Run_Options` up front and panics on
missing buffers or images, mismatched `Image_Format`, negative dimensions or
strides, strides smaller than `width * texel_bytes`, or undersized `data`.
On `Storage_Image` (`stride == 0` means `width *
texel_bytes`), `rgba8` writes `round(clamp(x, 0, 1) * 255)`, `rgba16f`
rounds to nearest even (Vulkan allows drivers to round toward zero),
integer formats are exact, single-channel `r32*` loads fill `.gba = (0, 0,
1)` in bounds, out-of-bounds `imageLoad` returns `(0, 0, 0, 0)`, and
out-of-bounds `imageStore` and buffer writes are dropped. Concurrent
writes to the same word or texel have unspecified results.

`texture` in a fragment shader passes the sampler the level a GPU picks:
`0.5 * log2(max(|duv/dx|², |duv/dy|²))` in the channel's texels
(`Run_Options.channel_sizes`), one per quad from its top row and left
column, plus the bias. It is fractional and below 0 where magnified (-inf
where uv is constant), so the sampler clamps it to the levels it has; a
channel without a size samples at the bias. GPUs approximate the level
differently (on a bent mapping radv and Apple pick up to 0.17 levels
higher), and where uv jumps inside a quad (a `fract` seam) which pixels
they difference decides it; `tools/gpu_parity` samples a mip chain to
show both.

For flat 2D float maps owned by the caller, `Run_Options.channel_images`
(`[]Channel_Image`, indexed by channel; `oksl.channel_image(texels, width,
height, filter)` accepts `[]f32`, `[][2]f32` or `[][4]f32`) binds tightly
packed 1-, 2- or 4-component `f32` texels directly to any `run*` entry
point without calling `sampler` per lane (lowered to in-JIT AVX2 gathers
on x86-64). Missing components read `(0, 0, 0, 1)`. Both filters clamp to
the edge (`.Nearest`: `clamp(floor(uv * size), 0, size - 1)`; `.Linear`:
`uv * size - 0.5` with fractional weights). A bound image has one level:
`texture`, `textureLod` and `textureGrad` at any lod read level 0,
`texelFetch` reads its texel or `(0, 0, 0, 0)` out of range (flipping `y`
in the Shadertoy profile), and `textureSize` returns `(width, height)`.
Bindings are validated up front via `ensure`, which panics on a
`sampler2DArray` channel, invalid component count, filter or dimensions,
or `data` shorter than `width * height * components`. Any channel with
`len(data) == 0` or past `len(channel_images)` stays unbound and continues
to use `Run_Options.sampler` (for mipmaps, wrap modes, `sampler2DArray` or
sRGB `u8` textures), and bound and unbound channels may mix in the same
pass.

- **Inlining:** every function is inlined into one flat `[]Instr` program
  over SoA registers. A slot is one scalar component for `LANES` (64)
  lanes: 16 2x2 quads, an 8x8 block of the layer, so each instruction
  dispatched works on 64 pixels and neighbouring pixels diverge little.
  Variables and consecutive swizzles of them (`.xy`,
  `.rgb`) are read in place; a call keeps a return mask only when a
  `return` can leave the function early.
- **Dispatch:** one switch per instruction, never per lane. Arithmetic,
  comparisons, conversions, the float built-ins and the vector and matrix
  ones (`dot`, `normalize`, `cross`, `reflect`, products; not
  `determinant` and `inverse`) run as whole-register loops, each
  specialized at compile time (`$` parameters and `when`), so no lane loop
  branches on the operation. Execution masks are
  `bit_set`s of lanes. Masked stores and selects are branchless, and
  `floor` avoids `core:math`'s modf.
- **JIT:** on x86-64 Linux and macOS when the CPU has AVX2, `compile` also
  translates vertex and fragment programs into 8-lane (two 2x2 quads per
  256-bit YMM register) native AVX2 code in W^X pages (allocated
  read-write, switched read-execute, and released by `destroy`). Compute
  shaders, other targets, machines without AVX2, environments that refuse
  executable memory (`PROT_EXEC`), and builds with `-define:OKSL_JIT=false`
  run on the 64-lane interpreter with the same results, apart from NaN bit
  patterns and values computed in lanes that aren't running (both undefined
  on GPUs too). CI runs the tests and goldens both ways and requires
  identical output.
- **Uniform work:** what follows from constants and uniforms alone moves
  to the front of the program and runs once per run, before the batches,
  even from inside loops and branches: a raymarcher's `rot(iTime)` is
  computed once rather than at every step of every batch. An instruction
  moves when its op reads only its operands (not texture samples,
  derivatives or dynamic indexing), its operands are such values, and
  nothing else writes its registers; variables are written only by
  stores, so their registers are never a moved instruction's.
- **Divergence:** divergent control flow runs every path under execution
  masks (break, continue, return and discard masks per construct; a
  `switch` lets each label's lanes join the running ones). Only stores to
  variables honour the mask, so helper lanes keep `dFdx`/`dFdy` (fine
  derivatives) correct. Discarded pixels are left untouched.
- **Precision:** f32 throughout, matching GPU semantics: GLSL `mod`, C
  integer division and remainder, guarded division by zero and
  float-to-int conversion, and shift amounts masked to 31.

## Design

Data-oriented throughout:

- **Flat arrays:** tokens, syntax, symbols, functions and struct fields
  are flat arrays of plain structs in one `Unit`, referenced by integer ids
  (`Expr_Id`, `Stmt_Id`, `Sym_Id`, ...). Child lists are ranges into one id
  array.
- **Types:** interned in `Unit.types` (`Type_Id`).
- **Annotations:** the checker annotates expressions in place by index and
  adds `Convert` expressions instead of rewriting trees.
- **Dialects, profiles and stages are data:** `DIALECT_RULES` (the
  language's spelling: SkSL or GLSL), `PROFILE_RULES` (a host's
  conventions on a dialect: SkSL effects, Shadertoy, GLSL fragment shaders,
  with their entry points, predeclared inputs, uv flip and macros) and
  `STAGE_RULES` (the pipeline stage: SPIR-V execution model and modes,
  SDL_GPU descriptor sets, the Metal qualifier, location limits). A feature
  is available when the dialect has it and neither the profile nor the
  stage takes it away, and each table says what using a missing one
  reports. The shared passes consult them; there is no per-dialect code
  path or class hierarchy. `SOURCE_EXTENSIONS` maps the tools' file
  extensions to profiles and stages.
- **Inputs are a table:** `INPUTS` (name, type, block offset or binding)
  drives SkSL's uniform binding, GLSL's predeclarations, the block layout
  every backend emits and the host's `write_input`.
- **Built-ins and operators are tables:** `BUILTINS` gives each built-in
  its name, argument count, argument rule, result rule and the arguments
  that broadcast from scalars; the checker types every call from it. Each
  backend lowers from its own table indexed by built-in (or operator) and
  operand base: `VM_BUILTINS` / `VM_BINOPS` / `VM_CONVERT`,
  `SPV_EXT_BUILTINS` / `SPV_OP_BUILTINS` / `SPV_BINOPS` / `SPV_CONVERT`,
  `MSL_BUILTINS`, with a short hand-written list for the few that are not
  one instruction or call.
- **The stage is IR plus a table:** the checker synthesizes the entry
  point (`stage.odin`: the layer coordinate, `sk_FragCoord`, global
  initializers, a vertex shader's taps, the call to `main`) as ordinary IR,
  a tap being a copy of the statements its position depends on, and lists
  what it reads and writes in `Checked.interface` (stage inputs and
  outputs, the uniform block, the channels, with locations, sets and
  bindings). Backends declare those rows and emit the entry like any other
  function.
- **Errors are values:** `Error` is a position, an `Error_Code` and plain
  operands; `ERROR_TEXT` makes the message only in `format_errors` /
  `error_message`.
- **Backends:** keep per-id scratch arrays (SPIR-V ids, register slots)
  rather than fields on nodes.
- **Memory:** a compile allocates from one arena owned by the `Module`,
  released by `destroy` (along with any executable pages the JIT mapped);
  the preprocessor expands into a scratch arena of its own.

The pipeline: `lexer.odin` (tokens) -> `preprocess.odin` (macro table
indexed by id, a frame stack for rescans, flat tokens out) -> `parser.odin`
-> `check.odin` (names, types, lowering, `dialect.odin`'s rules, the
`inputs.odin` and `builtins.odin` tables, the Shadertoy entry adapter in
`shadertoy.odin`, the stage entry and interface in `stage.odin`, both
built with `nodes.odin`) ->
`spirv.odin` / `spirv_emit.odin`, `msl.odin` and `vm.odin` / `vm_run.odin`
/ `vm_jit.odin`.
`errors.odin` holds the error codes and their text. `switch` reaches each
backend natively: SPIR-V as `OpSwitch` (fallthrough as the following
labels' code, since structured SPIR-V only falls into the next operand),
MSL as `switch`, the interpreter under masks.

`program.odin` builds multipass programs on these pieces: one
`Preprocessor` runs Common, marks its macro table and resets to the mark
for each pass (`preprocess_unit` / `preprocess_mark` / `preprocess_reset`);
Common is parsed into one `Unit` whose syntax arrays seed each pass's unit
(`unit_copy_syntax`), and each pass is checked, with the entry adapter,
and emitted like a single compile.

## Test

```sh
odin test . -vet -strict-style                      # SkSL, GLSL/Shadertoy, the preprocessor, diagnostics,
                                                    # built-ins on the CPU, control flow and switch, discard,
                                                    # uniforms, children and channels, programs, corpus
                                                    # (+ spirv-val when installed)
odin test . -vet -strict-style -define:OKSL_JIT=false # force the portable interpreter on x86-64
odin run tools/gpu_parity -- out_dir                # SDL_GPU vs CPU interpreter on testdata/ through the layer vertex shader, and a tap of it
odin run tools/okslc -- a.sksl b.glsl ...           # compile (by extension: SOURCE_EXTENSIONS); -o prefix writes .spv / .metal
odin run tools/cpu_render -- shader.sksl out.png    # CPU render
odin run tools/golden -- snapshot ref                # before a refactor: ~95k generated cases + every shader
odin run tools/golden -- compare ref out             # after (snapshot out first): what changed, by kind
odin check . -no-entry-point -target:darwin_arm64
```

`testdata/` holds representative shaders across SkSL (`.sksl`), Shadertoy
code (`.glsl`), GLSL fragment (`.frag`) and compute (`.comp`) shaders, and
the layer vertex shader in GLSL (`layer.vert`): gradients, value noise and fbm, SDF raymarching, a
textured tunnel, swizzles, out parameters, structs and arrays, defines,
derivatives, premultiplied alpha, SkSL types and built-ins, matrices,
integer hashing, loops, uniforms, ternaries and short-circuiting,
matrix comparison, children, storage buffers, RGBA-to-NV12 conversion,
storage images, explicit-LOD compute sampling and bound float channels, and Shadertoy-style snippets (`st_*.glsl`) with function-like
macros, `texture` / `texelFetch` / `textureSize`, `switch` and `discard`.

## Limitations

- **Stages:** fragment, GLSL vertex and GLSL compute shaders, on the GPU
  and the CPU (`run_vertex`, `run_fragment`, `run_compute`). Compute has no
  `shared` workgroup memory, barriers or atomics yet (phase 2), and runs on
  the 64-lane interpreter rather than the AVX2 JIT. Storage images and
  writable storage buffers are compute-only. Programs run their passes as
  fragment shaders and the host owns the float targets and swaps them each
  frame. GLSL shaders have no `gl_FragDepth` or `gl_InstanceID` (draw
  instances as vertex ranges and divide `gl_VertexID`), no user uniform
  blocks, and no array stage variables.
- **MSL is compiled only on macOS:** Metal has no Linux compiler, so on
  Linux the emitter is checked textually; on macOS `tools/gpu_parity` runs
  the MSL through SDL_GPU's Metal driver against the CPU interpreter.
- **No DXIL:** on Windows, run SDL_GPU's Vulkan driver.
- **Not supported:** struct uniforms, arrays of arrays,
  `colorFilter` / `blender` children; in GLSL, cubemap and 3D samplers,
  arrays of samplers and Shadertoy's keyboard, sound and webcam inputs. Children
  and channels are images bound by the
  host (no shader trees); `sk_FragCoord` is layer-local.
- **Precision:** transcendental functions differ by a few ulps between
  GPU and CPU, and thresholds (`step`, comparisons) right at such values
  can flip. `tools/gpu_parity` reports these as threshold edges.
- **Runaway loops:** the interpreter has no iteration cap, so a shader
  that loops forever hangs it, as it would hang a GPU.
- **Uniform blocks past 4 KB on Vulkan:** SDL_GPU's Vulkan backend binds
  4,096 bytes of each uniform push (`MAX_UBO_SECTION_SIZE` in SDL 3.4), so
  a larger block (`Module.block_size`) reads zeros beyond it there, with
  no error; Metal and the CPU see all of it. Put large tables in storage
  buffers.

## Roadmap

What would make oksl useful well beyond onimite:

- **DXIL / HLSL output** for D3D12, SDL_GPU's preferred driver on Windows.
- **Compute shared memory, barriers and atomics** (phase 2).
- **WGSL output** for WebGPU and browsers.
- **A loop iteration cap** (and SkSL's ES2 loop rules enforced) so untrusted
  shaders can't hang the CPU interpreter.

## License

MIT, see [LICENSE](LICENSE).

The mascot, a firefly whose lantern is the shader, is drawn by `tools/mascot`
(`odin run tools/mascot -- assets`): `assets/logo.svg` keeps its lantern (lit or
out), two glow halos, eyes and mouths as layers a script can switch by id,
each one path per colour, and a hidden `silhouette`: the whole firefly as one
outline, to morph, extrude or collide.
