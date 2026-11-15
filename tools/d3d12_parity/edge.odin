#+build linux amd64, windows amd64
package d3d12_parity

import oksl "../.."
import parity "../parity"

// Fragment shaders for what the DXBC emitter lowers itself and testdata/
// exercises little: each is rendered through D3D12 and compared with the
// CPU, in 8 bits under the testdata shaders' rule, or (`f32`) on a float
// target under `tol` (parity.close: NaN matching NaN, infinities and zeros
// exact), a Shadertoy colour unclamped. Shadertoy unless `glsl`; `setup` writes uniforms the
// shared block does not have, `skip` leaves out pixels GLSL leaves
// undefined.

Edge_Shader :: struct {
	name, source: string,
	glsl:         bool,
	f32:          bool,
	tol:          parity.Tolerance,
	setup:        proc(m: ^oksl.Module, block: []u8),
	skip:         proc(x, y: int) -> bool,
}

EDGE_SHADERS := [?]Edge_Shader {
	// Early returns from nested loops and a switch inside inlined functions, a void return.
	{name = "edge_returns", source = `int findFirst(int n, int target) {
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (i * 8 + j == target) return i * 10 + j;
            if (j > n) break;
        }
        switch (i) {
        case 3: if (n == 2) return -1; break;
        case 5: continue;
        default: break;
        }
    }
    return -2;
}
float early(float x) { if (x < 0.3) return 0.1; if (x < 0.6) { return 0.5; } return 0.9; }
void noret(inout float v) { if (v > 0.5) { v = 1.0; return; } v *= 0.5; }
void mainImage(out vec4 c, in vec2 f) {
    int t = int(f.x) % 64; int n = int(f.y) % 9;
    int r = findFirst(n, t);
    float k = f.y / iResolution.y;
    noret(k);
    c = vec4(float(r + 2) / 80.0, early(f.x / iResolution.x), k, 1.0);
}`},
	// Continue in a for with a step and through a switch, fallthrough, do-while with continue, while (true) with break.
	{name = "edge_loops", source = `void mainImage(out vec4 c, in vec2 f) {
    int x = int(f.x), y = int(f.y);
    int acc = 0;
    for (int i = 0; i < 10; i += 2) {
        if (i == 4) continue;
        switch ((x + i) % 4) {
        case 0: acc += 1;
        case 1: acc += 2; break;
        case 2: continue;
        default: acc += 5;
        }
        acc += 100;
    }
    int k = 0; int s = 0;
    do { k++; if (k % 3 == 0) continue; s += k; } while (k < (y % 7) + 3);
    int w = 0;
    while (true) { w += 3; if (w > x % 50) break; }
    c = vec4(float(acc % 256) / 255.0, float(s) / 40.0, float(w) / 60.0, 1.0);
}`},
	// Run-time indexes into local arrays, arrays of structs with array and matrix fields, matrix columns and elements, vector components (read, written, through a swizzle and as inout arguments), uniform arrays and array constructors.
	{name = "edge_index", source = `uniform vec4 uPalette[3];
struct S { vec2 a; float b[3]; mat2 m; };
void bump(inout float v, float d) { v += d; }
void mainImage(out vec4 c, in vec2 f) {
    int x = int(f.x), y = int(f.y);
    float arr[5] = float[5](0.1, 0.2, 0.3, 0.4, 0.5);
    arr[x % 5] = 0.9;
    S ss[2];
    ss[0].b[1] = 0.25; ss[1].a = vec2(0.5, 0.75); ss[y % 2].b[x % 3] = float(x % 7) / 7.0;
    ss[1].m = mat2(1.0, 2.0, 3.0, 4.0);
    mat3 m = mat3(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9);
    vec3 col = m[x % 3];
    m[y % 3] = vec3(1.0);
    m[x % 3][y % 3] = 0.0;
    vec4 v = vec4(0.0, 0.25, 0.5, 0.75);
    float comp = v[y % 4];
    v[x % 4] = 1.0;
    v.zw[x % 2] = 0.125;
    bump(arr[(x + 1) % 5], 0.05);
    bump(v[y % 4], 0.01);
    c = vec4(arr[y % 5] + ss[y % 2].b[(x + 1) % 3] * 0.1, col.y + m[y % 3].x * 0.1 + m[1][x % 3] * 0.05,
             comp + v.x * 0.1 + v[(x + y) % 4] * 0.2 + ss[1].m[x % 2][y % 2] * 0.01, 1.0);
    c.rgb += uPalette[x % 3].rgb * 0.05 + float[3](0.01, 0.02, 0.03)[y % 3];
}`},
	// Integer / and % by zero and by -1 (INT_MIN), negative operands, unsigned / and %, shifts by 32 or more, vectors.
	{name = "edge_integers", source = `void mainImage(out vec4 c, in vec2 f) {
    int x = int(f.x) - 128, y = int(f.y) - 72;
    int q = (y == 0) ? 0 : x / y;
    int d0 = x / (y % 3);
    int r0 = x % (y % 5);
    int mn = (-2147483647 - 1) / ((y > 0) ? -1 : 1);
    int mr = (-2147483647 - 1) % ((y > 0) ? -1 : 3);
    uint ux = uint(x + 200), uy = uint(y + 100);
    uint ud = ux / (uy % 4u), ur = ux % (uy % 6u);
    int sh = (x << (y & 63)) ^ (x >> ((y + 3) & 63));
    uint ush = ux >> (uy & 63u);
    int nm = -x % 7;
    ivec2 vd = ivec2(x, -x) / ivec2(3, y % 4);
    c = vec4(float((q + d0 + r0 + vd.x + vd.y) & 255) / 255.0, float((ud + ur) & 255u) / 255.0,
             float((sh ^ int(ush)) & 255) / 255.0,
             float((mn == (-2147483647 - 1) ? 1 : 0) + (nm & 3) + mr % 2) / 6.0);
}`},
	// Left-to-right evaluation with side effects, && and || that skip a call, ?: that calls, ++ and --, swizzled self-assignment, global initializers.
	{name = "edge_order", source = `int counter = 0;
float g = 0.25;
int tick() { counter++; return counter; }
bool side(bool v) { counter += 10; return v; }
void mainImage(out vec4 c, in vec2 f) {
    int x = int(f.x);
    bool a = (x % 2 == 0) && side(x % 3 == 0);
    bool b = (x % 5 == 0) || side(true);
    int t = (x % 4 == 0) ? tick() : tick() + tick();
    int e = counter + (counter = 5) * 2 + counter;
    float h = g; g += 0.5;
    int i = 2; int j = i++ + i++ * ++i;
    vec2 v = vec2(1.0, 2.0); v = v.yx; v.xy = v.yx * 2.0;
    c = vec4(float(counter) / 32.0, float(a) * 0.5 + float(b) * 0.25, float(t + e + j) / 64.0, h + g * 0.1 + v.x * 0.01);
}`},
	// A uniform bool, bvec construction, any, all, not, mix by bvec, == on bvecs, ^^.
	{name = "edge_bools", source = `uniform bool uInvert;
void mainImage(out vec4 c, in vec2 f) {
    bvec3 bv = bvec3(f.x > 100.0, f.y > 50.0, !uInvert);
    bool any_ = any(bv), all_ = all(bv);
    vec3 m = mix(vec3(0.1), vec3(0.9), bv);
    bvec3 nb = not(bv);
    bool eq = bv == bvec3(true, false, true);
    bool x = bv.x ^^ bv.y;
    c = vec4(m.x * float(any_), m.y * float(all_), float(nb.z) + float(eq) * 0.5 + float(x) * 0.25, uInvert ? 0.0 : 1.0);
}`},
	// Discard inside a loop and in a while (true).
	{name = "edge_discard", source = `void mainImage(out vec4 c, in vec2 f) {
    int x = int(f.x);
    for (int i = 0; i < 100; i++) {
        if (i == x % 13 && x % 3 == 0) discard;
    }
    while (true) { if (x % 7 == 1) discard; break; }
    c = vec4(f / iResolution.xy, 0.5, 1.0);
}`},
	// The built-ins DXBC has no instruction for (atan, atan2, asin, acos, hyperbolics, round, pow, exp, log), a 4x4 inverse and determinant, refract, reflect, faceforward.
	{name = "edge_math", source = `void mainImage(out vec4 c, in vec2 f) {
    vec2 p = f / iResolution.xy * 2.0 - 1.0;
    float a = atan(p.y, p.x) / 6.2831853 + 0.5;
    float b = asin(clamp(p.x * 1.1, -1.0, 1.0)) * 0.3 + acos(clamp(p.y, -1.0, 1.0)) * 0.2;
    float s = sinh(p.x) * 0.2 + cosh(p.y) * 0.2 + tanh(p.x * 3.0) * 0.3 + asinh(p.y) * 0.1 + atanh(p.x * 0.9) * 0.1 + acosh(1.0 + abs(p.y)) * 0.1;
    float r = round(p.x * 7.5) / 8.0 + roundEven(p.y * 7.5) / 16.0 + trunc(p.x * 3.3) * 0.1 + sign(p.y) * 0.05;
    float e = exp(p.x) * 0.1 + log(abs(p.y) + 0.01) * 0.05 + pow(abs(p.x) + 0.1, 1.7) + exp2(p.y) * 0.1 + log2(abs(p.x) + 0.5) * 0.1 + inversesqrt(abs(p.y) + 0.2) * 0.1;
    mat4 m = mat4(1.0 + p.x, 0.2, 0.1, 0.0, 0.3, 1.0, 0.0, 0.1, 0.0, 0.2, 1.0 + p.y, 0.3, 0.1, 0.0, 0.2, 1.0);
    mat4 inv = inverse(m);
    float d = determinant(m) * 0.3 + inv[2][1] + inv[0][0] * 0.2;
    vec3 rf = refract(normalize(vec3(p, -1.0)), vec3(0.0, 0.0, 1.0), 0.75) + reflect(vec3(p, 1.0), normalize(vec3(0.3, 0.4, 0.5))) * 0.1;
    vec3 ff = faceforward(vec3(1.0, 0.0, 0.0), vec3(p, 0.5), vec3(0.0, 1.0, 0.0));
    c = vec4(fract(a + b + s), fract(r + e + d), fract(rf.x + rf.y + ff.x * 0.1), 1.0);
}`},
	// Every loop's continue inside a switch: a do-while's test (with a side effect) and a for's step run once per pass, not again at the continue.
	{name = "edge_flow", source = `void mainImage(out vec4 c, in vec2 f) {
    int x = int(f.x) % 5, y = int(f.y) % 4;
    int k = 0, after = 0, evals = 0;
    do {
        k++;
        switch ((x + k) % 3) {
        case 1: continue;
        case 2: after += 10; break;
        }
        after += 100;
    } while ((++evals) < y + 2);
    int steps = 0, s2 = 0;
    for (int i = 0; i < 6; i++, steps += 3) {
        switch ((i + y) % 3) {
        case 0: continue;
        default: s2 += i;
        }
        s2 += 7;
    }
    int w = 0, s3 = 0;
    while (w < x + 3) {
        w++;
        switch (w % 2) { case 0: continue; }
        s3 += w;
    }
    c = vec4(float(k * 1000 + after), float(evals * 100 + steps), float(s2), float(s3 * 100 + w));
}`, f32 = true},
	// Run-time indexes below 0 and at or past the end of arrays, vectors and matrices (clamped), and of storage buffers (0 past the end): element * stride wraps in 32-bit words, not bytes (rows[1 << 28] is past the end, rows[1 << 30] is rows[0]), a field's offset is added in words, a component index is clamped.
	{name = "edge_bounds", source = `#version 310 es
precision highp float;
struct P { vec2 a; float b; vec4 c; };
readonly buffer Rows { vec4 rows[]; };
readonly buffer Ps { P ps[]; };
uniform int uMode;
out vec4 o;
void main() {
    int k = uMode;
    int big = k << 28;
    int px = int(gl_FragCoord.x) % 8, py = int(gl_FragCoord.y) % 6;
    int i = (px - 3) * 3 * k;
    float arr[4] = float[4](1.0, 2.0, 3.0, 4.0);
    vec4 v = vec4(10.0, 20.0, 30.0, 40.0);
    mat3 m = mat3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    int rix[8] = int[8](big, -big, 1023 * k, 1024 * k, 0x0fffffff * k, -k, k << 30, 7 * k);
    int pix[8] = int[8](big, -big, 511 * k, 512 * k, 0x0fffffff * k, -k, k << 29, 3 * k);
    if (py == 0) {
        o = vec4(arr[i], arr[-k], arr[k * 7], v[i]);
    } else if (py == 1) {
        o = vec4(v[-k], v[k * 5], m[i].y, m[k][i]);
    } else if (py == 2) {
        o = rows[rix[px]];
    } else if (py == 3) {
        o = vec4(rows[px * k][i], rows[rix[px]][k], rows[rix[px]].w, rows[1023 * k][i]);
    } else if (py == 4) {
        o = ps[pix[px]].c;
    } else {
        o = vec4(ps[pix[px]].a, ps[pix[px]].b, ps[big].b);
    }
}`, glsl = true, f32 = true},
	// textureGrad with different x and y gradients, cross of vectors that are not symmetric.
	{name = "edge_grad_cross", source = `void mainImage(out vec4 c, in vec2 f) {
    vec2 uv = f / iResolution.xy;
    vec4 g = textureGrad(iChannel0, uv * 3.0, vec2(0.03, 0.002), vec2(0.0005, 0.004));
    vec3 a = vec3(uv, 0.3), b = vec3(0.2, uv.y * 0.5, 1.0 - uv.x);
    vec3 x = cross(a, b) * 0.5 + 0.5;
    c = vec4(g.r, g.g, x.x, x.y * 0.5 + x.z * 0.5);
}`},
	// Channels other than 0 passed through functions (each channel's image differs).
	{name = "edge_channels", source = `vec4 tap(sampler2D s, vec2 p) { return texture(s, p); }
vec4 pair(sampler2D a, sampler2D b, vec2 p) { return tap(b, p) * 0.75 + tap(a, p.yx) * 0.25; }
void mainImage(out vec4 c, in vec2 f) {
    vec2 uv = f / iResolution.xy;
    c = vec4(pair(iChannel0, iChannel1, uv).rgb * 0.5 + tap(iChannel2, uv * 2.0).gbr * 0.5, 1.0);
}`},
	// texelFetch at a uint lod of 2^31 (the coarsest level, not level 0) and at level 2.
	{name = "edge_texel_lod", source = `uniform int uMode;
void mainImage(out vec4 c, in vec2 f) {
    ivec2 p = ivec2(f) % 64;
    uint far = uint(uMode) << 31;
    c = vec4(texelFetch(iChannel0, p, far).rgb, 1.0) * 0.5 + texelFetch(iChannel1, p / 4, 2) * 0.5;
}`},
	// A discard before a loop that only ends where nothing discards: discard ends the invocation (the cap keeps a regression from wedging the GPU).
	{name = "edge_discard_loop", source = `uniform int uInvert;
void mainImage(out vec4 c, in vec2 f) {
    float stepv = (int(f.x) % 4 < 2) ? float(uInvert) : 0.25;
    if (stepv <= 0.0) discard;
    float s = 0.0;
    int n = 0;
    while (s < 1.0 && n < 65536) { s += stepv; n++; }
    c = vec4(float(n), s, stepv, 1.0);
}`, f32 = true},
	// The expansions DXBC has no instruction for, at NaN, infinities, signed zeros, tiny, large and ordinary inputs from uniforms (nothing folds them): column f / 16 is the function, row f / 9 the input (MATH_INPUTS).
	{name = "edge_math_values", source = `uniform float uIn[16];
void mainImage(out vec4 c, in vec2 f) {
    int fn = int(f.x) / 16, i = int(f.y) / 9;
    float v = uIn[i];
    float r = 0.0;
    switch (fn) {
    case 0: r = asin(v); break;
    case 1: r = acos(v); break;
    case 2: r = atan(v); break;
    case 3: r = sinh(v); break;
    case 4: r = cosh(v); break;
    case 5: r = tanh(v); break;
    case 6: r = asinh(v); break;
    case 7: r = acosh(v); break;
    case 8: r = atanh(v); break;
    case 9: r = atan(v, 1.0); break;
    case 10: r = atan(1.0, v); break;
    case 11: r = atan(v, -1.0); break;
    case 12: r = atan(v, v); break;
    case 13: r = atan(v, -v); break;
    case 14: r = atan(-v, v); break;
    default: r = atan(v, uIn[(i + 1) % 16]); break;
    }
    c = vec4(r, 0.0, 0.0, 1.0);
}`, f32 = true, tol = {rel = 2e-5}, setup = math_setup, skip = math_skip},
}

// edge_math_values' inputs.
MATH_INPUTS := [16]f32{0h7fc00000, 0h7f800000, 0hff800000, 0, 0h80000000, 1e-8, -1e-8, 0.3, -0.75, 0.999, 3, -20, 70000, -88, 88.5, 1e20}

math_setup :: proc(m: ^oksl.Module, block: []u8) {
	for v, i in MATH_INPUTS {
		oksl.set_uniform(m, block, "uIn", {v}, i)
	}
}

// atan(y, x) with both zero, which GLSL leaves undefined: columns 12 to 14
// at the zeros, column 15 at (0, -0). Image row y is fragCoord.y H - 1 - y
// (Shadertoy's origin is the bottom left).
math_skip :: proc(x, y: int) -> bool {
	fn, i := x / 16, (parity.H - 1 - y) / 9
	return fn >= 12 && fn <= 14 && (i == 3 || i == 4) || fn == 15 && i == 3
}
