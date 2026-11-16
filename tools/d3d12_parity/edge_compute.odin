#+build linux amd64, windows amd64
package d3d12_parity

import parity "../parity"

// A compute shader of the DXBC emitter's own, for what testdata does not
// cover, checked as testdata's are (compute_check, parity.compute_compare:
// every word of every writable buffer and texel of every writable image).
// Each invocation writes only words and texels no other one touches, or
// folds contended results so they do not depend on order.
Edge_Compute :: struct {
	name, source: string,
	shape:        parity.Compute_Shape,
}

EDGE_COMPUTE := [?]Edge_Compute {
	// The system values in a 4x2x3 thread group over 8x10x3 groups, and
	// gl_WorkGroupSize; invocations past the buffer's end write nothing.
	{"edge_cs_ids", `#version 310 es
layout(local_size_x = 4, local_size_y = 2, local_size_z = 3) in;
buffer Out { uint w[]; } o;
void main() {
    uvec3 g = gl_GlobalInvocationID;
    uvec3 l = gl_LocalInvocationID;
    uvec3 wg = gl_WorkGroupID;
    uint i = (g.z * 20u + g.y) * 32u + g.x;
    uint flat = l.z * gl_WorkGroupSize.x * gl_WorkGroupSize.y + l.y * gl_WorkGroupSize.x + l.x;
    o.w[i] = l.x | (l.y << 2u) | (l.z << 3u) | (gl_LocalInvocationIndex << 5u) | ((flat ^ gl_LocalInvocationIndex) << 10u)
        | (wg.x << 12u) | (wg.y << 16u) | (wg.z << 20u) | (gl_WorkGroupSize.z << 24u);
}`, {29, 17, 4096, false}},
	// Storage buffers of 4093 words, read-only and read-write: reads and
	// writes past the end, a vec4, uvec3, uvec2 and struct straddling it
	// (the words inside are written, the others dropped, as on the CPU;
	// reads give 0 past it), indexes that
	// wrap at 2^32 (element 2^30 of a vec4 buffer is element 0), are past
	// 2^30 words (whose byte addresses would wrap without the clamp) or are
	// negative, a component chosen at run time (clamped), a swizzled store,
	// compound assignment, ++, and an inout argument.
	{"edge_cs_buffers", `#version 310 es
layout(local_size_x = 64) in;
readonly buffer In { uvec4 r[]; } src;
readonly buffer In2 { vec2 q[]; } src2;
buffer V { vec4 v[]; } bv;
buffer U3 { uvec3 u[]; } b3;
buffer U2 { uvec2 p[]; } b2;
buffer Out { uvec4 o[]; } bo;
buffer RW { uvec4 r[]; } rr;
struct T { float x; vec2 y; uint n; };
buffer St { T s[]; } bs;
void bump(inout vec4 x, float by) { x = x * 0.5 + by; }
void main() {
    uint t = gl_GlobalInvocationID.y * 128u + gl_GlobalInvocationID.x;
    if (t < 16u) {
        bv.v[1015u + t] = vec4(float(t), float(t) + 0.5, -float(t), 3.0);
        b3.u[1015u + t] = uvec3(t, t * 3u, 0xffffffffu - t);
        b2.p[2040u + t] = uvec2(t + 7u, t * 11u);
        bv.v[0x40000000u + t] = vec4(7.0, float(t), 8.0, 9.0);
        bv.v[-1 - int(t)] = vec4(100.0);
        bv.v[0x10000000u + t] = vec4(55.0);
        b2.p[0x40000000u + t] = uvec2(99u, 98u);
        bo.o[t] = src.r[1015u + t];
        bo.o[16u + t] = uvec4(floatBitsToUint(src2.q[2040u + t]), src.r[0x40000000u + t].xy);
        bo.o[32u + t] = rr.r[1020u + t] + uvec4(rr.r[0x40000000u + t].w);
        bo.o[48u + t] = src.r[-1 - int(t)] + uvec4(src.r[t][int(t) - 2]);
    } else if (t < 32u) {
        uint k = t - 16u;
        bv.v[900u + k][int(k) - 2] = float(k) * 0.25;
        bv.v[920u + k].wy = vec2(float(k), -1.0);
        bv.v[940u + k].x += float(k);
        b3.u[940u + k].z++;
        ++b3.u[960u + k][int(k) % 5];
        bump(bv.v[960u + k], float(k));
        bs.s[675u + k] = T(float(k), vec2(1.5, float(k)), k * 5u);
        bs.s[640u + k].y.y -= 2.0;
        bo.o[64u + k] = uvec4(floatBitsToUint(bs.s[675u + k].y), bs.s[675u + k].n, floatBitsToUint(bs.s[700u + k].x));
    }
}`, {29, 17, 4093, false}},
	// Every atomic on int and uint words of its own, in buffers and in
	// shared memory, with the old value used (every op, compare-and-swap
	// matching and not) and unused; contended compare-and-swap loops
	// adding to one word (the sums do not depend on order); buffer atomics
	// past the end (dropped, 0) and at negative and wrapping indexes.
	{"edge_cs_atomics", `#version 310 es
layout(local_size_x = 32) in;
buffer I { int w[]; } bi;
buffer U { uint w[]; } bu;
buffer RI { int w[]; } ri;
buffer RU { uint w[]; } ru;
buffer SI { int w[]; } si;
buffer SU { uint w[]; } su;
shared int sh_i[512];
shared uint sh_u[512];
shared uint sh_count;
shared int sh_signed;
void main() {
    uint lid = gl_LocalInvocationIndex;
    uint wg = gl_WorkGroupID.y * 2u + gl_WorkGroupID.x;
    uint t = wg * 32u + lid;
    int v = int(t * 2654435761u) >> 3;
    uint uv = t * 2246822519u;
    if (t < 240u) {
        uint b = t * 16u;
        ri.w[b + 0u] = atomicAdd(bi.w[b + 0u], v);
        ri.w[b + 1u] = atomicMin(bi.w[b + 1u], v);
        ri.w[b + 2u] = atomicMax(bi.w[b + 2u], v);
        ri.w[b + 3u] = atomicAnd(bi.w[b + 3u], v);
        ri.w[b + 4u] = atomicOr(bi.w[b + 4u], v);
        ri.w[b + 5u] = atomicXor(bi.w[b + 5u], v);
        ri.w[b + 6u] = atomicExchange(bi.w[b + 6u], v);
        ri.w[b + 7u] = atomicCompSwap(bi.w[b + 7u], bi.w[b + 7u], v);
        ri.w[b + 8u] = atomicCompSwap(bi.w[b + 8u], 12345, v);
        atomicAdd(bi.w[b + 9u], v);
        atomicMin(bi.w[b + 10u], v);
        atomicMax(bi.w[b + 11u], v);
        atomicAnd(bi.w[b + 12u], v);
        atomicOr(bi.w[b + 13u], v);
        atomicXor(bi.w[b + 14u], v);
        atomicExchange(bi.w[b + 15u], v);
        ru.w[b + 0u] = atomicAdd(bu.w[b + 0u], uv);
        ru.w[b + 1u] = atomicMin(bu.w[b + 1u], uv);
        ru.w[b + 2u] = atomicMax(bu.w[b + 2u], uv);
        ru.w[b + 3u] = atomicAnd(bu.w[b + 3u], uv);
        ru.w[b + 4u] = atomicOr(bu.w[b + 4u], uv);
        ru.w[b + 5u] = atomicXor(bu.w[b + 5u], uv);
        ru.w[b + 6u] = atomicExchange(bu.w[b + 6u], uv);
        ru.w[b + 7u] = atomicCompSwap(bu.w[b + 7u], bu.w[b + 7u], uv);
        ru.w[b + 8u] = atomicCompSwap(bu.w[b + 8u], 12345u, uv);
        atomicAdd(bu.w[b + 9u], uv);
        atomicMin(bu.w[b + 10u], uv);
        atomicMax(bu.w[b + 11u], uv);
        atomicAnd(bu.w[b + 12u], uv);
        atomicOr(bu.w[b + 13u], uv);
        atomicXor(bu.w[b + 14u], uv);
        atomicExchange(bu.w[b + 15u], uv);
    } else if (t < 272u) {
        uint k = 3840u + (t - 240u);
        ri.w[k] = atomicAdd(bi.w[4096u + t], 5) + atomicExchange(bi.w[-1 - int(t)], 6);
        ru.w[k] = atomicMax(bu.w[0x40000000u + t], 1u) + atomicCompSwap(bu.w[0xffffffffu - t], 0u, 1u);
        atomicOr(bi.w[5000u + t], 1);
        atomicMin(bu.w[uint(-int(t))], 0u);
    }
    // Contended: every invocation adds (t % 7 + 1) to one word.
    uint add = t % 7u + 1u;
    uint seen = bu.w[3900u];
    for (int i = 0; i < 100000; ++i) {
        uint got = atomicCompSwap(bu.w[3900u], seen, seen + add);
        if (got == seen) {
            break;
        }
        seen = got;
    }
    int sseen = bi.w[3901u];
    for (int i = 0; i < 100000; ++i) {
        int got = atomicCompSwap(bi.w[3901u], sseen, sseen - int(add));
        if (got == sseen) {
            break;
        }
        sseen = got;
    }

    // Shared memory: each lane's 16 words of each.
    uint s = lid * 16u;
    for (uint k = 0u; k < 16u; ++k) {
        sh_i[s + k] = int((t + k) * 40503u) - 30000;
        sh_u[s + k] = (t ^ k) * 0x9e3779b9u;
    }
    if (lid == 0u) {
        sh_count = 0u;
        sh_signed = 0;
    }
    barrier();
    int ri_[9];
    uint ru_[9];
    ri_[0] = atomicAdd(sh_i[s + 0u], v);
    ri_[1] = atomicMin(sh_i[s + 1u], v);
    ri_[2] = atomicMax(sh_i[s + 2u], v);
    ri_[3] = atomicAnd(sh_i[s + 3u], v);
    ri_[4] = atomicOr(sh_i[s + 4u], v);
    ri_[5] = atomicXor(sh_i[s + 5u], v);
    ri_[6] = atomicExchange(sh_i[s + 6u], v);
    ri_[7] = atomicCompSwap(sh_i[s + 7u], sh_i[s + 7u], v);
    ri_[8] = atomicCompSwap(sh_i[s + 8u], 12345, v);
    atomicAdd(sh_i[s + 9u], v);
    atomicMin(sh_i[s + 10u], v);
    atomicMax(sh_i[s + 11u], v);
    atomicAnd(sh_i[s + 12u], v);
    atomicOr(sh_i[s + 13u], v);
    atomicXor(sh_i[s + 14u], v);
    atomicExchange(sh_i[s + 15u], v);
    ru_[0] = atomicAdd(sh_u[s + 0u], uv);
    ru_[1] = atomicMin(sh_u[s + 1u], uv);
    ru_[2] = atomicMax(sh_u[s + 2u], uv);
    ru_[3] = atomicAnd(sh_u[s + 3u], uv);
    ru_[4] = atomicOr(sh_u[s + 4u], uv);
    ru_[5] = atomicXor(sh_u[s + 5u], uv);
    ru_[6] = atomicExchange(sh_u[s + 6u], uv);
    ru_[7] = atomicCompSwap(sh_u[s + 7u], sh_u[s + 7u], uv);
    ru_[8] = atomicCompSwap(sh_u[s + 8u], 12345u, uv);
    atomicAdd(sh_u[s + 9u], uv);
    atomicMin(sh_u[s + 10u], uv);
    atomicMax(sh_u[s + 11u], uv);
    atomicAnd(sh_u[s + 12u], uv);
    atomicOr(sh_u[s + 13u], uv);
    atomicXor(sh_u[s + 14u], uv);
    atomicExchange(sh_u[s + 15u], uv);
    uint cs = sh_count;
    for (int i = 0; i < 100000; ++i) {
        uint got = atomicCompSwap(sh_count, cs, cs + lid + 1u);
        if (got == cs) {
            break;
        }
        cs = got;
    }
    atomicAdd(sh_signed, -int(lid));
    barrier();
    if (wg < 8u) {
        uint o = wg * 512u + s;
        for (uint k = 0u; k < 16u; ++k) {
            si.w[o + k] = sh_i[s + k];
            su.w[o + k] = sh_u[s + k];
        }
        if (lid < 9u) {
            ri.w[3900u + wg * 9u + lid] = ri_[lid];
            ru.w[3900u + wg * 9u + lid] = ru_[lid];
        }
        if (lid == 0u) {
            ri.w[3990u + wg] = sh_signed;
            ru.w[3990u + wg] = sh_count;
        }
    }
}`, {29, 17, 4096, false}},
	// barrier() in loops, in a helper, after a uniform early return from
	// main and from a helper; memoryBarrierShared, groupMemoryBarrier (with
	// buffer words written and read back across the group) and
	// memoryBarrierBuffer / memoryBarrierImage / memoryBarrier between.
	{"edge_cs_barriers", `#version 310 es
layout(local_size_x = 64) in;
buffer O { uint w[]; } o;
buffer X { uint w[]; } x;
shared uint s[64];
shared uint acc;
void rotate(uint lid, uint k) {
    if (k == 3u) {
        return;
    }
    uint v = s[(lid + k + 1u) & 63u];
    barrier();
    s[lid] = v * 3u + k;
    memoryBarrierShared();
    barrier();
}
void main() {
    uint lid = gl_LocalInvocationIndex;
    uint wg = gl_WorkGroupID.y * 2u + gl_WorkGroupID.x;
    if (gl_WorkGroupID.y == 3u) {
        return;
    }
    s[lid] = lid * 7u + wg;
    if (lid == 0u) {
        acc = 0u;
    }
    barrier();
    for (uint k = 0u; k < 6u; ++k) {
        rotate(lid, k);
    }
    for (int k = 0; k < 4; ++k) {
        uint v = s[63u - lid];
        barrier();
        s[lid] = v ^ uint(k);
        barrier();
    }
    x.w[wg * 64u + lid] = s[lid] + 1u;
    groupMemoryBarrier();
    barrier();
    uint other = x.w[wg * 64u + (lid ^ 1u)];
    memoryBarrierBuffer();
    memoryBarrierImage();
    memoryBarrier();
    atomicAdd(acc, s[lid]);
    barrier();
    o.w[wg * 64u + lid] = (s[lid] ^ acc) + other;
}`, {29, 17, 4096, false}},
	// Shared structs (a vec3, a bool, a mat2, an int[3]), arrays of them,
	// a mat3, a bool array and a float array: whole-struct and whole-array
	// loads and stores, fields, matrix columns and components at run-time
	// indexes, in range and clamped out of it, as on the CPU.
	{"edge_cs_shared", `#version 310 es
layout(local_size_x = 16) in;
struct P { vec3 pos; bool alive; mat2 m; int ids[3]; };
shared P parts[16];
shared mat3 sm;
shared bool flags[8];
shared float row[5];
buffer O { vec4 v[]; } o;
buffer OI { ivec4 v[]; } oi;
void main() {
    uint lid = gl_LocalInvocationIndex;
    uint wg = gl_WorkGroupID.y * 4u + gl_WorkGroupID.x;
    int i = int(lid);
    parts[lid].pos = vec3(float(i), float(2 * i), -float(i));
    parts[lid].alive = (lid & 1u) == 0u;
    parts[lid].m = mat2(float(i), 1.0, 2.0, float(wg));
    parts[lid].ids = int[3](i, -i, i * i);
    parts[lid].ids[i % 3] += 100;
    if (lid < 8u) {
        flags[lid] = lid > 2u;
    }
    if (lid == 0u) {
        sm = mat3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, float(wg));
        row = float[5](0.5, 1.5, 2.5, 3.5, 4.5);
    }
    barrier();
    P q = parts[(lid + 5u) & 15u];
    float arr[5] = row;
    barrier();
    parts[lid] = q;
    parts[lid].m[i % 2][i - 8] = -1.0;
    if (lid == 3u) {
        sm[i - 5][i % 4] += float(i);
    }
    barrier();
    vec4 a = vec4(q.pos, q.alive ? 1.0 : 0.0) + vec4(q.m[1], q.m[0]);
    float e = parts[i - 4].pos[i - 1] + parts[i + 9].m[1][i];
    vec3 c = sm[i % 3] + vec3(sm[2][i - 10]);
    bool f = flags[i - 3] || flags[i + 20];
    uint k = wg * 32u + lid * 2u;
    o.v[k] = a + vec4(e);
    o.v[k + 1u] = vec4(c, (f ? 10.0 : 20.0) + arr[i % 5] + row[i]);
    oi.v[k] = ivec4(q.ids[0], q.ids[1], q.ids[2], parts[i - 2].ids[i]);
    oi.v[k + 1u] = ivec4(parts[lid].ids[0] - parts[15u - lid].ids[2], int(parts[lid].alive), i, -i);
}`, {29, 17, 4096, false}},
	// Sampler and image parameters of a function called again in a later
	// argument (each names its own argument), a writable r32f image passed
	// to a readonly image2D parameter, read at the invocation's own texel.
	{"edge_cs_params", `#version 310 es
layout(local_size_x = 8, local_size_y = 8) in;
uniform highp sampler2D tA;
uniform highp sampler2D tB;
layout(r32f) uniform highp image2D rw;
layout(r32f) readonly uniform highp image2D ro;
buffer O { vec4 v[]; } o;
float tap(readonly highp image2D im, ivec2 q, float bias) {
    return imageLoad(im, q).x * 2.0 + bias;
}
vec4 fetch(highp sampler2D s, ivec2 q) {
    return texelFetch(s, q, 0);
}
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    uint k = uint(p.y * 32 + p.x);
    vec4 b = fetch(tA, ivec2(fetch(tB, p).xy * 63.0));
    o.v[k] = vec4(tap(rw, p, tap(ro, p, 1.0)), b.xyz);
}`, {29, 17, 4096, false}},
	// Every storage image format read in and out of bounds (negative and
	// past the size; out of bounds every component is 0, a single channel
	// format's w too, as on the CPU), the read-write r32 formats loaded at
	// their own texel and outside, every writable format stored in and out
	// of bounds (dropped), values UNORM8 and half floats clamp or round,
	// imageSize of each; 10 UAVs, so u8 and u9 (SFI0's 64-UAV flag).
	{"edge_cs_images", `#version 310 es
layout(local_size_x = 8, local_size_y = 8) in;
layout(rgba8) readonly uniform highp image2D i8;
layout(rgba16f) readonly uniform highp image2D i16;
layout(rgba32f) readonly uniform highp image2D i32;
layout(r32f) readonly uniform highp image2D if_;
layout(r32ui) readonly uniform highp uimage2D iu;
layout(r32i) readonly uniform highp iimage2D ii;
layout(r32f) uniform highp image2D rf;
layout(r32ui) uniform highp uimage2D ru;
layout(r32i) uniform highp iimage2D ri;
layout(rgba8) writeonly uniform highp image2D w8;
layout(rgba16f) writeonly uniform highp image2D w16;
layout(rgba32f) writeonly uniform highp image2D w32;
buffer OF { vec4 v[]; } of;
buffer OF2 { vec4 v[]; } of2;
buffer OU { uvec4 v[]; } ou;
buffer OI { ivec4 v[]; } oi;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    ivec2 q = p * 2 - ivec2(3, 5);
    ivec2 neg = ivec2(-1 - p.x, p.y);
    uint k = uint(p.y * 32 + p.x) * 2u;
    of.v[k] = imageLoad(i8, q);
    of.v[k + 1u] = imageLoad(i16, q);
    of2.v[k] = imageLoad(i32, q);
    of2.v[k + 1u] = imageLoad(if_, q) + imageLoad(rf, neg) * 2.0 + imageLoad(if_, neg) * 4.0;
    ou.v[k] = imageLoad(iu, q) + imageLoad(ru, neg) * 3u;
    oi.v[k] = imageLoad(ii, q) + imageLoad(ri, neg) * 5;
    vec4 f0 = imageLoad(rf, p);
    uvec4 u0 = imageLoad(ru, p);
    ivec4 i0 = imageLoad(ri, p);
    ou.v[k + 1u] = uvec4(floatBitsToUint(f0.x), floatBitsToUint(f0.w), u0.x, u0.w);
    oi.v[k + 1u] = i0;
    imageStore(rf, p, vec4(f0.x * 2.0 + 1.0, 5.0, 6.0, 7.0));
    imageStore(ru, p, uvec4(u0.x * 3u + uint(p.x), 1u, 2u, 3u));
    imageStore(ri, p, ivec4(i0.x - p.y * 9, 1, 2, 3));
    vec4 c = vec4(float(p.x) / 28.0, float(p.y) / 16.0 * 1.5 - 0.25, -0.5, 0.5 + float(p.x & 3) / 255.0);
    imageStore(w8, p, c);
    imageStore(w16, p, c * 300.0 + vec4(0.0, 0.0, 1.0 / 3.0, 1e-6));
    imageStore(w32, p, c * c - 0.125);
    imageStore(w8, neg, vec4(1.0));
    imageStore(w16, ivec2(p.x, p.y + 17), vec4(9.0));
    imageStore(w32, ivec2(p.x + 29, -1 - p.y), vec4(9.0));
    imageStore(rf, neg, vec4(9.0));
    imageStore(ru, ivec2(p.x + 29, p.y), uvec4(9u));
    imageStore(ri, ivec2(p.x, -1), ivec4(9));
    if (p == ivec2(0)) {
        oi.v[1600] = ivec4(imageSize(i8), imageSize(i16));
        oi.v[1601] = ivec4(imageSize(i32), imageSize(if_));
        oi.v[1602] = ivec4(imageSize(iu), imageSize(ii));
        oi.v[1603] = ivec4(imageSize(rf), imageSize(ru));
        oi.v[1604] = ivec4(imageSize(ri), imageSize(w8));
        oi.v[1605] = ivec4(imageSize(w16), imageSize(w32));
    }
}`, {29, 17, 8192, false}},
}
