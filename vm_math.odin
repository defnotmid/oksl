package oksl

import "base:intrinsics"

// The CPU's float built-ins that libm would otherwise compute: sin, cos,
// tan, asin, acos, atan (one and two arguments), sinh, cosh, tanh, asinh,
// acosh, atanh, exp, exp2, log, log2 and pow, for the interpreter and the
// JIT's helper alike. Each works on eight f32 lanes in f32 with IEEE-754
// basic operations only (+, -, *, /, sqrt, comparisons, conversions and bit
// operations: no FMA, which Odin never contracts into, and no libm), so its
// result is the same bits on every IEEE platform. The accuracy is a GPU's,
// within the Vulkan spec's bounds for GLSL.std.450 and a few ulp over the
// ranges shaders use (test_math_ulp states each function's bound). Lanes
// are whole SIMD vectors and every choice a select, but for pow's rare
// operands and its y = 2, tested once per call; NaN results are any NaN
// (outputs canonicalize, `canonical`).
//
// The polynomials are minimax fits of the relative error (Remez), each
// named with its interval and error, rounded to f32.

@(private)
F8 :: #simd[8]f32
@(private = "file")
W8 :: #simd[8]u32
@(private = "file")
I8 :: #simd[8]i32

@(private = "file")
RND :: f32(12582912) // 1.5 * 2^23: v + RND - RND rounds v (|v| < 2^22) to an integer
@(private = "file")
RND_BITS :: i32(0x4B40_0000)
@(private = "file")
PIO2_HI :: f32(0h3FC90FDB)
@(private = "file")
PIO2_LO :: f32(0hB33BBD2E) // pi/2 - PIO2_HI
@(private = "file")
PI_HI :: f32(0h40490FDB)
@(private = "file")
PI_LO :: f32(0hB3BBBD2E)
@(private = "file")
PIO4 :: f32(0h3F490FDB)
@(private = "file")
LN2_HI :: f32(0h3F317200) // 15 bits: k * LN2_HI is exact for |k| < 2^9
@(private = "file")
LN2_LO :: f32(0h35BFBE8E)
@(private = "file")
LN2 :: f32(0h3F317218)
@(private = "file")
LOG2E :: f32(0h3FB8AA3B)
@(private = "file")
INF :: f32(0h7F80_0000)
@(private = "file")
NAN :: f32(0h7FC0_0000)
@(private = "file")
SIGN :: u32(0x8000_0000)

@(private = "file")
sel :: #force_inline proc "contextless" (c: $M, a, b: $T) -> T {
	return intrinsics.simd_select(c, a, b)
}

// x's sign bits on r's.
@(private = "file")
with_sign :: #force_inline proc "contextless" (r, x: F8) -> F8 {
	return transmute(F8)(transmute(W8)r ~ (transmute(W8)x & W8(SIGN)))
}

@(private = "file")
is_nan :: #force_inline proc "contextless" (x: F8) -> W8 {
	return intrinsics.simd_lanes_ne(x, x)
}

// 2^k for k in [-126, 127].
@(private = "file")
pow2 :: #force_inline proc "contextless" (k: I8) -> F8 {
	return transmute(F8)intrinsics.simd_shl(transmute(W8)(k + 127), W8(23))
}

// x clamped to [lo, hi], NaN kept.
@(private = "file")
clamp_nan :: #force_inline proc "contextless" (x: F8, lo, hi: f32) -> F8 {
	c := sel(intrinsics.simd_lanes_gt(x, F8(hi)), F8(hi), x)
	return sel(intrinsics.simd_lanes_lt(c, F8(lo)), F8(lo), c)
}

// ---- sin, cos, tan -----------------------------------------------------------------------

// |x| as r + n * pi/2: r and n's low two bits. n is the integer nearest
// |x| * 2/pi in f32, so |r| exceeds pi/4 by up to 0.0146 (|x| < 10^5).
// Cody-Waite: pi/2 in three parts, the first two of 12 bits, so n * P1, n *
// P2 and |x| - n * P1 are exact below n = 2^12 (|x| < 6434), and the three
// within 2^-57 of pi/2: r is within about an ulp of itself plus n * 2^-37
// there. Above, r's error grows as |x|'s ulp (the argument's own
// uncertainty); from |x| = 10^5 on r is clamped to the kernel's interval,
// so sin and cos stay within [-1, 1] even where n is no longer the nearest
// integer (|x| * 2/pi >= 2^22), at no particular value there. +-inf and NaN
// give NaN.
@(private = "file")
reduce_pio2 :: #force_inline proc "contextless" (ax: F8) -> (r: F8, q: W8) {
	P1 :: f32(0h3FC91000)
	P2 :: f32(0hB6957000)
	P3 :: f32(0hB06F4B9F)
	TWO_OVER_PI :: f32(0h3F22F983)
	w := ax * TWO_OVER_PI + RND
	n := w - RND
	r = clamp_nan(((ax - n * P1) - n * P2) - n * P3, -0.8, 0.8)
	q = transmute(W8)w
	return
}

// sin(r) and cos(r) for |r| <= 0.8.
@(private = "file")
sincos_kernel :: #force_inline proc "contextless" (r: F8) -> (s, c: F8) {
	// (sin(r) - r) / r^3 in z = r^2 on [0, 0.8^2], degree 2, 2^-27.8
	S0 :: f32(0hBE2AAAA2)
	S1 :: f32(0h3C08833D)
	S2 :: f32(0hB94C8127)
	// (cos(r) - 1 + r^2/2) / r^4 on the same interval, degree 2, 2^-32.7
	C0 :: f32(0h3D2AAAA4)
	C1 :: f32(0hBAB605B1)
	C2 :: f32(0h37CCD7D3)
	z := r * r
	z2 := z * z
	s = r + r * z * ((S0 + z * S1) + z2 * S2)
	c = (1 - 0.5 * z) + z2 * ((C0 + z * C1) + z2 * C2)
	return
}

// sin(q * pi/2 + r) from the kernel's values: q & 2 flips the sign bit.
@(private = "file")
quadrant :: #force_inline proc "contextless" (s, c: F8, q: W8) -> F8 {
	v := sel(intrinsics.simd_lanes_ne(q & 1, W8(0)), c, s)
	return transmute(F8)(transmute(W8)v ~ intrinsics.simd_shl(q & 2, W8(30)))
}

@(private)
fsin8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	r, q := reduce_pio2(intrinsics.simd_abs(x))
	s, c := sincos_kernel(r)
	return with_sign(quadrant(s, c, q), x)
}

@(private)
fcos8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	r, q := reduce_pio2(intrinsics.simd_abs(x))
	s, c := sincos_kernel(r)
	return quadrant(s, c, q + 1)
}

@(private)
ftan8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	r, q := reduce_pio2(intrinsics.simd_abs(x))
	s, c := sincos_kernel(r)
	odd := intrinsics.simd_lanes_ne(q & 1, W8(0))
	return with_sign(sel(odd, -c, s) / sel(odd, s, c), x)
}

// ---- exp, exp2 and the hyperbolic functions ----------------------------------------------

// e^r - 1 for |r| <= ln(2)/2 (1 + 2^-9).
@(private = "file")
expm1_kernel :: #force_inline proc "contextless" (r: F8) -> F8 {
	// (e^r - 1 - r) / r^2 on [-0.3473, 0.3473], degree 4, 2^-26.2 of e^r - 1
	E0 :: f32(0h3EFFFFFF)
	E1 :: f32(0h3E2AAA57)
	E2 :: f32(0h3D2AAB3B)
	E3 :: f32(0h3C091443)
	E4 :: f32(0h3AB5F5EF)
	r2 := r * r
	return r + r2 * ((E0 + r * E1) + r2 * ((E2 + r * E3) + r2 * E4))
}

// x = k * ln(2) + r for the integer k nearest x * log2(e) (|k| < 2^9), r
// rounded once: k * LN2_HI and x - k * LN2_HI are exact.
@(private = "file")
reduce_ln2 :: #force_inline proc "contextless" (x: F8) -> (r: F8, k: I8) {
	w := x * LOG2E + RND
	kf := w - RND
	r = (x - kf * LN2_HI) - kf * LN2_LO
	k = transmute(I8)w - RND_BITS
	return
}

// p * 2^k for k in [-252, 254], in two exact steps (the second rounds once
// into the subnormals or overflows to inf).
@(private = "file")
scale :: #force_inline proc "contextless" (p: F8, k: I8) -> F8 {
	h := intrinsics.simd_shr(k, W8(1))
	return p * pow2(h) * pow2(k - h)
}

// 2^x, x clamped to [-151, 129] (beyond f32's range either way).
@(private)
fexp28 :: #force_inline proc "contextless" (x: F8) -> F8 {
	xc := clamp_nan(x, -151, 129)
	w := xc + RND
	return scale(1 + expm1_kernel((xc - (w - RND)) * LN2), transmute(I8)w - RND_BITS)
}

@(private)
fexp8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	r, k := reduce_ln2(clamp_nan(x, -105, 90))
	return scale(1 + expm1_kernel(r), k)
}

// For a = |x| clamped to [0, 89.5] (sinh and cosh overflow from 89.42):
// e^a / 4 = t (1 + m) with t = 2^(k - 2) and m = e^r - 1, so (e^a - 1) / 4
// = (t - 1/4) + t m without cancellation (m / 4 exactly at k = 0) and every
// value finite.
@(private = "file")
exp_quarter :: #force_inline proc "contextless" (x: F8) -> (t, m: F8) {
	r, k := reduce_ln2(clamp_nan(intrinsics.simd_abs(x), 0, 89.5))
	return pow2(k - 2), expm1_kernel(r)
}

// sinh(a) = (E - 1/E) / 2 = 2 (q + q / (4q + 1)) with q = (e^a - 1) / 4.
@(private)
fsinh8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	t, m := exp_quarter(x)
	q := (t - 0.25) + t * m
	return with_sign(2 * (q + q / (4 * q + 1)), x)
}

// cosh(a) = 2 (e + 1 / (16 e)) with e = e^a / 4.
@(private)
fcosh8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	t, m := exp_quarter(x)
	e := t + t * m
	return 2 * (e + 0.0625 / e)
}

// tanh(a) = m / (m + 2) with m = e^(2a) - 1, a clamped to 9.5 (tanh rounds
// to 1 from 9.01).
@(private)
ftanh8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	r, k := reduce_ln2(2 * clamp_nan(intrinsics.simd_abs(x), 0, 9.5))
	t := pow2(k)
	m := (t - 1) + t * expm1_kernel(r)
	return with_sign(m / (m + 2), x)
}

// ---- log, log2, pow and the inverse hyperbolic functions ---------------------------------

// For finite x > 0, subnormals included: x = 2^e (1 + f) with 1 + f in
// [sqrt(2)/2, sqrt(2)) (f exact), e as an integer and a float.
@(private = "file")
log_split :: #force_inline proc "contextless" (x: F8) -> (e, f: F8, ei: I8) {
	sub := intrinsics.simd_lanes_lt(x, F8(0h0080_0000))
	xs := sel(sub, x * 8388608, x)
	ei = intrinsics.simd_shr(transmute(I8)xs - 0x3F35_04F3, W8(23))
	f = transmute(F8)(transmute(I8)xs - intrinsics.simd_shl(ei, W8(23))) - 1
	ei -= sel(sub, I8(23), I8(0))
	e = cast(F8)ei
	return
}

// log(1 + f) for f in [sqrt(2)/2 - 1, sqrt(2) - 1].
@(private = "file")
log1p_kernel :: #force_inline proc "contextless" (f: F8) -> F8 {
	// (log(1 + f) - f + f^2/2) / f^3 on [-0.2930, 0.4144], degree 7, 2^-27.3 of log(1 + f)
	L0 :: f32(0h3EAAAAAA)
	L1 :: f32(0hBE800114)
	L2 :: f32(0h3E4CD007)
	L3 :: f32(0hBE2A38DB)
	L4 :: f32(0h3E116C8C)
	L5 :: f32(0hBE06C504)
	L6 :: f32(0h3E02B2C2)
	L7 :: f32(0hBD9C5A7A)
	f2 := f * f
	f4 := f2 * f2
	p := ((L0 + f * L1) + f2 * (L2 + f * L3)) + f4 * ((L4 + f * L5) + f2 * (L6 + f * L7))
	return f - 0.5 * f2 + f * f2 * p
}

// log(x) or log2(x) for x's lanes, with log(+-0) = -inf, log(x < 0) = NaN
// and log(+inf) = +inf.
@(private = "file")
log_special :: #force_inline proc "contextless" (x, r: F8) -> F8 {
	res := sel(intrinsics.simd_lanes_eq(x, F8(0)), F8(-INF), r)
	res = sel(intrinsics.simd_lanes_eq(x, F8(INF)), x, res)
	return sel(intrinsics.simd_lanes_ge(x, F8(0)), res, F8(NAN))
}

@(private)
flog8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	e, f, _ := log_split(x)
	return log_special(x, e * LN2_HI + (log1p_kernel(f) + e * LN2_LO))
}

@(private)
flog28 :: #force_inline proc "contextless" (x: F8) -> F8 {
	e, f, _ := log_split(x)
	return log_special(x, e + log1p_kernel(f) * LOG2E)
}

// log(1 + u) for finite u >= 0: the log of w = 1 + u rounded, with the
// rounding, u - (w - 1), folded into w's fraction. Lanes with big take
// log(2a) instead (asinh's and acosh's value from 4096 on).
@(private = "file")
log1p_f :: #force_inline proc "contextless" (u, a: F8, big: W8) -> F8 {
	w := 1 + u
	c := u - (w - 1)
	e, f, ei := log_split(sel(big, a, w))
	f += sel(big, F8(0), c * pow2(-ei))
	e += sel(big, F8(1), F8(0))
	return e * LN2_HI + (log1p_kernel(f) + e * LN2_LO)
}

@(private)
fasinh8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	a := intrinsics.simd_abs(x)
	a2 := a * a
	r := log1p_f(a + a2 / (1 + intrinsics.sqrt(1 + a2)), a, intrinsics.simd_lanes_gt(a, F8(4096)))
	return with_sign(sel(intrinsics.simd_lanes_eq(a, F8(INF)), a, r), x)
}

@(private)
facosh8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	t := x - 1
	r := log1p_f(t + intrinsics.sqrt(t * (t + 2)), x, intrinsics.simd_lanes_gt(x, F8(4096)))
	r = sel(intrinsics.simd_lanes_eq(x, F8(INF)), x, r)
	return sel(intrinsics.simd_lanes_ge(x, F8(1)), r, F8(NAN))
}

@(private)
fatanh8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	a := intrinsics.simd_abs(x)
	r := 0.5 * log1p_f(2 * a / (1 - a), a, W8(0))
	r = sel(intrinsics.simd_lanes_eq(a, F8(1)), F8(INF), r)
	return sel(intrinsics.simd_lanes_le(a, F8(1)), with_sign(r, x), F8(NAN))
}

// C99's powf: |x|^y = 2^(y * log2|x|), and pow(+0, y) = +0, 1 or +inf by y's
// sign. Lanes with x < 0 or -0, either operand infinite or NaN (rare enough
// to test for once per call) take the rest of C99's rules: negative for x <
// 0 (or -0) at odd integer y; pow(x < 0, non-integer y) = NaN (not for
// -inf); pow(x, +-0) = pow(1, y) = pow(-1, +-inf) = 1, NaN included; zeros
// and infinities by the sign of y. Lanes with y = 2, the commonest power,
// take x * x (tested once per call).
@(private)
fpow8 :: #force_inline proc "contextless" (x, y: F8) -> F8 {
	two := intrinsics.simd_lanes_eq(y, F8(2))
	if intrinsics.simd_reduce_and(two) != 0 {
		return x * x
	}
	ax := intrinsics.simd_abs(x)
	e, f, _ := log_split(ax)
	res := fexp28(y * (e + log1p_kernel(f) * LOG2E))
	y_pos := intrinsics.simd_lanes_gt(y, F8(0))
	zero := sel(y_pos, F8(0), sel(intrinsics.simd_lanes_eq(y, F8(0)), F8(1), F8(INF)))
	res = sel(intrinsics.simd_lanes_eq(ax, F8(0)), zero, res)
	plain := intrinsics.simd_lanes_lt(transmute(W8)x, W8(0x7f80_0000)) & intrinsics.simd_lanes_lt(intrinsics.simd_abs(y), F8(INF))
	if intrinsics.simd_reduce_and(plain) == 0 {
		res = sel(intrinsics.simd_lanes_eq(ax, F8(INF)), sel(y_pos, F8(INF), F8(0)), res)
		ay := intrinsics.simd_abs(y)
		yi := cast(I8)sel(intrinsics.simd_lanes_lt(ay, F8(16777216)), y, F8(0))
		small_int := intrinsics.simd_lanes_eq(cast(F8)yi, y)
		y_int := small_int | intrinsics.simd_lanes_ge(ay, F8(16777216))
		y_odd := intrinsics.simd_lanes_ne(transmute(W8)yi & W8(1), W8(0)) & small_int
		x_neg := intrinsics.simd_lanes_ge(transmute(W8)x, W8(SIGN))
		res = sel(y_odd & x_neg, -res, res)
		neg_base := intrinsics.simd_lanes_lt(x, F8(0)) & intrinsics.simd_lanes_gt(x, F8(-INF))
		nan := (neg_base & ~y_int) | is_nan(x) | is_nan(y)
		res = sel(nan, F8(NAN), res)
		one := intrinsics.simd_lanes_eq(x, F8(1)) | intrinsics.simd_lanes_eq(y, F8(0)) | (intrinsics.simd_lanes_eq(x, F8(-1)) & intrinsics.simd_lanes_eq(ay, F8(INF)))
		res = sel(one, F8(1), res)
	}
	if intrinsics.simd_reduce_or(two) != 0 {
		res = sel(two, x * x, res)
	}
	return res
}

// ---- atan (both forms), asin, acos -------------------------------------------------------

// GLSL's atan(y, x), with C99's atan2 at zeros and infinities: a = atan(u)
// for u = min / max of |y| and |x|, in [0, 1] (0 / 0 as 0; both infinite,
// pi/4), then by the octant off - a or off + a for off 0, pi/2 or pi, off's
// rounding added to a first, and y's sign.
@(private)
fatan28 :: #force_inline proc "contextless" (y, x: F8) -> F8 {
	// (atan(u) - u) / u^3 in z = u^2 on [0, 1], degree 7, 2^-25.8
	A0 :: f32(0hBEAAAA6E)
	A1 :: f32(0h3E4CBC7A)
	A2 :: f32(0hBE11856E)
	A3 :: f32(0h3DDA7095)
	A4 :: f32(0hBD9AAB5B)
	A5 :: f32(0h3D30FEED)
	A6 :: f32(0hBC86160A)
	A7 :: f32(0h3B3F6919)
	ay, ax := intrinsics.simd_abs(y), intrinsics.simd_abs(x)
	swap := intrinsics.simd_lanes_gt(ay, ax)
	den := sel(swap, ay, ax)
	u := sel(swap, ax, ay) / sel(intrinsics.simd_lanes_eq(den, F8(0)), F8(1), den)
	z := u * u
	z2 := z * z
	p := ((A0 + z * A1) + z2 * (A2 + z * A3)) + (z2 * z2) * ((A4 + z * A5) + z2 * (A6 + z * A7))
	both_inf := intrinsics.simd_lanes_eq(ay, F8(INF)) & intrinsics.simd_lanes_eq(ax, F8(INF))
	a := sel(both_inf, F8(PIO4), u + u * z * p)
	x_neg := intrinsics.simd_lanes_ge(transmute(W8)x, W8(SIGN))
	a = sel(swap ~ x_neg, -a, a)
	hi := sel(swap, F8(PIO2_HI), sel(x_neg, F8(PI_HI), F8(0)))
	lo := sel(swap, F8(PIO2_LO), sel(x_neg, F8(PI_LO), F8(0)))
	res := with_sign(hi + (lo + a), y)
	return sel(is_nan(x) | is_nan(y), F8(NAN), res)
}

@(private)
fatan8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	return fatan28(x, F8(1))
}

// asin(x) = atan(x, sqrt(1 - x^2)) and acos(x) = atan(sqrt(1 - x^2), x),
// with 1 - x^2 as (1 - |x|)(1 + |x|) (NaN outside [-1, 1]).
@(private = "file")
cos_of_asin :: #force_inline proc "contextless" (x: F8) -> F8 {
	a := intrinsics.simd_abs(x)
	return intrinsics.sqrt((1 - a) * (1 + a))
}

@(private)
fasin8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	return fatan28(x, cos_of_asin(x))
}

@(private)
facos8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	return fatan28(cos_of_asin(x), x)
}

// ---- for callers compiled for AVX2 ----------------------------------------------------

// A built-in of the above (op F_Fn1 or F_Fn2, b its Builtin) for eight lanes
// from operands at px and py into pd, for the JIT's helper, compiled for
// AVX2: by pointer, as builds that keep each file a module of its own and
// leave calls between them uninlined can't pass vector values between
// functions of different target features.
@(private)
math_8 :: #force_inline proc "contextless" (op: VM_Op, b: Builtin, px, py, pd: ^[8]u32) {
	x := intrinsics.unaligned_load((^F8)(px))
	r := x
	if op == .F_Fn2 {
		y := intrinsics.unaligned_load((^F8)(py))
		r = fatan28(x, y) if b == .Atan2 else fpow8(x, y)
	} else {
		#partial switch b {
		case .Sin:
			r = fsin8(x)
		case .Cos:
			r = fcos8(x)
		case .Tan:
			r = ftan8(x)
		case .Asin:
			r = fasin8(x)
		case .Acos:
			r = facos8(x)
		case .Atan:
			r = fatan8(x)
		case .Sinh:
			r = fsinh8(x)
		case .Cosh:
			r = fcosh8(x)
		case .Tanh:
			r = ftanh8(x)
		case .Asinh:
			r = fasinh8(x)
		case .Acosh:
			r = facosh8(x)
		case .Atanh:
			r = fatanh8(x)
		case .Exp:
			r = fexp8(x)
		case .Log:
			r = flog8(x)
		case .Exp2:
			r = fexp28(x)
		case .Log2:
			r = flog28(x)
		}
	}
	intrinsics.unaligned_store((^F8)(pd), r)
}
