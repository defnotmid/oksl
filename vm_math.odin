package oksl

import "base:intrinsics"

// The CPU's float built-ins that libm would otherwise compute: sin, cos,
// tan, asin, acos, atan (one and two arguments), sinh, cosh, tanh, asinh,
// acosh, atanh, exp, exp2, log, log2 and pow, for the interpreter and the
// JIT's helper alike. Each works on eight f32 lanes in f64 with IEEE-754
// basic operations only (+, -, *, /, sqrt, comparisons, exact conversions
// and bit operations: no FMA, which Odin never contracts into, and no
// libm), so its result is the same bits on every IEEE platform: rounded
// once to f32 from a value within about 2^-45 of the exact one, so within
// 0.5000003 ulp of it, and the correctly rounded f32 but for at most 25
// inputs in a billion (pow: exact halfway cases, such as x^3 halfway
// between two subnormals). Lanes are whole SIMD vectors and every choice a
// select, but for cases tested once per call (sin, cos and tan of |x| >=
// 2^20; pow at y = 2, and at x < 0, -0, or infinite or NaN operands); NaN results
// are any NaN (outputs canonicalize, `canonical`).
//
// The polynomials are minimax fits of the relative error (Remez, in f64),
// each named with its interval and error, evaluated by Estrin's scheme:
// a JIT helper call is one register of lanes, so its latency, not its
// throughput, is what a shader waits for.

@(private)
F8 :: #simd[8]f32
@(private = "file")
D8 :: #simd[8]f64
@(private = "file")
N8 :: #simd[8]u64
@(private = "file")
L8 :: #simd[8]i64
@(private = "file")
W8 :: #simd[8]u32
@(private = "file")
I8 :: #simd[8]i32

@(private = "file")
SHIFT :: 0h4338000000000000 // 1.5 * 2^52: x + SHIFT - SHIFT rounds x to an integer
@(private = "file")
PIO2 :: 0h3FF921FB54442D18
@(private = "file")
PIO4 :: 0h3FE921FB54442D18
@(private = "file")
PI :: 0h400921FB54442D18
@(private = "file")
LN2 :: 0h3FE62E42FEFA39EF
@(private = "file")
LOG2E :: 0h3FF71547652B82FE
@(private = "file")
TAN_PI_8 :: 0h3FDA827999FCEF34
@(private = "file")
TAN_3PI_8 :: 0h4003504F333F9DE6
@(private = "file")
INF :: 0h7FF0000000000000
@(private = "file")
F32_INF :: f32(0h7F80_0000)
@(private = "file")
F32_NAN :: f32(0h7FC0_0000)
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

// x as f64 lanes; x != x (NaN lanes).
@(private = "file")
wide :: #force_inline proc "contextless" (x: F8) -> D8 {
	return cast(D8)x
}

@(private = "file")
is_nan :: #force_inline proc "contextless" (x: F8) -> W8 {
	return intrinsics.simd_lanes_ne(x, x)
}

// ---- sin, cos, tan -----------------------------------------------------------------------

// |x| as r + q * pi/2 with |r| <= pi/4 (1 + 2^-24): r and q's low two bits.
// Below 2^20, Cody-Waite: n the integer nearest |x| * 2/pi and |x| - n *
// pi/2 with pi/2 in three parts, the first two (33 bits each) times n and
// their differences exact, so r is within 2^-52 of itself plus n * 2^-118.
// From 2^20 up (the lanes of a call that have any finite one), Payne-Hanek:
// |x| = m * 2^s (m the 24-bit integer mantissa) and |x| * 2/pi mod 4 = m *
// TWO_OVER_PI[s + 3] mod 4, whose three parts make m * H exact (a multiple
// of 2^-27 below 2^26), m * M exact (of 2^-56 below 2^-3) and m * L within
// 2^-85, so the fraction left after the nearest integer is within about
// 2^-84 of the exact one. Either way r's relative error is near 2^-52,
// since no f32 lies within 2^-30 of a multiple of pi/2.
@(private = "file")
reduce_pio2 :: #force_inline proc "contextless" (ax: F8) -> (r: D8, q: N8) {
	P1 :: 0h3FF921FB54400000
	P2 :: 0h3DD0B4611A600000
	P3 :: 0h3BA3198A2E037073
	INV_PIO2 :: 0h3FE45F306DC9C883
	a := wide(ax)
	w := a * INV_PIO2 + SHIFT
	n := w - SHIFT
	r = ((a - n * P1) - n * P2) - n * P3
	q = transmute(N8)w
	far := intrinsics.simd_lanes_ge(ax, F8(1048576)) & intrinsics.simd_lanes_lt(ax, F8(F32_INF))
	if intrinsics.simd_reduce_or(far) == 0 {
		return
	}
	b := transmute(W8)ax
	e := transmute(I8)intrinsics.simd_shr(b, W8(23)) - 147
	e = intrinsics.simd_clamp(e, I8(0), I8(len(TWO_OVER_PI) - 1))
	row := cast(#simd[8]uintptr)(cast(N8)e * size_of([3]f64)) + uintptr(&TWO_OVER_PI[0][0])
	ALL :: N8(max(u64))
	t0 := intrinsics.simd_gather(transmute(#simd[8]rawptr)row, D8(0), ALL)
	t1 := intrinsics.simd_gather(transmute(#simd[8]rawptr)(row + 8), D8(0), ALL)
	t2 := intrinsics.simd_gather(transmute(#simd[8]rawptr)(row + 16), D8(0), ALL)
	m := cast(D8)transmute(I8)((b & W8(0x7f_ffff)) | W8(0x80_0000))
	p := m * t0
	h := m * t1
	wt := (p + h) + SHIFT
	nt := wt - SHIFT
	f := ((p - nt) + h) + m * t2
	r = sel(far, f * PIO2, r)
	q = sel(far, transmute(N8)wt, q)
	return
}

// sin(r) and cos(r) for |r| <= pi/4 (1 + 2^-24).
@(private = "file")
sincos_kernel :: #force_inline proc "contextless" (r: D8) -> (s, c: D8) {
	// (sin(r) - r) / r^3 in z = r^2 on [0, (pi/4)^2 * 1.0005], degree 4, 2^-42.4
	S0 :: 0hBFC555555555515B
	S1 :: 0h3F81111110FD0566
	S2 :: 0hBF2A019FD93EA600
	S3 :: 0h3EC71D9A511D3B9B
	S4 :: 0hBE5AA264C6AC12DA
	// (cos(r) - 1 + r^2/2) / r^4 on the same interval, degree 4, 2^-44.2
	C0 :: 0h3FA5555555555433
	C1 :: 0hBF56C16C16B5FF1E
	C2 :: 0h3EFA019FF50CB2EB
	C3 :: 0hBE927E25D5E6CF33
	C4 :: 0h3E21C80F2DF6F37C
	z := r * r
	z2 := z * z
	s = r + r * z * ((S0 + z * S1) + z2 * ((S2 + z * S3) + z2 * S4))
	c = (1 - 0.5 * z) + z2 * ((C0 + z * C1) + z2 * ((C2 + z * C3) + z2 * C4))
	return
}

// sin(q * pi/2 + r) from the kernel's values: q & 2 flips the sign bit.
@(private = "file")
quadrant :: #force_inline proc "contextless" (s, c: D8, q: N8) -> D8 {
	v := sel(intrinsics.simd_lanes_ne(q & 1, N8(0)), c, s)
	return transmute(D8)(transmute(N8)v ~ intrinsics.simd_shl(q & 2, N8(62)))
}

// At +-inf and NaN the reduction gives NaN (inf - inf), and so do these.
@(private)
fsin8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	r, q := reduce_pio2(intrinsics.simd_abs(x))
	s, c := sincos_kernel(r)
	return with_sign(cast(F8)quadrant(s, c, q), x)
}

@(private)
fcos8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	r, q := reduce_pio2(intrinsics.simd_abs(x))
	s, c := sincos_kernel(r)
	return cast(F8)quadrant(s, c, q + 1)
}

@(private)
ftan8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	r, q := reduce_pio2(intrinsics.simd_abs(x))
	s, c := sincos_kernel(r)
	odd := intrinsics.simd_lanes_ne(q & 1, N8(0))
	return with_sign(cast(F8)(sel(odd, -c, s) / sel(odd, s, c)), x)
}

// ---- exp, exp2 and the hyperbolic functions ----------------------------------------------

// 2^y = scale * (1 + fq), scale = 2^k for the integer k nearest y, y clamped
// to [-160, 160] (beyond f32's range either way; NaN stays NaN).
@(private = "file")
exp2_kernel :: #force_inline proc "contextless" (y: D8) -> (scale, fq: D8, k_zero: N8) {
	// (2^f - 1) / f on [-0.5005, 0.5005], degree 9, 2^-49.1
	Q0 :: 0h3FE62E42FEFA39F8
	Q1 :: 0h3FCEBFBDFF82C537
	Q2 :: 0h3FAC6B08D70441ED
	Q3 :: 0h3F83B2AB6FBB5D43
	Q4 :: 0h3F55D87FE90686B0
	Q5 :: 0h3F2430912C287BEE
	Q6 :: 0h3EEFFCB76F249CF1
	Q7 :: 0h3EB62C0902085C87
	Q8 :: 0h3E7B656AF5A327A2
	Q9 :: 0h3E3E4BA47D196460
	yc := sel(intrinsics.simd_lanes_gt(y, D8(160)), D8(160), y)
	yc = sel(intrinsics.simd_lanes_lt(yc, D8(-160)), D8(-160), yc)
	w := yc + SHIFT
	kd := w - SHIFT
	f := yc - kd
	f2 := f * f
	f4 := f2 * f2
	q03 := (Q0 + f * Q1) + f2 * (Q2 + f * Q3)
	q47 := (Q4 + f * Q5) + f2 * (Q6 + f * Q7)
	fq = f * (q03 + f4 * (q47 + f4 * (Q8 + f * Q9)))
	k := transmute(N8)w - N8(transmute(u64)f64(SHIFT))
	scale = transmute(D8)intrinsics.simd_shl(k + N8(1023), N8(52))
	k_zero = intrinsics.simd_lanes_eq(kd, D8(0))
	return
}

@(private = "file")
exp2_d :: #force_inline proc "contextless" (y: D8) -> D8 {
	scale, fq, _ := exp2_kernel(y)
	return scale * (1 + fq)
}

// e^y - 1 without cancellation near 0.
@(private = "file")
expm1_d :: #force_inline proc "contextless" (y: D8) -> D8 {
	scale, fq, k_zero := exp2_kernel(y * LOG2E)
	return sel(k_zero, fq, (scale - 1) + scale * fq)
}

@(private)
fexp28 :: #force_inline proc "contextless" (x: F8) -> F8 {
	return cast(F8)exp2_d(wide(x))
}

@(private)
fexp8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	return cast(F8)exp2_d(wide(x) * LOG2E)
}

// sinh(|x|) = (E - 1/E) / 2 = (m + m / (m + 1)) / 2 with m = e^|x| - 1.
@(private)
fsinh8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	m := expm1_d(wide(intrinsics.simd_abs(x)))
	res := with_sign(cast(F8)(0.5 * (m + m / (m + 1))), x)
	return sel(is_nan(x), F8(F32_NAN), res)
}

@(private)
fcosh8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	e := exp2_d(wide(intrinsics.simd_abs(x)) * LOG2E)
	res := cast(F8)(0.5 * (e + 1 / e))
	return sel(is_nan(x), F8(F32_NAN), res)
}

// tanh(|x|) = m / (m + 2) with m = e^(2|x|) - 1.
@(private)
ftanh8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	m := expm1_d(2 * wide(intrinsics.simd_abs(x)))
	res := with_sign(cast(F8)(m / (m + 2)), x)
	return sel(is_nan(x), F8(F32_NAN), res)
}

// ---- log, log2, pow and the inverse hyperbolic functions ---------------------------------

// For finite x > 0: x = 2^e * m with m in [sqrt(2)/2, sqrt(2)), and log(m).
@(private = "file")
log_kernel :: #force_inline proc "contextless" (x: D8) -> (e, lm: D8) {
	// (log((1 + s) / (1 - s)) - 2s) / s^3 in z = s^2 on [0, 0.17157^2 * 1.0005], degree 5, 2^-43.7
	R0 :: 0h3FE55555555553B5
	R1 :: 0h3FD9999999B895FD
	R2 :: 0h3FD24924627B342D
	R3 :: 0h3FCC71FCF6C7FD36
	R4 :: 0h3FC73827220574C2
	R5 :: 0h3FC547021AA96CFC
	ix := transmute(N8)x
	ei := intrinsics.simd_shr(transmute(L8)(ix - N8(0x3FE6A09E667F3BCD)), N8(52))
	m := transmute(D8)(ix - intrinsics.simd_shl(transmute(N8)ei, N8(52)))
	f := m - 1
	s := f / (2 + f)
	z := s * s
	z2 := z * z
	lm = 2 * s + s * z * ((R0 + z * R1) + z2 * ((R2 + z * R3) + z2 * (R4 + z * R5)))
	// The integer as an f64, exactly (|ei| < 2^51).
	e = transmute(D8)(transmute(N8)ei + N8(transmute(u64)f64(SHIFT))) - SHIFT
	return
}

// log(x) or log2(x) for x's lanes, with log(+-0) = -inf, log(x < 0) = NaN
// and log(+inf) = +inf.
@(private = "file")
log_special :: #force_inline proc "contextless" (x, r: F8) -> F8 {
	res := sel(intrinsics.simd_lanes_eq(x, F8(0)), F8(-F32_INF), r)
	res = sel(intrinsics.simd_lanes_eq(x, F8(F32_INF)), x, res)
	return sel(intrinsics.simd_lanes_ge(x, F8(0)), res, F8(F32_NAN))
}

@(private)
flog8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	e, lm := log_kernel(wide(x))
	return log_special(x, cast(F8)(e * LN2 + lm))
}

@(private)
flog28 :: #force_inline proc "contextless" (x: F8) -> F8 {
	e, lm := log_kernel(wide(x))
	return log_special(x, cast(F8)(e + lm * LOG2E))
}

// log(1 + u) for u >= 0 (+inf included): the log of w = 1 + u rounded, plus
// the rounding's share, (u - (w - 1)) / w.
@(private = "file")
log1p_d :: #force_inline proc "contextless" (u: D8) -> D8 {
	w := 1 + u
	c := (u - (w - 1)) / w
	e, lm := log_kernel(w)
	return sel(intrinsics.simd_lanes_lt(u, D8(INF)), e * LN2 + (lm + c), u)
}

@(private)
fasinh8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	a := wide(intrinsics.simd_abs(x))
	a2 := a * a
	u := a + a2 / (1 + intrinsics.sqrt(1 + a2))
	u = sel(intrinsics.simd_lanes_eq(a, D8(INF)), a, u)
	res := with_sign(cast(F8)log1p_d(u), x)
	return sel(is_nan(x), F8(F32_NAN), res)
}

@(private)
facosh8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	t := wide(x) - 1
	res := cast(F8)log1p_d(t + intrinsics.sqrt(t * (t + 2)))
	return sel(intrinsics.simd_lanes_ge(x, F8(1)), res, F8(F32_NAN))
}

@(private)
fatanh8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	ax := intrinsics.simd_abs(x)
	a := wide(ax)
	res := with_sign(cast(F8)(0.5 * log1p_d(2 * a / (1 - a))), x)
	return sel(intrinsics.simd_lanes_le(ax, F8(1)), res, F8(F32_NAN))
}

// C99's powf: |x|^y = 2^(y * log2|x|) in f64 (the product within about
// 2^-45 of the exact one wherever the result is a finite f32), and pow(+0,
// y) = +0, 1 or +inf by y's sign. Lanes with x < 0 or -0, either operand
// infinite or NaN (rare enough to test for once per call) take the rest of
// C99's rules: negative for x < 0 (or -0) at odd integer y; pow(x < 0,
// non-integer y) = NaN (not for -inf); pow(x, +-0) = pow(1, y) = pow(-1,
// +-inf) = 1, NaN included; zeros and infinities by the sign of y. Lanes
// with y = 2, the commonest power, take pow_square (tested once per call).
@(private)
fpow8 :: #force_inline proc "contextless" (x, y: F8) -> F8 {
	two := intrinsics.simd_lanes_eq(y, F8(2))
	if intrinsics.simd_reduce_and(two) != 0 {
		return pow_square(x)
	}
	ax := intrinsics.simd_abs(x)
	e, lm := log_kernel(wide(ax))
	res := cast(F8)exp2_d(wide(y) * (e + lm * LOG2E))
	y_pos := intrinsics.simd_lanes_gt(y, F8(0))
	zero := sel(y_pos, F8(0), sel(intrinsics.simd_lanes_eq(y, F8(0)), F8(1), F8(F32_INF)))
	res = sel(intrinsics.simd_lanes_eq(ax, F8(0)), zero, res)
	plain := intrinsics.simd_lanes_lt(transmute(W8)x, W8(0x7f80_0000)) & intrinsics.simd_lanes_lt(intrinsics.simd_abs(y), F8(F32_INF))
	if intrinsics.simd_reduce_and(plain) == 0 {
		res = sel(intrinsics.simd_lanes_eq(ax, F8(F32_INF)), sel(y_pos, F8(F32_INF), F8(0)), res)
		ay := intrinsics.simd_abs(y)
		yi := cast(I8)sel(intrinsics.simd_lanes_lt(ay, F8(16777216)), y, F8(0))
		small_int := intrinsics.simd_lanes_eq(cast(F8)yi, y)
		y_int := small_int | intrinsics.simd_lanes_ge(ay, F8(16777216))
		y_odd := intrinsics.simd_lanes_ne(transmute(W8)yi & W8(1), W8(0)) & small_int
		x_neg := intrinsics.simd_lanes_ge(transmute(W8)x, W8(SIGN))
		res = sel(y_odd & x_neg, -res, res)
		neg_base := intrinsics.simd_lanes_lt(x, F8(0)) & intrinsics.simd_lanes_gt(x, F8(-F32_INF))
		nan := (neg_base & ~y_int) | is_nan(x) | is_nan(y)
		res = sel(nan, F8(F32_NAN), res)
		one := intrinsics.simd_lanes_eq(x, F8(1)) | intrinsics.simd_lanes_eq(y, F8(0)) | (intrinsics.simd_lanes_eq(x, F8(-1)) & intrinsics.simd_lanes_eq(ay, F8(F32_INF)))
		res = sel(one, F8(1), res)
	}
	if intrinsics.simd_reduce_or(two) != 0 {
		res = sel(two, pow_square(x), res)
	}
	return res
}

// pow(x, 2.0): x * x in f64 is exact, so this is x^2 correctly rounded
// (and NaN, inf, +0 at the special cases, as C99's pow).
@(private = "file")
pow_square :: #force_inline proc "contextless" (x: F8) -> F8 {
	xd := wide(x)
	return cast(F8)(xd * xd)
}

// ---- asin, acos, atan --------------------------------------------------------------------

// asin(v) for |v| <= 0.5.
@(private = "file")
asin_kernel :: #force_inline proc "contextless" (v: D8) -> D8 {
	// (asin(v) - v) / v^3 in z = v^2 on [0, 0.25 * 1.0005], degree 11, 2^-49.4
	P0 :: 0h3FC555555555554E
	P1 :: 0h3FB33333333372C2
	P2 :: 0h3FA6DB6DB67E9D3B
	P3 :: 0h3F9F1C71FC293C58
	P4 :: 0h3F96E8B25C0E5322
	P5 :: 0h3F91C59A01C51A6C
	P6 :: 0h3F8C8697EDA71755
	P7 :: 0h3F88564B492EA7F1
	P8 :: 0h3F7FCEAD56AE1892
	P9 :: 0h3F9088E106DE9B8A
	P10 :: 0hBF86666729089375
	P11 :: 0h3F9CFC7C8D3AADBB
	z := v * v
	z2 := z * z
	z4 := z2 * z2
	p03 := (P0 + z * P1) + z2 * (P2 + z * P3)
	p47 := (P4 + z * P5) + z2 * (P6 + z * P7)
	p811 := (P8 + z * P9) + z2 * (P10 + z * P11)
	return v + v * z * (p03 + z4 * (p47 + z4 * p811))
}

// asin(|x|) for |x| > 0.5 (big) is pi/2 - 2 asin(w), w = sqrt((1 - |x|) /
// 2): the kernel's value at |x| or at w.
@(private = "file")
asin_parts :: #force_inline proc "contextless" (x: F8) -> (k: D8, big: N8) {
	a := wide(intrinsics.simd_abs(x))
	big = intrinsics.simd_lanes_gt(a, D8(0.5))
	k = asin_kernel(sel(big, intrinsics.sqrt((1 - a) * 0.5), a))
	return
}

@(private)
fasin8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	k, big := asin_parts(x)
	return with_sign(cast(F8)sel(big, PIO2 - 2 * k, k), x)
}

@(private)
facos8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	k, big := asin_parts(x)
	neg := intrinsics.simd_lanes_lt(wide(x), D8(0))
	small := sel(neg, PIO2 + k, PIO2 - k)
	large := sel(neg, PI - 2 * k, 2 * k)
	return cast(F8)sel(big, large, small)
}

// atan(num / den) + off, num / den within [-tan(pi/8), tan(pi/8)] (2^-46 over).
@(private = "file")
atan_kernel :: #force_inline proc "contextless" (num, den, off: D8) -> D8 {
	// (atan(u) - u) / u^3 in z = u^2 on [0, tan(pi/8)^2 * 1.0005], degree 8, 2^-43.7
	A0 :: 0hBFD55555555553AC
	A1 :: 0h3FC99999998D460E
	A2 :: 0hBFC249248ABCE938
	A3 :: 0h3FBC71C38A1CE567
	A4 :: 0hBFB7456484AC88FE
	A5 :: 0h3FB3A9E13D5A4622
	A6 :: 0hBFB0C566F090EB0C
	A7 :: 0h3FAA784604A62524
	A8 :: 0hBF9BE6CD666B63EB
	u := num / den
	z := u * u
	z2 := z * z
	z4 := z2 * z2
	a03 := (A0 + z * A1) + z2 * (A2 + z * A3)
	a47 := (A4 + z * A5) + z2 * (A6 + z * A7)
	return off + (u + u * z * (a03 + z4 * (a47 + z4 * A8)))
}

@(private)
fatan8 :: #force_inline proc "contextless" (x: F8) -> F8 {
	t := wide(intrinsics.simd_abs(x))
	mid := intrinsics.simd_lanes_gt(t, D8(TAN_PI_8))
	big := intrinsics.simd_lanes_gt(t, D8(TAN_3PI_8))
	num := sel(big, D8(-1), sel(mid, t - 1, t))
	den := sel(big, t, sel(mid, t + 1, D8(1)))
	off := sel(big, D8(PIO2), sel(mid, D8(PIO4), D8(0)))
	return with_sign(cast(F8)atan_kernel(num, den, off), x)
}

// GLSL's atan(y, x), with C99's atan2 at zeros and infinities.
@(private)
fatan28 :: #force_inline proc "contextless" (y, x: F8) -> F8 {
	ay, ax := wide(intrinsics.simd_abs(y)), wide(intrinsics.simd_abs(x))
	mid := intrinsics.simd_lanes_gt(ay, ax * TAN_PI_8)
	big := intrinsics.simd_lanes_gt(ay, ax * TAN_3PI_8)
	num := sel(big, -ax, sel(mid, ay - ax, ay))
	den := sel(big, ay, sel(mid, ay + ax, ax))
	off := sel(big, D8(PIO2), sel(mid, D8(PIO4), D8(0)))
	// atan2(0, x) is 0 before the quadrant (0 / 0 at x = 0); with both
	// infinite, pi/4.
	both_inf := intrinsics.simd_lanes_eq(ay, D8(INF)) & intrinsics.simd_lanes_eq(ax, D8(INF))
	fix := intrinsics.simd_lanes_eq(ay, D8(0)) | both_inf
	num = sel(fix, D8(0), num)
	den = sel(fix, D8(1), den)
	off = sel(both_inf, D8(PIO4), off)
	a := atan_kernel(num, den, off)
	a = sel(intrinsics.simd_lanes_ge(transmute(W8)x, W8(SIGN)), PI - a, a)
	res := with_sign(cast(F8)a, y)
	return sel(is_nan(x) | is_nan(y), F8(F32_NAN), res)
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

// (2^s * 2/pi) mod 4 for s = -3 .. 104 (|x| from 2^20 to f32's largest): H
// holds its bits from 2^1 to 2^-27, M those from 2^-28 to 2^-56 and L the
// rest rounded to f64, from 2/pi to 600 bits.
@(private = "file", rodata)
TWO_OVER_PI := [108][3]f64 {
	{0h3FB45F3060000000, 0h3E2B939104000000, 0h3C64A7F09D5F47D5}, // -3
	{0h3FC45F3060000000, 0h3E3B939105000000, 0h3C529FC2757D1F53}, // -2
	{0h3FD45F3068000000, 0h3E3727220A000000, 0h3C629FC2757D1F53}, // -1
	{0h3FE45F306C000000, 0h3E2C9C882A000000, 0h3C44FE13ABE8FA9A}, // 0
	{0h3FF45F306C000000, 0h3E3C9C882A000000, 0h3C54FE13ABE8FA9A}, // 1
	{0h40045F306D000000, 0h3E39391054000000, 0h3C64FE13ABE8FA9A}, // 2
	{0h3FF17CC1B6000000, 0h3E327220A9000000, 0h3C53F84EAFA3EA6A}, // 3
	{0h40017CC1B7000000, 0h3E13910548000000, 0h3C63F84EAFA3EA6A}, // 4
	{0h3FD7CC1B70000000, 0h3E2391054A000000, 0h3C4FC2757D1F534E}, // 5
	{0h3FE7CC1B70000000, 0h3E3391054A000000, 0h3C5FC2757D1F534E}, // 6
	{0h3FF7CC1B72000000, 0h3E1C882A50000000, 0h3C6FC2757D1F534E}, // 7
	{0h4007CC1B72000000, 0h3E2C882A52000000, 0h3C6F84EAFA3EA69C}, // 8
	{0h3FFF306DC8000000, 0h3E3C882A53000000, 0h3C6F09D5F47D4D37}, // 9
	{0h400F306DC9000000, 0h3E391054A7000000, 0h3C6E13ABE8FA9A6F}, // 10
	{0h400E60DB93000000, 0h3E3220A94F000000, 0h3C6C2757D1F534DE}, // 11
	{0h400CC1B727000000, 0h3E11054A7C000000, 0h3C684EAFA3EA69BC}, // 12
	{0h4009836E4E000000, 0h3E21054A7E000000, 0h3C609D5F47D4D377}, // 13
	{0h400306DC9C000000, 0h3E31054A7F000000, 0h3C23ABE8FA9A6EE0}, // 14
	{0h3FE836E4E4000000, 0h3E0054A7F0000000, 0h3C33ABE8FA9A6EE0}, // 15
	{0h3FF836E4E4000000, 0h3E1054A7F0000000, 0h3C43ABE8FA9A6EE0}, // 16
	{0h400836E4E4000000, 0h3E2054A7F0000000, 0h3C53ABE8FA9A6EE0}, // 17
	{0h40006DC9C8000000, 0h3E3054A7F0000000, 0h3C63ABE8FA9A6EE0}, // 18
	{0h3FBB727220000000, 0h3DE529FC20000000, 0h3C4D5F47D4D37703}, // 19
	{0h3FCB727220000000, 0h3DF529FC20000000, 0h3C5D5F47D4D37703}, // 20
	{0h3FDB727220000000, 0h3E0529FC20000000, 0h3C6D5F47D4D37703}, // 21
	{0h3FEB727220000000, 0h3E1529FC24000000, 0h3C6ABE8FA9A6EE07}, // 22
	{0h3FFB727220000000, 0h3E2529FC26000000, 0h3C657D1F534DDC0E}, // 23
	{0h400B727220000000, 0h3E3529FC27000000, 0h3C55F47D4D377037}, // 24
	{0h4006E4E441000000, 0h3E24A7F09C000000, 0h3C65F47D4D377037}, // 25
	{0h3FFB939104000000, 0h3E34A7F09D000000, 0h3C57D1F534DDC0DB}, // 26
	{0h400B939105000000, 0h3E229FC274000000, 0h3C67D1F534DDC0DB}, // 27
	{0h400727220A000000, 0h3E329FC275000000, 0h3C5F47D4D377036E}, // 28
	{0h3FFC9C882A000000, 0h3E14FE13A8000000, 0h3C6F47D4D377036E}, // 29
	{0h400C9C882A000000, 0h3E24FE13AA000000, 0h3C6E8FA9A6EE06DB}, // 30
	{0h4009391054000000, 0h3E34FE13AB000000, 0h3C6D1F534DDC0DB6}, // 31
	{0h40027220A9000000, 0h3E23F84EAE000000, 0h3C6A3EA69BB81B6C}, // 32
	{0h3FE3910548000000, 0h3E33F84EAF000000, 0h3C647D4D377036D9}, // 33
	{0h3FF391054A000000, 0h3E1FC2757C000000, 0h3C51F534DDC0DB63}, // 34
	{0h400391054A000000, 0h3E2FC2757C000000, 0h3C61F534DDC0DB63}, // 35
	{0h3FEC882A50000000, 0h3E3FC2757D000000, 0h3C3F534DDC0DB629}, // 36
	{0h3FFC882A52000000, 0h3E3F84EAFA000000, 0h3C4F534DDC0DB629}, // 37
	{0h400C882A53000000, 0h3E3F09D5F4000000, 0h3C5F534DDC0DB629}, // 38
	{0h40091054A7000000, 0h3E3E13ABE8000000, 0h3C6F534DDC0DB629}, // 39
	{0h400220A94F000000, 0h3E3C2757D1000000, 0h3C6EA69BB81B6C53}, // 40
	{0h3FE1054A7C000000, 0h3E384EAFA3000000, 0h3C6D4D377036D8A5}, // 41
	{0h3FF1054A7E000000, 0h3E309D5F47000000, 0h3C6A9A6EE06DB14B}, // 42
	{0h4001054A7F000000, 0h3DF3ABE8F0000000, 0h3C6534DDC0DB6296}, // 43
	{0h3FD054A7F0000000, 0h3E03ABE8F8000000, 0h3C54D377036D8A56}, // 44
	{0h3FE054A7F0000000, 0h3E13ABE8F8000000, 0h3C64D377036D8A56}, // 45
	{0h3FF054A7F0000000, 0h3E23ABE8FA000000, 0h3C534DDC0DB6295A}, // 46
	{0h400054A7F0000000, 0h3E33ABE8FA000000, 0h3C634DDC0DB6295A}, // 47
	{0h3FB529FC20000000, 0h3E1D5F47D4000000, 0h3C4A6EE06DB14ACD}, // 48
	{0h3FC529FC20000000, 0h3E2D5F47D4000000, 0h3C5A6EE06DB14ACD}, // 49
	{0h3FD529FC20000000, 0h3E3D5F47D4000000, 0h3C6A6EE06DB14ACD}, // 50
	{0h3FE529FC24000000, 0h3E3ABE8FA9000000, 0h3C64DDC0DB629599}, // 51
	{0h3FF529FC26000000, 0h3E357D1F53000000, 0h3C5377036D8A5665}, // 52
	{0h400529FC27000000, 0h3E25F47D4C000000, 0h3C6377036D8A5665}, // 53
	{0h3FF4A7F09C000000, 0h3E35F47D4D000000, 0h3C4BB81B6C52B328}, // 54
	{0h4004A7F09D000000, 0h3E27D1F534000000, 0h3C5BB81B6C52B328}, // 55
	{0h3FF29FC274000000, 0h3E37D1F534000000, 0h3C6BB81B6C52B328}, // 56
	{0h40029FC275000000, 0h3E2F47D4D2000000, 0h3C677036D8A5664F}, // 57
	{0h3FE4FE13A8000000, 0h3E3F47D4D3000000, 0h3C5DC0DB6295993C}, // 58
	{0h3FF4FE13AA000000, 0h3E3E8FA9A6000000, 0h3C6DC0DB6295993C}, // 59
	{0h4004FE13AB000000, 0h3E3D1F534D000000, 0h3C6B81B6C52B3279}, // 60
	{0h3FF3F84EAE000000, 0h3E3A3EA69B000000, 0h3C67036D8A5664F1}, // 61
	{0h4003F84EAF000000, 0h3E347D4D37000000, 0h3C5C0DB6295993C4}, // 62
	{0h3FEFC2757C000000, 0h3E21F534DC000000, 0h3C6C0DB6295993C4}, // 63
	{0h3FFFC2757C000000, 0h3E31F534DD000000, 0h3C681B6C52B32788}, // 64
	{0h400FC2757D000000, 0h3E0F534DD8000000, 0h3C6036D8A5664F11}, // 65
	{0h400F84EAFA000000, 0h3E1F534DDC000000, 0h3C0B6C52B3278872}, // 66
	{0h400F09D5F4000000, 0h3E2F534DDC000000, 0h3C1B6C52B3278872}, // 67
	{0h400E13ABE8000000, 0h3E3F534DDC000000, 0h3C2B6C52B3278872}, // 68
	{0h400C2757D1000000, 0h3E3EA69BB8000000, 0h3C3B6C52B3278872}, // 69
	{0h40084EAFA3000000, 0h3E3D4D3770000000, 0h3C4B6C52B3278872}, // 70
	{0h40009D5F47000000, 0h3E3A9A6EE0000000, 0h3C5B6C52B3278872}, // 71
	{0h3FC3ABE8F0000000, 0h3E3534DDC0000000, 0h3C6B6C52B3278872}, // 72
	{0h3FD3ABE8F8000000, 0h3E24D37702000000, 0h3C66D8A5664F10E4}, // 73
	{0h3FE3ABE8F8000000, 0h3E34D37703000000, 0h3C5B6295993C4390}, // 74
	{0h3FF3ABE8FA000000, 0h3E234DDC0C000000, 0h3C6B6295993C4390}, // 75
	{0h4003ABE8FA000000, 0h3E334DDC0D000000, 0h3C66C52B32788721}, // 76
	{0h3FED5F47D4000000, 0h3E1A6EE06C000000, 0h3C5B14ACC9E21C82}, // 77
	{0h3FFD5F47D4000000, 0h3E2A6EE06C000000, 0h3C6B14ACC9E21C82}, // 78
	{0h400D5F47D4000000, 0h3E3A6EE06D000000, 0h3C66295993C43904}, // 79
	{0h400ABE8FA9000000, 0h3E34DDC0DB000000, 0h3C58A5664F10E410}, // 80
	{0h40057D1F53000000, 0h3E2377036C000000, 0h3C68A5664F10E410}, // 81
	{0h3FF5F47D4C000000, 0h3E3377036D000000, 0h3C614ACC9E21C821}, // 82
	{0h4005F47D4D000000, 0h3E1BB81B6C000000, 0h3C34ACC9E21C8210}, // 83
	{0h3FF7D1F534000000, 0h3E2BB81B6C000000, 0h3C44ACC9E21C8210}, // 84
	{0h4007D1F534000000, 0h3E3BB81B6C000000, 0h3C54ACC9E21C8210}, // 85
	{0h3FFF47D4D2000000, 0h3E377036D8000000, 0h3C64ACC9E21C8210}, // 86
	{0h400F47D4D3000000, 0h3E2DC0DB62000000, 0h3C52B32788720840}, // 87
	{0h400E8FA9A6000000, 0h3E3DC0DB62000000, 0h3C62B32788720840}, // 88
	{0h400D1F534D000000, 0h3E3B81B6C5000000, 0h3C45993C439041FE}, // 89
	{0h400A3EA69B000000, 0h3E37036D8A000000, 0h3C55993C439041FE}, // 90
	{0h40047D4D37000000, 0h3E2C0DB628000000, 0h3C65993C439041FE}, // 91
	{0h3FF1F534DC000000, 0h3E3C0DB629000000, 0h3C5664F10E4107F9}, // 92
	{0h4001F534DD000000, 0h3E381B6C52000000, 0h3C6664F10E4107F9}, // 93
	{0h3FDF534DD8000000, 0h3E3036D8A5000000, 0h3C5993C439041FE5}, // 94
	{0h3FEF534DDC000000, 0h3DDB6C5280000000, 0h3C6993C439041FE5}, // 95
	{0h3FFF534DDC000000, 0h3DEB6C52A0000000, 0h3C63278872083FCA}, // 96
	{0h400F534DDC000000, 0h3DFB6C52B0000000, 0h3C493C439041FE51}, // 97
	{0h400EA69BB8000000, 0h3E0B6C52B0000000, 0h3C593C439041FE51}, // 98
	{0h400D4D3770000000, 0h3E1B6C52B0000000, 0h3C693C439041FE51}, // 99
	{0h400A9A6EE0000000, 0h3E2B6C52B2000000, 0h3C6278872083FCA3}, // 100
	{0h400534DDC0000000, 0h3E3B6C52B3000000, 0h3C43C439041FE516}, // 101
	{0h3FF4D37702000000, 0h3E36D8A566000000, 0h3C53C439041FE516}, // 102
	{0h4004D37703000000, 0h3E2B629598000000, 0h3C63C439041FE516}, // 103
	{0h3FF34DDC0C000000, 0h3E3B629599000000, 0h3C4E21C820FF28B2}, // 104
}
