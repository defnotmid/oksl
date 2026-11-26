// pow with a literal 2.0 exponent over negative bases: x * x on every
// backend (GLSL leaves pow of a negative base undefined, and exp2(2 log2 x)
// is NaN on WARP), as a scalar, a vector and the Gaussian falloff
// exp(-pow(d / w, 2.0)) that onimite's motion_poster draws with.
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    float r = pow(p.x, 2.0);
    vec2 q = pow(p - 0.25, vec2(2));
    float g = exp(-pow((length(p) - 0.6) / -0.2, 2.0));
    fragColor = vec4(r, q.y, g, 1.0);
}
