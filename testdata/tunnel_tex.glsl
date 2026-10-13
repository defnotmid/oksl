// Classic tunnel: polar coordinates looked up in a texture (iChannel0, a
// 64x64 checker in the tests), with Shadertoy's normalized uv.
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 p = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    float a = atan(p.y, p.x);
    float r = length(p);
    vec2 uv = vec2(0.3 / r + 0.2 * iTime, a / 3.1415927);
    vec3 col = texture(iChannel0, fract(uv)).rgb;
    col *= r;
    fragColor = vec4(col, 1.0);
}
