// Swizzle reads and writes, constructors and length().
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    vec4 v = vec4(uv, uv.yx * 0.5);
    v.wz = v.xy;
    v.x += v.y;
    v.yzw *= vec3(0.5, 0.75, 1.0);
    vec3 c = v.zyx;
    c.rg = c.gr + vec2(v.length()) * 0.05;
    vec2 s = vec2(1.0).yx;
    float f = c.b;
    vec4 w = vec4(c.xy, s);
    w.stpq = w.qpts;
    fragColor = vec4(clamp(vec3(w.x, c.y, f) * vec3(1.0, 1.0, 1.0), 0.0, 1.0), 1.0);
}
