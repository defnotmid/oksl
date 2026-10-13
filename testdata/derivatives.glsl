// Anti-aliased grid lines from screen-space derivatives.
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.y * 8.0;
    vec2 g = abs(fract(uv - 0.5) - 0.5) / fwidth(uv);
    float line = min(g.x, g.y);
    float dx = dFdx(uv.x);
    float dy = dFdy(uv.y);
    vec3 col = vec3(1.0 - min(line, 1.0));
    col.r += dx * 4.0;
    col.b += abs(dy) * 4.0;
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
