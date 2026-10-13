// Ternaries, short-circuit evaluation with side effects, bvec mix and steps.
int counter = 0;

bool bumpTrue()
{
    counter += 1;
    return true;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    bool left = uv.x < 0.5;
    bool top = uv.y > 0.5;
    if (left && bumpTrue()) {}
    if (top || bumpTrue()) {}
    vec3 a = vec3(1.0, 0.2, 0.2);
    vec3 b = vec3(0.2, 0.2, 1.0);
    vec3 col = left ? a : b;
    col = mix(col, vec3(0.2, 1.0, 0.2), lessThan(vec3(uv, 0.5), vec3(0.25, 0.25, 0.0)));
    col *= 0.5 + 0.5 * smoothstep(0.2, 0.8, uv.y) * step(0.3, uv.x);
    col.g += float(counter) * 0.1;
    fragColor = vec4(left ^^ top ? col : col.bgr, 1.0);
}
