// Shadertoy code with object-like #defines (expanded by the rewrite,
// including defines that use earlier ones) and a precision statement.
precision highp float;
#define PI 3.14159265
#define TAU (2.0 * PI) // a full turn
#define RINGS 6
#define DETAIL 1.0

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 p = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    float r = length(p);
    float a = atan(p.y, p.x) / TAU + 0.5;
    float rings = smoothstep(0.45, 0.5, fract(r * float(RINGS)));
    float sectors = step(0.5, fract(a * 8.0));
    vec3 col = vec3(rings, sectors * DETAIL, 1.0 - r);
    fragColor = vec4(col, 1.0);
}
