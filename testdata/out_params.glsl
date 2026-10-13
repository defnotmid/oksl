// Functions with out and inout parameters, and overloads.
void polar(in vec2 p, out float r, out float a)
{
    r = length(p);
    a = atan(p.y, p.x);
}

void bump(inout vec3 c, float amount)
{
    c += amount;
    c = min(c, vec3(1.0));
}

float shade(float x) { return x * x; }
vec3 shade(vec3 x) { return x * x; }

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 p = fragCoord / iResolution.xy - 0.5;
    float r, a;
    polar(p, r, a);
    vec3 col = vec3(r, 0.5 + 0.5 * sin(a * 3.0), shade(0.8));
    bump(col, 0.1);
    col = shade(col);
    fragColor = vec4(col, 1.0);
}
