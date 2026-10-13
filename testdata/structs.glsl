// Structs, arrays of structs and dynamic indexing.
struct Light {
    vec3 dir;
    vec3 color;
};

struct Material {
    vec3 albedo;
    float shine;
};

Light lights[3];

vec3 lit(Material m, vec3 n)
{
    vec3 sum = vec3(0.0);
    for (int i = 0; i < 3; i++) {
        float d = max(dot(n, normalize(lights[i].dir)), 0.0);
        sum += m.albedo * lights[i].color * pow(d, m.shine);
    }
    return sum;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    lights[0] = Light(vec3(1.0, 1.0, 0.5), vec3(1.0, 0.3, 0.2));
    lights[1] = Light(vec3(-1.0, 0.5, 0.5), vec3(0.2, 1.0, 0.3));
    lights[2] = Light(vec3(0.0, -1.0, 0.5), vec3(0.2, 0.3, 1.0));
    vec2 p = (fragCoord / iResolution.xy) * 2.0 - 1.0;
    float z2 = 1.0 - dot(p, p);
    if (z2 < 0.0) {
        fragColor = vec4(0.05, 0.05, 0.08, 1.0);
        return;
    }
    vec3 n = vec3(p, sqrt(z2));
    Material m = Material(vec3(0.9), 2.0);
    float weights[4] = float[4](0.25, 0.5, 0.75, 1.0);
    int k = int(clamp(fragCoord.x / iResolution.x * 4.0, 0.0, 3.0));
    fragColor = vec4(lit(m, n) * weights[k], 1.0);
}
