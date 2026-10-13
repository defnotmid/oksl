// Truchet tiles picked with a switch (with fallthrough and a default in
// the middle), function-like macros and a quality switch on Shadertoy's
// HW_PERFORMANCE.
#define S(a, b, t) smoothstep(a, b, t)
#define ROT(a) mat2(cos(a), -sin(a), sin(a), cos(a))
// An integer hash: exact on every backend (sin-based hashes are not).
#define HASH(p) hash21(ivec2(p) + 1000)
#if HW_PERFORMANCE == 0
#define AA 1
#else
#define AA 2
#endif

float hash21(ivec2 p) {
    uvec2 q = uvec2(p);
    uint h = (q.x * 1597334677u) ^ (q.y * 3812015801u);
    h *= 1597334677u;
    return float(h >> 8u) / 16777216.0;
}

float tile(vec2 p, int kind) {
    float d = 1e9;
    switch (kind) {
    case 0:
        p.x = -p.x;
        // falls through: a mirrored arc
    case 1:
        d = min(abs(length(p - 0.5) - 0.5), abs(length(p + 0.5) - 0.5));
        break;
    default:
        d = abs(p.x);
        break;
    case 3:
        d = min(abs(p.x), abs(p.y));
    }
    return d;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec3 col = vec3(0.0);
    for (int m = 0; m < AA; m++)
    for (int n = 0; n < AA; n++) {
        vec2 o = vec2(float(m), float(n)) / float(AA) - 0.5;
        vec2 uv = (fragCoord + o - 0.5 * iResolution.xy) / iResolution.y;
        uv = ROT(0.1 * iTime) * uv * 6.0;
        vec2 id = floor(uv);
        vec2 gv = fract(uv) - 0.5;
        int kind = int(HASH(id) * 4.0);
        float d = tile(gv, kind);
        float line = S(0.08, 0.05, d);
        vec3 tint = 0.5 + 0.5 * cos(6.2831 * (HASH(id + 3.0) + vec3(0.0, 0.33, 0.67)));
        col += mix(vec3(0.05), tint, line);
    }
    fragColor = vec4(col / float(AA * AA), 1.0);
}
