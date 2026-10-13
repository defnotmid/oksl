// Sphere and box SDFs, raymarched with an early break, normals by central
// differences and simple lighting.
float sdSphere(vec3 p, float r) { return length(p) - r; }

float sdBox(vec3 p, vec3 b)
{
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float map(vec3 p)
{
    float s = sdSphere(p - vec3(-0.6, 0.0, 0.0), 0.5);
    float b = sdBox(p - vec3(0.6, 0.0, 0.0), vec3(0.4));
    float ground = p.y + 0.5;
    return min(min(s, b), ground);
}

vec3 calcNormal(vec3 p)
{
    const float h = 0.001;
    vec2 k = vec2(1.0, -1.0);
    return normalize(k.xyy * map(p + k.xyy * h) +
                     k.yyx * map(p + k.yyx * h) +
                     k.yxy * map(p + k.yxy * h) +
                     k.xxx * map(p + k.xxx * h));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    vec3 ro = vec3(0.0, 0.3, 2.5);
    vec3 rd = normalize(vec3(uv, -1.5));
    float t = 0.0;
    bool hit = false;
    for (int i = 0; i < 80; i++) {
        float d = map(ro + rd * t);
        if (d < 0.001) {
            hit = true;
            break;
        }
        t += d;
        if (t > 20.0) break;
    }
    vec3 col = vec3(0.6, 0.75, 0.9) - 0.4 * uv.y;
    if (hit) {
        vec3 p = ro + rd * t;
        vec3 n = calcNormal(p);
        vec3 l = normalize(vec3(0.6, 0.8, 0.4));
        float diff = clamp(dot(n, l), 0.0, 1.0);
        col = vec3(0.2) + vec3(0.8, 0.7, 0.5) * diff;
    }
    fragColor = vec4(pow(col, vec3(0.4545)), 1.0);
}
