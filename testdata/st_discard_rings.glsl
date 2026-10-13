// Rings cut out with discard (in a helper and inside a loop); discarded
// pixels keep what was there.
#define RING(r, w, d) (abs((d) - (r)) < (w))

void cut(float d) {
    if (d > 0.48) discard;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 p = (2.0 * fragCoord - iResolution.xy) / iResolution.y;
    float d = length(p);
    cut(d * 0.5);
    for (int i = 1; i < 4; i++) {
        if (RING(0.25 * float(i), 0.02, d)) {
            discard;
        }
    }
    float a = atan(p.y, p.x);
    vec3 col = 0.5 + 0.5 * cos(a * 3.0 + iTime + vec3(0.0, 2.0, 4.0));
    fragColor = vec4(col * (1.0 - 0.5 * d), 1.0);
}
