// Seven-segment digits counting the frames: a switch per digit with
// grouped labels, break inside a loop inside a case, continue through a
// switch, token pasting and #ifdef.
#define SEG_A 1
#define SEG_B 2
#define SEG_C 4
#define SEG_D 8
#define SEG_E 16
#define SEG_F 32
#define SEG_G 64
#define SEG(n) SEG_ ## n
#define HAS(mask, n) (((mask) & SEG(n)) != 0)
#define SHOW_GRID

int segments(int digit) {
    int mask = 0;
    switch (digit) {
    case 0: mask = SEG(A) | SEG(B) | SEG(C) | SEG(D) | SEG(E) | SEG(F); break;
    case 1: mask = SEG(B) | SEG(C); break;
    case 2: mask = SEG(A) | SEG(B) | SEG(G) | SEG(E) | SEG(D); break;
    case 3: mask = SEG(A) | SEG(B) | SEG(G) | SEG(C) | SEG(D); break;
    case 4: mask = SEG(F) | SEG(G) | SEG(B) | SEG(C); break;
    case 5: mask = SEG(A) | SEG(F) | SEG(G) | SEG(C) | SEG(D); break;
    case 6:
    case 8:
        mask = SEG(A) | SEG(F) | SEG(G) | SEG(E) | SEG(C) | SEG(D);
        if (digit == 8) mask |= SEG(B);
        break;
    case 7: mask = SEG(A) | SEG(B) | SEG(C); break;
    default:
        // 9: count the segments in.
        for (int i = 0; i < 7; i++) {
            if (i == 4) continue;
            mask |= 1 << i;
            if (mask == 0x6F) break;
        }
    }
    return mask;
}

float box(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float digit(vec2 p, int n) {
    int m = segments(n);
    float d = 1e9;
    const float w = 0.08;
    if (HAS(m, A)) d = min(d, box(p - vec2(0.0, 0.8), vec2(0.35, w)));
    if (HAS(m, B)) d = min(d, box(p - vec2(0.4, 0.4), vec2(w, 0.35)));
    if (HAS(m, C)) d = min(d, box(p - vec2(0.4, -0.4), vec2(w, 0.35)));
    if (HAS(m, D)) d = min(d, box(p - vec2(0.0, -0.8), vec2(0.35, w)));
    if (HAS(m, E)) d = min(d, box(p - vec2(-0.4, -0.4), vec2(w, 0.35)));
    if (HAS(m, F)) d = min(d, box(p - vec2(-0.4, 0.4), vec2(w, 0.35)));
    if (HAS(m, G)) d = min(d, box(p, vec2(0.35, w)));
    return d;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y * 4.0;
    int value = iFrame;
    vec3 col = vec3(0.04, 0.05, 0.08);
#ifdef SHOW_GRID
    vec2 g = abs(fract(uv) - 0.5);
    col += 0.03 * step(0.48, max(g.x, g.y));
#endif
    for (int k = 0; k < 4; k++) {
        int place = 3 - k;
        int d = value % 10;
        value /= 10;
        switch (place) {
        case 0:
        case 1:
        case 2:
        case 3: {
            vec2 q = uv - vec2(1.5 - float(k), 0.0);
            float s = digit(q, d);
            col = mix(col, vec3(1.0, 0.45, 0.2), smoothstep(0.03, 0.0, s));
            continue;
        }
        }
    }
    fragColor = vec4(col, 1.0);
}
