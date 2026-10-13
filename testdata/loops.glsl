// while, do-while, continue, nested loops and returns from inside loops.
float firstMultiple(int n, int of)
{
    for (int i = 1; i < 64; i++) {
        if (i * of >= n) {
            return float(i);
        }
    }
    return 64.0;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    ivec2 ip = ivec2(fragCoord) / 4;
    int count = 0;
    int k = ip.x;
    while (k > 0) {
        if (k % 3 == 0) {
            k -= 1;
            continue;
        }
        count += 1;
        k -= 2;
    }
    int j = 0;
    do {
        j += 3;
    } while (j < ip.y);
    float sum = 0.0;
    for (int a = 0; a < 4; a++) {
        for (int b = 0; b < 4; b++) {
            if (b > a) break;
            sum += float(a * b);
        }
    }
    float f = firstMultiple(ip.x + ip.y, 5);
    fragColor = vec4(float(count) / 16.0, float(j % 32) / 32.0, sum / 36.0 + f / 128.0, 1.0);
}
