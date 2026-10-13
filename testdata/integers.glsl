// Integer and bit operations: a PCG hash, shifts, % and bit casts.
uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    ivec2 ip = ivec2(fragCoord);
    uint seed = uint(ip.x) + uint(ip.y) * 1024u;
    uint h = pcg(seed);
    float r = float(h & 255u) / 255.0;
    int checker = (ip.x / 8 + ip.y / 8) % 2;
    int bits = (ip.x << 2) | (ip.y & 3);
    float g = float(bits % 64) / 63.0;
    uint fb = floatBitsToUint(fragCoord.x);
    float back = uintBitsToFloat(fb);
    int neg = -ip.x;
    float b = float(checker) * 0.5 + float(abs(neg) % 7) / 14.0 + (back - fragCoord.x);
    fragColor = vec4(r, g, b, 1.0);
}
