// A ripple over iChannel0 with Shadertoy's texture calls: normalized uv,
// textureLod, texelFetch with textureSize, and a sampler2D parameter.
#define PI 3.14159265
#define RIPPLE(d, t) (0.02 * sin(40.0 * (d) - 6.0 * (t)) / (1.0 + 30.0 * (d)))

vec3 blur(sampler2D tex, vec2 uv, float r) {
    vec3 sum = vec3(0.0);
    for (int i = 0; i < 4; i++) {
        float a = float(i) * PI * 0.5;
        sum += texture(tex, uv + r * vec2(cos(a), sin(a))).rgb;
    }
    return sum * 0.25;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec2 c = uv - 0.5;
    float d = length(c);
    vec2 warped = uv + normalize(c + 1e-5) * RIPPLE(d, iTime);
    vec3 col = texture(iChannel0, warped).rgb;
    col = mix(col, blur(iChannel0, warped, 0.01), 0.5);
    col *= 0.8 + 0.2 * textureLod(iChannel0, uv, 0.0).g;
    // A texel strip along the bottom, by integer coordinates.
    ivec2 size = textureSize(iChannel0, 0);
    if (fragCoord.y < 8.0) {
        ivec2 texel = ivec2(int(uv.x * float(size.x)), 0);
        col = texelFetch(iChannel0, texel, 0).rgb;
    }
    fragColor = vec4(col, 1.0);
}
