// User uniforms of several types, read alongside the built-ins.
uniform float uScale;
uniform vec3 uTint;
uniform vec4 uPalette[3];
uniform mat2 uRot;
uniform int uMode;
uniform bool uInvert;

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 p = (fragCoord / iResolution.xy - 0.5) * uScale;
    p = uRot * p;
    int idx = int(clamp(floor((p.x + 0.5) * 3.0), 0.0, 2.0));
    vec3 col = uPalette[idx].rgb * uTint;
    if (uMode == 1) {
        col *= 0.5 + 0.5 * sin(p.y * 10.0 + iTime);
    }
    if (uInvert) {
        col = 1.0 - col;
    }
    fragColor = vec4(col, 1.0);
}
