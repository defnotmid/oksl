#version 300 es
precision highp float;

// A GLSL ES 3.0 fragment shader on the layer: uv from the layer vertex
// shader at location 0, gl_FragCoord, a sampler without Shadertoy's flip,
// discard, functions that read and write the stage variables, a swizzled
// write to the output and a global initializer.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

uniform float uSpeed;
uniform sampler2D tex;

const float RINGS = 6.0;
vec2 centre = vec2(0.5, 0.45);

float ring(void)
{
    float d = length(uv - centre) * RINGS;
    return 0.35 + 0.65 * smoothstep(0.2, 0.3, abs(fract(d) - 0.5));
}

void shade()
{
    vec4 t = texture(tex, uv * 2.0);
    float checker = mod(floor(gl_FragCoord.x / 8.0) + floor(gl_FragCoord.y / 8.0), 2.0);
    color = vec4(mix(t.rgb, vec3(checker), 0.25) * ring(), 1.0);
}

void main()
{
    if (uv.x > 0.95) {
        discard;
    }
    shade();
    color.b += uSpeed * 0.1;
}
