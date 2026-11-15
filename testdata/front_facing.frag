#version 300 es
precision highp float;

// gl_FrontFacing and gl_FragCoord.zw on the layer quad, which hosts draw
// counter-clockwise on screen: front-facing under SDL_GPU's default front
// face, as the CPU's fragments are, at depth 0 and w 1. A back-facing quad
// would show the uv mirrored; a wrong z or w, a different blue or alpha.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

void main()
{
    vec2 face = gl_FrontFacing ? uv : vec2(1.0) - uv.yx;
    color = vec4(face, gl_FragCoord.z + 0.25, gl_FragCoord.w * 0.75);
}
