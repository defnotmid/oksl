#version 300 es
precision highp float;

// A storage buffer of structs holding a struct (std430: Inner at word 4,
// 4 words with its padding; the element 12 words): the element read whole
// and passed to a function, an inner struct read alone, an inner struct's
// field, and a component of it chosen at run time.
struct Inner { vec2 x; float y; };
struct Outer { vec4 a; Inner b; uint c; float d; };
readonly buffer Rows { Outer rows[]; };

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

vec4 shade(Outer o)
{
    return vec4(o.b.x, o.b.y, o.a.w * float(o.c & 3u) * 0.25 + o.d * 0.25);
}

void main()
{
    int i = int(uv.x * 32.0) + 32 * int(uv.y * 8.0);
    Outer whole = rows[i];
    Inner inner = rows[i + 1].b;
    float y = rows[i + 2].b.y;
    float x = rows[i + 3].b.x[int(uv.x * 7.0) % 2];
    color = shade(whole) * 0.5 + vec4(inner.x, inner.y * 0.5 + y * 0.5, x) * 0.5;
}
