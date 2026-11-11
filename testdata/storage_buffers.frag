#version 310 es
precision highp float;
// Read-only storage buffers (tools/gpu_parity binds its test words to
// each): struct elements at their std430 offsets, a loop whose trip count
// is a buffer word, and a helper reading by index.
struct Cell { vec3 tint; float weight; vec2 offset; };
layout(std430) readonly buffer Cells { Cell cells[]; } grid;
readonly buffer Words { uint words[]; };
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
uint word(int i) { return words[i]; }
void main() {
    ivec2 p = ivec2(uv * vec2(16.0, 9.0));
    Cell c = grid.cells[p.y * 16 + p.x];
    int n = int(word(p.x * 5 + p.y) >> 16u) & 7;
    float acc = 0.0;
    for (int k = 0; k < n; k++) {
        acc += grid.cells[k + p.y].weight;
    }
    color = vec4(c.tint * c.weight + vec3(c.offset, acc / 8.0), 1.0);
}
