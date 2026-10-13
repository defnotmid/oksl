// Matrix == and != (column by column in SPIR-V), on matrices that agree
// in some cells and not others.
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 cell = floor(fragCoord / iResolution.xy * vec2(4.0, 3.0));
    mat2 a = mat2(cell.x, 1.0, 2.0, cell.y);
    mat2 b = mat2(2.0, 1.0, 2.0, 1.0);
    mat2x3 c = mat2x3(cell.x, 0.0, 1.0, 0.0, cell.y, 1.0);
    mat2x3 d = mat2x3(1.0, 0.0, 1.0, 0.0, 2.0, 1.0);
    mat3 e = mat3(cell.x);
    float r = a == b ? 1.0 : 0.0;
    float g = c != d ? 1.0 : 0.25;
    float bl = e == mat3(3.0) ? 1.0 : (e != mat3(0.0) ? 0.5 : 0.0);
    fragColor = vec4(r, g, bl, 1.0);
}
