// Matrix construction, products, inverse, transpose and friends.
mat3 rotY(float a)
{
    float c = cos(a), s = sin(a);
    return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / iResolution.xy;
    mat3 r = rotY(uv.x * 3.0);
    vec3 v = r * vec3(uv, 1.0);
    mat3 inv = inverse(r);
    vec3 back = inv * v;
    mat2 m2 = mat2(2.0, 1.0, 1.0, 3.0);
    float det = determinant(m2);
    mat2 t = transpose(m2) * (1.0 / det);
    mat3 o = outerProduct(vec3(uv, 0.5), vec3(0.5, uv.yx));
    mat3 cm = matrixCompMult(o, mat3(2.0));
    vec2 w = vec2(uv.y, uv.x) * t;
    mat4 big = mat4(1.0);
    big[3] = vec4(0.1, 0.2, 0.3, 1.0);
    vec4 h = big * vec4(back, 1.0);
    mat2 small = mat2(big);
    vec3 col = vec3(clamp(w.x, 0.0, 1.0), cm[1][1] + small[0][0] * 0.1, fract(h.z + v.x * 0.1));
    fragColor = vec4(col, 1.0);
}
