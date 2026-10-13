// The vertex shader hosts draw oksl layers with under SDL_GPU: a quad in
// target pixels (y down) with a uv per vertex (0..1, origin top-left), which
// oksl's fragment shaders read at location 0.
//
// SDL_GPU binding convention: vertex uniforms in set 1.

// Target size in pixels (xy; zw unused), like ovg's View.
struct View {
    float4 size;
};

[[vk::binding(0, 1)]] ConstantBuffer<View> view : register(b0, space1);

struct VS_In {
    float2 position : TEXCOORD0;
    float2 uv : TEXCOORD1;
};

struct VS_Out {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

// SDL_GPU puts NDC +1 at the top row of every target.
[shader("vertex")]
VS_Out vs_main(VS_In i) {
    VS_Out o;
    o.uv = i.uv;
    o.position = float4(2.0 * i.position.x / view.size.x - 1.0, 1.0 - 2.0 * i.position.y / view.size.y, 0, 1);
    return o;
}
