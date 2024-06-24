struct VertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.position = vec4<f32>(in.pos, 1.0);
    out.color = vec4<f32>(in.color, 1.0);

    return out;
}

@fragment
fn fs_main(out: VertexOut) -> @location(0) vec4<f32> {
    return out.color;
}
