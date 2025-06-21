@group(0) @binding(0) var texture_out: texture_storage_2d<rgba32float, write>; // For writing
@group(0) @binding(1) var<uniform> resolution: vec2<u32>;
@group(0) @binding(2) var<uniform> uniforms: vec4<u32>;

@compute
@workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {

    let pos = vec2(id.x, id.y);
    var fragColor: vec4<f32>;

    var current_color = vec4f(0.);

    // Determine the new color based on the current time
    var new_color = current_color; // Default to current sample
    
    let fres = vec2f(resolution);

    var p = vec2f(pos);
    var uv = (vec2f(p.x, fres.y - p.y) - 0.5*fres.xy)/min(fres.x, fres.y);

    fragColor = reflection_demo(uv);

    new_color = current_color + fragColor;

    // Write to output texture
    textureStore(texture_out, pos, new_color);
}

fn reflection_demo(uv: vec2f) -> vec4<f32> {
    let fragColor = vec4f(uv.x, uv.y, 1., 1.);
    return fragColor;
}
