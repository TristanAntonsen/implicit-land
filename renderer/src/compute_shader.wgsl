@group(0) @binding(0) var texture_out: texture_storage_2d<rgba32float, write>; // For writing
@group(0) @binding(1) var<uniform> resolution: vec2<u32>;
@group(0) @binding(2) var<uniform> uniforms: vec4<u32>;

fn linear_to_srgb(c: f32) -> f32 {
    return select(12.92 * c, 1.055 * pow(c, 1.0 / 2.4) - 0.055, c > 0.0031308);
}

fn gamma_correct(color: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        linear_to_srgb(color.r),
        linear_to_srgb(color.g),
        linear_to_srgb(color.b),
        color.a
    );
}

@compute
@workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {

    let pos = vec2(id.x, id.y);
    var fragColor: vec4<f32>;

    var current_color = vec4f(0.);
    
    let fres = vec2f(resolution);

    var p = vec2f(pos);
    var uv = (vec2f(p.x, fres.y - p.y) - 0.5*fres.xy)/min(fres.x, fres.y);
    
    fragColor = render(uv);

    // Write to output texture
    textureStore(texture_out, pos, fragColor);
}


fn circle(p: vec2f, c: vec2f, r: f32) -> f32 {

    return length(p-c) - r;
    
}

fn box(p: vec2f, c: vec2f, s: vec2f) -> f32 {

    return max(abs(p.x-c.x)-s.x/2., abs(p.y-c.y)-s.y/2.);
    
}

fn map(p: vec2<f32>) -> f32 {

    let d = circle(p, vec2f(0.), 0.5);
    
    return d;
}

fn overlay(startColor: vec4<f32>, color: vec4<f32>, d: f32) -> vec4<f32> {
    return mix(color, startColor, smoothstep(-0.5, 0.5, d));
}

fn contourFac(d: f32, scale: f32, thick: f32) -> f32 {
    let halfContour = 0.5 * scale;
    let c = abs((abs(d) - halfContour) % scale - halfContour);
    return smoothstep(0.0, 1.0, c / thick);
}

fn render(uv: vec2<f32>) -> vec4<f32> {
    let res = map(uv);
    let d = res;


    let field = d;
    let contour = 0.03;

    let outerColor = vec4<f32>(38., 70., 83., 255.) / 255.;
    let innerColor = vec4<f32>(42., 157., 143., 255.) / 255.;

    let lineWidth = 0.003;

    var fragColor = mix(innerColor, outerColor, smoothstep(-0.5, 0.5, d / lineWidth));
    
    fragColor = mix(fragColor * 1.25, fragColor, contourFac(d, contour, lineWidth));

    fragColor = overlay(fragColor, vec4<f32>(1.0), abs(d) * 200.0);
    fragColor.w = 1.;
    fragColor = clamp(fragColor, vec4f(0.0), vec4f(1.0));
    return fragColor;
}

