@group(0) @binding(0) var texture_out: texture_storage_2d<rgba32float, write>; // For writing
@group(0) @binding(1) var<uniform> resolution: vec2<u32>;
@group(0) @binding(2) var<uniform> uniforms: vec4<u32>;

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

// Maximum/minumum elements of a vector
// https://mercury.sexy/hg_sdf/
fn vmax(v: vec2f) -> f32{
	return max(v.x, v.y);
}

// Box: correct distance to corners
// https://mercury.sexy/hg_sdf/ (modified)
fn box(p: vec2f, b: vec2f) -> f32 {
	let d: vec2f = abs(p) - b * 0.5;
	return length(max(d, vec2f(0))) + vmax(min(d, vec2f(0)));
}

fn boxSharp(p: vec2f, b: vec2f) -> f32 {
	let d: vec2f = abs(p) - b * 0.5;
	return max(d.x, d.y);
}

// https://mercury.sexy/hg_sdf/ (modified)
fn line(p: vec2f, a: vec2f, b: vec2f) -> f32 {
	var ab = b - a;
	var t = clamp(dot(p - a, ab) / dot(ab, ab), 0., 1.);
	return length((ab*t + a) - p);
}

fn plane(p: vec2f, c: vec2f, n: vec2f) -> f32 {
    return dot((p-c), normalize(n));
}

// The "Round" variant uses a quarter-circle to join the two objects smoothly:
fn roundUnion(a: f32, b: f32, r: f32) -> f32 {
	var u = max(vec2<f32>(r - a,r - b), vec2<f32>(0));
	return max(r, min (a, b)) - length(u);
}

fn roundIntersection(a: f32, b: f32, r: f32) -> f32 {
	var u = max(vec2<f32>(r + a,r + b), vec2<f32>(0));
	return min(-r, max (a, b)) + length(u);
}

fn roundDifference(a: f32, b: f32, r: f32) -> f32 {
	return roundIntersection(a, -b, r);
}

fn map(p: vec2<f32>) -> f32 {

    let d = MAP_FUNCTION;
    
    return d;
}

fn overlay(startColor: vec4<f32>, color: vec4<f32>, d: f32) -> vec4<f32> {
    return mix(color, startColor, smoothstep(-0.5, 0.5, d));
}

fn contourFac(d: f32, scale: f32, thick: f32) -> f32 {
    let c = abs(d + scale / 2.) % scale;
    let edge = 0.5 * thick;
    return smoothstep(edge, 0.0, abs(c - 0.5 * scale));
}

// Mostly hard coded stuff, will expose more params later
fn render(uv: vec2<f32>) -> vec4<f32> {
    let res = map(uv);
    let d = res;


    let field = d;

    let outerColor = vec4<f32>([[outer_color]]);
    let innerColor = vec4<f32>([[inner_color]]);
    let border_color = vec4<f32>([[border_color]]);
    var contour_color_inner = vec4<f32>([[contour_color_inner]]);
    var contour_color_outer = vec4<f32>([[contour_color_outer]]);
    let contour_spacing = [[contour_spacing]];

    let aaWidth = 0.001;
    let lineWidth = 0.002;
    let borderWidth = 0.002;
    // Draw base shape
    var fragColor = mix(innerColor, outerColor, smoothstep(-0.5, 0.5, d / aaWidth));

    // Draw isolines
    var cfac = contourFac(d, contour_spacing, lineWidth);
    var io = sign(d) * 0.5 + 0.5;

    // contour_color_outer = mix(contour_color_outer, fragColor, smoothstep(0., 1., field / 0.25));
    // contour_color_inner = mix(contour_color_inner, fragColor, smoothstep(0., 1., field / 0.25));

    fragColor = mix(fragColor, contour_color_inner, cfac * (1. - io));
    fragColor = mix(fragColor, contour_color_outer, cfac * io);

    // Draw Border
    fragColor = overlay(fragColor, border_color, (abs(d) - borderWidth / 2.) / aaWidth);

    // Clamp to 0-1 for u8
    fragColor = clamp(fragColor, vec4f(0.), vec4f(1.0));

    return fragColor;
}

