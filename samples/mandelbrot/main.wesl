import package::fullscreen_quad::{fullscreen_quad, QuadVertex};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> QuadVertex {
    return fullscreen_quad(vertex_index);
}


@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    let mandelbrotEscaped = package::mandelbrot::mandelbrot(uv * 3.0 - vec2f(2.0, 1.5));    
    
    @if(DEBUG)
    let color = random_wgsl::sinRand(vec2f(mandelbrotEscaped));
    @if(!DEBUG)
    let color = mandelbrotEscaped;
    
    return vec4(vec3f(color), 1.0);
}