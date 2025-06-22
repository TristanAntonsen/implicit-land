use pyo3::prelude::*;
//////// Python binding stuff

/// Renders the result from python
#[pyfunction]
fn render_with_wgsl(shader: &str, resolution: usize, output_path: &str) -> PyResult<()> {
    Ok(render(shader.to_string(), resolution as u32, output_path))
}

/// A Python module implemented in Rust.cl
#[pymodule]
fn renderer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(render_with_wgsl, m)?)?;
    Ok(())
}

//////// Renderer stuff

use std::fs::File;
use std::io::BufWriter;
use std::{thread, time};
use wgpu::util::DeviceExt;
use wgpu::Adapter;

const GRID_SIZE: u32 = 16;

pub fn render(shader: String, resolution: u32, output_path: &str) {
    let time = time::Duration::from_millis(3);
    thread::sleep(time);

    pollster::block_on(run(shader, resolution, output_path));
}

pub struct ComputeState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
    resolution: u32,
    output_texture: wgpu::Texture,
    _uniform_buf: wgpu::Buffer,
}

impl ComputeState {
    async fn new(adapter: Adapter, shader: String, resolution: u32) -> ComputeState {
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    // required_features: wgpu::Features::BGRA8UNORM_STORAGE,
                    // required_features: ['bgra8unorm-storage'],
                    required_limits: wgpu::Limits::default(),
                    label: None,
                    memory_hints: Default::default(),
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });

        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Compute Output Texture"),
            size: wgpu::Extent3d {
                width: resolution,
                height: resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let size_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Resolution Buffer"),
            contents: bytemuck::cast_slice(&[resolution, resolution]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let _uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[0, 0, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create a view for the output texture
        let output_texture_view =
            output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    // Output texture (storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Resolution buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Time buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&output_texture_view), // Bind output texture
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &size_buf,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &_uniform_buf,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            compute_pipeline,
            compute_bind_group,
            resolution,
            output_texture,
            _uniform_buf,
        }
    }

    fn update(&mut self) -> () {}

    async fn save_rgba32float_to_rgba8_image(
        &self,
        output_staging_buffer: &wgpu::Buffer,
        resolution: (u32, u32),
        output_file: &str,
    ) {
        // Map the buffer to access the data
        let buffer_slice = output_staging_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.receive().await.unwrap().unwrap();

        // Retrieve the raw data
        let data = buffer_slice.get_mapped_range();
        let texture_width = resolution.0 as usize;
        let texture_height = resolution.1 as usize;
        let bytes_per_row = ((texture_width * 16 + 255) / 256) * 256; // Adjust for 256-byte alignment

        // Convert the f32 data to u8
        let mut rgba_u8_data = Vec::with_capacity(texture_width * texture_height * 4);
        for row in 0..texture_height {
            let row_start = row * bytes_per_row;
            let row_end = row_start + texture_width * 16; // 16 bytes per pixel for RGBA32Float
            let row_data = &data[row_start..row_end];

            for chunk in row_data.chunks_exact(16) {
                let r = f32::from_ne_bytes(chunk[0..4].try_into().unwrap());
                let g = f32::from_ne_bytes(chunk[4..8].try_into().unwrap());
                let b = f32::from_ne_bytes(chunk[8..12].try_into().unwrap());
                let a = f32::from_ne_bytes(chunk[12..16].try_into().unwrap());

                // Normalize and convert to u8
                rgba_u8_data.push((r.clamp(0.0, 1.0) * 255.0).round() as u8);
                rgba_u8_data.push((g.clamp(0.0, 1.0) * 255.0).round() as u8);
                rgba_u8_data.push((b.clamp(0.0, 1.0) * 255.0).round() as u8);
                rgba_u8_data.push((a.clamp(0.0, 1.0) * 255.0).round() as u8);

                // Why the hell is color space stuff so confusing
                // rgba_u8_data.push((linear_to_srgb(r.clamp(0.0, 1.0)) * 255.0).round() as u8);
                // rgba_u8_data.push((linear_to_srgb(g.clamp(0.0, 1.0)) * 255.0).round() as u8);
                // rgba_u8_data.push((linear_to_srgb(b.clamp(0.0, 1.0)) * 255.0).round() as u8);
                // rgba_u8_data.push((a.clamp(0.0, 1.0) * 255.0).round() as u8); // leave alpha linear
            }
        }
        // Save to an image
        use png::Encoder;

        let file = File::create(output_file).expect("Failed to create output file");
        let w = &mut BufWriter::new(file);

        // Configure PNG encoder
        let mut encoder = Encoder::new(w, self.resolution, self.resolution);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);

        // Mark image as using sRGB transfer function
        encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);

        let mut writer = encoder.write_header().expect("Failed to write PNG header");

        writer
            .write_image_data(&rgba_u8_data)
            .expect("Failed to write image data");

        // Unmap the buffer
        drop(data); // Ensure no references remain to the data slice
        output_staging_buffer.unmap();
    }

    fn render(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            compute_pass.set_pipeline(&self.compute_pipeline);

            let workgroups_x = self.resolution / GRID_SIZE;
            let workgroups_y = self.resolution / GRID_SIZE;

            compute_pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, 1);
        }

        // self.device.poll(wgpu::Maintain::Wait);
        self.device.poll(wgpu::Maintain::Poll);

        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
    }
}

pub async fn run(shader: String, resolution: u32, output_path: &str) {
    let instance = wgpu::Instance::default();

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let mut compute_state = ComputeState::new(adapter, shader, resolution).await;
    compute_state.render();

    // println!("FINISHED RENDER");
    finish_and_export(compute_state, output_path);
}

pub fn finish_and_export(compute_state: ComputeState, output_path: &str) {
    let mut encoder =
        compute_state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Copy Command Encoder"),
            });

    let output_staging_buffer = compute_state.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (compute_state.resolution as u64
            * compute_state.resolution as u64
            * 4
            * std::mem::size_of::<f32>() as u64),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &compute_state.output_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &output_staging_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                // This needs to be padded to 256.
                bytes_per_row: Some(compute_state.resolution * 16),
                rows_per_image: Some(compute_state.resolution),
            },
        },
        wgpu::Extent3d {
            width: compute_state.resolution,
            height: compute_state.resolution,
            depth_or_array_layers: 1,
        },
    );
    compute_state
        .queue
        .submit(std::iter::once(encoder.finish()));
    // println!("SAVING");

    // Save the texture to an image before exiting
    pollster::block_on(async {
        compute_state
            .save_rgba32float_to_rgba8_image(
                &output_staging_buffer,
                (compute_state.resolution, compute_state.resolution),
                output_path,
            )
            .await;
    });
}


// See above comment about disliking color spaces

// fn linear_to_gamma(linear: f32) -> f32 {
//     if linear > 0. {
//         return linear.sqrt();
//     };

//     return 0.;
// }

// fn linear_to_srgb(c: f32) -> f32 {
//     if c <= 0.0031308 {
//         12.92 * c
//     } else {
//         1.055 * c.powf(1.0 / 2.4) - 0.055
//     }
// }
