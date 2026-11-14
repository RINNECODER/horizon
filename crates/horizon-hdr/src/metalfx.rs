//! MetalFX's spatial and temporal scalers over wgpu's Metal textures, for the renderer's
//! Upscaler setting: the Metal calls the crates that forbid unsafe code can't make.
//!
//! wgpu forbids raw Metal commands in an encoder that records wgpu ones, so MetalFX gets
//! an encoder of its own: the frame so far and MetalFX are submitted together (one Metal
//! queue runs them in order) and the caller's encoder is swapped for a fresh one.
//! Wherever MetalFX can't run (another backend, a format it refuses, the 16-bit HDR frame
//! whose alpha marks it would not keep), nothing is drawn and the caller falls back.

use objc2::Message;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLDevice, MTLPixelFormat, MTLTexture};
use objc2_metal_fx::{
    MTLFXSpatialScaler, MTLFXSpatialScalerBase, MTLFXSpatialScalerColorProcessingMode,
    MTLFXSpatialScalerDescriptor, MTLFXTemporalScaler, MTLFXTemporalScalerBase,
    MTLFXTemporalScalerDescriptor,
};
use wgpu::hal::api::Metal;

type Raw<T> = Retained<ProtocolObject<T>>;
/// Input size, output size, input format: what a scaler is built for.
type Key = ([u32; 2], [u32; 2], MTLPixelFormat);

const OUTPUT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const MOTION: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;

struct Spatial {
    key: Key,
    scaler: Raw<dyn MTLFXSpatialScaler>,
    output: (Raw<dyn MTLTexture>, wgpu::TextureView),
}

struct Temporal {
    key: Key,
    scaler: Raw<dyn MTLFXTemporalScaler>,
    output: (Raw<dyn MTLTexture>, wgpu::TextureView),
    motion: (Raw<dyn MTLTexture>, wgpu::TextureView),
}

/// The jittered world MetalFX Temporal rebuilds.
pub struct Frame<'a> {
    pub color: &'a wgpu::TextureView,
    /// Reversed depth (1 nearest), `Depth32Float`.
    pub depth: &'a wgpu::TextureView,
    pub render_size: [u32; 2],
    pub output_size: [u32; 2],
    /// This frame's jitter in rendered pixels, y down (FSR's `jitterOffset`).
    pub jitter_pixels: [f32; 2],
    /// The history is unusable: a cut, a resize.
    pub reset: bool,
}

/// One world's scalers and their textures, rebuilt when the sizes or format change.
pub struct State {
    device: wgpu::Device,
    spatial: Option<Spatial>,
    temporal: Option<Temporal>,
    /// What MetalFX refused, not asked again until the sizes or format change.
    refused: Vec<(bool, Key)>,
    /// -1 flips the jitter handed to MetalFX (`HORIZON_METALFX_JITTER=-1`), a developer's
    /// check of the convention.
    jitter_sign: f32,
}

// Safety: Metal devices, textures and MetalFX scalers may be retained and released on any
// thread; the renderer that owns a `State` uses it from one thread at a time.
unsafe impl Send for State {}
unsafe impl Sync for State {}

fn raw(texture: &wgpu::Texture) -> Option<Raw<dyn MTLTexture>> {
    // SAFETY: the handle is only read and retained; wgpu keeps owning the texture.
    unsafe { texture.as_hal::<Metal>().map(|t| t.raw_handle().retain()) }
}

fn target(
    device: &wgpu::Device,
    label: &str,
    size: [u32; 2],
    format: wgpu::TextureFormat,
) -> Option<(Raw<dyn MTLTexture>, wgpu::TextureView)> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        // ShaderRead, ShaderWrite and RenderTarget: what MetalFX's outputs ask for.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    Some((raw(&texture)?, texture.create_view(&Default::default())))
}

/// The 8-bit frame only: the 16-bit HDR frame keeps light marks in alpha (docs/hdr-passes.md).
fn takes(format: MTLPixelFormat) -> bool {
    [
        MTLPixelFormat::BGRA8Unorm,
        MTLPixelFormat::RGBA8Unorm,
        MTLPixelFormat::BGRA8Unorm_sRGB,
        MTLPixelFormat::RGBA8Unorm_sRGB,
    ]
    .contains(&format)
}

/// Submit the frame so far and MetalFX's commands (`f`) after it; `encoder` continues
/// the frame.
fn encode(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    encoder: &mut wgpu::CommandEncoder,
    f: impl FnOnce(&ProtocolObject<dyn MTLCommandBuffer>),
) {
    let mut own = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MetalFX"),
    });
    // SAFETY: the buffer only gets MetalFX's encode; wgpu ends and submits it.
    unsafe {
        own.as_hal_mut::<Metal, _, _>(|e| {
            if let Some(buffer) = e.and_then(|e| e.raw_command_buffer()) {
                f(buffer);
            }
        });
    }
    let before = std::mem::replace(
        encoder,
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Horizon frame after MetalFX"),
        }),
    );
    queue.submit([before.finish(), own.finish()]);
}

impl State {
    pub fn new(device: &wgpu::Device) -> Self {
        let flip = std::env::var("HORIZON_METALFX_JITTER").is_ok_and(|v| v.trim() == "-1");
        Self {
            device: device.clone(),
            spatial: None,
            temporal: None,
            refused: Vec::new(),
            jitter_sign: if flip { -1.0 } else { 1.0 },
        }
    }

    fn metal_device(&self) -> Option<Raw<dyn MTLDevice>> {
        // SAFETY: read and retained only.
        unsafe {
            self.device
                .as_hal::<Metal>()
                .map(|d| d.raw_device().clone())
        }
    }

    /// Notes once that MetalFX refused `key` (`temporal`: which scaler) and why.
    fn refuse<T>(&mut self, temporal: bool, key: Key, why: &str) -> Option<T> {
        log::warn!(
            "Horizon renderer: MetalFX {} unavailable for {:?} -> {:?} ({why}); using {}",
            if temporal { "Temporal" } else { "Spatial" },
            key.0,
            key.1,
            if temporal { "TAA" } else { "the bilinear copy" }
        );
        self.refused.push((temporal, key));
        None
    }

    pub fn spatial(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        input_size: [u32; 2],
        output_size: [u32; 2],
    ) -> Option<wgpu::TextureView> {
        let color = raw(input.texture())?;
        // SAFETY: plain property reads and writes on live MetalFX/Metal objects.
        unsafe {
            let key = (input_size, output_size, color.pixelFormat());
            if self.refused.contains(&(false, key)) {
                return None;
            }
            if self.spatial.as_ref().is_none_or(|s| s.key != key) {
                self.spatial = None;
                if !takes(key.2) {
                    return self.refuse(false, key, "16-bit frame");
                }
                let Some(device) = self.metal_device() else {
                    return self.refuse(false, key, "not the Metal backend");
                };
                let desc = MTLFXSpatialScalerDescriptor::new();
                desc.setColorTextureFormat(key.2);
                desc.setOutputTextureFormat(MTLPixelFormat::RGBA16Float);
                desc.setInputWidth(input_size[0] as usize);
                desc.setInputHeight(input_size[1] as usize);
                desc.setOutputWidth(output_size[0] as usize);
                desc.setOutputHeight(output_size[1] as usize);
                // The picture is the game's gamma colours, as shown.
                desc.setColorProcessingMode(MTLFXSpatialScalerColorProcessingMode::Perceptual);
                let Some(scaler) = desc.newSpatialScalerWithDevice(&device) else {
                    return self.refuse(false, key, "no scaler for this device or format");
                };
                let Some(output) = target(&self.device, "MetalFX Spatial", output_size, OUTPUT)
                else {
                    return self.refuse(false, key, "no output texture");
                };
                if !output.0.usage().contains(scaler.outputTextureUsage())
                    || !color.usage().contains(scaler.colorTextureUsage())
                {
                    return self.refuse(false, key, "texture usage");
                }
                self.spatial = Some(Spatial {
                    key,
                    scaler,
                    output,
                });
            }
            let s = self.spatial.as_ref()?;
            s.scaler.setColorTexture(Some(&color));
            s.scaler.setOutputTexture(Some(&s.output.0));
            s.scaler.setInputContentWidth(input_size[0] as usize);
            s.scaler.setInputContentHeight(input_size[1] as usize);
            encode(&self.device, queue, encoder, |b| {
                s.scaler.encodeToCommandBuffer(b)
            });
            Some(s.output.1.clone())
        }
    }

    pub fn temporal(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &Frame<'_>,
        motion: impl FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView),
    ) -> Option<wgpu::TextureView> {
        let color = raw(frame.color.texture())?;
        let depth = raw(frame.depth.texture())?;
        // SAFETY: plain property reads and writes on live MetalFX/Metal objects.
        unsafe {
            let key = (frame.render_size, frame.output_size, color.pixelFormat());
            if self.refused.contains(&(true, key)) {
                return None;
            }
            if self.temporal.as_ref().is_none_or(|t| t.key != key) {
                self.temporal = None;
                if !takes(key.2) {
                    return self.refuse(true, key, "16-bit frame");
                }
                let Some(device) = self.metal_device() else {
                    return self.refuse(true, key, "not the Metal backend");
                };
                let desc = MTLFXTemporalScalerDescriptor::new();
                desc.setColorTextureFormat(key.2);
                desc.setDepthTextureFormat(depth.pixelFormat());
                desc.setMotionTextureFormat(MTLPixelFormat::RG16Float);
                desc.setOutputTextureFormat(MTLPixelFormat::RGBA16Float);
                desc.setInputWidth(key.0[0] as usize);
                desc.setInputHeight(key.0[1] as usize);
                desc.setOutputWidth(key.1[0] as usize);
                desc.setOutputHeight(key.1[1] as usize);
                desc.setAutoExposureEnabled(false);
                let Some(scaler) = desc.newTemporalScalerWithDevice(&device) else {
                    return self.refuse(true, key, "no scaler for this device or format");
                };
                let (Some(output), Some(motion)) = (
                    target(&self.device, "MetalFX Temporal", key.1, OUTPUT),
                    target(&self.device, "MetalFX motion", key.0, MOTION),
                ) else {
                    return self.refuse(true, key, "no output texture");
                };
                if !output.0.usage().contains(scaler.outputTextureUsage())
                    || !motion.0.usage().contains(scaler.motionTextureUsage())
                    || !color.usage().contains(scaler.colorTextureUsage())
                    || !depth.usage().contains(scaler.depthTextureUsage())
                {
                    return self.refuse(true, key, "texture usage");
                }
                // Horizon's depth is reversed: 1 nearest.
                scaler.setDepthReversed(true);
                self.temporal = Some(Temporal {
                    key,
                    scaler,
                    output,
                    motion,
                });
            }
            let t = self.temporal.as_ref()?;
            motion(encoder, &t.motion.1);
            let s = &t.scaler;
            s.setColorTexture(Some(&color));
            s.setDepthTexture(Some(&depth));
            s.setMotionTexture(Some(&t.motion.0));
            s.setOutputTexture(Some(&t.output.0));
            s.setInputContentWidth(key.0[0] as usize);
            s.setInputContentHeight(key.0[1] as usize);
            s.setJitterOffsetX(frame.jitter_pixels[0] * self.jitter_sign);
            s.setJitterOffsetY(frame.jitter_pixels[1] * self.jitter_sign);
            // The motion is in UV: times the render size, in rendered pixels.
            s.setMotionVectorScaleX(key.0[0] as f32);
            s.setMotionVectorScaleY(key.0[1] as f32);
            s.setReset(frame.reset);
            encode(&self.device, queue, encoder, |b| s.encodeToCommandBuffer(b));
            Some(t.output.1.clone())
        }
    }
}
