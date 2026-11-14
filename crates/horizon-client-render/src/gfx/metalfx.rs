//! MetalFX upscaling (macOS, Metal backend): the Upscaler setting's Spatial and Temporal.
//! Spatial rebuilds a picture smaller than the window (the world below 100% resolution
//! scale, a shader pack's `final`); Temporal takes TAA's place on the jittered world, with
//! the camera's motion vectors (`world_post.wgsl`'s `fs_motion`).
//!
//! The Metal calls live in `horizon_hdr::metalfx` (this crate forbids unsafe code). It may
//! submit the frame recorded so far and hand back a fresh encoder for the rest of it.
//! Wherever MetalFX can't run (another backend or OS, a format it refuses, the 16-bit HDR
//! frame whose alpha marks it would not keep), nothing is drawn and the caller falls back
//! to TAA or the bilinear copy.

use crate::graphics::Upscaler;

pub(super) struct MetalFx {
    pub mode: Upscaler,
    device: wgpu::Device,
    #[cfg(target_os = "macos")]
    imp: horizon_hdr::metalfx::State,
}

/// Both scalers' output size for an input size and the window's: never smaller than the
/// input (MetalFX only upscales), at most 8192.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
fn output_size(input: [u32; 2], window: [u32; 2]) -> Option<[u32; 2]> {
    let out = window.map(|v| v.clamp(1, 8192));
    (out != input && out[0] >= input[0] && out[1] >= input[1]).then_some(out)
}

impl MetalFx {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            mode: Upscaler::Taa,
            device: device.clone(),
            #[cfg(target_os = "macos")]
            imp: horizon_hdr::metalfx::State::new(device),
        }
    }

    /// `input` (its first `input_size` pixels) upscaled to `window` by MetalFX Spatial, as
    /// an `Rgba16Float` view at `window`; `None` when the setting, the sizes or the device
    /// don't call for it.
    pub fn spatial(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView,
        input_size: [u32; 2],
        window: [u32; 2],
    ) -> Option<wgpu::TextureView> {
        if self.mode == Upscaler::Taa {
            return None;
        }
        #[cfg(target_os = "macos")]
        {
            let out = output_size(input_size, window)?;
            self.imp.spatial(encoder, input, input_size, out)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (encoder, input, input_size, window);
            None
        }
    }

    /// MetalFX Temporal over TAA's frame: `motion` records the motion vectors into the view
    /// it is given (`Rg16Float`, UV, at the render size) before MetalFX runs.
    #[cfg(target_os = "macos")]
    pub fn temporal(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &super::world_post::UpscaleFrame<'_>,
        motion: impl FnOnce(&mut wgpu::CommandEncoder, &wgpu::TextureView),
    ) -> Option<wgpu::TextureView> {
        if self.mode != Upscaler::MetalFxTemporal {
            return None;
        }
        let frame = horizon_hdr::metalfx::Frame {
            color: frame.color,
            depth: frame.depth,
            render_size: frame.render_size,
            output_size: frame.output_size,
            jitter_pixels: frame.jitter_pixels,
            reset: frame.reset,
        };
        self.imp.temporal(encoder, &frame, motion)
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// The command buffers the frame was split into so far, to submit before its own.
    pub fn take_pending(&mut self) -> Vec<wgpu::CommandBuffer> {
        #[cfg(target_os = "macos")]
        return self.imp.take_pending();
        #[cfg(not(target_os = "macos"))]
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::output_size;

    #[test]
    fn output_is_the_window_when_it_is_larger() {
        assert_eq!(output_size([960, 540], [1920, 1080]), Some([1920, 1080]));
        assert_eq!(output_size([1280, 720], [1920, 1080]), Some([1920, 1080]));
        // Same size, or a world larger than the window (supersampling): no upscale.
        assert_eq!(output_size([1920, 1080], [1920, 1080]), None);
        assert_eq!(output_size([3840, 2160], [1920, 1080]), None);
        assert_eq!(output_size([1000, 2000], [1920, 1080]), None);
        assert_eq!(output_size([960, 540], [20000, 1080]), Some([8192, 1080]));
    }
}
