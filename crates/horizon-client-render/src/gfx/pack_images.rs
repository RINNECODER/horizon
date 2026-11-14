//! A shader pack's own images (`image.<name>`) and storage buffers (`bufferObject.<n>`) on the
//! GPU: created at the render size they depend on, kept across frames, images cleared before a
//! frame when the pack asks.

use std::collections::BTreeMap;

use horizon_shaders::{BufferObject, CustomImage, Format, Kind};
use wgpu::util::DeviceExt;

pub(super) struct PackImage {
    pub(super) def: CustomImage,
    pub(super) format: wgpu::TextureFormat,
    pub(super) dimension: wgpu::TextureViewDimension,
    pub(super) view: Option<wgpu::TextureView>,
    /// Last frame's contents, for a pass that samples and stores the image (`copied`).
    pub(super) copy: Option<wgpu::TextureView>,
    pub(super) copied: bool,
    /// The contents before a compute pass that reads them so (`snapshot`), apart from `copy`,
    /// which a later pass may still sample.
    pub(super) before: Option<wgpu::TextureView>,
    pub(super) snapshotted: bool,
    /// The render size the image was made for (relative images follow it).
    made_for: [u32; 2],
}

impl PackImage {
    pub(super) fn reset_history(&mut self) {
        self.view = None;
        self.copy = None;
        self.before = None;
        self.made_for = [0; 2];
    }

    pub(super) fn sample_type(&self, features: wgpu::Features) -> wgpu::TextureSampleType {
        self.format
            .sample_type(None, Some(features))
            .unwrap_or(wgpu::TextureSampleType::Float { filterable: false })
    }
}

/// The images `image.*` declares, each made in `backing` of its format; `packed_float`: this
/// device stores into `Rg11b10Ufloat` (`format`).
pub(super) fn images(
    defs: Vec<CustomImage>,
    packed_float: bool,
    backing: &dyn Fn(wgpu::TextureFormat) -> wgpu::TextureFormat,
) -> Vec<PackImage> {
    defs.into_iter()
        .map(|def| PackImage {
            format: backing(format(def.format, packed_float)),
            dimension: if def.is_1d() {
                wgpu::TextureViewDimension::D1
            } else if def.is_3d() {
                wgpu::TextureViewDimension::D3
            } else {
                wgpu::TextureViewDimension::D2
            },
            def,
            view: None,
            copy: None,
            copied: false,
            before: None,
            snapshotted: false,
            made_for: [0; 2],
        })
        .collect()
}

/// The formats, in order, an image of `format` can be made in where the device can't both read
/// and write `format` (Metal: no two-channel, 16-bit normalized, snorm or packed format): one
/// of the same kind that holds its values in its channels or all four, then the 32-bit
/// four-channel format of its kind. Its programs still see `format` (`widen_images`).
pub(super) fn twins(format: wgpu::TextureFormat) -> [wgpu::TextureFormat; 2] {
    use wgpu::TextureFormat as T;
    let wide = match format.sample_type(None, None) {
        Some(wgpu::TextureSampleType::Uint) => T::Rgba32Uint,
        Some(wgpu::TextureSampleType::Sint) => T::Rgba32Sint,
        _ => T::Rgba32Float,
    };
    let near = match format {
        T::R8Snorm => T::R16Float,
        T::Rg8Snorm | T::Rgba8Snorm | T::Rg16Float | T::Rg11b10Ufloat | T::Rgb10a2Unorm => {
            T::Rgba16Float
        }
        T::R16Unorm | T::R16Snorm => T::R32Float,
        T::Rg8Unorm => T::Rgba8Unorm,
        T::Rg8Uint => T::Rgba8Uint,
        T::Rg8Sint => T::Rgba8Sint,
        T::Rg16Uint | T::Rgb10a2Uint => T::Rgba16Uint,
        T::Rg16Sint => T::Rgba16Sint,
        _ => wide,
    };
    [near, wide]
}

/// The storage format of a format `twins` gives.
fn storage_format(format: wgpu::TextureFormat) -> Option<naga::StorageFormat> {
    use naga::StorageFormat as S;
    use wgpu::TextureFormat as T;
    Some(match format {
        T::R16Float => S::R16Float,
        T::R32Float => S::R32Float,
        T::Rgba8Unorm => S::Rgba8Unorm,
        T::Rgba8Uint => S::Rgba8Uint,
        T::Rgba8Sint => S::Rgba8Sint,
        T::Rgba16Float => S::Rgba16Float,
        T::Rgba16Uint => S::Rgba16Uint,
        T::Rgba16Sint => S::Rgba16Sint,
        T::Rgba32Float => S::Rgba32Float,
        T::Rgba32Uint => S::Rgba32Uint,
        T::Rgba32Sint => S::Rgba32Sint,
        _ => return None,
    })
}

/// What a store into an image of `declared` keeps of a value: the range its channels clamp to
/// (normalized: [0, 1] or [-1, 1]; unsigned floats: 0 and up).
fn store_range(declared: wgpu::TextureFormat) -> Option<(f32, Option<f32>)> {
    use wgpu::TextureFormat as T;
    match declared {
        T::R8Unorm | T::Rg8Unorm | T::Rgba8Unorm | T::R16Unorm | T::Rg16Unorm => {
            Some((0.0, Some(1.0)))
        }
        T::Rgba16Unorm | T::Rgb10a2Unorm => Some((0.0, Some(1.0))),
        T::R8Snorm | T::Rg8Snorm | T::Rgba8Snorm | T::R16Snorm | T::Rg16Snorm => {
            Some((-1.0, Some(1.0)))
        }
        T::Rgba16Snorm => Some((-1.0, Some(1.0))),
        T::Rg11b10Ufloat => Some((0.0, None)),
        _ => None,
    }
}

/// How the stores into a widened image change their value.
#[derive(Clone, Copy)]
struct Widened {
    scalar: naga::Scalar,
    /// The declared format's channels when fewer than the four made: the rest store 0 (and w 1).
    channels: Option<u8>,
    range: Option<(f32, Option<f32>)>,
}

/// Has `module`'s storage images at `bindings` (set 0: binding, declared format, the format
/// it is made in, one of its `twins`) declared in the format made, and their stores write what
/// a store into the declared format reads back: its channels (the rest z = 0, w = 1), clamped
/// to its range.
// shortcut: stores are not quantized to the declared precision, and texels never stored (cleared
// to zero) read w = 0 where the declared format reads 1; wrap loads if a pack depends on either.
pub(super) fn widen_images(
    module: &mut naga::Module,
    bindings: &[(u32, wgpu::TextureFormat, wgpu::TextureFormat)],
) {
    use naga::{ImageClass, TypeInner};
    let mut widened = Vec::new();
    for (h, g) in module.global_variables.iter_mut() {
        let Some(b) = g.binding.as_ref().filter(|b| b.group == 0) else {
            continue;
        };
        let Some(&(_, declared, made)) = bindings.iter().find(|w| w.0 == b.binding) else {
            continue;
        };
        let (Some(twin), false) = (storage_format(made), declared == made) else {
            continue;
        };
        let TypeInner::Image {
            dim,
            arrayed,
            class: ImageClass::Storage { access, .. },
        } = module.types[g.ty].inner
        else {
            continue;
        };
        let scalar = match made.sample_type(None, None) {
            Some(wgpu::TextureSampleType::Uint) => naga::Scalar::U32,
            Some(wgpu::TextureSampleType::Sint) => naga::Scalar::I32,
            _ => naga::Scalar::F32,
        };
        let (n, m) = (declared.components(), made.components());
        let class = ImageClass::Storage {
            format: twin,
            access,
        };
        let inner = TypeInner::Image {
            dim,
            arrayed,
            class,
        };
        let name = module.types[g.ty].name.clone();
        g.ty = module
            .types
            .insert(naga::Type { name, inner }, naga::Span::UNDEFINED);
        let channels = (m == 4 && n < 4).then_some(n);
        let range = store_range(declared);
        widened.push((
            h,
            Widened {
                scalar,
                channels,
                range,
            },
        ));
    }
    if widened.is_empty() {
        return;
    }
    let functions = (module.functions.iter_mut().map(|(_, f)| f))
        .chain(module.entry_points.iter_mut().map(|e| &mut e.function));
    for f in functions {
        let body = std::mem::take(&mut f.body);
        f.body = widen_stores(body, &mut f.expressions, &mut module.types, &widened);
    }
}

/// `block` with each store into one of `widened`'s images storing its value as `Widened` says.
fn widen_stores(
    block: naga::Block,
    expressions: &mut naga::Arena<naga::Expression>,
    types: &mut naga::UniqueArena<naga::Type>,
    widened: &[(naga::Handle<naga::GlobalVariable>, Widened)],
) -> naga::Block {
    use naga::{Expression as E, Statement as St};
    let mut out = naga::Block::new();
    let inner = |b: &mut naga::Block,
                 expressions: &mut naga::Arena<naga::Expression>,
                 types: &mut naga::UniqueArena<naga::Type>| {
        *b = widen_stores(std::mem::take(b), expressions, types, widened);
    };
    for (mut statement, span) in block.span_into_iter() {
        match &mut statement {
            St::Block(b) => inner(b, expressions, types),
            St::If { accept, reject, .. } => {
                inner(accept, expressions, types);
                inner(reject, expressions, types);
            }
            St::Switch { cases, .. } => {
                for c in cases {
                    inner(&mut c.body, expressions, types);
                }
            }
            St::Loop {
                body, continuing, ..
            } => {
                inner(body, expressions, types);
                inner(continuing, expressions, types);
            }
            St::ImageStore { image, value, .. } => {
                let w = match expressions[*image] {
                    E::GlobalVariable(g) => widened.iter().find(|w| w.0 == g).map(|w| w.1),
                    _ => None,
                };
                if let Some(w) = w {
                    let literal = |v: f32| match w.scalar.kind {
                        naga::ScalarKind::Uint => naga::Literal::U32(v as u32),
                        naga::ScalarKind::Sint => naga::Literal::I32(v as i32),
                        _ => naga::Literal::F32(v),
                    };
                    // Literals are in scope without an `Emit`: they go first, then the
                    // expressions emitted as one range.
                    let mut lit = |v: f32| expressions.append(E::Literal(literal(v)), span);
                    let (zero, one) = (lit(0.0), lit(1.0));
                    let range = w.range.map(|(lo, hi)| (lit(lo), hi.map(&mut lit)));
                    let first = expressions.len();
                    let ty = types.insert(
                        naga::Type {
                            name: None,
                            inner: naga::TypeInner::Vector {
                                size: naga::VectorSize::Quad,
                                scalar: w.scalar,
                            },
                        },
                        naga::Span::UNDEFINED,
                    );
                    let mut wide = *value;
                    if let Some(n) = w.channels {
                        let mut components: Vec<_> = (0..u32::from(n))
                            .map(|index| {
                                expressions.append(E::AccessIndex { base: wide, index }, span)
                            })
                            .collect();
                        components.resize(3, zero);
                        components.push(one);
                        wide = expressions.append(E::Compose { ty, components }, span);
                    }
                    if let Some((lo, hi)) = range {
                        let mut splat = |value| {
                            let size = naga::VectorSize::Quad;
                            expressions.append(E::Splat { size, value }, span)
                        };
                        let (lo, hi) = (splat(lo), hi.map(&mut splat));
                        let fun = match hi {
                            Some(_) => naga::MathFunction::Clamp,
                            None => naga::MathFunction::Max,
                        };
                        wide = expressions.append(
                            E::Math {
                                fun,
                                arg: wide,
                                arg1: Some(lo),
                                arg2: hi,
                                arg3: None,
                            },
                            span,
                        );
                    }
                    if expressions.len() > first {
                        let range = expressions.range_from(first);
                        out.push(St::Emit(range), span);
                        *value = wide;
                    }
                }
            }
            _ => {}
        }
        out.push(statement, span);
    }
    out
}

/// The texture dimension a view of `view` needs.
fn texture_dimension(view: wgpu::TextureViewDimension) -> wgpu::TextureDimension {
    match view {
        wgpu::TextureViewDimension::D1 => wgpu::TextureDimension::D1,
        wgpu::TextureViewDimension::D3 => wgpu::TextureDimension::D3,
        _ => wgpu::TextureDimension::D2,
    }
}

/// The wgpu format of an image's internal format: a storage format of the same channels.
/// `R11F_G11F_B10F` is itself where the device stores into it (`packed_float`): no alpha, so it
/// reads 1 there, as Iris 1.11.7 read it with a probe pack of ours (#10490); elsewhere an
/// `Rgba16Float` stand-in, which keeps an alpha.
pub(super) fn format(f: Format, packed_float: bool) -> wgpu::TextureFormat {
    use wgpu::TextureFormat as T;
    let pick = |r, rg, rgba| match f.channels {
        1 => r,
        2 => rg,
        _ => rgba,
    };
    match (f.kind, f.bits) {
        (Kind::Unorm, 10) => T::Rgb10a2Unorm,
        (Kind::Uint, 10) => T::Rgb10a2Uint,
        (Kind::Snorm, 16) => pick(T::R16Snorm, T::Rg16Snorm, T::Rgba16Snorm),
        (Kind::Int, 8) => pick(T::R8Sint, T::Rg8Sint, T::Rgba8Sint),
        (Kind::Int, 16) => pick(T::R16Sint, T::Rg16Sint, T::Rgba16Sint),
        (Kind::Int, _) => pick(T::R32Sint, T::Rg32Sint, T::Rgba32Sint),
        (Kind::Uint, 8) => pick(T::R8Uint, T::Rg8Uint, T::Rgba8Uint),
        (Kind::Uint, 16) => pick(T::R16Uint, T::Rg16Uint, T::Rgba16Uint),
        (Kind::Uint, _) => pick(T::R32Uint, T::Rg32Uint, T::Rgba32Uint),
        (Kind::Float, 32) => pick(T::R32Float, T::Rg32Float, T::Rgba32Float),
        (Kind::Float, 11) if packed_float => T::Rg11b10Ufloat,
        (Kind::Float, _) => pick(T::R16Float, T::Rg16Float, T::Rgba16Float),
        (Kind::Snorm, _) => pick(T::R8Snorm, T::Rg8Snorm, T::Rgba8Snorm),
        (Kind::Unorm, 16) => pick(T::R16Unorm, T::Rg16Unorm, T::Rgba16Unorm),
        (Kind::Unorm, _) => pick(T::R8Unorm, T::Rg8Unorm, T::Rgba8Unorm),
    }
}

/// Whether this device can make `def`: its format's features, and a size in pixels within the
/// device's limits.
pub(super) fn makeable(
    device: &wgpu::Device,
    def: &CustomImage,
    packed_float: bool,
) -> Result<(), String> {
    let format = format(def.format, packed_float);
    let missing = format.required_features() - device.features();
    if !missing.is_empty() {
        return Err(format!(
            "{format:?} needs {missing:?}, which this GPU lacks"
        ));
    }
    let limits = device.limits();
    let largest = if def.is_1d() {
        limits.max_texture_dimension_1d
    } else if def.is_3d() {
        limits.max_texture_dimension_3d
    } else {
        limits.max_texture_dimension_2d
    };
    let [width, height, depth] = def.extent([1, 1]);
    let relative = matches!(def.size, horizon_shaders::ImageSize::Relative(_));
    if !relative && width.max(height).max(depth) > largest {
        return Err(format!(
            "{width}x{height}x{depth} is larger than this GPU's {largest}"
        ));
    }
    Ok(())
}

/// Makes the images that are missing or sized for another render size; whether it made any.
pub(super) fn resize_images(
    device: &wgpu::Device,
    images: &mut [PackImage],
    size: [u32; 2],
) -> bool {
    let largest = device.limits().max_texture_dimension_2d;
    let mut made = false;
    for image in images {
        let relative = matches!(image.def.size, horizon_shaders::ImageSize::Relative(_));
        if image.view.is_some() && (!relative || image.made_for == size) {
            continue;
        }
        made = true;
        // A relative image larger than the render size stops at the device's limit.
        let [width, height, depth] = image.def.extent(size);
        let [width, height] = [width.min(largest), height.min(largest)];
        let make = |usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(&image.def.name),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: depth,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: texture_dimension(image.dimension),
                    format: image.format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        // Atomics on a texture need it made for them (Metal ignores them otherwise).
        let atomic = device.features().contains(wgpu::Features::TEXTURE_ATOMIC)
            && matches!(
                image.format,
                wgpu::TextureFormat::R32Sint | wgpu::TextureFormat::R32Uint
            );
        let atomic = if atomic {
            wgpu::TextureUsages::STORAGE_ATOMIC
        } else {
            wgpu::TextureUsages::empty()
        };
        image.view = Some(make(
            wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC
                | atomic,
        ));
        image.copy = image
            .copied
            .then(|| make(wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::STORAGE_BINDING));
        image.before = image
            .snapshotted
            .then(|| make(wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::STORAGE_BINDING));
        image.made_for = size;
    }
    made
}

/// The most images a pack may have.
pub(super) const IMAGES: usize = 64;

/// Copies `image` as it is before a compute pass whose loads read it so.
pub(super) fn snapshot(encoder: &mut wgpu::CommandEncoder, image: &PackImage) {
    if let (Some(view), Some(before)) = (&image.view, &image.before) {
        encoder.copy_texture_to_texture(
            view.texture().as_image_copy(),
            before.texture().as_image_copy(),
            view.texture().size(),
        );
    }
}

/// Copies the images of `mask` (bit n: image n), which a pass samples and stores, for that pass
/// to sample.
pub(super) fn copy(encoder: &mut wgpu::CommandEncoder, images: &[PackImage], mask: u64) {
    for (_, image) in images
        .iter()
        .enumerate()
        .filter(|(k, _)| mask & 1 << k != 0)
    {
        if let (Some(view), Some(copy)) = (&image.view, &image.copy) {
            encoder.copy_texture_to_texture(
                view.texture().as_image_copy(),
                copy.texture().as_image_copy(),
                view.texture().size(),
            );
        }
    }
}

/// Clears the images the pack clears before each frame: with `clear_texture` where the device
/// has it (`clear`), else by copying from `zeros`, a buffer of zeros made large enough.
pub(super) fn clear(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    images: &[PackImage],
    clear: bool,
    zeros: &mut Option<wgpu::Buffer>,
) {
    for view in images
        .iter()
        .filter(|i| i.def.clear)
        .filter_map(|i| i.view.as_ref())
    {
        let texture = view.texture();
        if clear {
            encoder.clear_texture(texture, &Default::default());
            continue;
        }
        let size = texture.size();
        let texel = u64::from(texture.format().block_copy_size(None).unwrap_or(4));
        let row = (u64::from(size.width) * texel).next_multiple_of(256);
        // Zeros for a slice of rows at a time, the whole image taken in slices, so a large
        // volume doesn't need a buffer of its size.
        let rows = (ZEROS / row).clamp(1, u64::from(size.height)) as u32;
        let bytes = row * u64::from(rows);
        if zeros.as_ref().is_none_or(|z| z.size() < bytes) {
            // New buffers start zeroed.
            *zeros = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shader pack image zeros"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        }
        let Some(buffer) = zeros.as_ref() else {
            continue;
        };
        for z in 0..size.depth_or_array_layers {
            for y in (0..size.height).step_by(rows as usize) {
                let height = rows.min(size.height - y);
                encoder.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(row as u32),
                            rows_per_image: Some(height),
                        },
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y, z },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: size.width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
            }
        }
    }
}

/// The most bytes of zeros kept to clear images without `CLEAR_TEXTURE`.
const ZEROS: u64 = 4 << 20;

/// What an image no `image.*` gives binds. Iris 1.11.7 runs the program, and its loads read
/// zero, also after it stored into the image (observed with a probe pack). Here it is one texel
/// of the declared format, cleared before every pass that binds it; loads and stores outside
/// that texel are out of bounds (with the same probe pack on Vulkan, the loads read zero).
pub(super) struct StandIn {
    format: wgpu::TextureFormat,
    dimension: wgpu::TextureViewDimension,
    pub(super) view: wgpu::TextureView,
    zeros: wgpu::Buffer,
}

impl StandIn {
    /// Zero again, before a pass that binds it.
    pub(super) fn clear(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &self.zeros,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
            },
            self.view.texture().as_image_copy(),
            self.view.texture().size(),
        );
    }
}

pub(super) fn reset_stand_ins(device: &wgpu::Device, images: &mut Vec<StandIn>) {
    let wanted = images.iter().map(|i| (i.format, i.dimension)).collect();
    *images = stand_ins(device, wanted);
}

/// One stand-in for each format and dimension of `wanted`.
pub(super) fn stand_ins(
    device: &wgpu::Device,
    wanted: Vec<(wgpu::TextureFormat, wgpu::TextureViewDimension)>,
) -> Vec<StandIn> {
    wanted
        .into_iter()
        .map(|(format, dimension)| {
            let atomic = device.features().contains(wgpu::Features::TEXTURE_ATOMIC)
                && matches!(
                    format,
                    wgpu::TextureFormat::R32Sint | wgpu::TextureFormat::R32Uint
                );
            let view = device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("Shader pack image stand-in"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: texture_dimension(dimension),
                    format,
                    usage: wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::COPY_SRC
                        | if atomic {
                            wgpu::TextureUsages::STORAGE_ATOMIC
                        } else {
                            wgpu::TextureUsages::empty()
                        },
                    view_formats: &[],
                })
                .create_view(&Default::default());
            // New buffers start zeroed; one texel is at most 16 bytes.
            let zeros = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Shader pack image stand-in zeros"),
                size: 16,
                usage: wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            StandIn {
                format,
                dimension,
                view,
                zeros,
            }
        })
        .collect()
}

/// The stand-in of this format and dimension.
pub(super) fn stand_in(
    stand_ins: &[StandIn],
    format: wgpu::TextureFormat,
    dimension: wgpu::TextureViewDimension,
) -> Option<&StandIn> {
    stand_ins
        .iter()
        .find(|s| s.format == format && s.dimension == dimension)
}

/// The pack's storage buffers, bound in set 4.
pub(super) struct Buffers {
    pub(super) layout: wgpu::BindGroupLayout,
    objects: BTreeMap<u32, BufferObject>,
    /// The fewest bytes each one's shaders bind it as (`Post::buffer_minimums`).
    minimums: BTreeMap<u32, u64>,
    pub(super) group: Option<wgpu::BindGroup>,
    buffers: Vec<wgpu::Buffer>,
    made_for: [u32; 2],
}

impl Buffers {
    pub(super) fn reset_history(&mut self) {
        self.group = None;
        self.buffers.clear();
        self.made_for = [0; 2];
    }

    /// The layout of `objects`, bound in `stages`; each buffer is at least its entry in
    /// `minimums` long.
    pub(super) fn new(
        device: &wgpu::Device,
        objects: BTreeMap<u32, BufferObject>,
        minimums: BTreeMap<u32, u64>,
        stages: wgpu::ShaderStages,
    ) -> Self {
        let entries: Vec<_> = objects
            .keys()
            .map(|&binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: stages,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shader pack buffers"),
            entries: &entries,
        });
        Self {
            layout,
            objects,
            minimums,
            group: None,
            buffers: Vec::new(),
            made_for: [0; 2],
        }
    }

    /// Makes the buffers (again, when one depends on the render size and it changed): their
    /// contents start zeroed and persist across frames.
    pub(super) fn resize(&mut self, device: &wgpu::Device, size: [u32; 2]) {
        let relative = self
            .objects
            .values()
            .any(|o| matches!(o, BufferObject::PerPixel { .. }));
        if self.group.is_some() && (!relative || self.made_for == size) {
            return;
        }
        let limits = device.limits();
        let largest = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size)
            & !3;
        self.buffers = self
            .objects
            .iter()
            .map(|(binding, o)| {
                let least = self.minimums.get(binding).copied().unwrap_or(0);
                let wanted = buffer_size(o, size, least);
                if wanted > largest {
                    log::warn!(
                        "shader pack: bufferObject.{binding} wants {wanted} bytes, this device \
                         binds at most {largest}"
                    );
                }
                let size = wanted.min(largest);
                // A buffer the pack initializes from a file starts with the file's bytes, the
                // rest of it zero.
                match o.initial_data() {
                    Some(data) => {
                        let mut contents = data.to_vec();
                        contents.resize(size as usize, 0);
                        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("Shader pack buffer"),
                            contents: &contents,
                            usage: wgpu::BufferUsages::STORAGE,
                        })
                    }
                    None => device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Shader pack buffer"),
                        size,
                        usage: wgpu::BufferUsages::STORAGE,
                        mapped_at_creation: false,
                    }),
                }
            })
            .collect();
        let entries: Vec<_> = self
            .objects
            .keys()
            .zip(&self.buffers)
            .map(|(&binding, b)| wgpu::BindGroupEntry {
                binding,
                resource: b.as_entire_binding(),
            })
            .collect();
        self.group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shader pack buffers"),
            layout: &self.layout,
            entries: &entries,
        }));
        self.made_for = size;
    }

    /// Whether the pack declares buffer `binding`.
    pub(super) fn has(&self, binding: u32) -> bool {
        self.objects.contains_key(&binding)
    }
}

/// The bytes a storage buffer is made with for a render size of `render`: what the pack
/// declares, or more when a shader binds it as a larger block (`least`), which wgpu needs where
/// OpenGL binds the smaller buffer; a multiple of 4.
fn buffer_size(object: &BufferObject, render: [u32; 2], least: u64) -> u64 {
    object.size(render).max(least).next_multiple_of(4)
}

/// A debug readback of one of the pack's images (`HORIZON_SHADER_PACK_DUMP=<name>`): every 60
/// frames, how many of its texels are not zero, logged.
pub(super) struct Dump {
    name: String,
    frame: u32,
    pending: Option<(wgpu::Buffer, usize, [u32; 3], u32)>,
}

impl Dump {
    pub(super) fn new() -> Option<Self> {
        let name = std::env::var("HORIZON_SHADER_PACK_DUMP").ok()?;
        Some(Self {
            name,
            frame: 0,
            pending: None,
        })
    }

    /// Reads last frame's copy, once its commands ran.
    pub(super) fn read(&mut self, device: &wgpu::Device) {
        let Some((buffer, texel, [w, h, d], row)) = self.pending.take() else {
            return;
        };
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        let Ok(data) = buffer.slice(..).get_mapped_range() else {
            return;
        };
        let (mut set, mut total) = (0usize, 0usize);
        // Also: the rows (y) holding set texels, and the set texels' values (first 4 bytes) by count.
        let (mut rows, mut values, mut other) = ((u32::MAX, 0), Vec::<(u32, usize)>::new(), 0usize);
        for z in 0..d as usize {
            for y in 0..h as usize {
                let start = (z * h as usize + y) * row as usize;
                for x in 0..w as usize {
                    let at = start + x * texel;
                    total += 1;
                    if data[at..at + texel].iter().any(|&b| b != 0) {
                        set += 1;
                        rows = (rows.0.min(y as u32), rows.1.max(y as u32));
                        let mut v = [0u8; 4];
                        let n = texel.min(4);
                        v[..n].copy_from_slice(&data[at..at + n]);
                        let v = u32::from_le_bytes(v);
                        match values.iter().position(|e| e.0 == v) {
                            Some(i) => values[i].1 += 1,
                            None if values.len() < 40 => values.push((v, 1)),
                            None => other += 1,
                        }
                    }
                }
            }
        }
        log::info!(
            "shader pack image {}: {set} of {total} texels set, rows {}..={}, values {values:x?}, other {other}",
            self.name,
            rows.0,
            rows.1
        );
    }

    /// Copies the image for `read` every 60 frames.
    pub(super) fn copy(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        images: &[PackImage],
    ) {
        self.frame += 1;
        if !self.frame.is_multiple_of(60) {
            return;
        }
        let Some(image) = images.iter().find(|i| i.def.name == self.name) else {
            return;
        };
        let Some(view) = &image.view else { return };
        let size = view.texture().size();
        let texel = image.format.block_copy_size(None).unwrap_or(4) as usize;
        let row = (size.width as usize * texel).next_multiple_of(256) as u32;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shader pack image dump"),
            size: u64::from(row) * u64::from(size.height) * u64::from(size.depth_or_array_layers),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            view.texture().as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(size.height),
                },
            },
            size,
        );
        let dims = [size.width, size.height, size.depth_or_array_layers];
        self.pending = Some((buffer, texel, dims, row));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Images made in a twin of their format (Metal reads and writes neither `rg32ui` nor `r16`):
    /// the module declares the twin and still validates; a store into the `rg32ui` image (here
    /// inside an `if`) writes x and y with z = 0, w = 1, one into the `r16` image clamps to
    /// [0, 1], and an image made in its own format is left alone.
    #[test]
    fn a_widened_image_stores_what_its_format_keeps() {
        use wgpu::TextureFormat as T;
        let glsl = "#version 450\nlayout(local_size_x = 1) in;\n\
                    layout(set = 0, binding = 1, rg32ui) uniform uimage2D wide;\n\
                    layout(set = 0, binding = 2, rg32ui) uniform uimage2D kept;\n\
                    layout(set = 0, binding = 3, r16) uniform image2D unorm;\n\
                    void main() {\n    uvec4 v = imageLoad(wide, ivec2(0)) + imageLoad(kept, ivec2(0));\n    \
                    if (v.x == 0u) { imageStore(wide, ivec2(1), uvec4(7u, 8u, 9u, 10u)); }\n    \
                    imageStore(kept, ivec2(1), v);\n    \
                    imageStore(unorm, ivec2(1), imageLoad(unorm, ivec2(0)) * 2.0);\n}\n";
        let options = naga::front::glsl::Options::from(naga::ShaderStage::Compute);
        let mut module = naga::front::glsl::Frontend::default()
            .parse(&options, glsl)
            .expect("parses");
        widen_images(
            &mut module,
            &[
                (1, T::Rg32Uint, T::Rgba32Uint),
                (2, T::Rg32Uint, T::Rg32Uint),
                (3, T::R16Unorm, T::R32Float),
            ],
        );
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("validates");
        let format = |binding: u32| {
            let (_, g) = (module.global_variables.iter())
                .find(|(_, g)| g.binding.as_ref().is_some_and(|b| b.binding == binding))
                .expect("declared");
            match module.types[g.ty].inner {
                naga::TypeInner::Image {
                    class: naga::ImageClass::Storage { format, .. },
                    ..
                } => format,
                _ => panic!("not a storage image"),
            }
        };
        use naga::StorageFormat as S;
        assert_eq!(
            [1, 2, 3].map(format),
            [S::Rgba32Uint, S::Rg32Uint, S::R32Float]
        );
        // What each store's value is: as written, filled out (its last two components'
        // literals), or clamped.
        // The GLSL front puts `main` in a function the entry point calls.
        let (_, f) = module.functions.iter().next().expect("main");
        let mut stored = Vec::new();
        let mut blocks = vec![&f.body];
        while let Some(block) = blocks.pop() {
            for s in block.iter() {
                match s {
                    naga::Statement::If { accept, reject, .. } => blocks.extend([accept, reject]),
                    naga::Statement::Block(b) => blocks.push(b),
                    naga::Statement::ImageStore { value, .. } => {
                        stored.push(match &f.expressions[*value] {
                            naga::Expression::Compose { components, .. } => {
                                let literals =
                                    components[2..].iter().map(|&c| match f.expressions[c] {
                                        naga::Expression::Literal(naga::Literal::U32(u)) => u,
                                        _ => u32::MAX,
                                    });
                                format!("filled {:?}", literals.collect::<Vec<_>>())
                            }
                            naga::Expression::Math {
                                fun: naga::MathFunction::Clamp,
                                ..
                            } => "clamped".to_owned(),
                            _ => "as written".to_owned(),
                        })
                    }
                    _ => {}
                }
            }
        }
        stored.sort();
        assert_eq!(stored, ["as written", "clamped", "filled [0, 1]"]);
    }

    /// A buffer the pack declares smaller than a shader's block grows to the block (40 bytes
    /// bound as a 48-byte block failed wgpu's validation, #7178); a larger one keeps its size.
    #[test]
    fn a_buffer_is_at_least_its_shaders_block() {
        assert_eq!(buffer_size(&BufferObject::Bytes(40), [1, 1], 48), 48);
        assert_eq!(
            buffer_size(&BufferObject::Bytes(40980), [1, 1], 40992),
            40992
        );
        assert_eq!(buffer_size(&BufferObject::Bytes(100), [1, 1], 48), 100);
        assert_eq!(buffer_size(&BufferObject::Bytes(41), [1, 1], 0), 44);
        let per_pixel = BufferObject::PerPixel {
            bytes: 4,
            scale: [0.5, 0.5],
        };
        assert_eq!(buffer_size(&per_pixel, [10, 10], 64), 100);
        assert_eq!(buffer_size(&per_pixel, [4, 4], 64), 64);
    }
}
