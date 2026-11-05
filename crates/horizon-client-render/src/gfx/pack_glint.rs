//! The enchantment glint under a shader pack (`gbuffers_armor_glint` and its fallbacks): the
//! glint quads of entities and held items drawn again over themselves, textured with the glint
//! (`enchanted_glint_item`, `enchanted_glint_armor`) at the glint's UV and texture matrix, as
//! vanilla's glint render types draw them. Also how a run of quads splits by kind (glint,
//! translucent), for the programs that draw each kind.

use glam::{Mat4, Vec3};

use super::Vertex;

/// What draws a quad's glint: none, the item glint (`GLINT_TEXTURING`, its UV already in
/// `Vertex::glint_uv`) or the armour glint (`ARMOR_ENTITY_GLINT_TEXTURING`: the model UV in
/// `glint_uv`, scrolled by the texture matrix).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glint {
    None,
    Item,
    Armor,
}

/// The glint of a quad with overlay code `overlay` (entity.wgsl: 7, 8 and 33 the item foil,
/// 36 the armour's).
pub fn glint(overlay: u32) -> Glint {
    match overlay & 0xffff {
        7 | 8 | super::TRANSLUCENT_GLINT_ITEM_OVERLAY => Glint::Item,
        36 => Glint::Armor,
        _ => Glint::None,
    }
}

/// `count` quads from quad `first` of `vertices`, cut where `kind` of a quad's overlay changes:
/// (first quad, quads, kind) of each part.
pub fn split<K: PartialEq + Copy>(
    vertices: &[Vertex],
    first: u32,
    count: u32,
    kind: impl Fn(u32) -> K,
) -> Vec<(u32, u32, K)> {
    let mut out: Vec<(u32, u32, K)> = Vec::new();
    for q in first..first + count {
        let k = kind(vertices[q as usize * 4].layer);
        match out.last_mut() {
            Some(last) if last.2 == k => last.1 += 1,
            _ => out.push((q, 1, k)),
        }
    }
    out
}

/// The glint's vertex inputs over `base` (entities' or the hand's, which read their vertices at
/// set 1 as `hz_entity_vertices`): the glint UV as `gl_MultiTexCoord0`, and with `armor`, the
/// armour glint's scroll as `gl_TextureMatrix[0]` from set 1, binding 1 (`HzGlint`). No alpha
/// test, as vanilla's glint has none.
pub fn inputs(
    mut base: horizon_shaders::GbuffersInputs,
    armor: bool,
) -> horizon_shaders::GbuffersInputs {
    let words = std::mem::size_of::<Vertex>() / 4;
    let at = std::mem::offset_of!(Vertex, glint_uv) / 4;
    if armor {
        base.declarations
            .push_str("layout(set = 1, binding = 1) uniform HzGlint { mat4 hz_glint_armor; };\n");
    }
    base.body.push_str(&format!(
        "    uint hz_gw = uint(gl_VertexIndex) * {words}u + {at}u;\n    \
         hz_MultiTexCoord0 = vec4(uintBitsToFloat(uvec2(hz_entity_vertices[hz_gw], hz_entity_vertices[hz_gw + 1u])), 0.0, 1.0);\n"
    ));
    if armor {
        base.body.push_str(
            "    if ((hz_v_overlay & 65535u) == 36u) hz_TextureMatrix[0] = hz_glint_armor;\n",
        );
    }
    for (name, value) in &mut base.attributes {
        if name == "vaUV0" {
            *value = "hz_MultiTexCoord0.xy".to_owned();
        }
    }
    base.alpha_test = None;
    base
}

/// `ARMOR_ENTITY_GLINT_TEXTURING` (`TextureTransform.setupGlintTexturing(0.16)`) at the glint
/// clock `ticks` (`Frame::glint_ticks`): scale, a 10 degree turn, then the two scrolls.
pub fn armor_matrix(ticks: f32) -> Mat4 {
    let offset0 = (ticks % 110000.0) / 110000.0;
    let offset1 = (ticks % 30000.0) / 30000.0;
    Mat4::from_translation(Vec3::new(-offset0, offset1, 0.0))
        * Mat4::from_rotation_z(10f32.to_radians())
        * Mat4::from_scale(Vec3::new(0.16, 0.16, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(overlay: u32) -> [Vertex; 4] {
        [Vertex {
            pos: [0.0; 3],
            uv: [0.0; 2],
            layer: overlay,
            color: 0,
            light: 0,
            normal: [0; 4],
            glint_uv: [0.0; 2],
        }; 4]
    }

    #[test]
    fn runs_split_where_the_kind_changes() {
        let vertices: Vec<Vertex> = [0, 7, 7, 36, 0, 33 | 3 << 16]
            .iter()
            .flat_map(|&o| quad(o))
            .collect();
        assert_eq!(
            split(&vertices, 0, 6, glint),
            [
                (0, 1, Glint::None),
                (1, 2, Glint::Item),
                (3, 1, Glint::Armor),
                (4, 1, Glint::None),
                (5, 1, Glint::Item),
            ]
        );
    }

    /// The model UV's glint place, as vanilla's texture matrix gives it.
    #[test]
    fn armor_glint_scrolls_the_scaled_and_turned_uv() {
        let at = |ticks: f32, uv: [f32; 2]| {
            let p = armor_matrix(ticks) * glam::Vec4::new(uv[0], uv[1], 0.0, 1.0);
            [p.x, p.y]
        };
        let close =
            |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5;
        assert!(close(at(0.0, [1.0, 0.0]), [0.157_569, 0.027_784]));
        assert!(close(at(55000.0, [0.0, 0.0]), [-0.5, 25000.0 / 30000.0]));
    }
}
