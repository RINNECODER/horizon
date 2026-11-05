//! A buffer in a format the device can't both read and write (Metal: R11F_G11F_B10F), read and
//! written as `colorimgN` by a compute program, is made in a twin format and still runs. The
//! programs are original.
use super::destination_tests::Fixture;
use super::*;
use std::io::{Cursor, Write};

fn pack() -> Post {
    let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
    let mut add = |name: &str, source: &str| {
        zip.start_file(
            format!("shaders/{name}"),
            zip::write::SimpleFileOptions::default(),
        )
        .unwrap();
        zip.write_all(source.as_bytes()).unwrap();
    };
    add(
        "composite.csh",
        "#version 430\nlayout(local_size_x=1,local_size_y=1) in;\nconst ivec3 workGroups=ivec3(8,8,1);\n\
         layout(r11f_g11f_b10f) uniform image2D colorimg1;\n\
         void main() { ivec2 p=ivec2(gl_GlobalInvocationID.xy);\n\
         imageStore(colorimg1, p, imageLoad(colorimg1, p) + vec4(0.25,0.5,0,0)); }\n",
    );
    add(
        "final.vsh",
        "#version 120\nvarying vec2 uv;\nvoid main() { uv=gl_MultiTexCoord0.xy; gl_Position=ftransform(); }\n",
    );
    add(
        "final.fsh",
        "#version 330 compatibility\n/*\nconst int colortex1Format = R11F_G11F_B10F;\n*/\n\
         const vec4 colortex1ClearColor = vec4(0.0);\nuniform sampler2D colortex1;\nin vec2 uv;\n\
         void main() { gl_FragColor = vec4(texelFetch(colortex1, ivec2(gl_FragCoord.xy), 0).rgb, 1); }\n",
    );
    let pack = horizon_shaders::Pack::from_zip_bytes(
        "original-read-write-buffer",
        zip.finish().unwrap().into_inner(),
    )
    .unwrap();
    let mut macros = horizon_shaders::Macros::standard();
    let (properties, _) = horizon_shaders::read_properties(&pack, &mut macros);
    horizon_shaders::load_post(&pack, "", &properties, &macros, &Default::default())
}

#[test]
fn a_buffer_read_and_written_as_an_image_runs_in_any_format() {
    // The game asks for these (gfx.rs): with them Metal keeps the buffer R11F_G11F_B10F.
    let Some(gpu) = Fixture::with_features(
        wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            | wgpu::Features::RG11B10UFLOAT_RENDERABLE,
    ) else {
        return;
    };
    let post = pack();
    assert!(post.notes.is_empty(), "{:?}", post.notes);
    let mut post = gpu.build(post);
    assert!(post.notes.is_empty(), "{:?}", post.notes);
    let pixels = gpu.render(&mut post, [8, 8]);
    let want = [64, 128, 0, 255];
    for (k, got) in pixels.iter().enumerate() {
        assert!(
            got.iter().zip(want).all(|(g, w)| g.abs_diff(w) <= 1),
            "pixel {k}: {got:?}, expected {want:?}"
        );
    }
}
