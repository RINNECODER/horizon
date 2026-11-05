//! `shadowcolorimgN` is shadowcolor N as an image in any program (shaders.properties docs,
//! "shadowcolor"): a shadowcomp compute program stores into the buffer it samples, through a
//! copy, and a composite compute program stores into another. The programs are original.
use super::destination_tests::Fixture;
use super::*;
use std::io::{Cursor, Write};

const VERTEX: &str = "#version 120\nvarying vec2 uv;\nvoid main() { uv=gl_MultiTexCoord0.xy; gl_Position=ftransform(); }\n";

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
        "shadowcomp.csh",
        "#version 430\nlayout(local_size_x=1,local_size_y=1) in;\nconst ivec3 workGroups=ivec3(16,16,1);\n\
         uniform sampler2D shadowcolor0;\nlayout(rgba8) writeonly uniform image2D shadowcolorimg0;\n\
         void main() { ivec2 p=ivec2(gl_GlobalInvocationID.xy);\n\
         imageStore(shadowcolorimg0, p, texelFetch(shadowcolor0, p, 0) + vec4(0.25,0,0,0)); }\n",
    );
    add(
        "composite.csh",
        "#version 430\nlayout(local_size_x=1,local_size_y=1) in;\nconst ivec3 workGroups=ivec3(16,16,1);\n\
         layout(rgba8) writeonly uniform image2D shadowcolorimg1;\n\
         void main() { imageStore(shadowcolorimg1, ivec2(gl_GlobalInvocationID.xy), vec4(0,0.5,0,1)); }\n",
    );
    add("final.vsh", VERTEX);
    // Pixel column x shows shadowcolor 0 and 1 in turn.
    add(
        "final.fsh",
        "#version 330 compatibility\nconst int shadowMapResolution=16;\n\
         const vec4 shadowcolor0ClearColor=vec4(0,0,0,1);\n\
         uniform sampler2D shadowcolor0;\nuniform sampler2D shadowcolor1;\nin vec2 uv;\n\
         void main() {\n    vec4 c[2] = vec4[2](texture(shadowcolor0,uv), texture(shadowcolor1,uv));\n\
             gl_FragColor = c[int(gl_FragCoord.x) % 2];\n}\n",
    );
    add("shaders.properties", "shadow.enabled=true\n");
    let pack = horizon_shaders::Pack::from_zip_bytes(
        "original-shadowcolor-images",
        zip.finish().unwrap().into_inner(),
    )
    .unwrap();
    let mut macros = horizon_shaders::Macros::standard();
    let (properties, _) = horizon_shaders::read_properties(&pack, &mut macros);
    horizon_shaders::load_post(&pack, "", &properties, &macros, &Default::default())
}

#[test]
fn shadowcolor_images_are_stored_in_any_program() {
    let Some(gpu) = Fixture::new() else { return };
    let post = pack();
    assert!(post.notes.is_empty(), "{:?}", post.notes);
    let mut post = gpu.build(post);
    // shadowcolor0: its clear colour plus a quarter red, each frame; 1: green.
    let pixels = gpu.render(&mut post, [8, 8]);
    for (x, want) in [[64, 0, 0, 255], [0, 128, 0, 255]].iter().enumerate() {
        let got = pixels[x];
        assert!(
            got.iter().zip(want).all(|(g, w)| g.abs_diff(*w) <= 1),
            "column {x}: {got:?}, expected {want:?}; notes {:?}",
            post.notes
        );
    }
}
