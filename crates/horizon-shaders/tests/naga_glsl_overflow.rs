//! GLSL float arithmetic overflows to infinity (IEEE) where the program runs, but a Naga literal
//! can't hold infinity: naga's GLSL frontend leaves such an expression to run instead of failing
//! the shader. A global `const` must fold, so it still fails there, at the call.

fn parse(globals: &str, body: &str) -> Result<naga::Module, naga::front::glsl::ParseErrors> {
    let source = format!(
        "#version 450\nlayout(location = 0) out vec4 color;\n{globals}\n\
         void main() {{\n{body}\n}}\n"
    );
    naga::front::glsl::Frontend::default().parse(
        &naga::front::glsl::Options::from(naga::ShaderStage::Fragment),
        &source,
    )
}

fn validates(globals: &str, body: &str) {
    let module = parse(globals, body).unwrap_or_else(|e| panic!("{body}\n{e:?}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{body}\n{e:?}"));
}

#[test]
fn overflowing_builtin_in_a_function() {
    validates(
        "",
        "float x = exp2(200.0); color = vec4(x, exp(100.0), 0.0, 1.0);",
    );
}

#[test]
fn overflowing_arithmetic_in_a_const_local() {
    validates("", "const float x = 1e30 * 1e30; color = vec4(x);");
}

#[test]
fn overflowing_global_initializer() {
    validates("float x = pow(10.0, 50.0);", "color = vec4(x);");
}

#[test]
fn overflowing_global_const_is_blamed_on_its_call() {
    let globals = "const float x = exp2(200.0);";
    let errors = parse(globals, "color = vec4(x);").unwrap_err().errors;
    let at = errors[0].meta.to_range().unwrap();
    let source_start = "#version 450\nlayout(location = 0) out vec4 color;\n".len();
    assert_eq!(
        &globals[at.start - source_start..at.end - source_start],
        "exp2(200.0)"
    );
}
