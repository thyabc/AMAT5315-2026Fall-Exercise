unsafe extern "C" {
    fn enzyme_cube(
        x: f64,
        out: *mut f64,
    );
}

fn main() {
    let mut out = [0.0_f64; 3];

    unsafe {
        enzyme_cube(
            2.0,
            out.as_mut_ptr(),
        );
    }

    println!("Enzyme smoke test");
    println!("primal  = {}", out[0]);
    println!("forward = {}", out[1]);
    println!("reverse = {}", out[2]);

    assert!((out[0] - 8.0).abs() < 1e-12);
    assert!((out[1] - 12.0).abs() < 1e-12);
    assert!((out[2] - 12.0).abs() < 1e-12);

    println!("PASS: Enzyme is working.");
}
