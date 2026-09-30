use std::{
    env,
    path::PathBuf,
    process::Command,
};

fn main() {
    println!("cargo:rerun-if-changed=src/kernel.rs");

    let out =
        PathBuf::from(env::var_os("OUT_DIR").unwrap());

    let rustc =
        env::var_os("RUSTC").unwrap();

    let output =
        out.join("libenzyme_kernel.a");

    let status =
        Command::new(rustc)
            .args([
                "--edition=2024",
                "--crate-name",
                "enzyme_kernel",
                "--crate-type",
                "staticlib",
                "-C",
                "opt-level=3",
                "-C",
                "lto=fat",
                "-C",
                "panic=abort",
                "-Zautodiff=Enable",
                "src/kernel.rs",
                "-o",
            ])
            .arg(&output)
            .status()
            .expect("failed to compile Enzyme kernel");

    assert!(
        status.success(),
        "Enzyme kernel compilation failed"
    );

    println!(
        "cargo:rustc-link-search=native={}",
        out.display()
    );

    println!(
        "cargo:rustc-link-lib=static=enzyme_kernel"
    );
}
