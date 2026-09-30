unsafe extern "C" {
    fn enzyme_timestep_forward(
        state_ptr: *const f64,
        dstate_ptr: *const f64,

        c_ptr: *const f64,
        dc_ptr: *const f64,

        sigma_ptr: *const f64,
        source_ptr: *const f64,

        nx: usize,
        nz: usize,
        dx: f64,
        dt: f64,

        out_ptr: *mut f64,
        dout_ptr: *mut f64,
    );

    fn enzyme_timestep_reverse(
        state_ptr: *const f64,
        astate_ptr: *mut f64,

        c_ptr: *const f64,
        ac_ptr: *mut f64,

        sigma_ptr: *const f64,
        source_ptr: *const f64,

        nx: usize,
        nz: usize,
        dx: f64,
        dt: f64,

        out_ptr: *mut f64,
        aout_ptr: *mut f64,
    );
}


fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}


fn main() {
    let nx = 5usize;
    let nz = 5usize;

    let ncell = nx * nz;

    let dx = 1.0_f64;
    let dt = 0.2_f64;

    let mut state =
        vec![0.0_f64; 2 * ncell];

    let mut dstate =
        vec![0.0_f64; 2 * ncell];

    let mut c =
        vec![0.0_f64; ncell];

    let mut dc =
        vec![0.0_f64; ncell];

    let mut sigma =
        vec![0.0_f64; ncell];

    let mut source =
        vec![0.0_f64; ncell];

    for i in 0..ncell {
        state[i] =
            0.001 * (i as f64 + 1.0);

        state[ncell + i] =
            0.002 * (i as f64 + 1.0);

        dstate[i] =
            0.0003 * (i as f64 + 1.0);

        dstate[ncell + i] =
            -0.0002 * (i as f64 + 1.0);

        c[i] =
            1.5 + 0.001 * i as f64;

        dc[i] =
            0.0001
            * ((i % 7) as f64 - 3.0);

        source[i] =
            0.0005
            * ((i % 5) as f64);

        let x = i % nx;
        let z = i / nx;

        let dist = x
            .min(z)
            .min(nx - 1 - x)
            .min(nz - 1 - z);

        if dist == 0 {
            sigma[i] = 0.4;
        }
    }

    let mut out =
        vec![0.0_f64; 2 * ncell];

    let mut dout =
        vec![0.0_f64; 2 * ncell];

    unsafe {
        enzyme_timestep_forward(
            state.as_ptr(),
            dstate.as_ptr(),

            c.as_ptr(),
            dc.as_ptr(),

            sigma.as_ptr(),
            source.as_ptr(),

            nx,
            nz,
            dx,
            dt,

            out.as_mut_ptr(),
            dout.as_mut_ptr(),
        );
    }

    // Arbitrary output adjoint / receiver-like weights.
    let mut w =
        vec![0.0_f64; 2 * ncell];

    for i in 0..2 * ncell {
        w[i] =
            0.01
            * ((i % 11) as f64 - 5.0);
    }

    let lhs = dot(&dout, &w);

    let mut astate =
        vec![0.0_f64; 2 * ncell];

    let mut ac =
        vec![0.0_f64; ncell];

    let mut reverse_out =
        vec![0.0_f64; 2 * ncell];

    let mut aout = w.clone();

    unsafe {
        enzyme_timestep_reverse(
            state.as_ptr(),
            astate.as_mut_ptr(),

            c.as_ptr(),
            ac.as_mut_ptr(),

            sigma.as_ptr(),
            source.as_ptr(),

            nx,
            nz,
            dx,
            dt,

            reverse_out.as_mut_ptr(),
            aout.as_mut_ptr(),
        );
    }

    let rhs =
        dot(&dstate, &astate)
        + dot(&dc, &ac);

    let denominator =
        lhs.abs()
            .max(rhs.abs())
            .max(1e-30);

    let relative_difference =
        (lhs - rhs).abs()
        / denominator;

    println!(
        "timestep transpose test"
    );

    println!("lhs = {:.16e}", lhs);
    println!("rhs = {:.16e}", rhs);

    println!(
        "relative difference = {:.3e}",
        relative_difference
    );

    assert!(
        relative_difference < 1e-10,
        "timestep JVP/VJP transpose check failed"
    );

    println!(
        "PASS: timestep JVP/VJP agree."
    );
}
