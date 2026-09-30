#![no_std]
#![feature(autodiff)]

use core::autodiff::{
    autodiff_forward,
    autodiff_reverse,
};

unsafe extern "C" {
    fn abort() -> !;
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe { abort() }
}

#[autodiff_forward(
    cube_forward,
    Dual,
    Dual
)]
#[autodiff_reverse(
    cube_reverse,
    Active,
    Active
)]
fn cube(x: f64) -> f64 {
    x * x * x
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_cube(
    x: f64,
    out: *mut f64,
) {
    let (y, dy) = cube_forward(x, 1.0);

    let (_, adj) = cube_reverse(x, 1.0);

    unsafe {
        *out = y;
        *out.add(1) = dy;
        *out.add(2) = adj;
    }
}
// ==========================================================
// Acoustic timestep
//
// state = [u^(n-1), u^n]
// out   = [u^n, u^(n+1)]
//
// c is active.
// sigma and source are fixed.
// ==========================================================

#[autodiff_forward(
    timestep_forward,
    Dual,       // state
    Dual,       // c
    Const,      // sigma
    Const,      // source
    Const,      // nx
    Const,      // nz
    Const,      // dx
    Const,      // dt
    Dual        // out
)]
#[autodiff_reverse(
    timestep_reverse,
    Duplicated, // state
    Duplicated, // c
    Const,      // sigma
    Const,      // source
    Const,      // nx
    Const,      // nz
    Const,      // dx
    Const,      // dt
    Duplicated  // out
)]
fn timestep(
    state: &[f64],
    c: &[f64],
    sigma: &[f64],
    source: &[f64],
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
    out: &mut [f64],
) {
    let ncell = nx * nz;

    let dx2 = dx * dx;
    let dt2 = dt * dt;

    // state:
    // [0 .. ncell)       = u^(n-1)
    // [ncell .. 2*ncell) = u^n
    //
    // output:
    // [0 .. ncell)       = u^n
    // [ncell .. 2*ncell) = u^(n+1)

    let mut i = 0usize;

    while i < ncell {
        // First component of new state is the old current field.
        out[i] = state[ncell + i];

        // New field begins as zero so the outer boundary remains zero.
        out[ncell + i] = 0.0;

        i += 1;
    }

    let mut z = 1usize;

    while z + 1 < nz {
        let mut x = 1usize;

        while x + 1 < nx {
            let i = z * nx + x;

            let prev = state[i];
            let u = state[ncell + i];

            let lap = (
                state[ncell + i + 1]
                    + state[ncell + i - 1]
                    + state[ncell + i + nx]
                    + state[ncell + i - nx]
                    - 4.0 * u
            ) / dx2;

            let s = sigma[i];
            let velocity = c[i];

            let numerator =
                2.0 * u
                - (1.0 - s * dt) * prev
                + dt2
                    * (
                        velocity * velocity * lap
                        + source[i]
                    );

            out[ncell + i] =
                numerator / (1.0 + s * dt);

            x += 1;
        }

        z += 1;
    }
}


// ==========================================================
// C ABI: primal timestep
// ==========================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_timestep_primal(
    state_ptr: *const f64,
    c_ptr: *const f64,
    sigma_ptr: *const f64,
    source_ptr: *const f64,
    nx: usize,
    nz: usize,
    dx: f64,
    dt: f64,
    out_ptr: *mut f64,
) {
    let ncell = nx * nz;

    let state = unsafe {
        core::slice::from_raw_parts(
            state_ptr,
            2 * ncell
        )
    };

    let c = unsafe {
        core::slice::from_raw_parts(
            c_ptr,
            ncell
        )
    };

    let sigma = unsafe {
        core::slice::from_raw_parts(
            sigma_ptr,
            ncell
        )
    };

    let source = unsafe {
        core::slice::from_raw_parts(
            source_ptr,
            ncell
        )
    };

    let out = unsafe {
        core::slice::from_raw_parts_mut(
            out_ptr,
            2 * ncell
        )
    };

    timestep(
        state,
        c,
        sigma,
        source,
        nx,
        nz,
        dx,
        dt,
        out,
    );
}


// ==========================================================
// C ABI: timestep JVP
// ==========================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_timestep_forward(
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
) {
    let ncell = nx * nz;

    let state = unsafe {
        core::slice::from_raw_parts(
            state_ptr,
            2 * ncell
        )
    };

    let dstate = unsafe {
        core::slice::from_raw_parts(
            dstate_ptr,
            2 * ncell
        )
    };

    let c = unsafe {
        core::slice::from_raw_parts(
            c_ptr,
            ncell
        )
    };

    let dc = unsafe {
        core::slice::from_raw_parts(
            dc_ptr,
            ncell
        )
    };

    let sigma = unsafe {
        core::slice::from_raw_parts(
            sigma_ptr,
            ncell
        )
    };

    let source = unsafe {
        core::slice::from_raw_parts(
            source_ptr,
            ncell
        )
    };

    let out = unsafe {
        core::slice::from_raw_parts_mut(
            out_ptr,
            2 * ncell
        )
    };

    let dout = unsafe {
        core::slice::from_raw_parts_mut(
            dout_ptr,
            2 * ncell
        )
    };

    timestep_forward(
        state,
        dstate,

        c,
        dc,

        sigma,
        source,

        nx,
        nz,
        dx,
        dt,

        out,
        dout,
    );
}


// ==========================================================
// C ABI: timestep VJP
// ==========================================================

#[unsafe(no_mangle)]
pub unsafe extern "C" fn enzyme_timestep_reverse(
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
) {
    let ncell = nx * nz;

    let state = unsafe {
        core::slice::from_raw_parts(
            state_ptr,
            2 * ncell
        )
    };

    let astate = unsafe {
        core::slice::from_raw_parts_mut(
            astate_ptr,
            2 * ncell
        )
    };

    let c = unsafe {
        core::slice::from_raw_parts(
            c_ptr,
            ncell
        )
    };

    let ac = unsafe {
        core::slice::from_raw_parts_mut(
            ac_ptr,
            ncell
        )
    };

    let sigma = unsafe {
        core::slice::from_raw_parts(
            sigma_ptr,
            ncell
        )
    };

    let source = unsafe {
        core::slice::from_raw_parts(
            source_ptr,
            ncell
        )
    };

    let out = unsafe {
        core::slice::from_raw_parts_mut(
            out_ptr,
            2 * ncell
        )
    };

    let aout = unsafe {
        core::slice::from_raw_parts_mut(
            aout_ptr,
            2 * ncell
        )
    };

    timestep_reverse(
        state,
        astate,

        c,
        ac,

        sigma,
        source,

        nx,
        nz,
        dx,
        dt,

        out,
        aout,
    );
}
