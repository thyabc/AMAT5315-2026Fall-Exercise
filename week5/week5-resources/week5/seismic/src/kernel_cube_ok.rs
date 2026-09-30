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
