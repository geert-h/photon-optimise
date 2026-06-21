use crate::pipeline::PlanarImage;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
pub(super) unsafe fn div9_u16x8_simd(
    values: core::arch::wasm32::v128,
) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;

    // Each lane contains a 3x3 box-blur sum, so the valid range is 0..=2295.
    // For this limited unsigned range we can replace integer division by 9 with
    // a fixed-point multiply: floor(x / 9) == (x * ceil(2^16 / 9)) >> 16.
    // ceil(65536 / 9) is 7282, and it is exact for all values this function sees.
    let magic = u32x4_splat(7282);

    // wasm SIMD has no u16x8 multiply that gives the high half of the product,
    // so widen to u32x4, multiply there, shift, then narrow back to u16x8.
    let lo = u32x4_extend_low_u16x8(values);
    let hi = u32x4_extend_high_u16x8(values);
    let div_lo = u32x4_shr(i32x4_mul(lo, magic), 16);
    let div_hi = u32x4_shr(i32x4_mul(hi, magic), 16);

    u16x8_narrow_i32x4(div_lo, div_hi)
}

/// We have added this function to stay compatible with the library
pub(super) fn restore_alpha_if_filter_zeroed_it(
    src: &PlanarImage,
    dst: &mut PlanarImage,
    width: usize,
    height: usize,
) {
    if !dst.a.iter().all(|alpha| *alpha == 0) {
        return;
    }

    for y in 0..height.saturating_sub(1) {
        let row = y * width;

        for x in 0..width.saturating_sub(1) {
            dst.a[row + x] = src.a[row + x];
        }
    }
}
