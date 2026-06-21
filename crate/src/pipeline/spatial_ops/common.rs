use crate::pipeline::PlanarImage;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(super) const fn magic_u16_divisor(divisor: u32) -> u32 {
    ((1u32 << 16) + divisor - 1) / divisor
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
macro_rules! div_const_u16x8_simd {
    ($values:expr, $divisor:literal) => {{
        const MAGIC: u32 = super::common::magic_u16_divisor($divisor);
        super::common::div_u16x8_by_magic_simd($values, MAGIC)
    }};
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
pub(super) unsafe fn div_u16x8_by_magic_simd(
    values: core::arch::wasm32::v128,
    magic: u32,
) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;

    // For bounded unsigned values we can replace division by a constant with a
    // fixed-point multiply: floor(x / d) == (x * ceil(2^16 / d)) >> 16.
    // The caller must only use this for value ranges where that identity has
    // been checked. For the current box blur case, x is 0..=2295 and d is 9.
    let magic = u32x4_splat(magic);

    // wasm SIMD has no u16x8 multiply that gives the high half of the product,
    // so widen to u32x4, multiply there, shift, then narrow back to u16x8.
    let lo = u32x4_extend_low_u16x8(values);
    let hi = u32x4_extend_high_u16x8(values);
    let div_lo = u32x4_shr(u32x4_mul(lo, magic), 16);
    let div_hi = u32x4_shr(u32x4_mul(hi, magic), 16);

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
