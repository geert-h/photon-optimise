use crate::pipeline::PlanarImage;

use super::common::restore_alpha_if_filter_zeroed_it;

#[target_feature(enable = "simd128")]
pub(super) unsafe fn prewitt_horizontal_simd(src: &PlanarImage, dst: &mut PlanarImage) {
    let width = src.width() as usize;
    let height = src.height() as usize;

    dst.r.fill(0);
    dst.g.fill(0);
    dst.b.fill(0);
    dst.a.fill(0);

    prewitt_horizontal_channel_simd(&src.r, &mut dst.r, width, height);
    prewitt_horizontal_channel_simd(&src.g, &mut dst.g, width, height);
    prewitt_horizontal_channel_simd(&src.b, &mut dst.b, width, height);
    prewitt_horizontal_channel_simd(&src.a, &mut dst.a, width, height);
    restore_alpha_if_filter_zeroed_it(src, dst, width, height);
}

#[target_feature(enable = "simd128")]
unsafe fn prewitt_horizontal_channel_simd(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
) {
    use core::arch::wasm32::*;

    if width < 3 || height < 3 {
        return;
    }

    for y in 1..height - 1 {
        let top = (y - 1) * width;
        let mid = y * width;
        let bot = (y + 1) * width;
        let mut x = 1;

        while x + 16 <= width - 1 {
            let top_left = v128_load(src.as_ptr().add(top + x - 1) as *const v128);
            let top_center = v128_load(src.as_ptr().add(top + x) as *const v128);
            let top_right = v128_load(src.as_ptr().add(top + x + 1) as *const v128);
            let mid_left = v128_load(src.as_ptr().add(mid + x - 1) as *const v128);
            let mid_right = v128_load(src.as_ptr().add(mid + x + 1) as *const v128);
            let bot_left = v128_load(src.as_ptr().add(bot + x - 1) as *const v128);
            let bot_center = v128_load(src.as_ptr().add(bot + x) as *const v128);
            let bot_right = v128_load(src.as_ptr().add(bot + x + 1) as *const v128);

            let value_lo = prewitt_horizontal_lanes(
                u16x8_extend_low_u8x16(top_left),
                u16x8_extend_low_u8x16(top_center),
                u16x8_extend_low_u8x16(top_right),
                u16x8_extend_low_u8x16(mid_left),
                u16x8_extend_low_u8x16(mid_right),
                u16x8_extend_low_u8x16(bot_left),
                u16x8_extend_low_u8x16(bot_center),
                u16x8_extend_low_u8x16(bot_right),
            );
            let value_hi = prewitt_horizontal_lanes(
                u16x8_extend_high_u8x16(top_left),
                u16x8_extend_high_u8x16(top_center),
                u16x8_extend_high_u8x16(top_right),
                u16x8_extend_high_u8x16(mid_left),
                u16x8_extend_high_u8x16(mid_right),
                u16x8_extend_high_u8x16(bot_left),
                u16x8_extend_high_u8x16(bot_center),
                u16x8_extend_high_u8x16(bot_right),
            );
            let out = u8x16_narrow_i16x8(value_lo, value_hi);

            v128_store(dst.as_mut_ptr().add(mid + x) as *mut v128, out);

            x += 16;
        }

        while x < width - 1 {
            let value = 5 * src[top + x - 1] as i32
                - 3 * src[top + x] as i32
                - 3 * src[top + x + 1] as i32
                + 5 * src[mid + x - 1] as i32
                - 3 * src[mid + x + 1] as i32
                + 5 * src[bot + x - 1] as i32
                - 3 * src[bot + x] as i32
                - 3 * src[bot + x + 1] as i32;

            dst[mid + x] = value.clamp(0, 255) as u8;
            x += 1;
        }
    }
}

#[target_feature(enable = "simd128")]
unsafe fn prewitt_horizontal_lanes(
    top_left: core::arch::wasm32::v128,
    top_center: core::arch::wasm32::v128,
    top_right: core::arch::wasm32::v128,
    mid_left: core::arch::wasm32::v128,
    mid_right: core::arch::wasm32::v128,
    bot_left: core::arch::wasm32::v128,
    bot_center: core::arch::wasm32::v128,
    bot_right: core::arch::wasm32::v128,
) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;

    let positive = i16x8_add(
        mul5_i16x8(top_left),
        i16x8_add(mul5_i16x8(mid_left), mul5_i16x8(bot_left)),
    );
    let negative = i16x8_add(
        i16x8_add(mul3_i16x8(top_center), mul3_i16x8(top_right)),
        i16x8_add(
            mul3_i16x8(mid_right),
            i16x8_add(mul3_i16x8(bot_center), mul3_i16x8(bot_right)),
        ),
    );

    i16x8_sub(positive, negative)
}

#[target_feature(enable = "simd128")]
unsafe fn mul3_i16x8(values: core::arch::wasm32::v128) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;

    i16x8_add(i16x8_shl(values, 1), values)
}

#[target_feature(enable = "simd128")]
unsafe fn mul5_i16x8(values: core::arch::wasm32::v128) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;

    i16x8_add(i16x8_shl(values, 2), values)
}
