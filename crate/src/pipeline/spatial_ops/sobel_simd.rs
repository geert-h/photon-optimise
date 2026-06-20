use crate::pipeline::PlanarImage;

use super::common::restore_alpha_if_filter_zeroed_it;

#[target_feature(enable = "simd128")]
pub(super) unsafe fn sobel_horizontal_simd(
    src: &PlanarImage,
    dst: &mut PlanarImage,
    scratch: &mut [u16],
) {
    let width = src.width() as usize;
    let height = src.height() as usize;

    dst.r.fill(0);
    dst.g.fill(0);
    dst.b.fill(0);
    dst.a.fill(0);

    sobel_horizontal_channel_simd(&src.r, &mut dst.r, scratch, width, height);
    sobel_horizontal_channel_simd(&src.g, &mut dst.g, scratch, width, height);
    sobel_horizontal_channel_simd(&src.b, &mut dst.b, scratch, width, height);
    sobel_horizontal_channel_simd(&src.a, &mut dst.a, scratch, width, height);
    restore_alpha_if_filter_zeroed_it(src, dst, width, height);
}

#[target_feature(enable = "simd128")]
unsafe fn sobel_horizontal_channel_simd(
    src: &[u8],
    dst: &mut [u8],
    scratch: &mut [u16],
    width: usize,
    height: usize,
) {
    use core::arch::wasm32::*;

    if width < 3 || height < 3 {
        return;
    }

    for y in 0..height {
        let row = y * width;
        let mut x = 1;

        while x + 16 <= width - 1 {
            let left = v128_load(src.as_ptr().add(row + x - 1) as *const v128);
            let center = v128_load(src.as_ptr().add(row + x) as *const v128);
            let right = v128_load(src.as_ptr().add(row + x + 1) as *const v128);

            let center_lo = u16x8_extend_low_u8x16(center);
            let center_hi = u16x8_extend_high_u8x16(center);
            let sum_lo = u16x8_add(
                u16x8_add(u16x8_extend_low_u8x16(left), u16x8_shl(center_lo, 1)),
                u16x8_extend_low_u8x16(right),
            );
            let sum_hi = u16x8_add(
                u16x8_add(u16x8_extend_high_u8x16(left), u16x8_shl(center_hi, 1)),
                u16x8_extend_high_u8x16(right),
            );

            v128_store(scratch.as_mut_ptr().add(row + x) as *mut v128, sum_lo);
            v128_store(scratch.as_mut_ptr().add(row + x + 8) as *mut v128, sum_hi);

            x += 16;
        }

        while x < width - 1 {
            scratch[row + x] = src[row + x - 1] as u16
                + 2 * src[row + x] as u16
                + src[row + x + 1] as u16;
            x += 1;
        }
    }

    for y in 1..height - 1 {
        let top = (y - 1) * width;
        let mid = y * width;
        let bot = (y + 1) * width;
        let mut x = 1;

        while x + 16 <= width - 1 {
            let top_lo = v128_load(scratch.as_ptr().add(top + x) as *const v128);
            let bot_lo = v128_load(scratch.as_ptr().add(bot + x) as *const v128);
            let top_hi = v128_load(scratch.as_ptr().add(top + x + 8) as *const v128);
            let bot_hi = v128_load(scratch.as_ptr().add(bot + x + 8) as *const v128);

            let diff_lo = i16x8_sub(bot_lo, top_lo);
            let diff_hi = i16x8_sub(bot_hi, top_hi);
            let out = u8x16_narrow_i16x8(diff_lo, diff_hi);

            v128_store(dst.as_mut_ptr().add(mid + x) as *mut v128, out);

            x += 16;
        }

        while x < width - 1 {
            let value = scratch[bot + x] as i32 - scratch[top + x] as i32;
            dst[mid + x] = value.clamp(0, 255) as u8;
            x += 1;
        }
    }
}

#[target_feature(enable = "simd128")]
pub(super) unsafe fn sobel_vertical_simd(src: &PlanarImage, dst: &mut PlanarImage) {
    let width = src.width() as usize;
    let height = src.height() as usize;

    dst.r.fill(0);
    dst.g.fill(0);
    dst.b.fill(0);
    dst.a.fill(0);

    sobel_vertical_channel_simd(&src.r, &mut dst.r, width, height);
    sobel_vertical_channel_simd(&src.g, &mut dst.g, width, height);
    sobel_vertical_channel_simd(&src.b, &mut dst.b, width, height);
    sobel_vertical_channel_simd(&src.a, &mut dst.a, width, height);
    restore_alpha_if_filter_zeroed_it(src, dst, width, height);
}

#[target_feature(enable = "simd128")]
unsafe fn sobel_vertical_channel_simd(
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
            // We have to do more work than the horizontal pass because the data is not contiguous
            let top_left = v128_load(src.as_ptr().add(top + x - 1) as *const v128);
            let top_right = v128_load(src.as_ptr().add(top + x + 1) as *const v128);
            let mid_left = v128_load(src.as_ptr().add(mid + x - 1) as *const v128);
            let mid_right = v128_load(src.as_ptr().add(mid + x + 1) as *const v128);
            let bot_left = v128_load(src.as_ptr().add(bot + x - 1) as *const v128);
            let bot_right = v128_load(src.as_ptr().add(bot + x + 1) as *const v128);

            let top_diff_lo = i16x8_sub(
                u16x8_extend_low_u8x16(top_right),
                u16x8_extend_low_u8x16(top_left),
            );
            let top_diff_hi = i16x8_sub(
                u16x8_extend_high_u8x16(top_right),
                u16x8_extend_high_u8x16(top_left),
            );
            let mid_diff_lo = i16x8_sub(
                u16x8_extend_low_u8x16(mid_right),
                u16x8_extend_low_u8x16(mid_left),
            );
            let mid_diff_hi = i16x8_sub(
                u16x8_extend_high_u8x16(mid_right),
                u16x8_extend_high_u8x16(mid_left),
            );
            let bot_diff_lo = i16x8_sub(
                u16x8_extend_low_u8x16(bot_right),
                u16x8_extend_low_u8x16(bot_left),
            );
            let bot_diff_hi = i16x8_sub(
                u16x8_extend_high_u8x16(bot_right),
                u16x8_extend_high_u8x16(bot_left),
            );

            let value_lo = i16x8_add(
                i16x8_add(top_diff_lo, i16x8_add(mid_diff_lo, mid_diff_lo)),
                bot_diff_lo,
            );
            let value_hi = i16x8_add(
                i16x8_add(top_diff_hi, i16x8_add(mid_diff_hi, mid_diff_hi)),
                bot_diff_hi,
            );
            let out = u8x16_narrow_i16x8(value_lo, value_hi);

            v128_store(dst.as_mut_ptr().add(mid + x) as *mut v128, out);

            x += 16;
        }

        while x < width - 1 {
            let top_diff = src[top + x + 1] as i32 - src[top + x - 1] as i32;
            let mid_diff = src[mid + x + 1] as i32 - src[mid + x - 1] as i32;
            let bot_diff = src[bot + x + 1] as i32 - src[bot + x - 1] as i32;
            let value = top_diff + 2 * mid_diff + bot_diff;

            dst[mid + x] = value.clamp(0, 255) as u8;
            x += 1;
        }
    }
}
