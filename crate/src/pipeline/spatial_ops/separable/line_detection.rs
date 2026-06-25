use crate::pipeline::PlanarImage;

use crate::pipeline::spatial_ops::common::restore_alpha_if_filter_zeroed_it;

#[target_feature(enable = "simd128")]
pub(in crate::pipeline::spatial_ops) unsafe fn detect_horizontal_lines_simd(
    src: &PlanarImage,
    dst: &mut PlanarImage,
    scratch: &mut [i16],
) {
    let width = src.width() as usize;
    let height = src.height() as usize;

    dst.r.fill(0);
    dst.g.fill(0);
    dst.b.fill(0);
    dst.a.fill(0);

    detect_horizontal_lines_channel_simd(&src.r, &mut dst.r, scratch, width, height);
    detect_horizontal_lines_channel_simd(&src.g, &mut dst.g, scratch, width, height);
    detect_horizontal_lines_channel_simd(&src.b, &mut dst.b, scratch, width, height);
    detect_horizontal_lines_channel_simd(&src.a, &mut dst.a, scratch, width, height);
    restore_alpha_if_filter_zeroed_it(src, dst, width, height);
}

#[target_feature(enable = "simd128")]
unsafe fn detect_horizontal_lines_channel_simd(
    src: &[u8],
    dst: &mut [u8],
    scratch: &mut [i16],
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

            let sum_lo = i16x8_add(
                i16x8_add(i16x8_extend_low_u8x16(left), i16x8_extend_low_u8x16(center)),
                i16x8_extend_low_u8x16(right),
            );
            let sum_hi = i16x8_add(
                i16x8_add(
                    i16x8_extend_high_u8x16(left),
                    i16x8_extend_high_u8x16(center),
                ),
                i16x8_extend_high_u8x16(right),
            );

            v128_store(scratch.as_mut_ptr().add(row + x) as *mut v128, sum_lo);
            v128_store(scratch.as_mut_ptr().add(row + x + 8) as *mut v128, sum_hi);

            x += 16;
        }

        while x < width - 1 {
            scratch[row + x] =
                src[row + x - 1] as i16 + src[row + x] as i16 + src[row + x + 1] as i16;
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
            let mid_lo = v128_load(scratch.as_ptr().add(mid + x) as *const v128);
            let bot_lo = v128_load(scratch.as_ptr().add(bot + x) as *const v128);
            let top_hi = v128_load(scratch.as_ptr().add(top + x + 8) as *const v128);
            let mid_hi = v128_load(scratch.as_ptr().add(mid + x + 8) as *const v128);
            let bot_hi = v128_load(scratch.as_ptr().add(bot + x + 8) as *const v128);

            let value_lo = i16x8_sub(i16x8_sub(i16x8_shl(mid_lo, 1), top_lo), bot_lo);
            let value_hi = i16x8_sub(i16x8_sub(i16x8_shl(mid_hi, 1), top_hi), bot_hi);
            let out = u8x16_narrow_i16x8(value_lo, value_hi);

            v128_store(dst.as_mut_ptr().add(mid + x) as *mut v128, out);

            x += 16;
        }

        while x < width - 1 {
            let value = 2 * scratch[mid + x] as i32
                - scratch[top + x] as i32
                - scratch[bot + x] as i32;
            dst[mid + x] = value.clamp(0, 255) as u8;
            x += 1;
        }
    }
}

#[target_feature(enable = "simd128")]
pub(in crate::pipeline::spatial_ops) unsafe fn detect_vertical_lines_simd(
    src: &PlanarImage,
    dst: &mut PlanarImage,
) {
    let width = src.width() as usize;
    let height = src.height() as usize;

    dst.r.fill(0);
    dst.g.fill(0);
    dst.b.fill(0);
    dst.a.fill(0);

    detect_vertical_lines_channel_simd(&src.r, &mut dst.r, width, height);
    detect_vertical_lines_channel_simd(&src.g, &mut dst.g, width, height);
    detect_vertical_lines_channel_simd(&src.b, &mut dst.b, width, height);
    detect_vertical_lines_channel_simd(&src.a, &mut dst.a, width, height);
    restore_alpha_if_filter_zeroed_it(src, dst, width, height);
}

/// Maybe surprisingly, this function runs faster if we do not separate the kernels into two passes.
/// This is because the middle column are only zeroes, so we do roughly as much work when we do it directly compared to the separable kernels.
/// However, if we do it directly on the 3x3 kernel, do not suffer from the overhead of writing to and reading from the scratch image.
#[target_feature(enable = "simd128")]
unsafe fn detect_vertical_lines_channel_simd(
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
            let mid_center = v128_load(src.as_ptr().add(mid + x) as *const v128);
            let mid_right = v128_load(src.as_ptr().add(mid + x + 1) as *const v128);
            let bot_left = v128_load(src.as_ptr().add(bot + x - 1) as *const v128);
            let bot_center = v128_load(src.as_ptr().add(bot + x) as *const v128);
            let bot_right = v128_load(src.as_ptr().add(bot + x + 1) as *const v128);

            let top_lo = second_derivative_low(top_left, top_center, top_right);
            let top_hi = second_derivative_high(top_left, top_center, top_right);
            let mid_lo = second_derivative_low(mid_left, mid_center, mid_right);
            let mid_hi = second_derivative_high(mid_left, mid_center, mid_right);
            let bot_lo = second_derivative_low(bot_left, bot_center, bot_right);
            let bot_hi = second_derivative_high(bot_left, bot_center, bot_right);

            let value_lo = i16x8_add(i16x8_add(top_lo, mid_lo), bot_lo);
            let value_hi = i16x8_add(i16x8_add(top_hi, mid_hi), bot_hi);
            let out = u8x16_narrow_i16x8(value_lo, value_hi);

            v128_store(dst.as_mut_ptr().add(mid + x) as *mut v128, out);

            x += 16;
        }

        while x < width - 1 {
            let top_value = 2 * src[top + x] as i32
                - src[top + x - 1] as i32
                - src[top + x + 1] as i32;
            let mid_value = 2 * src[mid + x] as i32
                - src[mid + x - 1] as i32
                - src[mid + x + 1] as i32;
            let bot_value = 2 * src[bot + x] as i32
                - src[bot + x - 1] as i32
                - src[bot + x + 1] as i32;
            let value = top_value + mid_value + bot_value;

            dst[mid + x] = value.clamp(0, 255) as u8;
            x += 1;
        }
    }
}

#[target_feature(enable = "simd128")]
unsafe fn second_derivative_low(
    left: core::arch::wasm32::v128,
    center: core::arch::wasm32::v128,
    right: core::arch::wasm32::v128,
) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;

    i16x8_sub(
        i16x8_sub(
            i16x8_shl(u16x8_extend_low_u8x16(center), 1),
            u16x8_extend_low_u8x16(left),
        ),
        u16x8_extend_low_u8x16(right),
    )
}

#[target_feature(enable = "simd128")]
unsafe fn second_derivative_high(
    left: core::arch::wasm32::v128,
    center: core::arch::wasm32::v128,
    right: core::arch::wasm32::v128,
) -> core::arch::wasm32::v128 {
    use core::arch::wasm32::*;

    i16x8_sub(
        i16x8_sub(
            i16x8_shl(u16x8_extend_high_u8x16(center), 1),
            u16x8_extend_high_u8x16(left),
        ),
        u16x8_extend_high_u8x16(right),
    )
}
