#[allow(dead_code)]
pub(super) enum PixelOp {
    GrayScale,
    Monochrome {
        r_offset: u8,
        g_offset: u8,
        b_offset: u8,
    },
    Invert,
    AlterChannels {
        r: i16,
        g: i16,
        b: i16,
    },
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub(super) fn apply_pixel_op_scalar(op: &PixelOp, r: &mut u8, g: &mut u8, b: &mut u8) {
    match op {
        PixelOp::GrayScale => {
            let avg = ((*r as u32 + *g as u32 + *b as u32) / 3) as u8;
            *r = avg;
            *g = avg;
            *b = avg;
        }
        PixelOp::Monochrome {
            r_offset,
            g_offset,
            b_offset,
        } => {
            let avg = (*r as u32 + *g as u32 + *b as u32) / 3;
            *r = (avg + *r_offset as u32).min(255) as u8;
            *g = (avg + *g_offset as u32).min(255) as u8;
            *b = (avg + *b_offset as u32).min(255) as u8;
        }
        PixelOp::Invert => {
            *r = 255 - *r;
            *g = 255 - *g;
            *b = 255 - *b;
        }
        PixelOp::AlterChannels {
            r: dr,
            g: dg,
            b: db,
        } => {
            *r = (*r as i16 + dr).clamp(0, 255) as u8;
            *g = (*g as i16 + dg).clamp(0, 255) as u8;
            *b = (*b as i16 + db).clamp(0, 255) as u8;
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
pub(super) unsafe fn invert_planes_simd(r: &mut [u8], g: &mut [u8], b: &mut [u8]) {
    use core::arch::wasm32::*;

    let simd_len = r.len() - (r.len() % 16);
    let mask = u8x16_splat(255);

    for i in (0..simd_len).step_by(16) {
        let rv = v128_load(r.as_ptr().add(i) as *const v128);
        let gv = v128_load(g.as_ptr().add(i) as *const v128);
        let bv = v128_load(b.as_ptr().add(i) as *const v128);

        v128_store(r.as_mut_ptr().add(i) as *mut v128, v128_xor(rv, mask));
        v128_store(g.as_mut_ptr().add(i) as *mut v128, v128_xor(gv, mask));
        v128_store(b.as_mut_ptr().add(i) as *mut v128, v128_xor(bv, mask));
    }

    for i in simd_len..r.len() {
        r[i] = 255 - r[i];
        g[i] = 255 - g[i];
        b[i] = 255 - b[i];
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
pub(super) unsafe fn alter_channels_planes_simd(
    r: &mut [u8],
    g: &mut [u8],
    b: &mut [u8],
    r_amt: i16,
    g_amt: i16,
    b_amt: i16,
) {
    alter_plane_simd(r, r_amt);
    alter_plane_simd(g, g_amt);
    alter_plane_simd(b, b_amt);
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn alter_plane_simd(channel: &mut [u8], amt: i16) {
    use core::arch::wasm32::*;

    let simd_len = channel.len() - (channel.len() % 16);
    let magnitude = amt.unsigned_abs().min(255) as u8;
    let delta = u8x16_splat(magnitude);

    for i in (0..simd_len).step_by(16) {
        let values = v128_load(channel.as_ptr().add(i) as *const v128);
        let output = if amt >= 0 {
            u8x16_add_sat(values, delta)
        } else {
            u8x16_sub_sat(values, delta)
        };
        v128_store(channel.as_mut_ptr().add(i) as *mut v128, output);
    }

    for value in &mut channel[simd_len..] {
        *value = (*value as i16 + amt).clamp(0, 255) as u8;
    }
}
