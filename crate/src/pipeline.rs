use crate::PhotonImage;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::wasm_bindgen;

#[cfg_attr(feature = "enable_wasm", wasm_bindgen)]
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PlanarImage {
    r: Vec<u8>,
    g: Vec<u8>,
    b: Vec<u8>,
    a: Vec<u8>,
    width: u32,
    height: u32,
}

impl PlanarImage {
    pub fn from_photon_image(img: &PhotonImage) -> PlanarImage {
        let pixels = (img.width * img.height) as usize;

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        unsafe {
            PlanarImage::from_photon_image_simd(img, pixels)
        }

        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            PlanarImage::from_photon_image_scalar(img, pixels)
        }
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    fn from_photon_image_scalar(img: &PhotonImage, pixels: usize) -> PlanarImage {
        let mut r = vec![0; pixels];
        let mut g = vec![0; pixels];
        let mut b = vec![0; pixels];
        let mut a = vec![0; pixels];

        for (i, px) in img.raw_pixels.chunks_exact(4).enumerate() {
            r[i] = px[0];
            g[i] = px[1];
            b[i] = px[2];
            a[i] = px[3];
        }

        PlanarImage {
            r,
            g,
            b,
            a,
            width: img.width,
            height: img.height,
        }
    }

    pub fn to_photon_image(&self) -> PhotonImage {
        let pixels = (self.width * self.height) as usize;

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        unsafe {
            self.to_photon_image_simd(pixels)
        }

        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            self.to_photon_image_scalar(pixels)
        }
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    fn to_photon_image_scalar(&self, pixels: usize) -> PhotonImage {
        let mut raw_pixels = vec![0; pixels * 4];

        for i in 0..pixels {
            let base = i * 4;
            raw_pixels[base] = self.r[i];
            raw_pixels[base + 1] = self.g[i];
            raw_pixels[base + 2] = self.b[i];
            raw_pixels[base + 3] = self.a[i];
        }

        PhotonImage {
            raw_pixels,
            width: self.width,
            height: self.height,
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[target_feature(enable = "simd128")]
    unsafe fn from_photon_image_simd(img: &PhotonImage, pixels: usize) -> PlanarImage {
        use core::arch::wasm32::*;

        let mut r = vec![0; pixels];
        let mut g = vec![0; pixels];
        let mut b = vec![0; pixels];
        let mut a = vec![0; pixels];

        let simd_pixels = pixels - (pixels % 16);

        for i in (0..simd_pixels).step_by(16) {
            let base = i * 4;
            let px0 = v128_load(img.raw_pixels.as_ptr().add(base) as *const v128);
            let px1 = v128_load(img.raw_pixels.as_ptr().add(base + 16) as *const v128);
            let px2 = v128_load(img.raw_pixels.as_ptr().add(base + 32) as *const v128);
            let px3 = v128_load(img.raw_pixels.as_ptr().add(base + 48) as *const v128);

            let r01 = u8x16_shuffle::<0, 4, 8, 12, 16, 20, 24, 28, 0, 0, 0, 0, 0, 0, 0, 0>(
                px0, px1,
            );
            let r23 = u8x16_shuffle::<0, 4, 8, 12, 16, 20, 24, 28, 0, 0, 0, 0, 0, 0, 0, 0>(
                px2, px3,
            );
            let g01 = u8x16_shuffle::<1, 5, 9, 13, 17, 21, 25, 29, 0, 0, 0, 0, 0, 0, 0, 0>(
                px0, px1,
            );
            let g23 = u8x16_shuffle::<1, 5, 9, 13, 17, 21, 25, 29, 0, 0, 0, 0, 0, 0, 0, 0>(
                px2, px3,
            );
            let b01 =
                u8x16_shuffle::<2, 6, 10, 14, 18, 22, 26, 30, 0, 0, 0, 0, 0, 0, 0, 0>(
                    px0, px1,
                );
            let b23 =
                u8x16_shuffle::<2, 6, 10, 14, 18, 22, 26, 30, 0, 0, 0, 0, 0, 0, 0, 0>(
                    px2, px3,
                );
            let a01 =
                u8x16_shuffle::<3, 7, 11, 15, 19, 23, 27, 31, 0, 0, 0, 0, 0, 0, 0, 0>(
                    px0, px1,
                );
            let a23 =
                u8x16_shuffle::<3, 7, 11, 15, 19, 23, 27, 31, 0, 0, 0, 0, 0, 0, 0, 0>(
                    px2, px3,
                );

            let r_vec =
                u8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(
                    r01, r23,
                );
            let g_vec =
                u8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(
                    g01, g23,
                );
            let b_vec =
                u8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(
                    b01, b23,
                );
            let a_vec =
                u8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(
                    a01, a23,
                );

            v128_store(r.as_mut_ptr().add(i) as *mut v128, r_vec);
            v128_store(g.as_mut_ptr().add(i) as *mut v128, g_vec);
            v128_store(b.as_mut_ptr().add(i) as *mut v128, b_vec);
            v128_store(a.as_mut_ptr().add(i) as *mut v128, a_vec);
        }

        for (offset, px) in img.raw_pixels[simd_pixels * 4..]
            .chunks_exact(4)
            .enumerate()
        {
            let i = simd_pixels + offset;
            r[i] = px[0];
            g[i] = px[1];
            b[i] = px[2];
            a[i] = px[3];
        }

        PlanarImage {
            r,
            g,
            b,
            a,
            width: img.width,
            height: img.height,
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[target_feature(enable = "simd128")]
    unsafe fn to_photon_image_simd(&self, pixels: usize) -> PhotonImage {
        use core::arch::wasm32::*;

        let mut raw_pixels = vec![0; pixels * 4];
        let simd_pixels = pixels - (pixels % 16);

        for i in (0..simd_pixels).step_by(16) {
            let r = v128_load(self.r.as_ptr().add(i) as *const v128);
            let g = v128_load(self.g.as_ptr().add(i) as *const v128);
            let b = v128_load(self.b.as_ptr().add(i) as *const v128);
            let a = v128_load(self.a.as_ptr().add(i) as *const v128);

            let rg_lo =
                u8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                    r, g,
                );
            let rg_hi = u8x16_shuffle::<
                8,
                24,
                9,
                25,
                10,
                26,
                11,
                27,
                12,
                28,
                13,
                29,
                14,
                30,
                15,
                31,
            >(r, g);
            let ba_lo =
                u8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                    b, a,
                );
            let ba_hi = u8x16_shuffle::<
                8,
                24,
                9,
                25,
                10,
                26,
                11,
                27,
                12,
                28,
                13,
                29,
                14,
                30,
                15,
                31,
            >(b, a);

            let out0 =
                u8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(
                    rg_lo, ba_lo,
                );
            let out1 = u8x16_shuffle::<
                8,
                9,
                24,
                25,
                10,
                11,
                26,
                27,
                12,
                13,
                28,
                29,
                14,
                15,
                30,
                31,
            >(rg_lo, ba_lo);
            let out2 =
                u8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(
                    rg_hi, ba_hi,
                );
            let out3 = u8x16_shuffle::<
                8,
                9,
                24,
                25,
                10,
                11,
                26,
                27,
                12,
                13,
                28,
                29,
                14,
                15,
                30,
                31,
            >(rg_hi, ba_hi);

            let base = i * 4;
            v128_store(raw_pixels.as_mut_ptr().add(base) as *mut v128, out0);
            v128_store(raw_pixels.as_mut_ptr().add(base + 16) as *mut v128, out1);
            v128_store(raw_pixels.as_mut_ptr().add(base + 32) as *mut v128, out2);
            v128_store(raw_pixels.as_mut_ptr().add(base + 48) as *mut v128, out3);
        }

        // Scalar tail
        for i in simd_pixels..pixels {
            let base = i * 4;
            raw_pixels[base] = self.r[i];
            raw_pixels[base + 1] = self.g[i];
            raw_pixels[base + 2] = self.b[i];
            raw_pixels[base + 3] = self.a[i];
        }

        PhotonImage {
            raw_pixels,
            width: self.width,
            height: self.height,
        }
    }
}

pub struct Pipeline {
    image: PlanarImage,
    pending: Vec<PixelOp>,
}

impl Pipeline {
    pub fn from_photon_image(img: &PhotonImage) -> Self {
        let planar_image = PlanarImage::from_photon_image(img);
        Pipeline {
            image: planar_image,
            pending: Vec::new(),
        }
    }

    pub fn finish(mut self) -> PhotonImage {
        self.flush_pixel_ops();
        self.image.to_photon_image()
    }

    pub fn gray_scale(mut self) -> Self {
        self.pending.push(PixelOp::GrayScale);
        self
    }

    pub fn monochrome(mut self, r_offset: u8, g_offset: u8, b_offset: u8) -> Self {
        self.pending.push(PixelOp::Monochrome {
            r_offset,
            g_offset,
            b_offset,
        });
        self
    }

    pub fn invert(mut self) -> Self {
        self.pending.push(PixelOp::Invert);
        self
    }

    pub fn alter_channels(mut self, r: i16, g: i16, b: i16) -> Self {
        self.pending.push(PixelOp::AlterChannels { r, g, b });
        self
    }

    pub fn swap_channels(mut self, mut channel1: usize, mut channel2: usize) -> Self {
        if channel1 > 2 {
            panic!(
                "Invalid channel index passed. Channel1 must be equal to 0, 1, or 2."
            );
        }
        if channel2 > 2 {
            panic!(
                "Invalid channel index passed. Channel2 must be equal to 0, 1, or 2."
            );
        }

        self.flush_pixel_ops();

        if channel1 == channel2 {
            return self;
        }

        if channel1 > channel2 {
            std::mem::swap(&mut channel1, &mut channel2);
        }

        match (channel1, channel2) {
            (0, 1) => std::mem::swap(&mut self.image.r, &mut self.image.g),
            (0, 2) => std::mem::swap(&mut self.image.r, &mut self.image.b),
            (1, 2) => std::mem::swap(&mut self.image.g, &mut self.image.b),
            _ => unreachable!(),
        }

        self
    }

    fn flush_pixel_ops(&mut self) {
        if self.pending.is_empty() {
            return;
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        unsafe {
            self.flush_pixel_ops_simd();
        }

        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            self.flush_pixel_ops_scalar();
        }
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    fn flush_pixel_ops_scalar(&mut self) {
        for i in 0..self.image.r.len() {
            let mut r = self.image.r[i];
            let mut g = self.image.g[i];
            let mut b = self.image.b[i];

            for op in &self.pending {
                apply_pixel_op_scalar(op, &mut r, &mut g, &mut b);
            }

            self.image.r[i] = r;
            self.image.g[i] = g;
            self.image.b[i] = b;
        }

        self.pending.clear();
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[target_feature(enable = "simd128")]
    unsafe fn flush_pixel_ops_simd(&mut self) {
        for op in &self.pending {
            match op {
                PixelOp::Invert => invert_planes_simd(
                    &mut self.image.r,
                    &mut self.image.g,
                    &mut self.image.b,
                ),
                PixelOp::AlterChannels { r, g, b } => alter_channels_planes_simd(
                    &mut self.image.r,
                    &mut self.image.g,
                    &mut self.image.b,
                    *r,
                    *g,
                    *b,
                ),
                _ => todo!(
                    "There is no SIMD-optimized implementation for this operation yet"
                ),
            }
        }

        self.pending.clear();
    }
}

#[cfg_attr(feature = "enable_wasm", wasm_bindgen)]
pub fn pipeline_conversion_roundtrip(img: &mut PhotonImage) {
    let output = Pipeline::from_photon_image(img).finish();
    img.raw_pixels = output.raw_pixels;
}

#[cfg_attr(feature = "enable_wasm", wasm_bindgen)]
pub fn pipeline_invert(img: &mut PhotonImage) {
    let output = Pipeline::from_photon_image(img).invert().finish();
    img.raw_pixels = output.raw_pixels;
}

#[cfg_attr(feature = "enable_wasm", wasm_bindgen)]
pub fn pipeline_invert_alter_channels(img: &mut PhotonImage, r: i16, g: i16, b: i16) {
    let output = Pipeline::from_photon_image(img)
        .invert()
        .alter_channels(r, g, b)
        .finish();
    img.raw_pixels = output.raw_pixels;
}

#[allow(dead_code)]
enum PixelOp {
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
fn apply_pixel_op_scalar(op: &PixelOp, r: &mut u8, g: &mut u8, b: &mut u8) {
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
unsafe fn invert_planes_simd(r: &mut [u8], g: &mut [u8], b: &mut [u8]) {
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
unsafe fn alter_channels_planes_simd(
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
