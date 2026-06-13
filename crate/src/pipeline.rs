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
        let mut r: Vec<u8> = Vec::with_capacity((img.width * img.height) as usize);
        let mut g: Vec<u8> = Vec::with_capacity((img.width * img.height) as usize);
        let mut b: Vec<u8> = Vec::with_capacity((img.width * img.height) as usize);
        let mut a: Vec<u8> = Vec::with_capacity((img.width * img.height) as usize);

        for px in img.raw_pixels.chunks_exact(4) {
            r.push(px[0]);
            g.push(px[1]);
            b.push(px[2]);
            a.push(px[3]);
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
        let mut raw_pixels: Vec<u8> = vec![];

        for i in 0..(self.width * self.height) as usize {
            raw_pixels.push(self.r[i]);
            raw_pixels.push(self.g[i]);
            raw_pixels.push(self.b[i]);
            raw_pixels.push(self.a[i]);
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

        for i in 0..self.image.r.len() {
            let mut r = self.image.r[i];
            let mut g = self.image.g[i];
            let mut b = self.image.b[i];

            for op in &self.pending {
                match op {
                    PixelOp::GrayScale => {
                        let avg = ((r as u32 + g as u32 + b as u32) / 3) as u8;
                        r = avg;
                        g = avg;
                        b = avg;
                    }
                    PixelOp::Monochrome {
                        r_offset,
                        g_offset,
                        b_offset,
                    } => {
                        let avg = (r as u32 + g as u32 + b as u32) / 3;
                        r = (avg + *r_offset as u32).min(255) as u8;
                        g = (avg + *g_offset as u32).min(255) as u8;
                        b = (avg + *b_offset as u32).min(255) as u8;
                    }
                    PixelOp::Invert => {
                        r = 255 - r;
                        g = 255 - g;
                        b = 255 - b;
                    }
                    PixelOp::AlterChannels {
                        r: dr,
                        g: dg,
                        b: db,
                    } => {
                        r = (r as i16 + dr).clamp(0, 255) as u8;
                        g = (g as i16 + dg).clamp(0, 255) as u8;
                        b = (b as i16 + db).clamp(0, 255) as u8;
                    }
                }
            }

            self.image.r[i] = r;
            self.image.g[i] = g;
            self.image.b[i] = b;
        }

        self.pending.clear();
    }
}

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
