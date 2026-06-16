use crate::pipeline::{Pipeline, PlanarImage};

impl Pipeline {
    pub fn convolve_3x3(mut self, kernel: [f32; 9]) -> Self {
        self.flush_pixel_ops();
        self.ensure_scratch();

        let scratch = self.scratch.as_mut().unwrap();
        convolve_3x3(&self.image, scratch, kernel);
        std::mem::swap(&mut self.image, scratch);
        self
    }

    pub fn blur_3x3(self) -> Self {
        self.convolve_3x3([0.0; 9])
    }
}

pub fn convolve_3x3(src: &PlanarImage, dst: &mut PlanarImage, kernel: [f32; 9]) {
    let width = src.width() as usize;
    let height = src.height() as usize;

    // If we want to match the result from the photon library, then we skip the borders
    // So we must fill the lanes with 0 beforehand so that we do not have accidental stale data in there
    dst.r.fill(0);
    dst.g.fill(0);
    dst.b.fill(0);
    // Not modified but does need to be copied over
    dst.a.copy_from_slice(&src.a);

    convolve_3x3_channel(&src.r, &mut dst.r, width, height, kernel);
    convolve_3x3_channel(&src.g, &mut dst.g, width, height, kernel);
    convolve_3x3_channel(&src.b, &mut dst.b, width, height, kernel);
}

pub fn convolve_3x3_channel(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    kernel: [f32; 9],
) {
    if width < 3 || height < 3 {
        return;
    }

    let sum: f32 = kernel.iter().sum();
    let divisor = if sum == 0.0 { 1.0 } else { sum };

    for y in 0..height {
        let top = (y - 1) * width;
        let mid = y * width;
        let bot = (y + 1) * width;

        for x in 1..width - 1 {
            let value = src[top + x - 1] as f32 * kernel[0]
                + src[top + x] as f32 * kernel[1]
                + src[top + x + 1] as f32 * kernel[2]
                + src[mid + x - 1] as f32 * kernel[3]
                + src[mid + x] as f32 * kernel[4]
                + src[mid + x + 1] as f32 * kernel[5]
                + src[bot + x - 1] as f32 * kernel[6]
                + src[bot + x] as f32 * kernel[7]
                + src[bot + x + 1] as f32 * kernel[8];

            dst[mid + x] = (value / divisor).clamp(0.0, 255.0) as u8;
        }
    }
}
