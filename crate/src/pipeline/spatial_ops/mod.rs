use crate::pipeline::{Pipeline, PlanarImage};

#[macro_use]
mod common;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod box_blur_simd;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod line_detection;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod prewitt_simd;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod sobel_simd;

use common::restore_alpha_if_filter_zeroed_it;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use box_blur_simd::box_blur_3x3_simd;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use line_detection::{detect_horizontal_lines_simd, detect_vertical_lines_simd};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use prewitt_simd::prewitt_horizontal_simd;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use sobel_simd::{sobel_horizontal_simd, sobel_vertical_simd};

const NOISE_REDUCTION: [f32; 9] = [0.0, -1.0, 7.0, -1.0, 5.0, 9.0, 0.0, 7.0, 9.0];
const SHARPEN: [f32; 9] = [0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0];
const EDGE_DETECTION: [f32; 9] = [-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0];
const DETECT_45_DEG_LINES: [f32; 9] =
    [-1.0, -1.0, 2.0, -1.0, 2.0, -1.0, 2.0, -1.0, -1.0];
const DETECT_135_DEG_LINES: [f32; 9] =
    [2.0, -1.0, -1.0, -1.0, 2.0, -1.0, -1.0, -1.0, 2.0];
const LAPLACE: [f32; 9] = [0.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 0.0];
const EDGE_ONE: [f32; 9] = [0.0, -2.2, -0.6, -0.4, 2.8, -0.3, -0.8, -1.0, 2.7];
const EMBOSS: [f32; 9] = [-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
const PREWITT_HORIZONTAL: [f32; 9] = [5.0, -3.0, -3.0, 5.0, 0.0, -3.0, 5.0, -3.0, -3.0];

impl Pipeline {
    pub fn convolve_3x3(mut self, kernel: [f32; 9]) -> Self {
        self.apply_direct_3x3(kernel);
        self
    }

    pub fn convolve_separable_3x3(
        mut self,
        horizontal: [f32; 3],
        vertical: [f32; 3],
    ) -> Self {
        self.apply_separable_3x3(horizontal, vertical);
        self
    }

    pub fn box_blur(mut self) -> Self {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        unsafe {
            self.apply_box_blur_simd();
        }

        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            self.apply_separable_3x3([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]);
        }

        self
    }

    pub fn noise_reduction(self) -> Self {
        self.convolve_3x3(NOISE_REDUCTION)
    }

    pub fn sharpen(self) -> Self {
        self.convolve_3x3(SHARPEN)
    }

    pub fn edge_detection(self) -> Self {
        self.convolve_3x3(EDGE_DETECTION)
    }

    pub fn identity(self) -> Self {
        self
    }

    pub fn detect_horizontal_lines(mut self) -> Self {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        unsafe {
            self.apply_detect_horizontal_lines_simd();
        }

        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            self.apply_separable_3x3([1.0, 1.0, 1.0], [-1.0, 2.0, -1.0]);
        }

        self
    }

    pub fn detect_vertical_lines(mut self) -> Self {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        unsafe {
            self.apply_detect_vertical_lines_simd();
        }

        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            self.apply_separable_3x3([-1.0, 2.0, -1.0], [1.0, 1.0, 1.0]);
        }

        self
    }

    pub fn detect_45_deg_lines(self) -> Self {
        self.convolve_3x3(DETECT_45_DEG_LINES)
    }

    pub fn detect_135_deg_lines(self) -> Self {
        self.convolve_3x3(DETECT_135_DEG_LINES)
    }

    pub fn laplace(self) -> Self {
        self.convolve_3x3(LAPLACE)
    }

    pub fn edge_one(self) -> Self {
        self.convolve_3x3(EDGE_ONE)
    }

    pub fn emboss(self) -> Self {
        self.convolve_3x3(EMBOSS)
    }

    pub fn sobel_horizontal(mut self) -> Self {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        unsafe {
            self.apply_sobel_horizontal_simd();
        }

        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            self.apply_separable_3x3([1.0, 2.0, 1.0], [-1.0, 0.0, 1.0]);
        }

        self
    }

    pub fn prewitt_horizontal(mut self) -> Self {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        unsafe {
            self.apply_prewitt_horizontal_simd();
        }

        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            self.apply_direct_3x3(PREWITT_HORIZONTAL);
        }

        self
    }

    pub fn sobel_vertical(mut self) -> Self {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        unsafe {
            self.apply_sobel_vertical_simd();
        }

        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            self.apply_separable_3x3([-1.0, 0.0, 1.0], [1.0, 2.0, 1.0]);
        }

        self
    }

    fn apply_direct_3x3(&mut self, kernel: [f32; 9]) {
        self.flush_pixel_ops();
        self.ensure_scratch();

        let scratch = self.scratch.as_mut().unwrap();
        convolve_3x3(&self.image, scratch, kernel);
        std::mem::swap(&mut self.image, scratch);
    }

    fn apply_separable_3x3(&mut self, horizontal: [f32; 3], vertical: [f32; 3]) {
        self.flush_pixel_ops();
        self.ensure_scratch();
        self.ensure_f32_scratch();

        let scratch = self.scratch.as_mut().unwrap();
        let f32_scratch = self.f32_scratch.as_mut().unwrap();

        convolve_separable_3x3(&self.image, scratch, f32_scratch, horizontal, vertical);
        std::mem::swap(&mut self.image, scratch);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[target_feature(enable = "simd128")]
    unsafe fn apply_box_blur_simd(&mut self) {
        self.flush_pixel_ops();
        self.ensure_scratch();
        self.ensure_i16_scratch();

        let scratch = self.scratch.as_mut().unwrap();
        let i16_scratch = self.i16_scratch.as_mut().unwrap();

        box_blur_3x3_simd(&self.image, scratch, i16_scratch);
        std::mem::swap(&mut self.image, scratch);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[target_feature(enable = "simd128")]
    unsafe fn apply_sobel_horizontal_simd(&mut self) {
        self.flush_pixel_ops();
        self.ensure_scratch();
        self.ensure_i16_scratch();

        let scratch = self.scratch.as_mut().unwrap();
        let i16_scratch = self.i16_scratch.as_mut().unwrap();

        sobel_horizontal_simd(&self.image, scratch, i16_scratch);
        std::mem::swap(&mut self.image, scratch);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[target_feature(enable = "simd128")]
    unsafe fn apply_sobel_vertical_simd(&mut self) {
        self.flush_pixel_ops();
        self.ensure_scratch();
        self.ensure_i16_scratch();

        let scratch = self.scratch.as_mut().unwrap();
        let i16_scratch = self.i16_scratch.as_mut().unwrap();

        sobel_vertical_simd(&self.image, scratch, i16_scratch);
        std::mem::swap(&mut self.image, scratch);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[target_feature(enable = "simd128")]
    unsafe fn apply_detect_horizontal_lines_simd(&mut self) {
        self.flush_pixel_ops();
        self.ensure_scratch();
        self.ensure_i16_scratch();

        let scratch = self.scratch.as_mut().unwrap();
        let i16_scratch = self.i16_scratch.as_mut().unwrap();

        detect_horizontal_lines_simd(&self.image, scratch, i16_scratch);
        std::mem::swap(&mut self.image, scratch);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[target_feature(enable = "simd128")]
    unsafe fn apply_detect_vertical_lines_simd(&mut self) {
        self.flush_pixel_ops();
        self.ensure_scratch();

        let scratch = self.scratch.as_mut().unwrap();

        detect_vertical_lines_simd(&self.image, scratch);
        std::mem::swap(&mut self.image, scratch);
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    #[target_feature(enable = "simd128")]
    unsafe fn apply_prewitt_horizontal_simd(&mut self) {
        self.flush_pixel_ops();
        self.ensure_scratch();

        let scratch = self.scratch.as_mut().unwrap();

        prewitt_horizontal_simd(&self.image, scratch);
        std::mem::swap(&mut self.image, scratch);
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
    dst.a.fill(0);

    convolve_3x3_channel(&src.r, &mut dst.r, width, height, kernel);
    convolve_3x3_channel(&src.g, &mut dst.g, width, height, kernel);
    convolve_3x3_channel(&src.b, &mut dst.b, width, height, kernel);
    convolve_3x3_channel(&src.a, &mut dst.a, width, height, kernel);
    restore_alpha_if_filter_zeroed_it(src, dst, width, height);
}

fn convolve_3x3_channel(
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

    for y in 1..height - 1 {
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

pub fn convolve_separable_3x3(
    src: &PlanarImage,
    dst: &mut PlanarImage,
    scratch: &mut [f32],
    horizontal: [f32; 3],
    vertical: [f32; 3],
) {
    let width = src.width() as usize;
    let height = src.height() as usize;

    dst.r.fill(0);
    dst.g.fill(0);
    dst.b.fill(0);
    dst.a.fill(0);

    convolve_separable_3x3_channel(
        &src.r, &mut dst.r, scratch, width, height, horizontal, vertical,
    );
    convolve_separable_3x3_channel(
        &src.g, &mut dst.g, scratch, width, height, horizontal, vertical,
    );
    convolve_separable_3x3_channel(
        &src.b, &mut dst.b, scratch, width, height, horizontal, vertical,
    );
    convolve_separable_3x3_channel(
        &src.a, &mut dst.a, scratch, width, height, horizontal, vertical,
    );
    restore_alpha_if_filter_zeroed_it(src, dst, width, height);
}

fn convolve_separable_3x3_channel(
    src: &[u8],
    dst: &mut [u8],
    scratch: &mut [f32],
    width: usize,
    height: usize,
    horizontal: [f32; 3],
    vertical: [f32; 3],
) {
    if width < 3 || height < 3 {
        return;
    }

    let divisor = {
        let sum = horizontal.iter().sum::<f32>() * vertical.iter().sum::<f32>();
        if sum == 0.0 {
            1.0
        } else {
            sum
        }
    };

    for y in 0..height {
        let row = y * width;

        for x in 1..width - 1 {
            scratch[row + x] = src[row + x - 1] as f32 * horizontal[0]
                + src[row + x] as f32 * horizontal[1]
                + src[row + x + 1] as f32 * horizontal[2];
        }
    }

    for y in 1..height - 1 {
        let top = (y - 1) * width;
        let mid = y * width;
        let bot = (y + 1) * width;

        for x in 1..width - 1 {
            let value = scratch[top + x] * vertical[0]
                + scratch[mid + x] * vertical[1]
                + scratch[bot + x] * vertical[2];

            dst[mid + x] = (value / divisor).clamp(0.0, 255.0) as u8;
        }
    }
}
