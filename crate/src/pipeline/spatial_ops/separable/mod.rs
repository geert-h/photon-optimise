#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod box_blur_simd;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod line_detection;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod sobel_simd;
