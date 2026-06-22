#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[macro_use]
mod coeff;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[macro_use]
mod division;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[macro_use]
mod kernel;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[macro_use]
mod channel;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
macro_rules! apply_direct_3x3_simd {
    (
        $pipeline:ident,
        [$k0:expr, $k1:expr, $k2:expr,
         $k3:expr, $k4:expr, $k5:expr,
         $k6:expr, $k7:expr, $k8:expr $(,)?]
    ) => {{
        $pipeline.flush_pixel_ops();
        $pipeline.ensure_scratch();

        let width = $pipeline.image.width() as usize;
        let height = $pipeline.image.height() as usize;
        let scratch = $pipeline.scratch.as_mut().unwrap();

        scratch.r.fill(0);
        scratch.g.fill(0);
        scratch.b.fill(0);
        scratch.a.fill(0);

        direct_3x3_simd_channel!(
            &$pipeline.image.r,
            &mut scratch.r,
            width,
            height,
            [$k0, $k1, $k2, $k3, $k4, $k5, $k6, $k7, $k8]
        );
        direct_3x3_simd_channel!(
            &$pipeline.image.g,
            &mut scratch.g,
            width,
            height,
            [$k0, $k1, $k2, $k3, $k4, $k5, $k6, $k7, $k8]
        );
        direct_3x3_simd_channel!(
            &$pipeline.image.b,
            &mut scratch.b,
            width,
            height,
            [$k0, $k1, $k2, $k3, $k4, $k5, $k6, $k7, $k8]
        );
        direct_3x3_simd_channel!(
            &$pipeline.image.a,
            &mut scratch.a,
            width,
            height,
            [$k0, $k1, $k2, $k3, $k4, $k5, $k6, $k7, $k8]
        );

        restore_alpha_if_filter_zeroed_it(&$pipeline.image, scratch, width, height);

        std::mem::swap(&mut $pipeline.image, scratch);
    }};
}
