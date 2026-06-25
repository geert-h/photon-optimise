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
pub mod edge_one_simd;

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
macro_rules! apply_direct_3x3_simd {
    (
        $pipeline:ident,
        [$($kernel:tt)+]
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
            raw [$($kernel)+],
            expr [$($kernel)+]
        );
        direct_3x3_simd_channel!(
            &$pipeline.image.g,
            &mut scratch.g,
            width,
            height,
            raw [$($kernel)+],
            expr [$($kernel)+]
        );
        direct_3x3_simd_channel!(
            &$pipeline.image.b,
            &mut scratch.b,
            width,
            height,
            raw [$($kernel)+],
            expr [$($kernel)+]
        );
        direct_3x3_simd_channel!(
            &$pipeline.image.a,
            &mut scratch.a,
            width,
            height,
            raw [$($kernel)+],
            expr [$($kernel)+]
        );

        restore_alpha_if_filter_zeroed_it(&$pipeline.image, scratch, width, height);

        std::mem::swap(&mut $pipeline.image, scratch);
    }};
}
