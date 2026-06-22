#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
macro_rules! kernel_3x3_i16x8 {
    (
        [$k0:expr, $k1:expr, $k2:expr,
         $k3:expr, $k4:expr, $k5:expr,
         $k6:expr, $k7:expr, $k8:expr $(,)?],
        [$p0:expr, $p1:expr, $p2:expr,
         $p3:expr, $p4:expr, $p5:expr,
         $p6:expr, $p7:expr, $p8:expr $(,)?]
    ) => {{
        let mut acc = i16x8_splat(0);
        add_coeff_i16x8!(acc, $p0, $k0);
        add_coeff_i16x8!(acc, $p1, $k1);
        add_coeff_i16x8!(acc, $p2, $k2);
        add_coeff_i16x8!(acc, $p3, $k3);
        add_coeff_i16x8!(acc, $p4, $k4);
        add_coeff_i16x8!(acc, $p5, $k5);
        add_coeff_i16x8!(acc, $p6, $k6);
        add_coeff_i16x8!(acc, $p7, $k7);
        add_coeff_i16x8!(acc, $p8, $k8);
        acc
    }};
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
macro_rules! scalar_kernel_3x3_i32 {
    (
        [$k0:expr, $k1:expr, $k2:expr,
         $k3:expr, $k4:expr, $k5:expr,
         $k6:expr, $k7:expr, $k8:expr],
        [$p0:expr, $p1:expr, $p2:expr,
         $p3:expr, $p4:expr, $p5:expr,
         $p6:expr, $p7:expr, $p8:expr]
    ) => {
        scalar_coeff_i32!($p0, $k0)
            + scalar_coeff_i32!($p1, $k1)
            + scalar_coeff_i32!($p2, $k2)
            + scalar_coeff_i32!($p3, $k3)
            + scalar_coeff_i32!($p4, $k4)
            + scalar_coeff_i32!($p5, $k5)
            + scalar_coeff_i32!($p6, $k6)
            + scalar_coeff_i32!($p7, $k7)
            + scalar_coeff_i32!($p8, $k8)
    };
}
