// Wasm benchmarks can be run with:
// `wasm-pack test --node --release crate --test pipeline_spatial_benchmark -- --nocapture`
#![cfg(target_arch = "wasm32")]

mod common;

use common::*;
use photon_rs::conv::{
    box_blur, detect_135_deg_lines, detect_45_deg_lines, detect_horizontal_lines,
    detect_vertical_lines, edge_detection, edge_one, emboss, identity, laplace,
    noise_reduction, prewitt_horizontal, sharpen, sobel_horizontal, sobel_vertical,
};
use wasm_bindgen_test::*;

pub const SPATIAL_IMAGES: &[(&str, &[u8])] = &[
    ("Lena 512x512", include_bytes!("assets/lena.png")),
    ("Perlin 1280x720", include_bytes!("assets/1280x720.png")),
];

const SPATIAL_BENCH_CONFIG: BenchConfig = BenchConfig {
    images: SPATIAL_IMAGES,
    iterations: 50,
    warmups: 10,
};

#[wasm_bindgen_test]
fn run_pipeline_spatial_benchmarks() {
    let benches = vec![
        Bench {
            name: "box_blur",
            original: Box::new(box_blur),
            pipeline: pipeline_to_fn(|p| p.box_blur()),
        },
        Bench {
            name: "noise_reduction",
            original: Box::new(noise_reduction),
            pipeline: pipeline_to_fn(|p| p.noise_reduction()),
        },
        Bench {
            name: "sharpen",
            original: Box::new(sharpen),
            pipeline: pipeline_to_fn(|p| p.sharpen()),
        },
        Bench {
            name: "edge_detection",
            original: Box::new(edge_detection),
            pipeline: pipeline_to_fn(|p| p.edge_detection()),
        },
        Bench {
            name: "identity",
            original: Box::new(identity),
            pipeline: pipeline_to_fn(|p| p.identity()),
        },
        Bench {
            name: "detect_horizontal_lines",
            original: Box::new(detect_horizontal_lines),
            pipeline: pipeline_to_fn(|p| p.detect_horizontal_lines()),
        },
        Bench {
            name: "detect_vertical_lines",
            original: Box::new(detect_vertical_lines),
            pipeline: pipeline_to_fn(|p| p.detect_vertical_lines()),
        },
        Bench {
            name: "detect_45_deg_lines",
            original: Box::new(detect_45_deg_lines),
            pipeline: pipeline_to_fn(|p| p.detect_45_deg_lines()),
        },
        Bench {
            name: "detect_135_deg_lines",
            original: Box::new(detect_135_deg_lines),
            pipeline: pipeline_to_fn(|p| p.detect_135_deg_lines()),
        },
        Bench {
            name: "laplace",
            original: Box::new(laplace),
            pipeline: pipeline_to_fn(|p| p.laplace()),
        },
        Bench {
            name: "edge_one",
            original: Box::new(edge_one),
            pipeline: pipeline_to_fn(|p| p.edge_one()),
        },
        Bench {
            name: "emboss",
            original: Box::new(emboss),
            pipeline: pipeline_to_fn(|p| p.emboss()),
        },
        Bench {
            name: "sobel_horizontal",
            original: Box::new(sobel_horizontal),
            pipeline: pipeline_to_fn(|p| p.sobel_horizontal()),
        },
        Bench {
            name: "prewitt_horizontal",
            original: Box::new(prewitt_horizontal),
            pipeline: pipeline_to_fn(|p| p.prewitt_horizontal()),
        },
        Bench {
            name: "sobel_vertical",
            original: Box::new(sobel_vertical),
            pipeline: pipeline_to_fn(|p| p.sobel_vertical()),
        },
        Bench {
            name: "long_spatial_chain",
            original: Box::new(|img| {
                box_blur(img);
                sharpen(img);
                edge_detection(img);
                sobel_horizontal(img);
                sobel_vertical(img);
                laplace(img);
                emboss(img);
                noise_reduction(img);
                detect_horizontal_lines(img);
                detect_vertical_lines(img);
                detect_45_deg_lines(img);
                detect_135_deg_lines(img);
                edge_one(img);
                prewitt_horizontal(img);
                identity(img);
            }),
            pipeline: pipeline_to_fn(|p| {
                p.box_blur()
                    .sharpen()
                    .edge_detection()
                    .sobel_horizontal()
                    .sobel_vertical()
                    .laplace()
                    .emboss()
                    .noise_reduction()
                    .detect_horizontal_lines()
                    .detect_vertical_lines()
                    .detect_45_deg_lines()
                    .detect_135_deg_lines()
                    .edge_one()
                    .prewitt_horizontal()
                    .identity()
            }),
        },
    ];

    bench_with_config(benches, SPATIAL_BENCH_CONFIG);
}
