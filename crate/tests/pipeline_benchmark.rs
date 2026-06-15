#![cfg(target_arch = "wasm32")]

mod common;

use common::*;
use photon_rs::channels::{alter_channels, invert, swap_channels};
use photon_rs::monochrome::{grayscale, monochrome};
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn run_pipeline_benchmarks() {
  let mut benches = vec![
    // Single operations.
    Bench {
      name: "invert",
      original: Box::new(|img| invert(img)),
      pipeline: pipeline_to_fn(|p| p.invert()),
    },
    // Multiple chained operations.
    Bench {
      name: "chain_of_2",
      original: Box::new(|img| {
        invert(img);
        alter_channels(img, 20, -10, 5);
      }),
      pipeline: pipeline_to_fn(|p| p.invert().alter_channels(20, -10, 5)),
    },
    Bench {
      name: "chain_of_3",
      original: Box::new(|img| {
        grayscale(img);
        invert(img);
        alter_channels(img, 10, -20, 30);
      }),
      pipeline: pipeline_to_fn(|p| p.grayscale().invert().alter_channels(10, -20, 30)),
    },
  ];

  bench(benches);
}
