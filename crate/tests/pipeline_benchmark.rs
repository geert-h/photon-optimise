// Wasm benchmarks can be run with `wasm-pack test --node --release crate -- --nocapture`
#![cfg(target_arch = "wasm32")]

mod common;

use common::*;
use photon_rs::channels::{alter_channels, invert, swap_channels};
use photon_rs::monochrome::{grayscale, monochrome};
use wasm_bindgen_test::*;

#[wasm_bindgen_test]
fn run_pipeline_benchmarks() {
  let benches = vec![
    // Single operations.
    Bench {
      name: "invert",
      original: Box::new(|img| invert(img)),
      pipeline: pipeline_to_fn(|p| p.invert()),
    },
    Bench {
      name: "monochrome",
      original: Box::new(|img| monochrome(img, 40, 50, 100)),
      pipeline: pipeline_to_fn(|p| p.monochrome(40, 50, 100)),
    },
    Bench {
      name: "swap_channels",
      original: Box::new(|img| swap_channels(img, 0, 2)),
      pipeline: pipeline_to_fn(|p| p.swap_channels(0, 2)),
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
    Bench {
      name: "chain_of_4",
      original: Box::new(|img| {
        grayscale(img);
        alter_channels(img, 10, -20, 30);
        swap_channels(img, 0, 2);
        invert(img);
      }),
      pipeline: pipeline_to_fn(|p| {
        p.grayscale()
          .grayscale()
          .alter_channels(10, -20, 30)
          .swap_channels(0, 2)
          .invert()
      }),
    },
    Bench {
      name: "chain_of_12",
      original: Box::new(|img| {
        grayscale(img);
        invert(img);
        alter_channels(img, 10, -20, 30);
        monochrome(img, 40, 50, 100);
        invert(img);
        alter_channels(img, -5, 25, -10);
        monochrome(img, 0, 20, 40);
        invert(img);
        alter_channels(img, 12, -8, 4);
        monochrome(img, 5, 15, 25);
        invert(img);
        alter_channels(img, -30, 5, 10);
      }),
      pipeline: pipeline_to_fn(|p| {
        p.grayscale()
          .invert()
          .alter_channels(10, -20, 30)
          .monochrome(40, 50, 100)
          .invert()
          .alter_channels(-5, 25, -10)
          .monochrome(0, 20, 40)
          .invert()
          .alter_channels(12, -8, 4)
          .monochrome(5, 15, 25)
          .invert()
          .alter_channels(-30, 5, 10)
      }),
    },
  ];

  bench(benches);
}
