// This benchmark suite can be run with `wasm-pack test --node --release crate -- --nocapture`
#![cfg(target_arch = "wasm32")]
use js_sys::Date;
use photon_rs::PhotonImage;
use wasm_bindgen_test::*;

// `println!` doesn't work with Wasm so we make a macro for console logging (with web_sys)
macro_rules! log {
    ($($t:tt)*) => (web_sys::console::log_1(&format!($($t)*).into()))
}

// We need to make a fake image because Wasm has no access to the filesystem
fn synthetic_image(w: u32, h: u32) -> PhotonImage {
  let pixels = (0..(w * h * 4)).map(|i| (i % 256) as u8).collect();
  PhotonImage::new(pixels, w, h)
}

fn time_fn(f: &dyn Fn(&mut PhotonImage), img: &PhotonImage, iters: u32) -> f64 {
  // warm-up, not timed
  for _ in 0..3 {
    let mut img_clone = img.clone();
    f(&mut img_clone);
  }
  let start = Date::now();
  for _ in 0..iters {
    let mut img_clone = img.clone();
    f(&mut img_clone);
  }
  (Date::now() - start) / iters as f64
}

struct BenchCase {
  name: &'static str,
  scalar: Box<dyn Fn(&mut PhotonImage)>,
  simd: Box<dyn Fn(&mut PhotonImage)>,
}

const SIZES: &[(u32, u32)] = &[(512, 512), (1280, 720), (1000, 500), (1920, 1080)];
const ITERS: u32 = 50;

#[wasm_bindgen_test]
fn run_benchmark_suite() {
  let cases: Vec<BenchCase> = vec![BenchCase {
    name: "monochrome",
    scalar: Box::new(|img| photon_rs::monochrome::monochrome(img, 10, 20, 30)),
    simd: Box::new(|img| unsafe {
      photon_rs::monochrome::monochrome_simd(img, 10, 20, 30)
    }),
  }];

  log!("| function | size | scalar (ms) | simd (ms) | speedup |");

  for case in &cases {
    for &(w, h) in SIZES {
      let img = synthetic_image(w, h);
      let scalar_ms = time_fn(&case.scalar, &img, ITERS);
      let simd_ms = time_fn(&case.simd, &img, ITERS);
      log!(
        "| {} | {}x{} | {:.4} | {:.4} | {:.2}x |",
        case.name,
        w,
        h,
        scalar_ms,
        simd_ms,
        scalar_ms / simd_ms
      );
    }
  }
}
