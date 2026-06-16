#![cfg(target_arch = "wasm32")]
use js_sys::Date;
use photon_rs::pipeline::Pipeline;
use photon_rs::PhotonImage;

// `println!` doesn't work with Wasm so we make a macro for console logging (with web_sys)
macro_rules! log {
    ($($t:tt)*) => (web_sys::console::log_1(&format!($($t)*).into()))
}

const SIZES: &[(u32, u32)] = &[(512, 512), (1280, 720), (1000, 500), (1920, 1080)];
const ITERS: u32 = 500;
const WARMUP: u32 = 200;

pub struct Bench {
  pub name: &'static str,
  pub original: Box<dyn Fn(&mut PhotonImage)>,
  pub pipeline: Box<dyn Fn(&mut PhotonImage)>,
}

// Wrap a `Pipeline` chain (like `|p| p.invert().alter_channels(10, 0, -10)`)
// as a `Fn(&mut PhotonImage)`, so it can be used as `Bench::pipeline`
pub fn pipeline_to_fn(
  build: impl Fn(Pipeline) -> Pipeline + 'static,
) -> Box<dyn Fn(&mut PhotonImage)> {
  Box::new(move |img: &mut PhotonImage| {
    *img = build(Pipeline::from_photon_image(img)).finish();
  })
}

fn synthetic_image(width: u32, height: u32) -> PhotonImage {
  let pixels = (width * height) as usize;
  let mut raw = Vec::with_capacity(pixels * 4);

  for i in 0..pixels {
    let x = (i % width as usize) as u32;
    let y = (i / width as usize) as u32;

    raw.push(((x * 13 + y * 3) & 0xff) as u8);
    raw.push(((x * 5 + y * 11) & 0xff) as u8);
    raw.push(((x * 17 + y * 7) & 0xff) as u8);
    raw.push(255);
  }

  PhotonImage::new(raw, width, height)
}

fn validate_and_measure(bench: &Bench, img: &PhotonImage) -> (f64, f64) {
  // Correctness check: original and pipeline must have the same output
  let mut original_out = img.clone();
  let mut pipeline_out = img.clone();
  (bench.original)(&mut original_out);
  (bench.pipeline)(&mut pipeline_out);
  assert_eq!(
    original_out.get_raw_pixels(),
    pipeline_out.get_raw_pixels(),
    "{}: original/pipeline output mismatch at {}x{}",
    bench.name,
    img.get_width(),
    img.get_height()
  );

  // Benchmark: original vs pipeline

  // Original
  // Not timed, warm-up
  let mut img_clone = img.clone();
  for _ in 0..WARMUP {
    (bench.original)(&mut img_clone);
  }

  // Timed
  let mut img_clone = img.clone();
  let start = Date::now();
  for _ in 0..ITERS {
    (bench.original)(&mut img_clone);
  }
  let original_ms = (Date::now() - start) / ITERS as f64;

  // Pipeline
  // Not timed, warm-up
  let mut img_clone = img.clone();
  for _ in 0..WARMUP {
    (bench.pipeline)(&mut img_clone);
  }

  // Timed
  let mut img_clone = img.clone();
  let start = Date::now();
  for _ in 0..ITERS {
    (bench.pipeline)(&mut img_clone);
  }
  let pipeline_ms = (Date::now() - start) / ITERS as f64;

  (original_ms, pipeline_ms)
}

pub fn bench(benches: Vec<Bench>) {
  let variant = if cfg!(all(target_arch = "wasm32", target_feature = "simd128")) {
    "pipeline variant: SIMD"
  } else {
    "pipeline variant: scalar"
  };
  log!("{}", variant);

  log!("| benchmark\t\t\t| size\t\t| original (ms)\t| pipeline (ms)\t| speedup |");
  log!("|-----------------------------------------------------------------------------------------|");

  for bench in &benches {
    for &(w, h) in SIZES {
      let img = synthetic_image(w, h);
      let (original_ms, pipeline_ms) = validate_and_measure(&bench, &img);
      log!(
        "| {}\t\t\t| {}x{}\t| {:.4}\t| {:.4}\t| {:.2}x   |",
        bench.name,
        w,
        h,
        original_ms,
        pipeline_ms,
        original_ms / pipeline_ms
      );
    }
    log!("|-----------------------------------------------------------------------------------------|");
  }
}
