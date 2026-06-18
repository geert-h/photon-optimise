#![cfg(target_arch = "wasm32")]
#![allow(dead_code)]
use js_sys::Date;
use photon_rs::pipeline::Pipeline;
use photon_rs::PhotonImage;

// `println!` doesn't work with Wasm so we make a macro for console logging (with web_sys)
macro_rules! log {
    ($($t:tt)*) => (web_sys::console::log_1(&format!($($t)*).into()))
}

pub const DEFAULT_IMAGES: &[(&str, &[u8])] = &[
    ("Lena 512x512", include_bytes!("../assets/lena.png")),
    ("Perlin 512x512", include_bytes!("../assets/512x512.png")),
    ("Perlin 1000x500", include_bytes!("../assets/1000x500.png")),
    ("Perlin 1280x720", include_bytes!("../assets/1280x720.png")),
    (
        "Perlin 1920x1080",
        include_bytes!("../assets/1920x1080.png"),
    ),
];

const DEFAULT_ITERS: u32 = 500;
const DEFAULT_WARMUP: u32 = 200;

pub const DEFAULT_BENCH_CONFIG: BenchConfig = BenchConfig {
    images: DEFAULT_IMAGES,
    iterations: DEFAULT_ITERS,
    warmups: DEFAULT_WARMUP,
};

#[derive(Clone, Copy)]
pub struct BenchConfig {
    pub images: &'static [(&'static str, &'static [u8])],
    pub iterations: u32,
    pub warmups: u32,
}

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

fn load_image(bytes: &[u8]) -> PhotonImage {
    let rgba = image::load_from_memory(bytes)
        .expect("Failed to load Lena image")
        .to_rgba8();

    let (width, height) = rgba.dimensions();
    PhotonImage::new(rgba.into_raw(), width, height)
}

fn validate_and_measure(
    bench: &Bench,
    img: &PhotonImage,
    config: BenchConfig,
) -> (f64, f64) {
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
    for _ in 0..config.warmups {
        (bench.original)(&mut img_clone);
    }

    // Timed
    let mut img_clone = img.clone();
    let start = Date::now();
    for _ in 0..config.iterations {
        (bench.original)(&mut img_clone);
    }
    let original_ms = (Date::now() - start) / config.iterations as f64;

    // Pipeline
    // Not timed, warm-up
    let mut img_clone = img.clone();
    for _ in 0..config.warmups {
        (bench.pipeline)(&mut img_clone);
    }

    // Timed
    let mut img_clone = img.clone();
    let start = Date::now();
    for _ in 0..config.iterations {
        (bench.pipeline)(&mut img_clone);
    }
    let pipeline_ms = (Date::now() - start) / config.iterations as f64;

    (original_ms, pipeline_ms)
}

pub fn bench(benches: Vec<Bench>) {
    bench_with_config(benches, DEFAULT_BENCH_CONFIG);
}

pub fn bench_with_config(benches: Vec<Bench>, config: BenchConfig) {
    let variant = if cfg!(all(target_arch = "wasm32", target_feature = "simd128")) {
        "pipeline variant: SIMD"
    } else {
        "pipeline variant: scalar"
    };
    log!("{}", variant);

    log!(
        "| {:<24} | {:<16} | {:>13} | {:>13} | {:>7} |",
        "benchmark",
        "size",
        "original (ms)",
        "pipeline (ms)",
        "speedup"
    );
    log!("|--------------------------|------------------|---------------|---------------|---------|");

    for bench in &benches {
        for (name, bytes) in config.images {
            let img = load_image(bytes);
            let (original_ms, pipeline_ms) = validate_and_measure(bench, &img, config);
            log!(
                "| {:<24} | {:<16} | {:>13.4} | {:>13.4} | {:>6.2}x |",
                bench.name,
                name,
                original_ms,
                pipeline_ms,
                original_ms / pipeline_ms
            );
        }
        log!("|--------------------------|------------------|---------------|---------------|---------|");
    }
}
