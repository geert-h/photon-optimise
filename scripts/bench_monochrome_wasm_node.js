// wasm package is built with:
// wasm-pack build crate --target nodejs --release --out-dir pkg-node

// usage:
// node scripts/bench_monochrome_wasm_node.js [width] [height] [iterations] [warmups]

"use strict";

const {performance} = require("node:perf_hooks");
const path = require("node:path");

const pkgPath = path.join(__dirname, "..", "crate", "pkg-node", "photon_rs.js");
let photon;

try {
    photon = require(pkgPath);
} catch (error) {
    console.error("Could not load the Node wasm package.");
    console.error("Build it first with:");
    console.error("  wasm-pack build crate --target nodejs --release --out-dir pkg-node");
    throw error;
}

const width = Number(process.argv[2] || 1920);
const height = Number(process.argv[3] || 1080);
const iterations = Number(process.argv[4] || 500);
const warmups = Number(process.argv[5] || 100);
const rOffset = 40;
const gOffset = 50;
const bOffset = 100;

function makePixels(width, height) {
    const pixels = new Uint8Array(width * height * 4);

    for (let i = 0, p = 0; i < pixels.length; i += 4, p += 1) {
        pixels[i] = (p * 13 + 17) & 0xff;
        pixels[i + 1] = (p * 7 + 53) & 0xff;
        pixels[i + 2] = (p * 3 + 101) & 0xff;
        pixels[i + 3] = 255;
    }

    return pixels;
}

function checksum(bytes) {
    let hash = 2166136261;

    for (let i = 0; i < bytes.length; i += 1) {
        hash ^= bytes[i];
        hash = Math.imul(hash, 16777619);
    }

    return hash >>> 0;
}

function assertSameOutput(input) {
    const scalar = new photon.PhotonImage(input.slice(), width, height);
    const dispatch = new photon.PhotonImage(input.slice(), width, height);

    try {
        photon.monochrome(scalar, rOffset, gOffset, bOffset);
        photon.monochrome_simd(dispatch, rOffset, gOffset, bOffset);

        const scalarBytes = scalar.get_raw_pixels();
        const dispatchBytes = dispatch.get_raw_pixels();

        if (scalarBytes.length !== dispatchBytes.length) {
            throw new Error(`Output lengths differ: ${scalarBytes.length} != ${dispatchBytes.length}`);
        }

        for (let i = 0; i < scalarBytes.length; i += 1) {
            if (scalarBytes[i] !== dispatchBytes[i]) {
                throw new Error(
                    `Outputs differ at byte ${i}: scalar=${scalarBytes[i]}, dispatch=${dispatchBytes[i]}`,
                );
            }
        }

        return checksum(dispatchBytes);
    } finally {
        scalar.free();
        dispatch.free();
    }
}

function bench(name, fn, input) {
    for (let i = 0; i < warmups; i += 1) {
        const img = new photon.PhotonImage(input.slice(), width, height);
        try {
            fn(img);
        } finally {
            img.free();
        }
    }

    const start = performance.now();
    let lastChecksum = 0;

    for (let i = 0; i < iterations; i += 1) {
        const img = new photon.PhotonImage(input.slice(), width, height);
        try {
            fn(img);

            if (i === iterations - 1) {
                lastChecksum = checksum(img.get_raw_pixels());
            }
        } finally {
            img.free();
        }
    }

    const elapsed = performance.now() - start;
    const msPerIteration = elapsed / iterations;
    const megapixelsPerSecond = (width * height) / (msPerIteration * 1000);

    console.log(
        `${name.padEnd(10)} ${msPerIteration.toFixed(3).padStart(8)} ms/iter  ` +
        `${megapixelsPerSecond.toFixed(1).padStart(8)} MP/s  checksum=${lastChecksum}`,
    );
}

const input = makePixels(width, height);
const expectedChecksum = assertSameOutput(input);

console.log(`image=${width}x${height} iterations=${iterations} warmups=${warmups}`);
console.log(`scalar and dispatch outputs are byte-identical, checksum=${expectedChecksum}`);

bench(
    "scalar",
    (img) => photon.monochrome(img, rOffset, gOffset, bOffset),
    input,
);
bench("dispatch", (img) => photon.monochrome_simd(img, rOffset, gOffset, bOffset), input);
