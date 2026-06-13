// wasm package is built with:
// wasm-pack build crate --target nodejs --release --out-dir pkg-node

// usage:
// node scripts/bench_pipeline_wasm_node.js [width] [height] [iterations] [warmups]

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
const rAmount = 10;
const gAmount = -20;
const bAmount = 30;

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

function assertSameBytes(label, left, right) {
    if (left.length !== right.length) {
        throw new Error(`${label}: output lengths differ: ${left.length} != ${right.length}`);
    }

    for (let i = 0; i < left.length; i += 1) {
        if (left[i] !== right[i]) {
            throw new Error(`${label}: outputs differ at byte ${i}: ${left[i]} != ${right[i]}`);
        }
    }
}

function assertSameOutput(input) {
    const original = new photon.PhotonImage(input.slice(), width, height);
    const roundtrip = new photon.PhotonImage(input.slice(), width, height);
    const scalarInvert = new photon.PhotonImage(input.slice(), width, height);
    const pipelineInvert = new photon.PhotonImage(input.slice(), width, height);
    const scalarInvertAlter = new photon.PhotonImage(input.slice(), width, height);
    const pipelineInvertAlter = new photon.PhotonImage(input.slice(), width, height);

    try {
        photon.pipeline_conversion_roundtrip(roundtrip);
        photon.invert(scalarInvert);
        photon.pipeline_invert(pipelineInvert);
        photon.invert(scalarInvertAlter);
        photon.alter_channels(scalarInvertAlter, rAmount, gAmount, bAmount);
        photon.pipeline_invert_alter_channels(
            pipelineInvertAlter,
            rAmount,
            gAmount,
            bAmount,
        );

        const originalBytes = original.get_raw_pixels();
        const roundtripBytes = roundtrip.get_raw_pixels();
        const scalarBytes = scalarInvert.get_raw_pixels();
        const pipelineBytes = pipelineInvert.get_raw_pixels();
        const scalarInvertAlterBytes = scalarInvertAlter.get_raw_pixels();
        const pipelineInvertAlterBytes = pipelineInvertAlter.get_raw_pixels();

        assertSameBytes("conversion roundtrip", originalBytes, roundtripBytes);
        assertSameBytes("pipeline invert", scalarBytes, pipelineBytes);
        assertSameBytes(
            "pipeline invert alter_channels",
            scalarInvertAlterBytes,
            pipelineInvertAlterBytes,
        );

        return {
            roundtrip: checksum(roundtripBytes),
            invert: checksum(pipelineBytes),
            invertAlter: checksum(pipelineInvertAlterBytes),
        };
    } finally {
        original.free();
        roundtrip.free();
        scalarInvert.free();
        pipelineInvert.free();
        scalarInvertAlter.free();
        pipelineInvertAlter.free();
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
        `${name.padEnd(28)} ${msPerIteration.toFixed(3).padStart(8)} ms/iter  ` +
            `${megapixelsPerSecond.toFixed(1).padStart(8)} MP/s  checksum=${lastChecksum}`,
    );
}

const input = makePixels(width, height);
const expected = assertSameOutput(input);

console.log(`image=${width}x${height} iterations=${iterations} warmups=${warmups}`);
console.log(
    `roundtrip, invert, and invert+alter outputs are byte-identical, ` +
        `roundtrip_checksum=${expected.roundtrip}, invert_checksum=${expected.invert}, ` +
        `invert_alter_checksum=${expected.invertAlter}`,
);

bench("scalar_invert", (img) => photon.invert(img), input);
bench("pipeline_conversion", (img) => photon.pipeline_conversion_roundtrip(img), input);
bench("pipeline_invert", (img) => photon.pipeline_invert(img), input);
bench(
    "scalar_invert_alter",
    (img) => {
        photon.invert(img);
        photon.alter_channels(img, rAmount, gAmount, bAmount);
    },
    input,
);
bench(
    "pipeline_invert_alter",
    (img) => photon.pipeline_invert_alter_channels(img, rAmount, gAmount, bAmount),
    input,
);
