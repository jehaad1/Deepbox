/**
 * Dependency-free PNG encoder (8-bit RGBA, no interlacing).
 *
 * Uses `node:zlib` for compression when running under Node.js and falls back to stored
 * (uncompressed) deflate blocks elsewhere.
 *
 * @module plot/renderers/png
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
import { assertPositiveInt } from "../utils/validation";

/** Largest width or height the PNG format allows (2^31 - 1). */
const PNG_MAX_DIMENSION = 0x7fffffff;

function isNodeEnvironment(): boolean {
  return (
    typeof process !== "undefined" &&
    typeof process.versions !== "undefined" &&
    typeof process.versions.node !== "undefined"
  );
}

function u32be(n: number): Uint8Array {
  return Uint8Array.of((n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255);
}

function ascii4(s: string): Uint8Array {
  return Uint8Array.of(
    s.charCodeAt(0) & 255,
    s.charCodeAt(1) & 255,
    s.charCodeAt(2) & 255,
    s.charCodeAt(3) & 255
  );
}

function concatBytes(chunks: readonly Uint8Array[]): Uint8Array {
  let total = 0;
  for (const c of chunks) total += c.length;
  const out = new Uint8Array(total);
  let o = 0;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}

const CRC_TABLE: Uint32Array = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = (c & 1) !== 0 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

/** CRC-32 (IEEE 802.3) as used by PNG chunks, over the concatenation of `parts`. */
function crc32(...parts: Uint8Array[]): number {
  let crc = 0xffffffff;
  for (const buf of parts) {
    for (let i = 0; i < buf.length; i++) {
      crc = (CRC_TABLE[(crc ^ (buf[i] ?? 0)) & 0xff] ?? 0) ^ (crc >>> 8);
    }
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunk(type: string, data: Uint8Array): Uint8Array {
  const t = ascii4(type);
  const len = u32be(data.length);
  const crc = u32be(crc32(t, data));
  return concatBytes([len, t, data, crc]);
}

/** Adler-32 checksum, reducing modulo 65521 once per 5552 bytes instead of per byte. */
function adler32(data: Uint8Array): number {
  let a = 1;
  let b = 0;
  for (let start = 0; start < data.length; start += 5552) {
    const end = Math.min(start + 5552, data.length);
    for (let i = start; i < end; i++) {
      a += data[i] ?? 0;
      b += a;
    }
    a %= 65521;
    b %= 65521;
  }
  return ((b << 16) | a) >>> 0;
}

/** Zlib stream made of stored (uncompressed) deflate blocks. */
function deflateUncompressed(data: Uint8Array): Uint8Array {
  const maxBlockSize = 65535;
  // An empty input still needs one final (empty) block to be a valid stream.
  const numBlocks = Math.max(1, Math.ceil(data.length / maxBlockSize));

  let totalSize = 0;
  for (let i = 0; i < numBlocks; i++) {
    const blockStart = i * maxBlockSize;
    const blockSize = Math.min(maxBlockSize, data.length - blockStart);
    totalSize += 5 + blockSize;
  }

  const output = new Uint8Array(2 + totalSize + 4);
  let outPos = 0;

  output[outPos++] = 0x78;
  output[outPos++] = 0x01;

  for (let i = 0; i < numBlocks; i++) {
    const blockStart = i * maxBlockSize;
    const blockSize = Math.min(maxBlockSize, data.length - blockStart);
    const isLast = i === numBlocks - 1;

    output[outPos++] = isLast ? 0x01 : 0x00;

    output[outPos++] = blockSize & 0xff;
    output[outPos++] = (blockSize >> 8) & 0xff;

    const nlen = ~blockSize & 0xffff;
    output[outPos++] = nlen & 0xff;
    output[outPos++] = (nlen >> 8) & 0xff;

    output.set(data.subarray(blockStart, blockStart + blockSize), outPos);
    outPos += blockSize;
  }

  const adler = adler32(data);

  output[outPos++] = (adler >>> 24) & 0xff;
  output[outPos++] = (adler >>> 16) & 0xff;
  output[outPos++] = (adler >>> 8) & 0xff;
  output[outPos++] = adler & 0xff;

  return output;
}

/**
 * Encodes non-premultiplied 8-bit RGBA pixels (row-major, top row first) as a PNG file.
 * @param width - Image width in pixels (1 to 2^31 - 1)
 * @param height - Image height in pixels (1 to 2^31 - 1)
 * @param rgba - Pixel data, exactly `width * height * 4` bytes
 * @throws {InvalidParameterError} If a dimension is not a positive integer within the PNG limit
 *   or `rgba` has the wrong length.
 * @internal
 */
export async function pngEncodeRGBA(
  width: number,
  height: number,
  rgba: Uint8ClampedArray
): Promise<Uint8Array> {
  assertPositiveInt("width", width);
  assertPositiveInt("height", height);
  if (width > PNG_MAX_DIMENSION || height > PNG_MAX_DIMENSION) {
    throw new InvalidParameterError(
      `PNG dimensions must not exceed ${PNG_MAX_DIMENSION}; received ${width}x${height}`,
      "width/height",
      { width, height }
    );
  }
  if (rgba.length !== width * height * 4)
    throw new InvalidParameterError("RGBA buffer has incorrect length", "rgba", rgba.length);

  const signature = Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10);
  const ihdrData = concatBytes([u32be(width), u32be(height), Uint8Array.of(8, 6, 0, 0, 0)]);
  const ihdr = pngChunk("IHDR", ihdrData);

  const stride = width * 4;
  const raw = new Uint8Array(height * (1 + stride));
  for (let y = 0; y < height; y++) {
    const rowOff = y * (1 + stride);
    raw[rowOff] = 0;
    raw.set(rgba.subarray(y * stride, y * stride + stride), rowOff + 1);
  }

  let compressed: Uint8Array;
  if (isNodeEnvironment()) {
    try {
      const zlib = await import("node:zlib");
      compressed =
        typeof zlib.deflateSync === "function"
          ? zlib.deflateSync(raw, { level: 9 })
          : deflateUncompressed(raw);
    } catch {
      compressed = deflateUncompressed(raw);
    }
  } else {
    compressed = deflateUncompressed(raw);
  }

  const idat = pngChunk("IDAT", compressed);
  const iend = pngChunk("IEND", new Uint8Array(0));
  return concatBytes([signature, ihdr, idat, iend]);
}

/**
 * Checks if PNG is supported.
 * @internal
 */
export function isPNGSupported(): boolean {
  return isNodeEnvironment();
}

/**
 * Checks if Node environment.
 * @internal
 */
export function isNodeEnvironment_export(): boolean {
  return isNodeEnvironment();
}
