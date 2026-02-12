import { InvalidParameterError } from "../../core";
import { assertPositiveInt } from "../utils/validation";

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

function crc32(buf: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < buf.length; i++) {
    let x = (crc ^ (buf[i] ?? 0)) & 0xff;
    for (let k = 0; k < 8; k++) {
      x = (x & 1) !== 0 ? 0xedb88320 ^ (x >>> 1) : x >>> 1;
    }
    crc = (crc >>> 8) ^ x;
  }
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunk(type: string, data: Uint8Array): Uint8Array {
  const t = ascii4(type);
  const len = u32be(data.length);
  const crc = u32be(crc32(concatBytes([t, data])));
  return concatBytes([len, t, data, crc]);
}

function deflateUncompressed(data: Uint8Array): Uint8Array {
  const maxBlockSize = 65535;
  const numBlocks = Math.ceil(data.length / maxBlockSize);

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

  let a = 1;
  let b = 0;
  for (let i = 0; i < data.length; i++) {
    a = (a + (data[i] ?? 0)) % 65521;
    b = (b + a) % 65521;
  }
  const adler32 = ((b << 16) | a) >>> 0;

  output[outPos++] = (adler32 >>> 24) & 0xff;
  output[outPos++] = (adler32 >>> 16) & 0xff;
  output[outPos++] = (adler32 >>> 8) & 0xff;
  output[outPos++] = adler32 & 0xff;

  return output;
}

/**
 * Encodes RGBA to PNG.
 * @internal
 */
export async function pngEncodeRGBA(
  width: number,
  height: number,
  rgba: Uint8ClampedArray
): Promise<Uint8Array> {
  assertPositiveInt("width", width);
  assertPositiveInt("height", height);
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
