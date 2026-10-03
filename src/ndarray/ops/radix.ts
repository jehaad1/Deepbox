/**
 * LSD radix sort for float64 lanes (with optional index payload).
 *
 * Doubles are mapped to order-preserving unsigned 64-bit keys (sign bit
 * flipped for positives, all bits inverted for negatives; NaN forced to the
 * all-ones key so every NaN sorts last regardless of payload/sign, matching
 * TypedArray.prototype.sort and NumPy). Keys are then sorted with four
 * 16-bit counting passes: O(n) versus O(n log n), ~3-4x faster than V8's
 * built-in typed-array sort for large lanes. A pass whose digit is the same
 * for every key (common for float32 data widened to float64) is skipped.
 *
 * Signed zeros: {@link radixSortF64} keeps `-0` and `+0` as they are and orders
 * `-0` first, exactly like `TypedArray.prototype.sort`. {@link radixArgsortF64}
 * treats them as equal and keeps their input order, like NumPy's argsort.
 *
 * The key layout assumes little-endian memory; on a big-endian host both
 * functions fall back to a comparison sort with the same ordering.
 *
 * @internal
 *
 * @see {@link https://deepbox.dev/docs/ndarray-sorting | Deepbox Sorting Functions}
 */

/** Lane length at which radix beats V8's built-in typed sort. */
export const RADIX_SORT_THRESHOLD = 8192;

const RADIX = 65536;

const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;

/** Digit histogram shared by all calls (the sorts are synchronous and not re-entrant). */
const COUNT = new Uint32Array(RADIX);

/**
 * Write the order-preserving key of every value into `words` (two uint32 per
 * value, low word first), which must already hold the raw bits of `values`.
 * With `mergeZeros`, `-0` and `+0` share the `+0` key so they compare equal.
 */
function encodeF64(values: Float64Array, words: Uint32Array, mergeZeros: boolean): void {
  const n = values.length;
  for (let i = 0; i < n; i++) {
    const hi = words[2 * i + 1] as number;
    const lo = words[2 * i] as number;
    const v = values[i] as number;
    if (Number.isNaN(v)) {
      words[2 * i + 1] = 0xffffffff;
      words[2 * i] = 0xffffffff;
    } else if (mergeZeros && v === 0) {
      words[2 * i + 1] = 0x80000000;
      words[2 * i] = 0;
    } else if (hi & 0x80000000) {
      words[2 * i + 1] = ~hi >>> 0;
      words[2 * i] = ~lo >>> 0;
    } else {
      words[2 * i + 1] = (hi | 0x80000000) >>> 0;
    }
  }
}

function decodeF64(words: Uint32Array): void {
  const n = words.length >>> 1;
  for (let i = 0; i < n; i++) {
    const hi = words[2 * i + 1] as number;
    const lo = words[2 * i] as number;
    if (hi & 0x80000000) {
      words[2 * i + 1] = (hi & 0x7fffffff) >>> 0;
    } else {
      words[2 * i + 1] = ~hi >>> 0;
      words[2 * i] = ~lo >>> 0;
    }
  }
}

/**
 * Count the digits of pass `pass` into {@link COUNT} and turn the counts into
 * start offsets. Returns false when every key has the same digit (the pass
 * would not move anything).
 */
function prepareHistogram(src: Uint32Array, n: number, pass: number): boolean {
  const shift = (pass & 1) * 16;
  const word = pass >> 1;
  COUNT.fill(0);
  for (let i = 0; i < n; i++) {
    const digit = ((src[2 * i + word] as number) >>> shift) & 0xffff;
    COUNT[digit] = (COUNT[digit] as number) + 1;
  }
  const firstDigit = ((src[word] as number) >>> shift) & 0xffff;
  if ((COUNT[firstDigit] as number) === n) return false;
  let sum = 0;
  for (let j = 0; j < RADIX; j++) {
    const c = COUNT[j] as number;
    COUNT[j] = sum;
    sum += c;
  }
  return true;
}

/**
 * Sort a Float64Array ascending in place, NaNs last, `-0` before `+0`.
 */
export function radixSortF64(values: Float64Array): void {
  const n = values.length;
  if (n < 2) return;
  if (!LITTLE_ENDIAN) {
    values.sort();
    return;
  }

  const encBuf = new ArrayBuffer(n * 8);
  let src: Uint32Array = new Uint32Array(encBuf);
  let dst: Uint32Array = new Uint32Array(n * 2);
  new Float64Array(encBuf).set(values);
  encodeF64(values, src, false);

  for (let pass = 0; pass < 4; pass++) {
    if (!prepareHistogram(src, n, pass)) continue;
    const shift = (pass & 1) * 16;
    const word = pass >> 1;
    for (let i = 0; i < n; i++) {
      const digit = ((src[2 * i + word] as number) >>> shift) & 0xffff;
      const o = (COUNT[digit] as number)++;
      dst[2 * o] = src[2 * i] as number;
      dst[2 * o + 1] = src[2 * i + 1] as number;
    }
    const tmp = src;
    src = dst;
    dst = tmp;
  }

  decodeF64(src);
  values.set(new Float64Array(src.buffer, 0, n));
}

/**
 * Argsort a Float64Array ascending (NaNs last, stable, `-0` equal to `+0`),
 * writing the sorted order's original indices into `indicesOut` (length >= n).
 */
export function radixArgsortF64(values: Float64Array, indicesOut: Int32Array): void {
  const n = values.length;
  if (n === 0) return;
  if (n === 1) {
    indicesOut[0] = 0;
    return;
  }
  if (!LITTLE_ENDIAN) {
    const idx = Array.from({ length: n }, (_, i) => i);
    idx.sort((a, b) => {
      const x = values[a] as number;
      const y = values[b] as number;
      const xNaN = Number.isNaN(x);
      const yNaN = Number.isNaN(y);
      if (xNaN || yNaN) return xNaN === yNaN ? 0 : xNaN ? 1 : -1;
      return x < y ? -1 : x > y ? 1 : 0;
    });
    indicesOut.set(idx);
    return;
  }

  const encBuf = new ArrayBuffer(n * 8);
  let src: Uint32Array = new Uint32Array(encBuf);
  let dst: Uint32Array = new Uint32Array(n * 2);
  let idxSrc: Int32Array = new Int32Array(n);
  let idxDst: Int32Array = new Int32Array(n);
  new Float64Array(encBuf).set(values);
  encodeF64(values, src, true);
  for (let i = 0; i < n; i++) idxSrc[i] = i;

  for (let pass = 0; pass < 4; pass++) {
    if (!prepareHistogram(src, n, pass)) continue;
    const shift = (pass & 1) * 16;
    const word = pass >> 1;
    for (let i = 0; i < n; i++) {
      const digit = ((src[2 * i + word] as number) >>> shift) & 0xffff;
      const o = (COUNT[digit] as number)++;
      dst[2 * o] = src[2 * i] as number;
      dst[2 * o + 1] = src[2 * i + 1] as number;
      idxDst[o] = idxSrc[i] as number;
    }
    let tmp: Uint32Array | Int32Array = src;
    src = dst;
    dst = tmp as Uint32Array;
    tmp = idxSrc;
    idxSrc = idxDst;
    idxDst = tmp as Int32Array;
  }

  indicesOut.set(idxSrc);
}
