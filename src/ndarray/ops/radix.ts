/**
 * LSD radix sort for float64 lanes (with optional index payload).
 *
 * Doubles are mapped to order-preserving unsigned 64-bit keys (sign bit
 * flipped for positives, all bits inverted for negatives; NaN forced to the
 * all-ones key so every NaN sorts last regardless of payload/sign, matching
 * TypedArray.prototype.sort and NumPy). Keys are then sorted with four
 * 16-bit counting passes — O(n) versus O(n log n), ~3-4x faster than V8's
 * built-in typed-array sort for large lanes.
 *
 * @internal
 *
 * @see {@link https://deepbox.dev/docs/ndarray-sorting | Deepbox Sorting Functions}
 */

/** Lane length at which radix beats V8's built-in typed sort. */
export const RADIX_SORT_THRESHOLD = 8192;

const RADIX = 65536;

function encodeF64(values: Float64Array, words: Uint32Array): void {
  const n = values.length;
  for (let i = 0; i < n; i++) {
    const hi = words[2 * i + 1] as number;
    const lo = words[2 * i] as number;
    const v = values[i] as number;
    if (Number.isNaN(v)) {
      words[2 * i + 1] = 0xffffffff;
      words[2 * i] = 0xffffffff;
    } else if (v === 0) {
      // Normalize -0 and +0 to the +0 key so signed zeros compare equal
      // (matches numpy and the comparator fallback); for argsort the index
      // payload then preserves their input order.
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
 * Sort a Float64Array ascending in place, NaNs last.
 */
export function radixSortF64(values: Float64Array): void {
  const n = values.length;
  if (n < 2) return;

  const encBuf = new ArrayBuffer(n * 8);
  let src: Uint32Array = new Uint32Array(encBuf);
  let dst: Uint32Array = new Uint32Array(n * 2);
  new Float64Array(encBuf).set(values);
  encodeF64(values, src);

  const count = new Uint32Array(RADIX);
  for (let pass = 0; pass < 4; pass++) {
    count.fill(0);
    const shift = (pass & 1) * 16;
    const word = pass >> 1;
    for (let i = 0; i < n; i++) {
      const digit = ((src[2 * i + word] as number) >>> shift) & 0xffff;
      count[digit] = (count[digit] as number) + 1;
    }
    let sum = 0;
    for (let j = 0; j < RADIX; j++) {
      const c = count[j] as number;
      count[j] = sum;
      sum += c;
    }
    for (let i = 0; i < n; i++) {
      const digit = ((src[2 * i + word] as number) >>> shift) & 0xffff;
      const o = (count[digit] as number)++;
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
 * Argsort a Float64Array ascending (NaNs last, stable), writing the sorted
 * order's original indices into `indicesOut` (length n).
 */
export function radixArgsortF64(values: Float64Array, indicesOut: Int32Array): void {
  const n = values.length;
  if (n === 0) return;
  if (n === 1) {
    indicesOut[0] = 0;
    return;
  }

  const encBuf = new ArrayBuffer(n * 8);
  let src: Uint32Array = new Uint32Array(encBuf);
  let dst: Uint32Array = new Uint32Array(n * 2);
  let idxSrc: Int32Array = new Int32Array(n);
  let idxDst: Int32Array = new Int32Array(n);
  new Float64Array(encBuf).set(values);
  encodeF64(values, src);
  for (let i = 0; i < n; i++) idxSrc[i] = i;

  const count = new Uint32Array(RADIX);
  for (let pass = 0; pass < 4; pass++) {
    count.fill(0);
    const shift = (pass & 1) * 16;
    const word = pass >> 1;
    for (let i = 0; i < n; i++) {
      const digit = ((src[2 * i + word] as number) >>> shift) & 0xffff;
      count[digit] = (count[digit] as number) + 1;
    }
    let sum = 0;
    for (let j = 0; j < RADIX; j++) {
      const c = count[j] as number;
      count[j] = sum;
      sum += c;
    }
    for (let i = 0; i < n; i++) {
      const digit = ((src[2 * i + word] as number) >>> shift) & 0xffff;
      const o = (count[digit] as number)++;
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
