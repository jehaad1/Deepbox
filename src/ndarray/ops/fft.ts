/**
 * Fast Fourier Transform operations.
 *
 * Implements an iterative radix-2 Cooley-Tukey FFT with Bluestein's chirp-z
 * algorithm for lengths that are not a power of two. Twiddle factors are
 * computed directly (not by repeated multiplication), so the error stays at
 * a few ulps for long transforms. JavaScript has no native complex type, so
 * every transform returns a pair of tensors (real part, imaginary part).
 *
 * Dtype rules follow NumPy: float32, float16 and bfloat16 inputs give float32
 * results; every other numeric dtype (float64, int32, int64, uint8, bool, ...)
 * gives float64 results. Arithmetic is always carried out in float64.
 *
 * @module ndarray/ops/fft
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import type { DType } from "../../core";
import { DTypeError, InvalidParameterError, normalizeAxis, ShapeError } from "../../core";
import { Tensor } from "../tensor/Tensor";
import { readNumbers } from "./_internal";
import { clone, roll } from "./utils";

/** Result of an FFT operation: real and imaginary parts as separate tensors. */
export interface FFTResult {
  real: Tensor;
  imag: Tensor;
}

/**
 * Normalization mode of an FFT, matching NumPy's `norm` argument.
 *
 * - `"backward"` (default): the forward transform is unscaled and the inverse
 *   is scaled by `1/n`.
 * - `"ortho"`: both directions are scaled by `1/sqrt(n)` (unitary transform).
 * - `"forward"`: the forward transform is scaled by `1/n` and the inverse is
 *   unscaled.
 */
export type FFTNorm = "backward" | "ortho" | "forward";

// ---- Plans ----

/** Precomputed tables for a radix-2 transform of one power-of-two length. */
interface Radix2Plan {
  readonly size: number;
  /** cos(2*pi*k/size) for k < size/2. */
  readonly cos: Float64Array;
  /** -sin(2*pi*k/size) for k < size/2 (the imaginary part of the forward twiddle). */
  readonly sin: Float64Array;
}

/** Precomputed tables for Bluestein's algorithm at one length. */
interface BluesteinPlan {
  readonly n: number;
  readonly m: number;
  readonly radix: Radix2Plan;
  /** Chirp w_j = exp(-i*pi*j^2/n). */
  readonly wRe: Float64Array;
  readonly wIm: Float64Array;
  /** Forward FFT (length m) of the zero-padded, wrapped conjugate chirp. */
  readonly bRe: Float64Array;
  readonly bIm: Float64Array;
  /** Scratch buffers, reused between calls (all transforms here are synchronous). */
  readonly aRe: Float64Array;
  readonly aIm: Float64Array;
}

type Plan =
  | { readonly kind: "trivial" }
  | { readonly kind: "radix2"; readonly plan: Radix2Plan }
  | { readonly kind: "bluestein"; readonly plan: BluesteinPlan };

/** Plans are cached by length; the cache holds at most this many bytes of tables. */
const PLAN_CACHE_MAX_BYTES = 32 * 1024 * 1024;
const planCache = new Map<number, { plan: Plan; bytes: number }>();
let planCacheBytes = 0;

function isPowerOf2(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

function nextPowerOf2(n: number): number {
  let p = 1;
  while (p < n) p *= 2;
  return p;
}

function makeRadix2Plan(size: number): Radix2Plan {
  const half = size >> 1;
  const cos = new Float64Array(half);
  const sin = new Float64Array(half);
  for (let k = 0; k < half; k++) {
    if (k === 0) {
      cos[k] = 1;
      sin[k] = 0;
    } else if (4 * k === size) {
      // Exact quarter turn: cos(pi/2) in floating point is 6e-17, not 0.
      cos[k] = 0;
      sin[k] = -1;
    } else {
      const angle = (2 * Math.PI * k) / size;
      cos[k] = Math.cos(angle);
      sin[k] = -Math.sin(angle);
    }
  }
  return { size, cos, sin };
}

function makeBluesteinPlan(n: number): BluesteinPlan {
  const m = nextPowerOf2(2 * n - 1);
  const radix = makeRadix2Plan(m);
  const wRe = new Float64Array(n);
  const wIm = new Float64Array(n);
  // j^2 mod 2n, advanced incrementally so it stays exact for any n.
  const twoN = 2 * n;
  let sq = 0;
  for (let j = 0; j < n; j++) {
    const angle = (Math.PI * sq) / n;
    wRe[j] = Math.cos(angle);
    wIm[j] = -Math.sin(angle);
    sq = (sq + 2 * j + 1) % twoN;
  }
  const bRe = new Float64Array(m);
  const bIm = new Float64Array(m);
  bRe[0] = wRe[0] as number;
  bIm[0] = -(wIm[0] as number);
  for (let j = 1; j < n; j++) {
    const cr = wRe[j] as number;
    const ci = -(wIm[j] as number);
    bRe[j] = cr;
    bIm[j] = ci;
    bRe[m - j] = cr;
    bIm[m - j] = ci;
  }
  radix2Forward(bRe, bIm, radix);
  return {
    n,
    m,
    radix,
    wRe,
    wIm,
    bRe,
    bIm,
    aRe: new Float64Array(m),
    aIm: new Float64Array(m),
  };
}

/** Approximate memory held by a plan's tables, in bytes. */
function planBytes(plan: Plan): number {
  if (plan.kind === "trivial") return 0;
  if (plan.kind === "radix2") return plan.plan.size * 8;
  const { n, m } = plan.plan;
  return m * 8 + 2 * n * 8 + 4 * m * 8;
}

function getPlan(n: number): Plan {
  if (n <= 1) return { kind: "trivial" };
  const cached = planCache.get(n);
  if (cached) return cached.plan;
  const plan: Plan = isPowerOf2(n)
    ? { kind: "radix2", plan: makeRadix2Plan(n) }
    : { kind: "bluestein", plan: makeBluesteinPlan(n) };
  const bytes = planBytes(plan);
  if (bytes <= PLAN_CACHE_MAX_BYTES / 2) {
    // Evict the oldest plans until the new one fits.
    for (const [key, entry] of planCache) {
      if (planCacheBytes + bytes <= PLAN_CACHE_MAX_BYTES) break;
      planCache.delete(key);
      planCacheBytes -= entry.bytes;
    }
    planCache.set(n, { plan, bytes });
    planCacheBytes += bytes;
  }
  return plan;
}

// ---- 1-D kernels (forward direction, unscaled, in place) ----

function radix2Forward(re: Float64Array, im: Float64Array, plan: Radix2Plan): void {
  const n = plan.size;
  const cosT = plan.cos;
  const sinT = plan.sin;

  // Bit-reversal permutation.
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      const tr = re[i] as number;
      const ti = im[i] as number;
      re[i] = re[j] as number;
      im[i] = im[j] as number;
      re[j] = tr;
      im[j] = ti;
    }
    let m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const step = n / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0, tw = 0; k < halfLen; k++, tw += step) {
        const wr = cosT[tw] as number;
        const wi = sinT[tw] as number;
        const even = i + k;
        const odd = even + halfLen;
        const oRe = re[odd] as number;
        const oIm = im[odd] as number;
        const tRe = wr * oRe - wi * oIm;
        const tIm = wr * oIm + wi * oRe;
        const eRe = re[even] as number;
        const eIm = im[even] as number;
        re[even] = eRe + tRe;
        im[even] = eIm + tIm;
        re[odd] = eRe - tRe;
        im[odd] = eIm - tIm;
      }
    }
  }
}

/** Bluestein chirp-z transform of the first `plan.n` entries of `re`/`im`, in place. */
function bluesteinForward(re: Float64Array, im: Float64Array, plan: BluesteinPlan): void {
  const { n, m, radix, wRe, wIm, bRe, bIm, aRe, aIm } = plan;

  // a_j = x_j * w_j, zero padded to length m.
  for (let j = 0; j < n; j++) {
    const xr = re[j] as number;
    const xi = im[j] as number;
    const wr = wRe[j] as number;
    const wi = wIm[j] as number;
    aRe[j] = xr * wr - xi * wi;
    aIm[j] = xr * wi + xi * wr;
  }
  aRe.fill(0, n);
  aIm.fill(0, n);

  radix2Forward(aRe, aIm, radix);

  // Pointwise product with FFT(b); store the conjugate so that a forward
  // transform computes the inverse (conj(FFT(conj(x))) = unscaled IFFT(x)).
  for (let k = 0; k < m; k++) {
    const ar = aRe[k] as number;
    const ai = aIm[k] as number;
    const br = bRe[k] as number;
    const bi = bIm[k] as number;
    aRe[k] = ar * br - ai * bi;
    aIm[k] = -(ar * bi + ai * br);
  }

  radix2Forward(aRe, aIm, radix);

  // c_k = conj(result) / m, then X_k = w_k * c_k.
  const invM = 1 / m;
  for (let k = 0; k < n; k++) {
    const cr = (aRe[k] as number) * invM;
    const ci = -(aIm[k] as number) * invM;
    const wr = wRe[k] as number;
    const wi = wIm[k] as number;
    re[k] = cr * wr - ci * wi;
    im[k] = cr * wi + ci * wr;
  }
}

/** In-place 1-D transform of `re`/`im` (length n), unscaled. */
function transform1d(re: Float64Array, im: Float64Array, plan: Plan, inverse: boolean): void {
  if (plan.kind === "trivial") return;
  const n = re.length;
  if (inverse) {
    for (let i = 0; i < n; i++) im[i] = -(im[i] as number);
  }
  if (plan.kind === "radix2") radix2Forward(re, im, plan.plan);
  else bluesteinForward(re, im, plan.plan);
  if (inverse) {
    for (let i = 0; i < n; i++) im[i] = -(im[i] as number);
  }
}

// ---- Tensor plumbing ----

interface Spectrum {
  re: Float64Array;
  im: Float64Array;
  shape: number[];
}

const FFT_NORMS: readonly FFTNorm[] = ["backward", "ortho", "forward"];

function checkNorm(norm: FFTNorm): void {
  if (!FFT_NORMS.includes(norm)) {
    throw new InvalidParameterError(
      `norm must be one of "backward", "ortho" or "forward"; received ${String(norm)}`,
      "norm",
      norm
    );
  }
}

function requireNumeric(t: Tensor, op: string, param: string): void {
  if (t.dtype === "string") {
    throw new DTypeError(`${op} requires numeric input; ${param} has string dtype`);
  }
}

/** Logical row-major values of `t` as float64 (may alias the tensor buffer; never written). */
function toFloat64(t: Tensor, op: string): Float64Array {
  const src = readNumbers(t, op, false);
  return src instanceof Float64Array ? src : new Float64Array(src);
}

/** Output dtype: float32 for single/half precision inputs, float64 otherwise. */
function fftDtype(...dtypes: DType[]): "float32" | "float64" {
  for (const d of dtypes) {
    if (d !== "float32" && d !== "float16" && d !== "bfloat16" && d !== "complex64") {
      return "float64";
    }
  }
  return "float32";
}

function makeTensor(data: Float64Array, shape: number[], dtype: "float32" | "float64"): Tensor {
  return Tensor.fromTypedArray({
    data: dtype === "float32" ? new Float32Array(data) : data,
    shape,
    dtype,
    device: "cpu",
  });
}

function makeResult(spec: Spectrum, dtype: "float32" | "float64"): FFTResult {
  return {
    real: makeTensor(spec.re, spec.shape, dtype),
    imag: makeTensor(spec.im, spec.shape, dtype),
  };
}

function validateLength(n: number, param: string, hint?: string): void {
  if (!Number.isInteger(n) || n <= 0) {
    throw new InvalidParameterError(
      `n must be a positive integer; received ${String(n)}${hint ? ` (${hint})` : ""}`,
      param,
      n
    );
  }
}

function requireSameShape(real: Tensor, imag: Tensor, op: string): void {
  if (real.ndim !== imag.ndim || real.shape.some((d, i) => d !== imag.shape[i])) {
    throw ShapeError.mismatch(real.shape, imag.shape, `${op}: real and imag parts`);
  }
}

function scaleFor(norm: FFTNorm, n: number, inverse: boolean): number {
  if (norm === "ortho") return 1 / Math.sqrt(n);
  const scaled: FFTNorm = inverse ? "backward" : "forward";
  return norm === scaled ? 1 / n : 1;
}

/** Product of `shape[from..to)`. */
function span(shape: readonly number[], from: number, to: number): number {
  let p = 1;
  for (let d = from; d < to; d++) p *= shape[d] as number;
  return p;
}

/**
 * Transform every line of `shape` along `axis`, cropping or zero-padding each
 * line to `outLen`, and scale the result. Input arrays are never modified.
 */
function transformAxis(
  re: Float64Array,
  im: Float64Array | null,
  shape: readonly number[],
  axis: number,
  outLen: number,
  inverse: boolean,
  norm: FFTNorm
): Spectrum {
  const len = shape[axis] as number;
  const outer = span(shape, 0, axis);
  const inner = span(shape, axis + 1, shape.length);
  const outShape = shape.slice();
  outShape[axis] = outLen;
  const total = outer * outLen * inner;
  const outRe = new Float64Array(total);
  const outIm = new Float64Array(total);
  if (total === 0) return { re: outRe, im: outIm, shape: outShape };

  const plan = getPlan(outLen);
  const scale = scaleFor(norm, outLen, inverse);
  const copyLen = Math.min(len, outLen);
  const lr = new Float64Array(outLen);
  const li = new Float64Array(outLen);

  for (let o = 0; o < outer; o++) {
    const inBase = o * len * inner;
    const outBase = o * outLen * inner;
    for (let i = 0; i < inner; i++) {
      let p = inBase + i;
      for (let j = 0; j < copyLen; j++, p += inner) {
        lr[j] = re[p] as number;
        li[j] = im ? (im[p] as number) : 0;
      }
      if (copyLen < outLen) {
        lr.fill(0, copyLen);
        li.fill(0, copyLen);
      }
      transform1d(lr, li, plan, inverse);
      let q = outBase + i;
      for (let j = 0; j < outLen; j++, q += inner) {
        outRe[q] = (lr[j] as number) * scale;
        outIm[q] = (li[j] as number) * scale;
      }
    }
  }
  return { re: outRe, im: outIm, shape: outShape };
}

/** Keep the first `count` entries along `axis`. */
function cropAxis(spec: Spectrum, axis: number, count: number): Spectrum {
  const len = spec.shape[axis] as number;
  if (count === len) return spec;
  const outer = span(spec.shape, 0, axis);
  const inner = span(spec.shape, axis + 1, spec.shape.length);
  const outShape = spec.shape.slice();
  outShape[axis] = count;
  const re = new Float64Array(outer * count * inner);
  const im = new Float64Array(outer * count * inner);
  const run = count * inner;
  for (let o = 0; o < outer; o++) {
    const from = o * len * inner;
    re.set(spec.re.subarray(from, from + run), o * run);
    im.set(spec.im.subarray(from, from + run), o * run);
  }
  return { re, im, shape: outShape };
}

/**
 * Build the full length-`n` Hermitian spectrum along `axis` from a one-sided
 * spectrum. Like NumPy, the imaginary parts of the DC term and (for even `n`)
 * the Nyquist term are ignored.
 */
function hermitianExtend(
  re: Float64Array,
  im: Float64Array,
  shape: readonly number[],
  axis: number,
  n: number
): Spectrum {
  const half = shape[axis] as number;
  const outer = span(shape, 0, axis);
  const inner = span(shape, axis + 1, shape.length);
  const outShape = shape.slice();
  outShape[axis] = n;
  const outRe = new Float64Array(outer * n * inner);
  const outIm = new Float64Array(outer * n * inner);
  const top = Math.floor(n / 2);
  for (let o = 0; o < outer; o++) {
    const inBase = o * half * inner;
    const outBase = o * n * inner;
    for (let i = 0; i < inner; i++) {
      for (let k = 0; k <= top && k < half; k++) {
        const src = inBase + k * inner + i;
        const dst = outBase + k * inner + i;
        const r = re[src] as number;
        const isReal = k === 0 || 2 * k === n;
        const v = isReal ? 0 : (im[src] as number);
        outRe[dst] = r;
        outIm[dst] = v;
        if (!isReal) {
          const mirror = outBase + (n - k) * inner + i;
          outRe[mirror] = r;
          outIm[mirror] = -v;
        }
      }
    }
  }
  return { re: outRe, im: outIm, shape: outShape };
}

function resolveAxes(axes: readonly number[] | undefined, ndim: number): number[] {
  if (axes === undefined) return Array.from({ length: ndim }, (_, i) => i);
  return axes.map((ax) => normalizeAxis(ax, ndim));
}

function axisLength(shape: readonly number[], axis: number): number {
  const len = shape[axis] as number;
  if (len === 0) {
    throw new InvalidParameterError(
      `Invalid number of FFT data points (0) along axis ${axis}`,
      "axes",
      axis
    );
  }
  return len;
}

// ---- Public API ----

/**
 * Compute the one-dimensional discrete Fourier transform along one axis.
 *
 * @param t - Real-valued input tensor.
 * @param n - Length of the transformed axis. A shorter input is zero-padded and
 *   a longer one is cropped. Default: the input length along `axis`.
 * @param axis - Axis to transform (negative values count from the end). Default: -1.
 * @param norm - Normalization mode. Default: `"backward"`.
 * @returns Object with `real` and `imag` tensors; the transformed axis has length `n`.
 * @throws {DTypeError} If `t` has string dtype.
 * @throws {ShapeError} If `t` is 0-D.
 * @throws {InvalidParameterError} If `n` is not a positive integer or `axis` is invalid.
 *
 * @example
 * ```ts
 * import { fft, tensor } from "deepbox/ndarray";
 *
 * const { real, imag } = fft(tensor([1, 0, 0, 0]));
 * // real = [1, 1, 1, 1], imag = [0, 0, 0, 0]
 * ```
 */
export function fft(t: Tensor, n?: number, axis = -1, norm: FFTNorm = "backward"): FFTResult {
  requireNumeric(t, "fft", "t");
  if (t.ndim === 0) throw new ShapeError("fft requires at least 1D input");
  checkNorm(norm);
  const ax = normalizeAxis(axis, t.ndim);
  const len = n ?? axisLength(t.shape, ax);
  validateLength(len, "n");
  const spec = transformAxis(toFloat64(t, "fft"), null, t.shape, ax, len, false, norm);
  return makeResult(spec, fftDtype(t.dtype));
}

/**
 * Compute the one-dimensional inverse discrete Fourier transform along one axis.
 *
 * @param real - Real part of the input spectrum.
 * @param imag - Imaginary part of the input spectrum (same shape as `real`).
 * @param n - Length of the transformed axis. Default: the input length along `axis`.
 * @param axis - Axis to transform. Default: -1.
 * @param norm - Normalization mode. Default: `"backward"` (the inverse is scaled by `1/n`).
 * @returns Object with `real` and `imag` tensors.
 * @throws {DTypeError} If an input has string dtype.
 * @throws {ShapeError} If an input is 0-D or the two parts differ in shape.
 *
 * @example
 * ```ts
 * const spectrum = fft(tensor([1, 2, 3, 4]));
 * const { real } = ifft(spectrum.real, spectrum.imag);
 * // real is approximately [1, 2, 3, 4]
 * ```
 */
export function ifft(
  real: Tensor,
  imag: Tensor,
  n?: number,
  axis = -1,
  norm: FFTNorm = "backward"
): FFTResult {
  requireNumeric(real, "ifft", "real");
  requireNumeric(imag, "ifft", "imag");
  if (real.ndim === 0 || imag.ndim === 0) throw new ShapeError("ifft requires at least 1D input");
  requireSameShape(real, imag, "ifft");
  checkNorm(norm);
  const ax = normalizeAxis(axis, real.ndim);
  const len = n ?? axisLength(real.shape, ax);
  validateLength(len, "n");
  const spec = transformAxis(
    toFloat64(real, "ifft"),
    toFloat64(imag, "ifft"),
    real.shape,
    ax,
    len,
    true,
    norm
  );
  return makeResult(spec, fftDtype(real.dtype, imag.dtype));
}

/**
 * Compute the one-dimensional FFT of real input, keeping only the
 * non-negative frequencies.
 *
 * The spectrum of a real signal is Hermitian-symmetric, so the transformed
 * axis has length `floor(n / 2) + 1`.
 *
 * @param t - Real-valued input tensor.
 * @param n - Length of the underlying FFT. Default: the input length along `axis`.
 * @param axis - Axis to transform. Default: -1.
 * @param norm - Normalization mode. Default: `"backward"`.
 * @returns Object with `real` and `imag` tensors of shape `[..., floor(n/2) + 1]`
 *   (along `axis`).
 *
 * @example
 * ```ts
 * const { real, imag } = rfft(tensor([1, 2, 3, 4]));
 * // real = [10, -2, -2], imag = [0, 2, 0]
 * ```
 */
export function rfft(t: Tensor, n?: number, axis = -1, norm: FFTNorm = "backward"): FFTResult {
  requireNumeric(t, "rfft", "t");
  if (t.ndim === 0) throw new ShapeError("rfft requires at least 1D input");
  checkNorm(norm);
  const ax = normalizeAxis(axis, t.ndim);
  const len = n ?? axisLength(t.shape, ax);
  validateLength(len, "n");
  const full = transformAxis(toFloat64(t, "rfft"), null, t.shape, ax, len, false, norm);
  return makeResult(cropAxis(full, ax, Math.floor(len / 2) + 1), fftDtype(t.dtype));
}

/**
 * Compute the inverse of {@link rfft}: a real signal from a one-sided spectrum.
 *
 * The imaginary parts of the DC term and, for even `n`, the Nyquist term are
 * ignored, as in NumPy. The returned `imag` tensor is all zeros; it is kept so
 * the result has the same `{ real, imag }` form as the other transforms.
 *
 * @param real - Real part of the one-sided spectrum (length `floor(n/2) + 1` along `axis`).
 * @param imag - Imaginary part of the one-sided spectrum (same shape as `real`).
 * @param n - Length of the output axis. Default: `2 * (m - 1)` for an input length `m`.
 * @param axis - Axis to transform. Default: -1.
 * @param norm - Normalization mode. Default: `"backward"`.
 * @returns Object with `real` (the signal) and a zero `imag` tensor of shape `[..., n]`.
 *
 * @example
 * ```ts
 * const { real, imag } = rfft(tensor([1, 2, 3, 4]));
 * irfft(real, imag).real;  // [1, 2, 3, 4]
 * ```
 */
export function irfft(
  real: Tensor,
  imag: Tensor,
  n?: number,
  axis = -1,
  norm: FFTNorm = "backward"
): FFTResult {
  requireNumeric(real, "irfft", "real");
  requireNumeric(imag, "irfft", "imag");
  if (real.ndim === 0 || imag.ndim === 0) throw new ShapeError("irfft requires at least 1D input");
  requireSameShape(real, imag, "irfft");
  checkNorm(norm);
  const ax = normalizeAxis(axis, real.ndim);
  const half = real.shape[ax] as number;
  const len = n ?? 2 * (half - 1);
  validateLength(
    len,
    "n",
    n === undefined ? `the default 2 * (${half} - 1) is not usable; pass n explicitly` : undefined
  );
  if (half === 0) {
    throw new InvalidParameterError(
      `Invalid number of FFT data points (0) along axis ${ax}`,
      "axis",
      ax
    );
  }
  const full = hermitianExtend(
    toFloat64(real, "irfft"),
    toFloat64(imag, "irfft"),
    real.shape,
    ax,
    len
  );
  const spec = transformAxis(full.re, full.im, full.shape, ax, len, true, norm);
  const dtype = fftDtype(real.dtype, imag.dtype);
  return {
    real: makeTensor(spec.re, spec.shape, dtype),
    imag: makeTensor(new Float64Array(spec.re.length), spec.shape, dtype),
  };
}

/** Shared driver for fftn/ifftn (and the 2-D variants). */
function transformN(
  real: Tensor,
  imag: Tensor | null,
  axes: readonly number[] | undefined,
  inverse: boolean,
  norm: FFTNorm,
  op: string
): FFTResult {
  checkNorm(norm);
  const resolved = resolveAxes(axes, real.ndim);
  let re = toFloat64(real, op);
  let im: Float64Array | null = imag ? toFloat64(imag, op) : null;
  let shape: number[] = real.shape.slice();
  for (const axis of resolved) {
    const len = axisLength(shape, axis);
    const spec = transformAxis(re, im, shape, axis, len, inverse, norm);
    re = spec.re;
    im = spec.im;
    shape = spec.shape;
  }
  const dtype = fftDtype(real.dtype, ...(imag ? [imag.dtype] : []));
  if (resolved.length === 0) {
    // No axes were transformed: the result is a copy of the input as a complex signal.
    re = re.slice();
    im = im ? im.slice() : new Float64Array(re.length);
  }
  if (im === null) im = new Float64Array(re.length);
  return makeResult({ re, im, shape }, dtype);
}

/**
 * Compute the 2-dimensional discrete Fourier transform.
 *
 * @param t - Real-valued input with at least 2 dimensions.
 * @param axes - The two axes to transform. Default: the last two, `[-2, -1]`.
 * @param norm - Normalization mode. Default: `"backward"`.
 * @returns Object with `real` and `imag` tensors of the same shape as `t`.
 *
 * @example
 * ```ts
 * const { real, imag } = fft2(tensor([[1, 0], [0, 0]]));
 * // real = [[1, 1], [1, 1]], imag = zeros
 * ```
 */
export function fft2(
  t: Tensor,
  axes: readonly [number, number] = [-2, -1],
  norm: FFTNorm = "backward"
): FFTResult {
  requireNumeric(t, "fft2", "t");
  if (t.ndim < 2) throw new ShapeError("fft2 requires at least 2D input");
  return transformN(t, null, axes, false, norm, "fft2");
}

/**
 * Compute the 2-dimensional inverse discrete Fourier transform.
 *
 * @param real - Real part of the 2-D spectrum (at least 2 dimensions).
 * @param imag - Imaginary part of the 2-D spectrum (same shape as `real`).
 * @param axes - The two axes to transform. Default: `[-2, -1]`.
 * @param norm - Normalization mode. Default: `"backward"`.
 * @returns Object with `real` and `imag` tensors of the same shape as the input.
 */
export function ifft2(
  real: Tensor,
  imag: Tensor,
  axes: readonly [number, number] = [-2, -1],
  norm: FFTNorm = "backward"
): FFTResult {
  requireNumeric(real, "ifft2", "real");
  requireNumeric(imag, "ifft2", "imag");
  if (real.ndim < 2 || imag.ndim < 2) throw new ShapeError("ifft2 requires at least 2D input");
  requireSameShape(real, imag, "ifft2");
  return transformN(real, imag, axes, true, norm, "ifft2");
}

/**
 * Compute the n-dimensional discrete Fourier transform.
 *
 * The 1-D transform is applied along each listed axis in turn. If `axes` is
 * omitted, every axis is transformed.
 *
 * @param t - Real-valued input tensor.
 * @param axes - Axes to transform (negative values count from the end). Default: all axes.
 * @param norm - Normalization mode. Default: `"backward"`.
 * @returns Object with `real` and `imag` tensors of the same shape as `t`.
 *
 * @example
 * ```ts
 * const { real } = fftn(tensor([[1, 0], [0, 0]]));
 * ```
 */
export function fftn(t: Tensor, axes?: number[], norm: FFTNorm = "backward"): FFTResult {
  requireNumeric(t, "fftn", "t");
  if (t.ndim === 0) throw new ShapeError("fftn requires at least 1D input");
  return transformN(t, null, axes, false, norm, "fftn");
}

/**
 * Compute the n-dimensional inverse discrete Fourier transform.
 *
 * @param real - Real part of the n-D spectrum.
 * @param imag - Imaginary part of the n-D spectrum (same shape as `real`).
 * @param axes - Axes to transform. Default: all axes.
 * @param norm - Normalization mode. Default: `"backward"`.
 * @returns Object with `real` and `imag` tensors of the same shape as the input.
 */
export function ifftn(
  real: Tensor,
  imag: Tensor,
  axes?: number[],
  norm: FFTNorm = "backward"
): FFTResult {
  requireNumeric(real, "ifftn", "real");
  requireNumeric(imag, "ifftn", "imag");
  if (real.ndim === 0 || imag.ndim === 0) throw new ShapeError("ifftn requires at least 1D input");
  requireSameShape(real, imag, "ifftn");
  return transformN(real, imag, axes, true, norm, "ifftn");
}

// ---- Frequency helpers ----

/**
 * Sample frequencies of an `n`-point FFT: `[0, 1, ..., ceil(n/2) - 1, -floor(n/2), ..., -1] / (n * d)`.
 *
 * @param n - Window length (positive integer).
 * @param d - Sample spacing (non-zero). Default: 1.
 * @returns 1-D float64 tensor of length `n`.
 *
 * @example
 * ```ts
 * fftfreq(4);  // [0, 0.25, -0.5, -0.25]
 * ```
 */
export function fftfreq(n: number, d = 1): Tensor {
  validateLength(n, "n");
  if (!Number.isFinite(d) || d === 0) {
    throw new InvalidParameterError(`d must be a finite non-zero number; received ${d}`, "d", d);
  }
  const val = 1 / (n * d);
  const out = new Float64Array(n);
  const positive = Math.floor((n - 1) / 2) + 1;
  for (let i = 0; i < positive; i++) out[i] = i * val;
  for (let i = positive; i < n; i++) out[i] = (i - n) * val;
  return Tensor.fromTypedArray({ data: out, shape: [n], dtype: "float64", device: "cpu" });
}

/**
 * Sample frequencies of an `n`-point {@link rfft}: `[0, 1, ..., floor(n/2)] / (n * d)`.
 *
 * @param n - Window length (positive integer).
 * @param d - Sample spacing (non-zero). Default: 1.
 * @returns 1-D float64 tensor of length `floor(n / 2) + 1`.
 */
export function rfftfreq(n: number, d = 1): Tensor {
  validateLength(n, "n");
  if (!Number.isFinite(d) || d === 0) {
    throw new InvalidParameterError(`d must be a finite non-zero number; received ${d}`, "d", d);
  }
  const val = 1 / (n * d);
  const count = Math.floor(n / 2) + 1;
  const out = new Float64Array(count);
  for (let i = 0; i < count; i++) out[i] = i * val;
  return Tensor.fromTypedArray({ data: out, shape: [count], dtype: "float64", device: "cpu" });
}

function shiftAxes(t: Tensor, axes: number[] | undefined, inverse: boolean): Tensor {
  const resolved = resolveAxes(axes, t.ndim);
  if (resolved.length === 0) return clone(t);
  let out = t;
  for (const axis of resolved) {
    const half = Math.floor((t.shape[axis] as number) / 2);
    out = roll(out, inverse ? -half : half, axis);
  }
  return out;
}

/**
 * Shift the zero-frequency component to the center of the spectrum.
 *
 * @param t - Input tensor (for example one half of an {@link fft} result).
 * @param axes - Axes to shift. Default: all axes.
 * @returns Shifted copy of `t`.
 *
 * @example
 * ```ts
 * fftshift(fftfreq(4));  // [-0.5, -0.25, 0, 0.25]
 * ```
 */
export function fftshift(t: Tensor, axes?: number[]): Tensor {
  return shiftAxes(t, axes, false);
}

/**
 * Inverse of {@link fftshift}.
 *
 * @param t - Input tensor.
 * @param axes - Axes to shift. Default: all axes.
 * @returns Shifted copy of `t`.
 */
export function ifftshift(t: Tensor, axes?: number[]): Tensor {
  return shiftAxes(t, axes, true);
}
