/**
 * Signal processing: window functions, convolution, and correlation.
 *
 * Window functions follow NumPy (`np.hanning`, `np.hamming`, `np.blackman`,
 * `np.bartlett`, `np.kaiser`): the default window is symmetric, which is the
 * right choice for filter design. Pass `{ periodic: true }` for the periodic
 * form used for spectral analysis (the same as `scipy.signal.get_window` with
 * `fftbins=True` and `torch.hann_window` with `periodic=True`).
 *
 * @module ndarray/ops/signal
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */
import { InvalidParameterError } from "../../core";
import { Tensor } from "../tensor/Tensor";
import { readNumbers } from "./_internal";

/** Options shared by the window functions. */
export type WindowOptions = {
  /**
   * Build the periodic window (length `n` of an `n + 1` point symmetric
   * window with the last point dropped) instead of the symmetric one.
   * Default: `false`.
   */
  readonly periodic?: boolean;
};

/** Boundary handling for {@link convolve} and {@link correlate}. */
export type ConvolveMode = "full" | "same" | "valid";

function validateWinLen(n: number, name: string): void {
  if (!Number.isInteger(n) || n < 1) {
    throw new InvalidParameterError(`${name}: n must be a positive integer`, "n", n);
  }
}

function mkTensor(data: Float64Array, n: number): Tensor {
  return Tensor.fromTypedArray({
    data,
    shape: [n],
    dtype: "float64",
    device: "cpu",
  });
}

/**
 * Build a cosine-family window from a per-index function of the phase
 * `i / denominator`, where the denominator is `n - 1` (symmetric) or `n`
 * (periodic).
 */
function buildWindow(
  n: number,
  name: string,
  options: WindowOptions | undefined,
  at: (i: number, denom: number) => number
): Tensor {
  validateWinLen(n, name);
  if (n === 1) return mkTensor(new Float64Array([1]), 1);
  const denom = options?.periodic === true ? n : n - 1;
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) d[i] = at(i, denom);
  return mkTensor(d, n);
}

/**
 * Exponentially scaled modified Bessel function of the first kind and order
 * zero: `exp(-|x|) * I0(x)`.
 *
 * The scaling keeps the result finite for large arguments, where `I0(x)`
 * itself overflows a double near `x = 713`. Moderate arguments use the power
 * series (all terms positive, so no cancellation); large arguments use the
 * asymptotic expansion.
 */
function besselI0Scaled(x: number): number {
  const ax = Math.abs(x);
  if (ax < 30) {
    const h2 = (ax / 2) * (ax / 2);
    let sum = 1;
    let term = 1;
    for (let k = 1; k < 500; k++) {
      term *= h2 / (k * k);
      sum += term;
      if (term < 1e-17 * sum) break;
    }
    return sum * Math.exp(-ax);
  }
  let sum = 1;
  let term = 1;
  const eightX = 8 * ax;
  for (let k = 1; k < 200; k++) {
    const next = (term * (2 * k - 1) * (2 * k - 1)) / (k * eightX);
    // The asymptotic series diverges eventually; stop at the smallest term.
    if (next >= term) break;
    term = next;
    sum += term;
    if (term < 1e-17 * sum) break;
  }
  return sum / Math.sqrt(2 * Math.PI * ax);
}

/**
 * Hann window (raised cosine).
 *
 * `w[i] = 0.5 - 0.5 * cos(2 * pi * i / (n - 1))`. Matches `np.hanning(n)`.
 *
 * @param n - Number of points; a positive integer
 * @param options - `{ periodic: true }` for the periodic form
 * @returns float64 tensor of shape `[n]`
 * @throws {InvalidParameterError} If `n` is not a positive integer
 *
 * @example
 * ```ts
 * hannWindow(5); // [0, 0.5, 1, 0.5, 0]
 * ```
 */
export function hannWindow(n: number, options?: WindowOptions): Tensor {
  return buildWindow(
    n,
    "hannWindow",
    options,
    (i, denom) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / denom)
  );
}

/**
 * Hamming window.
 *
 * `w[i] = 0.54 - 0.46 * cos(2 * pi * i / (n - 1))`. Matches `np.hamming(n)`.
 *
 * @param n - Number of points; a positive integer
 * @param options - `{ periodic: true }` for the periodic form
 * @returns float64 tensor of shape `[n]`
 * @throws {InvalidParameterError} If `n` is not a positive integer
 *
 * @example
 * ```ts
 * hammingWindow(5); // [0.08, 0.54, 1, 0.54, 0.08]
 * ```
 */
export function hammingWindow(n: number, options?: WindowOptions): Tensor {
  return buildWindow(
    n,
    "hammingWindow",
    options,
    (i, denom) => 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / denom)
  );
}

/**
 * Blackman window.
 *
 * `w[i] = 0.42 - 0.5 * cos(2 * pi * i / (n - 1)) + 0.08 * cos(4 * pi * i / (n - 1))`.
 * Matches `np.blackman(n)`.
 *
 * @param n - Number of points; a positive integer
 * @param options - `{ periodic: true }` for the periodic form
 * @returns float64 tensor of shape `[n]`
 * @throws {InvalidParameterError} If `n` is not a positive integer
 *
 * @example
 * ```ts
 * blackmanWindow(5); // [-1.4e-17, 0.34, 1, 0.34, -1.4e-17]
 * ```
 */
export function blackmanWindow(n: number, options?: WindowOptions): Tensor {
  return buildWindow(
    n,
    "blackmanWindow",
    options,
    (i, denom) =>
      0.42 - 0.5 * Math.cos((2 * Math.PI * i) / denom) + 0.08 * Math.cos((4 * Math.PI * i) / denom)
  );
}

/**
 * Bartlett (triangular) window with zero end points.
 *
 * Matches `np.bartlett(n)`.
 *
 * @param n - Number of points; a positive integer
 * @param options - `{ periodic: true }` for the periodic form
 * @returns float64 tensor of shape `[n]`
 * @throws {InvalidParameterError} If `n` is not a positive integer
 *
 * @example
 * ```ts
 * bartlettWindow(5); // [0, 0.5, 1, 0.5, 0]
 * ```
 */
export function bartlettWindow(n: number, options?: WindowOptions): Tensor {
  return buildWindow(n, "bartlettWindow", options, (i, denom) =>
    i <= denom / 2 ? (2 * i) / denom : 2 - (2 * i) / denom
  );
}

/**
 * Kaiser window.
 *
 * `w[i] = I0(beta * sqrt(1 - r^2)) / I0(beta)` with `r = 2 * i / (n - 1) - 1`,
 * where `I0` is the modified Bessel function of order zero. Matches
 * `np.kaiser(n, beta)`. Larger `beta` gives a narrower main lobe and lower
 * side lobes; `beta = 0` is the rectangular window.
 *
 * @param n - Number of points; a positive integer
 * @param beta - Shape parameter; a finite number (default 12)
 * @param options - `{ periodic: true }` for the periodic form
 * @returns float64 tensor of shape `[n]`
 * @throws {InvalidParameterError} If `n` is not a positive integer or `beta` is not finite
 *
 * @example
 * ```ts
 * kaiserWindow(8, 5); // [0.0367, 0.2707, 0.6517, 0.9552, 0.9552, 0.6517, 0.2707, 0.0367]
 * ```
 */
export function kaiserWindow(n: number, beta = 12, options?: WindowOptions): Tensor {
  validateWinLen(n, "kaiserWindow");
  if (!Number.isFinite(beta)) {
    throw new InvalidParameterError("kaiserWindow: beta must be a finite number", "beta", beta);
  }
  const absBeta = Math.abs(beta);
  const i0Beta = besselI0Scaled(absBeta);
  return buildWindow(n, "kaiserWindow", options, (i, denom) => {
    const r = (2 * i) / denom - 1;
    const x = absBeta * Math.sqrt(Math.max(0, 1 - r * r));
    // I0(x) / I0(beta) computed from the scaled values so large beta cannot overflow.
    return (besselI0Scaled(x) / i0Beta) * Math.exp(x - absBeta);
  });
}

function validateConvInputs(a: Tensor, v: Tensor, mode: ConvolveMode, name: string): void {
  if (a.dtype === "string" || v.dtype === "string") {
    throw new InvalidParameterError(`${name} requires numeric input`, "a", a.dtype);
  }
  if (a.ndim !== 1) {
    throw new InvalidParameterError(
      `${name} requires 1-D inputs; a has ndim ${a.ndim}`,
      "a",
      a.shape
    );
  }
  if (v.ndim !== 1) {
    throw new InvalidParameterError(
      `${name} requires 1-D inputs; v has ndim ${v.ndim}`,
      "v",
      v.shape
    );
  }
  if (a.size === 0) {
    throw new InvalidParameterError(`${name}: a cannot be empty`, "a", a.shape);
  }
  if (v.size === 0) {
    throw new InvalidParameterError(`${name}: v cannot be empty`, "v", v.shape);
  }
  if (mode !== "full" && mode !== "same" && mode !== "valid") {
    throw new InvalidParameterError(
      `${name}: mode must be "full", "same" or "valid"; received ${String(mode)}`,
      "mode",
      mode
    );
  }
}

/**
 * Trim a full-length convolution to the requested mode. `same` returns the
 * central `max(aLen, vLen)` samples (the extra sample of an even-length
 * overhang is dropped from the end, as in NumPy); `valid` returns only the
 * samples computed without zero padding.
 */
function trimArray(
  full: Float64Array,
  aLen: number,
  vLen: number,
  mode: ConvolveMode
): Float64Array {
  if (mode === "full") return full;
  if (mode === "same") {
    const len = Math.max(aLen, vLen);
    const start = Math.floor((full.length - len) / 2);
    return full.slice(start, start + len);
  }
  const len = Math.max(aLen, vLen) - Math.min(aLen, vLen) + 1;
  const start = Math.min(aLen, vLen) - 1;
  return full.slice(start, start + len);
}

/** Direct `O(aLen * vLen)` full convolution of two non-empty arrays. */
function fullConvolution(a: ArrayLike<number>, v: ArrayLike<number>): Float64Array {
  const aLen = a.length;
  const vLen = v.length;
  const full = new Float64Array(aLen + vLen - 1);
  for (let i = 0; i < aLen; i++) {
    const ai = a[i] as number;
    for (let j = 0; j < vLen; j++) {
      full[i + j] = (full[i + j] as number) + ai * (v[j] as number);
    }
  }
  return full;
}

/** Cross-correlation with `a` at least as long as `v`, trimmed to `mode`. */
function correlateLongFirst(
  a: ArrayLike<number>,
  v: ArrayLike<number>,
  mode: ConvolveMode
): Float64Array {
  const reversed = new Float64Array(v.length);
  for (let j = 0; j < v.length; j++) reversed[j] = v[v.length - 1 - j] as number;
  return trimArray(fullConvolution(a, reversed), a.length, v.length, mode);
}

/**
 * Discrete linear convolution of two 1-D tensors.
 *
 * Matches `np.convolve(a, v, mode)`. The result is a float64 tensor.
 *
 * - `"full"` (default): every overlap, length `a.size + v.size - 1`
 * - `"same"`: length `max(a.size, v.size)`, centered with respect to `"full"`
 * - `"valid"`: only complete overlaps, length `max(a.size, v.size) - min(a.size, v.size) + 1`
 *
 * @param a - First 1-D input
 * @param v - Second 1-D input
 * @param mode - Output size, default `"full"`
 * @returns float64 tensor
 * @throws {InvalidParameterError} If an input is not a non-empty 1-D numeric tensor or `mode` is unknown
 *
 * @example
 * ```ts
 * convolve(tensor([1, 2, 3]), tensor([0, 1, 0.5])); // [0, 1, 2.5, 4, 1.5]
 * ```
 */
export function convolve(a: Tensor, v: Tensor, mode: ConvolveMode = "full"): Tensor {
  validateConvInputs(a, v, mode, "convolve");
  const aD = readNumbers(a, "convolve", false);
  const vD = readNumbers(v, "convolve", false);
  const result = trimArray(fullConvolution(aD, vD), a.size, v.size, mode);
  return mkTensor(result, result.length);
}

/**
 * Cross-correlation of two 1-D tensors.
 *
 * Matches `np.correlate(a, v, mode)`: `c[k] = sum_n a[n + k] * v[n]`, which is
 * `convolve(a, reversed(v), mode)`. The result is a float64 tensor.
 *
 * @param a - First 1-D input
 * @param v - Second 1-D input
 * @param mode - Output size, default `"full"` (see {@link convolve})
 * @returns float64 tensor
 * @throws {InvalidParameterError} If an input is not a non-empty 1-D numeric tensor or `mode` is unknown
 *
 * @example
 * ```ts
 * correlate(tensor([1, 2, 3]), tensor([0, 1, 0.5])); // [0.5, 2, 3.5, 3, 0]
 * ```
 */
export function correlate(a: Tensor, v: Tensor, mode: ConvolveMode = "full"): Tensor {
  validateConvInputs(a, v, mode, "correlate");
  const aD = readNumbers(a, "correlate", false);
  const vD = readNumbers(v, "correlate", false);
  // NumPy computes the correlation with the longer input first and flips the
  // result when the inputs had to be swapped; this matters for "same".
  if (aD.length < vD.length) {
    const flipped = correlateLongFirst(vD, aD, mode);
    return mkTensor(flipped.reverse(), flipped.length);
  }
  const result = correlateLongFirst(aD, vD, mode);
  return mkTensor(result, result.length);
}
