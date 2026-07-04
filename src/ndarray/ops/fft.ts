/**
 * Fast Fourier Transform operations.
 *
 * Implements the Cooley-Tukey radix-2 FFT algorithm with Bluestein's
 * algorithm fallback for non-power-of-2 lengths. All operations work
 * on real-valued tensors and return results as pairs of tensors
 * (real part, imaginary part) since JS has no native complex type.
 *
 * @module ndarray/ops/fft
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import type { DType } from "../../core";
import { InvalidParameterError, ShapeError } from "../../core";
import { computeStrides, Tensor } from "../tensor/Tensor";

/** Result of an FFT operation: real and imaginary parts as separate tensors. */
export interface FFTResult {
  real: Tensor;
  imag: Tensor;
}

// ---- Internal helpers ----

function isPowerOf2(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

function nextPowerOf2(n: number): number {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/**
 * In-place radix-2 Cooley-Tukey FFT.
 * Arrays `re` and `im` must have length that is a power of 2.
 * `inverse` = true computes the IFFT (without the 1/N scaling).
 */
function fftRadix2(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length;

  // Bit-reversal permutation
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      const tmpRe = re[i] ?? 0;
      const tmpIm = im[i] ?? 0;
      re[i] = re[j] ?? 0;
      im[i] = im[j] ?? 0;
      re[j] = tmpRe;
      im[j] = tmpIm;
    }
    let m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }

  // Cooley-Tukey butterfly
  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (sign * 2 * Math.PI) / len;
    const wRe = Math.cos(angle);
    const wIm = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < halfLen; k++) {
        const evenIdx = i + k;
        const oddIdx = i + k + halfLen;
        const eRe = re[evenIdx] ?? 0;
        const eIm = im[evenIdx] ?? 0;
        const oRe = re[oddIdx] ?? 0;
        const oIm = im[oddIdx] ?? 0;

        const tRe = curRe * oRe - curIm * oIm;
        const tIm = curRe * oIm + curIm * oRe;

        re[evenIdx] = eRe + tRe;
        im[evenIdx] = eIm + tIm;
        re[oddIdx] = eRe - tRe;
        im[oddIdx] = eIm - tIm;

        const newCurRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = newCurRe;
      }
    }
  }
}

/**
 * Bluestein's algorithm for arbitrary-length FFT.
 * Reduces to convolution via power-of-2 FFT.
 */
function fftBluestein(
  re: Float64Array,
  im: Float64Array,
  inverse: boolean
): { real: Float64Array; imag: Float64Array } {
  const n = re.length;
  const m = nextPowerOf2(2 * n - 1);
  // The identity jk = (j^2 + k^2 - (k-j)^2) / 2 turns the DFT kernel
  // exp(-i*2*pi*jk/n) into chirp products with exponent -sign, so the chirp
  // sign must be the OPPOSITE of the radix-2 butterfly sign convention.
  const sign = inverse ? -1 : 1;

  // Chirp: w_k = exp(sign * i * pi * k^2 / n)
  const chirpRe = new Float64Array(n);
  const chirpIm = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const angle = (sign * Math.PI * k * k) / n;
    chirpRe[k] = Math.cos(angle);
    chirpIm[k] = Math.sin(angle);
  }

  // a_k = x_k * conj(chirp_k)
  const aRe = new Float64Array(m);
  const aIm = new Float64Array(m);
  for (let k = 0; k < n; k++) {
    const xr = re[k] ?? 0;
    const xi = im[k] ?? 0;
    const cr = chirpRe[k] ?? 0;
    const ci = chirpIm[k] ?? 0;
    // conj(chirp) = (cr, -ci)
    aRe[k] = xr * cr + xi * ci;
    aIm[k] = xi * cr - xr * ci;
  }

  // b_k = chirp_k, zero-padded with wrap-around
  const bRe = new Float64Array(m);
  const bIm = new Float64Array(m);
  bRe[0] = chirpRe[0] ?? 0;
  bIm[0] = chirpIm[0] ?? 0;
  for (let k = 1; k < n; k++) {
    const cr = chirpRe[k] ?? 0;
    const ci = chirpIm[k] ?? 0;
    bRe[k] = cr;
    bIm[k] = ci;
    bRe[m - k] = cr;
    bIm[m - k] = ci;
  }

  // Forward FFT of a and b
  fftRadix2(aRe, aIm, false);
  fftRadix2(bRe, bIm, false);

  // Pointwise multiply
  const cRe = new Float64Array(m);
  const cIm = new Float64Array(m);
  for (let k = 0; k < m; k++) {
    const ar = aRe[k] ?? 0;
    const ai = aIm[k] ?? 0;
    const br = bRe[k] ?? 0;
    const bi = bIm[k] ?? 0;
    cRe[k] = ar * br - ai * bi;
    cIm[k] = ar * bi + ai * br;
  }

  // Inverse FFT
  fftRadix2(cRe, cIm, true);
  const invM = 1 / m;
  for (let k = 0; k < m; k++) {
    cRe[k] = (cRe[k] ?? 0) * invM;
    cIm[k] = (cIm[k] ?? 0) * invM;
  }

  // Multiply by conj(chirp) and extract first n elements
  const outRe = new Float64Array(n);
  const outIm = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const cr = chirpRe[k] ?? 0;
    const ci = chirpIm[k] ?? 0;
    const vr = cRe[k] ?? 0;
    const vi = cIm[k] ?? 0;
    outRe[k] = vr * cr + vi * ci;
    outIm[k] = vi * cr - vr * ci;
  }

  return { real: outRe, imag: outIm };
}

/**
 * Compute 1D FFT on arrays of length n (any positive integer).
 */
function fft1d(
  re: Float64Array,
  im: Float64Array,
  inverse: boolean
): { real: Float64Array; imag: Float64Array } {
  const n = re.length;
  if (n <= 1) {
    return { real: new Float64Array(re), imag: new Float64Array(im) };
  }

  if (isPowerOf2(n)) {
    const outRe = new Float64Array(re);
    const outIm = new Float64Array(im);
    fftRadix2(outRe, outIm, inverse);
    return { real: outRe, imag: outIm };
  }

  return fftBluestein(re, im, inverse);
}

/**
 * Read a numeric value from a tensor at a physical offset.
 */
function readVal(t: Tensor, offset: number): number {
  const raw = t.data[offset];
  if (typeof raw === "number") return raw;
  if (typeof raw === "bigint") return Number(raw);
  return 0;
}

/**
 * Read the i-th element in logical (row-major) order, honoring the tensor's
 * strides and offset so non-contiguous views (transposes, slices) are read
 * correctly.
 */
function readLogical(t: Tensor, i: number, logicalStrides: readonly number[]): number {
  let rem = i;
  let off = t.offset;
  for (let d = 0; d < t.ndim; d++) {
    const ls = logicalStrides[d] ?? 1;
    const coord = Math.floor(rem / ls);
    rem -= coord * ls;
    off += coord * (t.strides[d] ?? 0);
  }
  return readVal(t, off);
}

/**
 * Normalize FFT axes (supporting negative indices) and validate.
 */
function normalizeFftAxes(axes: number[] | undefined, ndim: number): number[] {
  const raw = axes ?? Array.from({ length: ndim }, (_, i) => i);
  return raw.map((ax) => {
    const normalized = ax < 0 ? ax + ndim : ax;
    if (normalized < 0 || normalized >= ndim || !Number.isInteger(normalized)) {
      throw new InvalidParameterError(`Invalid axis ${ax} for ${ndim}D input`, "axes", ax);
    }
    return normalized;
  });
}

/**
 * Determine the output dtype for FFT operations.
 */
function fftDtype(inputDtype: DType): "float32" | "float64" {
  return inputDtype === "float64" ? "float64" : "float32";
}

/**
 * Create a tensor from Float64Array with optional dtype downcast.
 */
function makeTensor(data: Float64Array, shape: number[], dtype: "float32" | "float64"): Tensor {
  if (dtype === "float32") {
    return Tensor.fromTypedArray({
      data: new Float32Array(data),
      shape,
      dtype: "float32",
      device: "cpu",
    });
  }
  return Tensor.fromTypedArray({
    data,
    shape,
    dtype: "float64",
    device: "cpu",
  });
}

// ---- Public API ----

/**
 * Compute the one-dimensional discrete Fourier Transform.
 *
 * @param t - Input real-valued tensor. FFT is computed along the last axis.
 * @param n - Length of the transformed axis. If shorter than the input, the
 *   input is cropped. If longer, the input is zero-padded. Default: input length.
 * @returns Object with `real` and `imag` tensors of the same shape.
 *
 * @example
 * ```ts
 * import { fft } from 'deepbox/ndarray';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const t = tensor([1, 0, 0, 0]);
 * const { real, imag } = fft(t);
 * // real ≈ [1, 1, 1, 1], imag ≈ [0, 0, 0, 0]
 * ```
 */
export function fft(t: Tensor, n?: number): FFTResult {
  if (t.dtype === "string") {
    throw new InvalidParameterError("fft requires numeric input", "t");
  }
  if (t.ndim === 0) {
    throw new ShapeError("fft requires at least 1D input");
  }

  const lastAxis = t.ndim - 1;
  const inputLen = t.shape[lastAxis] ?? 0;
  const fftLen = n ?? inputLen;

  if (fftLen <= 0 || !Number.isInteger(fftLen)) {
    throw new InvalidParameterError("n must be a positive integer", "n", fftLen);
  }

  const outDtype = fftDtype(t.dtype);
  const outerShape = t.shape.slice(0, lastAxis);
  const outerSize = outerShape.reduce((a, b) => a * b, 1);
  const outShape = [...outerShape, fftLen];

  const totalOut = outerSize * fftLen;
  const realData = new Float64Array(totalOut);
  const imagData = new Float64Array(totalOut);

  for (let outer = 0; outer < outerSize; outer++) {
    // Compute base offset for this slice along all but last axis
    let baseOffset = t.offset;
    let rem = outer;
    for (let d = lastAxis - 1; d >= 0; d--) {
      const dimSize = t.shape[d] ?? 1;
      const coord = rem % dimSize;
      rem = Math.floor(rem / dimSize);
      baseOffset += coord * (t.strides[d] ?? 0);
    }

    // Extract input signal
    const re = new Float64Array(fftLen);
    const im = new Float64Array(fftLen);
    const copyLen = Math.min(inputLen, fftLen);
    const lastStride = t.strides[lastAxis] ?? 1;
    for (let i = 0; i < copyLen; i++) {
      re[i] = readVal(t, baseOffset + i * lastStride);
    }

    // Compute FFT
    const result = fft1d(re, im, false);

    // Store result
    const outBase = outer * fftLen;
    for (let i = 0; i < fftLen; i++) {
      realData[outBase + i] = result.real[i] ?? 0;
      imagData[outBase + i] = result.imag[i] ?? 0;
    }
  }

  return {
    real: makeTensor(realData, outShape, outDtype),
    imag: makeTensor(imagData, outShape, outDtype),
  };
}

/**
 * Compute the one-dimensional inverse discrete Fourier Transform.
 *
 * @param real - Real part of the input spectrum
 * @param imag - Imaginary part of the input spectrum
 * @param n - Length of the transformed axis. Default: input length.
 * @returns Object with `real` and `imag` tensors.
 *
 * @example
 * ```ts
 * const spectrum = fft(tensor([1, 2, 3, 4]));
 * const { real } = ifft(spectrum.real, spectrum.imag);
 * // real ≈ [1, 2, 3, 4]
 * ```
 */
export function ifft(real: Tensor, imag: Tensor, n?: number): FFTResult {
  if (real.dtype === "string" || imag.dtype === "string") {
    throw new InvalidParameterError("ifft requires numeric input", "real");
  }
  if (real.ndim === 0 || imag.ndim === 0) {
    throw new ShapeError("ifft requires at least 1D input");
  }

  const lastAxis = real.ndim - 1;
  const inputLen = real.shape[lastAxis] ?? 0;
  const fftLen = n ?? inputLen;

  if (fftLen <= 0 || !Number.isInteger(fftLen)) {
    throw new InvalidParameterError("n must be a positive integer", "n", fftLen);
  }

  const outDtype = fftDtype(real.dtype);
  const outerShape = real.shape.slice(0, lastAxis);
  const outerSize = outerShape.reduce((a, b) => a * b, 1);
  const outShape = [...outerShape, fftLen];

  const totalOut = outerSize * fftLen;
  const realData = new Float64Array(totalOut);
  const imagData = new Float64Array(totalOut);

  for (let outer = 0; outer < outerSize; outer++) {
    // Compute base offsets
    let realBase = real.offset;
    let imagBase = imag.offset;
    let rem = outer;
    for (let d = lastAxis - 1; d >= 0; d--) {
      const dimSize = real.shape[d] ?? 1;
      const coord = rem % dimSize;
      rem = Math.floor(rem / dimSize);
      realBase += coord * (real.strides[d] ?? 0);
      imagBase += coord * (imag.strides[d] ?? 0);
    }

    const re = new Float64Array(fftLen);
    const im = new Float64Array(fftLen);
    const copyLen = Math.min(inputLen, fftLen);
    const realLastStride = real.strides[lastAxis] ?? 1;
    const imagLastStride = imag.strides[lastAxis] ?? 1;
    for (let i = 0; i < copyLen; i++) {
      re[i] = readVal(real, realBase + i * realLastStride);
      im[i] = readVal(imag, imagBase + i * imagLastStride);
    }

    const result = fft1d(re, im, true);

    // Scale by 1/N
    const invN = 1 / fftLen;
    const outBase = outer * fftLen;
    for (let i = 0; i < fftLen; i++) {
      realData[outBase + i] = (result.real[i] ?? 0) * invN;
      imagData[outBase + i] = (result.imag[i] ?? 0) * invN;
    }
  }

  return {
    real: makeTensor(realData, outShape, outDtype),
    imag: makeTensor(imagData, outShape, outDtype),
  };
}

/**
 * Compute the one-dimensional FFT for real input.
 *
 * Since the input is real, the output is Hermitian-symmetric and only
 * the positive-frequency terms are returned (length floor(n/2) + 1).
 *
 * @param t - Real-valued input tensor. FFT is computed along the last axis.
 * @param n - Length of the FFT. Default: input length along last axis.
 * @returns Object with `real` and `imag` tensors of shape [..., floor(n/2)+1].
 */
export function rfft(t: Tensor, n?: number): FFTResult {
  const full = fft(t, n);
  const fftLen = full.real.shape[full.real.ndim - 1] ?? 0;
  const halfLen = Math.floor(fftLen / 2) + 1;

  const lastAxis = full.real.ndim - 1;
  const outerShape = full.real.shape.slice(0, lastAxis);
  const outerSize = outerShape.reduce((a, b) => a * b, 1);
  const outShape = [...outerShape, halfLen];

  const totalOut = outerSize * halfLen;
  const realData = new Float64Array(totalOut);
  const imagData = new Float64Array(totalOut);

  const fullLastStride = full.real.strides[lastAxis] ?? 1;

  for (let outer = 0; outer < outerSize; outer++) {
    let realBase = full.real.offset;
    let imagBase = full.imag.offset;
    let rem = outer;
    for (let d = lastAxis - 1; d >= 0; d--) {
      const dimSize = full.real.shape[d] ?? 1;
      const coord = rem % dimSize;
      rem = Math.floor(rem / dimSize);
      realBase += coord * (full.real.strides[d] ?? 0);
      imagBase += coord * (full.imag.strides[d] ?? 0);
    }

    const outBase = outer * halfLen;
    for (let i = 0; i < halfLen; i++) {
      realData[outBase + i] = readVal(full.real, realBase + i * fullLastStride);
      imagData[outBase + i] = readVal(full.imag, imagBase + i * fullLastStride);
    }
  }

  const outDtype = fftDtype(t.dtype);
  return {
    real: makeTensor(realData, outShape, outDtype),
    imag: makeTensor(imagData, outShape, outDtype),
  };
}

/**
 * Compute the inverse FFT for rfft output (Hermitian-symmetric).
 *
 * @param real - Real part of the one-sided spectrum (length floor(n/2)+1)
 * @param imag - Imaginary part of the one-sided spectrum
 * @param n - Output length. Default: 2*(input_length - 1).
 * @returns Object with `real` and `imag` tensors of shape [..., n].
 */
export function irfft(real: Tensor, imag: Tensor, n?: number): FFTResult {
  if (real.dtype === "string" || imag.dtype === "string") {
    throw new InvalidParameterError("irfft requires numeric input", "real");
  }
  if (real.ndim === 0 || imag.ndim === 0) {
    throw new ShapeError("irfft requires at least 1D input");
  }

  const lastAxis = real.ndim - 1;
  const halfLen = real.shape[lastAxis] ?? 0;
  const fftLen = n ?? 2 * (halfLen - 1);

  if (fftLen <= 0 || !Number.isInteger(fftLen)) {
    throw new InvalidParameterError("n must be a positive integer", "n", fftLen);
  }

  const outDtype = fftDtype(real.dtype);
  const outerShape = real.shape.slice(0, lastAxis);
  const outerSize = outerShape.reduce((a, b) => a * b, 1);
  const outShape = [...outerShape, fftLen];

  const totalOut = outerSize * fftLen;
  const realData = new Float64Array(totalOut);
  const imagData = new Float64Array(totalOut);

  for (let outer = 0; outer < outerSize; outer++) {
    let realBase = real.offset;
    let imagBase = imag.offset;
    let rem = outer;
    for (let d = lastAxis - 1; d >= 0; d--) {
      const dimSize = real.shape[d] ?? 1;
      const coord = rem % dimSize;
      rem = Math.floor(rem / dimSize);
      realBase += coord * (real.strides[d] ?? 0);
      imagBase += coord * (imag.strides[d] ?? 0);
    }

    // Reconstruct full spectrum from half via Hermitian symmetry
    const fullRe = new Float64Array(fftLen);
    const fullIm = new Float64Array(fftLen);
    const realLastStride = real.strides[lastAxis] ?? 1;
    const imagLastStride = imag.strides[lastAxis] ?? 1;

    const copyLen = Math.min(halfLen, fftLen);
    for (let i = 0; i < copyLen; i++) {
      fullRe[i] = readVal(real, realBase + i * realLastStride);
      fullIm[i] = readVal(imag, imagBase + i * imagLastStride);
    }
    // Hermitian symmetry: X[n-k] = conj(X[k])
    for (let i = 1; i < fftLen - Math.floor(fftLen / 2); i++) {
      if (i < halfLen) {
        fullRe[fftLen - i] = readVal(real, realBase + i * realLastStride);
        fullIm[fftLen - i] = -readVal(imag, imagBase + i * imagLastStride);
      }
    }

    // Inverse FFT
    const result = fft1d(fullRe, fullIm, true);
    const invN = 1 / fftLen;
    const outBase = outer * fftLen;
    for (let i = 0; i < fftLen; i++) {
      realData[outBase + i] = (result.real[i] ?? 0) * invN;
      imagData[outBase + i] = (result.imag[i] ?? 0) * invN;
    }
  }

  return {
    real: makeTensor(realData, outShape, outDtype),
    imag: makeTensor(imagData, outShape, outDtype),
  };
}

/**
 * Compute the 2-dimensional discrete Fourier Transform.
 *
 * @param t - Input tensor with at least 2 dimensions. FFT is computed
 *   along the last two axes.
 * @returns Object with `real` and `imag` tensors of the same shape.
 *
 * @example
 * ```ts
 * const t = tensor([[1, 0], [0, 0]]);
 * const { real, imag } = fft2(t);
 * ```
 */
export function fft2(t: Tensor): FFTResult {
  if (t.ndim < 2) {
    throw new ShapeError("fft2 requires at least 2D input");
  }

  // FFT along last axis
  const pass1 = fft(t);

  // FFT along second-to-last axis: transpose last two axes, FFT, transpose back
  const ndim = pass1.real.ndim;
  const axis0 = ndim - 2;
  const axis1 = ndim - 1;
  const dim0 = pass1.real.shape[axis0] ?? 0;
  const dim1 = pass1.real.shape[axis1] ?? 0;

  const outerShape = pass1.real.shape.slice(0, axis0);
  const outerSize = outerShape.reduce((a, b) => a * b, 1);

  const outShape = [...pass1.real.shape];
  const totalOut = outerSize * dim0 * dim1;
  const realData = new Float64Array(totalOut);
  const imagData = new Float64Array(totalOut);

  for (let outer = 0; outer < outerSize; outer++) {
    // Compute base offsets for pass1 result
    let realBase = pass1.real.offset;
    let imagBase = pass1.imag.offset;
    let rem = outer;
    for (let d = axis0 - 1; d >= 0; d--) {
      const dimSize = pass1.real.shape[d] ?? 1;
      const coord = rem % dimSize;
      rem = Math.floor(rem / dimSize);
      realBase += coord * (pass1.real.strides[d] ?? 0);
      imagBase += coord * (pass1.imag.strides[d] ?? 0);
    }

    // FFT along axis0: for each column j, extract column, FFT, store back
    for (let j = 0; j < dim1; j++) {
      const colRe = new Float64Array(dim0);
      const colIm = new Float64Array(dim0);

      const realRowStride = pass1.real.strides[axis0] ?? 0;
      const realColStride = pass1.real.strides[axis1] ?? 0;
      const imagRowStride = pass1.imag.strides[axis0] ?? 0;
      const imagColStride = pass1.imag.strides[axis1] ?? 0;

      for (let i = 0; i < dim0; i++) {
        colRe[i] = readVal(pass1.real, realBase + i * realRowStride + j * realColStride);
        colIm[i] = readVal(pass1.imag, imagBase + i * imagRowStride + j * imagColStride);
      }

      const result = fft1d(colRe, colIm, false);

      const outBaseOuter = outer * dim0 * dim1;
      for (let i = 0; i < dim0; i++) {
        realData[outBaseOuter + i * dim1 + j] = result.real[i] ?? 0;
        imagData[outBaseOuter + i * dim1 + j] = result.imag[i] ?? 0;
      }
    }
  }

  const outDtype = fftDtype(t.dtype);
  return {
    real: makeTensor(realData, outShape, outDtype),
    imag: makeTensor(imagData, outShape, outDtype),
  };
}

/**
 * Compute the 2-dimensional inverse discrete Fourier Transform.
 *
 * @param real - Real part of the 2D spectrum
 * @param imag - Imaginary part of the 2D spectrum
 * @returns Object with `real` and `imag` tensors of the same shape.
 */
export function ifft2(real: Tensor, imag: Tensor): FFTResult {
  if (real.ndim < 2 || imag.ndim < 2) {
    throw new ShapeError("ifft2 requires at least 2D input");
  }

  const ndim = real.ndim;
  const axis0 = ndim - 2;
  const axis1 = ndim - 1;
  const dim0 = real.shape[axis0] ?? 0;
  const dim1 = real.shape[axis1] ?? 0;

  const outerShape = real.shape.slice(0, axis0);
  const outerSize = outerShape.reduce((a, b) => a * b, 1);
  const outShape = [...real.shape];

  // First pass: IFFT along last axis
  const mid = ifft(real, imag);

  // Second pass: IFFT along second-to-last axis
  const totalOut = outerSize * dim0 * dim1;
  const realData = new Float64Array(totalOut);
  const imagData = new Float64Array(totalOut);

  for (let outer = 0; outer < outerSize; outer++) {
    let midRealBase = mid.real.offset;
    let midImagBase = mid.imag.offset;
    let rem = outer;
    for (let d = axis0 - 1; d >= 0; d--) {
      const dimSize = mid.real.shape[d] ?? 1;
      const coord = rem % dimSize;
      rem = Math.floor(rem / dimSize);
      midRealBase += coord * (mid.real.strides[d] ?? 0);
      midImagBase += coord * (mid.imag.strides[d] ?? 0);
    }

    for (let j = 0; j < dim1; j++) {
      const colRe = new Float64Array(dim0);
      const colIm = new Float64Array(dim0);

      const realRowStride = mid.real.strides[axis0] ?? 0;
      const realColStride = mid.real.strides[axis1] ?? 0;
      const imagRowStride = mid.imag.strides[axis0] ?? 0;
      const imagColStride = mid.imag.strides[axis1] ?? 0;

      for (let i = 0; i < dim0; i++) {
        colRe[i] = readVal(mid.real, midRealBase + i * realRowStride + j * realColStride);
        colIm[i] = readVal(mid.imag, midImagBase + i * imagRowStride + j * imagColStride);
      }

      const result = fft1d(colRe, colIm, true);
      const invN = 1 / dim0;

      const outBaseOuter = outer * dim0 * dim1;
      for (let i = 0; i < dim0; i++) {
        realData[outBaseOuter + i * dim1 + j] = (result.real[i] ?? 0) * invN;
        imagData[outBaseOuter + i * dim1 + j] = (result.imag[i] ?? 0) * invN;
      }
    }
  }

  const outDtype = fftDtype(real.dtype);
  return {
    real: makeTensor(realData, outShape, outDtype),
    imag: makeTensor(imagData, outShape, outDtype),
  };
}

/**
 * Compute the n-dimensional discrete Fourier Transform.
 *
 * FFT is computed along each of the last `axes` dimensions sequentially.
 * If `axes` is not provided, all dimensions are transformed.
 *
 * @param t - Input real-valued tensor
 * @param axes - Axes over which to compute the FFT. Default: all axes.
 * @returns Object with `real` and `imag` tensors of the same shape.
 *
 * @example
 * ```ts
 * const t = tensor([[1, 0], [0, 0]]);
 * const { real, imag } = fftn(t);
 * ```
 */
export function fftn(t: Tensor, axes?: number[]): FFTResult {
  if (t.dtype === "string") {
    throw new InvalidParameterError("fftn requires numeric input", "t");
  }
  if (t.ndim === 0) {
    throw new ShapeError("fftn requires at least 1D input");
  }

  const allAxes = normalizeFftAxes(axes, t.ndim);

  const outDtype = fftDtype(t.dtype);
  const shape = t.shape.slice();
  const totalSize = shape.reduce((a, b) => a * b, 1);

  // Initialize real/imag arrays from input (stride-aware: input may be a view)
  const inLogicalStrides = computeStrides(t.shape);
  const realArr = new Float64Array(totalSize);
  const imagArr = new Float64Array(totalSize);
  for (let i = 0; i < totalSize; i++) {
    realArr[i] = readLogical(t, i, inLogicalStrides);
  }

  // Apply 1D FFT along each axis
  for (const axis of allAxes) {
    const dimLen = shape[axis] ?? 1;
    if (dimLen <= 1) continue;

    // Compute strides for the current working shape
    const strides: number[] = [];
    let s = 1;
    for (let d = shape.length - 1; d >= 0; d--) {
      strides[d] = s;
      s *= shape[d] ?? 1;
    }

    // Compute outer iteration count (all dims except the target axis)
    const outerSize = totalSize / dimLen;

    for (let outer = 0; outer < outerSize; outer++) {
      // Map outer index to multi-index skipping the target axis
      let rem = outer;
      let baseIdx = 0;
      for (let d = shape.length - 1; d >= 0; d--) {
        if (d === axis) continue;
        const dimSize = shape[d] ?? 1;
        const coord = rem % dimSize;
        rem = Math.floor(rem / dimSize);
        baseIdx += coord * (strides[d] ?? 0);
      }

      // Extract slice along axis
      const re = new Float64Array(dimLen);
      const im = new Float64Array(dimLen);
      const axisStride = strides[axis] ?? 1;
      for (let i = 0; i < dimLen; i++) {
        const idx = baseIdx + i * axisStride;
        re[i] = realArr[idx] ?? 0;
        im[i] = imagArr[idx] ?? 0;
      }

      const result = fft1d(re, im, false);

      // Write back
      for (let i = 0; i < dimLen; i++) {
        const idx = baseIdx + i * axisStride;
        realArr[idx] = result.real[i] ?? 0;
        imagArr[idx] = result.imag[i] ?? 0;
      }
    }
  }

  return {
    real: makeTensor(realArr, shape, outDtype),
    imag: makeTensor(imagArr, shape, outDtype),
  };
}

/**
 * Compute the n-dimensional inverse discrete Fourier Transform.
 *
 * @param real - Real part of the n-D spectrum
 * @param imag - Imaginary part of the n-D spectrum
 * @param axes - Axes over which to compute the IFFT. Default: all axes.
 * @returns Object with `real` and `imag` tensors of the same shape.
 */
export function ifftn(real: Tensor, imag: Tensor, axes?: number[]): FFTResult {
  if (real.dtype === "string" || imag.dtype === "string") {
    throw new InvalidParameterError("ifftn requires numeric input", "real");
  }
  if (real.ndim === 0 || imag.ndim === 0) {
    throw new ShapeError("ifftn requires at least 1D input");
  }

  const allAxes = normalizeFftAxes(axes, real.ndim);

  const outDtype = fftDtype(real.dtype);
  const shape = real.shape.slice();
  const totalSize = shape.reduce((a, b) => a * b, 1);

  const realLogicalStrides = computeStrides(real.shape);
  const imagLogicalStrides = computeStrides(imag.shape);
  const realArr = new Float64Array(totalSize);
  const imagArr = new Float64Array(totalSize);
  for (let i = 0; i < totalSize; i++) {
    realArr[i] = readLogical(real, i, realLogicalStrides);
    imagArr[i] = readLogical(imag, i, imagLogicalStrides);
  }

  for (const axis of allAxes) {
    const dimLen = shape[axis] ?? 1;
    if (dimLen <= 1) continue;

    const strides: number[] = [];
    let s = 1;
    for (let d = shape.length - 1; d >= 0; d--) {
      strides[d] = s;
      s *= shape[d] ?? 1;
    }

    const outerSize = totalSize / dimLen;

    for (let outer = 0; outer < outerSize; outer++) {
      let rem = outer;
      let baseIdx = 0;
      for (let d = shape.length - 1; d >= 0; d--) {
        if (d === axis) continue;
        const dimSize = shape[d] ?? 1;
        const coord = rem % dimSize;
        rem = Math.floor(rem / dimSize);
        baseIdx += coord * (strides[d] ?? 0);
      }

      const re = new Float64Array(dimLen);
      const im = new Float64Array(dimLen);
      const axisStride = strides[axis] ?? 1;
      for (let i = 0; i < dimLen; i++) {
        const idx = baseIdx + i * axisStride;
        re[i] = realArr[idx] ?? 0;
        im[i] = imagArr[idx] ?? 0;
      }

      const result = fft1d(re, im, true);
      const invN = 1 / dimLen;

      for (let i = 0; i < dimLen; i++) {
        const idx = baseIdx + i * axisStride;
        realArr[idx] = (result.real[i] ?? 0) * invN;
        imagArr[idx] = (result.imag[i] ?? 0) * invN;
      }
    }
  }

  return {
    real: makeTensor(realArr, shape, outDtype),
    imag: makeTensor(imagArr, shape, outDtype),
  };
}
