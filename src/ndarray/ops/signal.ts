/**
 * Signal processing: window functions, convolution, and correlation.
 * @module ndarray/ops/signal
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */
import { InvalidParameterError } from "../../core";
import { Tensor } from "../tensor/Tensor";

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

function readNum(t: Tensor, i: number): number {
  const raw = t.data[t.offset + i];
  return typeof raw === "number" ? raw : typeof raw === "bigint" ? Number(raw) : 0;
}

function besselI0(x: number): number {
  let sum = 1,
    term = 1;
  const h = x / 2;
  for (let k = 1; k <= 25; k++) {
    term *= (h / k) * (h / k);
    sum += term;
    if (term < 1e-16 * sum) break;
  }
  return sum;
}

export function hannWindow(n: number): Tensor {
  validateWinLen(n, "hannWindow");
  if (n === 1) return mkTensor(new Float64Array([1]), 1);
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) d[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  return mkTensor(d, n);
}

export function hammingWindow(n: number): Tensor {
  validateWinLen(n, "hammingWindow");
  if (n === 1) return mkTensor(new Float64Array([1]), 1);
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) d[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1));
  return mkTensor(d, n);
}

export function blackmanWindow(n: number): Tensor {
  validateWinLen(n, "blackmanWindow");
  if (n === 1) return mkTensor(new Float64Array([1]), 1);
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const a = (2 * Math.PI * i) / (n - 1);
    d[i] = 0.42 - 0.5 * Math.cos(a) + 0.08 * Math.cos(2 * a);
  }
  return mkTensor(d, n);
}

export function bartlettWindow(n: number): Tensor {
  validateWinLen(n, "bartlettWindow");
  if (n === 1) return mkTensor(new Float64Array([1]), 1);
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) d[i] = 1 - Math.abs((2 * i) / (n - 1) - 1);
  return mkTensor(d, n);
}

export function kaiserWindow(n: number, beta = 12): Tensor {
  validateWinLen(n, "kaiserWindow");
  if (n === 1) return mkTensor(new Float64Array([1]), 1);
  const d = new Float64Array(n);
  const i0b = besselI0(beta);
  for (let i = 0; i < n; i++) {
    const r = (2 * i) / (n - 1) - 1;
    d[i] = besselI0(beta * Math.sqrt(1 - r * r)) / i0b;
  }
  return mkTensor(d, n);
}

function validateConvInputs(a: Tensor, v: Tensor, name: string): void {
  if (a.dtype === "string" || v.dtype === "string")
    throw new InvalidParameterError(`${name} requires numeric input`, "a");
  if (a.ndim !== 1 || v.ndim !== 1)
    throw new InvalidParameterError(`${name} requires 1-D inputs`, "a");
}

function trimResult(
  full: Float64Array,
  fullLen: number,
  aLen: number,
  vLen: number,
  mode: "full" | "same" | "valid"
): Tensor {
  if (mode === "full") return mkTensor(full, fullLen);
  if (mode === "same") {
    const len = Math.max(aLen, vLen);
    const start = Math.floor((fullLen - len) / 2);
    return mkTensor(full.slice(start, start + len), len);
  }
  // valid
  const len = Math.max(aLen, vLen) - Math.min(aLen, vLen) + 1;
  const start = Math.min(aLen, vLen) - 1;
  return mkTensor(full.slice(start, start + len), len);
}

export function convolve(a: Tensor, v: Tensor, mode: "full" | "same" | "valid" = "full"): Tensor {
  validateConvInputs(a, v, "convolve");
  const aLen = a.size,
    vLen = v.size;
  const aD = new Float64Array(aLen);
  const vD = new Float64Array(vLen);
  for (let i = 0; i < aLen; i++) aD[i] = readNum(a, i);
  for (let i = 0; i < vLen; i++) vD[i] = readNum(v, i);
  const fullLen = aLen + vLen - 1;
  const full = new Float64Array(fullLen);
  for (let i = 0; i < aLen; i++)
    for (let j = 0; j < vLen; j++) full[i + j] = (full[i + j] ?? 0) + (aD[i] ?? 0) * (vD[j] ?? 0);
  return trimResult(full, fullLen, aLen, vLen, mode);
}

export function correlate(a: Tensor, v: Tensor, mode: "full" | "same" | "valid" = "full"): Tensor {
  validateConvInputs(a, v, "correlate");
  const aLen = a.size,
    vLen = v.size;
  const aD = new Float64Array(aLen);
  const vD = new Float64Array(vLen);
  for (let i = 0; i < aLen; i++) aD[i] = readNum(a, i);
  for (let i = 0; i < vLen; i++) vD[i] = readNum(v, i);
  const fullLen = aLen + vLen - 1;
  const full = new Float64Array(fullLen);
  for (let i = 0; i < aLen; i++)
    for (let j = 0; j < vLen; j++)
      full[i + j] = (full[i + j] ?? 0) + (aD[i] ?? 0) * (vD[vLen - 1 - j] ?? 0);
  return trimResult(full, fullLen, aLen, vLen, mode);
}
