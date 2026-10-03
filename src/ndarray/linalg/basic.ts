/**
 * @see {@link https://deepbox.dev/docs/ndarray-tensor | Deepbox documentation}
 */

import type { DType, Shape } from "../../core";
import {
  DataValidationError,
  DeviceError,
  DTypeError,
  dtypeToTypedArrayCtor,
  getBigIntElement,
  InvalidParameterError,
  ShapeError,
} from "../../core";

import { dispatchDot, dispatchMatmul } from "../ops/device_dispatch";
import { Tensor } from "../tensor/Tensor";
import { contract } from "./index";

const INT64_MIN = -(1n << 63n);
const INT64_MAX = (1n << 63n) - 1n;

/**
 * Matrix multiplication of two 2-D tensors.
 *
 * This is the strict 2-D variant: both operands must be 2-D. Use `dot` from
 * `deepbox/ndarray` for vectors and batched operands.
 *
 * Both operands must have the same dtype. The only mixed case allowed is
 * float32 with float64, which gives float64. Float products are accumulated in
 * float64, int32 results wrap around like NumPy, bool results are the logical
 * product, and int64 results throw if they leave the int64 range.
 *
 * @param a - Left matrix of shape (m, k)
 * @param b - Right matrix of shape (k, n)
 * @returns Product of shape (m, n)
 *
 * @throws {ShapeError} If an operand is not 2-D or the inner dimensions differ
 * @throws {DTypeError} If a dtype is `string` or the dtypes are incompatible
 * @throws {DataValidationError} If an int64 result overflows
 *
 * @example
 * ```ts
 * const c = matmul(tensor([[1, 2], [3, 4]]), tensor([[5, 6], [7, 8]]));
 * // [[19, 22], [43, 50]]
 * ```
 */
export function matmul(a: Tensor, b: Tensor): Tensor {
  if (a.device !== "cpu" || b.device !== "cpu") {
    const onDevice = dispatchMatmul(a, b);
    if (onDevice) return onDevice;
  }

  if (a.ndim !== 2 || b.ndim !== 2) {
    throw new ShapeError("matmul requires 2D tensors");
  }

  if (a.shape[1] !== b.shape[0]) {
    throw ShapeError.mismatch(a.shape, b.shape, "matmul");
  }

  return contract(a, b, "matmul");
}

/**
 * Inner product of two 1-D tensors, returned as a 0-d tensor.
 *
 * This is the strict 1-D variant. Use `dot` from `deepbox/ndarray` for matrices
 * and batched operands. Dtype rules are the same as for {@link matmul}.
 *
 * @param a - First vector of length n
 * @param b - Second vector of length n
 * @returns 0-d tensor holding the sum of `a[i] * b[i]`
 *
 * @throws {ShapeError} If an operand is not 1-D or the lengths differ
 * @throws {DTypeError} If a dtype is `string` or the dtypes are incompatible
 * @throws {DataValidationError} If an int64 result overflows
 */
export function dot(a: Tensor, b: Tensor): Tensor {
  if (a.ndim !== 1 || b.ndim !== 1) {
    throw new ShapeError("dot requires 1D tensors");
  }

  if (a.size !== b.size) {
    throw ShapeError.mismatch(a.shape, b.shape, "dot");
  }

  if (a.device !== "cpu" || b.device !== "cpu") {
    const onDevice = dispatchDot(a, b);
    if (onDevice) return onDevice;
  }

  return contract(a, b, "dot");
}

/** Axes argument of {@link tensordot}: a count, or one axis list per operand. */
export type TensordotAxes =
  | number
  | readonly [number | readonly number[], number | readonly number[]];

/** Rank used to pick the result dtype when two small integer dtypes are mixed. */
const SMALL_INT_RANK: Partial<Record<DType, number>> = { bool: 0, uint8: 1, int32: 2 };

/** Floating dtypes narrower than float64, which absorb bool and uint8 operands. */
const NARROW_FLOATS: ReadonlySet<DType> = new Set<DType>(["float16", "bfloat16", "float32"]);

/**
 * Result dtype of {@link tensordot}, following NumPy's promotion for the cases
 * that can be represented exactly.
 *
 * Equal dtypes are kept. bool, uint8 and int32 mixed together give the wider
 * of the two. A float16, bfloat16 or float32 operand combined with bool or
 * uint8 keeps its float dtype. Everything else (any other mix of float and
 * integer dtypes, or int64 with another dtype) gives float64.
 */
function resolveTensordotDtype(a: Exclude<DType, "string">, b: Exclude<DType, "string">): DType {
  if (a === b) return a;
  const ra = SMALL_INT_RANK[a];
  const rb = SMALL_INT_RANK[b];
  if (ra !== undefined && rb !== undefined) return ra >= rb ? a : b;
  if (NARROW_FLOATS.has(a) && (b === "bool" || b === "uint8")) return a;
  if (NARROW_FLOATS.has(b) && (a === "bool" || a === "uint8")) return b;
  return "float64";
}

/** Validate one axis list, wrap negative axes and reject duplicates. */
function normalizeTensordotAxes(
  spec: number | readonly number[],
  ndim: number,
  operand: "a" | "b",
  axes: unknown
): number[] {
  const list = typeof spec === "number" ? [spec] : Array.from(spec);
  const seen = new Set<number>();
  const out: number[] = [];
  for (const ax of list) {
    if (!Number.isInteger(ax)) {
      throw new InvalidParameterError(
        `tensordot: axes for ${operand} must be integers; received ${String(ax)}`,
        "axes",
        axes
      );
    }
    const norm = ax < 0 ? ax + ndim : ax;
    if (norm < 0 || norm >= ndim) {
      throw new InvalidParameterError(
        `tensordot: axis ${ax} is out of range for ${operand} with ${ndim} dimension(s)`,
        "axes",
        axes
      );
    }
    if (seen.has(norm)) {
      throw new InvalidParameterError(
        `tensordot: axis ${ax} is repeated in the axes for ${operand}`,
        "axes",
        axes
      );
    }
    seen.add(norm);
    out.push(norm);
  }
  return out;
}

/** Flat data offsets of every index tuple over `axes`, in row-major order of those axes. */
function axisOffsets(
  shape: readonly number[],
  strides: readonly number[],
  axes: readonly number[]
) {
  let offsets: number[] = [0];
  for (const ax of axes) {
    const dim = shape[ax] ?? 0;
    const stride = strides[ax] ?? 0;
    const next: number[] = new Array(offsets.length * dim);
    let idx = 0;
    for (const o of offsets) {
      for (let i = 0; i < dim; i++) next[idx++] = o + i * stride;
    }
    offsets = next;
  }
  return offsets;
}

/**
 * Copy `t` into a dense row-major (rows x cols) matrix, where the row index runs
 * over `rowAxes` and the column index over `colAxes` (both in the listed order).
 * int64 data stays exact (BigInt64Array) when `exactInt64` is set and is converted
 * to float64 otherwise.
 */
function packMatrix(
  t: Tensor,
  rowAxes: readonly number[],
  colAxes: readonly number[],
  exactInt64: boolean
): { rows: number; cols: number; values: Float64Array | BigInt64Array } {
  const rowOffs = axisOffsets(t.shape, t.strides, rowAxes);
  const colOffs = axisOffsets(t.shape, t.strides, colAxes);
  const rows = rowOffs.length;
  const cols = colOffs.length;
  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError("tensordot is not defined for string dtype");
  }
  if (data instanceof BigInt64Array && exactInt64) {
    const values = new BigInt64Array(rows * cols);
    for (let r = 0; r < rows; r++) {
      const base = t.offset + (rowOffs[r] as number);
      for (let c = 0; c < cols; c++) {
        values[r * cols + c] = getBigIntElement(data, base + (colOffs[c] as number));
      }
    }
    return { rows, cols, values };
  }
  const values = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const base = t.offset + (rowOffs[r] as number);
    for (let c = 0; c < cols; c++) {
      const raw = data[base + (colOffs[c] as number)];
      values[r * cols + c] = typeof raw === "bigint" ? Number(raw) : (raw as number);
    }
  }
  return { rows, cols, values };
}

/**
 * Tensor dot product along specified axes.
 *
 * Contracts `a` and `b` over the given axes, generalizing matmul and dot.
 *
 * If `axes` is a number N, contracts the last N axes of `a` with the first N axes of `b`.
 * If `axes` is a pair, `axes[0]` lists the axes of `a` and `axes[1]` the matching axes of
 * `b`; each entry may be a single axis or an array, and negative axes count from the end.
 * The output shape is the remaining axes of `a` followed by the remaining axes of `b`.
 *
 * Result dtype: equal dtypes are kept (float32 stays float32, int64 stays int64 and is
 * computed exactly), bool, uint8 and int32 mixed together give the wider one, a float16,
 * bfloat16 or float32 operand with bool or uint8 keeps its float dtype, and any other
 * combination (including int64 with another dtype) gives float64. Floats are accumulated
 * in float64.
 *
 * @param a - First tensor
 * @param b - Second tensor
 * @param axes - Number of axes to contract, or [axesA, axesB] (default: 2)
 * @returns Contracted tensor
 *
 * @throws {InvalidParameterError} If `axes` holds non-integer, out-of-range or repeated axes
 * @throws {ShapeError} If contracted axes have mismatched sizes
 * @throws {DTypeError} If tensors have string dtype
 * @throws {DeviceError} If a tensor lives on a device; call `await t.cpu()` first
 * @throws {DataValidationError} If an int64 result overflows
 *
 * @example
 * ```ts
 * const a = tensor([[1, 2, 3], [4, 5, 6]]);
 * const b = tensor([[1, 4], [2, 5], [3, 6]]);
 * tensordot(a, b, 1);           // same as matmul: [[14, 32], [32, 77]]
 * tensordot(a, b, [[1], [0]]);  // same result
 * tensordot(a, b, 0);           // outer product, shape [2, 3, 3, 2]
 * ```
 */
export function tensordot(a: Tensor, b: Tensor, axes: TensordotAxes = 2): Tensor {
  if (a.dtype === "string" || b.dtype === "string") {
    throw new DTypeError("tensordot is not defined for string dtype");
  }
  if (a.isDeviceTensor || b.isDeviceTensor) {
    throw new DeviceError(
      "tensordot is not available for device tensors; call `await t.cpu()` first"
    );
  }

  let axesA: number[];
  let axesB: number[];

  if (typeof axes === "number") {
    if (!Number.isInteger(axes) || axes < 0) {
      throw new InvalidParameterError("axes must be a non-negative integer", "axes", axes);
    }
    if (axes > a.ndim || axes > b.ndim) {
      throw new ShapeError(
        `tensordot: axes=${axes} exceeds the number of dimensions ` +
          `(a has ${a.ndim}, b has ${b.ndim})`
      );
    }
    axesA = [];
    axesB = [];
    for (let i = 0; i < axes; i++) {
      axesA.push(a.ndim - axes + i);
      axesB.push(i);
    }
  } else {
    if (!Array.isArray(axes) || axes.length !== 2) {
      throw new InvalidParameterError(
        "axes must be a number or a pair [axesA, axesB]",
        "axes",
        axes
      );
    }
    axesA = normalizeTensordotAxes(axes[0], a.ndim, "a", axes);
    axesB = normalizeTensordotAxes(axes[1], b.ndim, "b", axes);
    if (axesA.length !== axesB.length) {
      throw new InvalidParameterError(
        "axes[0] and axes[1] must have the same length",
        "axes",
        axes
      );
    }
  }

  for (let i = 0; i < axesA.length; i++) {
    const dimA = a.shape[axesA[i] as number];
    const dimB = b.shape[axesB[i] as number];
    if (dimA !== dimB) {
      throw new ShapeError(
        `tensordot: contracted axes have mismatched sizes: axis ${axesA[i]} of a is ${dimA}, ` +
          `axis ${axesB[i]} of b is ${dimB}`
      );
    }
  }

  const freeA: number[] = [];
  const freeB: number[] = [];
  for (let i = 0; i < a.ndim; i++) if (!axesA.includes(i)) freeA.push(i);
  for (let i = 0; i < b.ndim; i++) if (!axesB.includes(i)) freeB.push(i);

  const outShape: number[] = [];
  for (const ax of freeA) outShape.push(a.shape[ax] as number);
  for (const ax of freeB) outShape.push(b.shape[ax] as number);

  const outDtype = resolveTensordotDtype(a.dtype, b.dtype);

  // Reduce to one matrix product: A as (freeA x contracted), B as (contracted x freeB).
  const exactInt64 = outDtype === "int64";
  const A = packMatrix(a, freeA, axesA, exactInt64);
  const B = packMatrix(b, axesB, freeB, exactInt64);
  const rows = A.rows;
  const inner = A.cols;
  const cols = B.cols;

  const Ctor = dtypeToTypedArrayCtor(outDtype);
  const out = new Ctor(rows * cols);

  if (rows * cols > 0) {
    if (A.values instanceof BigInt64Array && B.values instanceof BigInt64Array) {
      if (!(out instanceof BigInt64Array)) {
        throw new DTypeError("Internal error: int64 tensordot requires an int64 output buffer");
      }
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          let sum = 0n;
          for (let p = 0; p < inner; p++) {
            sum += (A.values[i * inner + p] as bigint) * (B.values[p * cols + j] as bigint);
          }
          if (sum < INT64_MIN || sum > INT64_MAX) {
            throw new DataValidationError("int64 tensordot overflow");
          }
          out[i * cols + j] = sum;
        }
      }
    } else if (A.values instanceof Float64Array && B.values instanceof Float64Array) {
      if (out instanceof BigInt64Array) {
        throw new DTypeError("Internal error: unexpected int64 output buffer");
      }
      const av = A.values;
      const bv = B.values;
      if (outDtype === "int32") {
        // Exact two's-complement wraparound; float64 sums would lose low bits.
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) {
            let sum = 0;
            for (let p = 0; p < inner; p++) {
              sum = (sum + Math.imul(av[i * inner + p] as number, bv[p * cols + j] as number)) | 0;
            }
            out[i * cols + j] = sum;
          }
        }
      } else if (outDtype === "bool") {
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) {
            let any = 0;
            for (let p = 0; p < inner; p++) {
              if (av[i * inner + p] !== 0 && bv[p * cols + j] !== 0) {
                any = 1;
                break;
              }
            }
            out[i * cols + j] = any;
          }
        }
      } else {
        // i-k-j order with a float64 row accumulator keeps B reads contiguous; each
        // output element is still summed in ascending contracted-index order.
        const rowAcc = new Float64Array(cols);
        for (let i = 0; i < rows; i++) {
          rowAcc.fill(0);
          for (let p = 0; p < inner; p++) {
            const x = av[i * inner + p] as number;
            const bBase = p * cols;
            for (let j = 0; j < cols; j++) {
              rowAcc[j] = (rowAcc[j] as number) + x * (bv[bBase + j] as number);
            }
          }
          for (let j = 0; j < cols; j++) out[i * cols + j] = rowAcc[j] as number;
        }
      }
    } else {
      throw new DTypeError("Internal error: tensordot operands were packed inconsistently");
    }
  }

  const shape: Shape = outShape;
  return Tensor.fromTypedArray({
    data: out,
    shape,
    dtype: outDtype as Exclude<DType, "string">,
    device: a.device,
  });
}

/** Everything the covariance routines need from one input tensor. */
type ObservationMatrix = {
  /** Number of variables. */
  readonly m: number;
  /** Number of observations per variable. */
  readonly n: number;
  /** Row-major (m x n) values, one variable per row. */
  readonly values: Float64Array;
};

/** Read `x` as an (m variables x n observations) float64 matrix. */
function readObservations(x: Tensor, rowvar: boolean, op: string): ObservationMatrix {
  if (x.dtype === "string") {
    throw new DTypeError(`${op} is not defined for string dtype`);
  }
  if (x.isDeviceTensor) {
    throw new DeviceError(
      `${op} is not available for device tensors; call \`await t.cpu()\` first`
    );
  }
  if (x.ndim > 2) {
    throw new ShapeError(`${op} requires 1D or 2D input; got ndim=${x.ndim}`);
  }
  const data = x.data;
  if (Array.isArray(data)) {
    throw new DTypeError(`${op} is not defined for string dtype`);
  }

  // A 1-D input is one variable; a 0-d input is one variable with one observation.
  let m: number;
  let n: number;
  let varStride: number;
  let obsStride: number;
  if (x.ndim === 0) {
    m = 1;
    n = 1;
    varStride = 0;
    obsStride = 0;
  } else if (x.ndim === 1) {
    m = 1;
    n = x.shape[0] ?? 0;
    varStride = 0;
    obsStride = x.strides[0] ?? 0;
  } else if (rowvar) {
    m = x.shape[0] ?? 0;
    n = x.shape[1] ?? 0;
    varStride = x.strides[0] ?? 0;
    obsStride = x.strides[1] ?? 0;
  } else {
    m = x.shape[1] ?? 0;
    n = x.shape[0] ?? 0;
    varStride = x.strides[1] ?? 0;
    obsStride = x.strides[0] ?? 0;
  }

  const values = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    const base = x.offset + i * varStride;
    for (let j = 0; j < n; j++) {
      const raw = data[base + j * obsStride];
      values[i * n + j] = typeof raw === "bigint" ? Number(raw) : (raw as number);
    }
  }
  return { m, n, values };
}

/** Covariance matrix (m x m) of an observation matrix; centers `values` in place. */
function covarianceOf(obs: ObservationMatrix, ddof: number): Float64Array {
  const { m, n, values } = obs;

  // Center each variable. A second pass over the residuals corrects the rounding
  // error of the plain mean (corrected two-pass algorithm).
  for (let i = 0; i < m; i++) {
    const base = i * n;
    let sum = 0;
    for (let j = 0; j < n; j++) sum += values[base + j] as number;
    let mean = sum / n;
    let resid = 0;
    for (let j = 0; j < n; j++) resid += (values[base + j] as number) - mean;
    mean += resid / n;
    for (let j = 0; j < n; j++) values[base + j] = (values[base + j] as number) - mean;
  }

  const denom = n - ddof;
  const out = new Float64Array(m * m);
  for (let i = 0; i < m; i++) {
    const bi = i * n;
    for (let j = i; j < m; j++) {
      const bj = j * n;
      let sum = 0;
      for (let k = 0; k < n; k++) {
        sum += (values[bi + k] as number) * (values[bj + k] as number);
      }
      const val = sum / denom;
      out[i * m + j] = val;
      out[j * m + i] = val;
    }
  }
  return out;
}

/**
 * Compute the Pearson correlation coefficient matrix.
 *
 * Given a 1D or 2D tensor, returns the correlation matrix. With the default
 * `rowvar = true`, each row of a 2D input of shape (m, n) is a variable and each
 * column is an observation (like NumPy); with `rowvar = false` the roles are swapped.
 *
 * A variable with zero variance has no defined correlation, so its row and column
 * are NaN (as in NumPy). Off-diagonal values are clipped to [-1, 1] to remove
 * rounding noise, and the diagonal is exactly 1 for every non-constant variable.
 *
 * Note that `deepbox/stats` `corrcoef` treats columns as variables.
 *
 * @param x - Input tensor (1D or 2D)
 * @param rowvar - Whether rows are variables (default: true)
 * @returns Correlation matrix of shape (m, m) with float64 dtype
 *
 * @throws {ShapeError} If input has more than 2 dimensions
 * @throws {DTypeError} If input has string dtype
 * @throws {InvalidParameterError} If there are fewer than 2 observations
 * @throws {DeviceError} If the tensor lives on a device; call `await t.cpu()` first
 *
 * @example
 * ```ts
 * corrcoef(tensor([[1, 2, 3, 4], [4, 3, 2, 1]]));
 * // [[1, -1], [-1, 1]]
 * ```
 */
export function corrcoef(x: Tensor, rowvar: boolean = true): Tensor {
  const obs = readObservations(x, rowvar, "corrcoef");
  if (obs.n < 2) {
    throw new InvalidParameterError(
      `corrcoef requires at least 2 observations; got ${obs.n}`,
      "nObs",
      obs.n
    );
  }
  const m = obs.m;
  const covMatrix = covarianceOf(obs, 1);

  const std = new Float64Array(m);
  for (let i = 0; i < m; i++) std[i] = Math.sqrt(covMatrix[i * m + i] as number);

  // Dividing by each standard deviation separately avoids overflowing or
  // underflowing the product of two variances.
  const out = new Float64Array(m * m);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      if (i === j) {
        const s = std[i] as number;
        out[i * m + j] = s > 0 && Number.isFinite(s) ? 1 : Number.NaN;
        continue;
      }
      const r = (covMatrix[i * m + j] as number) / (std[i] as number) / (std[j] as number);
      out[i * m + j] = r > 1 ? 1 : r < -1 ? -1 : r;
    }
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: [m, m],
    dtype: "float64",
    device: x.device,
  });
}

/**
 * Compute the covariance matrix.
 *
 * Given a 1D or 2D tensor, returns the covariance matrix. With the default
 * `rowvar = true`, each row of a 2D input of shape (m, n) is a variable and each
 * column is an observation (like NumPy); with `rowvar = false` the roles are swapped.
 *
 * Uses Bessel's correction (ddof=1) by default. Entries are `sum / (n - ddof)`
 * over the n observations, computed from mean-centered values.
 *
 * Note that `deepbox/stats` `cov` treats columns as variables.
 *
 * @param x - Input tensor (1D or 2D)
 * @param ddof - Delta degrees of freedom (default: 1)
 * @param rowvar - Whether rows are variables (default: true)
 * @returns Covariance matrix of shape (m, m) with float64 dtype
 *
 * @throws {ShapeError} If input has more than 2 dimensions
 * @throws {DTypeError} If input has string dtype
 * @throws {InvalidParameterError} If `ddof` is negative or not finite, or the number of
 *   observations is not larger than `ddof`
 * @throws {DeviceError} If the tensor lives on a device; call `await t.cpu()` first
 *
 * @example
 * ```ts
 * cov(tensor([[1, 2, 3], [4, 5, 6]])); // [[1, 1], [1, 1]]
 * cov(tensor([2, 4, 6]), 0);           // [[8 / 3]]
 * ```
 */
export function cov(x: Tensor, ddof: number = 1, rowvar: boolean = true): Tensor {
  if (!Number.isFinite(ddof) || ddof < 0) {
    throw new InvalidParameterError("ddof must be a non-negative finite number", "ddof", ddof);
  }
  const obs = readObservations(x, rowvar, "cov");
  if (obs.n <= ddof) {
    throw new InvalidParameterError(
      `cov: ddof=${ddof} must be smaller than the number of observations (${obs.n})`,
      "ddof",
      ddof
    );
  }
  const m = obs.m;
  const out = covarianceOf(obs, ddof);

  return Tensor.fromTypedArray({
    data: out,
    shape: [m, m],
    dtype: "float64",
    device: x.device,
  });
}
