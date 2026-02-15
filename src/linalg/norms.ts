import {
  type Axis,
  InvalidParameterError,
  normalizeAxes,
  normalizeAxis,
  ShapeError,
} from "../core";
import { type Tensor, tensor } from "../ndarray";
import { isContiguous } from "../ndarray/tensor/strides";
import { assertFiniteTensor, at, getDim, getStride, toDenseVector1D } from "./_internal";
import { svd } from "./decomposition/svd";

/**
 * Matrix or vector norm.
 *
 * Computes various matrix and vector norms.
 *
 * **Parameters**:
 * @param x - Input array
 * @param ord - Order of the norm:
 *   For vectors:
 *   - undefined or 2: L2 norm (Euclidean)
 *   - 1: L1 norm (Manhattan)
 *   - Infinity: Max norm
 *   - -Infinity: Min norm
 *   - 0: L0 "norm" (number of non-zero elements)
 *   - p: Lp norm (p > 0). Negative p (except -Infinity) is invalid.
 *   For matrices:
 *   - 'fro': Frobenius norm
 *   - 'nuc': Nuclear norm (sum of singular values)
 *   - 1: Max column sum
 *   - -1: Min column sum
 *   - 2: Largest singular value
 *   - -2: Smallest singular value
 *   - Infinity: Max row sum
 *   - -Infinity: Min row sum
 * @param axis - Axis along which to compute norm
 * @param keepdims - Keep reduced dimensions
 *
 * **Returns**: Norm value (scalar or tensor)
 *
 * @throws {DTypeError} If input has string dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 * @throws {InvalidParameterError} If norm order or axis values are invalid
 * @throws {ShapeError} If axis configuration is incompatible with input
 *
 * @see {@link https://deepbox.dev/docs/linalg-properties | Deepbox Linear Algebra}
 */
export function norm(x: Tensor, ord?: number | "fro" | "nuc"): number;
export function norm(
  x: Tensor,
  ord: number | "fro" | "nuc" | undefined,
  axis: Axis | Axis[],
  keepdims?: boolean
): Tensor | number;
export function norm(
  x: Tensor,
  ord?: number | "fro" | "nuc",
  axis?: Axis | Axis[],
  keepdims?: boolean
): Tensor | number;
export function norm(
  x: Tensor,
  ord?: number | "fro" | "nuc",
  axis?: Axis | Axis[],
  keepdims = false
): Tensor | number {
  assertFiniteTensor(x, "norm()");

  const axes = axis !== undefined ? normalizeAxes(axis, x.ndim) : undefined;

  let p: number | "fro" | "nuc";
  if (ord === undefined) {
    if (axes === undefined) {
      p = x.ndim === 1 ? 2 : "fro";
    } else if (axes.length === 2) {
      p = "fro";
    } else {
      p = 2;
    }
  } else {
    p = ord;
  }

  const normalizeVectorOrder = (order: number | "fro"): number => {
    const ordValue = order === "fro" ? 2 : order;

    if (Number.isNaN(ordValue)) {
      throw new InvalidParameterError("ord must be a valid number", "ord", ordValue);
    }
    if (!Number.isFinite(ordValue)) {
      if (ordValue === Number.POSITIVE_INFINITY || ordValue === Number.NEGATIVE_INFINITY) {
        return ordValue;
      }
      throw new InvalidParameterError("ord must be finite or ±Infinity", "ord", ordValue);
    }
    if (ordValue < 0) {
      throw new InvalidParameterError(
        "Vector norm order must be non-negative or ±Infinity",
        "ord",
        ordValue
      );
    }
    return ordValue;
  };

  const vectorNormValue = (
    values: Iterable<number> & { readonly length: number },
    order: number
  ): number => {
    const n = values.length;
    if (n === 0) return 0;

    if (order === 0) {
      let count = 0;
      for (const v of values) if (v !== 0) count++;
      return count;
    }
    if (order === 1) {
      let sum = 0;
      for (const v of values) sum += Math.abs(v);
      return sum;
    }
    if (order === 2) {
      let sum = 0;
      for (const v of values) sum += v * v;
      return Math.sqrt(sum);
    }
    if (order === Number.POSITIVE_INFINITY) {
      let max = 0;
      for (const v of values) max = Math.max(max, Math.abs(v));
      return max;
    }
    if (order === Number.NEGATIVE_INFINITY) {
      let min = Infinity;
      for (const v of values) min = Math.min(min, Math.abs(v));
      return min === Infinity ? 0 : min;
    }

    let sum = 0;
    for (const v of values) sum += Math.abs(v) ** order;
    return sum ** (1 / order);
  };

  const vectorNormAxis = (axisValue: number, order: number, keep: boolean): Tensor | number => {
    const ax = normalizeAxis(axisValue, x.ndim);
    const axisDim = getDim(x, ax, "norm()");

    const outShape: number[] = [];
    for (let i = 0; i < x.ndim; i++) {
      if (i === ax) {
        if (keep) outShape.push(1);
      } else {
        outShape.push(getDim(x, i, "norm()"));
      }
    }

    const outSize = outShape.length === 0 ? 1 : outShape.reduce((a, b) => a * b, 1);
    const out = new Float64Array(outSize);

    const outStrides = new Array<number>(outShape.length);
    let stride = 1;
    for (let i = outShape.length - 1; i >= 0; i--) {
      outStrides[i] = stride;
      stride *= outShape[i] ?? 0;
    }

    for (let outFlat = 0; outFlat < outSize; outFlat++) {
      let rem = outFlat;
      const outIdx: number[] = new Array(outShape.length);

      let i = 0;
      for (const s of outStrides) {
        outIdx[i] = Math.floor(rem / s);
        rem %= s;
        i++;
      }

      const inIdx = new Array<number>(x.ndim);
      if (keep) {
        for (let i = 0; i < x.ndim; i++) {
          inIdx[i] = i === ax ? 0 : (outIdx[i] ?? 0);
        }
      } else {
        let outAxis = 0;
        for (let i = 0; i < x.ndim; i++) {
          if (i === ax) {
            inIdx[i] = 0;
          } else {
            inIdx[i] = outIdx[outAxis] ?? 0;
            outAxis++;
          }
        }
      }

      if (axisDim === 0) {
        out[outFlat] = 0;
        continue;
      }

      let baseOffset = x.offset;
      for (let i = 0; i < x.ndim; i++) {
        baseOffset += (inIdx[i] ?? 0) * getStride(x, i, "norm()");
      }

      const axisStride = getStride(x, ax, "norm()");

      if (order === 0) {
        let count = 0;
        for (let k = 0; k < axisDim; k++) {
          if (Number(x.data[baseOffset + k * axisStride]) !== 0) count++;
        }
        out[outFlat] = count;
      } else if (order === 1) {
        let sum = 0;
        for (let k = 0; k < axisDim; k++) {
          sum += Math.abs(Number(x.data[baseOffset + k * axisStride]));
        }
        out[outFlat] = sum;
      } else if (order === 2) {
        let sum = 0;
        for (let k = 0; k < axisDim; k++) {
          const v = Number(x.data[baseOffset + k * axisStride]);
          sum += v * v;
        }
        out[outFlat] = Math.sqrt(sum);
      } else if (order === Number.POSITIVE_INFINITY) {
        let max = 0;
        for (let k = 0; k < axisDim; k++) {
          max = Math.max(max, Math.abs(Number(x.data[baseOffset + k * axisStride])));
        }
        out[outFlat] = max;
      } else if (order === Number.NEGATIVE_INFINITY) {
        let min = Infinity;
        for (let k = 0; k < axisDim; k++) {
          min = Math.min(min, Math.abs(Number(x.data[baseOffset + k * axisStride])));
        }
        out[outFlat] = min === Infinity ? 0 : min;
      } else {
        let sum = 0;
        for (let k = 0; k < axisDim; k++) {
          sum += Math.abs(Number(x.data[baseOffset + k * axisStride])) ** order;
        }
        out[outFlat] = sum ** (1 / order);
      }
    }

    if (outShape.length === 0 && !keep) return at(out, 0);
    return tensor(out).view(outShape);
  };

  const collectValues = (): number[] => {
    const values = new Array<number>(x.size);
    if (x.size === 0) return values;

    if (x.ndim === 0) {
      values[0] = Number(x.data[x.offset]);
      return values;
    }

    // Fast path: contiguous zero-offset numeric tensor
    const data = x.data;
    if (
      !Array.isArray(data) &&
      !(data instanceof BigInt64Array) &&
      x.offset === 0 &&
      isContiguous(x.shape, x.strides)
    ) {
      for (let i = 0; i < x.size; i++) {
        values[i] = data[i] as number;
      }
      return values;
    }

    const idx = new Array<number>(x.ndim).fill(0);
    let offset = x.offset;

    for (let count = 0; count < x.size; count++) {
      values[count] = Number(data[offset]);

      for (let d = x.ndim - 1; d >= 0; d--) {
        const dim = getDim(x, d, "norm()");
        const stride = getStride(x, d, "norm()");
        const idxVal = idx[d] ?? 0;
        const nextIdx = idxVal + 1;

        idx[d] = nextIdx;
        offset += stride;

        if (nextIdx < dim) break;

        offset -= nextIdx * stride;
        idx[d] = 0;
      }
    }

    return values;
  };

  if (axis !== undefined) {
    if (p === "nuc") {
      throw new InvalidParameterError(
        "axis is only supported for vector norms, not nuclear norm",
        "axis",
        axis
      );
    }
    if (axes === undefined) {
      throw new ShapeError("axis has invalid length for input");
    }

    if (axes.length === 0) {
      const ordValue = normalizeVectorOrder(p === "fro" ? 2 : p);
      const value = vectorNormValue(collectValues(), ordValue);
      if (!keepdims) return value;

      const kdShape = new Array<number>(x.ndim).fill(1);
      return tensor([value]).view(kdShape);
    }

    if (axes.length === 1) {
      const ax = axes[0];
      if (ax === undefined) throw new ShapeError("axis has invalid length for input");
      const ordValue = normalizeVectorOrder(p === "fro" ? 2 : p);
      return vectorNormAxis(ax, ordValue, keepdims);
    }

    if (axes.length === 2) {
      if (x.ndim < 2) throw new ShapeError("axis has invalid length for input");

      const ax0 = axes[0];
      const ax1 = axes[1];
      if (ax0 === undefined || ax1 === undefined)
        throw new ShapeError("axis has invalid length for input");

      const dimRow = getDim(x, ax0, "norm()");
      const dimCol = getDim(x, ax1, "norm()");
      const strideRow = getStride(x, ax0, "norm()");
      const strideCol = getStride(x, ax1, "norm()");

      const outerShape: number[] = [];
      const outerStrides: number[] = [];
      for (let i = 0; i < x.ndim; i++) {
        if (i !== ax0 && i !== ax1) {
          outerShape.push(getDim(x, i, "norm()"));
          outerStrides.push(getStride(x, i, "norm()"));
        }
      }

      const outerSize = outerShape.length === 0 ? 1 : outerShape.reduce((a, b) => a * b, 1);

      const matOrd = p === "fro" ? "fro" : p;
      if (
        matOrd !== "fro" &&
        matOrd !== 1 &&
        matOrd !== -1 &&
        matOrd !== 2 &&
        matOrd !== -2 &&
        matOrd !== Number.POSITIVE_INFINITY &&
        matOrd !== Number.NEGATIVE_INFINITY
      ) {
        throw new InvalidParameterError(
          `Invalid norm order '${String(matOrd)}' for matrix norm. ` +
            "Valid orders are: 1, -1, 2, -2, Infinity, -Infinity, 'fro'.",
          "ord",
          matOrd
        );
      }

      const results = new Float64Array(outerSize);

      for (let outer = 0; outer < outerSize; outer++) {
        let baseOffset = x.offset;
        let rem = outer;

        for (let d = outerShape.length - 1; d >= 0; d--) {
          const dim = outerShape[d] ?? 0;
          const idx = rem % dim;
          rem = Math.floor(rem / dim);
          baseOffset += idx * (outerStrides[d] ?? 0);
        }

        let val = 0;

        if (matOrd === "fro") {
          let sum = 0;
          for (let i = 0; i < dimRow; i++) {
            for (let j = 0; j < dimCol; j++) {
              const v = Number(x.data[baseOffset + i * strideRow + j * strideCol]);
              sum += v * v;
            }
          }
          val = Math.sqrt(sum);
        } else if (matOrd === 1) {
          let maxSum = 0;
          for (let j = 0; j < dimCol; j++) {
            let colSum = 0;
            for (let i = 0; i < dimRow; i++) {
              colSum += Math.abs(Number(x.data[baseOffset + i * strideRow + j * strideCol]));
            }
            maxSum = Math.max(maxSum, colSum);
          }
          val = maxSum;
        } else if (matOrd === -1) {
          let minSum = Infinity;
          for (let j = 0; j < dimCol; j++) {
            let colSum = 0;
            for (let i = 0; i < dimRow; i++) {
              colSum += Math.abs(Number(x.data[baseOffset + i * strideRow + j * strideCol]));
            }
            minSum = Math.min(minSum, colSum);
          }
          val = minSum === Infinity ? 0 : minSum;
        } else if (matOrd === Number.POSITIVE_INFINITY) {
          let maxSum = 0;
          for (let i = 0; i < dimRow; i++) {
            let rowSum = 0;
            for (let j = 0; j < dimCol; j++) {
              rowSum += Math.abs(Number(x.data[baseOffset + i * strideRow + j * strideCol]));
            }
            maxSum = Math.max(maxSum, rowSum);
          }
          val = maxSum;
        } else if (matOrd === Number.NEGATIVE_INFINITY) {
          let minSum = Infinity;
          for (let i = 0; i < dimRow; i++) {
            let rowSum = 0;
            for (let j = 0; j < dimCol; j++) {
              rowSum += Math.abs(Number(x.data[baseOffset + i * strideRow + j * strideCol]));
            }
            minSum = Math.min(minSum, rowSum);
          }
          val = minSum === Infinity ? 0 : minSum;
        } else if (matOrd === 2 || matOrd === -2) {
          const sliceData = new Float64Array(dimRow * dimCol);
          for (let i = 0; i < dimRow; i++) {
            for (let j = 0; j < dimCol; j++) {
              sliceData[i * dimCol + j] = Number(
                x.data[baseOffset + i * strideRow + j * strideCol]
              );
            }
          }
          const sliceTensor = tensor(sliceData).view([dimRow, dimCol]);
          const [_U, s, _Vt] = svd(sliceTensor);
          const sDense = toDenseVector1D(s);

          if (sDense.length === 0) {
            val = 0;
          } else {
            val = matOrd === 2 ? at(sDense, 0) : Math.abs(at(sDense, sDense.length - 1));
          }
        }

        results[outer] = val;
      }

      if (keepdims) {
        const kdShape: number[] = [];
        for (let i = 0; i < x.ndim; i++) {
          kdShape.push(i === ax0 || i === ax1 ? 1 : getDim(x, i, "norm()"));
        }
        return tensor(results).view(kdShape);
      }

      if (outerSize === 1 && outerShape.length === 0) return at(results, 0);
      return tensor(results).view(outerShape);
    }

    throw new ShapeError("axis has invalid length for input");
  }

  // Scalar norms when axis is omitted
  if (x.ndim === 0) {
    if (p === "nuc") {
      throw new InvalidParameterError("Invalid norm order 'nuc' for scalar norm.", "ord", p);
    }
    const ordInput = p === "fro" ? 2 : p;
    if (typeof ordInput !== "number") {
      throw new InvalidParameterError(
        `Invalid norm order '${String(ordInput)}' for scalar norm.`,
        "ord",
        ordInput
      );
    }
    const ordValue = normalizeVectorOrder(ordInput);
    return vectorNormValue(collectValues(), ordValue);
  }

  if (x.ndim === 1) {
    if (p === "nuc") {
      throw new InvalidParameterError("Invalid norm order 'nuc' for vector norm.", "ord", p);
    }
    const ordInput = p === "fro" ? 2 : p;
    if (typeof ordInput !== "number") {
      throw new InvalidParameterError(
        `Invalid norm order '${String(ordInput)}' for vector norm.`,
        "ord",
        ordInput
      );
    }
    const ordValue = normalizeVectorOrder(ordInput);
    return vectorNormValue(collectValues(), ordValue);
  }

  if (x.ndim !== 2) {
    throw new ShapeError("norm requires 1D or 2D input when axis is omitted");
  }

  const s0 = getStride(x, 0, "norm()");
  const s1 = getStride(x, 1, "norm()");
  const rows = getDim(x, 0, "norm()");
  const cols = getDim(x, 1, "norm()");

  if (p === "fro") {
    let sum = 0;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const val = Number(x.data[x.offset + i * s0 + j * s1]);
        sum += val * val;
      }
    }
    return Math.sqrt(sum);
  }

  if (p === "nuc") {
    const [_U, s, _Vt] = svd(x);
    const sDense = toDenseVector1D(s);
    let sum = 0;
    for (let i = 0; i < sDense.length; i++) sum += Math.abs(at(sDense, i));
    return sum;
  }

  if (p === 1) {
    let maxSum = 0;
    for (let j = 0; j < cols; j++) {
      let colSum = 0;
      for (let i = 0; i < rows; i++) {
        colSum += Math.abs(Number(x.data[x.offset + i * s0 + j * s1]));
      }
      maxSum = Math.max(maxSum, colSum);
    }
    return maxSum;
  }

  if (p === -1) {
    let minSum = Infinity;
    for (let j = 0; j < cols; j++) {
      let colSum = 0;
      for (let i = 0; i < rows; i++) {
        colSum += Math.abs(Number(x.data[x.offset + i * s0 + j * s1]));
      }
      minSum = Math.min(minSum, colSum);
    }
    return minSum === Infinity ? 0 : minSum;
  }

  if (p === Number.POSITIVE_INFINITY) {
    let maxSum = 0;
    for (let i = 0; i < rows; i++) {
      let rowSum = 0;
      for (let j = 0; j < cols; j++) {
        rowSum += Math.abs(Number(x.data[x.offset + i * s0 + j * s1]));
      }
      maxSum = Math.max(maxSum, rowSum);
    }
    return maxSum;
  }

  if (p === Number.NEGATIVE_INFINITY) {
    let minSum = Infinity;
    for (let i = 0; i < rows; i++) {
      let rowSum = 0;
      for (let j = 0; j < cols; j++) {
        rowSum += Math.abs(Number(x.data[x.offset + i * s0 + j * s1]));
      }
      minSum = Math.min(minSum, rowSum);
    }
    return minSum === Infinity ? 0 : minSum;
  }

  if (p === 2 || p === -2) {
    const [_U, s, _Vt] = svd(x);
    const sDense = toDenseVector1D(s);
    if (sDense.length === 0) return 0;
    const sMax = at(sDense, 0);
    const sMin = at(sDense, sDense.length - 1);
    return p === 2 ? sMax : Math.abs(sMin);
  }

  throw new InvalidParameterError(
    `Invalid norm order '${String(p)}' for matrix norm. ` +
      "Valid orders are: 1, -1, 2, -2, Infinity, -Infinity, 'fro', 'nuc'.",
    "ord",
    p
  );
}

/**
 * Condition number of a matrix.
 *
 * Measures how sensitive the solution of A*x=b is to changes in b.
 * Large condition number indicates ill-conditioned matrix.
 *
 * **Formula**: cond(A) = ||A|| * ||A^(-1)||
 *
 * **Parameters**:
 * @param a - Input matrix
 * @param p - Norm order (same as norm())
 *
 * **Returns**: Condition number (>= 1, infinity for singular matrices)
 *
 * @throws {ShapeError} If input is not a 2D matrix
 * @throws {DTypeError} If input has string dtype
 * @throws {DataValidationError} If input contains non-finite values (NaN, Infinity)
 * @throws {InvalidParameterError} If p is unsupported
 */
export function cond(a: Tensor, _p?: number | "fro"): number {
  if (a.ndim !== 2) throw new ShapeError("Input must be 2D matrix");
  assertFiniteTensor(a, "cond()");

  if (_p !== undefined) {
    if (_p === "fro") {
      // ok
    } else if (typeof _p === "number") {
      if (!Number.isFinite(_p) || Number.isNaN(_p) || _p !== 2) {
        throw new InvalidParameterError(
          "Only 2-norm and Frobenius norm are supported for condition number",
          "p",
          _p
        );
      }
    } else {
      throw new InvalidParameterError(
        "Only 2-norm and Frobenius norm are supported for condition number",
        "p",
        _p
      );
    }
  }

  const m = getDim(a, 0, "cond()");
  const n = getDim(a, 1, "cond()");
  const k = Math.min(m, n);
  if (k === 0) return Infinity;

  const [_U, s, _Vt] = svd(a);
  const sDense = toDenseVector1D(s);

  if (_p === "fro") {
    if (sDense.length === 0) return Infinity;
    let sumSq = 0;
    let sumInvSq = 0;
    for (let i = 0; i < sDense.length; i++) {
      const si = at(sDense, i);
      sumSq += si * si;
      if (si === 0) return Infinity;
      sumInvSq += 1 / (si * si);
    }
    return Math.sqrt(sumSq) * Math.sqrt(sumInvSq);
  }

  const sMax = at(sDense, 0);
  const sMin = at(sDense, k - 1);
  if (sMin === 0) return Infinity;
  return sMax / sMin;
}
