/**
 * Numerical utility operations.
 *
 * This module provides common numerical computing functions:
 * - interp: 1D linear interpolation (NumPy equivalent)
 * - trapz: Trapezoidal numerical integration
 * - gradient: Numerical gradient via finite differences
 * - digitize: Bin continuous values into discrete bins
 * - vstack / hstack / column_stack: Convenience stacking functions
 *
 * All operations maintain type safety and proper error handling.
 */

import { DTypeError, InvalidParameterError, ShapeError } from "../../core";
import { isContiguous } from "../tensor/strides";
import { Tensor } from "../tensor/Tensor";
import { concatenate } from "./manipulation";

// ─── Internal helpers ─────────────────────────────────────────────────────────

function requireNumericFlat(t: Tensor, fnName: string): Float64Array {
  if (t.dtype === "string") {
    throw new DTypeError(`${fnName} is not defined for string dtype`);
  }
  const n = t.size;
  const out = new Float64Array(n);
  const data = t.data;
  if (Array.isArray(data)) {
    throw new DTypeError(`${fnName} requires numeric data`);
  }
  const contig = isContiguous(t.shape, t.strides);
  if (contig && t.offset === 0) {
    if (data instanceof BigInt64Array) {
      for (let i = 0; i < n; i++) {
        out[i] = Number(data[i] ?? 0n);
      }
    } else {
      for (let i = 0; i < n; i++) {
        out[i] = Number(data[i] ?? 0);
      }
    }
  } else {
    for (let i = 0; i < n; i++) {
      let physIdx = t.offset;
      let rem = i;
      for (let d = t.shape.length - 1; d >= 0; d--) {
        const dim = t.shape[d] ?? 1;
        const coord = rem % dim;
        rem = (rem - coord) / dim;
        physIdx += coord * (t.strides[d] ?? 0);
      }
      const val = data[physIdx];
      out[i] = typeof val === "bigint" ? Number(val) : Number(val ?? 0);
    }
  }
  return out;
}

// ─── interp ───────────────────────────────────────────────────────────────────

/**
 * One-dimensional linear interpolation.
 *
 * Returns the one-dimensional piecewise linear interpolant to a function
 * with given discrete data points (xp, fp), evaluated at x.
 *
 * Equivalent to `numpy.interp(x, xp, fp)`.
 *
 * **Complexity**: O(n * log(m)) where n = len(x), m = len(xp)
 *
 * @param x - The x-coordinates at which to evaluate the interpolated values
 * @param xp - The x-coordinates of the data points, must be increasing
 * @param fp - The y-coordinates of the data points, same length as xp
 * @param left - Value to return for x < xp[0] (default: fp[0])
 * @param right - Value to return for x > xp[-1] (default: fp[-1])
 * @returns Tensor of interpolated values, same shape as x
 *
 * @example
 * ```ts
 * const xp = tensor([0, 1, 2, 3]);
 * const fp = tensor([0, 1, 4, 9]);
 * const x = tensor([0.5, 1.5, 2.5]);
 * const y = interp(x, xp, fp);  // tensor([0.5, 2.5, 6.5])
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function interp(x: Tensor, xp: Tensor, fp: Tensor, left?: number, right?: number): Tensor {
  if (xp.ndim !== 1) {
    throw new ShapeError("interp: xp must be a 1D tensor");
  }
  if (fp.ndim !== 1) {
    throw new ShapeError("interp: fp must be a 1D tensor");
  }
  const m = xp.shape[0] ?? 0;
  if (m === 0) {
    throw new InvalidParameterError("interp: xp must not be empty", "xp", m);
  }
  if ((fp.shape[0] ?? 0) !== m) {
    throw new ShapeError("interp: xp and fp must have the same length");
  }

  const xpData = requireNumericFlat(xp, "interp");
  const fpData = requireNumericFlat(fp, "interp");
  const xData = requireNumericFlat(x, "interp");

  // Validate xp is sorted
  for (let i = 1; i < m; i++) {
    if ((xpData[i] ?? 0) < (xpData[i - 1] ?? 0)) {
      throw new InvalidParameterError(
        "interp: xp must be monotonically increasing",
        "xp",
        xpData[i]
      );
    }
  }

  const leftVal = left ?? fpData[0] ?? 0;
  const rightVal = right ?? fpData[m - 1] ?? 0;

  const n = xData.length;
  const out = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const xi = xData[i] ?? 0;

    if (xi <= (xpData[0] ?? 0)) {
      out[i] = leftVal;
      continue;
    }
    if (xi >= (xpData[m - 1] ?? 0)) {
      out[i] = rightVal;
      continue;
    }

    // Binary search for the interval
    let lo = 0;
    let hi = m - 1;
    while (lo < hi - 1) {
      const mid = (lo + hi) >>> 1;
      if ((xpData[mid] ?? 0) <= xi) {
        lo = mid;
      } else {
        hi = mid;
      }
    }

    const x0 = xpData[lo] ?? 0;
    const x1 = xpData[hi] ?? 0;
    const f0 = fpData[lo] ?? 0;
    const f1 = fpData[hi] ?? 0;

    // Linear interpolation
    const t = x1 !== x0 ? (xi - x0) / (x1 - x0) : 0;
    out[i] = f0 + t * (f1 - f0);
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: [...x.shape],
    dtype: "float64",
    device: x.device,
  });
}

// ─── trapz ────────────────────────────────────────────────────────────────────

/**
 * Integrate along the given axis using the composite trapezoidal rule.
 *
 * Equivalent to `numpy.trapz(y, x, dx)`.
 *
 * @param y - Input array to integrate
 * @param x - Optional sample points corresponding to y values. If not provided, spacing is uniform with step `dx`.
 * @param dx - Spacing between sample points when x is not given (default: 1.0)
 * @returns Scalar result of the integration
 *
 * @example
 * ```ts
 * const y = tensor([1, 2, 3, 4]);
 * trapz(y);           // 7.5 (dx=1)
 * trapz(y, undefined, 0.5);  // 3.75 (dx=0.5)
 *
 * const x = tensor([0, 1, 3, 5]);
 * const y2 = tensor([1, 2, 3, 4]);
 * trapz(y2, x);       // 12.0 (non-uniform spacing)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function trapz(y: Tensor, x?: Tensor, dx = 1.0): number {
  if (y.ndim !== 1) {
    throw new ShapeError("trapz: y must be a 1D tensor");
  }
  const n = y.shape[0] ?? 0;
  if (n < 2) {
    return 0;
  }

  const yData = requireNumericFlat(y, "trapz");

  if (x !== undefined) {
    if (x.ndim !== 1) {
      throw new ShapeError("trapz: x must be a 1D tensor");
    }
    if ((x.shape[0] ?? 0) !== n) {
      throw new ShapeError("trapz: x and y must have the same length");
    }
    const xData = requireNumericFlat(x, "trapz");
    let result = 0;
    for (let i = 1; i < n; i++) {
      const dx_i = (xData[i] ?? 0) - (xData[i - 1] ?? 0);
      result += (dx_i * ((yData[i] ?? 0) + (yData[i - 1] ?? 0))) / 2;
    }
    return result;
  }

  // Uniform spacing
  let result = 0;
  for (let i = 1; i < n; i++) {
    result += (yData[i] ?? 0) + (yData[i - 1] ?? 0);
  }
  return (result * dx) / 2;
}

// ─── gradient ─────────────────────────────────────────────────────────────────

/**
 * Return the numerical gradient of a 1D array using central finite differences.
 *
 * Interior points use second-order central differences.
 * Boundary points use first-order one-sided differences.
 *
 * Equivalent to `numpy.gradient(f, *varargs)` for 1D arrays.
 *
 * @param f - Input 1D tensor
 * @param spacing - Scalar spacing between samples (default: 1.0), or a 1D tensor of sample positions
 * @returns Tensor of the same shape as f containing the numerical gradient
 *
 * @example
 * ```ts
 * const f = tensor([1, 2, 4, 7, 11]);
 * gradient(f);         // tensor([1, 1.5, 2.5, 3.5, 4])
 * gradient(f, 0.5);    // tensor([2, 3, 5, 7, 8])
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function gradient(f: Tensor, spacing?: number | Tensor): Tensor {
  if (f.ndim !== 1) {
    throw new ShapeError("gradient: f must be a 1D tensor");
  }
  const n = f.shape[0] ?? 0;
  if (n < 2) {
    throw new InvalidParameterError("gradient: f must have at least 2 elements", "f", n);
  }

  const fData = requireNumericFlat(f, "gradient");
  const out = new Float64Array(n);

  if (spacing === undefined || typeof spacing === "number") {
    const h = spacing ?? 1.0;
    if (h <= 0) {
      throw new InvalidParameterError("gradient: spacing must be positive", "spacing", h);
    }

    // Forward difference at left boundary
    out[0] = ((fData[1] ?? 0) - (fData[0] ?? 0)) / h;
    // Central differences for interior
    for (let i = 1; i < n - 1; i++) {
      out[i] = ((fData[i + 1] ?? 0) - (fData[i - 1] ?? 0)) / (2 * h);
    }
    // Backward difference at right boundary
    out[n - 1] = ((fData[n - 1] ?? 0) - (fData[n - 2] ?? 0)) / h;
  } else {
    // Non-uniform spacing
    if (spacing.ndim !== 1) {
      throw new ShapeError("gradient: spacing tensor must be 1D");
    }
    if ((spacing.shape[0] ?? 0) !== n) {
      throw new ShapeError("gradient: spacing and f must have the same length");
    }
    const xData = requireNumericFlat(spacing, "gradient");

    // Forward difference at left boundary
    const h0 = (xData[1] ?? 0) - (xData[0] ?? 0);
    out[0] = h0 !== 0 ? ((fData[1] ?? 0) - (fData[0] ?? 0)) / h0 : 0;

    // Central differences for interior
    for (let i = 1; i < n - 1; i++) {
      const hPrev = (xData[i] ?? 0) - (xData[i - 1] ?? 0);
      const hNext = (xData[i + 1] ?? 0) - (xData[i] ?? 0);
      const hTotal = hPrev + hNext;
      if (hTotal === 0) {
        out[i] = 0;
      } else {
        // Weighted central difference for non-uniform grid
        out[i] =
          ((hPrev * ((fData[i + 1] ?? 0) - (fData[i] ?? 0))) / hNext +
            (hNext * ((fData[i] ?? 0) - (fData[i - 1] ?? 0))) / hPrev) /
          hTotal;
      }
    }

    // Backward difference at right boundary
    const hEnd = (xData[n - 1] ?? 0) - (xData[n - 2] ?? 0);
    out[n - 1] = hEnd !== 0 ? ((fData[n - 1] ?? 0) - (fData[n - 2] ?? 0)) / hEnd : 0;
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: [n],
    dtype: "float64",
    device: f.device,
  });
}

// ─── digitize ─────────────────────────────────────────────────────────────────

/**
 * Return the indices of the bins to which each value in the input array belongs.
 *
 * Equivalent to `numpy.digitize(x, bins, right)`.
 *
 * If `right` is false (default), then the bin index `i` satisfies:
 *   `bins[i-1] <= x < bins[i]`
 *
 * If `right` is true:
 *   `bins[i-1] < x <= bins[i]`
 *
 * @param x - Input tensor of values to be binned
 * @param bins - 1D monotonically increasing array of bin edges
 * @param right - If true, intervals are closed on the right (default: false)
 * @returns Tensor of bin indices (same shape as x)
 *
 * @example
 * ```ts
 * const x = tensor([0.5, 1.5, 2.5, 3.5]);
 * const bins = tensor([1, 2, 3]);
 * digitize(x, bins);  // tensor([0, 1, 2, 3])
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function digitize(x: Tensor, bins: Tensor, right = false): Tensor {
  if (bins.ndim !== 1) {
    throw new ShapeError("digitize: bins must be a 1D tensor");
  }
  const m = bins.shape[0] ?? 0;
  if (m === 0) {
    throw new InvalidParameterError("digitize: bins must not be empty", "bins", m);
  }

  const binsData = requireNumericFlat(bins, "digitize");
  const xData = requireNumericFlat(x, "digitize");

  // Validate bins is sorted
  for (let i = 1; i < m; i++) {
    if ((binsData[i] ?? 0) < (binsData[i - 1] ?? 0)) {
      throw new InvalidParameterError(
        "digitize: bins must be monotonically increasing",
        "bins",
        binsData[i]
      );
    }
  }

  const n = xData.length;
  const out = new Float64Array(n);

  for (let i = 0; i < n; i++) {
    const val = xData[i] ?? 0;
    // Binary search
    let lo = 0;
    let hi = m;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      const binVal = binsData[mid] ?? 0;
      if (right ? binVal < val : binVal <= val) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    out[i] = lo;
  }

  return Tensor.fromTypedArray({
    data: out,
    shape: [...x.shape],
    dtype: "float64",
    device: x.device,
  });
}

// ─── vstack / hstack / column_stack ───────────────────────────────────────────

/**
 * Stack tensors vertically (row-wise).
 *
 * Equivalent to `numpy.vstack()`. For 1D inputs, creates rows and stacks.
 * For 2D+ inputs, concatenates along axis 0.
 *
 * @param tensors - Sequence of tensors to stack
 * @returns Vertically stacked tensor
 *
 * @example
 * ```ts
 * const a = tensor([1, 2, 3]);
 * const b = tensor([4, 5, 6]);
 * vstack([a, b]);  // tensor([[1, 2, 3], [4, 5, 6]])
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function vstack(tensors: Tensor[]): Tensor {
  if (tensors.length === 0) {
    throw new InvalidParameterError("vstack requires at least one tensor", "tensors");
  }

  // For 1D inputs, reshape to (1, N) then concatenate along axis 0
  const first = tensors[0]!;
  if (first.ndim === 1) {
    const reshaped = tensors.map((t) => {
      if (t.ndim !== 1) {
        throw new ShapeError("vstack: all tensors must have the same number of dimensions");
      }
      const n = t.shape[0] ?? 0;
      const data = requireNumericFlat(t, "vstack");
      return Tensor.fromTypedArray({
        data: new Float64Array(data),
        shape: [1, n],
        dtype: "float64",
        device: t.device,
      });
    });
    return concatenate(reshaped, 0);
  }

  return concatenate(tensors, 0);
}

/**
 * Stack tensors horizontally (column-wise).
 *
 * Equivalent to `numpy.hstack()`. For 1D inputs, concatenates along axis 0.
 * For 2D+ inputs, concatenates along axis 1.
 *
 * @param tensors - Sequence of tensors to stack
 * @returns Horizontally stacked tensor
 *
 * @example
 * ```ts
 * const a = tensor([1, 2]);
 * const b = tensor([3, 4]);
 * hstack([a, b]);  // tensor([1, 2, 3, 4])
 *
 * const c = tensor([[1], [2]]);
 * const d = tensor([[3], [4]]);
 * hstack([c, d]);  // tensor([[1, 3], [2, 4]])
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function hstack(tensors: Tensor[]): Tensor {
  if (tensors.length === 0) {
    throw new InvalidParameterError("hstack requires at least one tensor", "tensors");
  }

  const first = tensors[0]!;
  if (first.ndim === 1) {
    return concatenate(tensors, 0);
  }

  return concatenate(tensors, 1);
}

/**
 * Stack 1D arrays as columns into a 2D array.
 *
 * Equivalent to `numpy.column_stack()`. Takes a sequence of 1D tensors
 * and stacks them as columns of a 2D array.
 *
 * @param tensors - Sequence of 1D tensors to stack as columns
 * @returns 2D tensor where each input is a column
 *
 * @example
 * ```ts
 * const a = tensor([1, 2, 3]);
 * const b = tensor([4, 5, 6]);
 * column_stack([a, b]);  // tensor([[1, 4], [2, 5], [3, 6]])
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 * @deprecated Prefer {@link columnStack}.
 */
export function column_stack(tensors: Tensor[]): Tensor {
  if (tensors.length === 0) {
    throw new InvalidParameterError("column_stack requires at least one tensor", "tensors");
  }

  const reshaped = tensors.map((t) => {
    if (t.ndim === 1) {
      const n = t.shape[0] ?? 0;
      const data = requireNumericFlat(t, "column_stack");
      return Tensor.fromTypedArray({
        data: new Float64Array(data),
        shape: [n, 1],
        dtype: "float64",
        device: t.device,
      });
    }
    if (t.ndim === 2) {
      return t;
    }
    throw new ShapeError("column_stack: all inputs must be 1D or 2D");
  });

  return concatenate(reshaped, 1);
}

/**
 * Canonical camelCase alias of {@link column_stack}. Prefer this spelling;
 * the snake_case original remains exported for backward compatibility.
 */
export const columnStack = column_stack;
