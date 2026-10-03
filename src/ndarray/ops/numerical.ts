/**
 * Numerical utility operations.
 *
 * This module provides common numerical computing functions:
 * - interp: 1D linear interpolation (NumPy equivalent)
 * - trapz / trapezoid: Trapezoidal numerical integration
 * - gradient: Numerical gradient via finite differences
 * - digitize: Bin continuous values into discrete bins
 * - vstack / hstack / columnStack: Convenience stacking functions
 *
 * Inputs may have any numeric dtype and may be strided views. `interp` and
 * `gradient` return the promoted float dtype of their inputs (`float32` for
 * integer input), `digitize` returns `int32` indices, and the stacking
 * functions keep the input dtype (mixed dtypes are promoted like binary ops).
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */

import { DTypeError, InvalidParameterError, ShapeError } from "../../core";
import { promoteTypes, toFloatDType } from "../../core/utils/dtype_utils";
import { Tensor } from "../tensor/Tensor";
import { allocFloat, floatResult, readNumbers } from "./_internal";
import { concatenate } from "./manipulation";

// ─── Internal helpers ─────────────────────────────────────────────────────────

/** Logical elements of `t` as numbers in row-major order (read-only, may alias the buffer). */
function requireNumericFlat(t: Tensor, fnName: string): ArrayLike<number> {
  return readNumbers(t, fnName, false);
}

/** Neumaier-compensated running sum; keeps the error independent of the number of terms. */
class CompensatedSum {
  private sum = 0;
  private comp = 0;

  add(v: number): void {
    const t = this.sum + v;
    // Only compensate while the running sum is finite, otherwise inf - inf would give NaN.
    if (Number.isFinite(t)) {
      if (Math.abs(this.sum) >= Math.abs(v)) this.comp += this.sum - t + v;
      else this.comp += v - t + this.sum;
    }
    this.sum = t;
  }

  get value(): number {
    return this.sum + this.comp;
  }
}

// ─── interp ───────────────────────────────────────────────────────────────────

/**
 * One-dimensional linear interpolation.
 *
 * Returns the one-dimensional piecewise linear interpolant to a function
 * with given discrete data points (xp, fp), evaluated at x.
 *
 * Equivalent to `numpy.interp(x, xp, fp, left, right)`. A value equal to
 * `xp[0]` or `xp[-1]` returns `fp[0]` or `fp[-1]`; `left` and `right` apply
 * only strictly outside `[xp[0], xp[-1]]`. A NaN in `x` gives NaN.
 *
 * **Complexity**: O(n * log(m)) where n = len(x), m = len(xp)
 *
 * @param x - The x-coordinates at which to evaluate the interpolated values
 * @param xp - The x-coordinates of the data points, must be increasing (repeated values allowed)
 * @param fp - The y-coordinates of the data points, same length as xp
 * @param left - Value to return for x < xp[0] (default: fp[0])
 * @param right - Value to return for x > xp[-1] (default: fp[-1])
 * @returns Tensor of interpolated values, same shape as x. Its dtype is the promoted float
 *   dtype of `x`, `xp` and `fp` (`float32` when all are integer or bool)
 * @throws {ShapeError} If xp or fp is not 1D, or their lengths differ
 * @throws {InvalidParameterError} If xp is empty, contains NaN, or is not increasing
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

  for (let i = 0; i < m; i++) {
    const v = xpData[i] as number;
    if (Number.isNaN(v)) {
      throw new InvalidParameterError("interp: xp must not contain NaN", "xp", v);
    }
    if (i > 0 && v < (xpData[i - 1] as number)) {
      throw new InvalidParameterError("interp: xp must be monotonically increasing", "xp", v);
    }
  }

  const xFirst = xpData[0] as number;
  const xLast = xpData[m - 1] as number;
  const fFirst = fpData[0] as number;
  const fLast = fpData[m - 1] as number;
  const leftVal = left ?? fFirst;
  const rightVal = right ?? fLast;

  const n = x.size;
  const outDtype = toFloatDType(promoteTypes(promoteTypes(x.dtype, xp.dtype), fp.dtype));
  const out = allocFloat(outDtype, n);

  for (let i = 0; i < n; i++) {
    const xi = xData[i] as number;

    if (Number.isNaN(xi)) {
      out[i] = Number.NaN;
      continue;
    }
    if (xi < xFirst) {
      out[i] = leftVal;
      continue;
    }
    if (xi > xLast) {
      out[i] = rightVal;
      continue;
    }
    if (xi === xLast) {
      out[i] = fLast;
      continue;
    }

    // Last index lo with xp[lo] <= xi; invariant xp[hi] > xi.
    let lo = 0;
    let hi = m - 1;
    while (lo < hi - 1) {
      const mid = (lo + hi) >>> 1;
      if ((xpData[mid] as number) <= xi) {
        lo = mid;
      } else {
        hi = mid;
      }
    }

    const x0 = xpData[lo] as number;
    const x1 = xpData[hi] as number;
    const f0 = fpData[lo] as number;
    const f1 = fpData[hi] as number;

    if (xi === x0) {
      out[i] = f0;
      continue;
    }

    // Same evaluation order as NumPy, including the fallbacks for infinite fp values.
    const slope = (f1 - f0) / (x1 - x0);
    let v = slope * (xi - x0) + f0;
    if (Number.isNaN(v)) {
      v = slope * (xi - x1) + f1;
      if (Number.isNaN(v) && f0 === f1) v = f0;
    }
    out[i] = v;
  }

  return floatResult(out, [...x.shape], outDtype, x.device);
}

// ─── trapz ────────────────────────────────────────────────────────────────────

/**
 * Integrate a 1D signal using the composite trapezoidal rule.
 *
 * Equivalent to `numpy.trapezoid(y, x, dx)` (formerly `numpy.trapz`) for 1D
 * input. The terms are added with compensated summation, so long inputs do not
 * accumulate rounding error.
 *
 * @param y - 1D tensor of function values
 * @param x - Optional 1D sample points corresponding to y values. If not provided, spacing is uniform with step `dx`.
 * @param dx - Spacing between sample points when x is not given (default: 1.0)
 * @returns Scalar result of the integration (0 when y has fewer than 2 samples)
 * @throws {ShapeError} If y or x is not 1D, or their lengths differ
 *
 * @example
 * ```ts
 * const y = tensor([1, 2, 3, 4]);
 * trapz(y);           // 7.5 (dx=1)
 * trapz(y, undefined, 0.5);  // 3.75 (dx=0.5)
 *
 * const x = tensor([0, 1, 3, 5]);
 * const y2 = tensor([1, 2, 3, 4]);
 * trapz(y2, x);       // 13.5 (non-uniform spacing)
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function trapz(y: Tensor, x?: Tensor, dx = 1.0): number {
  if (y.ndim !== 1) {
    throw new ShapeError("trapz: y must be a 1D tensor");
  }
  const n = y.shape[0] ?? 0;

  if (x !== undefined) {
    if (x.ndim !== 1) {
      throw new ShapeError("trapz: x must be a 1D tensor");
    }
    if ((x.shape[0] ?? 0) !== n) {
      throw new ShapeError("trapz: x and y must have the same length");
    }
  }
  if (n < 2) {
    return 0;
  }

  const yData = requireNumericFlat(y, "trapz");
  const acc = new CompensatedSum();

  if (x !== undefined) {
    const xData = requireNumericFlat(x, "trapz");
    for (let i = 1; i < n; i++) {
      const h = (xData[i] as number) - (xData[i - 1] as number);
      acc.add((h * ((yData[i] as number) + (yData[i - 1] as number))) / 2);
    }
    return acc.value;
  }

  for (let i = 1; i < n; i++) {
    acc.add((yData[i] as number) + (yData[i - 1] as number));
  }
  return (acc.value * dx) / 2;
}

/**
 * NumPy 2 name of {@link trapz}; both refer to the same function.
 */
export const trapezoid = trapz;

// ─── gradient ─────────────────────────────────────────────────────────────────

/**
 * Return the numerical gradient of a 1D array using central finite differences.
 *
 * Interior points use second-order central differences.
 * Boundary points use first-order one-sided differences.
 *
 * Equivalent to `numpy.gradient(f, *varargs)` for 1D arrays with the default
 * `edge_order=1`.
 *
 * @param f - Input 1D tensor
 * @param spacing - Scalar spacing between samples (default: 1.0, must be positive and finite),
 *   or a 1D tensor with the coordinate of every sample (consecutive coordinates must differ)
 * @returns Tensor of the same shape as f containing the numerical gradient. Its dtype is the
 *   float dtype of `f` (promoted with the spacing tensor when one is given; `float32` for
 *   integer or bool input)
 * @throws {ShapeError} If f or the spacing tensor is not 1D, or their lengths differ
 * @throws {InvalidParameterError} If f has fewer than 2 elements, the scalar spacing is not a
 *   positive finite number, or the coordinates contain repeated or non-finite steps
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
  const outDtype = toFloatDType(
    spacing === undefined || typeof spacing === "number"
      ? f.dtype
      : promoteTypes(f.dtype, spacing.dtype)
  );
  const out = allocFloat(outDtype, n);

  if (spacing === undefined || typeof spacing === "number") {
    const h = spacing ?? 1.0;
    if (!Number.isFinite(h) || h <= 0) {
      throw new InvalidParameterError(
        "gradient: spacing must be a positive finite number",
        "spacing",
        h
      );
    }

    out[0] = ((fData[1] as number) - (fData[0] as number)) / h;
    for (let i = 1; i < n - 1; i++) {
      out[i] = ((fData[i + 1] as number) - (fData[i - 1] as number)) / (2 * h);
    }
    out[n - 1] = ((fData[n - 1] as number) - (fData[n - 2] as number)) / h;
  } else {
    if (spacing.ndim !== 1) {
      throw new ShapeError("gradient: spacing tensor must be 1D");
    }
    if ((spacing.shape[0] ?? 0) !== n) {
      throw new ShapeError("gradient: spacing and f must have the same length");
    }
    const xData = requireNumericFlat(spacing, "gradient");

    for (let i = 1; i < n; i++) {
      const step = (xData[i] as number) - (xData[i - 1] as number);
      if (step === 0 || !Number.isFinite(step)) {
        throw new InvalidParameterError(
          "gradient: spacing coordinates must have finite, non-zero steps",
          "spacing",
          xData[i]
        );
      }
    }

    out[0] =
      ((fData[1] as number) - (fData[0] as number)) / ((xData[1] as number) - (xData[0] as number));

    // Second-order formula for a non-uniform grid (same weights as numpy.gradient).
    for (let i = 1; i < n - 1; i++) {
      const hPrev = (xData[i] as number) - (xData[i - 1] as number);
      const hNext = (xData[i + 1] as number) - (xData[i] as number);
      const hTotal = hPrev + hNext;
      if (hTotal === 0) {
        throw new InvalidParameterError(
          "gradient: spacing coordinates at i-1 and i+1 must differ",
          "spacing",
          xData[i + 1]
        );
      }
      out[i] =
        ((hPrev * ((fData[i + 1] as number) - (fData[i] as number))) / hNext +
          (hNext * ((fData[i] as number) - (fData[i - 1] as number))) / hPrev) /
        hTotal;
    }

    out[n - 1] =
      ((fData[n - 1] as number) - (fData[n - 2] as number)) /
      ((xData[n - 1] as number) - (xData[n - 2] as number));
  }

  return floatResult(out, [n], outDtype, f.device);
}

// ─── digitize ─────────────────────────────────────────────────────────────────

/**
 * Return the indices of the bins to which each value in the input array belongs.
 *
 * Equivalent to `numpy.digitize(x, bins, right)`.
 *
 * For increasing `bins`, with `right` false (default), the bin index `i` satisfies
 * `bins[i-1] <= x < bins[i]`; with `right` true it satisfies `bins[i-1] < x <= bins[i]`.
 * For decreasing `bins` the inequalities are reversed: `bins[i-1] > x >= bins[i]`
 * and `bins[i-1] >= x > bins[i]`. Values below the first bin get index 0 and
 * values past the last get `bins.length` (for increasing bins). NaN is placed
 * after the last increasing bin, and at index 0 for decreasing bins. An empty
 * `bins` maps everything to 0.
 *
 * The indices are returned as an `int32` tensor.
 *
 * @param x - Input tensor of values to be binned
 * @param bins - 1D monotonically increasing or decreasing array of bin edges (no NaN)
 * @param right - If true, intervals are closed on the right (default: false)
 * @returns `int32` tensor of bin indices (same shape as x)
 * @throws {ShapeError} If bins is not 1D
 * @throws {InvalidParameterError} If bins is not monotonic or contains NaN
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

  const binsData = requireNumericFlat(bins, "digitize");
  const xData = requireNumericFlat(x, "digitize");

  let increasing = true;
  let decreasing = true;
  for (let i = 0; i < m; i++) {
    const v = binsData[i] as number;
    if (Number.isNaN(v)) {
      throw new InvalidParameterError("digitize: bins must not contain NaN", "bins", v);
    }
    if (i > 0) {
      const prev = binsData[i - 1] as number;
      if (v < prev) increasing = false;
      if (v > prev) decreasing = false;
    }
  }
  if (!increasing && !decreasing) {
    throw new InvalidParameterError(
      "digitize: bins must be monotonically increasing or decreasing",
      "bins",
      Array.from(binsData)
    );
  }

  const n = x.size;
  const out = new Int32Array(n);

  for (let i = 0; i < n; i++) {
    const val = xData[i] as number;
    if (increasing && Number.isNaN(val)) {
      out[i] = m;
      continue;
    }
    // Number of leading bins that lie before `val`: binary search on a prefix-true predicate.
    let lo = 0;
    let hi = m;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      const b = binsData[mid] as number;
      const before = increasing ? (right ? b < val : b <= val) : right ? b >= val : b > val;
      if (before) {
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
    dtype: "int32",
    device: x.device,
  });
}

// ─── vstack / hstack / column_stack ───────────────────────────────────────────

/**
 * Common dtype of numeric tensors (PyTorch-style promotion, see `promoteTypes`): the wider
 * type of one category wins, an integer with a float gives the float, and `float16` with
 * `bfloat16` gives `float32`.
 */
function promoteDTypes(dtypes: readonly Tensor["dtype"][], fnName: string): Tensor["dtype"] {
  const first = dtypes[0] as Tensor["dtype"];
  if (dtypes.every((d) => d === first)) return first;
  if (dtypes.some((d) => d === "string")) {
    throw new DTypeError(`${fnName}: cannot combine string tensors with other dtypes`);
  }
  return dtypes.reduce((a, b) => promoteTypes(a, b));
}

/** Cast every tensor to their common dtype (no copy when the dtypes already agree). */
function unifyDTypes(tensors: Tensor[], fnName: string): Tensor[] {
  const target = promoteDTypes(
    tensors.map((t) => t.dtype),
    fnName
  );
  return tensors.map((t) => (t.dtype === target ? t : t.astype(target)));
}

/**
 * Stack tensors vertically (row-wise).
 *
 * Equivalent to `numpy.vstack()`. Tensors with fewer than two dimensions are
 * promoted first (a 1D tensor of length N becomes a `[1, N]` row, a scalar
 * becomes `[1, 1]`), then everything is concatenated along axis 0. The input
 * dtype is kept; tensors of different numeric dtypes are promoted to a common
 * one.
 *
 * @param tensors - Sequence of tensors to stack
 * @returns Vertically stacked tensor
 * @throws {InvalidParameterError} If `tensors` is empty
 * @throws {ShapeError} If the trailing dimensions do not match
 * @throws {DTypeError} If string and numeric tensors are mixed
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

  const rows = tensors.map((t) => {
    if (t.ndim === 0) return t.reshape([1, 1]);
    if (t.ndim === 1) return t.reshape([1, t.shape[0] ?? 0]);
    return t;
  });
  return concatenate(unifyDTypes(rows, "vstack"), 0);
}

/**
 * Stack tensors horizontally (column-wise).
 *
 * Equivalent to `numpy.hstack()`. Scalars are promoted to length-1 vectors.
 * 1D tensors are concatenated along axis 0, everything else along axis 1. The
 * input dtype is kept; tensors of different numeric dtypes are promoted to a
 * common one.
 *
 * @param tensors - Sequence of tensors to stack
 * @returns Horizontally stacked tensor
 * @throws {InvalidParameterError} If `tensors` is empty
 * @throws {ShapeError} If the other dimensions do not match
 * @throws {DTypeError} If string and numeric tensors are mixed
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

  const parts = tensors.map((t) => (t.ndim === 0 ? t.reshape([1]) : t));
  const first = parts[0] as Tensor;
  return concatenate(unifyDTypes(parts, "hstack"), first.ndim === 1 ? 0 : 1);
}

/**
 * Stack 1D arrays as columns into a 2D array.
 *
 * Equivalent to `numpy.column_stack()` for 0D, 1D and 2D inputs. Each 1D tensor
 * of length N becomes an `[N, 1]` column (a scalar becomes `[1, 1]`), 2D tensors
 * are kept as they are, and the result is concatenated along axis 1. The input
 * dtype is kept; tensors of different numeric dtypes are promoted to a common
 * one.
 *
 * @param tensors - Sequence of 0D, 1D or 2D tensors to stack as columns
 * @returns 2D tensor where each input is a column
 * @throws {InvalidParameterError} If `tensors` is empty
 * @throws {ShapeError} If an input has more than 2 dimensions or the row counts differ
 * @throws {DTypeError} If string and numeric tensors are mixed
 *
 * @example
 * ```ts
 * const a = tensor([1, 2, 3]);
 * const b = tensor([4, 5, 6]);
 * columnStack([a, b]);  // tensor([[1, 4], [2, 5], [3, 6]])
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ndarray-ops | Deepbox Tensor Operations}
 */
export function columnStack(tensors: Tensor[]): Tensor {
  if (tensors.length === 0) {
    throw new InvalidParameterError("columnStack requires at least one tensor", "tensors");
  }

  const columns = tensors.map((t) => {
    if (t.ndim === 0) return t.reshape([1, 1]);
    if (t.ndim === 1) return t.reshape([t.shape[0] ?? 0, 1]);
    if (t.ndim === 2) return t;
    throw new ShapeError("columnStack: all inputs must be 0D, 1D or 2D");
  });
  return concatenate(unifyDTypes(columns, "columnStack"), 1);
}

/**
 * Snake_case alias of {@link columnStack}, kept for backward compatibility.
 *
 * @deprecated Prefer {@link columnStack}.
 */
export const column_stack = columnStack;
