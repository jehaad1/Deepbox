/**
 * Isotonic regression: a monotone piecewise linear fit computed with the pool adjacent
 * violators algorithm.
 *
 * @module ml/linear/IsotonicRegression
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { assertContiguous, toFloat64View } from "../_validation";
import type { Regressor } from "../base";
import { r2ScoreOf } from "./LinearRegression";

/** How {@link IsotonicRegression.predict} treats `X` outside the training range. */
export type IsotonicOutOfBounds = "clip" | "nan" | "raise";

/** Constructor options of {@link IsotonicRegression}. */
export type IsotonicRegressionOptions = {
  /**
   * `true` for a non-decreasing fit, `false` for a non-increasing one, `"auto"` to pick the
   * direction from the sign of the Spearman correlation between `X` and `y` (default: true).
   */
  readonly increasing?: boolean | "auto";
  /**
   * Handling of prediction inputs outside the training range: `"clip"` uses the nearest fitted
   * value, `"nan"` returns NaN and `"raise"` throws (default: "clip").
   */
  readonly outOfBounds?: IsotonicOutOfBounds;
  /** Lower bound applied to the fitted values. */
  readonly yMin?: number;
  /** Upper bound applied to the fitted values. */
  readonly yMax?: number;
};

type Options = {
  increasing: boolean | "auto";
  outOfBounds: IsotonicOutOfBounds;
  yMin: number | undefined;
  yMax: number | undefined;
};

function resolveOptions(options: IsotonicRegressionOptions): Options {
  const resolved: Options = {
    increasing: options.increasing ?? true,
    outOfBounds: options.outOfBounds ?? "clip",
    yMin: options.yMin,
    yMax: options.yMax,
  };
  if (
    resolved.increasing !== true &&
    resolved.increasing !== false &&
    resolved.increasing !== "auto"
  ) {
    throw new InvalidParameterError(
      `increasing must be true, false or "auto"; received ${String(resolved.increasing)}`,
      "increasing",
      resolved.increasing
    );
  }
  if (
    resolved.outOfBounds !== "clip" &&
    resolved.outOfBounds !== "nan" &&
    resolved.outOfBounds !== "raise"
  ) {
    throw new InvalidParameterError(
      `outOfBounds must be "clip", "nan" or "raise"; received ${String(resolved.outOfBounds)}`,
      "outOfBounds",
      resolved.outOfBounds
    );
  }
  for (const key of ["yMin", "yMax"] as const) {
    const v = resolved[key];
    if (v !== undefined && Number.isNaN(v)) {
      throw new InvalidParameterError(`${key} must not be NaN`, key, v);
    }
  }
  if (resolved.yMin !== undefined && resolved.yMax !== undefined && resolved.yMin > resolved.yMax) {
    throw new InvalidParameterError(
      `yMin must be <= yMax; received yMin=${resolved.yMin}, yMax=${resolved.yMax}`,
      "yMin",
      resolved.yMin
    );
  }
  return resolved;
}

/** Average ranks (ties share the mean rank). */
function averageRanks(values: Float64Array): Float64Array {
  const n = values.length;
  const order = Array.from({ length: n }, (_, i) => i);
  order.sort((a, b) => (values[a] as number) - (values[b] as number));
  const ranks = new Float64Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && values[order[j + 1] as number] === values[order[i] as number]) j++;
    const rank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) ranks[order[k] as number] = rank;
    i = j + 1;
  }
  return ranks;
}

/** True when the Spearman rank correlation of x and y is >= 0 (NaN counts as negative). */
function spearmanNonNegative(x: Float64Array, y: Float64Array): boolean {
  const rx = averageRanks(x);
  const ry = averageRanks(y);
  const n = x.length;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += rx[i] as number;
    my += ry[i] as number;
  }
  mx /= n;
  my /= n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = (rx[i] as number) - mx;
    const dy = (ry[i] as number) - my;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  return sxy / Math.sqrt(sxx * syy) >= 0;
}

/**
 * Isotonic Regression.
 *
 * Fits a non-decreasing (or non-increasing) function to the data using the pool adjacent
 * violators algorithm (PAVA) and predicts by linear interpolation between the fitted
 * breakpoints, like `sklearn.isotonic.IsotonicRegression`. Samples with the same `X` are
 * merged into their weighted mean before pooling.
 *
 * Unlike most regressors, IsotonicRegression takes 1-D X and y (a single-column 2-D X is also
 * accepted).
 *
 * @example
 * ```ts
 * import { IsotonicRegression } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([1, 2, 3, 4, 5]);
 * const y = tensor([1, 3, 2, 4, 5]);
 * const model = new IsotonicRegression({ increasing: true });
 * model.fit(X, y);
 * const predictions = model.predict(tensor([1.5, 2.5]));
 * ```
 *
 * @category Linear Models
 * @implements {Regressor}
 */
export class IsotonicRegression implements Regressor {
  private options: Options;

  private xThresholds_?: Float64Array;
  private yThresholds_?: Float64Array;
  private increasing_ = true;
  private fitted = false;

  /**
   * Create a new isotonic regression model.
   *
   * @param options - Configuration options, see {@link IsotonicRegressionOptions}
   * @throws {InvalidParameterError} If an option is invalid
   */
  constructor(options: IsotonicRegressionOptions = {}) {
    this.options = resolveOptions(options);
  }

  /**
   * Fit the isotonic function.
   *
   * @param X - Training inputs, shape (n_samples,) or (n_samples, 1)
   * @param y - Training targets, shape (n_samples,) or (n_samples, 1)
   * @param sampleWeight - Optional non-negative weights of shape (n_samples,); samples with
   *   weight 0 are ignored
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X or y have the wrong shape or their lengths differ
   * @throws {DataValidationError} If the data is empty, non-finite, or all weights are zero
   */
  // biome-ignore lint/suspicious/noConfusingVoidType: `void` keeps this compatible with Estimator.fit(X, y, params)
  fit(X: Tensor, y: Tensor, sampleWeightArg?: Tensor | void): this {
    const sampleWeight = sampleWeightArg as Tensor | undefined;
    const xAll = this.extractArray(X, "X");
    const yAll = this.extractArray(y, "y");
    if (xAll.length !== yAll.length) {
      throw new ShapeError(
        `X and y must have the same number of samples; got ${xAll.length} and ${yAll.length}`
      );
    }
    if (xAll.length < 1) {
      throw new DataValidationError("X must have at least one sample");
    }
    let wAll: Float64Array | undefined;
    if (sampleWeight !== undefined) {
      wAll = this.extractArray(sampleWeight, "sampleWeight");
      if (wAll.length !== xAll.length) {
        throw new ShapeError(
          `sampleWeight must have one value per sample; got ${wAll.length} for ${xAll.length} samples`
        );
      }
      for (let i = 0; i < wAll.length; i++) {
        if ((wAll[i] as number) < 0) {
          throw new DataValidationError("sampleWeight must be non-negative");
        }
      }
    }

    // Drop zero-weight samples.
    const keep: number[] = [];
    for (let i = 0; i < xAll.length; i++) {
      if (wAll === undefined || (wAll[i] as number) > 0) keep.push(i);
    }
    if (keep.length === 0) {
      throw new DataValidationError("sampleWeight must contain at least one positive value");
    }
    const n = keep.length;
    const xs = new Float64Array(n);
    const ys = new Float64Array(n);
    const ws = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const src = keep[i] as number;
      xs[i] = xAll[src] as number;
      ys[i] = yAll[src] as number;
      ws[i] = wAll === undefined ? 1 : (wAll[src] as number);
    }

    const increasing =
      this.options.increasing === "auto" ? spearmanNonNegative(xs, ys) : this.options.increasing;

    // Sort by X, then merge samples with the same X into their weighted mean.
    const order = Array.from({ length: n }, (_, i) => i);
    order.sort(
      (a, b) => (xs[a] as number) - (xs[b] as number) || (ys[a] as number) - (ys[b] as number)
    );
    const ux: number[] = [];
    const uy: number[] = [];
    const uw: number[] = [];
    for (let k = 0; k < n; ) {
      const first = order[k] as number;
      const x0 = xs[first] as number;
      let sumW = 0;
      let sumWY = 0;
      let j = k;
      while (j < n && xs[order[j] as number] === x0) {
        const idx = order[j] as number;
        sumW += ws[idx] as number;
        sumWY += (ws[idx] as number) * (ys[idx] as number);
        j++;
      }
      ux.push(x0);
      uy.push(sumWY / sumW);
      uw.push(sumW);
      k = j;
    }
    const m = ux.length;

    // PAVA on the (possibly negated) values.
    const sign = increasing ? 1 : -1;
    const blockStart: number[] = [];
    const blockW: number[] = [];
    const blockV: number[] = [];
    for (let i = 0; i < m; i++) {
      blockStart.push(i);
      blockW.push(uw[i] as number);
      blockV.push(sign * (uy[i] as number));
      // Pool with the previous block while the ordering is violated; a merged block keeps
      // the start of the earlier one.
      while (blockV.length >= 2) {
        const last = blockV.length - 1;
        if ((blockV[last - 1] as number) <= (blockV[last] as number)) break;
        const wPrev = blockW[last - 1] as number;
        const wCur = blockW[last] as number;
        blockV[last - 1] =
          ((blockV[last - 1] as number) * wPrev + (blockV[last] as number) * wCur) / (wPrev + wCur);
        blockW[last - 1] = wPrev + wCur;
        blockStart.pop();
        blockW.pop();
        blockV.pop();
      }
    }

    const fitted = new Float64Array(m);
    for (let b = 0; b < blockV.length; b++) {
      const from = blockStart[b] as number;
      const to = b + 1 < blockV.length ? (blockStart[b + 1] as number) : m;
      let value = sign * (blockV[b] as number);
      if (this.options.yMin !== undefined && value < this.options.yMin) value = this.options.yMin;
      if (this.options.yMax !== undefined && value > this.options.yMax) value = this.options.yMax;
      for (let i = from; i < to; i++) fitted[i] = value;
    }

    // Keep only the ends of every run of equal fitted values; interpolation is unchanged.
    const xt: number[] = [];
    const yt: number[] = [];
    for (let i = 0; i < m; i++) {
      const v = fitted[i] as number;
      const interior = i > 0 && i < m - 1;
      if (interior && v === fitted[i - 1] && v === fitted[i + 1]) continue;
      xt.push(ux[i] as number);
      yt.push(v);
    }

    this.xThresholds_ = Float64Array.from(xt);
    this.yThresholds_ = Float64Array.from(yt);
    this.increasing_ = increasing;
    this.fitted = true;
    return this;
  }

  /**
   * Evaluate the fitted function.
   *
   * @param X - Inputs, shape (n_samples,) or (n_samples, 1)
   * @returns Predictions of shape (n_samples,) as a float64 tensor
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {DataValidationError} If `outOfBounds` is "raise" and an input is outside the training range
   */
  predict(X: Tensor): Tensor {
    const xt = this.xThresholds_;
    const yt = this.yThresholds_;
    if (!this.fitted || !xt || !yt) {
      throw new NotFittedError("IsotonicRegression must be fitted before predict");
    }
    const xs = this.extractArray(X, "X");
    const n = xs.length;
    const nT = xt.length;
    const xMin = xt[0] as number;
    const xMax = xt[nT - 1] as number;
    const out = new Float64Array(n);

    for (let i = 0; i < n; i++) {
      const xi = xs[i] as number;
      if (xi < xMin || xi > xMax) {
        switch (this.options.outOfBounds) {
          case "clip":
            out[i] = xi < xMin ? (yt[0] as number) : (yt[nT - 1] as number);
            break;
          case "nan":
            out[i] = Number.NaN;
            break;
          default:
            throw new DataValidationError(
              `X contains ${xi}, which is outside the training range [${xMin}, ${xMax}]`
            );
        }
        continue;
      }
      if (nT === 1) {
        out[i] = yt[0] as number;
        continue;
      }
      // Largest index lo with xt[lo] <= xi, capped so that lo + 1 exists.
      let lo = 0;
      let hi = nT - 1;
      while (lo < hi - 1) {
        const mid = (lo + hi) >>> 1;
        if ((xt[mid] as number) <= xi) lo = mid;
        else hi = mid;
      }
      const x0 = xt[lo] as number;
      const x1 = xt[hi] as number;
      const y0 = yt[lo] as number;
      const y1 = yt[hi] as number;
      out[i] = ((y1 - y0) / (x1 - x0)) * (xi - x0) + y0;
    }
    return tensor(out, { dtype: "float64" });
  }

  /**
   * Coefficient of determination R² of the predictions.
   *
   * A constant `y` scores 1 when predicted exactly and 0 otherwise.
   *
   * @param X - Inputs, shape (n_samples,) or (n_samples, 1)
   * @param y - True targets, shape (n_samples,) or (n_samples, 1)
   * @returns R² score (1 is perfect, can be negative)
   * @throws {NotFittedError} If the model has not been fitted
   */
  score(X: Tensor, y: Tensor): number {
    if (!this.fitted) {
      throw new NotFittedError("IsotonicRegression must be fitted before scoring");
    }
    const flat = y.ndim === 2 && y.shape[1] === 1 ? y.reshape([y.shape[0] ?? 0]) : y;
    return r2ScoreOf(flat, () => this.predict(X));
  }

  /** Breakpoints of the fitted function (strictly increasing). */
  get xThresholds(): Float64Array {
    if (!this.fitted || !this.xThresholds_) {
      throw new NotFittedError("IsotonicRegression must be fitted to access xThresholds");
    }
    return this.xThresholds_;
  }

  /** Fitted values at {@link IsotonicRegression.xThresholds}. */
  get yThresholds(): Float64Array {
    if (!this.fitted || !this.yThresholds_) {
      throw new NotFittedError("IsotonicRegression must be fitted to access yThresholds");
    }
    return this.yThresholds_;
  }

  /** Direction used by the last fit (resolves `increasing: "auto"`). */
  get increasing(): boolean {
    if (!this.fitted) {
      throw new NotFittedError("IsotonicRegression must be fitted to access increasing");
    }
    return this.increasing_;
  }

  /** Hyper-parameters of this estimator. `yMin` and `yMax` are included only when set. */
  getParams(): Record<string, unknown> {
    const params: Record<string, unknown> = {
      increasing: this.options.increasing,
      outOfBounds: this.options.outOfBounds,
    };
    if (this.options.yMin !== undefined) params["yMin"] = this.options.yMin;
    if (this.options.yMax !== undefined) params["yMax"] = this.options.yMax;
    return params;
  }

  /**
   * Set hyper-parameters. All values are validated before any is applied.
   *
   * @param params - Parameters to change (`increasing`, `outOfBounds`, `yMin`, `yMax`)
   * @returns this
   * @throws {InvalidParameterError} If a name is unknown or a value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    const next: Record<string, unknown> = { ...this.options };
    for (const [key, value] of Object.entries(params)) {
      if (key !== "increasing" && key !== "outOfBounds" && key !== "yMin" && key !== "yMax") {
        throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
      if ((key === "yMin" || key === "yMax") && value !== undefined && typeof value !== "number") {
        throw new InvalidParameterError(`${key} must be a number`, key, value);
      }
      next[key] = value;
    }
    this.options = resolveOptions(next as IsotonicRegressionOptions);
    return this;
  }

  /** Create an unfitted copy with the same hyper-parameters. */
  clone(): IsotonicRegression {
    return new IsotonicRegression(this.getParams() as IsotonicRegressionOptions);
  }

  /** Read a 1-D (or single-column 2-D) numeric tensor as a finite float64 array. */
  private extractArray(t: Tensor, name: string): Float64Array {
    const singleColumn = t.ndim === 2 && (t.shape[1] ?? 0) === 1;
    if (t.ndim !== 1 && !singleColumn) {
      throw new ShapeError(
        `${name} must be 1-D or 2-D with 1 column for IsotonicRegression; got shape (${t.shape.join(", ")})`
      );
    }
    assertContiguous(t, name);
    const view = toFloat64View(t);
    for (let i = 0; i < view.length; i++) {
      if (!Number.isFinite(view[i])) {
        throw new DataValidationError(`${name} contains non-finite values`);
      }
    }
    return view;
  }
}
