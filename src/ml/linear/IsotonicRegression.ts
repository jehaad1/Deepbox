/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import type { Regressor } from "../base";

/**
 * Isotonic Regression.
 *
 * Fits a non-decreasing (or non-increasing) piecewise constant function
 * to the data using the pool adjacent violators algorithm (PAVA).
 *
 * Unlike most regressors, IsotonicRegression takes 1-D X and y.
 *
 * @example
 * ```ts
 * import { IsotonicRegression } from 'deepbox/ml';
 *
 * const model = new IsotonicRegression({ increasing: true });
 * model.fit(X_train, y_train);  // X is 1-D or single-column 2-D
 * const predictions = model.predict(X_test);
 * ```
 *
 * @category Linear Models
 * @implements {Regressor}
 */
export class IsotonicRegression implements Regressor {
  private options: {
    increasing: boolean;
    outOfBounds: "clip" | "nan";
  };

  private xThresholds_?: Float64Array;
  private yThresholds_?: Float64Array;
  private nThresholds_?: number;
  private fitted = false;

  constructor(
    options: {
      readonly increasing?: boolean;
      readonly outOfBounds?: "clip" | "nan";
    } = {}
  ) {
    this.options = {
      increasing: options.increasing ?? true,
      outOfBounds: options.outOfBounds ?? "clip",
    };
  }

  fit(X: Tensor, y: Tensor): IsotonicRegression {
    // Extract 1-D arrays from X and y
    const xArr = this.extractArray(X, "X");
    const yArr = this.extractArray(y, "y");
    const n = xArr.length;

    if (n !== yArr.length) {
      throw new ShapeError(
        `X and y must have the same number of samples; got ${n} and ${yArr.length}`
      );
    }
    if (n < 1) {
      throw new DataValidationError("X must have at least one sample");
    }

    // Sort by X values
    const indices = Array.from({ length: n }, (_, i) => i);
    indices.sort((a, b) => (xArr[a] ?? 0) - (xArr[b] ?? 0));

    const sortedX = new Float64Array(n);
    const sortedY = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      sortedX[i] = xArr[indices[i] ?? 0] ?? 0;
      sortedY[i] = yArr[indices[i] ?? 0] ?? 0;
    }

    // If decreasing, negate Y, run PAVA, then negate back
    if (!this.options.increasing) {
      for (let i = 0; i < n; i++) {
        sortedY[i] = -(sortedY[i] ?? 0);
      }
    }

    // Pool Adjacent Violators Algorithm (PAVA)
    const blocks: { start: number; end: number; value: number }[] = [];
    for (let i = 0; i < n; i++) {
      blocks.push({ start: i, end: i, value: sortedY[i] ?? 0 });
      // Merge while violating monotonicity
      while (blocks.length >= 2) {
        const curr = blocks[blocks.length - 1]!;
        const prev = blocks[blocks.length - 2]!;
        if (prev.value <= curr.value) break;
        // Merge: weighted average
        const nPrev = prev.end - prev.start + 1;
        const nCurr = curr.end - curr.start + 1;
        prev.value = (prev.value * nPrev + curr.value * nCurr) / (nPrev + nCurr);
        prev.end = curr.end;
        blocks.pop();
      }
    }

    // Extract thresholds (unique x breakpoints with their y values)
    const xThresh: number[] = [];
    const yThresh: number[] = [];

    for (const block of blocks) {
      const xStart = sortedX[block.start] ?? 0;
      const xEnd = sortedX[block.end] ?? 0;
      let yVal = block.value;
      if (!this.options.increasing) yVal = -yVal;

      if (xThresh.length === 0 || xStart !== xThresh[xThresh.length - 1]) {
        xThresh.push(xStart);
        yThresh.push(yVal);
      }
      if (xStart !== xEnd) {
        xThresh.push(xEnd);
        yThresh.push(yVal);
      }
    }

    this.xThresholds_ = new Float64Array(xThresh);
    this.yThresholds_ = new Float64Array(yThresh);
    this.nThresholds_ = xThresh.length;
    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.xThresholds_ || !this.yThresholds_) {
      throw new NotFittedError("IsotonicRegression");
    }

    const xArr = this.extractArray(X, "X");
    const n = xArr.length;
    const predictions = new Float64Array(n);
    const nT = this.nThresholds_ ?? 0;

    for (let i = 0; i < n; i++) {
      const xi = xArr[i] ?? 0;
      const xMin = this.xThresholds_[0] ?? 0;
      const xMax = this.xThresholds_[nT - 1] ?? 0;

      if (xi <= xMin) {
        predictions[i] =
          this.options.outOfBounds === "clip" ? (this.yThresholds_[0] ?? 0) : Number.NaN;
      } else if (xi >= xMax) {
        predictions[i] =
          this.options.outOfBounds === "clip" ? (this.yThresholds_[nT - 1] ?? 0) : Number.NaN;
      } else {
        // Binary search for interval
        let lo = 0;
        let hi = nT - 1;
        while (lo < hi - 1) {
          const mid = Math.floor((lo + hi) / 2);
          if ((this.xThresholds_[mid] ?? 0) <= xi) lo = mid;
          else hi = mid;
        }
        // Linear interpolation
        const x0 = this.xThresholds_[lo] ?? 0;
        const x1 = this.xThresholds_[hi] ?? 0;
        const y0 = this.yThresholds_[lo] ?? 0;
        const y1 = this.yThresholds_[hi] ?? 0;
        const t = x1 !== x0 ? (xi - x0) / (x1 - x0) : 0;
        predictions[i] = y0 + t * (y1 - y0);
      }
    }

    return tensor(Array.from(predictions), { dtype: "float64" });
  }

  score(X: Tensor, y: Tensor): number {
    const pred = this.predict(X);
    const yArr = this.extractArray(y, "y");
    const n = yArr.length;
    let ssRes = 0;
    let ssTot = 0;
    let yMean = 0;
    for (let i = 0; i < n; i++) yMean += yArr[i] ?? 0;
    yMean /= n;
    for (let i = 0; i < n; i++) {
      const yi = yArr[i] ?? 0;
      const pi = Number(pred.data[i] ?? 0);
      ssRes += (yi - pi) ** 2;
      ssTot += (yi - yMean) ** 2;
    }
    return ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  }

  getParams(): Record<string, unknown> {
    return { ...this.options };
  }

  setParams(params: Record<string, unknown>): IsotonicRegression {
    if (params["increasing"] !== undefined)
      this.options.increasing = params["increasing"] as boolean;
    if (params["outOfBounds"] !== undefined)
      this.options.outOfBounds = params["outOfBounds"] as "clip" | "nan";
    return this;
  }

  private extractArray(t: Tensor, name: string): Float64Array {
    if (t.ndim === 1) {
      const arr = new Float64Array(t.size);
      for (let i = 0; i < t.size; i++) {
        const v = Number(t.data[(t.offset ?? 0) + i] ?? 0);
        if (!Number.isFinite(v)) {
          throw new DataValidationError(`${name} contains non-finite values`);
        }
        arr[i] = v;
      }
      return arr;
    }
    if (t.ndim === 2 && (t.shape[1] ?? 0) === 1) {
      const n = t.shape[0] ?? 0;
      const arr = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        const v = Number(t.data[(t.offset ?? 0) + i] ?? 0);
        if (!Number.isFinite(v)) {
          throw new DataValidationError(`${name} contains non-finite values`);
        }
        arr[i] = v;
      }
      return arr;
    }
    throw new ShapeError(
      `${name} must be 1-D or 2-D with 1 column for IsotonicRegression; got shape (${t.shape.join(", ")})`
    );
  }
}
