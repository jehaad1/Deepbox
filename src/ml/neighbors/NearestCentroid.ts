/**
 * Nearest Centroid classifier.
 *
 * Classifies samples based on the nearest class centroid (mean of
 * training samples per class).
 *
 * @module ml/neighbors/NearestCentroid
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier } from "../base";

export class NearestCentroid implements Classifier {
  private centroids_?: Float64Array;
  private classes_: number[] = [];
  private nFeaturesIn_ = 0;
  private fitted = false;

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nF;

    const classSet = new Set<number>();
    for (let i = 0; i < n; i++) classSet.add(Number(y.data[y.offset + i]));
    this.classes_ = [...classSet].sort((a, b) => a - b);
    const nClasses = this.classes_.length;
    const classMap = new Map<number, number>();
    for (let idx = 0; idx < this.classes_.length; idx++) {
      classMap.set(this.classes_[idx] ?? 0, idx);
    }

    this.centroids_ = new Float64Array(nClasses * nF);
    const counts = new Float64Array(nClasses);

    for (let i = 0; i < n; i++) {
      const cIdx = classMap.get(Number(y.data[y.offset + i])) ?? 0;
      counts[cIdx] = (counts[cIdx] ?? 0) + 1;
      for (let f = 0; f < nF; f++) {
        this.centroids_[cIdx * nF + f] =
          (this.centroids_[cIdx * nF + f] ?? 0) + Number(X.data[X.offset + i * nF + f]);
      }
    }

    for (let c = 0; c < nClasses; c++) {
      const cnt = counts[c] ?? 1;
      for (let f = 0; f < nF; f++) {
        this.centroids_[c * nF + f] = (this.centroids_[c * nF + f] ?? 0) / cnt;
      }
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) throw new NotFittedError("NearestCentroid must be fitted before predict");
    validatePredictInputs(X, this.nFeaturesIn_, "NearestCentroid");
    const nTest = X.shape[0] ?? 0;
    const nF = this.nFeaturesIn_;
    const nClasses = this.classes_.length;
    const labels = new Float64Array(nTest);

    for (let i = 0; i < nTest; i++) {
      let bestC = 0;
      let bestD = Infinity;
      for (let c = 0; c < nClasses; c++) {
        let d = 0;
        for (let f = 0; f < nF; f++) {
          const diff = Number(X.data[X.offset + i * nF + f]) - (this.centroids_![c * nF + f] ?? 0);
          d += diff * diff;
        }
        if (d < bestD) {
          bestD = d;
          bestC = c;
        }
      }
      labels[i] = this.classes_[bestC] ?? 0;
    }
    return tensor(Array.from(labels));
  }

  predictProba(X: Tensor): Tensor {
    if (!this.fitted)
      throw new NotFittedError("NearestCentroid must be fitted before predictProba");
    validatePredictInputs(X, this.nFeaturesIn_, "NearestCentroid");
    const nTest = X.shape[0] ?? 0;
    const nF = this.nFeaturesIn_;
    const nClasses = this.classes_.length;
    const result = new Float64Array(nTest * nClasses);

    for (let i = 0; i < nTest; i++) {
      let sumInvD = 0;
      const invDists = new Float64Array(nClasses);
      for (let c = 0; c < nClasses; c++) {
        let d = 0;
        for (let f = 0; f < nF; f++) {
          const diff = Number(X.data[X.offset + i * nF + f]) - (this.centroids_![c * nF + f] ?? 0);
          d += diff * diff;
        }
        const invD = 1 / (Math.sqrt(d) + 1e-10);
        invDists[c] = invD;
        sumInvD += invD;
      }
      for (let c = 0; c < nClasses; c++) {
        result[i * nClasses + c] = sumInvD > 0 ? (invDists[c] ?? 0) / sumInvD : 1 / nClasses;
      }
    }
    return tensor(Array.from(result)).reshape([nTest, nClasses]);
  }

  score(X: Tensor, y: Tensor): number {
    assertContiguous(y, "y");
    const pred = this.predict(X);
    const n = y.size;
    let correct = 0;
    for (let i = 0; i < n; i++) {
      if (Number(pred.data[pred.offset + i]) === Number(y.data[y.offset + i])) correct++;
    }
    return correct / n;
  }

  get classes(): Tensor {
    if (!this.fitted) throw new NotFittedError("NearestCentroid must be fitted to access classes");
    return tensor(this.classes_);
  }

  getParams(): Record<string, unknown> {
    return {};
  }
  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}
