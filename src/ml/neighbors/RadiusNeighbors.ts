/**
 * Radius-based Neighbors Classifier and Regressor.
 *
 * Classifies/predicts based on all training samples within a given radius.
 *
 * @module ml/neighbors/RadiusNeighbors
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier, Regressor } from "../base";

export class RadiusNeighborsClassifier implements Classifier {
  private readonly radius: number;
  private xTrain_?: Float64Array;
  private yTrain_?: Float64Array;
  private classes_: number[] = [];
  private nTrainSamples_ = 0;
  private nFeaturesIn_ = 0;
  private fitted = false;

  constructor(options: { readonly radius?: number } = {}) {
    this.radius = options.radius ?? 1.0;
    if (this.radius <= 0) {
      throw new InvalidParameterError("radius must be > 0", "radius", this.radius);
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;
    this.nTrainSamples_ = n;
    this.nFeaturesIn_ = nF;
    this.xTrain_ = new Float64Array(n * nF);
    this.yTrain_ = new Float64Array(n);
    for (let i = 0; i < n * nF; i++) this.xTrain_[i] = Number(X.data[X.offset + i]);
    for (let i = 0; i < n; i++) this.yTrain_[i] = Number(y.data[y.offset + i]);
    const classSet = new Set<number>();
    for (let i = 0; i < n; i++) classSet.add(this.yTrain_[i] ?? 0);
    this.classes_ = [...classSet].sort((a, b) => a - b);
    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted)
      throw new NotFittedError("RadiusNeighborsClassifier must be fitted before predict");
    validatePredictInputs(X, this.nFeaturesIn_, "RadiusNeighborsClassifier");
    const nTest = X.shape[0] ?? 0;
    const nF = this.nFeaturesIn_;
    const nTrain = this.nTrainSamples_;
    const r2 = this.radius * this.radius;
    const labels = new Float64Array(nTest);

    for (let i = 0; i < nTest; i++) {
      const votes = new Map<number, number>();
      for (let j = 0; j < nTrain; j++) {
        let d = 0;
        for (let f = 0; f < nF; f++) {
          const diff = Number(X.data[X.offset + i * nF + f]) - (this.xTrain_![j * nF + f] ?? 0);
          d += diff * diff;
        }
        if (d <= r2) {
          const cls = this.yTrain_![j] ?? 0;
          votes.set(cls, (votes.get(cls) ?? 0) + 1);
        }
      }
      if (votes.size === 0) {
        labels[i] = this.classes_[0] ?? 0;
      } else {
        let bestCls = 0;
        let bestCount = -1;
        for (const [cls, count] of votes) {
          if (count > bestCount) {
            bestCount = count;
            bestCls = cls;
          }
        }
        labels[i] = bestCls;
      }
    }
    return tensor(Array.from(labels));
  }

  predictProba(X: Tensor): Tensor {
    if (!this.fitted)
      throw new NotFittedError("RadiusNeighborsClassifier must be fitted before predictProba");
    validatePredictInputs(X, this.nFeaturesIn_, "RadiusNeighborsClassifier");
    const nTest = X.shape[0] ?? 0;
    const nF = this.nFeaturesIn_;
    const nTrain = this.nTrainSamples_;
    const nClasses = this.classes_.length;
    const classMap = new Map<number, number>();
    for (let idx = 0; idx < this.classes_.length; idx++) {
      classMap.set(this.classes_[idx] ?? 0, idx);
    }
    const r2 = this.radius * this.radius;
    const result = new Float64Array(nTest * nClasses);

    for (let i = 0; i < nTest; i++) {
      let total = 0;
      for (let j = 0; j < nTrain; j++) {
        let d = 0;
        for (let f = 0; f < nF; f++) {
          const diff = Number(X.data[X.offset + i * nF + f]) - (this.xTrain_![j * nF + f] ?? 0);
          d += diff * diff;
        }
        if (d <= r2) {
          const cIdx = classMap.get(this.yTrain_![j] ?? 0) ?? 0;
          result[i * nClasses + cIdx] = (result[i * nClasses + cIdx] ?? 0) + 1;
          total++;
        }
      }
      if (total > 0) {
        for (let c = 0; c < nClasses; c++) {
          result[i * nClasses + c] = (result[i * nClasses + c] ?? 0) / total;
        }
      } else {
        for (let c = 0; c < nClasses; c++) result[i * nClasses + c] = 1 / nClasses;
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
    if (!this.fitted) throw new NotFittedError("RadiusNeighborsClassifier must be fitted");
    return tensor(this.classes_);
  }

  getParams(): Record<string, unknown> {
    return { radius: this.radius };
  }
  setParams(_p: Record<string, unknown>): this {
    return this;
  }
}

export class RadiusNeighborsRegressor implements Regressor {
  private readonly radius: number;
  private xTrain_?: Float64Array;
  private yTrain_?: Float64Array;
  private nTrainSamples_ = 0;
  private nFeaturesIn_ = 0;
  private fitted = false;

  constructor(options: { readonly radius?: number } = {}) {
    this.radius = options.radius ?? 1.0;
    if (this.radius <= 0) {
      throw new InvalidParameterError("radius must be > 0", "radius", this.radius);
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;
    this.nTrainSamples_ = n;
    this.nFeaturesIn_ = nF;
    this.xTrain_ = new Float64Array(n * nF);
    this.yTrain_ = new Float64Array(n);
    for (let i = 0; i < n * nF; i++) this.xTrain_[i] = Number(X.data[X.offset + i]);
    for (let i = 0; i < n; i++) this.yTrain_[i] = Number(y.data[y.offset + i]);
    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted)
      throw new NotFittedError("RadiusNeighborsRegressor must be fitted before predict");
    validatePredictInputs(X, this.nFeaturesIn_, "RadiusNeighborsRegressor");
    const nTest = X.shape[0] ?? 0;
    const nF = this.nFeaturesIn_;
    const nTrain = this.nTrainSamples_;
    const r2 = this.radius * this.radius;
    const result = new Float64Array(nTest);

    for (let i = 0; i < nTest; i++) {
      let sum = 0;
      let count = 0;
      for (let j = 0; j < nTrain; j++) {
        let d = 0;
        for (let f = 0; f < nF; f++) {
          const diff = Number(X.data[X.offset + i * nF + f]) - (this.xTrain_![j * nF + f] ?? 0);
          d += diff * diff;
        }
        if (d <= r2) {
          sum += this.yTrain_![j] ?? 0;
          count++;
        }
      }
      result[i] = count > 0 ? sum / count : 0;
    }
    return tensor(Array.from(result));
  }

  score(X: Tensor, y: Tensor): number {
    const pred = this.predict(X);
    const n = y.size;
    let ssRes = 0;
    let ssTot = 0;
    let yMean = 0;
    for (let i = 0; i < n; i++) yMean += Number(y.data[y.offset + i]);
    yMean /= n;
    for (let i = 0; i < n; i++) {
      const yi = Number(y.data[y.offset + i]);
      const pi = Number(pred.data[pred.offset + i]);
      ssRes += (yi - pi) ** 2;
      ssTot += (yi - yMean) ** 2;
    }
    return ssTot > 0 ? 1 - ssRes / ssTot : 0;
  }

  getParams(): Record<string, unknown> {
    return { radius: this.radius };
  }
  setParams(_p: Record<string, unknown>): this {
    return this;
  }
}
