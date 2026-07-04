/**
 * Semi-supervised learning: Label Propagation and Label Spreading.
 *
 * Propagates labels from labeled to unlabeled samples using a
 * similarity graph (RBF kernel). Unlabeled samples should have
 * label = -1.
 *
 * @module ml/semi_supervised
 */

import { InvalidParameterError, NotFittedError, warn } from "../core";
import { type Tensor, tensor } from "../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "./_validation";
import type { Classifier } from "./base";

/**
 * Label Propagation algorithm.
 *
 * Uses an RBF similarity kernel to propagate labels from labeled
 * to unlabeled points iteratively.
 *
 * @example
 * ```ts
 * import { LabelPropagation } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0,0],[1,0],[0,1],[1,1],[5,5],[6,5]]);
 * const y = tensor([0, -1, -1, -1, 1, -1]); // -1 = unlabeled
 * const lp = new LabelPropagation();
 * lp.fit(X, y);
 * console.log(lp.predict(X));
 * ```
 */
export class LabelPropagation implements Classifier {
  private readonly maxIter: number;
  private readonly tol: number;
  private readonly gamma: number;

  private classes_: number[] = [];
  private labelDistributions_?: Float64Array;
  private xTrain_?: Float64Array;
  private nTrainSamples_ = 0;
  private nFeaturesIn_ = 0;
  private fitted = false;

  constructor(
    options: {
      readonly maxIter?: number;
      readonly tol?: number;
      readonly gamma?: number;
    } = {}
  ) {
    this.maxIter = options.maxIter ?? 30;
    this.tol = options.tol ?? 1e-3;
    this.gamma = options.gamma ?? 20;

    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new InvalidParameterError("maxIter must be >= 1", "maxIter", this.maxIter);
    }
    if (this.gamma <= 0) {
      throw new InvalidParameterError("gamma must be > 0", "gamma", this.gamma);
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;
    this.nTrainSamples_ = n;
    this.nFeaturesIn_ = nF;

    // Store training data
    this.xTrain_ = new Float64Array(n * nF);
    for (let i = 0; i < n * nF; i++) {
      this.xTrain_[i] = Number(X.data[X.offset + i]);
    }

    const yArr = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      yArr[i] = Number(y.data[y.offset + i]);
    }

    // Extract classes (excluding -1)
    const classSet = new Set<number>();
    for (let i = 0; i < n; i++) {
      if (yArr[i] !== -1) classSet.add(yArr[i] ?? 0);
    }
    this.classes_ = [...classSet].sort((a, b) => a - b);
    const nClasses = this.classes_.length;
    const classMap = new Map<number, number>();
    for (let idx = 0; idx < this.classes_.length; idx++) {
      classMap.set(this.classes_[idx] ?? 0, idx);
    }

    // Compute affinity matrix (RBF kernel)
    const W = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let sq = 0;
        for (let f = 0; f < nF; f++) {
          const diff = (this.xTrain_[i * nF + f] ?? 0) - (this.xTrain_[j * nF + f] ?? 0);
          sq += diff * diff;
        }
        const w = Math.exp(-this.gamma * sq);
        W[i * n + j] = w;
        W[j * n + i] = w;
      }
    }

    // Compute row-normalized transition matrix T = D^{-1} W
    // Initialize label distributions Y: n x nClasses
    const Y = new Float64Array(n * nClasses);
    const isLabeled = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      if ((yArr[i] ?? 0) !== -1) {
        const cIdx = classMap.get(yArr[i] ?? 0) ?? 0;
        Y[i * nClasses + cIdx] = 1;
        isLabeled[i] = 1;
      } else {
        // Uniform initialization for unlabeled
        for (let c = 0; c < nClasses; c++) {
          Y[i * nClasses + c] = 1 / nClasses;
        }
      }
    }

    // Iterative label propagation
    for (let iter = 0; iter < this.maxIter; iter++) {
      const Ynew = new Float64Array(n * nClasses);

      for (let i = 0; i < n; i++) {
        // Compute row sum of W for normalization
        let rowSum = 0;
        for (let j = 0; j < n; j++) rowSum += W[i * n + j] ?? 0;

        if (rowSum > 0) {
          for (let c = 0; c < nClasses; c++) {
            let sum = 0;
            for (let j = 0; j < n; j++) {
              sum += ((W[i * n + j] ?? 0) / rowSum) * (Y[j * nClasses + c] ?? 0);
            }
            Ynew[i * nClasses + c] = sum;
          }
        } else {
          for (let c = 0; c < nClasses; c++) {
            Ynew[i * nClasses + c] = Y[i * nClasses + c] ?? 0;
          }
        }
      }

      // Clamp labeled points
      for (let i = 0; i < n; i++) {
        if (isLabeled[i]) {
          for (let c = 0; c < nClasses; c++) Ynew[i * nClasses + c] = Y[i * nClasses + c] ?? 0;
        }
      }

      // Check convergence
      let maxDiff = 0;
      for (let i = 0; i < n * nClasses; i++) {
        const diff = Math.abs((Ynew[i] ?? 0) - (Y[i] ?? 0));
        if (diff > maxDiff) maxDiff = diff;
      }

      for (let i = 0; i < n * nClasses; i++) Y[i] = Ynew[i] ?? 0;

      if (maxDiff < this.tol) break;
    }

    this.labelDistributions_ = Y;
    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    const proba = this.predictProba(X);
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;
    const labels = new Float64Array(nSamples);

    for (let i = 0; i < nSamples; i++) {
      let bestC = 0;
      let bestP = -Infinity;
      for (let c = 0; c < nClasses; c++) {
        const p = Number(proba.data[proba.offset + i * nClasses + c]);
        if (p > bestP) {
          bestP = p;
          bestC = c;
        }
      }
      labels[i] = this.classes_[bestC] ?? 0;
    }
    return tensor(Array.from(labels));
  }

  predictProba(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("LabelPropagation must be fitted before predict");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "LabelPropagation");
    const nTest = X.shape[0] ?? 0;
    const nF = this.nFeaturesIn_;
    const nTrain = this.nTrainSamples_;
    const nClasses = this.classes_.length;

    const result = new Float64Array(nTest * nClasses);
    for (let i = 0; i < nTest; i++) {
      // Compute RBF similarity to all training points
      let rowSum = 0;
      const weights = new Float64Array(nTrain);
      for (let j = 0; j < nTrain; j++) {
        let sq = 0;
        for (let f = 0; f < nF; f++) {
          const diff = Number(X.data[X.offset + i * nF + f]) - (this.xTrain_![j * nF + f] ?? 0);
          sq += diff * diff;
        }
        weights[j] = Math.exp(-this.gamma * sq);
        rowSum += weights[j] ?? 0;
      }

      // Weighted average of training label distributions
      for (let c = 0; c < nClasses; c++) {
        let sum = 0;
        for (let j = 0; j < nTrain; j++) {
          sum +=
            ((weights[j] ?? 0) / (rowSum > 0 ? rowSum : 1)) *
            (this.labelDistributions_![j * nClasses + c] ?? 0);
        }
        result[i * nClasses + c] = sum;
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
    if (!this.fitted) throw new NotFittedError("LabelPropagation must be fitted to access classes");
    return tensor(this.classes_);
  }

  getParams(): Record<string, unknown> {
    return { maxIter: this.maxIter, tol: this.tol, gamma: this.gamma };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}

/**
 * Label Spreading algorithm.
 *
 * Similar to {@link LabelPropagation} but uses a soft clamping factor
 * `alpha` instead of hard clamping. At each iteration the label
 * distribution for labeled points is a weighted combination of the
 * propagated distribution and the original label, controlled by alpha.
 *
 * When `alpha = 0` the algorithm degenerates to hard clamping (like
 * LabelPropagation). When `alpha = 1` the initial labels are ignored
 * entirely.
 *
 * @example
 * ```ts
 * import { LabelSpreading } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0,0],[1,0],[0,1],[1,1],[5,5],[6,5]]);
 * const y = tensor([0, -1, -1, -1, 1, -1]); // -1 = unlabeled
 * const ls = new LabelSpreading({ alpha: 0.2 });
 * ls.fit(X, y);
 * console.log(ls.predict(X));
 * ```
 */
export class LabelSpreading implements Classifier {
  private readonly maxIter: number;
  private readonly tol: number;
  private readonly gamma: number;
  private readonly alpha: number;

  private classes_: number[] = [];
  private labelDistributions_?: Float64Array;
  private xTrain_?: Float64Array;
  private nTrainSamples_ = 0;
  private nFeaturesIn_ = 0;
  private fitted = false;

  constructor(
    options: {
      readonly maxIter?: number;
      readonly tol?: number;
      readonly gamma?: number;
      readonly alpha?: number;
    } = {}
  ) {
    this.maxIter = options.maxIter ?? 30;
    this.tol = options.tol ?? 1e-3;
    this.gamma = options.gamma ?? 20;
    this.alpha = options.alpha ?? 0.2;

    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new InvalidParameterError("maxIter must be >= 1", "maxIter", this.maxIter);
    }
    if (this.gamma <= 0) {
      throw new InvalidParameterError("gamma must be > 0", "gamma", this.gamma);
    }
    if (this.alpha < 0 || this.alpha > 1) {
      throw new InvalidParameterError("alpha must be in [0, 1]", "alpha", this.alpha);
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;
    this.nTrainSamples_ = n;
    this.nFeaturesIn_ = nF;

    this.xTrain_ = new Float64Array(n * nF);
    for (let i = 0; i < n * nF; i++) {
      this.xTrain_[i] = Number(X.data[X.offset + i]);
    }

    const yArr = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      yArr[i] = Number(y.data[y.offset + i]);
    }

    const classSet = new Set<number>();
    for (let i = 0; i < n; i++) {
      if (yArr[i] !== -1) classSet.add(yArr[i] ?? 0);
    }
    this.classes_ = [...classSet].sort((a, b) => a - b);
    const nClasses = this.classes_.length;
    const classMap = new Map<number, number>();
    for (let idx = 0; idx < this.classes_.length; idx++) {
      classMap.set(this.classes_[idx] ?? 0, idx);
    }

    // Compute affinity matrix (RBF kernel)
    const W = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let sq = 0;
        for (let f = 0; f < nF; f++) {
          const diff = (this.xTrain_[i * nF + f] ?? 0) - (this.xTrain_[j * nF + f] ?? 0);
          sq += diff * diff;
        }
        const w = Math.exp(-this.gamma * sq);
        W[i * n + j] = w;
        W[j * n + i] = w;
      }
    }

    // Compute normalized graph Laplacian: S = D^{-1/2} W D^{-1/2}
    const D = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let rowSum = 0;
      for (let j = 0; j < n; j++) rowSum += W[i * n + j] ?? 0;
      D[i] = rowSum > 0 ? 1 / Math.sqrt(rowSum) : 0;
    }
    const S = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        S[i * n + j] = (D[i] ?? 0) * (W[i * n + j] ?? 0) * (D[j] ?? 0);
      }
    }

    // Initial label distributions Y0
    const Y0 = new Float64Array(n * nClasses);
    const isLabeled = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      if ((yArr[i] ?? 0) !== -1) {
        const cIdx = classMap.get(yArr[i] ?? 0) ?? 0;
        Y0[i * nClasses + cIdx] = 1;
        isLabeled[i] = 1;
      } else {
        for (let c = 0; c < nClasses; c++) {
          Y0[i * nClasses + c] = 1 / nClasses;
        }
      }
    }

    const Y = new Float64Array(Y0);

    // Iterative spreading: Y = alpha * S * Y + (1 - alpha) * Y0
    for (let iter = 0; iter < this.maxIter; iter++) {
      const Ynew = new Float64Array(n * nClasses);

      // Ynew = alpha * S * Y
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < nClasses; c++) {
          let sum = 0;
          for (let j = 0; j < n; j++) {
            sum += (S[i * n + j] ?? 0) * (Y[j * nClasses + c] ?? 0);
          }
          Ynew[i * nClasses + c] =
            this.alpha * sum + (1 - this.alpha) * (Y0[i * nClasses + c] ?? 0);
        }
      }

      // Normalize rows to sum to 1
      for (let i = 0; i < n; i++) {
        let rowSum = 0;
        for (let c = 0; c < nClasses; c++) rowSum += Ynew[i * nClasses + c] ?? 0;
        if (rowSum > 0) {
          for (let c = 0; c < nClasses; c++)
            Ynew[i * nClasses + c] = (Ynew[i * nClasses + c] ?? 0) / rowSum;
        }
      }

      let maxDiff = 0;
      for (let i = 0; i < n * nClasses; i++) {
        const diff = Math.abs((Ynew[i] ?? 0) - (Y[i] ?? 0));
        if (diff > maxDiff) maxDiff = diff;
      }

      for (let i = 0; i < n * nClasses; i++) Y[i] = Ynew[i] ?? 0;

      if (maxDiff < this.tol) break;
    }

    this.labelDistributions_ = Y;
    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    const proba = this.predictProba(X);
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;
    const labels = new Float64Array(nSamples);

    for (let i = 0; i < nSamples; i++) {
      let bestC = 0;
      let bestP = -Infinity;
      for (let c = 0; c < nClasses; c++) {
        const p = Number(proba.data[proba.offset + i * nClasses + c]);
        if (p > bestP) {
          bestP = p;
          bestC = c;
        }
      }
      labels[i] = this.classes_[bestC] ?? 0;
    }
    return tensor(Array.from(labels));
  }

  predictProba(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("LabelSpreading must be fitted before predict");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "LabelSpreading");
    const nTest = X.shape[0] ?? 0;
    const nF = this.nFeaturesIn_;
    const nTrain = this.nTrainSamples_;
    const nClasses = this.classes_.length;

    const result = new Float64Array(nTest * nClasses);
    for (let i = 0; i < nTest; i++) {
      let rowSum = 0;
      const weights = new Float64Array(nTrain);
      for (let j = 0; j < nTrain; j++) {
        let sq = 0;
        for (let f = 0; f < nF; f++) {
          const diff = Number(X.data[X.offset + i * nF + f]) - (this.xTrain_![j * nF + f] ?? 0);
          sq += diff * diff;
        }
        weights[j] = Math.exp(-this.gamma * sq);
        rowSum += weights[j] ?? 0;
      }

      for (let c = 0; c < nClasses; c++) {
        let sum = 0;
        for (let j = 0; j < nTrain; j++) {
          sum +=
            ((weights[j] ?? 0) / (rowSum > 0 ? rowSum : 1)) *
            (this.labelDistributions_![j * nClasses + c] ?? 0);
        }
        result[i * nClasses + c] = sum;
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
    if (!this.fitted) throw new NotFittedError("LabelSpreading must be fitted to access classes");
    return tensor(this.classes_);
  }

  getParams(): Record<string, unknown> {
    return {
      maxIter: this.maxIter,
      tol: this.tol,
      gamma: this.gamma,
      alpha: this.alpha,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}

/**
 * Self-Training Classifier.
 *
 * A semi-supervised meta-estimator that iteratively labels unlabeled
 * data using predictions from a supervised base classifier. In each
 * iteration, the classifier is fit on all currently-labeled data, then
 * the most confident predictions on unlabeled data (above `threshold`)
 * are added to the labeled set.
 *
 * Unlabeled samples should have label = -1.
 *
 * @example
 * ```ts
 * import { SelfTrainingClassifier } from 'deepbox/ml';
 * import { LogisticRegression } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0,0],[1,0],[0,1],[1,1],[5,5],[6,5],[5,6],[6,6]]);
 * const y = tensor([0, -1, -1, -1, 1, -1, -1, -1]); // -1 = unlabeled
 * const base = new LogisticRegression();
 * const st = new SelfTrainingClassifier({ baseEstimator: base });
 * st.fit(X, y);
 * st.predict(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-advanced | Deepbox Semi-Supervised Learning}
 *
 * References:
 * - Yarowsky, D. (1995). Unsupervised word sense disambiguation rivaling supervised methods.
 * - Triguero, I., et al. (2015). Self-labeled techniques for semi-supervised learning.
 */
export class SelfTrainingClassifier implements Classifier {
  private readonly baseEstimator: Classifier;
  private readonly threshold: number;
  private readonly maxIter: number;
  private readonly verbose: boolean;

  private estimator_?: Classifier;
  private classes_: number[] = [];
  private nFeaturesIn_ = 0;
  private nIter_ = 0;
  private labeledIter_?: Int32Array;
  private fitted = false;

  constructor(options: {
    readonly baseEstimator: Classifier;
    readonly threshold?: number;
    readonly maxIter?: number;
    readonly verbose?: boolean;
  }) {
    this.baseEstimator = options.baseEstimator;
    this.threshold = options.threshold ?? 0.75;
    this.maxIter = options.maxIter ?? 10;
    this.verbose = options.verbose ?? false;

    if (this.threshold < 0 || this.threshold > 1) {
      throw new InvalidParameterError("threshold must be in [0, 1]", "threshold", this.threshold);
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new InvalidParameterError("maxIter must be >= 1", "maxIter", this.maxIter);
    }
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nF;

    const labels = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      labels[i] = Number(y.data[y.offset + i]);
    }

    const classSet = new Set<number>();
    for (let i = 0; i < n; i++) {
      if (labels[i] !== -1) classSet.add(labels[i] ?? 0);
    }
    this.classes_ = [...classSet].sort((a, b) => a - b);
    const nClasses = this.classes_.length;

    if (nClasses < 2) {
      throw new InvalidParameterError(
        "SelfTrainingClassifier requires at least 2 classes in the labeled data",
        "y",
        nClasses
      );
    }

    this.labeledIter_ = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      this.labeledIter_[i] = labels[i] !== -1 ? -1 : -2;
    }

    const xData = new Float64Array(n * nF);
    for (let i = 0; i < n * nF; i++) {
      xData[i] = Number(X.data[X.offset + i]);
    }

    const currentLabels = new Float64Array(labels);

    for (let iter = 0; iter < this.maxIter; iter++) {
      const labeledIdx: number[] = [];
      for (let i = 0; i < n; i++) {
        if (currentLabels[i] !== -1) labeledIdx.push(i);
      }

      const nLabeled = labeledIdx.length;
      const xLabeledArr: number[] = new Array(nLabeled * nF);
      const yLabeledArr: number[] = new Array(nLabeled);
      for (let li = 0; li < nLabeled; li++) {
        const idx = labeledIdx[li]!;
        for (let f = 0; f < nF; f++) {
          xLabeledArr[li * nF + f] = xData[idx * nF + f] ?? 0;
        }
        yLabeledArr[li] = currentLabels[idx] ?? 0;
      }

      const xLabeled = tensor(xLabeledArr).reshape([nLabeled, nF]);
      const yLabeled = tensor(yLabeledArr);

      const clonedEstimator = this.cloneBase();
      clonedEstimator.fit(xLabeled, yLabeled);

      const unlabeledIdx: number[] = [];
      for (let i = 0; i < n; i++) {
        if (currentLabels[i] === -1) unlabeledIdx.push(i);
      }

      if (unlabeledIdx.length === 0) {
        this.estimator_ = clonedEstimator;
        this.nIter_ = iter + 1;
        break;
      }

      const nUnlabeled = unlabeledIdx.length;
      const xUnlabeledArr: number[] = new Array(nUnlabeled * nF);
      for (let ui = 0; ui < nUnlabeled; ui++) {
        const idx = unlabeledIdx[ui]!;
        for (let f = 0; f < nF; f++) {
          xUnlabeledArr[ui * nF + f] = xData[idx * nF + f] ?? 0;
        }
      }
      const xUnlabeled = tensor(xUnlabeledArr).reshape([nUnlabeled, nF]);

      let added = 0;

      if (typeof clonedEstimator.predictProba === "function") {
        const proba = clonedEstimator.predictProba(xUnlabeled);
        for (let ui = 0; ui < nUnlabeled; ui++) {
          let maxProba = -Infinity;
          let bestClass = 0;
          for (let c = 0; c < nClasses; c++) {
            const p = Number(proba.data[proba.offset + ui * nClasses + c]);
            if (p > maxProba) {
              maxProba = p;
              bestClass = c;
            }
          }
          if (maxProba >= this.threshold) {
            const idx = unlabeledIdx[ui]!;
            currentLabels[idx] = this.classes_[bestClass] ?? 0;
            this.labeledIter_![idx] = iter;
            added++;
          }
        }
      } else {
        const preds = clonedEstimator.predict(xUnlabeled);
        for (let ui = 0; ui < nUnlabeled; ui++) {
          const idx = unlabeledIdx[ui]!;
          currentLabels[idx] = Number(preds.data[preds.offset + ui]);
          this.labeledIter_![idx] = iter;
          added++;
        }
      }

      this.estimator_ = clonedEstimator;
      this.nIter_ = iter + 1;

      if (added === 0) {
        if (this.verbose) {
          warn("SelfTrainingClassifier stopped: no new labels assigned", "ConvergenceWarning");
        }
        break;
      }
    }

    const finalLabeledIdx: number[] = [];
    for (let i = 0; i < n; i++) {
      if (currentLabels[i] !== -1) finalLabeledIdx.push(i);
    }
    const nFinal = finalLabeledIdx.length;
    const xFinalArr: number[] = new Array(nFinal * nF);
    const yFinalArr: number[] = new Array(nFinal);
    for (let li = 0; li < nFinal; li++) {
      const idx = finalLabeledIdx[li]!;
      for (let f = 0; f < nF; f++) {
        xFinalArr[li * nF + f] = xData[idx * nF + f] ?? 0;
      }
      yFinalArr[li] = currentLabels[idx] ?? 0;
    }

    const xFinal = tensor(xFinalArr).reshape([nFinal, nF]);
    const yFinal = tensor(yFinalArr);
    const finalEstimator = this.cloneBase();
    finalEstimator.fit(xFinal, yFinal);
    this.estimator_ = finalEstimator;

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.estimator_) {
      throw new NotFittedError("SelfTrainingClassifier must be fitted before predict");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "SelfTrainingClassifier");
    return this.estimator_.predict(X);
  }

  predictProba(X: Tensor): Tensor {
    if (!this.fitted || !this.estimator_) {
      throw new NotFittedError("SelfTrainingClassifier must be fitted before predictProba");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "SelfTrainingClassifier");
    if (typeof this.estimator_.predictProba !== "function") {
      throw new InvalidParameterError(
        "Base estimator does not support predictProba",
        "baseEstimator",
        this.baseEstimator
      );
    }
    return this.estimator_.predictProba(X);
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

  get nIterations(): number {
    return this.nIter_;
  }

  get transductionLabels(): Int32Array | undefined {
    return this.labeledIter_;
  }

  getParams(): Record<string, unknown> {
    return {
      baseEstimator: this.baseEstimator,
      threshold: this.threshold,
      maxIter: this.maxIter,
      verbose: this.verbose,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }

  private cloneBase(): Classifier {
    const params = this.baseEstimator.getParams();
    const Ctor = this.baseEstimator.constructor as new (
      params: Record<string, unknown>
    ) => Classifier;
    try {
      return new Ctor(params);
    } catch {
      return this.baseEstimator;
    }
  }
}
