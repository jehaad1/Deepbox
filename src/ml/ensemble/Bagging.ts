/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random } from "../../random/random";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier, Regressor } from "../base";
import { DecisionTreeClassifier, DecisionTreeRegressor } from "../tree/DecisionTree";

/**
 * Bagging Classifier (Bootstrap Aggregating).
 *
 * Trains multiple decision tree classifiers on bootstrap samples of the training
 * data and aggregates predictions via majority voting.
 *
 * @example
 * ```ts
 * import { BaggingClassifier } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const clf = new BaggingClassifier({ nEstimators: 10 });
 * clf.fit(X_train, y_train);
 * const predictions = clf.predict(X_test);
 * ```
 */
export class BaggingClassifier implements Classifier {
  private nEstimators: number;
  private maxSamples: number;
  private maxFeatures: number;
  private bootstrap: boolean;
  private maxDepth: number;
  private randomState: number | undefined;

  private estimators: DecisionTreeClassifier[] = [];
  private featureIndices: number[][] = [];
  private classLabels: number[] = [];
  private nFeatures = 0;
  private fitted = false;

  /**
   * @param options.nEstimators - Number of base estimators (default: 10)
   * @param options.maxSamples - Fraction of samples to draw for each estimator (default: 1.0)
   * @param options.maxFeatures - Fraction of features to draw for each estimator (default: 1.0)
   * @param options.bootstrap - Whether to use bootstrap sampling (default: true)
   * @param options.maxDepth - Maximum depth of each tree (default: unlimited/Infinity)
   * @param options.randomState - Random seed for reproducibility
   */
  constructor(
    options: {
      readonly nEstimators?: number;
      readonly maxSamples?: number;
      readonly maxFeatures?: number;
      readonly bootstrap?: boolean;
      readonly maxDepth?: number;
      readonly randomState?: number;
    } = {}
  ) {
    this.nEstimators = options.nEstimators ?? 10;
    this.maxSamples = options.maxSamples ?? 1.0;
    this.maxFeatures = options.maxFeatures ?? 1.0;
    this.bootstrap = options.bootstrap ?? true;
    this.maxDepth = options.maxDepth ?? Infinity;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.nEstimators) || this.nEstimators <= 0) {
      throw new InvalidParameterError(
        "nEstimators must be a positive integer",
        "nEstimators",
        this.nEstimators
      );
    }
    if (!Number.isFinite(this.maxSamples) || this.maxSamples <= 0 || this.maxSamples > 1) {
      throw new InvalidParameterError(
        "maxSamples must be in (0, 1]",
        "maxSamples",
        this.maxSamples
      );
    }
    if (!Number.isFinite(this.maxFeatures) || this.maxFeatures <= 0 || this.maxFeatures > 1) {
      throw new InvalidParameterError(
        "maxFeatures must be in (0, 1]",
        "maxFeatures",
        this.maxFeatures
      );
    }
  }

  private createRNG(): () => number {
    if (this.randomState !== undefined) {
      let seed = this.randomState;
      return () => {
        seed = (seed * 9301 + 49297) % 233280;
        return seed / 233280;
      };
    }
    return __random;
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeatures = nFeatures;

    const yData: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      yData.push(Number(y.data[y.offset + i]));
    }
    this.classLabels = [...new Set(yData)].sort((a, b) => a - b);

    const nSamplesDraw = Math.max(1, Math.round(this.maxSamples * nSamples));
    const nFeaturesDraw = Math.max(1, Math.round(this.maxFeatures * nFeatures));

    const rng = this.createRNG();

    this.estimators = [];
    this.featureIndices = [];

    for (let m = 0; m < this.nEstimators; m++) {
      // Select random feature subset
      const allFeatures = Array.from({ length: nFeatures }, (_, i) => i);
      const featIdx: number[] = [];
      if (nFeaturesDraw >= nFeatures) {
        featIdx.push(...allFeatures);
      } else {
        // Shuffle and take first nFeaturesDraw
        for (let i = allFeatures.length - 1; i > 0; i--) {
          const j = Math.floor(rng() * (i + 1));
          [allFeatures[i], allFeatures[j]] = [allFeatures[j]!, allFeatures[i]!];
        }
        featIdx.push(...allFeatures.slice(0, nFeaturesDraw));
      }
      this.featureIndices.push(featIdx);

      // Sample rows (with or without replacement)
      const rowIndices: number[] = [];
      if (this.bootstrap) {
        for (let s = 0; s < nSamplesDraw; s++) {
          rowIndices.push(Math.floor(rng() * nSamples));
        }
      } else {
        const allRows = Array.from({ length: nSamples }, (_, i) => i);
        for (let i = allRows.length - 1; i > 0; i--) {
          const j = Math.floor(rng() * (i + 1));
          [allRows[i], allRows[j]] = [allRows[j]!, allRows[i]!];
        }
        rowIndices.push(...allRows.slice(0, nSamplesDraw));
      }

      // Build subset data
      const xSubData: number[][] = [];
      const ySubData: number[] = [];
      for (const ri of rowIndices) {
        const row: number[] = [];
        for (const fi of featIdx) {
          row.push(Number(X.data[X.offset + ri * nFeatures + fi] ?? 0));
        }
        xSubData.push(row);
        ySubData.push(Number(y.data[y.offset + ri] ?? 0));
      }

      const treeOpts: {
        maxDepth?: number;
        minSamplesSplit: number;
        minSamplesLeaf: number;
      } = {
        minSamplesSplit: 2,
        minSamplesLeaf: 1,
      };
      if (Number.isFinite(this.maxDepth)) treeOpts.maxDepth = this.maxDepth;
      const tree = new DecisionTreeClassifier(treeOpts);
      tree.fit(tensor(xSubData), tensor(ySubData));
      this.estimators.push(tree);
    }

    this.fitted = true;
    return this;
  }

  private predictSubset(X: Tensor, featIdx: number[]): Tensor {
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const xSub: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (const fi of featIdx) {
        row.push(Number(X.data[X.offset + i * nFeatures + fi] ?? 0));
      }
      xSub.push(row);
    }
    return tensor(xSub);
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("BaggingClassifier must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "BaggingClassifier");

    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classLabels.length;
    const predictions: number[] = [];

    // Collect votes from all estimators
    const allPreds: number[][] = [];
    for (let m = 0; m < this.estimators.length; m++) {
      const tree = this.estimators[m]!;
      const xSub = this.predictSubset(X, this.featureIndices[m]!);
      const preds = tree.predict(xSub);
      const p: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        p.push(Number(preds.data[preds.offset + i]));
      }
      allPreds.push(p);
    }

    for (let i = 0; i < nSamples; i++) {
      const votes = new Array<number>(nClasses).fill(0);
      for (const p of allPreds) {
        const classIdx = this.classLabels.indexOf(p[i] ?? 0);
        if (classIdx >= 0) {
          votes[classIdx] = (votes[classIdx] ?? 0) + 1;
        }
      }
      let bestC = 0;
      let bestV = -1;
      for (let c = 0; c < nClasses; c++) {
        if ((votes[c] ?? 0) > bestV) {
          bestV = votes[c] ?? 0;
          bestC = c;
        }
      }
      predictions.push(this.classLabels[bestC] ?? 0);
    }

    return tensor(predictions, { dtype: "int32" });
  }

  predictProba(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("BaggingClassifier must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "BaggingClassifier");

    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classLabels.length;
    const probabilities: number[][] = [];

    // Average probability predictions from all estimators
    const allProba: number[][][] = [];
    for (let m = 0; m < this.estimators.length; m++) {
      const tree = this.estimators[m]!;
      const xSub = this.predictSubset(X, this.featureIndices[m]!);
      const proba = tree.predictProba(xSub);
      const treeNClasses = proba.shape[1] ?? 0;
      const p: number[][] = [];
      for (let i = 0; i < nSamples; i++) {
        const row: number[] = [];
        for (let c = 0; c < treeNClasses; c++) {
          row.push(Number(proba.data[proba.offset + i * treeNClasses + c] ?? 0));
        }
        p.push(row);
      }
      allProba.push(p);
    }

    for (let i = 0; i < nSamples; i++) {
      // Use voting-based probabilities for simplicity
      const votes = new Array<number>(nClasses).fill(0);
      for (const p of allProba) {
        const row = p[i];
        if (!row) continue;
        // Find which class this tree predicted (argmax of its proba)
        let maxC = 0;
        let maxP = -1;
        for (let c = 0; c < row.length; c++) {
          if ((row[c] ?? 0) > maxP) {
            maxP = row[c] ?? 0;
            maxC = c;
          }
        }
        // Map tree class index to global class index
        const treeClasses = this.estimators[allProba.indexOf(p)]?.classes;
        if (treeClasses) {
          const predLabel = Number(treeClasses.data[treeClasses.offset + maxC] ?? 0);
          const globalIdx = this.classLabels.indexOf(predLabel);
          if (globalIdx >= 0) {
            votes[globalIdx] = (votes[globalIdx] ?? 0) + 1;
          }
        }
      }
      const total = votes.reduce((a, b) => a + b, 0) || 1;
      probabilities.push(votes.map((v) => v / total));
    }

    return tensor(probabilities);
  }

  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    for (let i = 0; i < y.size; i++) {
      if (!Number.isFinite(y.data[y.offset + i] ?? 0)) {
        throw new DataValidationError("y contains non-finite values (NaN or Inf)");
      }
    }
    const predictions = this.predict(X);
    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if (Number(predictions.data[predictions.offset + i]) === Number(y.data[y.offset + i])) {
        correct++;
      }
    }
    return correct / y.size;
  }

  get classes(): Tensor | undefined {
    if (!this.fitted) return undefined;
    return tensor(this.classLabels, { dtype: "int32" });
  }

  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.nEstimators,
      maxSamples: this.maxSamples,
      maxFeatures: this.maxFeatures,
      bootstrap: this.bootstrap,
      maxDepth: this.maxDepth,
      randomState: this.randomState,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nEstimators":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "nEstimators must be an integer >= 1",
              "nEstimators",
              value
            );
          }
          this.nEstimators = value;
          break;
        case "maxSamples":
          if (typeof value !== "number" || value <= 0 || value > 1) {
            throw new InvalidParameterError("maxSamples must be in (0, 1]", "maxSamples", value);
          }
          this.maxSamples = value;
          break;
        case "maxFeatures":
          if (typeof value !== "number" || value <= 0 || value > 1) {
            throw new InvalidParameterError("maxFeatures must be in (0, 1]", "maxFeatures", value);
          }
          this.maxFeatures = value;
          break;
        case "bootstrap":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("bootstrap must be a boolean", "bootstrap", value);
          }
          this.bootstrap = value;
          break;
        case "maxDepth":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxDepth must be an integer >= 1", "maxDepth", value);
          }
          this.maxDepth = value;
          break;
        case "randomState":
          if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value))) {
            throw new InvalidParameterError(
              "randomState must be a finite number",
              "randomState",
              value
            );
          }
          this.randomState = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}

/**
 * Bagging Regressor (Bootstrap Aggregating for Regression).
 *
 * Trains multiple decision tree regressors on bootstrap samples and
 * averages their predictions.
 *
 * @example
 * ```ts
 * import { BaggingRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const reg = new BaggingRegressor({ nEstimators: 10 });
 * reg.fit(X_train, y_train);
 * const predictions = reg.predict(X_test);
 * ```
 */
export class BaggingRegressor implements Regressor {
  private nEstimators: number;
  private maxSamples: number;
  private maxFeatures: number;
  private bootstrap: boolean;
  private maxDepth: number;
  private randomState: number | undefined;

  private estimators: DecisionTreeRegressor[] = [];
  private featureIndices: number[][] = [];
  private nFeatures = 0;
  private fitted = false;

  constructor(
    options: {
      readonly nEstimators?: number;
      readonly maxSamples?: number;
      readonly maxFeatures?: number;
      readonly bootstrap?: boolean;
      readonly maxDepth?: number;
      readonly randomState?: number;
    } = {}
  ) {
    this.nEstimators = options.nEstimators ?? 10;
    this.maxSamples = options.maxSamples ?? 1.0;
    this.maxFeatures = options.maxFeatures ?? 1.0;
    this.bootstrap = options.bootstrap ?? true;
    this.maxDepth = options.maxDepth ?? Infinity;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.nEstimators) || this.nEstimators <= 0) {
      throw new InvalidParameterError(
        "nEstimators must be a positive integer",
        "nEstimators",
        this.nEstimators
      );
    }
    if (!Number.isFinite(this.maxSamples) || this.maxSamples <= 0 || this.maxSamples > 1) {
      throw new InvalidParameterError(
        "maxSamples must be in (0, 1]",
        "maxSamples",
        this.maxSamples
      );
    }
    if (!Number.isFinite(this.maxFeatures) || this.maxFeatures <= 0 || this.maxFeatures > 1) {
      throw new InvalidParameterError(
        "maxFeatures must be in (0, 1]",
        "maxFeatures",
        this.maxFeatures
      );
    }
  }

  private createRNG(): () => number {
    if (this.randomState !== undefined) {
      let seed = this.randomState;
      return () => {
        seed = (seed * 9301 + 49297) % 233280;
        return seed / 233280;
      };
    }
    return __random;
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeatures = nFeatures;

    const nSamplesDraw = Math.max(1, Math.round(this.maxSamples * nSamples));
    const nFeaturesDraw = Math.max(1, Math.round(this.maxFeatures * nFeatures));

    const rng = this.createRNG();

    this.estimators = [];
    this.featureIndices = [];

    for (let m = 0; m < this.nEstimators; m++) {
      const allFeatures = Array.from({ length: nFeatures }, (_, i) => i);
      const featIdx: number[] = [];
      if (nFeaturesDraw >= nFeatures) {
        featIdx.push(...allFeatures);
      } else {
        for (let i = allFeatures.length - 1; i > 0; i--) {
          const j = Math.floor(rng() * (i + 1));
          [allFeatures[i], allFeatures[j]] = [allFeatures[j]!, allFeatures[i]!];
        }
        featIdx.push(...allFeatures.slice(0, nFeaturesDraw));
      }
      this.featureIndices.push(featIdx);

      const rowIndices: number[] = [];
      if (this.bootstrap) {
        for (let s = 0; s < nSamplesDraw; s++) {
          rowIndices.push(Math.floor(rng() * nSamples));
        }
      } else {
        const allRows = Array.from({ length: nSamples }, (_, i) => i);
        for (let i = allRows.length - 1; i > 0; i--) {
          const j = Math.floor(rng() * (i + 1));
          [allRows[i], allRows[j]] = [allRows[j]!, allRows[i]!];
        }
        rowIndices.push(...allRows.slice(0, nSamplesDraw));
      }

      const xSubData: number[][] = [];
      const ySubData: number[] = [];
      for (const ri of rowIndices) {
        const row: number[] = [];
        for (const fi of featIdx) {
          row.push(Number(X.data[X.offset + ri * nFeatures + fi] ?? 0));
        }
        xSubData.push(row);
        ySubData.push(Number(y.data[y.offset + ri] ?? 0));
      }

      const regOpts: {
        maxDepth?: number;
        minSamplesSplit: number;
        minSamplesLeaf: number;
      } = {
        minSamplesSplit: 2,
        minSamplesLeaf: 1,
      };
      if (Number.isFinite(this.maxDepth)) regOpts.maxDepth = this.maxDepth;
      const tree = new DecisionTreeRegressor(regOpts);
      tree.fit(tensor(xSubData), tensor(ySubData));
      this.estimators.push(tree);
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("BaggingRegressor must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeatures, "BaggingRegressor");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const sums = new Float64Array(nSamples);

    for (let m = 0; m < this.estimators.length; m++) {
      const tree = this.estimators[m]!;
      const featIdx = this.featureIndices[m]!;
      const xSub: number[][] = [];
      for (let i = 0; i < nSamples; i++) {
        const row: number[] = [];
        for (const fi of featIdx) {
          row.push(Number(X.data[X.offset + i * nFeatures + fi] ?? 0));
        }
        xSub.push(row);
      }
      const preds = tree.predict(tensor(xSub));
      for (let i = 0; i < nSamples; i++) {
        sums[i] = (sums[i] ?? 0) + Number(preds.data[preds.offset + i]);
      }
    }

    const nEst = this.estimators.length;
    const result: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      result.push((sums[i] ?? 0) / nEst);
    }
    return tensor(result);
  }

  score(X: Tensor, y: Tensor): number {
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    for (let i = 0; i < y.size; i++) {
      if (!Number.isFinite(y.data[y.offset + i] ?? 0)) {
        throw new DataValidationError("y contains non-finite values (NaN or Inf)");
      }
    }
    const predictions = this.predict(X);
    if (predictions.size !== y.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${predictions.size}, y=${y.size}`
      );
    }
    let ssRes = 0;
    let ssTot = 0;
    let yMean = 0;
    for (let i = 0; i < y.size; i++) {
      yMean += Number(y.data[y.offset + i]);
    }
    yMean /= y.size;
    for (let i = 0; i < y.size; i++) {
      const yVal = Number(y.data[y.offset + i]);
      const pVal = Number(predictions.data[predictions.offset + i]);
      ssRes += (yVal - pVal) ** 2;
      ssTot += (yVal - yMean) ** 2;
    }
    return ssTot === 0 ? (ssRes === 0 ? 1.0 : 0.0) : 1 - ssRes / ssTot;
  }

  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.nEstimators,
      maxSamples: this.maxSamples,
      maxFeatures: this.maxFeatures,
      bootstrap: this.bootstrap,
      maxDepth: this.maxDepth,
      randomState: this.randomState,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nEstimators":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "nEstimators must be an integer >= 1",
              "nEstimators",
              value
            );
          }
          this.nEstimators = value;
          break;
        case "maxSamples":
          if (typeof value !== "number" || value <= 0 || value > 1) {
            throw new InvalidParameterError("maxSamples must be in (0, 1]", "maxSamples", value);
          }
          this.maxSamples = value;
          break;
        case "maxFeatures":
          if (typeof value !== "number" || value <= 0 || value > 1) {
            throw new InvalidParameterError("maxFeatures must be in (0, 1]", "maxFeatures", value);
          }
          this.maxFeatures = value;
          break;
        case "bootstrap":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("bootstrap must be a boolean", "bootstrap", value);
          }
          this.bootstrap = value;
          break;
        case "maxDepth":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxDepth must be an integer >= 1", "maxDepth", value);
          }
          this.maxDepth = value;
          break;
        case "randomState":
          if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value))) {
            throw new InvalidParameterError(
              "randomState must be a finite number",
              "randomState",
              value
            );
          }
          this.randomState = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
