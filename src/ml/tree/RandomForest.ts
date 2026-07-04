import { DataValidationError, InvalidParameterError, NotFittedError, ShapeError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random } from "../../random/random";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier, Regressor } from "../base";
import { DecisionTreeClassifier, DecisionTreeRegressor } from "./DecisionTree";

/**
 * Random Forest Classifier.
 *
 * An ensemble of decision trees trained on random subsets of data and features.
 * Predictions are made by majority voting.
 *
 * **Algorithm**:
 * 1. Create n_estimators bootstrap samples from training data
 * 2. Train a decision tree on each sample with random feature subsets
 * 3. Aggregate predictions via majority voting
 *
 * @example
 * ```ts
 * import { RandomForestClassifier } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const clf = new RandomForestClassifier({ nEstimators: 100 });
 * clf.fit(X_train, y_train);
 * const predictions = clf.predict(X_test);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-tree | Deepbox Decision Trees}
 */
export class RandomForestClassifier implements Classifier {
  private nEstimators: number;
  private maxDepth: number;
  private minSamplesSplit: number;
  private minSamplesLeaf: number;
  private maxFeatures: "sqrt" | "log2" | number;
  private bootstrap: boolean;
  private warmStart: boolean;
  private randomState: number | undefined;
  private maxSamples: number | undefined;
  private oobScoreEnabled: boolean;

  private trees: DecisionTreeClassifier[] = [];
  private classLabels?: number[];
  private nFeatures?: number;
  private fitted = false;
  private oobScore_?: number;
  private oobSampleIndices_: number[][] = [];

  constructor(
    options: {
      readonly nEstimators?: number;
      readonly maxDepth?: number;
      readonly minSamplesSplit?: number;
      readonly minSamplesLeaf?: number;
      readonly maxFeatures?: "sqrt" | "log2" | number;
      readonly bootstrap?: boolean;
      readonly warmStart?: boolean;
      readonly randomState?: number;
      readonly maxSamples?: number;
      readonly oobScore?: boolean;
    } = {}
  ) {
    this.nEstimators = options.nEstimators ?? 100;
    this.maxDepth = options.maxDepth ?? 10;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 1;
    this.maxFeatures = options.maxFeatures ?? "sqrt";
    this.bootstrap = options.bootstrap ?? true;
    this.warmStart = options.warmStart ?? false;
    this.oobScoreEnabled = options.oobScore ?? false;
    if (options.randomState !== undefined) {
      this.randomState = options.randomState;
    }
    if (options.maxSamples !== undefined) {
      this.maxSamples = options.maxSamples;
    }
    if (this.oobScoreEnabled && !this.bootstrap) {
      throw new InvalidParameterError("oobScore requires bootstrap=true", "oobScore", true);
    }
    if (this.maxSamples !== undefined) {
      if (this.maxSamples <= 0) {
        throw new InvalidParameterError(
          `maxSamples must be > 0; received ${this.maxSamples}`,
          "maxSamples",
          this.maxSamples
        );
      }
    }

    if (!Number.isInteger(this.nEstimators) || this.nEstimators < 1) {
      throw new InvalidParameterError(
        `nEstimators must be an integer >= 1; received ${this.nEstimators}`,
        "nEstimators",
        this.nEstimators
      );
    }
    if (!Number.isInteger(this.maxDepth) || this.maxDepth < 1) {
      throw new InvalidParameterError(
        `maxDepth must be an integer >= 1; received ${this.maxDepth}`,
        "maxDepth",
        this.maxDepth
      );
    }
    if (!Number.isInteger(this.minSamplesSplit) || this.minSamplesSplit < 2) {
      throw new InvalidParameterError(
        `minSamplesSplit must be an integer >= 2; received ${this.minSamplesSplit}`,
        "minSamplesSplit",
        this.minSamplesSplit
      );
    }
    if (!Number.isInteger(this.minSamplesLeaf) || this.minSamplesLeaf < 1) {
      throw new InvalidParameterError(
        `minSamplesLeaf must be an integer >= 1; received ${this.minSamplesLeaf}`,
        "minSamplesLeaf",
        this.minSamplesLeaf
      );
    }
    if (typeof this.maxFeatures === "number") {
      if (!Number.isInteger(this.maxFeatures) || this.maxFeatures < 1) {
        throw new InvalidParameterError(
          `maxFeatures must be an integer >= 1; received ${this.maxFeatures}`,
          "maxFeatures",
          this.maxFeatures
        );
      }
    } else if (this.maxFeatures !== "sqrt" && this.maxFeatures !== "log2") {
      throw new InvalidParameterError(
        `maxFeatures must be "sqrt", "log2", or a positive integer; received ${String(this.maxFeatures)}`,
        "maxFeatures",
        this.maxFeatures
      );
    }
    if (options.randomState !== undefined && !Number.isFinite(options.randomState)) {
      throw new InvalidParameterError(
        `randomState must be a finite number; received ${String(options.randomState)}`,
        "randomState",
        options.randomState
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

  /**
   * Fit the random forest classifier on training data.
   *
   * Builds an ensemble of decision trees, each trained on a bootstrapped
   * sample with random feature subsets.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target class labels of shape (n_samples,)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
   * @throws {DataValidationError} If X or y contain NaN/Inf values
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeatures = nFeatures;

    // Extract data
    const XData: number[][] = [];
    const yData: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      XData.push(row);
      yData.push(Number(y.data[y.offset + i]));
    }

    this.classLabels = [...new Set(yData)].sort((a, b) => a - b);

    // Determine number of features to select
    let nSelectFeatures: number;
    if (typeof this.maxFeatures === "number") {
      nSelectFeatures = Math.min(this.maxFeatures, nFeatures);
    } else if (this.maxFeatures === "sqrt") {
      nSelectFeatures = Math.floor(Math.sqrt(nFeatures));
    } else {
      nSelectFeatures = Math.floor(Math.log2(nFeatures));
    }
    nSelectFeatures = Math.max(1, nSelectFeatures);

    const rng = this.createRNG();

    // Warm start: keep existing trees and add more
    let startIdx = 0;
    if (this.warmStart && this.fitted && this.trees.length > 0) {
      startIdx = this.trees.length;
      if (startIdx >= this.nEstimators) {
        return this;
      }
      // Advance RNG past existing trees
      if (this.bootstrap) {
        for (let skip = 0; skip < startIdx * nSamples; skip++) {
          rng();
        }
      }
    } else {
      this.trees = [];
    }

    // Determine bootstrap draw size
    let drawSize = nSamples;
    if (this.maxSamples !== undefined) {
      if (this.maxSamples > 0 && this.maxSamples < 1) {
        drawSize = Math.max(1, Math.round(this.maxSamples * nSamples));
      } else if (Number.isInteger(this.maxSamples) && this.maxSamples >= 1) {
        drawSize = Math.min(this.maxSamples, nSamples);
      }
    }

    for (let t = startIdx; t < this.nEstimators; t++) {
      // Bootstrap sample
      const sampleIndices: number[] = [];
      if (this.bootstrap) {
        for (let i = 0; i < drawSize; i++) {
          sampleIndices.push(Math.floor(rng() * nSamples));
        }
      } else {
        for (let i = 0; i < nSamples; i++) {
          sampleIndices.push(i);
        }
      }

      // Track which samples were drawn (for OOB)
      this.oobSampleIndices_.push(sampleIndices);

      // Create subset of data (all features, bootstrapped samples)
      const XSubset: number[][] = [];
      const ySubset: number[] = [];
      for (const sampleIdx of sampleIndices) {
        // We copy the whole row (all features)
        XSubset.push(XData[sampleIdx] ?? []);
        ySubset.push(yData[sampleIdx] ?? 0);
      }

      // Train tree with maxFeatures set for random feature selection at each split
      const treeOptions: {
        maxDepth: number;
        minSamplesSplit: number;
        minSamplesLeaf: number;
        maxFeatures: number;
        randomState?: number;
      } = {
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
        minSamplesLeaf: this.minSamplesLeaf,
        maxFeatures: nSelectFeatures,
      };
      if (this.randomState !== undefined) {
        treeOptions.randomState = this.randomState + t;
      }
      const tree = new DecisionTreeClassifier(treeOptions);
      tree.fit(tensor(XSubset), tensor(ySubset, { dtype: "int32" }));
      this.trees.push(tree);
    }

    this.fitted = true;

    // Compute OOB score if enabled
    if (this.oobScoreEnabled && this.bootstrap) {
      this.oobScore_ = this.computeOobScore(XData, yData, nSamples);
    }

    return this;
  }

  /**
   * Predict class labels for samples in X.
   *
   * Aggregates predictions from all trees via majority voting.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted class labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("RandomForestClassifier must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeatures ?? 0, "RandomForestClassifier");

    const nSamples = X.shape[0] ?? 0;

    // Get predictions from all trees
    const allPredictions: number[][] = [];

    for (const tree of this.trees) {
      const preds = tree.predict(X);
      const treePreds: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        treePreds.push(Number(preds.data[preds.offset + i]));
      }
      allPredictions.push(treePreds);
    }

    // Majority voting
    const finalPredictions: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      const votes = new Map<number, number>();
      for (const treePreds of allPredictions) {
        const pred = treePreds[i] ?? 0;
        votes.set(pred, (votes.get(pred) ?? 0) + 1);
      }

      let maxVotes = 0;
      let prediction = 0;
      for (const [label, count] of votes) {
        if (count > maxVotes) {
          maxVotes = count;
          prediction = label;
        }
      }
      finalPredictions.push(prediction);
    }

    return tensor(finalPredictions, { dtype: "int32" });
  }

  /**
   * Predict class probabilities for samples in X.
   *
   * Averages the predicted class probabilities from all trees in the ensemble.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Class probability matrix of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predictProba(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("RandomForestClassifier must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeatures ?? 0, "RandomForestClassifier");
    assertContiguous(X, "X");

    const nSamples = X.shape[0] ?? 0;
    const classLabels = this.classLabels ?? [];
    const nClasses = classLabels.length;

    if (nClasses === 0) {
      throw new NotFittedError("RandomForestClassifier must be fitted before prediction");
    }

    const classIndex = new Map<number, number>();
    for (let i = 0; i < nClasses; i++) {
      const v = classLabels[i];
      if (v !== undefined) classIndex.set(v, i);
    }

    const proba: number[][] = Array.from({ length: nSamples }, () =>
      new Array<number>(nClasses).fill(0)
    );

    for (const tree of this.trees) {
      const treeProba = tree.predictProba(X);
      const treeClasses = tree.classes;
      if (!treeClasses) continue;
      assertContiguous(treeClasses, "classes");

      const k = treeClasses.size;
      for (let j = 0; j < k; j++) {
        const lbl = Number(treeClasses.data[treeClasses.offset + j] ?? 0);
        const globalJ = classIndex.get(lbl);
        if (globalJ === undefined) continue;

        for (let i = 0; i < nSamples; i++) {
          const row = proba[i];
          if (row) {
            row[globalJ] =
              (row[globalJ] ?? 0) + Number(treeProba.data[treeProba.offset + i * k + j] ?? 0);
          }
        }
      }
    }

    const invTrees = this.trees.length === 0 ? 0 : 1 / this.trees.length;
    for (let i = 0; i < nSamples; i++) {
      const row = proba[i];
      if (row) {
        for (let j = 0; j < nClasses; j++) {
          row[j] = (row[j] ?? 0) * invTrees;
        }
      }
    }

    return tensor(proba);
  }

  /**
   * Return the mean accuracy on the given test data and labels.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy score in range [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or sample counts mismatch
   * @throws {DataValidationError} If y contains NaN/Inf values
   */
  score(X: Tensor, y: Tensor): number {
    if (!this.fitted) {
      throw new NotFittedError("RandomForestClassifier must be fitted before scoring");
    }
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    for (let i = 0; i < y.size; i++) {
      const val = y.data[y.offset + i] ?? 0;
      if (!Number.isFinite(val)) {
        throw new DataValidationError("y contains non-finite values (NaN or Inf)");
      }
    }
    const predictions = this.predict(X);
    if (predictions.size !== y.size) {
      throw new ShapeError(
        `X and y must have the same number of samples; got X=${predictions.size}, y=${y.size}`
      );
    }
    let correct = 0;
    for (let i = 0; i < y.size; i++) {
      if (Number(predictions.data[predictions.offset + i]) === Number(y.data[y.offset + i])) {
        correct++;
      }
    }
    return correct / y.size;
  }

  /**
   * Get the out-of-bag accuracy score.
   * Only available when oobScore=true and bootstrap=true.
   *
   * @returns OOB accuracy score in range [0, 1]
   * @throws {NotFittedError} If the model has not been fitted or oobScore was not enabled
   */
  get oobScore(): number {
    if (!this.fitted || this.oobScore_ === undefined) {
      throw new NotFittedError(
        "RandomForestClassifier must be fitted with oobScore=true to access oob_score_"
      );
    }
    return this.oobScore_;
  }

  /**
   * Compute out-of-bag score by predicting each sample using only trees
   * that did NOT include it in their bootstrap sample.
   */
  private computeOobScore(XData: number[][], yData: number[], nSamples: number): number {
    const classLabels = this.classLabels ?? [];
    const nClasses = classLabels.length;
    const classIndex = new Map<number, number>();
    for (let c = 0; c < nClasses; c++) {
      const v = classLabels[c];
      if (v !== undefined) classIndex.set(v, c);
    }

    // For each sample, accumulate class votes from OOB trees
    const oobVotes: number[][] = Array.from({ length: nSamples }, () =>
      new Array<number>(nClasses).fill(0)
    );
    const oobCount = new Array<number>(nSamples).fill(0);

    for (let t = 0; t < this.trees.length; t++) {
      const tree = this.trees[t];
      if (!tree) continue;
      const bagIndices = new Set(this.oobSampleIndices_[t] ?? []);

      // Find OOB samples (not in this tree's bag)
      const oobRows: number[][] = [];
      const oobOrigIdx: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        if (!bagIndices.has(i)) {
          oobRows.push(XData[i] ?? []);
          oobOrigIdx.push(i);
        }
      }

      if (oobRows.length === 0) continue;

      const oobX = tensor(oobRows);
      const preds = tree.predict(oobX);
      for (let j = 0; j < oobOrigIdx.length; j++) {
        const idx = oobOrigIdx[j] ?? 0;
        const pred = Number(preds.data[preds.offset + j]);
        const ci = classIndex.get(pred);
        if (ci !== undefined) {
          const row = oobVotes[idx];
          if (row) {
            row[ci] = (row[ci] ?? 0) + 1;
          }
        }
        oobCount[idx] = (oobCount[idx] ?? 0) + 1;
      }
    }

    // Compute accuracy from OOB predictions
    let correct = 0;
    let total = 0;
    for (let i = 0; i < nSamples; i++) {
      if ((oobCount[i] ?? 0) === 0) continue;
      const row = oobVotes[i];
      if (!row) continue;
      let bestClass = 0;
      let bestVotes = -1;
      for (let c = 0; c < nClasses; c++) {
        if ((row[c] ?? 0) > bestVotes) {
          bestVotes = row[c] ?? 0;
          bestClass = c;
        }
      }
      if (classLabels[bestClass] === yData[i]) {
        correct++;
      }
      total++;
    }

    return total === 0 ? 0 : correct / total;
  }

  /**
   * Get the unique class labels discovered during fitting.
   *
   * @returns Tensor of class labels or undefined if not fitted
   */
  get classes(): Tensor | undefined {
    if (!this.fitted || !this.classLabels) {
      return undefined;
    }
    return tensor(this.classLabels, { dtype: "int32" });
  }

  /**
   * Get feature importances averaged across all trees.
   *
   * @returns Tensor of shape (n_features,) with importance values summing to 1
   * @throws {NotFittedError} If the model has not been fitted
   */
  get featureImportances(): Tensor {
    if (!this.fitted || this.trees.length === 0 || this.nFeatures === undefined) {
      throw new NotFittedError(
        "RandomForestClassifier must be fitted to access feature_importances_"
      );
    }
    const nF = this.nFeatures;
    const avg = new Array<number>(nF).fill(0);
    for (const tree of this.trees) {
      const treeImp = tree.featureImportances;
      for (let j = 0; j < nF; j++) {
        avg[j] = (avg[j] ?? 0) + Number(treeImp.data[treeImp.offset + j] ?? 0);
      }
    }
    const nTrees = this.trees.length;
    let total = 0;
    for (let j = 0; j < nF; j++) {
      avg[j] = (avg[j] ?? 0) / nTrees;
      total += avg[j] ?? 0;
    }
    if (total > 0) {
      for (let j = 0; j < nF; j++) {
        avg[j] = (avg[j] ?? 0) / total;
      }
    }
    return tensor(avg);
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.nEstimators,
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      minSamplesLeaf: this.minSamplesLeaf,
      maxFeatures: this.maxFeatures,
      bootstrap: this.bootstrap,
      warmStart: this.warmStart,
      randomState: this.randomState,
      maxSamples: this.maxSamples,
      oobScore: this.oobScoreEnabled,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param params - Parameters to set
   * @throws {InvalidParameterError} If a parameter value is invalid or unknown
   */
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
        case "maxDepth":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxDepth must be an integer >= 1", "maxDepth", value);
          }
          this.maxDepth = value;
          break;
        case "minSamplesSplit":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 2) {
            throw new InvalidParameterError(
              "minSamplesSplit must be an integer >= 2",
              "minSamplesSplit",
              value
            );
          }
          this.minSamplesSplit = value;
          break;
        case "minSamplesLeaf":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "minSamplesLeaf must be an integer >= 1",
              "minSamplesLeaf",
              value
            );
          }
          this.minSamplesLeaf = value;
          break;
        case "maxFeatures":
          if (value !== "sqrt" && value !== "log2" && (typeof value !== "number" || value < 1)) {
            throw new InvalidParameterError(
              'maxFeatures must be "sqrt", "log2", or a number >= 1',
              "maxFeatures",
              value
            );
          }
          this.maxFeatures = value;
          break;
        case "bootstrap":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("bootstrap must be a boolean", "bootstrap", value);
          }
          this.bootstrap = value;
          break;
        case "warmStart":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("warmStart must be a boolean", "warmStart", value);
          }
          this.warmStart = value;
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
        case "maxSamples":
          if (value !== undefined && (typeof value !== "number" || value <= 0)) {
            throw new InvalidParameterError(
              "maxSamples must be a positive number or undefined",
              "maxSamples",
              value
            );
          }
          this.maxSamples = value;
          break;
        case "oobScore":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("oobScore must be a boolean", "oobScore", value);
          }
          this.oobScoreEnabled = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  clone(): RandomForestClassifier {
    return new RandomForestClassifier(
      this.getParams() as {
        nEstimators?: number;
        maxDepth?: number;
        minSamplesSplit?: number;
        minSamplesLeaf?: number;
        maxFeatures?: "sqrt" | "log2" | number;
        bootstrap?: boolean;
        warmStart?: boolean;
        randomState?: number;
        maxSamples?: number;
        oobScore?: boolean;
      }
    );
  }
}

/**
 * Random Forest Regressor.
 *
 * An ensemble of decision tree regressors trained on random subsets of data
 * and features. Predictions are averaged across all trees.
 *
 * **Algorithm**:
 * 1. Create n_estimators bootstrap samples from training data
 * 2. Train a decision tree on each sample with random feature subsets
 * 3. Aggregate predictions via averaging
 *
 * @example
 * ```ts
 * import { RandomForestRegressor } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const reg = new RandomForestRegressor({ nEstimators: 100 });
 * reg.fit(X_train, y_train);
 * const predictions = reg.predict(X_test);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-tree | Deepbox Decision Trees}
 */
export class RandomForestRegressor implements Regressor {
  private nEstimators: number;
  private maxDepth: number;
  private minSamplesSplit: number;
  private minSamplesLeaf: number;
  private maxFeatures: "sqrt" | "log2" | number;
  private bootstrap: boolean;
  private warmStart: boolean;
  private randomState: number | undefined;
  private maxSamples: number | undefined;
  private oobScoreEnabled: boolean;

  private trees: DecisionTreeRegressor[] = [];
  private nFeatures?: number;
  private fitted = false;
  private oobScore_?: number;
  private oobSampleIndices_: number[][] = [];

  constructor(
    options: {
      readonly nEstimators?: number;
      readonly maxDepth?: number;
      readonly minSamplesSplit?: number;
      readonly minSamplesLeaf?: number;
      readonly maxFeatures?: "sqrt" | "log2" | number;
      readonly bootstrap?: boolean;
      readonly warmStart?: boolean;
      readonly randomState?: number;
      readonly maxSamples?: number;
      readonly oobScore?: boolean;
    } = {}
  ) {
    this.nEstimators = options.nEstimators ?? 100;
    this.maxDepth = options.maxDepth ?? 10;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 1;
    this.maxFeatures = options.maxFeatures ?? 1.0;
    this.bootstrap = options.bootstrap ?? true;
    this.warmStart = options.warmStart ?? false;
    this.oobScoreEnabled = options.oobScore ?? false;
    if (options.randomState !== undefined) {
      this.randomState = options.randomState;
    }
    if (options.maxSamples !== undefined) {
      this.maxSamples = options.maxSamples;
    }
    if (this.oobScoreEnabled && !this.bootstrap) {
      throw new InvalidParameterError("oobScore requires bootstrap=true", "oobScore", true);
    }
    if (this.maxSamples !== undefined) {
      if (this.maxSamples <= 0) {
        throw new InvalidParameterError(
          `maxSamples must be > 0; received ${this.maxSamples}`,
          "maxSamples",
          this.maxSamples
        );
      }
    }

    if (!Number.isInteger(this.nEstimators) || this.nEstimators < 1) {
      throw new InvalidParameterError(
        `nEstimators must be an integer >= 1; received ${this.nEstimators}`,
        "nEstimators",
        this.nEstimators
      );
    }
    if (!Number.isInteger(this.maxDepth) || this.maxDepth < 1) {
      throw new InvalidParameterError(
        `maxDepth must be an integer >= 1; received ${this.maxDepth}`,
        "maxDepth",
        this.maxDepth
      );
    }
    if (!Number.isInteger(this.minSamplesSplit) || this.minSamplesSplit < 2) {
      throw new InvalidParameterError(
        `minSamplesSplit must be an integer >= 2; received ${this.minSamplesSplit}`,
        "minSamplesSplit",
        this.minSamplesSplit
      );
    }
    if (!Number.isInteger(this.minSamplesLeaf) || this.minSamplesLeaf < 1) {
      throw new InvalidParameterError(
        `minSamplesLeaf must be an integer >= 1; received ${this.minSamplesLeaf}`,
        "minSamplesLeaf",
        this.minSamplesLeaf
      );
    }
    if (typeof this.maxFeatures === "number") {
      if (
        this.maxFeatures !== 1.0 &&
        (!Number.isInteger(this.maxFeatures) || this.maxFeatures < 1)
      ) {
        throw new InvalidParameterError(
          `maxFeatures must be 1.0, an integer >= 1, "sqrt", or "log2"; received ${this.maxFeatures}`,
          "maxFeatures",
          this.maxFeatures
        );
      }
    } else if (this.maxFeatures !== "sqrt" && this.maxFeatures !== "log2") {
      throw new InvalidParameterError(
        `maxFeatures must be "sqrt", "log2", or a positive integer; received ${String(this.maxFeatures)}`,
        "maxFeatures",
        this.maxFeatures
      );
    }
    if (options.randomState !== undefined && !Number.isFinite(options.randomState)) {
      throw new InvalidParameterError(
        `randomState must be a finite number; received ${String(options.randomState)}`,
        "randomState",
        options.randomState
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

  /**
   * Fit the random forest regressor on training data.
   *
   * Builds an ensemble of decision trees, each trained on a bootstrapped
   * sample with random feature subsets.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Target values of shape (n_samples,)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D or y is not 1D
   * @throws {ShapeError} If X and y have different number of samples
   * @throws {DataValidationError} If X or y contain NaN/Inf values
   */
  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeatures = nFeatures;

    const XData: number[][] = [];
    const yData: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      XData.push(row);
      yData.push(Number(y.data[y.offset + i]));
    }

    let nSelectFeatures: number;
    if (typeof this.maxFeatures === "number") {
      if (this.maxFeatures === 1.0) {
        nSelectFeatures = nFeatures;
      } else {
        nSelectFeatures = Math.min(this.maxFeatures, nFeatures);
      }
    } else if (this.maxFeatures === "sqrt") {
      nSelectFeatures = Math.floor(Math.sqrt(nFeatures));
    } else {
      nSelectFeatures = Math.floor(Math.log2(nFeatures));
    }
    nSelectFeatures = Math.max(1, nSelectFeatures);

    const rng = this.createRNG();

    // Warm start: keep existing trees and add more
    let startIdx = 0;
    if (this.warmStart && this.fitted && this.trees.length > 0) {
      startIdx = this.trees.length;
      if (startIdx >= this.nEstimators) {
        return this;
      }
      // Advance RNG past existing trees
      if (this.bootstrap) {
        for (let skip = 0; skip < startIdx * nSamples; skip++) {
          rng();
        }
      }
    } else {
      this.trees = [];
    }

    // Determine bootstrap draw size
    let drawSize = nSamples;
    if (this.maxSamples !== undefined) {
      if (this.maxSamples > 0 && this.maxSamples < 1) {
        drawSize = Math.max(1, Math.round(this.maxSamples * nSamples));
      } else if (Number.isInteger(this.maxSamples) && this.maxSamples >= 1) {
        drawSize = Math.min(this.maxSamples, nSamples);
      }
    }

    for (let t = startIdx; t < this.nEstimators; t++) {
      const sampleIndices: number[] = [];
      if (this.bootstrap) {
        for (let i = 0; i < drawSize; i++) {
          sampleIndices.push(Math.floor(rng() * nSamples));
        }
      } else {
        for (let i = 0; i < nSamples; i++) {
          sampleIndices.push(i);
        }
      }

      // Track which samples were drawn (for OOB)
      this.oobSampleIndices_.push(sampleIndices);

      // Create subset of data (all features, bootstrapped samples)
      const XSubset: number[][] = [];
      const ySubset: number[] = [];
      for (const sampleIdx of sampleIndices) {
        XSubset.push(XData[sampleIdx] ?? []);
        ySubset.push(yData[sampleIdx] ?? 0);
      }

      const treeOptions: {
        maxDepth: number;
        minSamplesSplit: number;
        minSamplesLeaf: number;
        maxFeatures: number;
        randomState?: number;
      } = {
        maxDepth: this.maxDepth,
        minSamplesSplit: this.minSamplesSplit,
        minSamplesLeaf: this.minSamplesLeaf,
        maxFeatures: nSelectFeatures,
      };
      if (this.randomState !== undefined) {
        treeOptions.randomState = this.randomState + t;
      }
      const tree = new DecisionTreeRegressor(treeOptions);
      tree.fit(tensor(XSubset), tensor(ySubset));
      this.trees.push(tree);
    }

    this.fitted = true;

    // Compute OOB score if enabled
    if (this.oobScoreEnabled && this.bootstrap) {
      this.oobScore_ = this.computeOobScore(XData, yData, nSamples);
    }

    return this;
  }

  /**
   * Predict target values for samples in X.
   *
   * Averages predictions from all trees in the ensemble.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted values of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("RandomForestRegressor must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeatures ?? 0, "RandomForestRegressor");

    const nSamples = X.shape[0] ?? 0;

    const allPredictions: number[][] = [];

    for (const tree of this.trees) {
      const preds = tree.predict(X);
      const treePreds: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        treePreds.push(Number(preds.data[preds.offset + i]));
      }
      allPredictions.push(treePreds);
    }

    // Average predictions
    const finalPredictions: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      let sum = 0;
      for (const treePreds of allPredictions) {
        sum += treePreds[i] ?? 0;
      }
      finalPredictions.push(sum / this.trees.length);
    }

    return tensor(finalPredictions);
  }

  /**
   * Return the R² score on the given test data and target values.
   *
   * R² = 1 - SS_res / SS_tot, where SS_res = Σ(y - ŷ)² and SS_tot = Σ(y - ȳ)².
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True target values of shape (n_samples,)
   * @returns R² score (best possible is 1.0, can be negative)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or sample counts mismatch
   * @throws {DataValidationError} If y contains NaN/Inf values
   */
  score(X: Tensor, y: Tensor): number {
    if (!this.fitted) {
      throw new NotFittedError("RandomForestRegressor must be fitted before scoring");
    }
    if (y.ndim !== 1) {
      throw new ShapeError(`y must be 1-dimensional; got ndim=${y.ndim}`);
    }
    assertContiguous(y, "y");
    for (let i = 0; i < y.size; i++) {
      const val = y.data[y.offset + i] ?? 0;
      if (!Number.isFinite(val)) {
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
      const yTrue = Number(y.data[y.offset + i]);
      const yPred = Number(predictions.data[predictions.offset + i]);
      ssRes += (yTrue - yPred) ** 2;
      ssTot += (yTrue - yMean) ** 2;
    }

    return ssTot === 0 ? (ssRes === 0 ? 1.0 : 0.0) : 1 - ssRes / ssTot;
  }

  /**
   * Get the out-of-bag R² score.
   * Only available when oobScore=true and bootstrap=true.
   *
   * @returns OOB R² score (best possible is 1.0, can be negative)
   * @throws {NotFittedError} If the model has not been fitted or oobScore was not enabled
   */
  get oobScore(): number {
    if (!this.fitted || this.oobScore_ === undefined) {
      throw new NotFittedError(
        "RandomForestRegressor must be fitted with oobScore=true to access oob_score_"
      );
    }
    return this.oobScore_;
  }

  /**
   * Compute out-of-bag R² score by predicting each sample using only trees
   * that did NOT include it in their bootstrap sample.
   */
  private computeOobScore(XData: number[][], yData: number[], nSamples: number): number {
    // For each sample, average OOB tree predictions
    const oobSum = new Array<number>(nSamples).fill(0);
    const oobCount = new Array<number>(nSamples).fill(0);

    for (let t = 0; t < this.trees.length; t++) {
      const tree = this.trees[t];
      if (!tree) continue;
      const bagIndices = new Set(this.oobSampleIndices_[t] ?? []);

      const oobRows: number[][] = [];
      const oobOrigIdx: number[] = [];
      for (let i = 0; i < nSamples; i++) {
        if (!bagIndices.has(i)) {
          oobRows.push(XData[i] ?? []);
          oobOrigIdx.push(i);
        }
      }

      if (oobRows.length === 0) continue;

      const oobX = tensor(oobRows);
      const preds = tree.predict(oobX);
      for (let j = 0; j < oobOrigIdx.length; j++) {
        const idx = oobOrigIdx[j] ?? 0;
        oobSum[idx] = (oobSum[idx] ?? 0) + Number(preds.data[preds.offset + j]);
        oobCount[idx] = (oobCount[idx] ?? 0) + 1;
      }
    }

    // Compute R² from OOB predictions
    let yMean = 0;
    let total = 0;
    for (let i = 0; i < nSamples; i++) {
      if ((oobCount[i] ?? 0) > 0) {
        yMean += yData[i] ?? 0;
        total++;
      }
    }
    if (total === 0) return 0;
    yMean /= total;

    let ssRes = 0;
    let ssTot = 0;
    for (let i = 0; i < nSamples; i++) {
      if ((oobCount[i] ?? 0) === 0) continue;
      const yTrue = yData[i] ?? 0;
      const yPred = (oobSum[i] ?? 0) / (oobCount[i] ?? 1);
      ssRes += (yTrue - yPred) ** 2;
      ssTot += (yTrue - yMean) ** 2;
    }

    return ssTot === 0 ? (ssRes === 0 ? 1.0 : 0.0) : 1 - ssRes / ssTot;
  }

  /**
   * Get feature importances averaged across all trees.
   *
   * @returns Tensor of shape (n_features,) with importance values summing to 1
   * @throws {NotFittedError} If the model has not been fitted
   */
  get featureImportances(): Tensor {
    if (!this.fitted || this.trees.length === 0 || this.nFeatures === undefined) {
      throw new NotFittedError(
        "RandomForestRegressor must be fitted to access feature_importances_"
      );
    }
    const nF = this.nFeatures;
    const avg = new Array<number>(nF).fill(0);
    for (const tree of this.trees) {
      const treeImp = tree.featureImportances;
      for (let j = 0; j < nF; j++) {
        avg[j] = (avg[j] ?? 0) + Number(treeImp.data[treeImp.offset + j] ?? 0);
      }
    }
    const nTrees = this.trees.length;
    let total = 0;
    for (let j = 0; j < nF; j++) {
      avg[j] = (avg[j] ?? 0) / nTrees;
      total += avg[j] ?? 0;
    }
    if (total > 0) {
      for (let j = 0; j < nF; j++) {
        avg[j] = (avg[j] ?? 0) / total;
      }
    }
    return tensor(avg);
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      nEstimators: this.nEstimators,
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      minSamplesLeaf: this.minSamplesLeaf,
      maxFeatures: this.maxFeatures,
      bootstrap: this.bootstrap,
      warmStart: this.warmStart,
      randomState: this.randomState,
      maxSamples: this.maxSamples,
      oobScore: this.oobScoreEnabled,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param params - Parameters to set
   * @throws {InvalidParameterError} If a parameter value is invalid or unknown
   */
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
        case "maxDepth":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxDepth must be an integer >= 1", "maxDepth", value);
          }
          this.maxDepth = value;
          break;
        case "minSamplesSplit":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 2) {
            throw new InvalidParameterError(
              "minSamplesSplit must be an integer >= 2",
              "minSamplesSplit",
              value
            );
          }
          this.minSamplesSplit = value;
          break;
        case "minSamplesLeaf":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "minSamplesLeaf must be an integer >= 1",
              "minSamplesLeaf",
              value
            );
          }
          this.minSamplesLeaf = value;
          break;
        case "maxFeatures":
          if (value !== "sqrt" && value !== "log2" && (typeof value !== "number" || value < 1)) {
            throw new InvalidParameterError(
              'maxFeatures must be "sqrt", "log2", or a number >= 1',
              "maxFeatures",
              value
            );
          }
          this.maxFeatures = value;
          break;
        case "bootstrap":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("bootstrap must be a boolean", "bootstrap", value);
          }
          this.bootstrap = value;
          break;
        case "warmStart":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("warmStart must be a boolean", "warmStart", value);
          }
          this.warmStart = value;
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
        case "maxSamples":
          if (value !== undefined && (typeof value !== "number" || value <= 0)) {
            throw new InvalidParameterError(
              "maxSamples must be a positive number or undefined",
              "maxSamples",
              value
            );
          }
          this.maxSamples = value;
          break;
        case "oobScore":
          if (typeof value !== "boolean") {
            throw new InvalidParameterError("oobScore must be a boolean", "oobScore", value);
          }
          this.oobScoreEnabled = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  clone(): RandomForestRegressor {
    return new RandomForestRegressor(
      this.getParams() as {
        nEstimators?: number;
        maxDepth?: number;
        minSamplesSplit?: number;
        minSamplesLeaf?: number;
        maxFeatures?: "sqrt" | "log2" | number;
        bootstrap?: boolean;
        warmStart?: boolean;
        randomState?: number;
        maxSamples?: number;
        oobScore?: boolean;
      }
    );
  }
}
