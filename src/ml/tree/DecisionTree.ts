import {
  DataValidationError,
  DeepboxError,
  InvalidParameterError,
  NotFittedError,
  NotImplementedError,
  ShapeError,
} from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "../_validation";
import type { Classifier, Regressor } from "../base";

type TreeNode = {
  readonly isLeaf: boolean;
  readonly prediction?: number | undefined;
  readonly classProbabilities?: number[] | undefined;
  readonly featureIndex?: number;
  readonly threshold?: number;
  readonly left?: TreeNode;
  readonly right?: TreeNode;
};

/**
 * Decision Tree Classifier.
 *
 * A non-parametric supervised learning method that learns simple decision rules
 * inferred from the data features.
 *
 * **Algorithm**: CART (Classification and Regression Trees)
 * - Uses Gini impurity for classification
 * - Recursively splits data based on feature thresholds
 * - Supports max_depth and min_samples_split for regularization
 *
 * @example
 * ```ts
 * import { DecisionTreeClassifier } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [3, 4], [5, 6], [7, 8]]);
 * const y = tensor([0, 0, 1, 1]);
 *
 * const clf = new DecisionTreeClassifier({ maxDepth: 3 });
 * clf.fit(X, y);
 * const predictions = clf.predict(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-tree | Deepbox Decision Trees}
 */
export class DecisionTreeClassifier implements Classifier {
  private maxDepth: number;
  private minSamplesSplit: number;
  private minSamplesLeaf: number;
  private maxFeatures: number | undefined;
  private randomState: number | undefined;

  private tree?: TreeNode;
  private nFeatures?: number;
  private classLabels?: number[];
  private fitted = false;

  constructor(
    options: {
      readonly maxDepth?: number;
      readonly minSamplesSplit?: number;
      readonly minSamplesLeaf?: number;
      readonly maxFeatures?: number;
      readonly randomState?: number;
    } = {}
  ) {
    this.maxDepth = options.maxDepth ?? 10;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 1;
    if (options.maxFeatures !== undefined) {
      this.maxFeatures = options.maxFeatures;
    }
    if (options.randomState !== undefined) {
      this.randomState = options.randomState;
    }

    if (this.randomState !== undefined && !Number.isFinite(this.randomState)) {
      throw new InvalidParameterError(
        `randomState must be a finite number; received ${String(this.randomState)}`,
        "randomState",
        this.randomState
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
    if (
      this.maxFeatures !== undefined &&
      (!Number.isInteger(this.maxFeatures) || this.maxFeatures < 1)
    ) {
      throw new InvalidParameterError(
        `maxFeatures must be an integer >= 1; received ${this.maxFeatures}`,
        "maxFeatures",
        this.maxFeatures
      );
    }
  }

  private getRng(): () => number {
    if (this.randomState === undefined) {
      return Math.random;
    }
    let seed = this.randomState;
    return () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    };
  }

  /**
   * Build a decision tree classifier from the training set (X, y).
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

    // Extract data as arrays
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

    // Get unique classes
    this.classLabels = [...new Set(yData)].sort((a, b) => a - b);

    // Build tree
    const indices = Array.from({ length: nSamples }, (_, i) => i);
    this.tree = this.buildTree(XData, yData, indices, 0);
    this.fitted = true;

    return this;
  }

  private buildTree(
    XData: number[][],
    yData: number[],
    indices: number[],
    depth: number
  ): TreeNode {
    const n = indices.length;

    // Check stopping conditions
    if (depth >= this.maxDepth || n < this.minSamplesSplit || n < this.minSamplesLeaf) {
      return {
        isLeaf: true,
        prediction: this.getMajorityClass(yData, indices),
        classProbabilities: this.getClassProbabilities(yData, indices),
      };
    }

    // Check if all samples have same class
    const classes = new Set(indices.map((i) => yData[i]));
    if (classes.size === 1) {
      const firstIdx = indices[0] ?? 0;
      return {
        isLeaf: true,
        prediction: yData[firstIdx] ?? 0,
        classProbabilities: this.getClassProbabilities(yData, indices),
      };
    }

    // Find best split
    const { featureIndex, threshold, leftIndices, rightIndices } = this.findBestSplit(
      XData,
      yData,
      indices
    );

    if (leftIndices.length === 0 || rightIndices.length === 0) {
      return {
        isLeaf: true,
        prediction: this.getMajorityClass(yData, indices),
        classProbabilities: this.getClassProbabilities(yData, indices),
      };
    }

    // Recursively build subtrees
    const left = this.buildTree(XData, yData, leftIndices, depth + 1);
    const right = this.buildTree(XData, yData, rightIndices, depth + 1);

    return {
      isLeaf: false,
      featureIndex,
      threshold,
      left,
      right,
    };
  }

  private getMajorityClass(yData: number[], indices: number[]): number {
    const counts = new Map<number, number>();
    for (const i of indices) {
      const label = yData[i] ?? 0;
      counts.set(label, (counts.get(label) ?? 0) + 1);
    }

    let maxCount = 0;
    let maxLabel = 0;
    for (const [label, count] of counts) {
      if (count > maxCount) {
        maxCount = count;
        maxLabel = label;
      }
    }
    return maxLabel;
  }

  private getClassProbabilities(yData: number[], indices: number[]): number[] {
    const labels = this.classLabels ?? [];
    if (labels.length === 0 || indices.length === 0) {
      return [];
    }

    const labelIndex = new Map<number, number>();
    for (let i = 0; i < labels.length; i++) {
      const v = labels[i];
      if (v !== undefined) labelIndex.set(v, i);
    }

    const counts = new Array<number>(labels.length).fill(0);
    for (const index of indices) {
      const label = yData[index] ?? 0;
      const idx = labelIndex.get(label);
      if (idx !== undefined) counts[idx] = (counts[idx] ?? 0) + 1;
    }

    const invN = 1 / indices.length;
    return counts.map((c) => c * invN);
  }

  private findBestSplit(
    XData: number[][],
    yData: number[],
    indices: number[]
  ): {
    featureIndex: number;
    threshold: number;
    leftIndices: number[];
    rightIndices: number[];
  } {
    let bestGini = Infinity;
    let bestFeature = 0;
    let bestThreshold = 0;
    let bestLeft: number[] = [];
    let bestRight: number[] = [];

    const nFeatures = XData[0]?.length ?? 0;
    let featureIndices = Array.from({ length: nFeatures }, (_, i) => i);

    if (this.maxFeatures !== undefined && this.maxFeatures < nFeatures) {
      const rng = this.getRng();
      // Fisher-Yates shuffle partial
      for (let i = 0; i < this.maxFeatures; i++) {
        const j = i + Math.floor(rng() * (nFeatures - i));
        const temp = featureIndices[i];
        if (temp !== undefined) {
          const swapVal = featureIndices[j];
          if (swapVal !== undefined) {
            featureIndices[i] = swapVal;
            featureIndices[j] = temp;
          }
        }
      }
      featureIndices = featureIndices.slice(0, this.maxFeatures);
    }

    const n = indices.length;
    // Pre-calculate total class counts
    const totalCounts = new Map<number, number>();
    for (const i of indices) {
      const label = yData[i] ?? 0;
      totalCounts.set(label, (totalCounts.get(label) ?? 0) + 1);
    }

    for (const f of featureIndices) {
      // Sort indices by feature value
      // Create a copy to sort
      const sortedIndices = [...indices].sort(
        (a, b) => (XData[a]?.[f] ?? 0) - (XData[b]?.[f] ?? 0)
      );

      const leftCounts = new Map<number, number>();
      const rightCounts = new Map<number, number>(totalCounts);
      let leftSize = 0;
      let rightSize = n;

      for (let i = 0; i < n - 1; i++) {
        const idx = sortedIndices[i];
        if (idx === undefined) continue;
        const label = yData[idx] ?? 0;
        const val = XData[idx]?.[f] ?? 0;
        const nextIdx = sortedIndices[i + 1];
        if (nextIdx === undefined) continue;
        const nextVal = XData[nextIdx]?.[f] ?? 0;

        // Move from Right to Left
        const currentRight = rightCounts.get(label) ?? 0;
        if (currentRight <= 1) rightCounts.delete(label);
        else rightCounts.set(label, currentRight - 1);
        rightSize--;

        leftCounts.set(label, (leftCounts.get(label) ?? 0) + 1);
        leftSize++;

        if (val === nextVal) continue; // Cannot split between same values

        if (leftSize < this.minSamplesLeaf || rightSize < this.minSamplesLeaf) continue;

        // Calculate weighted Gini
        const leftGini = this.giniFromCounts(leftCounts, leftSize);
        const rightGini = this.giniFromCounts(rightCounts, rightSize);
        const weightedGini = (leftSize * leftGini + rightSize * rightGini) / n;

        if (weightedGini < bestGini) {
          bestGini = weightedGini;
          bestFeature = f;
          bestThreshold = (val + nextVal) / 2;
          bestLeft = sortedIndices.slice(0, i + 1);
          bestRight = sortedIndices.slice(i + 1);
        }
      }
    }

    return {
      featureIndex: bestFeature,
      threshold: bestThreshold,
      leftIndices: bestLeft,
      rightIndices: bestRight,
    };
  }

  private giniFromCounts(counts: Map<number, number>, n: number): number {
    let impurity = 1.0;
    for (const count of counts.values()) {
      const p = count / n;
      impurity -= p * p;
    }
    return impurity;
  }

  /**
   * Predict class labels for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted class labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.tree) {
      throw new NotFittedError("DecisionTreeClassifier must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeatures ?? 0, "DecisionTreeClassifier");

    const predictions: number[] = [];
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    for (let i = 0; i < nSamples; i++) {
      const sample: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        sample.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      predictions.push(this.predictSample(sample, this.tree));
    }

    return tensor(predictions, { dtype: "int32" });
  }

  private predictSample(sample: number[], node: TreeNode): number {
    let current = node;
    while (!current.isLeaf) {
      const featureValue = sample[current.featureIndex ?? 0] ?? 0;
      if (featureValue <= (current.threshold ?? 0)) {
        if (!current.left)
          throw new DeepboxError("Corrupted tree: Internal node missing left child");
        current = current.left;
      } else {
        if (!current.right)
          throw new DeepboxError("Corrupted tree: Internal node missing right child");
        current = current.right;
      }
    }
    return current.prediction ?? 0;
  }

  /**
   * Predict class probabilities for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Class probability matrix of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predictProba(X: Tensor): Tensor {
    if (!this.fitted || !this.tree || !this.classLabels) {
      throw new NotFittedError("DecisionTreeClassifier must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeatures ?? 0, "DecisionTreeClassifier");
    assertContiguous(X, "X");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const nClasses = this.classLabels.length;

    const proba: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const sample: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        sample.push(Number(X.data[X.offset + i * nFeatures + j]));
      }

      const leaf = this.predictLeaf(sample, this.tree);
      const row = leaf.classProbabilities
        ? [...leaf.classProbabilities]
        : new Array(nClasses).fill(0);
      proba.push(row);
    }

    return tensor(proba);
  }

  private predictLeaf(sample: number[], node: TreeNode): TreeNode {
    if (node.isLeaf) {
      return node;
    }

    const featureValue = sample[node.featureIndex ?? 0] ?? 0;
    if (featureValue <= (node.threshold ?? 0)) {
      if (!node.left) {
        return node;
      }
      return this.predictLeaf(sample, node.left);
    }
    if (!node.right) {
      return node;
    }
    return this.predictLeaf(sample, node.right);
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
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      minSamplesLeaf: this.minSamplesLeaf,
      maxFeatures: this.maxFeatures,
      randomState: this.randomState,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param _params - Parameters to set
   * @throws {NotImplementedError} Always — parameters cannot be changed after construction
   */
  setParams(_params: Record<string, unknown>): this {
    throw new NotImplementedError(
      "DecisionTreeClassifier does not support setParams after construction"
    );
  }
}

/**
 * Decision Tree Regressor.
 *
 * Uses MSE reduction to find optimal splits for regression tasks.
 *
 * @see {@link https://deepbox.dev/docs/ml-tree | Deepbox Decision Trees}
 */
export class DecisionTreeRegressor implements Regressor {
  private maxDepth: number;
  private minSamplesSplit: number;
  private minSamplesLeaf: number;
  private maxFeatures: number | undefined;
  private randomState: number | undefined;

  private tree?: TreeNode;
  private nFeatures?: number;
  private fitted = false;

  constructor(
    options: {
      readonly maxDepth?: number;
      readonly minSamplesSplit?: number;
      readonly minSamplesLeaf?: number;
      readonly maxFeatures?: number;
      readonly randomState?: number;
    } = {}
  ) {
    this.maxDepth = options.maxDepth ?? 10;
    this.minSamplesSplit = options.minSamplesSplit ?? 2;
    this.minSamplesLeaf = options.minSamplesLeaf ?? 1;
    if (options.maxFeatures !== undefined) {
      this.maxFeatures = options.maxFeatures;
    }
    if (options.randomState !== undefined) {
      this.randomState = options.randomState;
    }

    if (this.randomState !== undefined && !Number.isFinite(this.randomState)) {
      throw new InvalidParameterError(
        `randomState must be a finite number; received ${String(this.randomState)}`,
        "randomState",
        this.randomState
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
    if (
      this.maxFeatures !== undefined &&
      (!Number.isInteger(this.maxFeatures) || this.maxFeatures < 1)
    ) {
      throw new InvalidParameterError(
        `maxFeatures must be an integer >= 1; received ${this.maxFeatures}`,
        "maxFeatures",
        this.maxFeatures
      );
    }
  }

  private getRng(): () => number {
    if (this.randomState === undefined) {
      return Math.random;
    }
    let seed = this.randomState;
    return () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    };
  }

  /**
   * Build a decision tree regressor from the training set (X, y).
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
    assertContiguous(X, "X");
    assertContiguous(y, "y");

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

    const indices = Array.from({ length: nSamples }, (_, i) => i);
    this.tree = this.buildTree(XData, yData, indices, 0);
    this.fitted = true;

    return this;
  }

  private buildTree(
    XData: number[][],
    yData: number[],
    indices: number[],
    depth: number
  ): TreeNode {
    const n = indices.length;
    if (n === 0) {
      throw new DataValidationError("Cannot build a decision tree from an empty dataset");
    }

    if (depth >= this.maxDepth || n < this.minSamplesSplit || n < this.minSamplesLeaf) {
      return { isLeaf: true, prediction: this.getMean(yData, indices) };
    }

    const { featureIndex, threshold, leftIndices, rightIndices } = this.findBestSplit(
      XData,
      yData,
      indices
    );

    if (leftIndices.length === 0 || rightIndices.length === 0) {
      return { isLeaf: true, prediction: this.getMean(yData, indices) };
    }

    const left = this.buildTree(XData, yData, leftIndices, depth + 1);
    const right = this.buildTree(XData, yData, rightIndices, depth + 1);

    return {
      isLeaf: false,
      featureIndex,
      threshold,
      left,
      right,
    };
  }

  private getMean(yData: number[], indices: number[]): number {
    let sum = 0;
    for (const i of indices) {
      sum += yData[i] ?? 0;
    }
    return sum / indices.length;
  }

  private findBestSplit(
    XData: number[][],
    yData: number[],
    indices: number[]
  ): {
    featureIndex: number;
    threshold: number;
    leftIndices: number[];
    rightIndices: number[];
  } {
    let bestScore = -Infinity; // We maximize the proxy score
    let bestFeature = 0;
    let bestThreshold = 0;
    let bestLeft: number[] = [];
    let bestRight: number[] = [];

    const nFeatures = XData[0]?.length ?? 0;
    let featureIndices = Array.from({ length: nFeatures }, (_, i) => i);

    if (this.maxFeatures !== undefined && this.maxFeatures < nFeatures) {
      const rng = this.getRng();
      for (let i = 0; i < this.maxFeatures; i++) {
        const j = i + Math.floor(rng() * (nFeatures - i));
        const a = featureIndices[i];
        const b = featureIndices[j];
        if (a === undefined || b === undefined) {
          throw new DeepboxError(`Internal error: featureIndices out of bounds: i=${i}, j=${j}`);
        }
        featureIndices[i] = b;
        featureIndices[j] = a;
      }
      featureIndices = featureIndices.slice(0, this.maxFeatures);
    }

    const n = indices.length;
    let totalSum = 0;

    for (const i of indices) {
      const yVal = yData[i] ?? 0;
      totalSum += yVal;
    }

    for (const f of featureIndices) {
      // Sort indices by feature value
      const sortedIndices = [...indices].sort(
        (a, b) => (XData[a]?.[f] ?? 0) - (XData[b]?.[f] ?? 0)
      );

      let leftSum = 0;
      let leftCnt = 0;
      let rightSum = totalSum;
      let rightCnt = n;

      for (let i = 0; i < n - 1; i++) {
        const idx = sortedIndices[i];
        if (idx === undefined) continue;
        const val = XData[idx]?.[f] ?? 0;
        const nextIdx = sortedIndices[i + 1];
        if (nextIdx === undefined) continue;
        const nextVal = XData[nextIdx]?.[f] ?? 0;
        const yVal = yData[idx] ?? 0;

        // Move from Right to Left
        leftSum += yVal;
        leftCnt++;
        rightSum -= yVal;
        rightCnt--;

        if (val === nextVal) continue; // Cannot split between same values

        if (leftCnt < this.minSamplesLeaf || rightCnt < this.minSamplesLeaf) continue;

        // Proxy score to maximize: (SumL^2 / nL) + (SumR^2 / nR)
        // This is equivalent to minimizing weighted MSE
        const score = (leftSum * leftSum) / leftCnt + (rightSum * rightSum) / rightCnt;

        if (score > bestScore) {
          bestScore = score;
          bestFeature = f;
          bestThreshold = (val + nextVal) / 2;
          bestLeft = sortedIndices.slice(0, i + 1);
          bestRight = sortedIndices.slice(i + 1);
        }
      }
    }

    // If no split found (e.g. pure node or all features constant), return empty
    if (bestScore === -Infinity) {
      return {
        featureIndex: 0,
        threshold: 0,
        leftIndices: [],
        rightIndices: [],
      };
    }

    return {
      featureIndex: bestFeature,
      threshold: bestThreshold,
      leftIndices: bestLeft,
      rightIndices: bestRight,
    };
  }

  /**
   * Predict target values for samples in X.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Predicted values of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has wrong dimensions or feature count
   * @throws {DataValidationError} If X contains NaN/Inf values
   */
  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.tree) {
      throw new NotFittedError("DecisionTreeRegressor must be fitted before prediction");
    }

    validatePredictInputs(X, this.nFeatures ?? 0, "DecisionTreeRegressor");
    assertContiguous(X, "X");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    const predictions: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const sample: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        sample.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      predictions.push(this.predictSample(sample, this.tree));
    }

    return tensor(predictions);
  }

  private predictSample(sample: number[], node: TreeNode): number {
    if (node.isLeaf) {
      return node.prediction ?? 0;
    }

    const featureValue = sample[node.featureIndex ?? 0] ?? 0;
    if (featureValue <= (node.threshold ?? 0)) {
      if (!node.left) throw new DeepboxError("Corrupted tree: Internal node missing left child");
      return this.predictSample(sample, node.left);
    } else {
      if (!node.right) throw new DeepboxError("Corrupted tree: Internal node missing right child");
      return this.predictSample(sample, node.right);
    }
  }

  /**
   * Return the R² score on the given test data and target values.
   *
   * R² = 1 - SS_res / SS_tot.
   *
   * @param X - Test samples of shape (n_samples, n_features)
   * @param y - True target values of shape (n_samples,)
   * @returns R² score (best possible is 1.0, can be negative)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-dimensional or sample counts mismatch
   * @throws {DataValidationError} If y contains NaN/Inf values
   */
  score(X: Tensor, y: Tensor): number {
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

    // Calculate R² score
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
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      maxDepth: this.maxDepth,
      minSamplesSplit: this.minSamplesSplit,
      minSamplesLeaf: this.minSamplesLeaf,
      maxFeatures: this.maxFeatures,
      randomState: this.randomState,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param _params - Parameters to set
   * @throws {NotImplementedError} Always — parameters cannot be changed after construction
   */
  setParams(_params: Record<string, unknown>): this {
    throw new NotImplementedError(
      "DecisionTreeRegressor does not support setParams after construction"
    );
  }
}
