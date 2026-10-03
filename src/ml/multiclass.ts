/**
 * Multiclass meta-estimators.
 *
 * Provides `OneVsRestClassifier` and `OneVsOneClassifier` that decompose
 * a multiclass classification problem into multiple binary classification
 * problems.
 *
 * @module ml/multiclass
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import {
  DataValidationError,
  InvalidParameterError,
  NotFittedError,
  NotImplementedError,
  ShapeError,
} from "../core";
import { type Tensor, tensor } from "../ndarray";
import { cloneEstimator } from "./_internal";
import { toFloat64View, validateFitInputs, validatePredictInputs } from "./_validation";
import type { Classifier } from "./base";
import { nbAccuracy, nbLabelTensor, nbMatrixTensor, nbValidateScoreTarget } from "./naive_bayes";

type BinaryEstimatorOptions = { readonly estimator: Classifier };

/** Check the `estimator` option shared by both meta-estimators. */
function validateEstimator(estimator: unknown, owner: string): Classifier {
  const candidate = estimator as Record<string, unknown> | null | undefined;
  if (
    candidate === null ||
    candidate === undefined ||
    typeof candidate !== "object" ||
    typeof candidate["fit"] !== "function" ||
    typeof candidate["predict"] !== "function" ||
    typeof candidate["getParams"] !== "function"
  ) {
    throw new InvalidParameterError(
      `${owner} needs an estimator with fit(), predict() and getParams()`,
      "estimator",
      estimator
    );
  }
  return estimator as Classifier;
}

/** Sorted unique labels and the index of every sample's class. */
function encodeClasses(y: Tensor, owner: string): { classes: number[]; labelIndex: Int32Array } {
  const values = toFloat64View(y);
  const unique = new Set<number>();
  for (let i = 0; i < values.length; i++) unique.add((values[i] as number) + 0);
  const classes = Array.from(unique).sort((a, b) => a - b);
  if (classes.length < 2) {
    throw new DataValidationError(`${owner} needs at least 2 classes in y; got ${classes.length}`);
  }
  const index = new Map<number, number>();
  for (let c = 0; c < classes.length; c++) index.set(classes[c] as number, c);
  const labelIndex = new Int32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    labelIndex[i] = index.get((values[i] as number) + 0) as number;
  }
  return { classes, labelIndex };
}

/** Positive-class column of a binary score tensor shaped (n,), (n, 1) or (n, 2). */
function positiveColumn(scores: Tensor, nSamples: number, source: string): Float64Array {
  const values = toFloat64View(scores);
  if (scores.ndim === 1 && scores.size === nSamples) return values;
  if (scores.ndim === 2 && scores.shape[0] === nSamples) {
    const nCols = scores.shape[1] ?? 0;
    if (nCols === 1) return values;
    if (nCols === 2) {
      const out = new Float64Array(nSamples);
      for (let i = 0; i < nSamples; i++) out[i] = values[2 * i + 1] as number;
      return out;
    }
  }
  throw new ShapeError(
    `${source} of the binary estimator returned shape [${scores.shape.join(", ")}]; expected (${nSamples},), (${nSamples}, 1) or (${nSamples}, 2)`
  );
}

type WithDecisionFunction = { decisionFunction?: (X: Tensor) => Tensor };

/**
 * Confidence that each sample belongs to the positive class of a binary estimator:
 * `decisionFunction` when the estimator has it, otherwise the positive-class column of
 * `predictProba`. Returns `null` when the estimator offers neither.
 */
function binaryConfidence(clf: Classifier, X: Tensor, nSamples: number): Float64Array | null {
  const decision = (clf as unknown as WithDecisionFunction).decisionFunction;
  // A method taking more than one argument is not the public `decisionFunction(X)`; some
  // estimators keep a private helper of that name, which must not be called with X alone.
  if (typeof decision === "function" && decision.length <= 1) {
    try {
      return positiveColumn(decision.call(clf, X), nSamples, "decisionFunction");
    } catch (error) {
      if (!(error instanceof NotImplementedError)) throw error;
    }
  }
  if (typeof clf.predictProba === "function") {
    try {
      return positiveColumn(clf.predictProba(X), nSamples, "predictProba");
    } catch (error) {
      if (!(error instanceof NotImplementedError)) throw error;
    }
  }
  return null;
}

/**
 * One-vs-Rest (OvR) multiclass strategy.
 *
 * Fits one binary classifier per class. For class k, the positive
 * examples are from class k and negative examples are all other classes.
 *
 * Prediction picks the class whose binary classifier is most confident. The confidence
 * is the estimator's `decisionFunction` when it has one, otherwise the positive-class
 * column of `predictProba`, otherwise its hard 0/1 `predict` output (ties go to the
 * smallest label). Each binary problem uses labels 0 (rest) and 1 (class k).
 *
 * @example
 * ```ts
 * import { OneVsRestClassifier, LogisticRegression } from 'deepbox/ml';
 * const ovr = new OneVsRestClassifier({ estimator: new LogisticRegression() });
 * ovr.fit(X_train, y_train);
 * const pred = ovr.predict(X_test);
 * ```
 */
export class OneVsRestClassifier implements Classifier {
  private estimator: Classifier;
  private classifiers_: Classifier[] = [];
  private classes_: number[] = [];
  private nFeaturesIn_ = 0;
  private fitted = false;

  /**
   * @param options.estimator - Binary classifier template. It is cloned for every class (through `clone()` or its constructor and `getParams()`) and never fitted itself.
   * @throws {InvalidParameterError} If `estimator` is missing or lacks `fit`, `predict` or `getParams`
   */
  constructor(options: BinaryEstimatorOptions) {
    this.estimator = validateEstimator(options?.estimator, "OneVsRestClassifier");
  }

  /**
   * Fit one binary classifier per class.
   *
   * @param X - Training samples of shape (n_samples, n_features)
   * @param y - Class labels of shape (n_samples,)
   * @returns this
   * @throws {DataValidationError} If y contains fewer than two classes
   * @throws {ShapeError} If X is not 2-D, y is not 1-D or their lengths differ
   */
  fit(X: Tensor, y: Tensor): this {
    this.fitted = false;
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const { classes } = encodeClasses(y, "OneVsRestClassifier");
    const values = toFloat64View(y);

    const classifiers: Classifier[] = [];
    for (const cls of classes) {
      const binaryY = new Float64Array(nSamples);
      for (let i = 0; i < nSamples; i++) binaryY[i] = (values[i] as number) + 0 === cls ? 1 : 0;
      const clf = cloneEstimator(this.estimator, "OneVsRestClassifier");
      clf.fit(X, tensor(binaryY));
      classifiers.push(clf);
    }

    this.classes_ = classes;
    this.classifiers_ = classifiers;
    this.nFeaturesIn_ = nFeatures;
    this.fitted = true;
    return this;
  }

  private assertFitted(method: string): void {
    if (!this.fitted) {
      throw new NotFittedError(`OneVsRestClassifier must be fitted before ${method}`);
    }
  }

  /** Confidence of every binary classifier, shape (n_samples * n_classes), row-major. */
  private confidences(X: Tensor): Float64Array {
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;
    const scores = new Float64Array(nSamples * nClasses);
    for (let c = 0; c < nClasses; c++) {
      const clf = this.classifiers_[c] as Classifier;
      const column = binaryConfidence(clf, X, nSamples) ?? toFloat64View(clf.predict(X));
      for (let i = 0; i < nSamples; i++) scores[i * nClasses + c] = column[i] as number;
    }
    return scores;
  }

  /**
   * Predict class labels.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  predict(X: Tensor): Tensor {
    this.assertFitted("predict");
    validatePredictInputs(X, this.nFeaturesIn_, "OneVsRestClassifier");
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;
    const scores = this.confidences(X);

    const labels = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let bestC = 0;
      let bestScore = -Infinity;
      for (let c = 0; c < nClasses; c++) {
        const s = scores[i * nClasses + c] as number;
        if (s > bestScore) {
          bestScore = s;
          bestC = c;
        }
      }
      labels[i] = this.classes_[bestC] as number;
    }
    return nbLabelTensor(labels, this.classes_);
  }

  /**
   * Confidence score of every class, one column per class.
   *
   * Each column is the `decisionFunction` of that class's binary estimator when it has
   * one, otherwise its positive-class probability, otherwise its 0/1 prediction.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Scores of shape (n_samples, n_classes), columns follow `classes`
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  decisionFunction(X: Tensor): Tensor {
    this.assertFitted("decisionFunction");
    validatePredictInputs(X, this.nFeaturesIn_, "OneVsRestClassifier");
    return nbMatrixTensor(this.confidences(X), X.shape[0] ?? 0, this.classes_.length);
  }

  /**
   * Estimate class probabilities.
   *
   * The positive-class probabilities of the binary estimators are normalized to sum to 1
   * per sample (rows where all of them are 0 become uniform).
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes), columns follow `classes`
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {NotImplementedError} If the base estimator has no `predictProba`
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  predictProba(X: Tensor): Tensor {
    this.assertFitted("predictProba");
    validatePredictInputs(X, this.nFeaturesIn_, "OneVsRestClassifier");
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;

    const result = new Float64Array(nSamples * nClasses);
    for (let c = 0; c < nClasses; c++) {
      const clf = this.classifiers_[c] as Classifier;
      if (typeof clf.predictProba !== "function") {
        throw new NotImplementedError(
          "The base estimator of OneVsRestClassifier has no predictProba"
        );
      }
      const column = positiveColumn(clf.predictProba(X), nSamples, "predictProba");
      for (let i = 0; i < nSamples; i++) result[i * nClasses + c] = column[i] as number;
    }

    for (let i = 0; i < nSamples; i++) {
      let sum = 0;
      for (let c = 0; c < nClasses; c++) sum += result[i * nClasses + c] as number;
      for (let c = 0; c < nClasses; c++) {
        result[i * nClasses + c] =
          sum === 0 ? 1 / nClasses : (result[i * nClasses + c] as number) / sum;
      }
    }
    return nbMatrixTensor(result, nSamples, nClasses);
  }

  /**
   * Mean accuracy on the given data and labels.
   *
   * @param X - Test samples
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy in [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-D or its length differs from the number of samples
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    nbValidateScoreTarget(y);
    return nbAccuracy(this.predict(X), y);
  }

  /**
   * Sorted class labels seen during `fit`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get classes(): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("OneVsRestClassifier must be fitted to access classes");
    }
    return nbLabelTensor(this.classes_, this.classes_);
  }

  /** Parameters of this estimator: the template `estimator`. */
  getParams(): Record<string, unknown> {
    return { estimator: this.estimator };
  }

  /**
   * Replace the template estimator. The model must be refitted afterwards.
   *
   * @throws {InvalidParameterError} If the key is unknown or `estimator` is not a valid estimator
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      if (key === "estimator") {
        this.estimator = validateEstimator(value, "OneVsRestClassifier");
      } else {
        throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /** Create an unfitted copy with a cloned template estimator. */
  clone(): OneVsRestClassifier {
    return new OneVsRestClassifier({
      estimator: cloneEstimator(this.estimator, "OneVsRestClassifier"),
    });
  }
}

/**
 * One-vs-One (OvO) multiclass strategy.
 *
 * Fits one classifier for each pair of classes (the lower label of the pair is passed
 * to the binary estimator as 0, the higher as 1). Each pairwise classifier votes for one
 * class. Ties between classes with equal votes are broken by the summed pairwise
 * confidences (`decisionFunction`, or the positive-class probability when the base
 * estimator has no decision function), as in scikit-learn.
 *
 * @example
 * ```ts
 * import { OneVsOneClassifier, LinearSVC } from 'deepbox/ml';
 * const ovo = new OneVsOneClassifier({ estimator: new LinearSVC() });
 * ovo.fit(X_train, y_train);
 * const pred = ovo.predict(X_test);
 * ```
 */
export class OneVsOneClassifier implements Classifier {
  private estimator: Classifier;
  private classifiers_: Classifier[] = [];
  private classPairs_: [number, number][] = []; // class indices (i, j) with i < j
  private classes_: number[] = [];
  private nFeaturesIn_ = 0;
  private fitted = false;

  /**
   * @param options.estimator - Binary classifier template. It is cloned for every pair of classes and never fitted itself.
   * @throws {InvalidParameterError} If `estimator` is missing or lacks `fit`, `predict` or `getParams`
   */
  constructor(options: BinaryEstimatorOptions) {
    this.estimator = validateEstimator(options?.estimator, "OneVsOneClassifier");
  }

  /**
   * Fit one binary classifier for every pair of classes.
   *
   * @param X - Training samples of shape (n_samples, n_features)
   * @param y - Class labels of shape (n_samples,)
   * @returns this
   * @throws {DataValidationError} If y contains fewer than two classes
   * @throws {ShapeError} If X is not 2-D, y is not 1-D or their lengths differ
   */
  fit(X: Tensor, y: Tensor): this {
    this.fitted = false;
    validateFitInputs(X, y);
    const nFeatures = X.shape[1] ?? 0;
    const { classes, labelIndex } = encodeClasses(y, "OneVsOneClassifier");
    const nClasses = classes.length;
    const xv = toFloat64View(X);

    // Sample indices of every class, ascending.
    const members: number[][] = classes.map(() => []);
    for (let s = 0; s < labelIndex.length; s++) {
      (members[labelIndex[s] as number] as number[]).push(s);
    }

    const classifiers: Classifier[] = [];
    const pairs: [number, number][] = [];
    for (let a = 0; a < nClasses; a++) {
      for (let b = a + 1; b < nClasses; b++) {
        const rowsA = members[a] as number[];
        const rowsB = members[b] as number[];
        const m = rowsA.length + rowsB.length;
        // Merge the two ascending index lists so the original sample order is kept.
        const subX = new Float64Array(m * nFeatures);
        const subY = new Float64Array(m);
        let pa = 0;
        let pb = 0;
        for (let r = 0; r < m; r++) {
          const takeA =
            pb >= rowsB.length ||
            (pa < rowsA.length && (rowsA[pa] as number) < (rowsB[pb] as number));
          const orig = takeA ? (rowsA[pa++] as number) : (rowsB[pb++] as number);
          subY[r] = takeA ? 0 : 1;
          subX.set(xv.subarray(orig * nFeatures, (orig + 1) * nFeatures), r * nFeatures);
        }

        const clf = cloneEstimator(this.estimator, "OneVsOneClassifier");
        clf.fit(nbMatrixTensor(subX, m, nFeatures), tensor(subY));
        classifiers.push(clf);
        pairs.push([a, b]);
      }
    }

    this.classes_ = classes;
    this.classifiers_ = classifiers;
    this.classPairs_ = pairs;
    this.nFeaturesIn_ = nFeatures;
    this.fitted = true;
    return this;
  }

  private assertFitted(method: string): void {
    if (!this.fitted) {
      throw new NotFittedError(`OneVsOneClassifier must be fitted before ${method}`);
    }
  }

  /**
   * Vote counts and summed pairwise confidences per sample and class (row-major).
   * `confidenceSums` stays zero when the base estimator has no confidence output.
   */
  private pairwiseStatistics(X: Tensor): { votes: Float64Array; confidenceSums: Float64Array } {
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;
    const votes = new Float64Array(nSamples * nClasses);
    const confidenceSums = new Float64Array(nSamples * nClasses);

    for (let k = 0; k < this.classifiers_.length; k++) {
      const clf = this.classifiers_[k] as Classifier;
      const [a, b] = this.classPairs_[k] as [number, number];
      const pred = toFloat64View(clf.predict(X));
      const confidence = binaryConfidence(clf, X, nSamples);
      for (let i = 0; i < nSamples; i++) {
        const winner = (pred[i] as number) > 0.5 ? b : a;
        votes[i * nClasses + winner] = (votes[i * nClasses + winner] as number) + 1;
        if (confidence !== null) {
          const conf = confidence[i] as number;
          confidenceSums[i * nClasses + a] = (confidenceSums[i * nClasses + a] as number) - conf;
          confidenceSums[i * nClasses + b] = (confidenceSums[i * nClasses + b] as number) + conf;
        }
      }
    }
    return { votes, confidenceSums };
  }

  /** Votes plus the scikit-learn confidence tie-breaker, which is below 1/3 in magnitude. */
  private scores(X: Tensor): Float64Array {
    const { votes, confidenceSums } = this.pairwiseStatistics(X);
    for (let i = 0; i < votes.length; i++) {
      const s = confidenceSums[i] as number;
      votes[i] = (votes[i] as number) + s / (3 * (Math.abs(s) + 1));
    }
    return votes;
  }

  /**
   * Predict class labels by majority vote of the pairwise classifiers.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  predict(X: Tensor): Tensor {
    this.assertFitted("predict");
    validatePredictInputs(X, this.nFeaturesIn_, "OneVsOneClassifier");
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;
    const scores = this.scores(X);

    const labels = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let bestC = 0;
      let bestScore = -Infinity;
      for (let c = 0; c < nClasses; c++) {
        const s = scores[i * nClasses + c] as number;
        if (s > bestScore) {
          bestScore = s;
          bestC = c;
        }
      }
      labels[i] = this.classes_[bestC] as number;
    }
    return nbLabelTensor(labels, this.classes_);
  }

  /**
   * Vote counts plus a small confidence-based tie-breaker, one column per class.
   *
   * The score of class k is its number of pairwise wins plus
   * `s / (3 * (|s| + 1))`, where `s` is the sum of the signed pairwise confidences; the
   * second term stays strictly between -1/3 and 1/3, so it only orders classes with equal
   * votes. This is scikit-learn's `decision_function` for `OneVsOneClassifier`.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Scores of shape (n_samples, n_classes), columns follow `classes`
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  decisionFunction(X: Tensor): Tensor {
    this.assertFitted("decisionFunction");
    validatePredictInputs(X, this.nFeaturesIn_, "OneVsOneClassifier");
    return nbMatrixTensor(this.scores(X), X.shape[0] ?? 0, this.classes_.length);
  }

  /**
   * Estimate class probabilities using normalized vote counts.
   *
   * For each sample, the probability of each class is proportional to
   * the number of votes it received across all pairwise classifiers. These are vote
   * shares, not calibrated probabilities, and tied classes get equal shares even though
   * `predict` breaks such ties with the pairwise confidences.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X has the wrong rank or feature count
   */
  predictProba(X: Tensor): Tensor {
    this.assertFitted("predictProba");
    validatePredictInputs(X, this.nFeaturesIn_, "OneVsOneClassifier");
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;
    const { votes } = this.pairwiseStatistics(X);

    const proba = new Float64Array(nSamples * nClasses);
    for (let i = 0; i < nSamples; i++) {
      let total = 0;
      for (let c = 0; c < nClasses; c++) total += votes[i * nClasses + c] as number;
      for (let c = 0; c < nClasses; c++) {
        proba[i * nClasses + c] =
          total > 0 ? (votes[i * nClasses + c] as number) / total : 1 / nClasses;
      }
    }
    return nbMatrixTensor(proba, nSamples, nClasses);
  }

  /**
   * Mean accuracy on the given data and labels.
   *
   * @param X - Test samples
   * @param y - True labels of shape (n_samples,)
   * @returns Accuracy in [0, 1]
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If y is not 1-D or its length differs from the number of samples
   * @throws {DataValidationError} If y is empty or contains NaN/Inf
   */
  score(X: Tensor, y: Tensor): number {
    nbValidateScoreTarget(y);
    return nbAccuracy(this.predict(X), y);
  }

  /**
   * Sorted class labels seen during `fit`.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get classes(): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("OneVsOneClassifier must be fitted to access classes");
    }
    return nbLabelTensor(this.classes_, this.classes_);
  }

  /** Parameters of this estimator: the template `estimator`. */
  getParams(): Record<string, unknown> {
    return { estimator: this.estimator };
  }

  /**
   * Replace the template estimator. The model must be refitted afterwards.
   *
   * @throws {InvalidParameterError} If the key is unknown or `estimator` is not a valid estimator
   */
  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      if (key === "estimator") {
        this.estimator = validateEstimator(value, "OneVsOneClassifier");
      } else {
        throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }

  /** Create an unfitted copy with a cloned template estimator. */
  clone(): OneVsOneClassifier {
    return new OneVsOneClassifier({
      estimator: cloneEstimator(this.estimator, "OneVsOneClassifier"),
    });
  }
}
