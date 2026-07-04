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

import { NotFittedError } from "../core";
import { type Tensor, tensor } from "../ndarray";
import { assertContiguous, validateFitInputs, validatePredictInputs } from "./_validation";
import type { Classifier } from "./base";

/**
 * One-vs-Rest (OvR) multiclass strategy.
 *
 * Fits one binary classifier per class. For class k, the positive
 * examples are from class k and negative examples are all other classes.
 *
 * The wrapped estimator must implement `fit(X, y)`, `predict(X)`,
 * and `predictProba(X)` (or at minimum `predict`).
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
  private readonly estimatorFactory: () => Classifier;
  private classifiers_: Classifier[] = [];
  private classes_: number[] = [];
  private nFeaturesIn_ = 0;
  private fitted = false;

  constructor(options: { readonly estimator: Classifier }) {
    // We need to clone the estimator for each binary subproblem.
    // Store a factory that creates fresh instances with the same params.
    const baseParams = options.estimator.getParams();
    const BaseClass = options.estimator.constructor as new (
      params: Record<string, unknown>
    ) => Classifier;
    this.estimatorFactory = () => new BaseClass(baseParams);
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    // Extract classes
    const classSet = new Set<number>();
    for (let i = 0; i < nSamples; i++) {
      classSet.add(Number(y.data[y.offset + i]));
    }
    this.classes_ = [...classSet].sort((a, b) => a - b);

    // Fit one classifier per class
    this.classifiers_ = [];
    for (const cls of this.classes_) {
      const binaryY = new Float64Array(nSamples);
      for (let i = 0; i < nSamples; i++) {
        binaryY[i] = Number(y.data[y.offset + i]) === cls ? 1 : 0;
      }
      const clf = this.estimatorFactory();
      clf.fit(X, tensor(Array.from(binaryY)));
      this.classifiers_.push(clf);
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("OneVsRestClassifier must be fitted before predict");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "OneVsRestClassifier");
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;

    // Get confidence scores from each classifier
    const scores = new Float64Array(nSamples * nClasses);
    for (let c = 0; c < nClasses; c++) {
      const clf = this.classifiers_[c]!;
      try {
        const proba = clf.predictProba(X);
        // Use probability of class 1 (positive)
        const nCols = proba.shape[1] ?? 1;
        for (let i = 0; i < nSamples; i++) {
          scores[i * nClasses + c] =
            nCols >= 2
              ? Number(proba.data[proba.offset + i * nCols + 1])
              : Number(proba.data[proba.offset + i * nCols]);
        }
      } catch {
        // Fall back to predict if predictProba is not available
        const pred = clf.predict(X);
        for (let i = 0; i < nSamples; i++) {
          scores[i * nClasses + c] = Number(pred.data[pred.offset + i]);
        }
      }
    }

    // Assign class with highest score
    const labels = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let bestC = 0;
      let bestScore = -Infinity;
      for (let c = 0; c < nClasses; c++) {
        const s = scores[i * nClasses + c] ?? 0;
        if (s > bestScore) {
          bestScore = s;
          bestC = c;
        }
      }
      labels[i] = this.classes_[bestC] ?? 0;
    }

    return tensor(Array.from(labels));
  }

  predictProba(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("OneVsRestClassifier must be fitted before predictProba");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "OneVsRestClassifier");
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;

    const result = new Float64Array(nSamples * nClasses);
    for (let c = 0; c < nClasses; c++) {
      const clf = this.classifiers_[c]!;
      const proba = clf.predictProba(X);
      const nCols = proba.shape[1] ?? 1;
      for (let i = 0; i < nSamples; i++) {
        result[i * nClasses + c] =
          nCols >= 2
            ? Number(proba.data[proba.offset + i * nCols + 1])
            : Number(proba.data[proba.offset + i * nCols]);
      }
    }

    // Normalize rows to sum to 1
    for (let i = 0; i < nSamples; i++) {
      let sum = 0;
      for (let c = 0; c < nClasses; c++) sum += result[i * nClasses + c] ?? 0;
      if (sum > 0) {
        for (let c = 0; c < nClasses; c++) {
          result[i * nClasses + c] = (result[i * nClasses + c] ?? 0) / sum;
        }
      }
    }

    return tensor(Array.from(result)).reshape([nSamples, nClasses]);
  }

  score(X: Tensor, y: Tensor): number {
    assertContiguous(y, "y");
    const pred = this.predict(X);
    const n = y.size;
    let correct = 0;
    for (let i = 0; i < n; i++) {
      if (Number(pred.data[pred.offset + i]) === Number(y.data[y.offset + i])) {
        correct++;
      }
    }
    return correct / n;
  }

  get classes(): Tensor {
    if (!this.fitted)
      throw new NotFittedError("OneVsRestClassifier must be fitted to access classes");
    return tensor(this.classes_);
  }

  getParams(): Record<string, unknown> {
    return {};
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}

/**
 * One-vs-One (OvO) multiclass strategy.
 *
 * Fits one classifier for each pair of classes. Uses majority voting
 * to determine the final prediction.
 *
 * @example
 * ```ts
 * import { OneVsOneClassifier, SVC } from 'deepbox/ml';
 * const ovo = new OneVsOneClassifier({ estimator: new SVC() });
 * ovo.fit(X_train, y_train);
 * const pred = ovo.predict(X_test);
 * ```
 */
export class OneVsOneClassifier implements Classifier {
  private readonly estimatorFactory: () => Classifier;
  private classifiers_: Classifier[] = [];
  private classPairs_: [number, number][] = [];
  private classes_: number[] = [];
  private nFeaturesIn_ = 0;
  private fitted = false;

  constructor(options: { readonly estimator: Classifier }) {
    const baseParams = options.estimator.getParams();
    const BaseClass = options.estimator.constructor as new (
      params: Record<string, unknown>
    ) => Classifier;
    this.estimatorFactory = () => new BaseClass(baseParams);
  }

  fit(X: Tensor, y: Tensor): this {
    validateFitInputs(X, y);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    const classSet = new Set<number>();
    for (let i = 0; i < nSamples; i++) {
      classSet.add(Number(y.data[y.offset + i]));
    }
    this.classes_ = [...classSet].sort((a, b) => a - b);
    const nClasses = this.classes_.length;

    this.classifiers_ = [];
    this.classPairs_ = [];

    // Fit one classifier for each pair of classes
    for (let i = 0; i < nClasses; i++) {
      for (let j = i + 1; j < nClasses; j++) {
        const classA = this.classes_[i]!;
        const classB = this.classes_[j]!;
        this.classPairs_.push([classA, classB]);

        // Extract samples belonging to classA or classB
        const indices: number[] = [];
        for (let s = 0; s < nSamples; s++) {
          const val = Number(y.data[y.offset + s]);
          if (val === classA || val === classB) indices.push(s);
        }

        const subX = new Float64Array(indices.length * nFeatures);
        const subY = new Float64Array(indices.length);
        for (let si = 0; si < indices.length; si++) {
          const origIdx = indices[si]!;
          for (let f = 0; f < nFeatures; f++) {
            subX[si * nFeatures + f] = Number(X.data[X.offset + origIdx * nFeatures + f]);
          }
          subY[si] = Number(y.data[y.offset + origIdx]);
        }

        const clf = this.estimatorFactory();
        clf.fit(
          tensor(Array.from(subX)).reshape([indices.length, nFeatures]),
          tensor(Array.from(subY))
        );
        this.classifiers_.push(clf);
      }
    }

    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("OneVsOneClassifier must be fitted before predict");
    }
    validatePredictInputs(X, this.nFeaturesIn_, "OneVsOneClassifier");
    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;

    // Voting
    const votes = new Float64Array(nSamples * nClasses);
    for (let ci = 0; ci < this.classifiers_.length; ci++) {
      const clf = this.classifiers_[ci]!;
      const [classA, classB] = this.classPairs_[ci]!;
      const pred = clf.predict(X);

      const idxA = this.classes_.indexOf(classA);
      const idxB = this.classes_.indexOf(classB);

      for (let i = 0; i < nSamples; i++) {
        const predVal = Number(pred.data[pred.offset + i]);
        if (predVal === classA) {
          votes[i * nClasses + idxA] = (votes[i * nClasses + idxA] ?? 0) + 1;
        } else {
          votes[i * nClasses + idxB] = (votes[i * nClasses + idxB] ?? 0) + 1;
        }
      }
    }

    const labels = new Float64Array(nSamples);
    for (let i = 0; i < nSamples; i++) {
      let bestC = 0;
      let bestVote = -1;
      for (let c = 0; c < nClasses; c++) {
        const v = votes[i * nClasses + c] ?? 0;
        if (v > bestVote) {
          bestVote = v;
          bestC = c;
        }
      }
      labels[i] = this.classes_[bestC] ?? 0;
    }

    return tensor(Array.from(labels));
  }

  /**
   * Estimate class probabilities using normalized vote counts.
   *
   * For each sample, the probability of each class is proportional to
   * the number of votes it received across all pairwise classifiers.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Probabilities of shape (n_samples, n_classes)
   * @throws {NotFittedError} If the model has not been fitted
   */
  predictProba(X: Tensor): Tensor {
    if (!this.fitted || !this.classifiers_ || !this.classPairs_ || this.classes_.length === 0) {
      throw new NotFittedError("OneVsOneClassifier must be fitted before predictProba");
    }

    const nSamples = X.shape[0] ?? 0;
    const nClasses = this.classes_.length;

    // Accumulate votes from all pairwise classifiers
    const votes = new Float64Array(nSamples * nClasses);
    for (let ci = 0; ci < this.classifiers_.length; ci++) {
      const clf = this.classifiers_[ci]!;
      const [classA, classB] = this.classPairs_[ci]!;
      const pred = clf.predict(X);

      const idxA = this.classes_.indexOf(classA);
      const idxB = this.classes_.indexOf(classB);

      for (let i = 0; i < nSamples; i++) {
        const predVal = Number(pred.data[pred.offset + i]);
        if (predVal === classA) {
          votes[i * nClasses + idxA] = (votes[i * nClasses + idxA] ?? 0) + 1;
        } else {
          votes[i * nClasses + idxB] = (votes[i * nClasses + idxB] ?? 0) + 1;
        }
      }
    }

    // Normalize votes to probabilities per sample
    const proba = new Float64Array(nSamples * nClasses);
    for (let i = 0; i < nSamples; i++) {
      let total = 0;
      for (let c = 0; c < nClasses; c++) {
        total += votes[i * nClasses + c] ?? 0;
      }
      for (let c = 0; c < nClasses; c++) {
        proba[i * nClasses + c] = total > 0 ? (votes[i * nClasses + c] ?? 0) / total : 1 / nClasses;
      }
    }

    return tensor(Array.from(proba)).reshape([nSamples, nClasses]);
  }

  score(X: Tensor, y: Tensor): number {
    assertContiguous(y, "y");
    const pred = this.predict(X);
    const n = y.size;
    let correct = 0;
    for (let i = 0; i < n; i++) {
      if (Number(pred.data[pred.offset + i]) === Number(y.data[y.offset + i])) {
        correct++;
      }
    }
    return correct / n;
  }

  get classes(): Tensor {
    if (!this.fitted)
      throw new NotFittedError("OneVsOneClassifier must be fitted to access classes");
    return tensor(this.classes_);
  }

  getParams(): Record<string, unknown> {
    return {};
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}
