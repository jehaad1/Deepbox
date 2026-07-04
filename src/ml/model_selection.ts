/**
 * Model selection utilities: cross-validation scoring, hyperparameter search.
 *
 * @module ml/model_selection
 * @see {@link https://deepbox.dev/docs/ml-model-selection | Deepbox Model Selection}
 */

import { InvalidParameterError, NotFittedError } from "../core";
import { type Tensor, tensor } from "../ndarray";
import { Generator } from "../random/Generator";
import { __random } from "../random/random";
import type { Classifier, Estimator, Regressor } from "./base";

function hasScore(e: Estimator): e is Estimator & { score(X: Tensor, y: Tensor): number } {
  return typeof (e as Classifier).score === "function";
}

/**
 * Assign each sample to a CV fold via a deterministic shuffle. Contiguous
 * folds (the previous behavior) fail catastrophically on class-sorted data —
 * a training fold could contain a single class, yielding 0% accuracy and
 * silently corrupting hyperparameter search. A fixed-seed shuffle keeps
 * results reproducible while breaking the sort-order pathology.
 */
function kfoldFoldOf(nSamples: number, cv: number, seed = 0): Int32Array {
  const perm = Array.from({ length: nSamples }, (_, i) => i);
  let s = seed >>> 0 || 1;
  const rng = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  for (let i = nSamples - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = perm[i]!;
    perm[i] = perm[j]!;
    perm[j] = tmp;
  }
  const foldSize = Math.floor(nSamples / cv);
  const foldOf = new Int32Array(nSamples);
  for (let pos = 0; pos < nSamples; pos++) {
    let fold = foldSize > 0 ? Math.floor(pos / foldSize) : 0;
    if (fold >= cv) fold = cv - 1;
    foldOf[perm[pos]!] = fold;
  }
  return foldOf;
}

/**
 * Evaluate an estimator using cross-validation.
 *
 * Splits data into `cv` folds, trains on (cv-1) folds, scores on the remaining fold,
 * and returns an array of scores.
 *
 * @param estimator - Estimator with `fit`, `predict`, and `score` methods
 * @param X - Features of shape (n_samples, n_features)
 * @param y - Targets of shape (n_samples,)
 * @param cv - Number of folds (default: 5)
 * @returns Array of scores, one per fold
 *
 * @example
 * ```ts
 * import { cross_val_score } from 'deepbox/ml';
 * import { LogisticRegression } from 'deepbox/ml';
 *
 * const scores = cross_val_score(new LogisticRegression(), X, y, 5);
 * console.log('Mean CV score:', scores.reduce((a,b) => a+b, 0) / scores.length);
 * ```
 */
export function cross_val_score(estimator: Estimator, X: Tensor, y: Tensor, cv = 5): number[] {
  if (!Number.isInteger(cv) || cv < 2) {
    throw new InvalidParameterError(`cv must be an integer >= 2; received ${cv}`, "cv", cv);
  }
  if (!hasScore(estimator)) {
    throw new InvalidParameterError("Estimator must implement score()", "estimator", estimator);
  }

  const nSamples = X.shape[0] ?? 0;
  const nFeatures = X.shape[1] ?? 0;

  if (nSamples < cv) {
    throw new InvalidParameterError(`Cannot have cv=${cv} with n_samples=${nSamples}`, "cv", cv);
  }

  // Create fold indices
  const foldOf = kfoldFoldOf(nSamples, cv);
  const scores: number[] = [];

  for (let fold = 0; fold < cv; fold++) {
    // Build train/test arrays
    const trainRows: number[] = [];
    const testRows: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      if (foldOf[i] === fold) {
        testRows.push(i);
      } else {
        trainRows.push(i);
      }
    }

    // Extract train data
    const XTrainData: number[] = [];
    const yTrainData: number[] = [];
    for (const i of trainRows) {
      for (let j = 0; j < nFeatures; j++) {
        XTrainData.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      yTrainData.push(Number(y.data[y.offset + i]));
    }

    // Extract test data
    const XTestData: number[] = [];
    const yTestData: number[] = [];
    for (const i of testRows) {
      for (let j = 0; j < nFeatures; j++) {
        XTestData.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      yTestData.push(Number(y.data[y.offset + i]));
    }

    const XTrain = tensor(XTrainData).reshape([trainRows.length, nFeatures]);
    const yTrain = tensor(yTrainData);
    const XTest = tensor(XTestData).reshape([testRows.length, nFeatures]);
    const yTest = tensor(yTestData);

    // Clone estimator: prefer clone() method if available, otherwise fall back to
    // constructor-based cloning via getParams.
    let cloned: Estimator;
    if (typeof estimator.clone === "function") {
      cloned = estimator.clone();
    } else {
      const params = estimator.getParams();
      const EstimatorClass = estimator.constructor as new (...args: unknown[]) => Estimator;
      try {
        cloned = new EstimatorClass(params);
      } catch {
        throw new InvalidParameterError(
          "Cannot clone estimator for cross-validation. " +
            "Implement clone() on your estimator or ensure the constructor accepts a params object.",
          "estimator",
          estimator
        );
      }
    }

    cloned.fit(XTrain, yTrain);
    const score = (cloned as Estimator & { score(X: Tensor, y: Tensor): number }).score(
      XTest,
      yTest
    );
    scores.push(score);
  }

  return scores;
}

/**
 * Result of cross_validate().
 */
export type CrossValidateResult = {
  /** Array of test scores per fold for each scoring metric */
  readonly testScores: Record<string, number[]>;
  /** Array of fit times per fold (ms) */
  readonly fitTime: number[];
  /** Array of score times per fold (ms) */
  readonly scoreTime: number[];
};

/**
 * Evaluate an estimator using cross-validation with multiple metrics.
 *
 * Unlike `cross_val_score` which returns only scores for a single metric,
 * `cross_validate` supports multiple scoring functions and also reports
 * fit/score times.
 *
 * @param estimator - Estimator with `fit` and `score` methods
 * @param X - Features of shape (n_samples, n_features)
 * @param y - Targets of shape (n_samples,)
 * @param options - Configuration options
 * @returns Object with testScores (per metric), fitTime, and scoreTime arrays
 *
 * @example
 * ```ts
 * import { cross_validate } from 'deepbox/ml';
 * import { LogisticRegression } from 'deepbox/ml';
 *
 * // Single metric (uses estimator.score)
 * const result = cross_validate(new LogisticRegression(), X, y, { cv: 5 });
 *
 * // Multiple metrics
 * const result = cross_validate(new LogisticRegression(), X, y, {
 *   cv: 5,
 *   scoring: {
 *     accuracy: (est, X, y) => est.score(X, y),
 *     custom: (est, X, y) => { ... },
 *   },
 * });
 * console.log(result.testScores['accuracy']);
 * ```
 */
export function cross_validate(
  estimator: Estimator,
  X: Tensor,
  y: Tensor,
  options: {
    cv?: number;
    scoring?:
      | Record<string, (est: Estimator, X: Tensor, y: Tensor) => number>
      | ((est: Estimator, X: Tensor, y: Tensor) => number);
  } = {}
): CrossValidateResult {
  const cv = options.cv ?? 5;
  if (!Number.isInteger(cv) || cv < 2) {
    throw new InvalidParameterError(`cv must be an integer >= 2; received ${cv}`, "cv", cv);
  }

  const nSamples = X.shape[0] ?? 0;
  const nFeatures = X.shape[1] ?? 0;

  if (nSamples < cv) {
    throw new InvalidParameterError(`Cannot have cv=${cv} with n_samples=${nSamples}`, "cv", cv);
  }

  // Build scoring functions
  let scorers: Record<string, (est: Estimator, X: Tensor, y: Tensor) => number>;
  if (options.scoring === undefined) {
    if (!hasScore(estimator)) {
      throw new InvalidParameterError(
        "Estimator must implement score() or provide scoring parameter",
        "estimator",
        estimator
      );
    }
    scorers = {
      score: (est, Xv, yv) =>
        (est as Estimator & { score(X: Tensor, y: Tensor): number }).score(Xv, yv),
    };
  } else if (typeof options.scoring === "function") {
    scorers = { score: options.scoring };
  } else {
    scorers = options.scoring;
  }

  const metricNames = Object.keys(scorers);
  const testScores: Record<string, number[]> = {};
  for (const name of metricNames) {
    testScores[name] = [];
  }
  const fitTime: number[] = [];
  const scoreTime: number[] = [];

  const foldOf = kfoldFoldOf(nSamples, cv);

  for (let fold = 0; fold < cv; fold++) {
    const trainRows: number[] = [];
    const testRows: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      if (foldOf[i] === fold) testRows.push(i);
      else trainRows.push(i);
    }

    const train = extractFoldData(X, y, trainRows, nFeatures);
    const test = extractFoldData(X, y, testRows, nFeatures);

    // Clone estimator: prefer clone() if available (e.g. Pipeline, whose flat
    // getParams() cannot be passed to its constructor), otherwise fall back to
    // constructor-based cloning via getParams. Mirrors cross_val_score.
    let cloned: Estimator;
    if (typeof estimator.clone === "function") {
      cloned = estimator.clone();
    } else {
      const EstimatorClass = estimator.constructor as new (p: Record<string, unknown>) => Estimator;
      try {
        cloned = new EstimatorClass(estimator.getParams());
      } catch {
        throw new InvalidParameterError(
          "Cannot clone estimator for cross-validation. " +
            "Implement clone() on your estimator or ensure the constructor accepts a params object for CV folds to avoid leaking fitted state.",
          "estimator",
          estimator
        );
      }
    }

    const fitStart = Date.now();
    cloned.fit(train.X, train.y);
    fitTime.push(Date.now() - fitStart);

    const scoreStart = Date.now();
    for (const name of metricNames) {
      const scorer = scorers[name]!;
      testScores[name]!.push(scorer(cloned, test.X, test.y));
    }
    scoreTime.push(Date.now() - scoreStart);
  }

  return { testScores, fitTime, scoreTime };
}

// ---- Helpers for grid/randomized search ----

function extractFoldData(
  X: Tensor,
  y: Tensor,
  rows: number[],
  nFeatures: number
): { X: Tensor; y: Tensor } {
  const xData: number[] = [];
  const yData: number[] = [];
  for (const i of rows) {
    for (let j = 0; j < nFeatures; j++) {
      xData.push(Number(X.data[X.offset + i * nFeatures + j]));
    }
    yData.push(Number(y.data[y.offset + i]));
  }
  return {
    X: tensor(xData).reshape([rows.length, nFeatures]),
    y: tensor(yData),
  };
}

function cartesianProduct(paramGrid: Record<string, unknown[]>): Record<string, unknown>[] {
  const keys = Object.keys(paramGrid);
  if (keys.length === 0) return [{}];

  const combos: Record<string, unknown>[] = [];
  const values = keys.map((k) => paramGrid[k] ?? []);

  function recurse(idx: number, current: Record<string, unknown>): void {
    if (idx === keys.length) {
      combos.push({ ...current });
      return;
    }
    const key = keys[idx]!;
    const vals = values[idx]!;
    for (const v of vals) {
      current[key] = v;
      recurse(idx + 1, current);
    }
  }

  recurse(0, {});
  return combos;
}

/** Result of a single parameter combination evaluation */
export type GridSearchResult = {
  readonly params: Record<string, unknown>;
  readonly meanScore: number;
  readonly scores: number[];
};

/**
 * Exhaustive search over specified parameter values for an estimator.
 *
 * @example
 * ```ts
 * import { GridSearchCV } from 'deepbox/ml';
 * import { Ridge } from 'deepbox/ml';
 *
 * const gs = new GridSearchCV(new Ridge(), { alpha: [0.1, 1, 10] }, { cv: 5 });
 * gs.fit(X, y);
 * console.log(gs.bestParams);
 * console.log(gs.bestScore);
 * ```
 */
export class GridSearchCV {
  private readonly paramGrid: Record<string, unknown[]>;
  private readonly cv: number;
  private fitted = false;

  /** Best parameters found */
  bestParams: Record<string, unknown> = {};
  /** Best mean CV score */
  bestScore = -Infinity;
  /** Best fitted estimator */
  bestEstimator: Estimator | undefined;
  /** All CV results */
  cvResults: GridSearchResult[] = [];

  constructor(
    private readonly estimator: Estimator,
    paramGrid: Record<string, unknown[]>,
    options: { cv?: number } = {}
  ) {
    this.paramGrid = paramGrid;
    this.cv = options.cv ?? 5;
  }

  fit(X: Tensor, y: Tensor): this {
    if (!hasScore(this.estimator)) {
      throw new InvalidParameterError(
        "Estimator must implement score()",
        "estimator",
        this.estimator
      );
    }

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const cv = this.cv;

    if (nSamples < cv) {
      throw new InvalidParameterError(`Cannot have cv=${cv} with n_samples=${nSamples}`, "cv", cv);
    }

    const combos = cartesianProduct(this.paramGrid);
    const foldOf = kfoldFoldOf(nSamples, cv);

    this.cvResults = [];
    this.bestScore = -Infinity;

    for (const params of combos) {
      const foldScores: number[] = [];

      for (let fold = 0; fold < cv; fold++) {
        const trainRows: number[] = [];
        const testRows: number[] = [];
        for (let i = 0; i < nSamples; i++) {
          if (foldOf[i] === fold) testRows.push(i);
          else trainRows.push(i);
        }

        const train = extractFoldData(X, y, trainRows, nFeatures);
        const test = extractFoldData(X, y, testRows, nFeatures);

        const EstimatorClass = this.estimator.constructor as new (
          p: Record<string, unknown>
        ) => Estimator;
        let cloned: Estimator;
        try {
          cloned = new EstimatorClass({
            ...this.estimator.getParams(),
            ...params,
          });
        } catch {
          throw new InvalidParameterError(
            "Cannot clone estimator for GridSearchCV cross-validation. " +
              "The estimator constructor must accept a params object for CV folds to avoid leaking fitted state.",
            "estimator",
            this.estimator
          );
        }

        cloned.fit(train.X, train.y);
        const score = (cloned as Estimator & { score(X: Tensor, y: Tensor): number }).score(
          test.X,
          test.y
        );
        foldScores.push(score);
      }

      const meanScore = foldScores.reduce((a, b) => a + b, 0) / foldScores.length;
      this.cvResults.push({ params, meanScore, scores: foldScores });

      if (meanScore > this.bestScore) {
        this.bestScore = meanScore;
        this.bestParams = params;
      }
    }

    // Refit best estimator on full data
    const EstimatorClass = this.estimator.constructor as new (
      p: Record<string, unknown>
    ) => Estimator;
    try {
      this.bestEstimator = new EstimatorClass({
        ...this.estimator.getParams(),
        ...this.bestParams,
      });
    } catch {
      throw new InvalidParameterError(
        "Cannot clone estimator for GridSearchCV final refit. " +
          "The estimator constructor must accept a params object.",
        "estimator",
        this.estimator
      );
    }
    this.bestEstimator.fit(X, y);
    this.fitted = true;

    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.bestEstimator) {
      throw new NotFittedError("GridSearchCV must be fitted before predict");
    }
    return (this.bestEstimator as Classifier | Regressor).predict(X);
  }

  score(X: Tensor, y: Tensor): number {
    if (!this.fitted || !this.bestEstimator) {
      throw new NotFittedError("GridSearchCV must be fitted before score");
    }
    return (this.bestEstimator as Estimator & { score(X: Tensor, y: Tensor): number }).score(X, y);
  }

  getParams(): Record<string, unknown> {
    return {
      estimator: this.estimator,
      paramGrid: this.paramGrid,
      cv: this.cv,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}

/**
 * Randomized search over parameter distributions.
 *
 * Samples `nIter` random parameter combinations instead of exhaustive search.
 *
 * @example
 * ```ts
 * import { RandomizedSearchCV } from 'deepbox/ml';
 * import { Ridge } from 'deepbox/ml';
 *
 * const rs = new RandomizedSearchCV(
 *   new Ridge(),
 *   { alpha: [0.01, 0.1, 1, 10, 100] },
 *   { nIter: 3, cv: 5 }
 * );
 * rs.fit(X, y);
 * ```
 */
export class RandomizedSearchCV {
  private readonly paramDistributions: Record<string, unknown[]>;
  private readonly cv: number;
  private readonly nIter: number;
  private readonly randomState: number | undefined;
  private fitted = false;

  bestParams: Record<string, unknown> = {};
  bestScore = -Infinity;
  bestEstimator: Estimator | undefined;
  cvResults: GridSearchResult[] = [];

  constructor(
    private readonly estimator: Estimator,
    paramDistributions: Record<string, unknown[]>,
    options: { nIter?: number; cv?: number; randomState?: number } = {}
  ) {
    this.paramDistributions = paramDistributions;
    this.nIter = options.nIter ?? 10;
    this.cv = options.cv ?? 5;
    this.randomState = options.randomState;
  }

  fit(X: Tensor, y: Tensor): this {
    // Sample nIter random parameter combinations
    const allCombos = cartesianProduct(this.paramDistributions);

    // Shuffle and take nIter using PCG-based RNG for statistical quality
    const rng = this.randomState !== undefined ? new Generator(this.randomState!) : null;

    for (let i = allCombos.length - 1; i > 0; i--) {
      const randVal = rng ? rng.random() : __random();
      const j = Math.floor(randVal * (i + 1));
      [allCombos[i], allCombos[j]] = [allCombos[j]!, allCombos[i]!];
    }

    const sampledCombos = allCombos.slice(0, Math.min(this.nIter, allCombos.length));

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const cv = this.cv;
    const foldOf = kfoldFoldOf(nSamples, cv);

    this.cvResults = [];
    this.bestScore = -Infinity;

    for (const params of sampledCombos) {
      const foldScores: number[] = [];

      for (let fold = 0; fold < cv; fold++) {
        const trainRows: number[] = [];
        const testRows: number[] = [];
        for (let i = 0; i < nSamples; i++) {
          if (foldOf[i] === fold) testRows.push(i);
          else trainRows.push(i);
        }

        const train = extractFoldData(X, y, trainRows, nFeatures);
        const test = extractFoldData(X, y, testRows, nFeatures);

        const EstimatorClass = this.estimator.constructor as new (
          p: Record<string, unknown>
        ) => Estimator;
        let cloned: Estimator;
        try {
          cloned = new EstimatorClass({
            ...this.estimator.getParams(),
            ...params,
          });
        } catch {
          throw new InvalidParameterError(
            "Cannot clone estimator for RandomizedSearchCV cross-validation. " +
              "The estimator constructor must accept a params object for CV folds to avoid leaking fitted state.",
            "estimator",
            this.estimator
          );
        }

        cloned.fit(train.X, train.y);
        const score = (cloned as Estimator & { score(X: Tensor, y: Tensor): number }).score(
          test.X,
          test.y
        );
        foldScores.push(score);
      }

      const meanScore = foldScores.reduce((a, b) => a + b, 0) / foldScores.length;
      this.cvResults.push({ params, meanScore, scores: foldScores });

      if (meanScore > this.bestScore) {
        this.bestScore = meanScore;
        this.bestParams = params;
      }
    }

    // Refit best
    const EstimatorClass = this.estimator.constructor as new (
      p: Record<string, unknown>
    ) => Estimator;
    try {
      this.bestEstimator = new EstimatorClass({
        ...this.estimator.getParams(),
        ...this.bestParams,
      });
    } catch {
      throw new InvalidParameterError(
        "Cannot clone estimator for RandomizedSearchCV final refit. " +
          "The estimator constructor must accept a params object.",
        "estimator",
        this.estimator
      );
    }
    this.bestEstimator.fit(X, y);
    this.fitted = true;

    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.bestEstimator) {
      throw new NotFittedError("RandomizedSearchCV must be fitted before predict");
    }
    return (this.bestEstimator as Classifier | Regressor).predict(X);
  }

  score(X: Tensor, y: Tensor): number {
    if (!this.fitted || !this.bestEstimator) {
      throw new NotFittedError("RandomizedSearchCV must be fitted before score");
    }
    return (this.bestEstimator as Estimator & { score(X: Tensor, y: Tensor): number }).score(X, y);
  }

  getParams(): Record<string, unknown> {
    return {
      estimator: this.estimator,
      paramDistributions: this.paramDistributions,
      nIter: this.nIter,
      cv: this.cv,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}
