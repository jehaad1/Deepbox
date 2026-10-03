/**
 * Model selection utilities: cross-validation scoring, hyperparameter search.
 *
 * @module ml/model_selection
 * @see {@link https://deepbox.dev/docs/ml-model-selection | Deepbox Model Selection}
 */

import {
  DataValidationError,
  DTypeError,
  InvalidParameterError,
  NotFittedError,
  ShapeError,
} from "../core";
import { type Tensor, tensor } from "../ndarray";
import { __random, __randomBelow, __SeededRandom, __seedToUint64 } from "../random/random";
import { cloneEstimator } from "./_internal";
import { type Classifier, type Estimator, getEstimatorTags, type Regressor } from "./base";

/** Scoring callback: higher is better. */
type Scorer = (est: Estimator, X: Tensor, y: Tensor) => number;

type Scorable = Estimator & { score(X: Tensor, y: Tensor): number };

function hasScore(e: Estimator): e is Scorable {
  return typeof (e as Classifier).score === "function";
}

function defaultScorer(est: Estimator, X: Tensor, y: Tensor): number {
  return (est as Scorable).score(X, y);
}

/**
 * Build a uniform [0, 1) generator. A seed gives a private deterministic stream;
 * without one the global Deepbox generator is used, so `setSeed` still applies.
 */
function createRng(seed: number | undefined): () => number {
  if (seed === undefined) return __random;
  const gen = new __SeededRandom(__seedToUint64(seed));
  return () => gen.next();
}

function validateCv(cv: unknown): asserts cv is number {
  if (typeof cv !== "number" || !Number.isInteger(cv) || cv < 2) {
    throw new InvalidParameterError(`cv must be an integer >= 2; received ${String(cv)}`, "cv", cv);
  }
}

/**
 * Check shapes of X and y and return the sample count.
 *
 * @throws {ShapeError} If X is not 2-D, y is not 1-D or 2-D, or their row counts differ
 * @throws {DataValidationError} If X has no rows or no columns
 */
function validateXY(X: Tensor, y: Tensor, cv: number): number {
  if (X.ndim !== 2) {
    throw new ShapeError(`X must be 2-dimensional; got ndim=${X.ndim}`);
  }
  if (y.ndim !== 1 && y.ndim !== 2) {
    throw new ShapeError(`y must be 1- or 2-dimensional; got ndim=${y.ndim}`);
  }
  const nSamples = X.shape[0] ?? 0;
  if (nSamples !== (y.shape[0] ?? 0)) {
    throw new ShapeError(
      `X and y must have the same number of samples; got X.shape[0]=${nSamples}, y.shape[0]=${y.shape[0]}`
    );
  }
  if (nSamples === 0 || (X.shape[1] ?? 0) === 0) {
    throw new DataValidationError("X must have at least one sample and one feature");
  }
  if (nSamples < cv) {
    throw new InvalidParameterError(`Cannot have cv=${cv} with n_samples=${nSamples}`, "cv", cv);
  }
  return nSamples;
}

/**
 * Select rows of a 1-D or 2-D tensor, honoring its strides. float32, float64 and
 * int32 inputs keep their dtype; every other numeric dtype becomes float64.
 * With `squeeze`, a single-column result is returned as a 1-D tensor.
 */
function takeRows(t: Tensor, rows: ArrayLike<number>, squeeze = false): Tensor {
  if (t.dtype === "string" || t.dtype === "complex64" || t.dtype === "complex128") {
    throw new DTypeError(`Cross-validation needs real numeric data; got dtype "${t.dtype}"`);
  }
  const data = t.data as ArrayLike<number | bigint>;
  const nCols = t.ndim === 1 ? 1 : (t.shape[1] ?? 0);
  const s0 = t.strides[0] ?? 1;
  const s1 = t.ndim === 2 ? (t.strides[1] ?? 1) : 0;
  const total = rows.length * nCols;
  const out =
    t.data instanceof Float32Array
      ? new Float32Array(total)
      : t.data instanceof Int32Array
        ? new Int32Array(total)
        : new Float64Array(total);
  for (let r = 0; r < rows.length; r++) {
    const base = t.offset + (rows[r] as number) * s0;
    for (let c = 0; c < nCols; c++) {
      out[r * nCols + c] = Number(data[base + c * s1]);
    }
  }
  const flat = tensor(out);
  return t.ndim === 1 || (squeeze && nCols === 1) ? flat : flat.reshape([rows.length, nCols]);
}

/**
 * Assign every sample to one of `cv` folds with a deterministic (fixed-seed)
 * shuffle. Contiguous folds would fail on class-sorted data: a training fold
 * could contain a single class and silently corrupt model selection.
 *
 * With `stratify`, the samples of each class are spread evenly over the folds,
 * so every fold keeps (as far as possible) the class proportions of `y`.
 * Fold sizes differ by at most one.
 */
function assignFolds(y: Tensor, nSamples: number, cv: number, stratify: boolean): Int32Array {
  const rng = createRng(0);
  const perm = Array.from({ length: nSamples }, (_, i) => i);
  for (let i = nSamples - 1; i > 0; i--) {
    const j = Math.min(__randomBelow(rng, i + 1), i);
    const tmp = perm[i] as number;
    perm[i] = perm[j] as number;
    perm[j] = tmp;
  }

  let order = perm;
  if (stratify && (y.ndim === 1 || y.shape[1] === 1)) {
    const label = (i: number): number => Number(y.data[y.offset + i * (y.strides[0] ?? 1)]);
    const classOf = new Map<number, number>();
    for (let i = 0; i < nSamples; i++) {
      const v = label(i);
      if (!classOf.has(v)) classOf.set(v, classOf.size);
    }
    // Continuous targets have (nearly) as many distinct values as samples.
    if (classOf.size >= 2 && classOf.size <= nSamples / 2) {
      // Stable sort keeps the shuffled order inside each class.
      order = [...perm].sort(
        (a, b) => (classOf.get(label(a)) as number) - (classOf.get(label(b)) as number)
      );
    }
  }

  const foldOf = new Int32Array(nSamples);
  if (order === perm) {
    const base = Math.floor(nSamples / cv);
    const extra = nSamples % cv;
    let pos = 0;
    for (let fold = 0; fold < cv; fold++) {
      const size = base + (fold < extra ? 1 : 0);
      for (let k = 0; k < size; k++) foldOf[perm[pos++] as number] = fold;
    }
  } else {
    for (let pos = 0; pos < nSamples; pos++) foldOf[order[pos] as number] = pos % cv;
  }
  return foldOf;
}

type FoldData = { XTrain: Tensor; yTrain: Tensor; XTest: Tensor; yTest: Tensor };

function buildFolds(X: Tensor, y: Tensor, foldOf: Int32Array, cv: number): FoldData[] {
  // Estimators expect a 1-D target, so an (n_samples, 1) column is flattened.
  const squeezeY = y.ndim === 2 && y.shape[1] === 1;
  const folds: FoldData[] = [];
  for (let fold = 0; fold < cv; fold++) {
    const trainRows: number[] = [];
    const testRows: number[] = [];
    for (let i = 0; i < foldOf.length; i++) {
      if (foldOf[i] === fold) testRows.push(i);
      else trainRows.push(i);
    }
    folds.push({
      XTrain: takeRows(X, trainRows),
      yTrain: takeRows(y, trainRows, squeezeY),
      XTest: takeRows(X, testRows),
      yTest: takeRows(y, testRows, squeezeY),
    });
  }
  return folds;
}

function isClassifier(estimator: Estimator): boolean {
  return getEstimatorTags(estimator).estimatorType === "classifier";
}

type CvOutcome = {
  scores: Record<string, number[]>;
  fitTime: number[];
  scoreTime: number[];
};

function runFolds(
  estimator: Estimator,
  overrides: Record<string, unknown>,
  folds: readonly FoldData[],
  scorers: Record<string, Scorer>,
  context: string
): CvOutcome {
  const names = Object.keys(scorers);
  const scores: Record<string, number[]> = {};
  for (const name of names) scores[name] = [];
  const fitTime: number[] = [];
  const scoreTime: number[] = [];

  for (const fold of folds) {
    const cloned = cloneEstimator(estimator, context, overrides);
    const fitStart = performance.now();
    cloned.fit(fold.XTrain, fold.yTrain);
    fitTime.push(performance.now() - fitStart);

    const scoreStart = performance.now();
    for (const name of names) {
      (scores[name] as number[]).push((scorers[name] as Scorer)(cloned, fold.XTest, fold.yTest));
    }
    scoreTime.push(performance.now() - scoreStart);
  }
  return { scores, fitTime, scoreTime };
}

/**
 * Evaluate an estimator using cross-validation.
 *
 * Splits data into `cv` folds, trains on (cv-1) folds, scores on the remaining fold,
 * and returns an array of scores.
 *
 * The folds come from a fixed-seed shuffle, so results are reproducible. For
 * classifiers (estimators exposing `classes` or `predictProba`) the folds are
 * stratified by the labels in `y`. Fold sizes differ by at most one sample.
 *
 * @param estimator - Estimator with `fit`, `predict`, and `score` methods
 * @param X - Features of shape (n_samples, n_features)
 * @param y - Targets of shape (n_samples,)
 * @param cv - Number of folds (default: 5)
 * @param scoring - Optional scoring function `(estimator, XTest, yTest) => number` used instead of `estimator.score`
 * @returns Array of scores, one per fold
 * @throws {InvalidParameterError} If `cv < 2`, `cv > n_samples`, or the estimator has no `score` method and no `scoring` function is given
 * @throws {ShapeError} If X is not 2-D or X and y disagree on the number of samples
 *
 * @example
 * ```ts
 * import { crossValScore } from 'deepbox/ml';
 * import { LogisticRegression } from 'deepbox/ml';
 *
 * const scores = crossValScore(new LogisticRegression(), X, y, 5);
 * console.log('Mean CV score:', scores.reduce((a,b) => a+b, 0) / scores.length);
 * ```
 *
 * @deprecated Prefer {@link crossValScore}.
 */
export function cross_val_score(
  estimator: Estimator,
  X: Tensor,
  y: Tensor,
  cv = 5,
  scoring?: Scorer
): number[] {
  validateCv(cv);
  if (scoring === undefined && !hasScore(estimator)) {
    throw new InvalidParameterError("Estimator must implement score()", "estimator", estimator);
  }
  if (scoring !== undefined && typeof scoring !== "function") {
    throw new InvalidParameterError("scoring must be a function", "scoring", scoring);
  }
  const nSamples = validateXY(X, y, cv);
  const foldOf = assignFolds(y, nSamples, cv, isClassifier(estimator));
  const folds = buildFolds(X, y, foldOf, cv);
  const { scores } = runFolds(
    estimator,
    {},
    folds,
    { score: scoring ?? defaultScorer },
    "cross-validation"
  );
  return scores["score"] as number[];
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
 * fit/score times. Folds are built as in {@link cross_val_score}.
 *
 * @param estimator - Estimator with `fit` and `score` methods
 * @param X - Features of shape (n_samples, n_features)
 * @param y - Targets of shape (n_samples,)
 * @param options - Configuration options
 * @returns Object with testScores (per metric), fitTime, and scoreTime arrays
 * @throws {InvalidParameterError} If `cv` is invalid, `scoring` is empty or holds a non-function, or no scorer is available
 *
 * @example
 * ```ts
 * import { crossValidate } from 'deepbox/ml';
 * import { LogisticRegression } from 'deepbox/ml';
 *
 * // Single metric (uses estimator.score)
 * const result = crossValidate(new LogisticRegression(), X, y, { cv: 5 });
 *
 * // Multiple metrics
 * const result = crossValidate(new LogisticRegression(), X, y, {
 *   cv: 5,
 *   scoring: {
 *     accuracy: (est, X, y) => est.score(X, y),
 *     custom: (est, X, y) => { ... },
 *   },
 * });
 * console.log(result.testScores['accuracy']);
 * ```
 *
 * @deprecated Prefer {@link crossValidate}.
 */
export function cross_validate(
  estimator: Estimator,
  X: Tensor,
  y: Tensor,
  options: {
    cv?: number;
    scoring?: Record<string, Scorer> | Scorer;
  } = {}
): CrossValidateResult {
  const cv = options.cv ?? 5;
  validateCv(cv);

  // Build scoring functions
  let scorers: Record<string, Scorer>;
  if (options.scoring === undefined) {
    if (!hasScore(estimator)) {
      throw new InvalidParameterError(
        "Estimator must implement score() or provide scoring parameter",
        "estimator",
        estimator
      );
    }
    scorers = { score: defaultScorer };
  } else if (typeof options.scoring === "function") {
    scorers = { score: options.scoring };
  } else {
    scorers = options.scoring;
    const names = Object.keys(scorers);
    if (names.length === 0) {
      throw new InvalidParameterError(
        "scoring must define at least one metric",
        "scoring",
        scorers
      );
    }
    for (const name of names) {
      if (typeof scorers[name] !== "function") {
        throw new InvalidParameterError(
          `scoring.${name} must be a function`,
          "scoring",
          scorers[name]
        );
      }
    }
  }

  const nSamples = validateXY(X, y, cv);
  const foldOf = assignFolds(y, nSamples, cv, isClassifier(estimator));
  const folds = buildFolds(X, y, foldOf, cv);
  const { scores, fitTime, scoreTime } = runFolds(
    estimator,
    {},
    folds,
    scorers,
    "cross-validation"
  );
  return { testScores: scores, fitTime, scoreTime };
}

// ---- Helpers for grid/randomized search ----

/** Cartesian product of the parameter lists; the last key varies fastest. */
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
    const key = keys[idx] as string;
    const vals = values[idx] as unknown[];
    for (const v of vals) {
      current[key] = v;
      recurse(idx + 1, current);
    }
  }

  recurse(0, {});
  return combos;
}

function validateParamSpace(space: unknown, name: string): Record<string, unknown[]> {
  if (typeof space !== "object" || space === null || Array.isArray(space)) {
    throw new InvalidParameterError(`${name} must be an object of parameter lists`, name, space);
  }
  const record = space as Record<string, unknown>;
  for (const key of Object.keys(record)) {
    const list = record[key];
    if (!Array.isArray(list) || list.length === 0) {
      throw new InvalidParameterError(
        `${name}.${key} must be a non-empty array of candidate values`,
        name,
        list
      );
    }
  }
  return record as Record<string, unknown[]>;
}

function validateOptionalScoring(scoring: unknown): Scorer | undefined {
  if (scoring !== undefined && typeof scoring !== "function") {
    throw new InvalidParameterError("scoring must be a function", "scoring", scoring);
  }
  return scoring as Scorer | undefined;
}

/** Result of a single parameter combination evaluation */
export type GridSearchResult = {
  readonly params: Record<string, unknown>;
  readonly meanScore: number;
  readonly scores: number[];
};

/**
 * Cross-validate every candidate, pick the best and refit it on all of the data.
 * Candidates whose mean score is NaN are never selected.
 */
function searchCandidates(
  base: Estimator,
  candidates: readonly Record<string, unknown>[],
  X: Tensor,
  y: Tensor,
  cv: number,
  scoring: Scorer | undefined,
  label: string
): {
  results: GridSearchResult[];
  bestIndex: number;
  bestEstimator: Estimator;
} {
  if (scoring === undefined && !hasScore(base)) {
    throw new InvalidParameterError("Estimator must implement score()", "estimator", base);
  }
  const nSamples = validateXY(X, y, cv);
  const foldOf = assignFolds(y, nSamples, cv, isClassifier(base));
  const folds = buildFolds(X, y, foldOf, cv);
  const scorers = { score: scoring ?? defaultScorer };
  const context = `${label} cross-validation`;

  const results: GridSearchResult[] = [];
  let bestIndex = -1;
  let bestScore = -Infinity;
  for (const params of candidates) {
    const { scores } = runFolds(base, params, folds, scorers, context);
    const foldScores = scores["score"] as number[];
    const meanScore = foldScores.reduce((a, b) => a + b, 0) / foldScores.length;
    results.push({ params, meanScore, scores: foldScores });
    if (meanScore > bestScore) {
      bestScore = meanScore;
      bestIndex = results.length - 1;
    }
  }
  if (bestIndex < 0) {
    throw new DataValidationError(
      `${label}: every parameter combination produced a NaN or -Infinity score`
    );
  }

  const bestEstimator = cloneEstimator(
    base,
    `${label} final refit`,
    (results[bestIndex] as GridSearchResult).params
  );
  bestEstimator.fit(X, y);
  return { results, bestIndex, bestEstimator };
}

/**
 * Exhaustive search over specified parameter values for an estimator.
 *
 * Every combination of the listed values is cross-validated (see
 * {@link cross_val_score} for how folds are built) and the best one, by mean
 * score, is refitted on the whole data set. Ties go to the first combination.
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
  private paramGrid: Record<string, unknown[]>;
  private cv: number;
  private scoring: Scorer | undefined;
  private fitted = false;

  /** Best parameters found */
  bestParams: Record<string, unknown> = {};
  /** Best mean CV score */
  bestScore = -Infinity;
  /** Best fitted estimator, refitted on all of the data */
  bestEstimator: Estimator | undefined;
  /** All CV results, in the order the combinations were evaluated */
  cvResults: GridSearchResult[] = [];
  /** Index of the best combination in `cvResults` (-1 before fitting) */
  bestIndex = -1;

  /**
   * @param estimator - Estimator to tune; its constructor must accept the object returned by `getParams()`
   * @param paramGrid - Candidate values per parameter name (each list must be non-empty)
   * @param options.cv - Number of folds (default: 5)
   * @param options.scoring - Scoring function `(estimator, XTest, yTest) => number`; defaults to `estimator.score`
   */
  constructor(
    private estimator: Estimator,
    paramGrid: Record<string, unknown[]>,
    options: { cv?: number; scoring?: Scorer } = {}
  ) {
    this.paramGrid = validateParamSpace(paramGrid, "paramGrid");
    this.cv = options.cv ?? 5;
    validateCv(this.cv);
    this.scoring = validateOptionalScoring(options.scoring);
  }

  fit(X: Tensor, y: Tensor): this {
    const { results, bestIndex, bestEstimator } = searchCandidates(
      this.estimator,
      cartesianProduct(this.paramGrid),
      X,
      y,
      this.cv,
      this.scoring,
      "GridSearchCV"
    );
    const best = results[bestIndex] as GridSearchResult;
    this.cvResults = results;
    this.bestIndex = bestIndex;
    this.bestParams = best.params;
    this.bestScore = best.meanScore;
    this.bestEstimator = bestEstimator;
    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.bestEstimator) {
      throw new NotFittedError("GridSearchCV must be fitted before predict");
    }
    return (this.bestEstimator as Classifier | Regressor).predict(X);
  }

  /**
   * Class probabilities of the best estimator.
   *
   * @throws {NotFittedError} If the search has not been fitted
   * @throws {InvalidParameterError} If the best estimator has no `predictProba`
   */
  predictProba(X: Tensor): Tensor {
    if (!this.fitted || !this.bestEstimator) {
      throw new NotFittedError("GridSearchCV must be fitted before predictProba");
    }
    const est = this.bestEstimator as Partial<Classifier>;
    if (typeof est.predictProba !== "function") {
      throw new InvalidParameterError(
        "The best estimator does not implement predictProba()",
        "estimator",
        this.bestEstimator
      );
    }
    return est.predictProba(X);
  }

  score(X: Tensor, y: Tensor): number {
    if (!this.fitted || !this.bestEstimator) {
      throw new NotFittedError("GridSearchCV must be fitted before score");
    }
    if (this.scoring) return this.scoring(this.bestEstimator, X, y);
    return (this.bestEstimator as Scorable).score(X, y);
  }

  getParams(): Record<string, unknown> {
    return {
      estimator: this.estimator,
      paramGrid: this.paramGrid,
      cv: this.cv,
      scoring: this.scoring,
    };
  }

  /**
   * Update the search configuration. The search must be fitted again afterwards.
   *
   * @throws {InvalidParameterError} On an unknown or invalid parameter
   */
  setParams(params: Record<string, unknown>): this {
    let { estimator, paramGrid, cv, scoring } = {
      estimator: this.estimator,
      paramGrid: this.paramGrid,
      cv: this.cv,
      scoring: this.scoring,
    };
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "estimator":
          estimator = value as Estimator;
          break;
        case "paramGrid":
          paramGrid = validateParamSpace(value, "paramGrid");
          break;
        case "cv":
          validateCv(value);
          cv = value;
          break;
        case "scoring":
          scoring = validateOptionalScoring(value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    this.estimator = estimator;
    this.paramGrid = paramGrid;
    this.cv = cv;
    this.scoring = scoring;
    return this;
  }
}

/**
 * Draw `count` distinct integers from `[0, total)` without building the range
 * (sparse Fisher-Yates), so very large parameter spaces cost O(count) memory.
 */
function sampleDistinct(total: number, count: number, rng: () => number): number[] {
  const swapped = new Map<number, number>();
  const chosen: number[] = [];
  for (let t = 0; t < count; t++) {
    const j = t + Math.min(__randomBelow(rng, total - t), total - t - 1);
    const valueAtJ = swapped.get(j) ?? j;
    swapped.set(j, swapped.get(t) ?? t);
    chosen.push(valueAtJ);
  }
  return chosen;
}

/**
 * Randomized search over parameter distributions.
 *
 * Samples `nIter` distinct parameter combinations (without replacement) from the
 * grid of listed values instead of trying all of them, cross-validates each and
 * refits the best one on the whole data set. If the grid has at most `nIter`
 * combinations, all of them are evaluated.
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
  private paramDistributions: Record<string, unknown[]>;
  private cv: number;
  private nIter: number;
  private randomState: number | undefined;
  private scoring: Scorer | undefined;
  private fitted = false;

  /** Best parameters found */
  bestParams: Record<string, unknown> = {};
  /** Best mean CV score */
  bestScore = -Infinity;
  /** Best fitted estimator, refitted on all of the data */
  bestEstimator: Estimator | undefined;
  /** CV results of the sampled combinations, in sampling order */
  cvResults: GridSearchResult[] = [];
  /** Index of the best combination in `cvResults` (-1 before fitting) */
  bestIndex = -1;

  /**
   * @param estimator - Estimator to tune; its constructor must accept the object returned by `getParams()`
   * @param paramDistributions - Candidate values per parameter name (each list must be non-empty)
   * @param options.nIter - Number of combinations to sample (default: 10)
   * @param options.cv - Number of folds (default: 5)
   * @param options.randomState - Seed of the sampler; without it the global Deepbox generator is used
   * @param options.scoring - Scoring function `(estimator, XTest, yTest) => number`; defaults to `estimator.score`
   */
  constructor(
    private estimator: Estimator,
    paramDistributions: Record<string, unknown[]>,
    options: { nIter?: number; cv?: number; randomState?: number; scoring?: Scorer } = {}
  ) {
    this.paramDistributions = validateParamSpace(paramDistributions, "paramDistributions");
    this.nIter = options.nIter ?? 10;
    this.cv = options.cv ?? 5;
    this.randomState = options.randomState;
    this.scoring = validateOptionalScoring(options.scoring);
    this.validateScalars();
  }

  private validateScalars(): void {
    if (!Number.isInteger(this.nIter) || this.nIter < 1) {
      throw new InvalidParameterError(
        `nIter must be a positive integer; received ${this.nIter}`,
        "nIter",
        this.nIter
      );
    }
    validateCv(this.cv);
    if (this.randomState !== undefined && !Number.isFinite(this.randomState)) {
      throw new InvalidParameterError(
        "randomState must be a finite number",
        "randomState",
        this.randomState
      );
    }
  }

  fit(X: Tensor, y: Tensor): this {
    const keys = Object.keys(this.paramDistributions);
    const sizes = keys.map((k) => (this.paramDistributions[k] as unknown[]).length);
    const total = sizes.reduce((a, b) => a * b, 1);
    if (!Number.isSafeInteger(total)) {
      throw new InvalidParameterError(
        "paramDistributions describes more than 2^53 combinations",
        "paramDistributions",
        this.paramDistributions
      );
    }

    let sampled: Record<string, unknown>[];
    if (total <= this.nIter) {
      sampled = cartesianProduct(this.paramDistributions);
    } else {
      const rng = createRng(this.randomState);
      sampled = sampleDistinct(total, this.nIter, rng).map((flat) => {
        // Mixed-radix decode; the last key varies fastest, as in cartesianProduct.
        const combo: Record<string, unknown> = {};
        let rest = flat;
        for (let k = keys.length - 1; k >= 0; k--) {
          const size = sizes[k] as number;
          combo[keys[k] as string] = (this.paramDistributions[keys[k] as string] as unknown[])[
            rest % size
          ];
          rest = Math.floor(rest / size);
        }
        return combo;
      });
    }

    const { results, bestIndex, bestEstimator } = searchCandidates(
      this.estimator,
      sampled,
      X,
      y,
      this.cv,
      this.scoring,
      "RandomizedSearchCV"
    );
    const best = results[bestIndex] as GridSearchResult;
    this.cvResults = results;
    this.bestIndex = bestIndex;
    this.bestParams = best.params;
    this.bestScore = best.meanScore;
    this.bestEstimator = bestEstimator;
    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.bestEstimator) {
      throw new NotFittedError("RandomizedSearchCV must be fitted before predict");
    }
    return (this.bestEstimator as Classifier | Regressor).predict(X);
  }

  /**
   * Class probabilities of the best estimator.
   *
   * @throws {NotFittedError} If the search has not been fitted
   * @throws {InvalidParameterError} If the best estimator has no `predictProba`
   */
  predictProba(X: Tensor): Tensor {
    if (!this.fitted || !this.bestEstimator) {
      throw new NotFittedError("RandomizedSearchCV must be fitted before predictProba");
    }
    const est = this.bestEstimator as Partial<Classifier>;
    if (typeof est.predictProba !== "function") {
      throw new InvalidParameterError(
        "The best estimator does not implement predictProba()",
        "estimator",
        this.bestEstimator
      );
    }
    return est.predictProba(X);
  }

  score(X: Tensor, y: Tensor): number {
    if (!this.fitted || !this.bestEstimator) {
      throw new NotFittedError("RandomizedSearchCV must be fitted before score");
    }
    if (this.scoring) return this.scoring(this.bestEstimator, X, y);
    return (this.bestEstimator as Scorable).score(X, y);
  }

  getParams(): Record<string, unknown> {
    return {
      estimator: this.estimator,
      paramDistributions: this.paramDistributions,
      nIter: this.nIter,
      cv: this.cv,
      randomState: this.randomState,
      scoring: this.scoring,
    };
  }

  /**
   * Update the search configuration. The search must be fitted again afterwards.
   *
   * @throws {InvalidParameterError} On an unknown or invalid parameter
   */
  setParams(params: Record<string, unknown>): this {
    const prev = {
      estimator: this.estimator,
      paramDistributions: this.paramDistributions,
      nIter: this.nIter,
      cv: this.cv,
      randomState: this.randomState,
      scoring: this.scoring,
    };
    try {
      for (const [key, value] of Object.entries(params)) {
        switch (key) {
          case "estimator":
            this.estimator = value as Estimator;
            break;
          case "paramDistributions":
            this.paramDistributions = validateParamSpace(value, "paramDistributions");
            break;
          case "nIter":
            this.nIter = value as number;
            break;
          case "cv":
            this.cv = value as number;
            break;
          case "randomState":
            this.randomState = value as number | undefined;
            break;
          case "scoring":
            this.scoring = validateOptionalScoring(value);
            break;
          default:
            throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
        }
      }
      this.validateScalars();
    } catch (error) {
      this.estimator = prev.estimator;
      this.paramDistributions = prev.paramDistributions;
      this.nIter = prev.nIter;
      this.cv = prev.cv;
      this.randomState = prev.randomState;
      this.scoring = prev.scoring;
      throw error;
    }
    return this;
  }
}

/**
 * Evaluate an estimator using cross-validation.
 *
 * Splits data into `cv` folds, trains on (cv-1) folds, scores on the remaining fold,
 * and returns an array of scores.
 *
 * The folds come from a fixed-seed shuffle, so results are reproducible. For
 * classifiers (estimators exposing `classes` or `predictProba`) the folds are
 * stratified by the labels in `y`. Fold sizes differ by at most one sample.
 *
 * @param estimator - Estimator with `fit`, `predict`, and `score` methods
 * @param X - Features of shape (n_samples, n_features)
 * @param y - Targets of shape (n_samples,)
 * @param cv - Number of folds (default: 5)
 * @param scoring - Optional scoring function `(estimator, XTest, yTest) => number` used instead of `estimator.score`
 * @returns Array of scores, one per fold
 * @throws {InvalidParameterError} If `cv < 2`, `cv > n_samples`, or the estimator has no `score` method and no `scoring` function is given
 * @throws {ShapeError} If X is not 2-D or X and y disagree on the number of samples
 *
 * @example
 * ```ts
 * import { crossValScore } from 'deepbox/ml';
 * import { LogisticRegression } from 'deepbox/ml';
 *
 * const scores = crossValScore(new LogisticRegression(), X, y, 5);
 * console.log('Mean CV score:', scores.reduce((a,b) => a+b, 0) / scores.length);
 * ```
 */
export const crossValScore = cross_val_score;

/**
 * Evaluate an estimator using cross-validation with multiple metrics.
 *
 * Unlike `cross_val_score` which returns only scores for a single metric,
 * `cross_validate` supports multiple scoring functions and also reports
 * fit/score times. Folds are built as in {@link cross_val_score}.
 *
 * @param estimator - Estimator with `fit` and `score` methods
 * @param X - Features of shape (n_samples, n_features)
 * @param y - Targets of shape (n_samples,)
 * @param options - Configuration options
 * @returns Object with testScores (per metric), fitTime, and scoreTime arrays
 * @throws {InvalidParameterError} If `cv` is invalid, `scoring` is empty or holds a non-function, or no scorer is available
 *
 * @example
 * ```ts
 * import { crossValidate } from 'deepbox/ml';
 * import { LogisticRegression } from 'deepbox/ml';
 *
 * // Single metric (uses estimator.score)
 * const result = crossValidate(new LogisticRegression(), X, y, { cv: 5 });
 *
 * // Multiple metrics
 * const result = crossValidate(new LogisticRegression(), X, y, {
 *   cv: 5,
 *   scoring: {
 *     accuracy: (est, X, y) => est.score(X, y),
 *     custom: (est, X, y) => { ... },
 *   },
 * });
 * console.log(result.testScores['accuracy']);
 * ```
 */
export const crossValidate = cross_validate;
