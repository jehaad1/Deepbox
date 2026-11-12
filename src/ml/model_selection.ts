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

/** One train/test split, as index lists into the rows of X. */
export type CvSplit = {
  readonly trainIndex: ArrayLike<number>;
  readonly testIndex: ArrayLike<number>;
};

/** Group label per sample, as a 1-D tensor or an array. */
export type CvGroups = Tensor | readonly (string | number | bigint)[];

/**
 * Anything that yields train/test index splits, such as the splitters of `deepbox/preprocess`
 * (`KFold`, `StratifiedKFold`, `GroupKFold`, `TimeSeriesSplit`, ...). `y` and `groups` are only
 * passed on; a splitter ignores the ones it does not need.
 */
export type CrossValidator = {
  split(X: Tensor, y?: Tensor, groups?: CvGroups): Iterable<CvSplit>;
};

/** Cross-validation strategy: a number of folds, or a splitter object. */
export type CvSpec = number | CrossValidator;

/** Options shared by the cross-validation helpers. */
export type CvOptions = {
  /** Group label per sample, forwarded to a splitter that uses groups (ignored for integer `cv`). */
  groups?: CvGroups | undefined;
  /** For integer `cv`: shuffle the samples before building folds (default: true). */
  shuffle?: boolean | undefined;
  /** For integer `cv` with `shuffle`: seed of the shuffle (default: 0). */
  randomState?: number | undefined;
};

function isCrossValidator(cv: unknown): cv is CrossValidator {
  return (
    typeof cv === "object" && cv !== null && typeof (cv as CrossValidator).split === "function"
  );
}

function validateCv(cv: unknown): asserts cv is CvSpec {
  if (isCrossValidator(cv)) return;
  if (typeof cv !== "number" || !Number.isInteger(cv) || cv < 2) {
    throw new InvalidParameterError(
      `cv must be an integer >= 2 or an object with a split() method; received ${String(cv)}`,
      "cv",
      cv
    );
  }
}

function validateCvOptions(options: CvOptions): void {
  if (options.shuffle !== undefined && typeof options.shuffle !== "boolean") {
    throw new InvalidParameterError("shuffle must be a boolean", "shuffle", options.shuffle);
  }
  if (options.randomState !== undefined && !Number.isFinite(options.randomState)) {
    throw new InvalidParameterError(
      "randomState must be a finite number",
      "randomState",
      options.randomState
    );
  }
}

/**
 * Check shapes of X and y and return the sample count.
 *
 * @throws {ShapeError} If X is not 2-D, y is not 1-D or 2-D, or their row counts differ
 * @throws {DataValidationError} If X has no rows or no columns
 */
function validateXY(X: Tensor, y: Tensor): number {
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
 * Class code (order of first appearance) of every sample, or `undefined` when `y` does not look
 * like a classification target: more than one column, a single class, or (nearly) as many
 * distinct values as samples.
 */
function classCodes(y: Tensor, nSamples: number): Int32Array | undefined {
  if (y.ndim !== 1 && y.shape[1] !== 1) return undefined;
  const classOf = new Map<number, number>();
  const codes = new Int32Array(nSamples);
  for (let i = 0; i < nSamples; i++) {
    const v = Number(y.data[y.offset + i * (y.strides[0] ?? 1)]);
    let code = classOf.get(v);
    if (code === undefined) {
      code = classOf.size;
      classOf.set(v, code);
    }
    codes[i] = code;
  }
  return classOf.size >= 2 && classOf.size <= nSamples / 2 ? codes : undefined;
}

/**
 * Assign every sample to one of `cv` folds. Fold sizes differ by at most one.
 *
 * With `shuffle`, the samples are permuted with a private generator seeded by `seed`. Contiguous
 * folds would fail on class-sorted data: a training fold could contain a single class and
 * silently corrupt model selection.
 *
 * With `stratify`, the samples of each class are spread evenly over the folds, so every fold
 * keeps (as far as possible) the class proportions of `y`. Without `shuffle` the folds equal
 * those of scikit-learn's `KFold` / `StratifiedKFold`.
 */
function assignFolds(
  y: Tensor,
  nSamples: number,
  cv: number,
  stratify: boolean,
  shuffle: boolean,
  seed: number
): Int32Array {
  const perm = Array.from({ length: nSamples }, (_, i) => i);
  if (shuffle) {
    const rng = createRng(seed);
    for (let i = nSamples - 1; i > 0; i--) {
      const j = Math.min(__randomBelow(rng, i + 1), i);
      const tmp = perm[i] as number;
      perm[i] = perm[j] as number;
      perm[j] = tmp;
    }
  }

  const codes = stratify ? classCodes(y, nSamples) : undefined;
  const foldOf = new Int32Array(nSamples);
  if (codes === undefined) {
    const base = Math.floor(nSamples / cv);
    const extra = nSamples % cv;
    let pos = 0;
    for (let fold = 0; fold < cv; fold++) {
      const size = base + (fold < extra ? 1 : 0);
      for (let k = 0; k < size; k++) foldOf[perm[pos++] as number] = fold;
    }
    return foldOf;
  }

  // Stable sort keeps the (shuffled) sample order inside each class.
  const order = [...perm].sort((a, b) => (codes[a] as number) - (codes[b] as number));
  if (shuffle) {
    for (let pos = 0; pos < nSamples; pos++) foldOf[order[pos] as number] = pos % cv;
    return foldOf;
  }

  // As scikit-learn's StratifiedKFold: the class counts of the folds come from dealing the
  // class-sorted samples round-robin, and each class then fills its folds in consecutive blocks.
  const nClasses = codes.reduce((m, c) => Math.max(m, c), 0) + 1;
  const allocation = Array.from({ length: cv }, () => new Int32Array(nClasses));
  for (let pos = 0; pos < nSamples; pos++) {
    const row = allocation[pos % cv] as Int32Array;
    const code = codes[order[pos] as number] as number;
    row[code] = (row[code] as number) + 1;
  }
  const nextFold = new Int32Array(nClasses);
  const used = new Int32Array(nClasses);
  for (let i = 0; i < nSamples; i++) {
    const code = codes[i] as number;
    let fold = nextFold[code] as number;
    while ((allocation[fold] as Int32Array)[code] === used[code]) {
      fold++;
      used[code] = 0;
    }
    nextFold[code] = fold;
    foldOf[i] = fold;
    used[code] = (used[code] as number) + 1;
  }
  return foldOf;
}

type FoldData = { XTrain: Tensor; yTrain: Tensor; XTest: Tensor; yTest: Tensor };

function foldData(
  X: Tensor,
  y: Tensor,
  trainRows: ArrayLike<number>,
  testRows: ArrayLike<number>
): FoldData {
  // Estimators expect a 1-D target, so an (n_samples, 1) column is flattened.
  const squeezeY = y.ndim === 2 && y.shape[1] === 1;
  return {
    XTrain: takeRows(X, trainRows),
    yTrain: takeRows(y, trainRows, squeezeY),
    XTest: takeRows(X, testRows),
    yTest: takeRows(y, testRows, squeezeY),
  };
}

function buildFolds(X: Tensor, y: Tensor, foldOf: Int32Array, cv: number): FoldData[] {
  const folds: FoldData[] = [];
  for (let fold = 0; fold < cv; fold++) {
    const trainRows: number[] = [];
    const testRows: number[] = [];
    for (let i = 0; i < foldOf.length; i++) {
      if (foldOf[i] === fold) testRows.push(i);
      else trainRows.push(i);
    }
    folds.push(foldData(X, y, trainRows, testRows));
  }
  return folds;
}

function checkIndices(indices: ArrayLike<number>, nSamples: number, name: string): void {
  if (indices.length === 0) {
    throw new InvalidParameterError(`cv produced a split with an empty ${name}`, "cv", indices);
  }
  for (let i = 0; i < indices.length; i++) {
    const idx = indices[i] as number;
    if (!Number.isInteger(idx) || idx < 0 || idx >= nSamples) {
      throw new InvalidParameterError(
        `cv produced ${name} entry ${String(idx)}, which is not a sample index in [0, ${nSamples})`,
        "cv",
        idx
      );
    }
  }
}

function checkGroups(groups: CvGroups, nSamples: number): void {
  const length = Array.isArray(groups) ? groups.length : (groups as Tensor).shape[0];
  if (length !== nSamples) {
    throw new InvalidParameterError(
      `groups must have one entry per sample; got ${String(length)} for ${nSamples} samples`,
      "groups",
      groups
    );
  }
}

/**
 * Build the train/test data of every fold. An integer `cv` makes folds as described in
 * {@link cross_val_score}; a splitter object supplies its own index lists.
 *
 * @throws {InvalidParameterError} If `cv` is an integer above the sample count, `groups` has the
 *   wrong length, or a splitter returns no splits, an empty side, or an index outside the data
 */
function resolveFolds(
  X: Tensor,
  y: Tensor,
  cv: CvSpec,
  options: CvOptions,
  stratify: boolean
): FoldData[] {
  validateCvOptions(options);
  const nSamples = validateXY(X, y);
  if (typeof cv === "number") {
    if (nSamples < cv) {
      throw new InvalidParameterError(`Cannot have cv=${cv} with n_samples=${nSamples}`, "cv", cv);
    }
    const foldOf = assignFolds(
      y,
      nSamples,
      cv,
      stratify,
      options.shuffle ?? true,
      options.randomState ?? 0
    );
    return buildFolds(X, y, foldOf, cv);
  }
  if (options.groups !== undefined) checkGroups(options.groups, nSamples);
  const yFlat = y.ndim === 2 && y.shape[1] === 1 ? y.reshape([nSamples]) : y;
  const folds: FoldData[] = [];
  for (const split of cv.split(X, yFlat, options.groups)) {
    checkIndices(split.trainIndex, nSamples, "trainIndex");
    checkIndices(split.testIndex, nSamples, "testIndex");
    folds.push(foldData(X, y, split.trainIndex, split.testIndex));
  }
  if (folds.length === 0) {
    throw new InvalidParameterError("cv produced no splits", "cv", cv);
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

/** Options object accepted as the 4th argument of {@link cross_val_score}. */
export type CrossValScoreOptions = CvOptions & {
  /** Number of folds or a splitter object (default: 5). */
  cv?: CvSpec;
  /** Scoring function `(estimator, XTest, yTest) => number` used instead of `estimator.score`. */
  scoring?: Scorer;
};

/**
 * Evaluate an estimator using cross-validation.
 *
 * With an integer `cv` the data is split into `cv` folds, the estimator is trained on `cv - 1`
 * of them and scored on the remaining one. The folds come from a shuffle with a fixed seed (set
 * `shuffle: false` for contiguous folds like scikit-learn's `KFold`, or `randomState` for another
 * seed), so results are reproducible. For classifiers (estimators exposing `classes` or
 * `predictProba`) the folds are stratified by the labels in `y`. Fold sizes differ by at most one
 * sample.
 *
 * `cv` may also be any splitter object with a `split(X, y?, groups?)` method returning
 * `{ trainIndex, testIndex }` pairs, such as `TimeSeriesSplit`, `GroupKFold` or `KFold` from
 * `deepbox/preprocess`. Then `groups` is passed on to it, and `shuffle` and `randomState` are
 * ignored.
 *
 * The 4th argument is either `cv` (a number or a splitter) followed by `scoring` and an options
 * object, or one options object holding `cv`, `scoring`, `groups`, `shuffle` and `randomState`.
 *
 * @param estimator - Estimator with `fit`, `predict`, and `score` methods
 * @param X - Features of shape (n_samples, n_features)
 * @param y - Targets of shape (n_samples,)
 * @param cv - Number of folds, a splitter object, or an options object (default: 5)
 * @param scoring - Optional scoring function `(estimator, XTest, yTest) => number` used instead of `estimator.score`
 * @param options - `groups`, `shuffle` and `randomState` (only with a `cv` that is not an options object)
 * @returns Array of scores, one per fold
 * @throws {InvalidParameterError} If `cv` is invalid, `cv > n_samples`, a splitter returns an invalid split, or the estimator has no `score` method and no `scoring` function is given
 * @throws {ShapeError} If X is not 2-D or X and y disagree on the number of samples
 *
 * @example
 * ```ts
 * import { crossValScore } from 'deepbox/ml';
 * import { LogisticRegression } from 'deepbox/ml';
 * import { TimeSeriesSplit } from 'deepbox/preprocess';
 *
 * const scores = crossValScore(new LogisticRegression(), X, y, 5);
 * console.log('Mean CV score:', scores.reduce((a,b) => a+b, 0) / scores.length);
 *
 * const ts = crossValScore(new LogisticRegression(), X, y, { cv: new TimeSeriesSplit({ nSplits: 4 }) });
 * ```
 *
 * @deprecated Prefer {@link crossValScore}.
 */
export function cross_val_score(
  estimator: Estimator,
  X: Tensor,
  y: Tensor,
  cv?: CvSpec | CrossValScoreOptions,
  scoring?: Scorer,
  options?: CvOptions
): number[] {
  const resolved = splitCrossValArgs(cv, scoring, options);
  validateCv(resolved.cv);
  if (resolved.scoring === undefined && !hasScore(estimator)) {
    throw new InvalidParameterError("Estimator must implement score()", "estimator", estimator);
  }
  if (resolved.scoring !== undefined && typeof resolved.scoring !== "function") {
    throw new InvalidParameterError("scoring must be a function", "scoring", resolved.scoring);
  }
  const folds = resolveFolds(X, y, resolved.cv, resolved.options, isClassifier(estimator));
  const { scores } = runFolds(
    estimator,
    {},
    folds,
    { score: resolved.scoring ?? defaultScorer },
    "cross-validation"
  );
  return scores["score"] as number[];
}

/**
 * Normalize the positional and options-object call forms of `cross_val_score`.
 * A splitter is told apart from an options object by its `split` method.
 */
function splitCrossValArgs(
  cv: CvSpec | CrossValScoreOptions | undefined,
  scoring: Scorer | undefined,
  options: CvOptions | undefined
): { cv: CvSpec; scoring: Scorer | undefined; options: CvOptions } {
  if (typeof cv === "object" && cv !== null && !isCrossValidator(cv)) {
    const { cv: innerCv, scoring: innerScoring, ...rest } = cv;
    return { cv: innerCv ?? 5, scoring: innerScoring ?? scoring, options: { ...options, ...rest } };
  }
  return { cv: cv ?? 5, scoring, options: options ?? {} };
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
 * fit/score times. Folds are built as in {@link cross_val_score}: `options.cv` is a number of
 * folds or a splitter object, and `options.groups`, `options.shuffle` and `options.randomState`
 * behave as there.
 *
 * @param estimator - Estimator with `fit` and `score` methods
 * @param X - Features of shape (n_samples, n_features)
 * @param y - Targets of shape (n_samples,)
 * @param options - Configuration options (`cv`, `scoring`, `groups`, `shuffle`, `randomState`)
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
  options: CvOptions & {
    cv?: CvSpec;
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

  const folds = resolveFolds(X, y, cv, options, isClassifier(estimator));
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
  cv: CvSpec,
  cvOptions: CvOptions,
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
  const folds = resolveFolds(X, y, cv, cvOptions, isClassifier(base));
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
  private cv: CvSpec;
  private shuffle: boolean | undefined;
  private randomState: number | undefined;
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
   * @param options.cv - Number of folds or a splitter object (default: 5)
   * @param options.shuffle - For integer `cv`: shuffle before building folds (default: true)
   * @param options.randomState - For integer `cv` with `shuffle`: seed of the shuffle (default: 0)
   * @param options.scoring - Scoring function `(estimator, XTest, yTest) => number`; defaults to `estimator.score`
   */
  constructor(
    private estimator: Estimator,
    paramGrid: Record<string, unknown[]>,
    options: { cv?: CvSpec; shuffle?: boolean; randomState?: number; scoring?: Scorer } = {}
  ) {
    this.paramGrid = validateParamSpace(paramGrid, "paramGrid");
    this.cv = options.cv ?? 5;
    this.shuffle = options.shuffle;
    this.randomState = options.randomState;
    validateCv(this.cv);
    validateCvOptions({ shuffle: this.shuffle, randomState: this.randomState });
    this.scoring = validateOptionalScoring(options.scoring);
  }

  /**
   * Cross-validate every combination and refit the best one.
   *
   * @param groups - Group label per sample, forwarded to a `cv` splitter that uses groups
   */
  fit(X: Tensor, y: Tensor, groups?: CvGroups): this {
    const { results, bestIndex, bestEstimator } = searchCandidates(
      this.estimator,
      cartesianProduct(this.paramGrid),
      X,
      y,
      this.cv,
      { groups, shuffle: this.shuffle, randomState: this.randomState },
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
      shuffle: this.shuffle,
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
    let { estimator, paramGrid, cv, shuffle, randomState, scoring } = {
      estimator: this.estimator,
      paramGrid: this.paramGrid,
      cv: this.cv,
      shuffle: this.shuffle,
      randomState: this.randomState,
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
        case "shuffle":
          shuffle = value as boolean | undefined;
          break;
        case "randomState":
          randomState = value as number | undefined;
          break;
        case "scoring":
          scoring = validateOptionalScoring(value);
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    validateCvOptions({ shuffle, randomState });
    this.estimator = estimator;
    this.paramGrid = paramGrid;
    this.cv = cv;
    this.shuffle = shuffle;
    this.randomState = randomState;
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
  private cv: CvSpec;
  private shuffle: boolean | undefined;
  private cvRandomState: number | undefined;
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
   * @param options.cv - Number of folds or a splitter object (default: 5)
   * @param options.shuffle - For integer `cv`: shuffle before building folds (default: true)
   * @param options.cvRandomState - For integer `cv` with `shuffle`: seed of the fold shuffle (default: 0)
   * @param options.randomState - Seed of the sampler; without it the global Deepbox generator is used
   * @param options.scoring - Scoring function `(estimator, XTest, yTest) => number`; defaults to `estimator.score`
   */
  constructor(
    private estimator: Estimator,
    paramDistributions: Record<string, unknown[]>,
    options: {
      nIter?: number;
      cv?: CvSpec;
      shuffle?: boolean;
      cvRandomState?: number;
      randomState?: number;
      scoring?: Scorer;
    } = {}
  ) {
    this.paramDistributions = validateParamSpace(paramDistributions, "paramDistributions");
    this.nIter = options.nIter ?? 10;
    this.cv = options.cv ?? 5;
    this.shuffle = options.shuffle;
    this.cvRandomState = options.cvRandomState;
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
    validateCvOptions({ shuffle: this.shuffle, randomState: this.cvRandomState });
    if (this.randomState !== undefined && !Number.isFinite(this.randomState)) {
      throw new InvalidParameterError(
        "randomState must be a finite number",
        "randomState",
        this.randomState
      );
    }
  }

  /**
   * Cross-validate the sampled combinations and refit the best one.
   *
   * @param groups - Group label per sample, forwarded to a `cv` splitter that uses groups
   */
  fit(X: Tensor, y: Tensor, groups?: CvGroups): this {
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
      { groups, shuffle: this.shuffle, randomState: this.cvRandomState },
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
      shuffle: this.shuffle,
      cvRandomState: this.cvRandomState,
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
      shuffle: this.shuffle,
      cvRandomState: this.cvRandomState,
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
            this.cv = value as CvSpec;
            break;
          case "shuffle":
            this.shuffle = value as boolean | undefined;
            break;
          case "cvRandomState":
            this.cvRandomState = value as number | undefined;
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
      this.shuffle = prev.shuffle;
      this.cvRandomState = prev.cvRandomState;
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
 * With an integer `cv` the data is split into `cv` folds, the estimator is trained on `cv - 1`
 * of them and scored on the remaining one. The folds come from a shuffle with a fixed seed (set
 * `shuffle: false` for contiguous folds like scikit-learn's `KFold`, or `randomState` for another
 * seed), so results are reproducible. For classifiers (estimators exposing `classes` or
 * `predictProba`) the folds are stratified by the labels in `y`. Fold sizes differ by at most one
 * sample.
 *
 * `cv` may also be any splitter object with a `split(X, y?, groups?)` method returning
 * `{ trainIndex, testIndex }` pairs, such as `TimeSeriesSplit`, `GroupKFold` or `KFold` from
 * `deepbox/preprocess`. Then `groups` is passed on to it, and `shuffle` and `randomState` are
 * ignored.
 *
 * The 4th argument is either `cv` (a number or a splitter) followed by `scoring` and an options
 * object, or one options object holding `cv`, `scoring`, `groups`, `shuffle` and `randomState`.
 *
 * @param estimator - Estimator with `fit`, `predict`, and `score` methods
 * @param X - Features of shape (n_samples, n_features)
 * @param y - Targets of shape (n_samples,)
 * @param cv - Number of folds, a splitter object, or an options object (default: 5)
 * @param scoring - Optional scoring function `(estimator, XTest, yTest) => number` used instead of `estimator.score`
 * @param options - `groups`, `shuffle` and `randomState` (only with a `cv` that is not an options object)
 * @returns Array of scores, one per fold
 * @throws {InvalidParameterError} If `cv` is invalid, `cv > n_samples`, a splitter returns an invalid split, or the estimator has no `score` method and no `scoring` function is given
 * @throws {ShapeError} If X is not 2-D or X and y disagree on the number of samples
 *
 * @example
 * ```ts
 * import { crossValScore } from 'deepbox/ml';
 * import { LogisticRegression } from 'deepbox/ml';
 * import { TimeSeriesSplit } from 'deepbox/preprocess';
 *
 * const scores = crossValScore(new LogisticRegression(), X, y, 5);
 * console.log('Mean CV score:', scores.reduce((a,b) => a+b, 0) / scores.length);
 *
 * const ts = crossValScore(new LogisticRegression(), X, y, { cv: new TimeSeriesSplit({ nSplits: 4 }) });
 * ```
 */
export const crossValScore = cross_val_score;

/**
 * Evaluate an estimator using cross-validation with multiple metrics.
 *
 * Unlike `cross_val_score` which returns only scores for a single metric,
 * `cross_validate` supports multiple scoring functions and also reports
 * fit/score times. Folds are built as in {@link cross_val_score}: `options.cv` is a number of
 * folds or a splitter object, and `options.groups`, `options.shuffle` and `options.randomState`
 * behave as there.
 *
 * @param estimator - Estimator with `fit` and `score` methods
 * @param X - Features of shape (n_samples, n_features)
 * @param y - Targets of shape (n_samples,)
 * @param options - Configuration options (`cv`, `scoring`, `groups`, `shuffle`, `randomState`)
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
