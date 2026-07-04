/**
 * Feature selection transformers.
 *
 * @module preprocess/feature_selection
 * @see {@link https://deepbox.dev/docs/preprocess-features | Deepbox Feature Selection}
 */

import { DTypeError, InvalidParameterError, NotFittedError } from "../core/errors";
import { type Tensor, Tensor as TensorClass, tensor, zeros } from "../ndarray";
import { assert2D, getShape2D, getStrides2D } from "./_internal";

/**
 * Feature selector that removes all low-variance features.
 *
 * Features with a variance lower than the threshold will be removed.
 * By default, removes all zero-variance features (constant features).
 *
 * @example
 * ```ts
 * import { VarianceThreshold } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]]);
 * const selector = new VarianceThreshold({ threshold: 0.0 });
 * selector.fit(X);
 * const Xt = selector.transform(X); // removes constant columns
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-features | Deepbox Feature Selection}
 */
export class VarianceThreshold {
  private readonly threshold: number;
  private variances_: number[] | undefined;
  private mask_: boolean[] | undefined;
  private nFeaturesIn_: number | undefined;

  constructor(options: { threshold?: number } = {}) {
    this.threshold = options.threshold ?? 0.0;
    if (!Number.isFinite(this.threshold) || this.threshold < 0) {
      throw new InvalidParameterError(
        "threshold must be a non-negative number",
        "threshold",
        this.threshold
      );
    }
  }

  fit(X: Tensor, _y?: Tensor): this {
    if (X.dtype === "string") {
      throw new DTypeError("VarianceThreshold requires numeric input");
    }
    assert2D(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const [rowStride, colStride] = getStrides2D(X);

    if (nSamples === 0) {
      throw new InvalidParameterError("Cannot fit on empty array", "X", nSamples);
    }

    const variances: number[] = [];
    const mask: boolean[] = [];

    for (let j = 0; j < nFeatures; j++) {
      let sum = 0;
      let sumSq = 0;
      for (let i = 0; i < nSamples; i++) {
        const val = Number(X.data[X.offset + i * rowStride + j * colStride]);
        sum += val;
        sumSq += val * val;
      }
      const mean = sum / nSamples;
      const variance = sumSq / nSamples - mean * mean;
      variances.push(variance);
      mask.push(variance > this.threshold);
    }

    this.variances_ = variances;
    this.mask_ = mask;
    this.nFeaturesIn_ = nFeatures;

    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.mask_ || !this.variances_ || this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("VarianceThreshold must be fitted before transform");
    }
    if (X.dtype === "string") {
      throw new DTypeError("VarianceThreshold requires numeric input");
    }
    assert2D(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const [rowStride, colStride] = getStrides2D(X);

    if (nFeatures !== this.nFeaturesIn_) {
      throw new InvalidParameterError(
        `Expected ${this.nFeaturesIn_} features, got ${nFeatures}`,
        "X",
        nFeatures
      );
    }

    const selectedCols: number[] = [];
    for (let j = 0; j < nFeatures; j++) {
      if (this.mask_[j]) {
        selectedCols.push(j);
      }
    }

    const nOut = selectedCols.length;
    if (nOut === 0) {
      return zeros([nSamples, 0], { dtype: X.dtype });
    }

    // Gather the selected columns straight into a flat buffer.
    const out = new Float64Array(nSamples * nOut);
    const src = X.data;
    const offset = X.offset;
    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      const rowBase = offset + i * rowStride;
      for (let c = 0; c < nOut; c++) {
        out[pos++] = Number(src[rowBase + (selectedCols[c] as number) * colStride]);
      }
    }

    return TensorClass.fromTypedArray({
      data: out,
      shape: [nSamples, nOut],
      dtype: "float64",
      device: X.device,
    });
  }

  fitTransform(X: Tensor, y?: Tensor): Tensor {
    this.fit(X, y);
    return this.transform(X);
  }

  /** Get the boolean mask of selected features. */
  getSupport(): boolean[] {
    if (!this.mask_) {
      throw new NotFittedError("VarianceThreshold must be fitted before getSupport");
    }
    return [...this.mask_];
  }

  /** Get the computed variances. */
  get variances(): number[] {
    if (!this.variances_) {
      throw new NotFittedError("VarianceThreshold must be fitted before accessing variances");
    }
    return [...this.variances_];
  }

  getParams(): Record<string, unknown> {
    return { threshold: this.threshold };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}

/** Scoring function type for SelectKBest. */
export type ScoreFunc = (X: Tensor, y: Tensor) => number[];

/**
 * ANOVA F-value between each feature and the target classes.
 * @deprecated Prefer {@link fClassif}.
 */
export function f_classif(X: Tensor, y: Tensor): number[] {
  assert2D(X, "X");
  const [nSamples, nFeatures] = getShape2D(X);
  const [rowStride, colStride] = getStrides2D(X);
  const classMap = new Map<number, number[]>();
  for (let i = 0; i < nSamples; i++) {
    const label = Number(y.data[y.offset + i]);
    let arr = classMap.get(label);
    if (!arr) {
      arr = [];
      classMap.set(label, arr);
    }
    arr.push(i);
  }
  const classes = [...classMap.values()];
  const nClasses = classes.length;
  const scores: number[] = [];
  for (let j = 0; j < nFeatures; j++) {
    let grandSum = 0;
    for (let i = 0; i < nSamples; i++)
      grandSum += Number(X.data[X.offset + i * rowStride + j * colStride]);
    const grandMean = grandSum / nSamples;
    let ssBetween = 0,
      ssWithin = 0;
    for (const gi of classes) {
      let gs = 0;
      for (const idx of gi) gs += Number(X.data[X.offset + idx * rowStride + j * colStride]);
      const gm = gs / gi.length;
      ssBetween += gi.length * (gm - grandMean) ** 2;
      for (const idx of gi) {
        const v = Number(X.data[X.offset + idx * rowStride + j * colStride]);
        ssWithin += (v - gm) ** 2;
      }
    }
    const dfB = nClasses - 1,
      dfW = nSamples - nClasses;
    scores.push(
      dfW <= 0 || ssWithin === 0
        ? ssBetween > 0
          ? Infinity
          : 0
        : ssBetween / dfB / (ssWithin / dfW)
    );
  }
  return scores;
}

/**
 * Univariate F-statistic from correlation between each feature and target.
 * @deprecated Prefer {@link fRegression}.
 */
export function f_regression(X: Tensor, y: Tensor): number[] {
  assert2D(X, "X");
  const [nSamples, nFeatures] = getShape2D(X);
  const [rowStride, colStride] = getStrides2D(X);
  let ySum = 0;
  for (let i = 0; i < nSamples; i++) ySum += Number(y.data[y.offset + i]);
  const yMean = ySum / nSamples;
  let ySS = 0;
  for (let i = 0; i < nSamples; i++) {
    const v = Number(y.data[y.offset + i]) - yMean;
    ySS += v * v;
  }
  const scores: number[] = [];
  for (let j = 0; j < nFeatures; j++) {
    let xSum = 0;
    for (let i = 0; i < nSamples; i++)
      xSum += Number(X.data[X.offset + i * rowStride + j * colStride]);
    const xMean = xSum / nSamples;
    let xSS = 0,
      xyCov = 0;
    for (let i = 0; i < nSamples; i++) {
      const xv = Number(X.data[X.offset + i * rowStride + j * colStride]) - xMean;
      const yv = Number(y.data[y.offset + i]) - yMean;
      xSS += xv * xv;
      xyCov += xv * yv;
    }
    if (xSS === 0 || ySS === 0) {
      scores.push(0);
      continue;
    }
    const r2 = (xyCov / Math.sqrt(xSS * ySS)) ** 2;
    const dfDen = nSamples - 2;
    scores.push(dfDen <= 0 || r2 >= 1 ? (r2 >= 1 ? Infinity : 0) : r2 / ((1 - r2) / dfDen));
  }
  return scores;
}

/**
 * Select features according to the K highest scores.
 *
 * @example
 * ```ts
 * import { SelectKBest, f_classif } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]);
 * const y = tensor([0,0,1,1]);
 * const skb = new SelectKBest({ scoreFunc: f_classif, k: 2 });
 * skb.fit(X, y);
 * const Xt = skb.transform(X);
 * ```
 */
export class SelectKBest {
  private readonly k: number;
  private readonly scoreFunc: ScoreFunc;
  private scores_: number[] | undefined;
  private mask_: boolean[] | undefined;
  private nFeaturesIn_: number | undefined;

  constructor(options: { scoreFunc?: ScoreFunc; k?: number } = {}) {
    this.k = options.k ?? 10;
    this.scoreFunc = options.scoreFunc ?? f_classif;
    if (!Number.isInteger(this.k) || this.k < 1) {
      throw new InvalidParameterError("k must be a positive integer", "k", this.k);
    }
  }

  fit(X: Tensor, y: Tensor): this {
    if (X.dtype === "string") throw new DTypeError("SelectKBest requires numeric input");
    assert2D(X, "X");
    const [, nFeatures] = getShape2D(X);
    if (this.k > nFeatures) {
      throw new InvalidParameterError(
        `k=${this.k} exceeds number of features (${nFeatures})`,
        "k",
        this.k
      );
    }
    const scores = this.scoreFunc(X, y);
    const indexed = scores.map((s, i) => ({ score: s, idx: i }));
    indexed.sort((a, b) => b.score - a.score);
    const selected = new Set<number>();
    for (let i = 0; i < this.k; i++) selected.add(indexed[i]!.idx);
    this.mask_ = Array.from({ length: nFeatures }, (_, j) => selected.has(j));
    this.scores_ = scores;
    this.nFeaturesIn_ = nFeatures;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.mask_ || this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("SelectKBest must be fitted before transform");
    }
    if (X.dtype === "string") throw new DTypeError("SelectKBest requires numeric input");
    assert2D(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    const [rowStride, colStride] = getStrides2D(X);
    if (nFeatures !== this.nFeaturesIn_) {
      throw new InvalidParameterError(
        `Expected ${this.nFeaturesIn_} features, got ${nFeatures}`,
        "X",
        nFeatures
      );
    }
    const cols: number[] = [];
    for (let j = 0; j < nFeatures; j++) {
      if (this.mask_[j]) cols.push(j);
    }
    const nOut = cols.length;
    if (nOut === 0) return zeros([nSamples, 0], { dtype: X.dtype });
    const data: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      for (const c of cols) data.push(Number(X.data[X.offset + i * rowStride + c * colStride]));
    }
    return tensor(data).reshape([nSamples, nOut]);
  }

  fitTransform(X: Tensor, y: Tensor): Tensor {
    this.fit(X, y);
    return this.transform(X);
  }
  getSupport(): boolean[] {
    if (!this.mask_) throw new NotFittedError("SelectKBest must be fitted before getSupport");
    return [...this.mask_];
  }
  get scores(): number[] {
    if (!this.scores_)
      throw new NotFittedError("SelectKBest must be fitted before accessing scores");
    return [...this.scores_];
  }
  getParams(): Record<string, unknown> {
    return { k: this.k };
  }
  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}

// ---- Helper: extract feature importances from a fitted estimator ----

/**
 * Estimator interface expected by SelectFromModel / RFE.
 * Must expose `fit(X, y)` and either `featureImportances_` or `coef_`.
 */
interface ImportanceEstimator {
  fit(X: Tensor, y: Tensor): unknown;
  featureImportances_?: number[] | undefined;
  coef_?: Tensor | undefined;
}

function extractImportances(estimator: ImportanceEstimator, nFeatures: number): number[] {
  if (estimator.featureImportances_) {
    return [...estimator.featureImportances_];
  }
  if (estimator.coef_) {
    const coef = estimator.coef_;
    const importances: number[] = [];
    if (coef.ndim === 1) {
      for (let j = 0; j < nFeatures; j++) {
        importances.push(Math.abs(Number(coef.data[coef.offset + j])));
      }
    } else if (coef.ndim === 2) {
      const [nClasses] = getShape2D(coef);
      const [rStride, cStride] = getStrides2D(coef);
      for (let j = 0; j < nFeatures; j++) {
        let sum = 0;
        for (let c = 0; c < nClasses; c++) {
          sum += Math.abs(Number(coef.data[coef.offset + c * rStride + j * cStride]));
        }
        importances.push(sum / nClasses);
      }
    } else {
      throw new InvalidParameterError("Estimator coef_ must be 1D or 2D", "estimator");
    }
    return importances;
  }
  throw new InvalidParameterError(
    "Estimator must have featureImportances_ or coef_ after fitting",
    "estimator"
  );
}

function selectColumnsFromMask(X: Tensor, cols: number[]): Tensor {
  const [nSamples] = getShape2D(X);
  const [rowStride, colStride] = getStrides2D(X);
  const nOut = cols.length;
  if (nOut === 0) return zeros([nSamples, 0], { dtype: X.dtype });
  const data: number[] = [];
  for (let i = 0; i < nSamples; i++) {
    for (const c of cols) {
      data.push(Number(X.data[X.offset + i * rowStride + c * colStride]));
    }
  }
  return tensor(data).reshape([nSamples, nOut]);
}

/**
 * Meta-transformer for selecting features based on importance weights
 * from a fitted estimator.
 *
 * The estimator must expose `featureImportances_` (e.g. tree-based models)
 * or `coef_` (e.g. linear models) after fitting.
 *
 * @example
 * ```ts
 * import { SelectFromModel } from 'deepbox/preprocess';
 * import { RandomForestClassifier } from 'deepbox/ml';
 *
 * const selector = new SelectFromModel({
 *   estimator: new RandomForestClassifier(),
 *   threshold: 'mean',
 * });
 * selector.fit(X, y);
 * const Xt = selector.transform(X);
 * ```
 */
export class SelectFromModel {
  private readonly estimator: ImportanceEstimator;
  private readonly threshold: "mean" | "median" | number;
  private readonly maxFeatures: number | undefined;
  private importances_: number[] | undefined;
  private mask_: boolean[] | undefined;
  private nFeaturesIn_: number | undefined;

  constructor(options: {
    estimator: ImportanceEstimator;
    threshold?: "mean" | "median" | number;
    maxFeatures?: number;
  }) {
    this.estimator = options.estimator;
    this.threshold = options.threshold ?? "mean";
    this.maxFeatures = options.maxFeatures;
    if (typeof this.threshold === "number" && this.threshold < 0) {
      throw new InvalidParameterError(
        "threshold must be non-negative",
        "threshold",
        this.threshold
      );
    }
    if (
      this.maxFeatures !== undefined &&
      (!Number.isInteger(this.maxFeatures) || this.maxFeatures < 1)
    ) {
      throw new InvalidParameterError(
        "maxFeatures must be a positive integer",
        "maxFeatures",
        this.maxFeatures
      );
    }
  }

  fit(X: Tensor, y: Tensor): this {
    assert2D(X, "X");
    const [, nFeatures] = getShape2D(X);
    this.estimator.fit(X, y);
    const importances = extractImportances(this.estimator, nFeatures);
    this.importances_ = importances;

    let thresh: number;
    if (this.threshold === "mean") {
      thresh = importances.reduce((a, b) => a + b, 0) / importances.length;
    } else if (this.threshold === "median") {
      const sorted = [...importances].sort((a, b) => a - b);
      const mid = Math.floor(sorted.length / 2);
      thresh =
        sorted.length % 2 === 0
          ? ((sorted[mid - 1] ?? 0) + (sorted[mid] ?? 0)) / 2
          : (sorted[mid] ?? 0);
    } else {
      thresh = this.threshold;
    }

    const mask = importances.map((imp) => imp >= thresh);

    if (this.maxFeatures !== undefined) {
      const selected = importances
        .map((imp, idx) => ({ imp, idx }))
        .filter((_, idx) => mask[idx])
        .sort((a, b) => b.imp - a.imp);
      if (selected.length > this.maxFeatures) {
        const keep = new Set(selected.slice(0, this.maxFeatures).map((s) => s.idx));
        for (let j = 0; j < nFeatures; j++) {
          mask[j] = keep.has(j);
        }
      }
    }

    this.mask_ = mask;
    this.nFeaturesIn_ = nFeatures;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.mask_ || this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("SelectFromModel must be fitted before transform");
    }
    assert2D(X, "X");
    const [, nFeatures] = getShape2D(X);
    if (nFeatures !== this.nFeaturesIn_) {
      throw new InvalidParameterError(
        `Expected ${this.nFeaturesIn_} features, got ${nFeatures}`,
        "X",
        nFeatures
      );
    }
    const cols: number[] = [];
    for (let j = 0; j < nFeatures; j++) {
      if (this.mask_[j]) cols.push(j);
    }
    return selectColumnsFromMask(X, cols);
  }

  fitTransform(X: Tensor, y: Tensor): Tensor {
    this.fit(X, y);
    return this.transform(X);
  }

  getSupport(): boolean[] {
    if (!this.mask_) {
      throw new NotFittedError("SelectFromModel must be fitted before getSupport");
    }
    return [...this.mask_];
  }

  get importances(): number[] {
    if (!this.importances_) {
      throw new NotFittedError("SelectFromModel must be fitted before accessing importances");
    }
    return [...this.importances_];
  }

  getParams(): Record<string, unknown> {
    return {
      threshold: this.threshold,
      maxFeatures: this.maxFeatures,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}

/**
 * Recursive Feature Elimination (RFE).
 *
 * Recursively removes the least important features, re-fitting the
 * estimator each time, until the desired number of features is reached.
 *
 * @example
 * ```ts
 * import { RFE } from 'deepbox/preprocess';
 * import { LogisticRegression } from 'deepbox/ml';
 *
 * const rfe = new RFE({
 *   estimator: new LogisticRegression(),
 *   nFeaturesToSelect: 3,
 * });
 * rfe.fit(X, y);
 * const Xt = rfe.transform(X);
 * ```
 */
export class RFE {
  private readonly estimator: ImportanceEstimator;
  private readonly nFeaturesToSelect: number;
  private readonly step: number;
  private ranking_: number[] | undefined;
  private mask_: boolean[] | undefined;
  private nFeaturesIn_: number | undefined;

  constructor(options: {
    estimator: ImportanceEstimator;
    nFeaturesToSelect?: number;
    step?: number;
  }) {
    this.estimator = options.estimator;
    this.nFeaturesToSelect = options.nFeaturesToSelect ?? 1;
    this.step = options.step ?? 1;
    if (!Number.isInteger(this.nFeaturesToSelect) || this.nFeaturesToSelect < 1) {
      throw new InvalidParameterError(
        "nFeaturesToSelect must be a positive integer",
        "nFeaturesToSelect",
        this.nFeaturesToSelect
      );
    }
    if (!Number.isInteger(this.step) || this.step < 1) {
      throw new InvalidParameterError("step must be a positive integer", "step", this.step);
    }
  }

  fit(X: Tensor, y: Tensor): this {
    assert2D(X, "X");
    const [, nFeatures] = getShape2D(X);
    if (this.nFeaturesToSelect > nFeatures) {
      throw new InvalidParameterError(
        `nFeaturesToSelect=${this.nFeaturesToSelect} exceeds number of features (${nFeatures})`,
        "nFeaturesToSelect",
        this.nFeaturesToSelect
      );
    }

    let activeIndices: number[] = Array.from({ length: nFeatures }, (_, i) => i);
    const ranking = new Array<number>(nFeatures).fill(0);
    let eliminationRound = 0;
    let currentX = X;

    while (activeIndices.length > this.nFeaturesToSelect) {
      this.estimator.fit(currentX, y);
      const importances = extractImportances(this.estimator, activeIndices.length);

      const nToRemove = Math.min(this.step, activeIndices.length - this.nFeaturesToSelect);

      const indexed = importances.map((imp, i) => ({ imp, i }));
      indexed.sort((a, b) => a.imp - b.imp);
      const removeSet = new Set<number>();
      for (let k = 0; k < nToRemove; k++) {
        removeSet.add(indexed[k]!.i);
      }

      eliminationRound++;
      for (const localIdx of removeSet) {
        ranking[activeIndices[localIdx]!] = eliminationRound + 1;
      }

      activeIndices = activeIndices.filter((_, i) => !removeSet.has(i));
      currentX = selectColumnsFromMask(X, activeIndices);
    }

    for (const idx of activeIndices) {
      ranking[idx] = 1;
    }

    this.ranking_ = ranking;
    this.mask_ = ranking.map((r) => r === 1);
    this.nFeaturesIn_ = nFeatures;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.mask_ || this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("RFE must be fitted before transform");
    }
    assert2D(X, "X");
    const [, nFeatures] = getShape2D(X);
    if (nFeatures !== this.nFeaturesIn_) {
      throw new InvalidParameterError(
        `Expected ${this.nFeaturesIn_} features, got ${nFeatures}`,
        "X",
        nFeatures
      );
    }
    const cols: number[] = [];
    for (let j = 0; j < nFeatures; j++) {
      if (this.mask_[j]) cols.push(j);
    }
    return selectColumnsFromMask(X, cols);
  }

  fitTransform(X: Tensor, y: Tensor): Tensor {
    this.fit(X, y);
    return this.transform(X);
  }

  getSupport(): boolean[] {
    if (!this.mask_) {
      throw new NotFittedError("RFE must be fitted before getSupport");
    }
    return [...this.mask_];
  }

  /** Feature ranking (1 = selected, higher = eliminated earlier). */
  get ranking(): number[] {
    if (!this.ranking_) {
      throw new NotFittedError("RFE must be fitted before accessing ranking");
    }
    return [...this.ranking_];
  }

  getParams(): Record<string, unknown> {
    return {
      nFeaturesToSelect: this.nFeaturesToSelect,
      step: this.step,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}

/**
 * Estimator with a predict method for cross-validation in RFECV.
 */
interface ScoringEstimator extends ImportanceEstimator {
  predict(X: Tensor): Tensor;
}

/**
 * Recursive Feature Elimination with Cross-Validation (RFECV).
 *
 * Performs RFE in a cross-validation loop to automatically determine the
 * optimal number of features. The number of features is selected based on
 * the cross-validated score across different feature counts.
 *
 * @example
 * ```ts
 * import { RFECV } from 'deepbox/preprocess';
 * import { LogisticRegression } from 'deepbox/ml';
 *
 * const rfecv = new RFECV({
 *   estimator: new LogisticRegression(),
 *   cv: 5,
 *   step: 1,
 * });
 * rfecv.fit(X, y);
 * console.log(rfecv.nFeatures); // optimal number of features
 * const Xt = rfecv.transform(X);
 * ```
 */
export class RFECV {
  private readonly estimator: ScoringEstimator;
  private readonly cv: number;
  private readonly step: number;
  private readonly minFeaturesToSelect: number;

  private ranking_: number[] | undefined;
  private mask_: boolean[] | undefined;
  private nFeaturesIn_: number | undefined;
  private nFeatures_: number | undefined;
  private cvScores_: Map<number, number> | undefined;

  constructor(options: {
    estimator: ScoringEstimator;
    cv?: number;
    step?: number;
    minFeaturesToSelect?: number;
  }) {
    this.estimator = options.estimator;
    this.cv = options.cv ?? 5;
    this.step = options.step ?? 1;
    this.minFeaturesToSelect = options.minFeaturesToSelect ?? 1;

    if (!Number.isInteger(this.cv) || this.cv < 2) {
      throw new InvalidParameterError("cv must be an integer >= 2", "cv", this.cv);
    }
    if (!Number.isInteger(this.step) || this.step < 1) {
      throw new InvalidParameterError("step must be a positive integer", "step", this.step);
    }
    if (!Number.isInteger(this.minFeaturesToSelect) || this.minFeaturesToSelect < 1) {
      throw new InvalidParameterError(
        "minFeaturesToSelect must be a positive integer",
        "minFeaturesToSelect",
        this.minFeaturesToSelect
      );
    }
  }

  fit(X: Tensor, y: Tensor): this {
    assert2D(X, "X");
    const [nSamples, nFeatures] = getShape2D(X);
    this.nFeaturesIn_ = nFeatures;

    const foldSize = Math.floor(nSamples / this.cv);
    const scores = new Map<number, number>();
    let activeIndices = Array.from({ length: nFeatures }, (_, i) => i);

    while (activeIndices.length >= this.minFeaturesToSelect) {
      const nActive = activeIndices.length;

      // Cross-validate with current feature set
      let totalScore = 0;
      for (let fold = 0; fold < this.cv; fold++) {
        const valStart = fold * foldSize;
        const valEnd = fold === this.cv - 1 ? nSamples : valStart + foldSize;

        const trainRows: number[] = [];
        const valRows: number[] = [];
        for (let i = 0; i < nSamples; i++) {
          if (i >= valStart && i < valEnd) {
            valRows.push(i);
          } else {
            trainRows.push(i);
          }
        }

        const Xsub = selectColumnsFromMask(X, activeIndices);
        const trainX = rfecvSelectRows(Xsub, trainRows);
        const trainY = rfecvSelectRows1D(y, trainRows);
        const valX = rfecvSelectRows(Xsub, valRows);
        const valY = rfecvSelectRows1D(y, valRows);

        this.estimator.fit(trainX, trainY);
        const pred = this.estimator.predict(valX);

        let correct = 0;
        const nVal = valRows.length;
        for (let i = 0; i < nVal; i++) {
          if (Number(pred.data[pred.offset + i]) === Number(valY.data[valY.offset + i])) {
            correct++;
          }
        }
        totalScore += nVal > 0 ? correct / nVal : 0;
      }

      scores.set(nActive, totalScore / this.cv);

      if (nActive <= this.minFeaturesToSelect) break;

      // Fit on full data to get importances for elimination
      const Xactive = selectColumnsFromMask(X, activeIndices);
      this.estimator.fit(Xactive, y);
      const importances = extractImportances(this.estimator, nActive);

      const nToRemove = Math.min(this.step, nActive - this.minFeaturesToSelect);
      if (nToRemove <= 0) break;

      const indexed = importances.map((imp, i) => ({ imp, i }));
      indexed.sort((a, b) => a.imp - b.imp);
      const removeSet = new Set<number>();
      for (let k = 0; k < nToRemove; k++) {
        removeSet.add(indexed[k]!.i);
      }

      activeIndices = activeIndices.filter((_, i) => !removeSet.has(i));
    }

    this.cvScores_ = scores;

    // Find the feature count with the best CV score
    let bestN = this.minFeaturesToSelect;
    let bestScore = -Infinity;
    for (const [n, score] of scores) {
      if (score > bestScore || (score === bestScore && n < bestN)) {
        bestScore = score;
        bestN = n;
      }
    }

    this.nFeatures_ = bestN;

    // Run standard RFE with nFeaturesToSelect = bestN
    activeIndices = Array.from({ length: nFeatures }, (_, i) => i);
    const ranking = new Array<number>(nFeatures).fill(0);
    let eliminationRound = 0;
    let currentX = X;

    while (activeIndices.length > bestN) {
      this.estimator.fit(currentX, y);
      const curImportances = extractImportances(this.estimator, activeIndices.length);

      const nToRemove = Math.min(this.step, activeIndices.length - bestN);

      const curIndexed = curImportances.map((imp, i) => ({ imp, i }));
      curIndexed.sort((a, b) => a.imp - b.imp);
      const removeSet = new Set<number>();
      for (let k = 0; k < nToRemove; k++) {
        removeSet.add(curIndexed[k]!.i);
      }

      eliminationRound++;
      for (const localIdx of removeSet) {
        ranking[activeIndices[localIdx]!] = eliminationRound + 1;
      }

      activeIndices = activeIndices.filter((_, i) => !removeSet.has(i));
      currentX = selectColumnsFromMask(X, activeIndices);
    }

    for (const idx of activeIndices) {
      ranking[idx] = 1;
    }

    this.ranking_ = ranking;
    this.mask_ = ranking.map((r) => r === 1);
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.mask_ || this.nFeaturesIn_ === undefined) {
      throw new NotFittedError("RFECV must be fitted before transform");
    }
    assert2D(X, "X");
    const [, nFeatures] = getShape2D(X);
    if (nFeatures !== this.nFeaturesIn_) {
      throw new InvalidParameterError(
        `Expected ${this.nFeaturesIn_} features, got ${nFeatures}`,
        "X",
        nFeatures
      );
    }
    const cols: number[] = [];
    for (let j = 0; j < nFeatures; j++) {
      if (this.mask_[j]) cols.push(j);
    }
    return selectColumnsFromMask(X, cols);
  }

  fitTransform(X: Tensor, y: Tensor): Tensor {
    this.fit(X, y);
    return this.transform(X);
  }

  getSupport(): boolean[] {
    if (!this.mask_) {
      throw new NotFittedError("RFECV must be fitted before getSupport");
    }
    return [...this.mask_];
  }

  /** Feature ranking (1 = selected, higher = eliminated earlier). */
  get ranking(): number[] {
    if (!this.ranking_) {
      throw new NotFittedError("RFECV must be fitted before accessing ranking");
    }
    return [...this.ranking_];
  }

  /** Optimal number of features selected by cross-validation. */
  get nFeatures(): number {
    if (this.nFeatures_ === undefined) {
      throw new NotFittedError("RFECV must be fitted before accessing nFeatures");
    }
    return this.nFeatures_;
  }

  /** Cross-validation scores for each feature count tested. */
  get gridScores(): Map<number, number> {
    if (!this.cvScores_) {
      throw new NotFittedError("RFECV must be fitted before accessing gridScores");
    }
    return new Map(this.cvScores_);
  }

  getParams(): Record<string, unknown> {
    return {
      cv: this.cv,
      step: this.step,
      minFeaturesToSelect: this.minFeaturesToSelect,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}

/** Select specific rows from a 2-D tensor for RFECV splits. */
function rfecvSelectRows(X: Tensor, rows: number[]): Tensor {
  const [, nCols] = getShape2D(X);
  const [rowStride, colStride] = getStrides2D(X);
  const out = new Float64Array(rows.length * nCols);
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i]!;
    for (let j = 0; j < nCols; j++) {
      out[i * nCols + j] = Number(X.data[X.offset + row * rowStride + j * colStride]);
    }
  }
  return tensor(Array.from(out)).reshape([rows.length, nCols]);
}

/** Select specific rows from a 1-D tensor for RFECV splits. */
function rfecvSelectRows1D(y: Tensor, rows: number[]): Tensor {
  const out = new Float64Array(rows.length);
  for (let i = 0; i < rows.length; i++) {
    out[i] = Number(y.data[y.offset + rows[i]!]);
  }
  return tensor(Array.from(out));
}

/** Canonical camelCase alias of {@link f_classif}. */
export const fClassif = f_classif;
/** Canonical camelCase alias of {@link f_regression}. */
export const fRegression = f_regression;
