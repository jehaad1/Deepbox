/**
 * Polynomial feature generation for ML pipelines.
 *
 * @module preprocess/polynomial
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox documentation}
 */

import {
  DeepboxError,
  DTypeError,
  InvalidParameterError,
  MemoryError,
  NotFittedError,
} from "../core";
import type { Transformer } from "../ml/base";
import { type Tensor, Tensor as TensorClass } from "../ndarray";
import { assert2D, assertNumericTensor, getShape2D, getStrides2D } from "./_internal";

/** Largest number of output columns PolynomialFeatures will generate. */
const MAX_OUTPUT_FEATURES = 2 ** 24;
/** Largest number of feature indices stored for the column definitions (4 bytes each). */
const MAX_COMBINATION_ENTRIES = 2 ** 27;
/** Largest number of output elements (rows x columns) PolynomialFeatures will allocate. */
const MAX_OUTPUT_ELEMENTS = 2 ** 31 - 1;

function assertKnownParams(
  params: Record<string, unknown>,
  known: readonly string[],
  who: string
): void {
  for (const key of Object.keys(params)) {
    if (!known.includes(key)) {
      throw new InvalidParameterError(
        `Invalid parameter '${key}' for ${who}; valid parameters are ${known.join(", ")}`,
        key,
        params[key]
      );
    }
  }
}

function definedEntries(params: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) out[key] = value;
  }
  return out;
}

function assertNumeric(X: Tensor, who: string): void {
  if (X.dtype === "string") {
    throw new DTypeError(`${who} requires numeric input`);
  }
  assertNumericTensor(X, "X");
}

/**
 * Size of the column definitions: the number of output columns and the number of stored
 * feature indices (the sum of the degrees of all columns). Both are `Infinity` once the
 * column count exceeds {@link MAX_OUTPUT_FEATURES} or the index count exceeds
 * {@link MAX_COMBINATION_ENTRIES}, so huge requests never loop or allocate for long.
 */
function sizeOutputFeatures(
  nFeatures: number,
  degree: number,
  interactionOnly: boolean,
  includeBias: boolean
): { columns: number; entries: number } {
  const tooLarge = { columns: Number.POSITIVE_INFINITY, entries: Number.POSITIVE_INFINITY };
  let columns = includeBias ? 1 : 0;
  let entries = 0;
  for (let d = 1; d <= degree; d++) {
    if (nFeatures === 0 || (interactionOnly && d > nFeatures)) break;
    // C(n, d) for interaction terms, C(n + d - 1, d) for terms with repetition. Use the
    // smaller of k and top - k so partial products never exceed the final count.
    const top = interactionOnly ? nFeatures : nFeatures + d - 1;
    const k = Math.min(d, top - d);
    let c = 1;
    for (let i = 0; i < k; i++) {
      c = (c * (top - i)) / (i + 1);
      if (c > MAX_OUTPUT_FEATURES) return tooLarge;
    }
    const count = Math.round(c);
    columns += count;
    entries += count * d;
    if (columns > MAX_OUTPUT_FEATURES || entries > MAX_COMBINATION_ENTRIES) return tooLarge;
  }
  return { columns, entries };
}

/** Throw a MemoryError when the expansion would be too large to describe or to allocate. */
function assertOutputSize(
  nFeatures: number,
  degree: number,
  interactionOnly: boolean,
  includeBias: boolean
): void {
  const size = sizeOutputFeatures(nFeatures, degree, interactionOnly, includeBias);
  if (!Number.isFinite(size.columns)) {
    throw new MemoryError(
      `PolynomialFeatures with ${nFeatures} features and degree ${degree} ` +
        `would generate more than ${MAX_OUTPUT_FEATURES} output features or more than ` +
        `${MAX_COMBINATION_ENTRIES} feature index entries`
    );
  }
}

/**
 * Column definitions in a flat encoding: column `c` multiplies the input features
 * `indices[offsets[c]] ... indices[offsets[c + 1] - 1]`. Two typed arrays replace one
 * `number[]` per column, which keeps the memory use near 4 bytes per entry.
 */
type ColumnDefinitions = {
  readonly offsets: Uint32Array;
  readonly indices: Int32Array;
  readonly columns: number;
};

/** Build the column definitions in scikit-learn order (by degree, then lexicographic). */
function buildColumnDefinitions(
  nFeatures: number,
  degree: number,
  interactionOnly: boolean,
  includeBias: boolean
): ColumnDefinitions {
  const size = sizeOutputFeatures(nFeatures, degree, interactionOnly, includeBias);
  const columns = size.columns;
  const offsets = new Uint32Array(columns + 1);
  const indices = new Int32Array(size.entries);
  let col = includeBias ? 1 : 0;
  let pos = 0;
  const idx = new Int32Array(degree);
  for (let d = 1; d <= degree; d++) {
    if (nFeatures === 0 || (interactionOnly && d > nFeatures)) break;
    // First combination: all zeros with repetition, 0..d-1 without.
    for (let t = 0; t < d; t++) idx[t] = interactionOnly ? t : 0;
    for (;;) {
      offsets[col++] = pos;
      for (let t = 0; t < d; t++) indices[pos++] = idx[t] as number;
      // Advance to the next combination in lexicographic order: find the rightmost
      // position that has not reached its largest value yet.
      let t = d - 1;
      while (
        t >= 0 &&
        (idx[t] as number) >= (interactionOnly ? nFeatures - d + t : nFeatures - 1)
      ) {
        t--;
      }
      if (t < 0) break;
      idx[t] = (idx[t] as number) + 1;
      for (let u = t + 1; u < d; u++) {
        idx[u] = interactionOnly ? (idx[u - 1] as number) + 1 : (idx[t] as number);
      }
    }
  }
  if (col !== columns || pos !== size.entries) {
    throw new DeepboxError("PolynomialFeatures internal error: column count mismatch");
  }
  offsets[columns] = pos;
  return { offsets, indices, columns };
}

/**
 * Generate polynomial and interaction features.
 *
 * Generates a new feature matrix consisting of all polynomial combinations
 * of the features with degree less than or equal to the specified degree.
 * The column order matches scikit-learn: by degree, and within a degree in
 * lexicographic order of the feature indices.
 *
 * For example, if input has features [a, b] and degree=2:
 * - interactionOnly=false: [1, a, b, a², ab, b²]
 * - interactionOnly=true:  [1, a, b, ab]
 *
 * The output is always float64. The number of output columns grows quickly with
 * the degree and the number of features; very large requests throw a
 * `MemoryError` instead of exhausting memory.
 *
 * @example
 * ```ts
 * import { PolynomialFeatures } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const poly = new PolynomialFeatures({ degree: 2 });
 * const X = tensor([[1, 2], [3, 4]]);
 * const Xpoly = poly.fitTransform(X);
 * // Xpoly has columns: [1, x1, x2, x1², x1*x2, x2²]
 * ```
 */
export class PolynomialFeatures implements Transformer {
  private degree: number;
  private interactionOnly: boolean;
  private includeBias: boolean;
  private nFeaturesIn = 0;
  private fitted = false;
  private combosKey: string | undefined;
  private combos: ColumnDefinitions | undefined;

  /**
   * @param options.degree - Maximum degree of the polynomial terms (default 2, must be >= 0)
   * @param options.interactionOnly - Only products of distinct features (default false)
   * @param options.includeBias - Include the constant column of ones (default true)
   */
  constructor(
    options: {
      readonly degree?: number;
      readonly interactionOnly?: boolean;
      readonly includeBias?: boolean;
    } = {}
  ) {
    this.degree = options.degree ?? 2;
    this.interactionOnly = options.interactionOnly ?? false;
    this.includeBias = options.includeBias ?? true;

    if (!Number.isInteger(this.degree) || this.degree < 0) {
      throw new InvalidParameterError(
        "degree must be a non-negative integer",
        "degree",
        this.degree
      );
    }
    if (typeof this.interactionOnly !== "boolean") {
      throw new InvalidParameterError(
        "interactionOnly must be a boolean",
        "interactionOnly",
        this.interactionOnly
      );
    }
    if (typeof this.includeBias !== "boolean") {
      throw new InvalidParameterError(
        "includeBias must be a boolean",
        "includeBias",
        this.includeBias
      );
    }
  }

  /**
   * Record the number of input features.
   *
   * @throws {ShapeError} If X is not 2D
   * @throws {MemoryError} If the requested expansion is too large
   */
  fit(X: Tensor): this {
    assert2D(X, "PolynomialFeatures.fit");
    assertNumeric(X, "PolynomialFeatures");
    const [, nFeatures] = getShape2D(X);
    // Check the size first so that a failed fit leaves the previous state untouched.
    assertOutputSize(nFeatures, this.degree, this.interactionOnly, this.includeBias);
    this.nFeaturesIn = nFeatures;
    this.fitted = true;
    this.combosKey = undefined;
    this.combos = undefined;
    return this;
  }

  /**
   * Expand X into polynomial features.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Float64 tensor of shape (n_samples, n_output_features)
   * @throws {MemoryError} If the output would be too large
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("PolynomialFeatures must be fitted before transform");
    }
    assert2D(X, "PolynomialFeatures.transform");
    assertNumeric(X, "PolynomialFeatures");
    const [nSamples, nFeatures] = getShape2D(X);

    if (nFeatures !== this.nFeaturesIn) {
      throw new InvalidParameterError(
        `Expected ${this.nFeaturesIn} features, got ${nFeatures}`,
        "X",
        nFeatures
      );
    }

    const combos = this.getCombinations();
    const nOutputFeatures = combos.columns;
    const { offsets, indices } = combos;
    if (nSamples * nOutputFeatures > MAX_OUTPUT_ELEMENTS) {
      throw new MemoryError(
        `PolynomialFeatures output would have ${nSamples} x ${nOutputFeatures} elements, ` +
          `more than the supported ${MAX_OUTPUT_ELEMENTS}`
      );
    }

    const result = new Float64Array(nSamples * nOutputFeatures);
    const [s0, s1] = getStrides2D(X);

    // Densify the input row-major once so the product loop reads monomorphic
    // Float64 values and the output wraps its buffer without another copy.
    const src = X.data as ArrayLike<number | bigint>;
    const offset = X.offset;
    const dense = new Float64Array(nSamples * nFeatures);
    let dp = 0;
    for (let i = 0; i < nSamples; i++) {
      const rowOffset = offset + i * s0;
      for (let j = 0; j < nFeatures; j++) dense[dp++] = Number(src[rowOffset + j * s1]);
    }

    for (let i = 0; i < nSamples; i++) {
      const denseBase = i * nFeatures;
      const outBase = i * nOutputFeatures;
      for (let c = 0; c < nOutputFeatures; c++) {
        let val = 1;
        const end = offsets[c + 1] as number;
        for (let t = offsets[c] as number; t < end; t++) {
          val *= dense[denseBase + (indices[t] as number)] as number;
        }
        result[outBase + c] = val;
      }
    }

    return TensorClass.fromTypedArray({
      data: result,
      shape: [nSamples, nOutputFeatures],
      dtype: "float64",
      device: X.device,
    });
  }

  fitTransform(X: Tensor): Tensor {
    this.fit(X);
    return this.transform(X);
  }

  getParams(): Record<string, unknown> {
    return {
      degree: this.degree,
      interactionOnly: this.interactionOnly,
      includeBias: this.includeBias,
    };
  }

  /**
   * Update parameters. The fitted input width is kept, so the new setting
   * applies to the next `transform`.
   *
   * @throws {InvalidParameterError} On an unknown parameter or invalid value
   * @throws {MemoryError} If the new setting would expand the fitted input too far
   */
  setParams(params: Record<string, unknown>): this {
    assertKnownParams(params, ["degree", "interactionOnly", "includeBias"], "PolynomialFeatures");
    const next = new PolynomialFeatures({
      ...this.getParams(),
      ...definedEntries(params),
    } as ConstructorParameters<typeof PolynomialFeatures>[0]);
    if (this.fitted) {
      assertOutputSize(this.nFeaturesIn, next.degree, next.interactionOnly, next.includeBias);
    }
    this.degree = next.degree;
    this.interactionOnly = next.interactionOnly;
    this.includeBias = next.includeBias;
    this.combosKey = undefined;
    this.combos = undefined;
    return this;
  }

  /** Create an unfitted copy with the same parameters. */
  clone(): PolynomialFeatures {
    return new PolynomialFeatures(
      this.getParams() as ConstructorParameters<typeof PolynomialFeatures>[0]
    );
  }

  /** Number of output features after transformation */
  get nOutputFeatures(): number {
    if (!this.fitted) {
      throw new NotFittedError("PolynomialFeatures must be fitted first");
    }
    return this.getCombinations().columns;
  }

  /**
   * Exponent of each input feature in each output column, shape
   * `[n_output_features][n_input_features]`.
   */
  get powers(): number[][] {
    if (!this.fitted) {
      throw new NotFittedError("PolynomialFeatures must be fitted first");
    }
    const { offsets, indices, columns } = this.getCombinations();
    const rows = new Array<number[]>(columns);
    for (let c = 0; c < columns; c++) {
      const row = new Array<number>(this.nFeaturesIn).fill(0);
      const end = offsets[c + 1] as number;
      for (let t = offsets[c] as number; t < end; t++) {
        const f = indices[t] as number;
        row[f] = (row[f] as number) + 1;
      }
      rows[c] = row;
    }
    return rows;
  }

  /**
   * Names of the output columns, in the same format as scikit-learn
   * (`1`, `x0`, `x0^2`, `x0 x1`).
   *
   * @param inputFeatures - Names of the input features (default `x0`, `x1`, ...)
   */
  getFeatureNamesOut(inputFeatures?: readonly string[]): string[] {
    if (!this.fitted) {
      throw new NotFittedError("PolynomialFeatures must be fitted first");
    }
    if (inputFeatures !== undefined && inputFeatures.length !== this.nFeaturesIn) {
      throw new InvalidParameterError(
        `inputFeatures must have ${this.nFeaturesIn} names, got ${inputFeatures.length}`,
        "inputFeatures",
        inputFeatures.length
      );
    }
    const names = inputFeatures ?? Array.from({ length: this.nFeaturesIn }, (_, i) => `x${i}`);
    return this.powers.map((row) => {
      const parts: string[] = [];
      for (let f = 0; f < row.length; f++) {
        const p = row[f] as number;
        if (p === 1) parts.push(names[f] as string);
        else if (p > 1) parts.push(`${names[f]}^${p}`);
      }
      return parts.length === 0 ? "1" : parts.join(" ");
    });
  }

  /** Column definitions for the current parameters (cached until they change). */
  private getCombinations(): ColumnDefinitions {
    const key = `${this.nFeaturesIn}|${this.degree}|${this.interactionOnly}|${this.includeBias}`;
    if (this.combos && this.combosKey === key) return this.combos;

    assertOutputSize(this.nFeaturesIn, this.degree, this.interactionOnly, this.includeBias);

    const combos = buildColumnDefinitions(
      this.nFeaturesIn,
      this.degree,
      this.interactionOnly,
      this.includeBias
    );
    this.combos = combos;
    this.combosKey = key;
    return combos;
  }
}

/**
 * Binarize data (set feature values to 0 or 1) according to a threshold.
 *
 * Values greater than the threshold map to 1, while values less than or
 * equal to the threshold (and NaN) map to 0. The output is always float64.
 *
 * @example
 * ```ts
 * import { Binarizer } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const binarizer = new Binarizer({ threshold: 1.5 });
 * const X = tensor([[1, 2], [3, 0.5]]);
 * const Xb = binarizer.fitTransform(X);
 * // Xb = [[0, 1], [1, 0]]
 * ```
 */
export class Binarizer implements Transformer {
  private threshold: number;
  private nFeaturesIn: number | undefined;
  private fitted = false;

  /**
   * @param options.threshold - Values strictly above it become 1 (default 0)
   */
  constructor(options: { readonly threshold?: number } = {}) {
    this.threshold = options.threshold ?? 0;
    if (typeof this.threshold !== "number" || Number.isNaN(this.threshold)) {
      throw new InvalidParameterError("threshold must be a number", "threshold", this.threshold);
    }
  }

  /**
   * Record the number of input features (the transformer is otherwise stateless).
   *
   * @throws {ShapeError} If X is not 2D
   */
  fit(X: Tensor): this {
    assert2D(X, "Binarizer.fit");
    assertNumeric(X, "Binarizer");
    this.nFeaturesIn = getShape2D(X)[1];
    this.fitted = true;
    return this;
  }

  /**
   * Binarize X.
   *
   * @param X - Data of shape (n_samples, n_features)
   * @returns Float64 tensor of zeros and ones with the same shape
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("Binarizer must be fitted before transform");
    }
    assert2D(X, "Binarizer.transform");
    assertNumeric(X, "Binarizer");
    const [nSamples, nFeatures] = getShape2D(X);
    if (this.nFeaturesIn !== undefined && nFeatures !== this.nFeaturesIn) {
      throw new InvalidParameterError(
        `Expected ${this.nFeaturesIn} features, got ${nFeatures}`,
        "X",
        nFeatures
      );
    }
    const result = new Float64Array(nSamples * nFeatures);
    const [s0, s1] = getStrides2D(X);

    const src = X.data as ArrayLike<number | bigint>;
    const offset = X.offset;
    const threshold = this.threshold;
    let pos = 0;
    for (let i = 0; i < nSamples; i++) {
      const rowBase = offset + i * s0;
      for (let j = 0; j < nFeatures; j++) {
        result[pos++] = Number(src[rowBase + j * s1]) > threshold ? 1 : 0;
      }
    }

    return TensorClass.fromTypedArray({
      data: result,
      shape: [nSamples, nFeatures],
      dtype: "float64",
      device: X.device,
    });
  }

  fitTransform(X: Tensor): Tensor {
    this.fit(X);
    return this.transform(X);
  }

  getParams(): Record<string, unknown> {
    return { threshold: this.threshold };
  }

  /**
   * Update parameters.
   *
   * @throws {InvalidParameterError} On an unknown parameter or invalid value
   */
  setParams(params: Record<string, unknown>): this {
    assertKnownParams(params, ["threshold"], "Binarizer");
    const next = new Binarizer({
      ...this.getParams(),
      ...definedEntries(params),
    } as { threshold?: number });
    this.threshold = next.threshold;
    return this;
  }

  /** Create an unfitted copy with the same parameters. */
  clone(): Binarizer {
    return new Binarizer({ threshold: this.threshold });
  }
}

/**
 * Constructs a transformer from an arbitrary callable.
 *
 * Useful for wrapping arbitrary functions in a Pipeline-compatible
 * transformer without subclassing. With no `func` the transform is the
 * identity and returns the input tensor itself. `inverseTransform` applies
 * `inverseFunc`, or is the identity when there is none; it does not require
 * `fit`.
 *
 * @example
 * ```ts
 * import { FunctionTransformer } from 'deepbox/preprocess';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const log1p = new FunctionTransformer({
 *   func: (X) => {
 *     // apply log1p element-wise
 *     const data = [];
 *     for (let i = 0; i < X.size; i++) data.push(Math.log1p(Number(X.data[X.offset + i])));
 *     return tensor(data).reshape(X.shape);
 *   },
 * });
 * ```
 */
export class FunctionTransformer implements Transformer {
  private func: ((X: Tensor) => Tensor) | undefined;
  private inverseFunc: ((X: Tensor) => Tensor) | undefined;
  private fitted = false;

  /**
   * @param options.func - Function applied by `transform` (default: identity)
   * @param options.inverseFunc - Function applied by `inverseTransform` (default: identity)
   */
  constructor(
    options: {
      readonly func?: (X: Tensor) => Tensor;
      readonly inverseFunc?: (X: Tensor) => Tensor;
    } = {}
  ) {
    this.func = options.func;
    this.inverseFunc = options.inverseFunc;
    if (this.func !== undefined && typeof this.func !== "function") {
      throw new InvalidParameterError("func must be a function", "func", this.func);
    }
    if (this.inverseFunc !== undefined && typeof this.inverseFunc !== "function") {
      throw new InvalidParameterError(
        "inverseFunc must be a function",
        "inverseFunc",
        this.inverseFunc
      );
    }
  }

  fit(_X: Tensor): this {
    this.fitted = true;
    return this;
  }

  /**
   * Apply `func` to X.
   *
   * @throws {InvalidParameterError} If `func` does not return a Tensor
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("FunctionTransformer must be fitted before transform");
    }
    return FunctionTransformer.apply(this.func, X, "func");
  }

  fitTransform(X: Tensor): Tensor {
    this.fit(X);
    return this.transform(X);
  }

  /**
   * Apply `inverseFunc` to X (identity when none was given).
   *
   * @throws {InvalidParameterError} If `inverseFunc` does not return a Tensor
   */
  inverseTransform(X: Tensor): Tensor {
    return FunctionTransformer.apply(this.inverseFunc, X, "inverseFunc");
  }

  getParams(): Record<string, unknown> {
    return { func: this.func, inverseFunc: this.inverseFunc };
  }

  /**
   * Update parameters.
   *
   * @throws {InvalidParameterError} On an unknown parameter or invalid value
   */
  setParams(params: Record<string, unknown>): this {
    assertKnownParams(params, ["func", "inverseFunc"], "FunctionTransformer");
    const next = new FunctionTransformer({
      ...definedEntries(this.getParams()),
      ...definedEntries(params),
    } as ConstructorParameters<typeof FunctionTransformer>[0]);
    this.func = next.func;
    this.inverseFunc = next.inverseFunc;
    return this;
  }

  /** Create an unfitted copy with the same parameters. */
  clone(): FunctionTransformer {
    return new FunctionTransformer(
      definedEntries(this.getParams()) as ConstructorParameters<typeof FunctionTransformer>[0]
    );
  }

  private static apply(fn: ((X: Tensor) => Tensor) | undefined, X: Tensor, name: string): Tensor {
    if (!fn) return X;
    const out = fn(X);
    if (!(out instanceof TensorClass)) {
      throw new InvalidParameterError(`${name} must return a Tensor`, name, typeof out);
    }
    return out;
  }
}
