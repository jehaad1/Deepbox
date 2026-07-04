/**
 * Pipeline: chain transformers and a final estimator.
 *
 * @module ml/pipeline
 * @see {@link https://deepbox.dev/docs/ml-model-selection | Deepbox Pipeline}
 */

import { InvalidParameterError, NotFittedError } from "../core";
import { tensor as createTensor, type Tensor } from "../ndarray";
import type { Classifier, Estimator, Transformer } from "./base";

type Step = {
  readonly name: string;
  readonly estimator: Estimator | Transformer;
};

function isTransformer(e: Estimator | Transformer): e is Transformer {
  return typeof (e as Transformer).transform === "function";
}

function hasPredict(e: Estimator): e is Estimator & { predict(X: Tensor): Tensor } {
  return typeof (e as Classifier).predict === "function";
}

function hasScore(e: Estimator): e is Estimator & { score(X: Tensor, y: Tensor): number } {
  return typeof (e as Classifier).score === "function";
}

function hasPredictProba(e: Estimator): e is Estimator & { predictProba(X: Tensor): Tensor } {
  return typeof (e as Classifier).predictProba === "function";
}

/**
 * Pipeline of transforms with a final estimator.
 *
 * Sequentially applies a list of transforms and a final estimator.
 * Intermediate steps must implement `transform()`. The final step
 * can be any estimator (classifier, regressor, transformer, etc.).
 *
 * @example
 * ```ts
 * import { Pipeline } from 'deepbox/ml';
 * import { StandardScaler } from 'deepbox/preprocess';
 * import { LogisticRegression } from 'deepbox/ml';
 *
 * const pipe = new Pipeline([
 *   ['scaler', new StandardScaler()],
 *   ['clf', new LogisticRegression()],
 * ]);
 * pipe.fit(X_train, y_train);
 * const pred = pipe.predict(X_test);
 * ```
 */
export class Pipeline {
  private readonly steps: Step[];
  private fitted = false;

  /**
   * @param steps - Array of [name, estimator] tuples. All but the last must be transformers.
   */
  constructor(steps: ReadonlyArray<readonly [string, Estimator | Transformer]>) {
    if (steps.length < 1) {
      throw new InvalidParameterError("Pipeline requires at least one step", "steps", steps);
    }

    // Validate intermediate steps are transformers
    for (let i = 0; i < steps.length - 1; i++) {
      const [name, est] = steps[i]!;
      if (!isTransformer(est)) {
        throw new InvalidParameterError(
          `Intermediate step '${name}' must implement transform()`,
          "steps",
          name
        );
      }
    }

    // Check for duplicate names
    const names = new Set<string>();
    for (const [name] of steps) {
      if (names.has(name)) {
        throw new InvalidParameterError(`Duplicate step name: '${name}'`, "steps", name);
      }
      names.add(name);
    }

    this.steps = steps.map(([name, estimator]) => ({ name, estimator }));
  }

  /**
   * Fit all steps. Transforms are fit_transformed, final step is fit.
   */
  fit(X: Tensor, y?: Tensor): this {
    let Xt = X;

    // Fit and transform intermediate steps
    for (let i = 0; i < this.steps.length - 1; i++) {
      const step = this.steps[i]!;
      const transformer = step.estimator as Transformer;
      if (typeof transformer.fitTransform === "function") {
        Xt = transformer.fitTransform(Xt, y);
      } else {
        transformer.fit(Xt, y);
        Xt = transformer.transform(Xt);
      }
    }

    // Fit final step
    const finalStep = this.steps[this.steps.length - 1]!;
    finalStep.estimator.fit(Xt, y);

    this.fitted = true;
    return this;
  }

  /**
   * Transform data through all steps (final step must be a transformer).
   */
  transform(X: Tensor): Tensor {
    this.checkFitted();
    let Xt = X;
    for (const step of this.steps) {
      if (!isTransformer(step.estimator)) {
        throw new InvalidParameterError(
          `Step '${step.name}' does not implement transform()`,
          "steps",
          step.name
        );
      }
      Xt = (step.estimator as Transformer).transform(Xt);
    }
    return Xt;
  }

  /**
   * Fit and transform (final step must be a transformer).
   */
  fitTransform(X: Tensor, y?: Tensor): Tensor {
    this.fit(X, y);
    const finalStep = this.steps[this.steps.length - 1]!;
    if (!isTransformer(finalStep.estimator)) {
      throw new InvalidParameterError(
        "Final step must implement transform() for fitTransform()",
        "steps",
        finalStep.name
      );
    }
    // Transform through all intermediate + final
    let Xt = X;
    for (const step of this.steps) {
      Xt = (step.estimator as Transformer).transform(Xt);
    }
    return Xt;
  }

  /**
   * Transform data through intermediate steps, then predict with final step.
   */
  predict(X: Tensor): Tensor {
    this.checkFitted();
    const Xt = this.transformIntermediate(X);
    const final = this.steps[this.steps.length - 1]!.estimator;
    if (!hasPredict(final)) {
      throw new InvalidParameterError(
        "Final step does not implement predict()",
        "steps",
        this.steps[this.steps.length - 1]!.name
      );
    }
    return final.predict(Xt);
  }

  /**
   * Transform data through intermediate steps, then predict probabilities.
   */
  predictProba(X: Tensor): Tensor {
    this.checkFitted();
    const Xt = this.transformIntermediate(X);
    const final = this.steps[this.steps.length - 1]!.estimator;
    if (!hasPredictProba(final)) {
      throw new InvalidParameterError(
        "Final step does not implement predictProba()",
        "steps",
        this.steps[this.steps.length - 1]!.name
      );
    }
    return final.predictProba(Xt);
  }

  /**
   * Transform data through intermediate steps, then score with final step.
   */
  score(X: Tensor, y: Tensor): number {
    this.checkFitted();
    const Xt = this.transformIntermediate(X);
    const final = this.steps[this.steps.length - 1]!.estimator;
    if (!hasScore(final)) {
      throw new InvalidParameterError(
        "Final step does not implement score()",
        "steps",
        this.steps[this.steps.length - 1]!.name
      );
    }
    return final.score(Xt, y);
  }

  /**
   * Get a named step by name.
   */
  getStep(name: string): Estimator | Transformer {
    const step = this.steps.find((s) => s.name === name);
    if (!step) {
      throw new InvalidParameterError(`No step named '${name}'`, "name", name);
    }
    return step.estimator;
  }

  /**
   * Get all step names.
   */
  get stepNames(): string[] {
    return this.steps.map((s) => s.name);
  }

  getParams(): Record<string, unknown> {
    const params: Record<string, unknown> = {};
    for (const step of this.steps) {
      params[step.name] = step.estimator.getParams();
    }
    return params;
  }

  setParams(_params: Record<string, unknown>): this {
    throw new InvalidParameterError(
      "Pipeline does not support setParams after construction",
      "params",
      _params
    );
  }

  /**
   * Create a fresh, unfitted clone of this pipeline with freshly cloned steps.
   *
   * Each step estimator is reconstructed (via its own `clone()` if available,
   * otherwise from `getParams()`), so the returned pipeline shares no fitted
   * state with the original. This lets a `Pipeline` be used directly with
   * `cross_validate` / `cross_val_score`, whose flat `getParams()` shape cannot
   * be passed back into the `Pipeline` constructor.
   */
  clone(): Pipeline {
    const clonedSteps = this.steps.map(({ name, estimator }) => {
      const withClone = estimator as { clone?: () => Estimator | Transformer };
      let cloned: Estimator | Transformer;
      if (typeof withClone.clone === "function") {
        cloned = withClone.clone();
      } else {
        const EstimatorClass = estimator.constructor as new (
          p: Record<string, unknown>
        ) => Estimator | Transformer;
        cloned = new EstimatorClass(estimator.getParams());
      }
      return [name, cloned] as readonly [string, Estimator | Transformer];
    });
    return new Pipeline(clonedSteps);
  }

  private transformIntermediate(X: Tensor): Tensor {
    let Xt = X;
    for (let i = 0; i < this.steps.length - 1; i++) {
      Xt = (this.steps[i]!.estimator as Transformer).transform(Xt);
    }
    return Xt;
  }

  private checkFitted(): void {
    if (!this.fitted) {
      throw new NotFittedError("Pipeline must be fitted before use");
    }
  }
}

/**
 * Convenience factory for creating a Pipeline with auto-generated step names.
 *
 * @example
 * ```ts
 * const pipe = makePipeline(new StandardScaler(), new LogisticRegression());
 * pipe.fit(X, y);
 * ```
 */
export function makePipeline(...estimators: Array<Estimator | Transformer>): Pipeline {
  if (estimators.length === 0) {
    throw new InvalidParameterError(
      "makePipeline requires at least one estimator",
      "estimators",
      estimators
    );
  }

  const steps: Array<[string, Estimator | Transformer]> = [];
  const nameCounts = new Map<string, number>();

  for (const est of estimators) {
    // Derive name from constructor
    let name = est.constructor.name.toLowerCase();
    if (!name || name === "object") name = "step";

    const count = nameCounts.get(name) ?? 0;
    nameCounts.set(name, count + 1);
    const stepName = count === 0 ? name : `${name}_${count}`;
    steps.push([stepName, est]);
  }

  return new Pipeline(steps);
}

/**
 * Concatenate results of multiple transformer objects.
 *
 * Applies each transformer independently and concatenates the results column-wise.
 * Useful for combining multiple feature extraction mechanisms into a single transformer.
 *
 * @example
 * ```ts
 * import { FeatureUnion } from 'deepbox/ml';
 * import { PolynomialFeatures, Binarizer } from 'deepbox/preprocess';
 *
 * const union = new FeatureUnion([
 *   ['poly', new PolynomialFeatures({ degree: 2 })],
 *   ['bin', new Binarizer({ threshold: 0.5 })],
 * ]);
 * const Xt = union.fitTransform(X);
 * ```
 */
export class FeatureUnion {
  private readonly transformers: Array<{
    name: string;
    transformer: Transformer;
  }>;
  private fitted = false;

  constructor(transformers: ReadonlyArray<readonly [string, Transformer]>) {
    if (transformers.length === 0) {
      throw new InvalidParameterError(
        "FeatureUnion requires at least one transformer",
        "transformers",
        transformers
      );
    }

    const names = new Set<string>();
    for (const [name] of transformers) {
      if (names.has(name)) {
        throw new InvalidParameterError(
          `Duplicate transformer name: '${name}'`,
          "transformers",
          name
        );
      }
      names.add(name);
    }

    this.transformers = transformers.map(([name, transformer]) => ({
      name,
      transformer,
    }));
  }

  fit(X: Tensor, y?: Tensor): this {
    for (const { transformer } of this.transformers) {
      transformer.fit(X, y);
    }
    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("FeatureUnion must be fitted before transform");
    }

    const results: Tensor[] = [];
    for (const { transformer } of this.transformers) {
      results.push(transformer.transform(X));
    }

    return this.hstack(results);
  }

  fitTransform(X: Tensor, y?: Tensor): Tensor {
    this.fit(X, y);
    return this.transform(X);
  }

  getParams(): Record<string, unknown> {
    const params: Record<string, unknown> = {};
    for (const { name, transformer } of this.transformers) {
      params[name] = transformer.getParams();
    }
    return params;
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }

  get transformerNames(): string[] {
    return this.transformers.map((t) => t.name);
  }

  private hstack(tensors: Tensor[]): Tensor {
    return hstackTensors(tensors);
  }
}

/**
 * ColumnTransformer applies transformers to specific column subsets of the data.
 *
 * Each transformer is applied to a specified subset of columns, and the results
 * are concatenated horizontally. This is useful when different feature types
 * require different preprocessing.
 *
 * @example
 * ```ts
 * import { ColumnTransformer } from 'deepbox/ml';
 * import { StandardScaler, OneHotEncoder } from 'deepbox/preprocess';
 *
 * const ct = new ColumnTransformer([
 *   ['num', new StandardScaler(), [0, 1, 2]],
 *   ['cat', new OneHotEncoder(), [3, 4]],
 * ]);
 * const Xt = ct.fitTransform(X);
 * ```
 */
export class ColumnTransformer {
  private readonly transformerSpecs: Array<{
    name: string;
    transformer: Transformer | "passthrough" | "drop";
    columns: number[];
  }>;
  private readonly remainder: "drop" | "passthrough";
  private remainderCols: number[] = [];
  private fitted = false;

  /**
   * @param transformers - Array of [name, transformer, columns] tuples.
   *   transformer can be a Transformer instance, "passthrough", or "drop".
   *   columns is an array of column indices.
   * @param options - Additional options.
   *   remainder: "drop" (default) or "passthrough" for columns not specified.
   */
  constructor(
    transformers: ReadonlyArray<readonly [string, Transformer | "passthrough" | "drop", number[]]>,
    options: { readonly remainder?: "drop" | "passthrough" } = {}
  ) {
    if (transformers.length === 0) {
      throw new InvalidParameterError(
        "ColumnTransformer requires at least one transformer spec",
        "transformers",
        transformers
      );
    }

    const names = new Set<string>();
    for (const [name] of transformers) {
      if (names.has(name)) {
        throw new InvalidParameterError(
          `Duplicate transformer name: '${name}'`,
          "transformers",
          name
        );
      }
      names.add(name);
    }

    this.transformerSpecs = transformers.map(([name, transformer, columns]) => ({
      name,
      transformer,
      columns: [...columns],
    }));
    this.remainder = options.remainder ?? "drop";
  }

  fit(X: Tensor, y?: Tensor): this {
    const nFeats = X.ndim === 1 ? 1 : (X.shape[1] ?? 1);
    this.computeRemainderCols(nFeats);

    for (const spec of this.transformerSpecs) {
      if (spec.transformer === "passthrough" || spec.transformer === "drop") {
        continue;
      }
      const subX = this.selectColumns(X, spec.columns);
      spec.transformer.fit(subX, y);
    }

    this.fitted = true;
    return this;
  }

  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("ColumnTransformer must be fitted before transform");
    }

    const parts: Tensor[] = [];
    for (const spec of this.transformerSpecs) {
      if (spec.transformer === "drop") continue;
      const subX = this.selectColumns(X, spec.columns);
      if (spec.transformer === "passthrough") {
        parts.push(subX);
      } else {
        parts.push(spec.transformer.transform(subX));
      }
    }

    if (this.remainder === "passthrough" && this.remainderCols.length > 0) {
      parts.push(this.selectColumns(X, this.remainderCols));
    }

    if (parts.length === 0) {
      const nSamples = X.shape[0] ?? 0;
      return createTensor(new Array(nSamples).fill(0)).reshape([nSamples, 0]);
    }

    return hstackTensors(parts);
  }

  fitTransform(X: Tensor, y?: Tensor): Tensor {
    this.fit(X, y);
    return this.transform(X);
  }

  getParams(): Record<string, unknown> {
    const params: Record<string, unknown> = { remainder: this.remainder };
    for (const spec of this.transformerSpecs) {
      if (typeof spec.transformer === "string") {
        params[spec.name] = spec.transformer;
      } else {
        params[spec.name] = spec.transformer.getParams();
      }
    }
    return params;
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }

  get transformerNames(): string[] {
    return this.transformerSpecs.map((s) => s.name);
  }

  private computeRemainderCols(nFeats: number): void {
    const used = new Set<number>();
    for (const spec of this.transformerSpecs) {
      for (const c of spec.columns) {
        used.add(c);
      }
    }
    this.remainderCols = [];
    for (let i = 0; i < nFeats; i++) {
      if (!used.has(i)) {
        this.remainderCols.push(i);
      }
    }
  }

  private selectColumns(X: Tensor, cols: number[]): Tensor {
    const nSamples = X.shape[0] ?? 0;
    const nCols = cols.length;
    if (nCols === 0) {
      return createTensor(new Array(nSamples).fill(0)).reshape([nSamples, 0]);
    }

    const nFeats = X.ndim === 1 ? 1 : (X.shape[1] ?? 1);
    const resultData: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      for (const c of cols) {
        if (X.ndim === 1) {
          resultData.push(Number(X.data[X.offset + i]));
        } else {
          resultData.push(Number(X.data[X.offset + i * nFeats + c]));
        }
      }
    }

    return createTensor(resultData).reshape([nSamples, nCols]);
  }
}

function hstackTensors(tensors: Tensor[]): Tensor {
  if (tensors.length === 0) {
    throw new InvalidParameterError("No tensors to concatenate", "tensors", tensors);
  }

  const first = tensors[0]!;
  const nSamples = first.shape[0] ?? 0;

  let totalFeatures = 0;
  const featureCounts: number[] = [];
  for (const t of tensors) {
    const tSamples = t.shape[0] ?? 0;
    if (tSamples !== nSamples) {
      throw new InvalidParameterError(
        "All transformers must produce same number of rows",
        "transformers",
        tSamples
      );
    }
    const nFeats = t.ndim === 1 ? 1 : (t.shape[1] ?? 1);
    featureCounts.push(nFeats);
    totalFeatures += nFeats;
  }

  const resultData: number[] = [];
  for (let i = 0; i < nSamples; i++) {
    for (let ti = 0; ti < tensors.length; ti++) {
      const t = tensors[ti]!;
      const nFeats = featureCounts[ti]!;
      if (t.ndim === 1) {
        resultData.push(Number(t.data[t.offset + i]));
      } else {
        for (let j = 0; j < nFeats; j++) {
          resultData.push(Number(t.data[t.offset + i * nFeats + j]));
        }
      }
    }
  }

  return createTensor(resultData).reshape([nSamples, totalFeatures]);
}
