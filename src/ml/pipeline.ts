/**
 * Pipeline: chain transformers and a final estimator.
 *
 * @module ml/pipeline
 * @see {@link https://deepbox.dev/docs/ml-model-selection | Deepbox Pipeline}
 */

import { InvalidParameterError, NotFittedError, ShapeError } from "../core";
import { type Tensor, tensor } from "../ndarray";
import { cloneEstimator } from "./_internal";
import { toFloat64View } from "./_validation";
import {
  type Classifier,
  type Estimator,
  type EstimatorTags,
  getEstimatorTags,
  type Transformer,
} from "./base";

type Step = {
  readonly name: string;
  readonly estimator: Estimator | Transformer;
};

/** Separator between a step name and one of its parameters, as in `scaler__withMean`. */
const PARAM_SEPARATOR = "__";

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

function hasDecisionFunction(
  e: Estimator
): e is Estimator & { decisionFunction(X: Tensor): Tensor } {
  return typeof (e as { decisionFunction?: unknown }).decisionFunction === "function";
}

function hasFitPredict(
  e: Estimator
): e is Estimator & { fitPredict(X: Tensor, y?: Tensor): Tensor } {
  return typeof (e as { fitPredict?: unknown }).fitPredict === "function";
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function validateName(name: unknown, owner: string): asserts name is string {
  if (typeof name !== "string" || name.length === 0) {
    throw new InvalidParameterError(`${owner} names must be non-empty strings`, "name", name);
  }
  if (name.includes(PARAM_SEPARATOR)) {
    throw new InvalidParameterError(
      `${owner} name '${name}' must not contain '${PARAM_SEPARATOR}'`,
      "name",
      name
    );
  }
}

/**
 * Split a `setParams` argument into per-step parameter objects.
 *
 * Accepted keys are a step name (value: object of that step's parameters), or
 * `name__param` (value: one parameter, which may itself be nested as `a__b`).
 * Keys listed in `ownKeys` are returned untouched for the caller.
 */
function groupStepParams(
  params: Record<string, unknown>,
  hasStep: (name: string) => boolean,
  ownKeys: ReadonlySet<string> = new Set()
): {
  perStep: Map<string, Record<string, unknown>>;
  direct: Array<[string, unknown]>;
  own: Array<[string, unknown]>;
} {
  const perStep = new Map<string, Record<string, unknown>>();
  const direct: Array<[string, unknown]> = [];
  const own: Array<[string, unknown]> = [];
  const bucket = (name: string): Record<string, unknown> => {
    let b = perStep.get(name);
    if (!b) {
      b = {};
      perStep.set(name, b);
    }
    return b;
  };

  for (const [key, value] of Object.entries(params)) {
    if (ownKeys.has(key)) {
      own.push([key, value]);
      continue;
    }
    const sep = key.indexOf(PARAM_SEPARATOR);
    if (sep > 0) {
      const name = key.slice(0, sep);
      if (!hasStep(name)) {
        throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
      bucket(name)[key.slice(sep + PARAM_SEPARATOR.length)] = value;
    } else if (hasStep(key)) {
      if (isPlainObject(value)) {
        Object.assign(bucket(key), value);
      } else {
        direct.push([key, value]);
      }
    } else {
      throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
    }
  }
  return { perStep, direct, own };
}

/**
 * Pipeline of transforms with a final estimator.
 *
 * Sequentially applies a list of transforms and a final estimator.
 * Intermediate steps must implement `transform()`. The final step
 * can be any estimator (classifier, regressor, transformer, etc.).
 *
 * Step parameters can be changed through `setParams` with the step name and the
 * parameter joined by two underscores, for example `{ scaler__withMean: false }`.
 * This also lets a `Pipeline` be tuned by `GridSearchCV`.
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
   * @throws {InvalidParameterError} If there are no steps, a name is empty, contains `__` or is
   *   used twice, an intermediate step has no `transform()` or the final step has no `fit()`
   */
  constructor(steps: ReadonlyArray<readonly [string, Estimator | Transformer]>) {
    if (steps.length < 1) {
      throw new InvalidParameterError("Pipeline requires at least one step", "steps", steps);
    }

    // Check names and duplicates
    const names = new Set<string>();
    for (const [name] of steps) {
      validateName(name, "Pipeline step");
      if (names.has(name)) {
        throw new InvalidParameterError(`Duplicate step name: '${name}'`, "steps", name);
      }
      names.add(name);
    }

    // Validate intermediate steps are transformers
    for (let i = 0; i < steps.length - 1; i++) {
      const [name, est] = steps[i] as readonly [string, Estimator | Transformer];
      if (!isTransformer(est)) {
        throw new InvalidParameterError(
          `Intermediate step '${name}' must implement transform()`,
          "steps",
          name
        );
      }
    }

    const [lastName, last] = steps[steps.length - 1] as readonly [string, Estimator | Transformer];
    if (typeof (last as Estimator | undefined)?.fit !== "function") {
      throw new InvalidParameterError(
        `Final step '${lastName}' must implement fit()`,
        "steps",
        lastName
      );
    }

    this.steps = steps.map(([name, estimator]) => ({ name, estimator }));
  }

  /**
   * Fit all steps. Transforms are fit_transformed, final step is fit.
   *
   * @param X - Training data
   * @param y - Targets (passed to every step)
   * @returns this
   */
  fit(X: Tensor, y?: Tensor): this {
    this.fitted = false;
    const Xt = this.fitIntermediate(X, y);
    this.finalStep.estimator.fit(Xt, y);
    this.fitted = true;
    return this;
  }

  /**
   * Transform data through all steps (final step must be a transformer).
   *
   * @param X - Data to transform
   * @returns Transformed data
   * @throws {NotFittedError} If the pipeline has not been fitted
   * @throws {InvalidParameterError} If a step does not implement `transform()`
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
      Xt = step.estimator.transform(Xt);
    }
    return Xt;
  }

  /**
   * Fit and transform (final step must be a transformer).
   *
   * @param X - Training data
   * @param y - Targets (passed to every step)
   * @returns Transformed training data
   * @throws {InvalidParameterError} If the final step does not implement `transform()`
   */
  fitTransform(X: Tensor, y?: Tensor): Tensor {
    const final = this.finalStep;
    if (!isTransformer(final.estimator)) {
      throw new InvalidParameterError(
        "Final step must implement transform() for fitTransform()",
        "steps",
        final.name
      );
    }
    this.fitted = false;
    const Xt = this.fitIntermediate(X, y);
    let result: Tensor;
    if (typeof final.estimator.fitTransform === "function") {
      result = final.estimator.fitTransform(Xt, y);
    } else {
      final.estimator.fit(Xt, y);
      result = final.estimator.transform(Xt);
    }
    this.fitted = true;
    return result;
  }

  /**
   * Fit the pipeline and return the final step's predictions for `X`.
   *
   * Uses the final step's `fitPredict()` when it has one (for example clusterers).
   *
   * @param X - Training data
   * @param y - Targets (passed to every step)
   * @returns Predictions for the training data
   * @throws {InvalidParameterError} If the final step implements neither `fitPredict()` nor `predict()`
   */
  fitPredict(X: Tensor, y?: Tensor): Tensor {
    const final = this.finalStep;
    if (!hasFitPredict(final.estimator) && !hasPredict(final.estimator)) {
      throw new InvalidParameterError(
        "Final step does not implement fitPredict() or predict()",
        "steps",
        final.name
      );
    }
    this.fitted = false;
    const Xt = this.fitIntermediate(X, y);
    let result: Tensor;
    if (hasFitPredict(final.estimator)) {
      result = final.estimator.fitPredict(Xt, y);
    } else {
      final.estimator.fit(Xt, y);
      result = (final.estimator as { predict(X: Tensor): Tensor }).predict(Xt);
    }
    this.fitted = true;
    return result;
  }

  /**
   * Transform data through intermediate steps, then predict with final step.
   *
   * @param X - Samples to predict
   * @returns Predictions of the final step
   * @throws {NotFittedError} If the pipeline has not been fitted
   * @throws {InvalidParameterError} If the final step does not implement `predict()`
   */
  predict(X: Tensor): Tensor {
    this.checkFitted();
    const final = this.finalStep;
    if (!hasPredict(final.estimator)) {
      throw new InvalidParameterError(
        "Final step does not implement predict()",
        "steps",
        final.name
      );
    }
    return final.estimator.predict(this.transformIntermediate(X));
  }

  /**
   * Transform data through intermediate steps, then predict probabilities.
   *
   * @param X - Samples to predict
   * @returns Class probabilities of the final step
   * @throws {NotFittedError} If the pipeline has not been fitted
   * @throws {InvalidParameterError} If the final step does not implement `predictProba()`
   */
  predictProba(X: Tensor): Tensor {
    this.checkFitted();
    const final = this.finalStep;
    if (!hasPredictProba(final.estimator)) {
      throw new InvalidParameterError(
        "Final step does not implement predictProba()",
        "steps",
        final.name
      );
    }
    return final.estimator.predictProba(this.transformIntermediate(X));
  }

  /**
   * Transform data through intermediate steps, then compute the final step's decision function.
   *
   * @param X - Samples to score
   * @returns Decision values of the final step
   * @throws {NotFittedError} If the pipeline has not been fitted
   * @throws {InvalidParameterError} If the final step does not implement `decisionFunction()`
   */
  decisionFunction(X: Tensor): Tensor {
    this.checkFitted();
    const final = this.finalStep;
    if (!hasDecisionFunction(final.estimator)) {
      throw new InvalidParameterError(
        "Final step does not implement decisionFunction()",
        "steps",
        final.name
      );
    }
    return final.estimator.decisionFunction(this.transformIntermediate(X));
  }

  /**
   * Transform data through intermediate steps, then score with final step.
   *
   * @param X - Test samples
   * @param y - True targets
   * @returns Score of the final step
   * @throws {NotFittedError} If the pipeline has not been fitted
   * @throws {InvalidParameterError} If the final step does not implement `score()`
   */
  score(X: Tensor, y: Tensor): number {
    this.checkFitted();
    const final = this.finalStep;
    if (!hasScore(final.estimator)) {
      throw new InvalidParameterError("Final step does not implement score()", "steps", final.name);
    }
    return final.estimator.score(this.transformIntermediate(X), y);
  }

  /**
   * Map transformed data back to the original feature space, undoing the steps in reverse order.
   *
   * @param X - Data in the output space of the pipeline
   * @returns Data in the input space
   * @throws {NotFittedError} If the pipeline has not been fitted
   * @throws {InvalidParameterError} If a step does not implement `inverseTransform()`
   */
  inverseTransform(X: Tensor): Tensor {
    this.checkFitted();
    let Xt = X;
    for (let i = this.steps.length - 1; i >= 0; i--) {
      const step = this.steps[i] as Step;
      const inverse = (step.estimator as Transformer).inverseTransform;
      if (typeof inverse !== "function") {
        throw new InvalidParameterError(
          `Step '${step.name}' does not implement inverseTransform()`,
          "steps",
          step.name
        );
      }
      Xt = inverse.call(step.estimator, Xt);
    }
    return Xt;
  }

  /**
   * Get a named step by name.
   *
   * @param name - Step name
   * @throws {InvalidParameterError} If no step has that name
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

  /**
   * Class labels of the final step, or `undefined` if it does not expose any.
   *
   * @throws {NotFittedError} If the pipeline has not been fitted
   */
  get classes(): Tensor | undefined {
    this.checkFitted();
    return (this.finalStep.estimator as Classifier).classes;
  }

  /**
   * Parameters of every step, keyed by step name.
   */
  getParams(): Record<string, unknown> {
    const params: Record<string, unknown> = {};
    for (const step of this.steps) {
      params[step.name] = step.estimator.getParams();
    }
    return params;
  }

  /**
   * Set step parameters.
   *
   * Use `{ stepName: { param: value } }` or the flat form `{ stepName__param: value }`.
   * Deeper levels (a pipeline inside a pipeline) chain further: `inner__scaler__withMean`.
   *
   * @throws {InvalidParameterError} If a key does not name a step or the step rejects a value
   */
  setParams(params: Record<string, unknown>): this {
    const { perStep, direct } = groupStepParams(params, (n) =>
      this.steps.some((s) => s.name === n)
    );
    for (const [name, value] of direct) {
      throw new InvalidParameterError(
        `Parameters of step '${name}' must be an object; received ${String(value)}`,
        name,
        value
      );
    }
    for (const [name, stepParams] of perStep) {
      this.getStep(name).setParams(stepParams);
    }
    return this;
  }

  /**
   * Tags of the final step, so that tools such as `cross_val_score` treat a pipeline
   * ending in a regressor as a regressor and one ending in a classifier as a classifier.
   *
   * @internal
   */
  _getTags(): EstimatorTags {
    return getEstimatorTags(this.finalStep.estimator as Estimator);
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
    return new Pipeline(
      this.steps.map(
        ({ name, estimator }) =>
          [name, cloneEstimator(estimator, "Pipeline")] as readonly [
            string,
            Estimator | Transformer,
          ]
      )
    );
  }

  private get finalStep(): Step {
    return this.steps[this.steps.length - 1] as Step;
  }

  /** Fit every step except the last and return the data that reaches the final step. */
  private fitIntermediate(X: Tensor, y?: Tensor): Tensor {
    let Xt = X;
    for (let i = 0; i < this.steps.length - 1; i++) {
      const transformer = (this.steps[i] as Step).estimator as Transformer;
      if (typeof transformer.fitTransform === "function") {
        Xt = transformer.fitTransform(Xt, y);
      } else {
        transformer.fit(Xt, y);
        Xt = transformer.transform(Xt);
      }
    }
    return Xt;
  }

  private transformIntermediate(X: Tensor): Tensor {
    let Xt = X;
    for (let i = 0; i < this.steps.length - 1; i++) {
      Xt = ((this.steps[i] as Step).estimator as Transformer).transform(Xt);
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
 * Names are the lower-cased constructor names; repeated names get a numeric suffix
 * (`standardscaler`, `standardscaler_1`, ...).
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
  const used = new Set<string>();
  const nameCounts = new Map<string, number>();

  for (const est of estimators) {
    // Derive name from constructor
    let name = est.constructor.name.toLowerCase();
    if (!name || name === "object") name = "step";

    let count = nameCounts.get(name) ?? 0;
    let stepName = count === 0 ? name : `${name}_${count}`;
    while (used.has(stepName)) {
      count++;
      stepName = `${name}_${count}`;
    }
    nameCounts.set(name, count + 1);
    used.add(stepName);
    steps.push([stepName, est]);
  }

  return new Pipeline(steps);
}

/**
 * Concatenate results of multiple transformer objects.
 *
 * Applies each transformer independently and concatenates the results column-wise.
 * Useful for combining multiple feature extraction mechanisms into a single transformer.
 * The output is float64.
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

  /**
   * @param transformers - Array of [name, transformer] tuples
   * @throws {InvalidParameterError} If the list is empty or a name is empty, contains `__`
   *   or is used twice
   */
  constructor(transformers: ReadonlyArray<readonly [string, Transformer]>) {
    if (transformers.length === 0) {
      throw new InvalidParameterError(
        "FeatureUnion requires at least one transformer",
        "transformers",
        transformers
      );
    }

    const names = new Set<string>();
    for (const [name, transformer] of transformers) {
      validateName(name, "FeatureUnion transformer");
      if (names.has(name)) {
        throw new InvalidParameterError(
          `Duplicate transformer name: '${name}'`,
          "transformers",
          name
        );
      }
      names.add(name);
      if (
        typeof (transformer as Transformer | undefined)?.fit !== "function" ||
        typeof (transformer as Transformer).transform !== "function"
      ) {
        throw new InvalidParameterError(
          `Transformer '${name}' must implement fit() and transform()`,
          "transformers",
          name
        );
      }
    }

    this.transformers = transformers.map(([name, transformer]) => ({
      name,
      transformer,
    }));
  }

  /**
   * Fit every transformer on `X`.
   *
   * @param X - Training data
   * @param y - Targets (passed to every transformer)
   * @returns this
   */
  fit(X: Tensor, y?: Tensor): this {
    this.fitted = false;
    for (const { transformer } of this.transformers) {
      transformer.fit(X, y);
    }
    this.fitted = true;
    return this;
  }

  /**
   * Transform `X` with every transformer and concatenate the outputs column-wise.
   *
   * @param X - Data to transform
   * @returns Float64 matrix with the columns of all transformer outputs in order
   * @throws {NotFittedError} If the union has not been fitted
   * @throws {ShapeError} If the transformers return different numbers of rows
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("FeatureUnion must be fitted before transform");
    }

    const results: Tensor[] = [];
    for (const { transformer } of this.transformers) {
      results.push(transformer.transform(X));
    }

    return hstackTensors(results);
  }

  /**
   * Fit every transformer and return the concatenated outputs.
   *
   * @param X - Training data
   * @param y - Targets (passed to every transformer)
   */
  fitTransform(X: Tensor, y?: Tensor): Tensor {
    this.fitted = false;
    const results: Tensor[] = [];
    for (const { transformer } of this.transformers) {
      if (typeof transformer.fitTransform === "function") {
        results.push(transformer.fitTransform(X, y));
      } else {
        transformer.fit(X, y);
        results.push(transformer.transform(X));
      }
    }
    const out = hstackTensors(results);
    this.fitted = true;
    return out;
  }

  /**
   * Parameters of every transformer, keyed by transformer name.
   */
  getParams(): Record<string, unknown> {
    const params: Record<string, unknown> = {};
    for (const { name, transformer } of this.transformers) {
      params[name] = transformer.getParams();
    }
    return params;
  }

  /**
   * Set transformer parameters with `{ name: { param: value } }` or `{ name__param: value }`.
   *
   * @throws {InvalidParameterError} If a key does not name a transformer or it rejects a value
   */
  setParams(params: Record<string, unknown>): this {
    const { perStep, direct } = groupStepParams(params, (n) =>
      this.transformers.some((t) => t.name === n)
    );
    for (const [name, value] of direct) {
      throw new InvalidParameterError(
        `Parameters of transformer '${name}' must be an object; received ${String(value)}`,
        name,
        value
      );
    }
    for (const [name, stepParams] of perStep) {
      this.transformers.find((t) => t.name === name)?.transformer.setParams(stepParams);
    }
    return this;
  }

  /**
   * Create an unfitted copy whose transformers are fresh clones.
   *
   * @returns A new FeatureUnion
   */
  clone(): FeatureUnion {
    return new FeatureUnion(
      this.transformers.map(
        ({ name, transformer }) =>
          [name, cloneEstimator(transformer, "FeatureUnion")] as readonly [string, Transformer]
      )
    );
  }

  get transformerNames(): string[] {
    return this.transformers.map((t) => t.name);
  }
}

/** Transformer spec accepted by {@link ColumnTransformer}. */
type ColumnSpec = {
  name: string;
  transformer: Transformer | "passthrough" | "drop";
  /** Columns as given by the user (may contain negative indices). */
  columns: number[];
  /** Non-negative column indices, resolved against the feature count at fit time. */
  resolved: number[];
};

/**
 * ColumnTransformer applies transformers to specific column subsets of the data.
 *
 * Each transformer is applied to a specified subset of columns, and the results
 * are concatenated horizontally. This is useful when different feature types
 * require different preprocessing. A spec with an empty column list is skipped.
 * The output is float64.
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
  private readonly transformerSpecs: ColumnSpec[];
  private remainder: "drop" | "passthrough";
  private remainderCols: number[] = [];
  private nFeaturesIn_ = 0;
  private fitted = false;

  /**
   * @param transformers - Array of [name, transformer, columns] tuples.
   *   transformer can be a Transformer instance, "passthrough", or "drop".
   *   columns is an array of column indices; negative indices count from the last column.
   * @param options - Additional options.
   *   remainder: "drop" (default) or "passthrough" for columns not specified.
   * @throws {InvalidParameterError} If the list is empty, a name is empty, contains `__` or is
   *   used twice, a transformer or column list is invalid, or `remainder` is unknown
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
    for (const [name, transformer, columns] of transformers) {
      validateName(name, "ColumnTransformer transformer");
      if (names.has(name)) {
        throw new InvalidParameterError(
          `Duplicate transformer name: '${name}'`,
          "transformers",
          name
        );
      }
      names.add(name);
      ColumnTransformer.checkTransformer(name, transformer);
      if (!Array.isArray(columns) || !columns.every((c) => Number.isInteger(c))) {
        throw new InvalidParameterError(
          `Columns of transformer '${name}' must be an array of integer indices`,
          "transformers",
          columns
        );
      }
    }

    this.transformerSpecs = transformers.map(([name, transformer, columns]) => ({
      name,
      transformer,
      columns: [...columns],
      resolved: [],
    }));
    this.remainder = ColumnTransformer.checkRemainder(options.remainder ?? "drop");
  }

  private static checkTransformer(name: string, transformer: unknown): void {
    if (transformer === "passthrough" || transformer === "drop") return;
    const t = transformer as Partial<Transformer> | null | undefined;
    if (
      typeof t !== "object" ||
      t === null ||
      typeof t.fit !== "function" ||
      typeof t.transform !== "function"
    ) {
      throw new InvalidParameterError(
        `Transformer '${name}' must be a transformer, "passthrough" or "drop"`,
        "transformers",
        name
      );
    }
  }

  private static checkRemainder(value: unknown): "drop" | "passthrough" {
    if (value !== "drop" && value !== "passthrough") {
      throw new InvalidParameterError(
        `remainder must be "drop" or "passthrough"; received ${String(value)}`,
        "remainder",
        value
      );
    }
    return value;
  }

  /**
   * Fit every transformer on its column subset.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Targets (passed to every transformer)
   * @returns this
   * @throws {ShapeError} If X is not 1-D or 2-D
   * @throws {InvalidParameterError} If a column index is out of range
   */
  fit(X: Tensor, y?: Tensor): this {
    this.fitted = false;
    const nFeats = ColumnTransformer.featureCount(X);
    this.resolveColumns(nFeats);
    const flat = toFloat64View(X);

    for (const spec of this.transformerSpecs) {
      if (spec.transformer === "passthrough" || spec.transformer === "drop") continue;
      if (spec.resolved.length === 0) continue;
      spec.transformer.fit(ColumnTransformer.selectColumns(flat, X, nFeats, spec.resolved), y);
    }

    this.nFeaturesIn_ = nFeats;
    this.fitted = true;
    return this;
  }

  /**
   * Transform `X` column group by column group and concatenate the results.
   *
   * @param X - Data with the same number of features as in `fit`
   * @returns Float64 matrix: transformer outputs in spec order, then remainder columns
   *   when `remainder` is "passthrough"
   * @throws {NotFittedError} If the transformer has not been fitted
   * @throws {ShapeError} If X has a different number of features than in `fit`
   */
  transform(X: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("ColumnTransformer must be fitted before transform");
    }
    const nFeats = ColumnTransformer.featureCount(X);
    if (nFeats !== this.nFeaturesIn_) {
      throw new ShapeError(
        `X has ${nFeats} features but ColumnTransformer was fitted with ${this.nFeaturesIn_} features`
      );
    }
    const flat = toFloat64View(X);

    const parts: Tensor[] = [];
    for (const spec of this.transformerSpecs) {
      if (spec.transformer === "drop" || spec.resolved.length === 0) continue;
      const subX = ColumnTransformer.selectColumns(flat, X, nFeats, spec.resolved);
      parts.push(spec.transformer === "passthrough" ? subX : spec.transformer.transform(subX));
    }

    if (this.remainder === "passthrough" && this.remainderCols.length > 0) {
      parts.push(ColumnTransformer.selectColumns(flat, X, nFeats, this.remainderCols));
    }

    if (parts.length === 0) {
      return tensor(new Float64Array(0)).reshape([X.shape[0] ?? 0, 0]);
    }

    return hstackTensors(parts);
  }

  /**
   * Fit and transform in one call.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param y - Targets (passed to every transformer)
   */
  fitTransform(X: Tensor, y?: Tensor): Tensor {
    this.fit(X, y);
    return this.transform(X);
  }

  /**
   * Remainder mode plus the parameters of every transformer, keyed by transformer name.
   */
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

  /**
   * Set `remainder`, replace a spec's transformer with "passthrough" or "drop", or set
   * transformer parameters with `{ name: { param: value } }` or `{ name__param: value }`.
   *
   * @throws {InvalidParameterError} If a key is unknown or a value is invalid
   */
  setParams(params: Record<string, unknown>): this {
    const { perStep, direct, own } = groupStepParams(
      params,
      (n) => this.transformerSpecs.some((s) => s.name === n),
      new Set(["remainder"])
    );
    for (const [, value] of own) {
      this.remainder = ColumnTransformer.checkRemainder(value);
    }
    for (const [name, value] of direct) {
      if (value !== "passthrough" && value !== "drop") {
        throw new InvalidParameterError(
          `Transformer '${name}' can only be replaced by "passthrough" or "drop"; received ${String(value)}`,
          name,
          value
        );
      }
      const spec = this.transformerSpecs.find((s) => s.name === name);
      if (spec) spec.transformer = value;
    }
    for (const [name, stepParams] of perStep) {
      const spec = this.transformerSpecs.find((s) => s.name === name);
      if (!spec || typeof spec.transformer === "string") {
        throw new InvalidParameterError(
          `Transformer '${name}' is "${String(spec?.transformer)}" and has no parameters to set`,
          name,
          stepParams
        );
      }
      spec.transformer.setParams(stepParams);
    }
    return this;
  }

  /**
   * Create an unfitted copy whose transformers are fresh clones.
   *
   * @returns A new ColumnTransformer
   */
  clone(): ColumnTransformer {
    return new ColumnTransformer(
      this.transformerSpecs.map(
        ({ name, transformer, columns }) =>
          [
            name,
            typeof transformer === "string"
              ? transformer
              : cloneEstimator(transformer, "ColumnTransformer"),
            [...columns],
          ] as const
      ),
      { remainder: this.remainder }
    );
  }

  get transformerNames(): string[] {
    return this.transformerSpecs.map((s) => s.name);
  }

  private static featureCount(X: Tensor): number {
    if (X.ndim === 1) return 1;
    if (X.ndim === 2) return X.shape[1] ?? 0;
    throw new ShapeError(`X must be 1-dimensional or 2-dimensional; got ndim=${X.ndim}`);
  }

  /** Resolve negative indices, validate ranges and compute the remainder columns. */
  private resolveColumns(nFeats: number): void {
    const used = new Set<number>();
    for (const spec of this.transformerSpecs) {
      spec.resolved = spec.columns.map((c) => {
        const index = c < 0 ? c + nFeats : c;
        if (index < 0 || index >= nFeats) {
          throw new InvalidParameterError(
            `Column index ${c} of transformer '${spec.name}' is out of range for ${nFeats} features`,
            "columns",
            c
          );
        }
        used.add(index);
        return index;
      });
    }
    this.remainderCols = [];
    for (let i = 0; i < nFeats; i++) {
      if (!used.has(i)) {
        this.remainderCols.push(i);
      }
    }
  }

  /** Copy the given columns of `X` into a new float64 matrix. */
  private static selectColumns(
    flat: Float64Array,
    X: Tensor,
    nFeats: number,
    cols: readonly number[]
  ): Tensor {
    const nSamples = X.shape[0] ?? 0;
    const nCols = cols.length;
    const out = new Float64Array(nSamples * nCols);
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nCols; j++) {
        out[i * nCols + j] = flat[i * nFeats + (cols[j] as number)] as number;
      }
    }
    return tensor(out).reshape([nSamples, nCols]);
  }
}

/**
 * Concatenate 1-D or 2-D tensors column-wise into one float64 matrix.
 * 1-D tensors count as a single column.
 */
function hstackTensors(tensors: Tensor[]): Tensor {
  if (tensors.length === 0) {
    throw new InvalidParameterError("No tensors to concatenate", "tensors", tensors);
  }

  const first = tensors[0] as Tensor;
  const nSamples = first.shape[0] ?? 0;

  let totalFeatures = 0;
  const featureCounts: number[] = [];
  for (const t of tensors) {
    if (t.ndim !== 1 && t.ndim !== 2) {
      throw new ShapeError(`Transformer output must be 1-D or 2-D; got ndim=${t.ndim}`);
    }
    const tSamples = t.shape[0] ?? 0;
    if (tSamples !== nSamples) {
      throw new ShapeError(
        `All transformers must produce the same number of rows; got ${nSamples} and ${tSamples}`
      );
    }
    const nFeats = t.ndim === 1 ? 1 : (t.shape[1] ?? 0);
    featureCounts.push(nFeats);
    totalFeatures += nFeats;
  }

  const flats = tensors.map((t) => toFloat64View(t));
  const result = new Float64Array(nSamples * totalFeatures);
  let colOffset = 0;
  for (let ti = 0; ti < tensors.length; ti++) {
    const flat = flats[ti] as Float64Array;
    const nFeats = featureCounts[ti] as number;
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeats; j++) {
        result[i * totalFeatures + colOffset + j] = flat[i * nFeats + j] as number;
      }
    }
    colOffset += nFeats;
  }

  return tensor(result).reshape([nSamples, totalFeatures]);
}
