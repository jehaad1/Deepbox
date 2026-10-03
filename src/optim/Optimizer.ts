/**
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError } from "../core";
import { GradTensor } from "../ndarray";
import { assertFinite, assertFiniteNonNegative, assertHasGradFloat } from "./_internal";

/**
 * Represents a group of parameters with optional per-group hyperparameters.
 *
 * Options left out of a group fall back to the optimizer defaults.
 *
 * @template Options - Type of optimizer-specific options
 * @property params - Iterable of parameters to optimize in this group
 */
export type ParamGroup<Options extends Record<string, unknown>> = {
  readonly params: Iterable<GradTensor>;
} & Partial<Options>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function ensureRecord(value: unknown, context: string) {
  if (!isRecord(value)) {
    throw new DataValidationError(`${context} must be an object`);
  }
  return value;
}

function ensureIntegerArray(value: unknown, context: string) {
  if (!Array.isArray(value)) {
    throw new DataValidationError(`${context} must be an array of integers`);
  }
  const output: number[] = [];
  for (const entry of value) {
    if (!Number.isInteger(entry)) {
      throw new DataValidationError(`${context} must contain integers only`);
    }
    output.push(entry);
  }
  return output;
}

type CopyableTypedArray =
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | Uint8Array
  | Uint8ClampedArray
  | Uint16Array
  | Uint32Array
  | BigInt64Array
  | BigUint64Array;

function isTypedArray(value: unknown): value is CopyableTypedArray {
  return ArrayBuffer.isView(value) && !(value instanceof DataView);
}

/**
 * Deep copy of the plain data held in optimizer state and options: typed arrays,
 * arrays, `Map`s and plain objects are copied. Tensors and other class instances
 * are shared by reference (optimizers replace them instead of mutating them).
 */
function cloneValue(value: unknown): unknown {
  if (typeof value !== "object" || value === null) return value;
  if (isTypedArray(value)) return value.slice();
  if (Array.isArray(value)) return value.map(cloneValue);
  if (value instanceof Map) {
    return new Map(Array.from(value.entries(), ([k, v]) => [k, cloneValue(v)]));
  }
  const proto = Object.getPrototypeOf(value);
  if (proto === Object.prototype || proto === null) {
    return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, cloneValue(v)]));
  }
  return value;
}

/**
 * Check a per-group option against the type of its default.
 *
 * Options whose default is `null` or `undefined` (for example `lineSearchFn: null`) accept any
 * value, and `null` is accepted for options whose default is a string.
 *
 * @returns The expected type name when the value does not fit, otherwise `undefined`.
 */
function optionTypeMismatch(defaultValue: unknown, value: unknown): string | undefined {
  if (defaultValue === null || defaultValue === undefined) return undefined;
  const expectedType = typeof defaultValue;
  if (value === null && expectedType === "string") return undefined;
  return typeof value === expectedType ? undefined : expectedType;
}

/**
 * Resolve the parameters of one group into a new array.
 *
 * Repeated parameters inside one group (for example tied weights yielded twice by
 * `Module.parameters()`) are kept once, so they are not updated twice per step.
 * A parameter that already belongs to another group is rejected.
 */
function collectParams(raw: unknown, context: string, seen: Set<GradTensor>): GradTensor[] {
  if (
    typeof raw !== "object" ||
    raw === null ||
    typeof (raw as { [Symbol.iterator]?: unknown })[Symbol.iterator] !== "function"
  ) {
    throw new InvalidParameterError(`${context} must be an iterable of parameters`, "params", raw);
  }
  const local = new Set<GradTensor>();
  for (const item of raw as Iterable<unknown>) {
    if (!GradTensor.isGradTensor(item)) {
      throw new InvalidParameterError(
        `${context} must contain GradTensor parameters only (create them with parameter())`,
        "params",
        item
      );
    }
    if (local.has(item)) continue;
    if (seen.has(item)) {
      throw new InvalidParameterError(
        `${context} contains a parameter that already belongs to another parameter group`,
        "params",
        item
      );
    }
    local.add(item);
  }
  for (const item of local) seen.add(item);
  return Array.from(local);
}

/**
 * Merge per-group options over the optimizer defaults. Options set to `undefined`
 * fall back to the default, and the type of every known option is checked.
 */
function mergeGroupOptions<Options extends Record<string, unknown>>(
  defaults: Readonly<Options>,
  groupOptions: Record<string, unknown>
): Options {
  const merged = cloneValue(defaults) as Options;
  const mergedRecord: Record<string, unknown> = merged;
  const defaultsRecord: Record<string, unknown> = defaults;
  for (const [key, value] of Object.entries(groupOptions)) {
    if (value === undefined) continue;
    if (Object.hasOwn(defaultsRecord, key)) {
      const expectedType = optionTypeMismatch(defaultsRecord[key], value);
      if (expectedType !== undefined) {
        throw new InvalidParameterError(
          `Invalid option '${key}' in parameter group: expected ${expectedType}, got ${typeof value}`,
          key,
          value
        );
      }
    }
    if (typeof value === "number" && Number.isNaN(value)) {
      throw new InvalidParameterError(
        `Invalid option '${key}' in parameter group: NaN`,
        key,
        value
      );
    }
    mergedRecord[key] = cloneValue(value);
  }
  return merged;
}

/**
 * Type guard to determine if params is an array of parameter groups.
 *
 * This function checks whether the provided params argument is a simple iterable
 * of parameters or an array of parameter groups with per-group options.
 *
 * @template Options - Type of optimizer-specific options
 * @param params - Either an iterable of parameters or array of parameter groups
 * @returns True if params is an array of parameter groups, false otherwise
 */
function isParamGroupArray<Options extends Record<string, unknown>>(
  params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<Options>>
): params is ReadonlyArray<ParamGroup<Options>> {
  // Parameter groups must be passed as an array.
  if (!Array.isArray(params)) return false;
  // An empty array is treated as an (empty) list of groups.
  if (params.length === 0) return true;
  // A group is a plain object with a 'params' property; a GradTensor has none.
  const first = params[0];
  if (!first || typeof first !== "object") return false;
  return "params" in first;
}

/**
 * Abstract base class for all optimizers.
 *
 * Concrete optimizers (SGD, Adam, ...) extend this class and implement `step()`.
 * The base class owns the parameter groups, the per-parameter state map, gradient
 * zeroing and checkpointing through `stateDict()` / `loadStateDict()`.
 *
 * Parameters are split into groups, and every group carries its own options (for
 * example a different learning rate per layer). Options a group does not set are
 * taken from the optimizer defaults, and every group is range checked when it is added
 * and again at each `step()`.
 *
 * A parameter with `requiresGrad = false` (for example one frozen with
 * `Module.freezeParameters()`) stays in its group but is skipped by `step()`. A
 * trainable parameter whose gradient is `null` (it took no part in the loss, or no
 * `backward()` has run yet) is skipped as well, as in PyTorch, and gets no optimizer state.
 * A failed `step()` (a non-finite gradient, an invalid option) changes neither the
 * parameters nor the optimizer state.
 *
 * @example
 * ```ts
 * import { SGD } from 'deepbox/optim';
 *
 * const optimizer = new SGD(model.parameters(), { lr: 0.01 });
 *
 * // Training loop
 * for (let epoch = 0; epoch < 100; epoch++) {
 *   optimizer.zeroGrad();
 *   const loss = computeLoss();
 *   loss.backward();
 *   optimizer.step();
 *   console.log(`epoch ${epoch}: loss ${loss.item()}`);
 * }
 * ```
 *
 * @example
 * ```ts
 * // Different learning rates per group
 * const optimizer = new SGD([
 *   { params: model.layer1.parameters(), lr: 0.01 },
 *   { params: model.layer2.parameters(), lr: 0.001 }
 * ], { lr: 0.01 });
 * ```
 *
 * @template Options - Type defining optimizer-specific hyperparameters
 * @template State - Type defining per-parameter state (e.g., momentum buffers)
 * @see {@link https://deepbox.dev/docs/optim-optimizers | Deepbox documentation}
 * @category Optimization
 */
export abstract class Optimizer<
  Options extends Record<string, unknown>,
  State extends Record<string, unknown>,
> {
  /**
   * Groups of parameters with their associated hyperparameters.
   * Each group can have different options (e.g., learning rates).
   * Exposed publicly to enable scheduler integrations.
   */
  public paramGroups: Array<{
    params: GradTensor[];
    options: Options;
  }>;

  /**
   * Learning rate of the first parameter group, or 0 when the optimizer has no
   * groups or the first group has no numeric `lr` option.
   */
  get lr(): number {
    const group = this.paramGroups[0];
    if (!group) {
      return 0;
    }
    const opts = group.options as Record<string, unknown>;
    const lrVal = opts["lr"];
    return typeof lrVal === "number" ? lrVal : 0;
  }

  /**
   * Per-parameter state storage.
   * Maps each parameter to its optimizer-specific state (momentum, adaptive rates, etc.).
   */
  protected state: Map<GradTensor, State> = new Map();

  /**
   * Create a new optimizer.
   *
   * @param params - Either an iterable of parameters or an array of parameter groups
   * @param defaults - Default hyperparameters applied to all groups
   * @throws {InvalidParameterError} If `params` holds something other than GradTensor
   *   parameters, a group option has the wrong type or is NaN, or a parameter appears
   *   in more than one group. A parameter repeated inside one group is kept once.
   */
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<Options>>,
    protected readonly defaults: Readonly<Options>
  ) {
    this.paramGroups = [];
    const seen = new Set<GradTensor>();

    // The hook is a prototype method, so it can run before the subclass constructor body.
    this.validateOptions(defaults);

    if (!isParamGroupArray<Options>(params)) {
      // Plain iterable of parameters: one group with the default options.
      this.paramGroups.push({
        params: collectParams(params, "params", seen),
        options: mergeGroupOptions<Options>(defaults, {}),
      });
    } else {
      params.forEach((group, index) => {
        if (!isRecord(group) || !("params" in group)) {
          throw new InvalidParameterError(
            `params[${index}] must be an object with a 'params' property`,
            "params",
            group
          );
        }
        const { params: groupParams, ...groupOptions } = group;
        const options = mergeGroupOptions<Options>(defaults, groupOptions);
        this.validateOptions(options);
        this.paramGroups.push({
          params: collectParams(groupParams, `params[${index}].params`, seen),
          options,
        });
      });
    }
  }

  /**
   * Check the range of every option of one parameter group.
   *
   * The base class only checks option types and NaN. Subclasses override this hook to
   * check ranges (for example `lr >= 0` or `0 <= beta1 < 1`). It runs on the defaults and
   * on the merged options of every group when the optimizer is built, in `addParamGroup`,
   * in `loadStateDict`, and at the start of every `step()`, so a per-group override such
   * as `{ params, lr: -1 }` is rejected as early as a bad default is. It must not read
   * instance fields, because the constructor calls it before the subclass is initialized.
   *
   * @param _options - Merged options of one group (or the optimizer defaults)
   * @throws {InvalidParameterError} If an option is out of range
   */
  protected validateOptions(_options: Readonly<Options>): void {
    // No range checks by default.
  }

  /** Number of `step()` calls made so far, restored by `loadStateDict()`. */
  private _stepCount = 0;

  /**
   * Total number of optimization steps performed, including the steps recorded in a
   * state dictionary that was loaded.
   */
  get stepCount(): number {
    return this._stepCount;
  }

  /** Record one `step()` call. Subclasses call this once per step. */
  protected countStep(): void {
    this._stepCount++;
  }

  /** Overwrite the step counter (used when restoring a legacy checkpoint). */
  protected restoreStepCount(count: number): void {
    this._stepCount = count;
  }

  /**
   * Parameters of a group that take part in an update: those with `requiresGrad = true`
   * and a gradient that is not `null`.
   *
   * A frozen parameter (for example after `Module.freezeParameters()`) or a parameter that
   * received no gradient stays in its group but is left alone by `step()`, as PyTorch
   * leaves a parameter whose `grad` is `None`.
   *
   * @param group - Parameter group
   * @returns The parameters to update, in group order
   */
  protected trainableParams(group: { readonly params: readonly GradTensor[] }): GradTensor[] {
    return group.params.filter((param) => param.requiresGrad && param.grad !== null);
  }

  /**
   * Parameters of a group with `requiresGrad = true`, whether or not they have a gradient.
   * Optimizers that lay all parameters out in one vector (L-BFGS) use this so the layout does
   * not change from step to step.
   *
   * @param group - Parameter group
   * @returns The trainable parameters, in group order
   */
  protected requiresGradParams(group: { readonly params: readonly GradTensor[] }): GradTensor[] {
    return group.params.filter((param) => param.requiresGrad);
  }

  /**
   * Validate a whole step before anything is modified, so a failed `step()` leaves the
   * parameters, the moment buffers and the step counter exactly as they were.
   *
   * Checks the options of every group, then for every trainable host parameter that its
   * gradient has a supported dtype and matching shape, and that its gradient and its
   * values are finite. Parameters without a gradient are skipped, and parameters on a
   * device are not read back.
   *
   * @param name - Optimizer name for error messages
   * @throws {InvalidParameterError} If an option is out of range, or a gradient or
   *   parameter value is not finite
   * @throws {DTypeError} If a parameter or gradient has an unsupported dtype
   * @throws {ShapeError} If a gradient shape differs from its parameter, or a parameter is
   *   a non-contiguous view
   */
  protected prepareStep(name: string): void {
    for (const group of this.paramGroups) {
      this.validateOptions(group.options);
      for (const param of this.trainableParams(group)) {
        if (param.tensor.isDeviceTensor) continue;
        const view = assertHasGradFloat(param, name);
        const size = param.tensor.size;
        for (let i = 0; i < size; i++) {
          const g = view.grad[view.gradOffset + i] as number;
          if (!Number.isFinite(g)) assertFinite("gradient", g);
          const p = view.param[view.paramOffset + i] as number;
          if (!Number.isFinite(p)) assertFinite("parameter", p);
        }
      }
    }
  }

  /**
   * Get the learning rate of a parameter group.
   *
   * @param groupIdx - Parameter group index (default: 0)
   * @returns Current learning rate of the group
   * @throws {InvalidParameterError} If the group index does not exist, or the group has no
   *   numeric `lr` option
   */
  getLearningRate(groupIdx = 0): number {
    const group = this.paramGroups[groupIdx];
    if (!group) {
      throw new InvalidParameterError(
        `Invalid group index: ${groupIdx} (valid range: [0, ${this.paramGroups.length}))`,
        "groupIdx",
        groupIdx
      );
    }
    const lr = (group.options as Record<string, unknown>)["lr"];
    if (typeof lr !== "number") {
      throw new InvalidParameterError("The optimizer has no 'lr' option", "lr", lr);
    }
    return lr;
  }

  /**
   * Set the learning rate of every parameter group.
   *
   * @param lr - New learning rate (finite and non-negative)
   * @throws {InvalidParameterError} If `lr` is negative or not finite
   */
  setLearningRate(lr: number): void {
    assertFiniteNonNegative("learning rate", lr);
    for (const group of this.paramGroups) {
      (group.options as Record<string, unknown>)["lr"] = lr;
    }
  }

  /**
   * Perform a single optimization step (parameter update).
   *
   * Implemented by every optimizer subclass; it updates all parameters from their
   * gradients.
   *
   * @param closure - Optional closure that reevaluates the model and returns the loss.
   *                  Used by optimizers (e.g., LBFGS) that need several function
   *                  evaluations per step.
   * @returns Loss value if a closure is provided, undefined otherwise
   */
  abstract step(closure?: () => number): number | undefined;

  /**
   * Reset the gradients of all optimized parameters to zero.
   *
   * Call this at the start of each training iteration, before computing new
   * gradients. Without it, gradients accumulate across iterations.
   *
   * @example
   * ```ts
   * // Typical training loop
   * optimizer.zeroGrad();              // Clear previous gradients
   * const output = model.forward(input);
   * const loss = criterion(output, target);
   * loss.backward();                   // Compute new gradients
   * optimizer.step();                  // Update parameters
   * console.log(loss.item());          // Read the scalar loss
   * ```
   */
  zeroGrad(): void {
    for (const group of this.paramGroups) {
      for (const param of group.params) {
        param.zeroGrad();
      }
    }
  }

  /**
   * Add a parameter group to the optimizer.
   *
   * Useful for fine-tuning (adding pre-trained layers with their own learning
   * rate), progressive unfreezing and models that grow during training.
   *
   * @param paramGroup - Parameter group to add with optional per-group options
   * @throws {InvalidParameterError} If the group is malformed, or contains a
   *   parameter the optimizer already holds in another group
   *
   * @example
   * ```ts
   * const optimizer = new SGD(model.backbone.parameters(), { lr: 0.001 });
   * // Later, add classifier with higher learning rate
   * optimizer.addParamGroup({
   *   params: model.classifier.parameters(),
   *   lr: 0.01
   * });
   * ```
   */
  addParamGroup(paramGroup: ParamGroup<Options>): void {
    if (!isRecord(paramGroup) || !("params" in paramGroup)) {
      throw new InvalidParameterError(
        "paramGroup must be an object with a 'params' property",
        "paramGroup",
        paramGroup
      );
    }
    const { params, ...options } = paramGroup;
    const seen = new Set<GradTensor>(this.paramGroups.flatMap((group) => group.params));
    const merged = mergeGroupOptions<Options>(this.defaults, options);
    this.validateOptions(merged);
    const collected = collectParams(params, "paramGroup.params", seen);
    this.paramGroups.push({ params: collected, options: merged });
  }

  /**
   * Validate that a given state object matches the optimizer's state type.
   *
   * @param state - The state object to validate
   * @returns True if the state object is valid, false otherwise
   */
  protected abstract isState(state: Record<string, unknown>): state is State;

  /**
   * Get a snapshot of the optimizer state for checkpointing.
   *
   * The result holds the per-parameter state (momentum buffers, adaptive moments,
   * step counters), the optimizer's own step count (`stepCount`) and the parameter groups
   * with their options. Buffers and options
   * are copied, so later optimizer steps or learning-rate changes do not alter the
   * returned object. Parameters are identified by `paramId`, their position in the
   * flattened parameter groups, so the state can be loaded into an optimizer built
   * over a different set of parameter objects with the same layout. The `param` and
   * `params` fields hold references to the live parameters for backward
   * compatibility.
   *
   * @returns Optimizer state dictionary containing state and parameter groups
   *
   * @example
   * ```ts
   * // Save checkpoint
   * const checkpoint = {
   *   model: model.stateDict(),
   *   optimizer: optimizer.stateDict(),
   *   epoch: currentEpoch
   * };
   * ```
   */
  stateDict() {
    // Ids follow the flattened group order, which is what loadStateDict resolves them against.
    const paramIdMap = new Map<GradTensor, number>();
    for (const group of this.paramGroups) {
      for (const param of group.params) {
        if (!paramIdMap.has(param)) paramIdMap.set(param, paramIdMap.size);
      }
    }
    const idOf = (param: GradTensor): number => {
      const id = paramIdMap.get(param);
      if (id === undefined) {
        // Only reachable when state exists for a parameter that is in no group.
        throw new DataValidationError("Optimizer state refers to a parameter outside its groups");
      }
      return id;
    };

    const stateEntries: Array<{ paramId: number; param: GradTensor; state: State }> = [];
    for (const [param, paramState] of this.state) {
      if (!paramIdMap.has(param)) continue;
      const copied = cloneValue(paramState);
      stateEntries.push({ paramId: idOf(param), param, state: copied as State });
    }

    return {
      stepCount: this._stepCount,
      state: stateEntries,
      paramGroups: this.paramGroups.map((group) => ({
        params: [...group.params],
        paramIds: group.params.map(idOf),
        options: cloneValue(group.options) as Options,
      })),
    };
  }

  /**
   * Load optimizer state from a state dictionary.
   *
   * Restores the per-parameter state and the parameter group options saved by
   * `stateDict()`. The optimizer must have the same type and the same parameter
   * layout as the one that produced the dictionary. The dictionary is validated
   * before anything is changed, so a failed load leaves the optimizer untouched,
   * and the loaded buffers are copied, so the dictionary can be loaded again or
   * into another optimizer.
   *
   * @param stateDict - State dictionary previously returned by stateDict()
   * @throws {DataValidationError} If the dictionary is malformed or does not match
   *   this optimizer's parameters
   *
   * @example
   * ```ts
   * // Resume from checkpoint
   * const checkpoint = loadCheckpoint('checkpoint.json');
   * model.loadStateDict(checkpoint.model);
   * optimizer.loadStateDict(checkpoint.optimizer);
   * ```
   */
  loadStateDict(stateDict: Record<string, unknown>): void {
    ensureRecord(stateDict, "stateDict");
    const currentParams = this.paramGroups.flatMap((group) => group.params);
    const currentParamCount = currentParams.length;
    const paramLookup = new Map<unknown, number>();
    for (let i = 0; i < currentParams.length; i++) {
      paramLookup.set(currentParams[i], i);
    }

    let nextGroups: Array<{ params: GradTensor[]; options: Options }> | undefined;
    let nextState: Map<GradTensor, State> | undefined;
    let nextStepCount: number | undefined;

    if (Object.hasOwn(stateDict, "stepCount")) {
      const rawCount = stateDict["stepCount"];
      if (typeof rawCount !== "number" || !Number.isInteger(rawCount) || rawCount < 0) {
        throw new DataValidationError("stepCount must be a non-negative integer");
      }
      nextStepCount = rawCount;
    }

    if (Object.hasOwn(stateDict, "paramGroups")) {
      const rawGroups = stateDict["paramGroups"];
      if (!Array.isArray(rawGroups)) {
        throw new DataValidationError("paramGroups must be an array");
      }
      const groupsArray: unknown[] = rawGroups;

      if (groupsArray.length === 0) {
        if (this.paramGroups.length !== 0) {
          throw new DataValidationError("paramGroups cannot be empty");
        }
        nextGroups = [];
      } else {
        if (groupsArray.length !== this.paramGroups.length) {
          throw new DataValidationError("paramGroups count mismatch");
        }

        const seenIndices = new Set<number>();
        let sawParamIds = false;
        let sawNoParamIds = false;
        const resolvedGroups: Array<{ params: GradTensor[]; options: Options }> = [];

        groupsArray.forEach((rawGroup, index) => {
          const groupRecord = ensureRecord(rawGroup, `paramGroups[${index}]`);
          const optionsRaw = ensureRecord(groupRecord["options"], `paramGroups[${index}].options`);
          const options = cloneValue(this.defaults) as Options;
          const optionsRecord: Record<string, unknown> = options;
          const defaultsRecord: Record<string, unknown> = { ...this.defaults };

          for (const [key, value] of Object.entries(optionsRaw)) {
            if (Object.hasOwn(defaultsRecord, key)) {
              const expectedType = optionTypeMismatch(defaultsRecord[key], value);

              if (expectedType !== undefined) {
                throw new DataValidationError(
                  `Type mismatch for option '${key}' in paramGroups[${index}]: expected ${expectedType}, got ${typeof value}`
                );
              }
              optionsRecord[key] = cloneValue(value);
            }
          }

          this.validateOptions(options);

          const paramIdsRaw = groupRecord["paramIds"];
          const paramsRaw = groupRecord["params"];
          let paramIds: number[] | undefined;
          if (paramIdsRaw !== undefined) {
            paramIds = ensureIntegerArray(paramIdsRaw, `paramGroups[${index}].paramIds`);
            sawParamIds = true;
          } else {
            sawNoParamIds = true;
          }

          // Indices into currentParams for this group.
          let indices: number[] | undefined;

          if (paramIds) {
            for (const id of paramIds) {
              if (id < 0 || id >= currentParamCount) {
                throw new DataValidationError(`Invalid paramId ${id} in paramGroups`);
              }
            }
            indices = paramIds;
          }

          if (paramsRaw !== undefined) {
            if (!Array.isArray(paramsRaw)) {
              throw new DataValidationError(`paramGroups[${index}].params must be an array`);
            }
            const fromParams: number[] = [];
            let hasUnknown = false;
            for (const paramRef of paramsRaw) {
              const paramIndex = paramLookup.get(paramRef);
              if (paramIndex === undefined) {
                hasUnknown = true;
              } else {
                fromParams.push(paramIndex);
              }
            }
            if (indices) {
              // Parameter references from another optimizer are ignored; ids decide.
              if (!hasUnknown) {
                if (indices.length !== fromParams.length) {
                  throw new DataValidationError("paramIds length does not match params length");
                }
                if (indices.some((id, i) => id !== fromParams[i])) {
                  throw new DataValidationError(
                    `paramGroups[${index}].paramIds do not match its params`
                  );
                }
              }
            } else if (hasUnknown) {
              throw new DataValidationError(
                `paramGroups[${index}].params contains parameters that are not part of this optimizer`
              );
            } else {
              indices = fromParams;
            }
          }

          if (!indices) {
            throw new DataValidationError(`paramGroups[${index}] must include params or paramIds`);
          }

          for (const id of indices) {
            if (seenIndices.has(id)) {
              throw new DataValidationError(`Duplicate paramId ${id} in paramGroups`);
            }
            seenIndices.add(id);
          }

          resolvedGroups.push({
            params: indices.map((id) => {
              const param = currentParams[id];
              if (!param) {
                throw new DataValidationError(`Invalid paramId ${id} in paramGroups`);
              }
              return param;
            }),
            options,
          });
        });

        if (sawParamIds && sawNoParamIds) {
          throw new DataValidationError("paramIds must be provided for all parameter groups");
        }

        if (seenIndices.size !== currentParamCount) {
          throw new DataValidationError(
            `Parameter count mismatch: expected ${currentParamCount}, got ${seenIndices.size}`
          );
        }

        nextGroups = resolvedGroups;
      }
    }

    if (Object.hasOwn(stateDict, "state")) {
      const rawState = stateDict["state"];
      if (!Array.isArray(rawState)) {
        throw new DataValidationError("state must be an array");
      }
      const stateArray: unknown[] = rawState;
      nextState = new Map();
      stateArray.forEach((rawEntry, index) => {
        const entryRecord = ensureRecord(rawEntry, `state[${index}]`);
        if (!Object.hasOwn(entryRecord, "state")) {
          throw new DataValidationError(`state[${index}].state is required`);
        }
        const entryStateValue = ensureRecord(entryRecord["state"], `state[${index}].state`);

        const paramIdRaw = entryRecord["paramId"];
        const paramRaw = entryRecord["param"];

        let resolvedParam: GradTensor | undefined;

        if (paramIdRaw !== undefined) {
          if (typeof paramIdRaw !== "number" || !Number.isInteger(paramIdRaw)) {
            throw new DataValidationError(`Invalid paramId ${String(paramIdRaw)} in state`);
          }
          resolvedParam = currentParams[paramIdRaw];
          if (!resolvedParam) {
            throw new DataValidationError(`Invalid paramId ${paramIdRaw} in state`);
          }
          // A reference from another optimizer is ignored; a reference to one of our own
          // parameters must agree with the id.
          const refIndex = paramRaw === undefined ? undefined : paramLookup.get(paramRaw);
          if (refIndex !== undefined && refIndex !== paramIdRaw) {
            throw new DataValidationError(`paramId ${paramIdRaw} does not match provided param`);
          }
        } else {
          if (paramRaw === undefined) {
            throw new DataValidationError("Missing param reference in state entry");
          }
          const paramIndex = paramLookup.get(paramRaw);
          resolvedParam = paramIndex === undefined ? undefined : currentParams[paramIndex];
          if (!resolvedParam) {
            throw new DataValidationError("Unknown param reference in state entry");
          }
        }

        if (nextState?.has(resolvedParam)) {
          throw new DataValidationError(`Duplicate state entry for state[${index}]`);
        }
        const copied = cloneValue(entryStateValue);
        if (!isRecord(copied) || !this.isState(copied)) {
          throw new DataValidationError(`state[${index}].state has invalid structure`);
        }
        nextState?.set(resolvedParam, copied);
      });
    }

    // Everything validated: apply atomically.
    if (nextStepCount !== undefined) this._stepCount = nextStepCount;
    if (nextGroups) this.paramGroups = nextGroups;
    if (nextState) {
      this.state.clear();
      for (const [param, paramState] of nextState) this.state.set(param, paramState);
    }
  }
}
