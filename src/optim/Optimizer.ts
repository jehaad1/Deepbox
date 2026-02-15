import { DataValidationError } from "../core";
import type { GradTensor } from "../ndarray";

/**
 * Base class for all optimizers.
 *
 * This abstract class provides the foundation for implementing optimization algorithms
 * used in training machine learning models. All concrete optimizers (SGD, Adam, etc.)
 * must extend this class and implement the abstract `step()` method.
 *
 * **Key Features:**
 * - Parameter groups with per-group hyperparameters
 * - State management for stateful optimizers (momentum, adaptive learning rates)
 * - Gradient zeroing utilities
 * - State serialization for checkpointing
 *
 * **Design Pattern:**
 * The optimizer maintains a list of parameter groups, where each group can have
 * different hyperparameters (e.g., different learning rates for different layers).
 * This enables fine-grained control over the optimization process.
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
 * }
 * ```
 *
 * @example
 * ```ts
 * // Using parameter groups with different learning rates
 * const optimizer = new SGD([
 *   { params: model.layer1.parameters(), lr: 0.01 },
 *   { params: model.layer2.parameters(), lr: 0.001 }
 * ], { lr: 0.01 });
 * ```
 *
 * References:
 * - Deepbox Optimizers: https://deepbox.dev/docs/optim-optimizers
 *
 * @category Optimization
 */

/**
 * Represents a group of parameters with optional per-group hyperparameters.
 *
 * @template Options - Type of optimizer-specific options
 * @property params - Iterable of parameters to optimize in this group
 */
export type ParamGroup<Options extends Record<string, unknown>> = {
  readonly params: Iterable<GradTensor>;
} & Partial<Options>;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function ensureRecord(value: unknown, context: string) {
  if (!isRecord(value)) {
    throw new DataValidationError(`${context} must be an object`);
  }
  return value;
}

function isStateRecord(value: unknown): value is Record<string, unknown> {
  return isRecord(value);
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
  // Check if params is an array (parameter groups must be arrays)
  if (!Array.isArray(params)) return false;
  // Empty array is considered a valid parameter group array
  if (params.length === 0) return true;
  // Check if first element has a 'params' property (indicating it's a group)
  const first = params[0];
  if (!first || typeof first !== "object") return false;
  return "params" in first;
}

/**
 * Abstract base class for all optimization algorithms.
 *
 * @template Options - Type defining optimizer-specific hyperparameters
 * @template State - Type defining per-parameter state (e.g., momentum buffers)
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
   * Get the current learning rate of the first parameter group.
   * Convenience property for optimizers with a single group.
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
   * Initializes the optimizer with either a simple list of parameters or
   * multiple parameter groups with per-group hyperparameters.
   *
   * @param params - Either an iterable of parameters or array of parameter groups
   * @param defaults - Default hyperparameters applied to all groups
   */
  constructor(
    params: Iterable<GradTensor> | ReadonlyArray<ParamGroup<Options>>,
    protected readonly defaults: Readonly<Options>
  ) {
    // Initialize empty parameter groups array
    this.paramGroups = [];

    // Handle both simple param list and param groups
    if (!isParamGroupArray<Options>(params)) {
      // Simple iterable of parameters - create single group with default options
      this.paramGroups.push({
        params: Array.from(params), // Convert iterable to array for efficient access
        options: { ...defaults }, // Clone defaults to avoid mutation
      });
    } else {
      // Array of parameter groups - create group for each with merged options
      for (const group of params) {
        // Destructure to separate params from group-specific options
        const { params: groupParams, ...groupOptions } = group;
        this.paramGroups.push({
          params: Array.from(groupParams), // Convert to array
          options: { ...defaults, ...groupOptions }, // Merge defaults with group options
        });
      }
    }
  }

  /**
   * Perform a single optimization step (parameter update).
   *
   * This abstract method must be implemented by all optimizer subclasses.
   * It applies the optimization algorithm to update all parameters based on
   * their gradients.
   *
   * @param closure - Optional closure that reevaluates the model and returns the loss.
   *                  Used by some optimizers (e.g., LBFGS) that require multiple
   *                  function evaluations per step.
   * @returns Loss value if closure is provided, undefined otherwise
   */
  abstract step(closure?: () => number): number | undefined;

  /**
   * Zero out the gradients of all optimized parameters.
   *
   * This method should be called at the beginning of each training iteration,
   * before computing new gradients. Without this call, gradients would accumulate
   * across iterations, leading to incorrect updates.
   *
   * **Implementation Note:**
   * For parameters wrapped in GradTensor, this calls zeroGrad() on each parameter,
   * which either sets the gradient to zero or initializes it if not yet created.
   *
   * @example
   * ```ts
   * // Typical training loop
   * optimizer.zeroGrad();              // Clear previous gradients
   * const output = model.forward(input);
   * const loss = criterion(output, target);
   * loss.backward();                   // Compute new gradients
   * optimizer.step();                  // Update parameters
   * ```
   */
  zeroGrad(): void {
    // Iterate through all parameter groups
    for (const group of this.paramGroups) {
      // Zero gradient for each parameter in the group
      for (const param of group.params) {
        param.zeroGrad(); // Delegate to GradTensor's zeroGrad method
      }
    }
  }

  /**
   * Add a parameter group to the optimizer.
   *
   * This method allows adding new parameters to optimize after the optimizer
   * has been created. This is particularly useful for:
   * - Fine-tuning: adding pre-trained layers with different learning rates
   * - Progressive training: gradually unfreezing layers
   * - Dynamic architectures: adding parameters while the model grows
   *
   * @param paramGroup - Parameter group to add with optional per-group options
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
    // Destructure to separate params from group-specific options
    const { params, ...options } = paramGroup;
    // Add new group with merged options (defaults + group-specific)
    this.paramGroups.push({
      params: Array.from(params), // Convert iterable to array
      options: { ...this.defaults, ...options }, // Merge with defaults
    });
  }

  /**
   * Validate that a given state object matches the optimizer's state type.
   *
   * @param state - The state object to validate
   * @returns True if the state object is valid, false otherwise
   */
  protected abstract isState(state: Record<string, unknown>): state is State;

  /**
   * Get the current state of the optimizer.
   *
   * Returns a dictionary containing all optimizer state that needs to be
   * saved for checkpointing. This includes per-parameter state (momentum buffers,
   * adaptive learning rates, etc.) and parameter group configurations.
   *
   * **Note:** In a production implementation, parameters would be identified by
   * unique IDs rather than object references for proper serialization.
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
    const paramIdMap = new Map<GradTensor, number>();
    const orderedParams: GradTensor[] = [];
    const getParamId = (param: GradTensor) => {
      const existing = paramIdMap.get(param);
      if (existing !== undefined) return existing;
      const id = orderedParams.length;
      orderedParams.push(param);
      paramIdMap.set(param, id);
      return id;
    };

    return {
      // Serialize per-parameter state
      state: Array.from(this.state.entries()).map(([param, state]) => ({
        paramId: getParamId(param),
        param: param, // Backward-compatible references
        state, // Optimizer-specific state (momentum, etc.)
      })),
      // Serialize parameter groups and their options
      paramGroups: this.paramGroups.map((group) => ({
        params: group.params, // Backward-compatible references
        paramIds: group.params.map((param) => getParamId(param)),
        options: group.options, // Hyperparameters for this group
      })),
    };
  }

  /**
   * Load optimizer state from a state dictionary.
   *
   * Restores the optimizer to a previously saved state, including all
   * per-parameter state and parameter group configurations. This is essential
   * for resuming training from checkpoints.
   *
   * **Important:** The loaded state must be compatible with the current
   * optimizer configuration (same parameters, same optimizer type).
   *
   * @param stateDict - State dictionary previously returned by stateDict()
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
    const currentParams = this.paramGroups.flatMap((group) => group.params);
    const currentParamCount = currentParams.length;
    const paramLookup = new Map<unknown, number>();
    for (let i = 0; i < currentParams.length; i++) {
      paramLookup.set(currentParams[i], i);
    }

    // Validate paramGroups if present
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
        this.paramGroups = [];
      } else {
        if (groupsArray.length !== this.paramGroups.length) {
          throw new DataValidationError("paramGroups count mismatch");
        }

        const seenParamIds = new Set<number>();
        let totalParamCount = 0;
        let sawParamIds = false;
        let sawNoParamIds = false;
        const nextGroups: Array<{
          params: GradTensor[];
          options: Options;
          paramIds?: number[];
        }> = [];

        groupsArray.forEach((rawGroup, index) => {
          const groupRecord = ensureRecord(rawGroup, `paramGroups[${index}]`);
          const optionsRaw = ensureRecord(groupRecord["options"], `paramGroups[${index}].options`);
          const options: Options = { ...this.defaults };
          const optionsRecord: Record<string, unknown> = options;
          const defaultsRecord: Record<string, unknown> = { ...this.defaults };

          for (const [key, value] of Object.entries(optionsRaw)) {
            if (Object.hasOwn(defaultsRecord, key)) {
              const defaultVal = defaultsRecord[key];
              const expectedType = typeof defaultVal;
              const actualType = typeof value;

              if (actualType !== expectedType) {
                throw new DataValidationError(
                  `Type mismatch for option '${key}' in paramGroups[${index}]: expected ${expectedType}, got ${actualType}`
                );
              }
              optionsRecord[key] = value;
            }
          }

          const paramIdsRaw = groupRecord["paramIds"];
          const paramsRaw = groupRecord["params"];
          let paramIds: number[] | undefined;
          if (paramIdsRaw !== undefined) {
            paramIds = ensureIntegerArray(paramIdsRaw, `paramGroups[${index}].paramIds`);
            sawParamIds = true;
          } else {
            sawNoParamIds = true;
          }

          let resolvedParams: GradTensor[] | undefined;

          if (paramIds) {
            for (const id of paramIds) {
              if (id < 0 || id >= currentParamCount) {
                throw new DataValidationError(`Invalid paramId ${id} in paramGroups`);
              }
              if (seenParamIds.has(id)) {
                throw new DataValidationError(`Duplicate paramId ${id} in paramGroups`);
              }
              seenParamIds.add(id);
            }
            totalParamCount += paramIds.length;
            resolvedParams = paramIds.map((id) => {
              const param = currentParams[id];
              if (!param) {
                throw new DataValidationError(`Invalid paramId ${id} in paramGroups`);
              }
              return param;
            });
          }

          if (paramsRaw !== undefined) {
            if (!Array.isArray(paramsRaw)) {
              throw new DataValidationError(`paramGroups[${index}].params must be an array`);
            }
            const resolvedFromParams: GradTensor[] = [];
            let hasUnknown = false;
            for (const paramRef of paramsRaw) {
              const paramIndex = paramLookup.get(paramRef);
              if (paramIndex === undefined) {
                hasUnknown = true;
                continue;
              }
              const param = currentParams[paramIndex];
              if (!param) {
                hasUnknown = true;
                continue;
              }
              resolvedFromParams.push(param);
            }
            if (!hasUnknown) {
              if (paramIds && paramIds.length !== resolvedFromParams.length) {
                throw new DataValidationError("paramIds length does not match params length");
              }
              if (!resolvedParams) {
                resolvedParams = resolvedFromParams;
              }
            }
          }

          if (!resolvedParams) {
            throw new DataValidationError(`paramGroups[${index}] must include params or paramIds`);
          }

          if (paramIds === undefined) {
            nextGroups.push({ params: resolvedParams, options });
          } else {
            nextGroups.push({ params: resolvedParams, options, paramIds });
          }
        });

        if (sawParamIds && sawNoParamIds) {
          throw new DataValidationError("paramIds must be provided for all parameter groups");
        }

        if (sawParamIds && totalParamCount !== currentParamCount) {
          throw new DataValidationError(
            `Parameter count mismatch: expected ${currentParamCount}, got ${totalParamCount}`
          );
        }

        this.paramGroups = nextGroups.map((group) => ({
          params: group.params,
          options: group.options,
        }));
      }
    }

    // Load per-parameter state if present in state dict
    if (Object.hasOwn(stateDict, "state")) {
      const rawState = stateDict["state"];
      if (!Array.isArray(rawState)) {
        throw new DataValidationError("state must be an array");
      }
      const stateArray: unknown[] = rawState;
      // Clear existing state before loading
      this.state.clear();
      // Restore each parameter's state
      stateArray.forEach((rawEntry, index) => {
        const entryRecord = ensureRecord(rawEntry, `state[${index}]`);
        if (!Object.hasOwn(entryRecord, "state")) {
          throw new DataValidationError(`state[${index}].state is required`);
        }
        const entryStateValue = ensureRecord(entryRecord["state"], `state[${index}].state`);
        if (!isStateRecord(entryStateValue)) {
          throw new DataValidationError(`state[${index}].state must be an object`);
        }

        const paramIdRaw = entryRecord["paramId"];
        const paramRaw = entryRecord["param"];

        let resolvedParam: GradTensor | undefined;

        if (paramIdRaw !== undefined) {
          if (
            paramIdRaw === null ||
            typeof paramIdRaw !== "number" ||
            !Number.isInteger(paramIdRaw)
          ) {
            throw new DataValidationError(`Invalid paramId ${String(paramIdRaw)} in state`);
          }
          if (paramIdRaw < 0 || paramIdRaw >= currentParamCount) {
            throw new DataValidationError(`Invalid paramId ${paramIdRaw} in state`);
          }
          const param = currentParams[paramIdRaw];
          if (!param) {
            throw new DataValidationError(`Invalid paramId ${paramIdRaw} in state`);
          }
          if (paramRaw !== undefined) {
            const paramIndex = paramLookup.get(paramRaw);
            if (paramIndex === undefined || paramIndex !== paramIdRaw) {
              throw new DataValidationError(`paramId ${paramIdRaw} does not match provided param`);
            }
          }
          resolvedParam = param;
        } else {
          if (paramRaw === undefined) {
            throw new DataValidationError("Missing param reference in state entry");
          }
          const paramIndex = paramLookup.get(paramRaw);
          if (paramIndex === undefined) {
            throw new DataValidationError("Unknown param reference in state entry");
          }
          const param = currentParams[paramIndex];
          if (!param) {
            throw new DataValidationError("Unknown param reference in state entry");
          }
          resolvedParam = param;
        }

        if (!resolvedParam) {
          throw new DataValidationError(`Unable to resolve parameter for state[${index}]`);
        }
        if (!this.isState(entryStateValue)) {
          throw new DataValidationError(`state[${index}].state has invalid structure`);
        }
        this.state.set(resolvedParam, entryStateValue);
      });
    }
  }
}
