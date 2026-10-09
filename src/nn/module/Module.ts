/**
 * Base class for neural network modules: parameter, buffer and child module
 * registration, train/eval mode, hooks, device transfer and state dicts.
 *
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import {
  DeepboxError,
  type Device,
  DeviceError,
  type DType,
  DTypeError,
  InvalidParameterError,
  isBackendAvailable,
  isDevice,
  ShapeError,
  shapesEqual,
} from "../../core";
import { getKernelBackend } from "../../core/backend/registry";
import type { AnyTensor, GradTensor, Tensor } from "../../ndarray";
import { isContiguous, offsetFromFlatIndex } from "../../ndarray/tensor/strides";
import { computeStrides } from "../../ndarray/tensor/Tensor";

type StateEntry = {
  data: Array<number | string | bigint>;
  dtype: DType;
  shape: number[];
};

function sizeFromShape(shape: readonly number[], context: string): number {
  let size = 1;
  for (const dim of shape) {
    if (!Number.isInteger(dim) || dim < 0) {
      throw new ShapeError(`${context} contains invalid dimension ${String(dim)}`);
    }
    size *= dim;
  }
  return size;
}

/**
 * Copy the elements of `t` in logical (row-major) order. Views with a storage
 * offset or non-trivial strides are gathered through their strides, so the
 * result always has exactly `t.size` elements.
 */
function cloneTensorData(t: Tensor): Array<number | string | bigint> {
  const data = t.data;
  const size = t.size;
  if (isContiguous(t.shape, t.strides)) {
    const start = t.offset;
    if (Array.isArray(data)) {
      return data.slice(start, start + size);
    }
    if (data instanceof BigInt64Array) {
      return Array.from(data.subarray(start, start + size));
    }
    const out = new Array<number>(size);
    for (let i = 0; i < size; i++) {
      const value = data[start + i];
      if (value === undefined) {
        throw new DeepboxError("Internal error: tensor data access out of bounds");
      }
      out[i] = value;
    }
    return out;
  }

  const logicalStrides = computeStrides(t.shape);
  const out = new Array<number | string | bigint>(size);
  for (let i = 0; i < size; i++) {
    const value = data[offsetFromFlatIndex(i, logicalStrides, t.strides, t.offset)];
    if (value === undefined) {
      throw new DeepboxError("Internal error: tensor data access out of bounds");
    }
    out[i] = value;
  }
  return out;
}

/**
 * Check that a state entry is well formed and can be loaded into `target`:
 * matching shape and dtype, data length equal to the shape size, and elements
 * of the type the dtype stores. Nothing is written.
 */
function validateStateEntry(
  name: string,
  kind: "parameter" | "buffer",
  target: Tensor,
  entry: StateEntry
): void {
  if (!Array.isArray(entry.shape) || !Array.isArray(entry.data)) {
    throw new InvalidParameterError(
      `${kind} ${name} must be an object with array "shape" and "data" fields`,
      `stateDict.${kind === "parameter" ? "parameters" : "buffers"}`,
      name
    );
  }
  const size = sizeFromShape(entry.shape, `${kind} ${name} shape`);
  if (entry.data.length !== size) {
    throw new ShapeError(
      `${kind} ${name} data length ${entry.data.length} does not match shape size ${size}`
    );
  }
  if (!shapesEqual(target.shape, entry.shape)) {
    throw new ShapeError(
      `${kind} ${name} shape mismatch: expected [${target.shape.join(", ")}], got [${entry.shape.join(", ")}]`
    );
  }
  if (target.dtype !== entry.dtype) {
    throw new DTypeError(
      `${kind} ${name} dtype mismatch: expected ${target.dtype}, got ${entry.dtype}`
    );
  }

  const data = target.data;
  let expected: "string" | "bigint" | "number";
  if (target.dtype === "string") {
    if (!Array.isArray(data)) {
      throw new DTypeError(`${kind} ${name} expected string data`);
    }
    expected = "string";
  } else if (data instanceof BigInt64Array) {
    expected = "bigint";
  } else if (Array.isArray(data)) {
    throw new DTypeError(`${kind} ${name} expected numeric data`);
  } else {
    expected = "number";
  }
  for (let i = 0; i < size; i++) {
    if (typeof entry.data[i] !== expected) {
      throw new DTypeError(`${kind} ${name} expects ${expected} data`);
    }
  }
}

/** Write a state entry that already passed {@link validateStateEntry} into `target`. */
function writeStateEntry(target: Tensor, entry: StateEntry): void {
  const size = entry.data.length;
  const logicalStrides = computeStrides(target.shape);
  const data = target.data;
  const contiguous = isContiguous(target.shape, target.strides);

  for (let i = 0; i < size; i++) {
    const offset = contiguous
      ? target.offset + i
      : offsetFromFlatIndex(i, logicalStrides, target.strides, target.offset);
    const value = entry.data[i];
    if (Array.isArray(data)) {
      data[offset] = value as string;
    } else if (data instanceof BigInt64Array) {
      data[offset] = value as bigint;
    } else {
      data[offset] = value as number;
    }
  }
}

/**
 * Hook function called before the forward pass.
 *
 * @param module - The module being called
 * @param inputs - The input tensors to the forward pass
 * @returns Modified inputs array, or undefined to keep original inputs
 */
export type ForwardPreHook = (module: Module, inputs: AnyTensor[]) => AnyTensor[] | undefined;

/**
 * Hook function called after the forward pass.
 *
 * @param module - The module being called
 * @param inputs - The input tensors to the forward pass
 * @param output - The output tensor from the forward pass
 * @returns Modified output tensor, or undefined to keep original output
 */
export type ForwardHook = (
  module: Module,
  inputs: AnyTensor[],
  output: AnyTensor
) => AnyTensor | undefined;

/**
 * Base class for all neural network modules.
 *
 * All models should subclass this class. Modules can contain other modules,
 * allowing to nest them in a tree structure.
 *
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox Module & Sequential}
 *
 * @example
 * ```ts
 * import { Module, Linear, ReLU } from 'deepbox/nn';
 * import type { AnyTensor, Tensor } from 'deepbox/ndarray';
 *
 * class MyModel extends Module {
 *   private fc1: Linear;
 *   private relu: ReLU;
 *   private fc2: Linear;
 *
 *   constructor() {
 *     super();
 *     this.fc1 = new Linear(10, 5);
 *     this.relu = new ReLU();
 *     this.fc2 = new Linear(5, 2);
 *     this.registerModule('fc1', this.fc1);
 *     this.registerModule('relu', this.relu);
 *     this.registerModule('fc2', this.fc2);
 *   }
 *
 *   forward(x: Tensor): AnyTensor {
 *     let out = this.fc1.forward(x);
 *     out = this.relu.forward(out);
 *     out = this.fc2.forward(out);
 *     return out;
 *   }
 * }
 * ```
 *
 * @category Neural Networks
 */
export abstract class Module {
  /** Child modules registered to this module - stores nested layers/modules */
  protected _modules: Map<string, Module> = new Map();

  /** Parameters of this module - trainable tensors (weights, biases) wrapped as GradTensor */
  protected _parameters: Map<string, GradTensor> = new Map();

  /** Buffers (non-trainable tensors) of this module - e.g., running stats in BatchNorm */
  protected _buffers: Map<string, Tensor> = new Map();

  /** Training mode flag - affects behavior of layers like Dropout and BatchNorm */
  protected _training = true;

  /** Forward pre-hooks registered on this module */
  private _forwardPreHooks: Map<number, ForwardPreHook> = new Map();
  /** Forward hooks registered on this module */
  private _forwardHooks: Map<number, ForwardHook> = new Map();
  /** Incrementing hook id */
  private _nextHookId = 0;

  /**
   * Forward pass of the module.
   *
   * Should be overridden by all subclasses. Accepts either regular Tensors
   * or GradTensors for automatic differentiation support.
   *
   * **Gradient tracking rule of the built-in layers:** a `GradTensor` input gives a
   * `GradTensor`. A plain `Tensor` input gives a `GradTensor` that tracks the weights
   * (the data itself is not tracked) while gradient tracking is on and at least one
   * parameter of the module requires grad, so a training step needs no wrapping of the
   * data. Inside `noGrad()`, or when no parameter requires grad (frozen or parameter-free
   * layers), a plain `Tensor` is returned. `eval()` does not switch tracking off: use
   * `noGrad()` for inference, as in PyTorch.
   *
   * @param inputs - Input tensors (Tensor or GradTensor)
   * @returns Output tensor (Tensor or GradTensor depending on input and layer type)
   *
   * @example
   * ```ts
   * // A plain tensor in training: the result tracks the weights
   * const pred = model.forward(inputTensor);
   * if (GradTensor.isGradTensor(pred)) mseLoss(pred, targets).backward();
   *
   * // Inference without a graph
   * const output = noGrad(() => model.forward(inputTensor)); // plain Tensor
   * ```
   */
  abstract forward(...inputs: AnyTensor[]): AnyTensor;

  /**
   * Run the module: forward pre-hooks, then {@link Module.forward}, then forward hooks.
   *
   * Calling `forward` directly skips the hooks; use `call` when hooks registered with
   * {@link Module.registerForwardPreHook} or {@link Module.registerForwardHook} should run.
   *
   * @param inputs - Input tensors (Tensor or GradTensor)
   * @returns Output of `forward`, possibly replaced by a forward hook
   */
  call(...inputs: AnyTensor[]): AnyTensor {
    let curInputs = inputs;
    for (const hook of this._forwardPreHooks.values()) {
      const result = hook(this, curInputs);
      if (Array.isArray(result)) {
        curInputs = result;
      }
    }
    let output = this.forward(...curInputs);
    for (const hook of this._forwardHooks.values()) {
      const result = hook(this, curInputs, output);
      if (result !== undefined) {
        output = result;
      }
    }
    return output;
  }

  /**
   * Register a child module.
   *
   * @param name - Name of the module
   * @param module - The module to register
   */
  protected registerModule(name: string, module: Module): void {
    Module.assertRegistrationName(name, "module");
    if (module === this) {
      throw new InvalidParameterError(
        "A module cannot be registered as its own child",
        "module",
        name
      );
    }
    // Store the child module in the modules map for hierarchical tracking
    this._modules.set(name, module);
  }

  /**
   * Register a parameter (trainable tensor).
   *
   * Parameters must be GradTensor instances with requiresGrad=true for
   * proper gradient computation during backpropagation.
   *
   * @param name - Name of the parameter
   * @param param - The parameter tensor (must be GradTensor)
   */
  protected registerParameter(name: string, param: GradTensor): void {
    Module.assertRegistrationName(name, "parameter");
    // Register a trainable parameter (weight or bias) for optimization
    this._parameters.set(name, param);
  }

  /** Retrieve a previously registered parameter by name (undefined if absent). */
  protected getParameter(name: string): GradTensor | undefined {
    return this._parameters.get(name);
  }

  /**
   * Register a buffer (non-trainable tensor).
   *
   * Buffers are typically used for running statistics in batch normalization.
   *
   * @param name - Name of the buffer
   * @param buffer - The buffer tensor
   */
  protected registerBuffer(name: string, buffer: Tensor): void {
    Module.assertRegistrationName(name, "buffer");
    // Register a non-trainable buffer (e.g., running mean/variance in BatchNorm)
    // Buffers are saved with the model but not updated by optimizers
    this._buffers.set(name, buffer);
  }

  private static assertRegistrationName(name: string, kind: string): void {
    if (typeof name !== "string" || name.length === 0) {
      throw new InvalidParameterError(`${kind} name must be a non-empty string`, "name", name);
    }
  }

  /**
   * Get all parameters of this module and its children.
   *
   * Returns GradTensor instances that are compatible with optimizers.
   * This enables direct usage with optimizer constructors:
   * ```ts
   * const optimizer = new Adam(model.parameters());
   * ```
   *
   * A parameter registered under several names (tied weights) or reachable through
   * a child module registered twice is yielded once, so an optimizer never updates
   * the same tensor twice per step.
   *
   * @param recurse - Whether to include parameters of child modules
   * @returns Iterator of GradTensor parameters
   */
  *parameters(recurse = true): Generator<GradTensor> {
    for (const [, param] of this.namedParameters("", recurse)) {
      yield param;
    }
  }

  /**
   * Get all named parameters of this module and its children.
   *
   * Names are dot-separated paths (for example `"encoder.fc1.weight"`). A parameter
   * shared under several names is reported once, under the first name found, unless
   * `removeDuplicate` is false.
   *
   * @param prefix - Prefix for parameter names
   * @param recurse - Whether to include parameters of child modules
   * @param removeDuplicate - Skip parameters that were already yielded (default: true)
   * @returns Iterator of [name, parameter] pairs
   */
  *namedParameters(
    prefix = "",
    recurse = true,
    removeDuplicate = true
  ): Generator<[string, GradTensor]> {
    const seen = removeDuplicate ? new Set<GradTensor>() : null;
    for (const [mPrefix, module] of this.namedModules(prefix, recurse, removeDuplicate)) {
      for (const [name, param] of module._parameters.entries()) {
        if (seen) {
          if (seen.has(param)) continue;
          seen.add(param);
        }
        yield [mPrefix ? `${mPrefix}.${name}` : name, param];
      }
    }
  }

  /**
   * Get all child modules.
   *
   * The module itself is yielded first, followed by its descendants in depth-first
   * order. A module reachable through several paths is yielded once.
   *
   * @param recurse - Whether to include nested child modules
   * @returns Iterator of modules
   */
  *modules(recurse = true): Generator<Module> {
    for (const [, module] of this.namedModules("", recurse)) {
      yield module;
    }
  }

  /**
   * Get all named child modules.
   *
   * The first pair is the module itself under `prefix`. A module reachable through
   * several paths is reported once, under the first path found, unless
   * `removeDuplicate` is false (which also assumes the module graph has no cycles).
   *
   * @param prefix - Prefix for module names
   * @param recurse - Whether to include nested child modules
   * @param removeDuplicate - Skip modules that were already yielded (default: true)
   * @returns Iterator of [name, module] pairs
   */
  *namedModules(prefix = "", recurse = true, removeDuplicate = true): Generator<[string, Module]> {
    yield* this.walkModules(prefix, recurse, removeDuplicate ? new Set<Module>() : null);
  }

  private *walkModules(
    prefix: string,
    recurse: boolean,
    seen: Set<Module> | null
  ): Generator<[string, Module]> {
    if (seen) {
      if (seen.has(this)) return;
      seen.add(this);
    }
    yield [prefix, this];
    if (!recurse) return;
    for (const [name, module] of this._modules.entries()) {
      const fullName = prefix ? `${prefix}.${name}` : name;
      yield* module.walkModules(fullName, true, seen);
    }
  }

  /**
   * Get the immediate child modules (not the module itself, no recursion).
   *
   * @returns Iterator of child modules; a module registered under two names is yielded once
   */
  *children(): Generator<Module> {
    for (const [, module] of this.namedChildren()) {
      yield module;
    }
  }

  /**
   * Get the immediate child modules together with their registered names.
   *
   * @returns Iterator of [name, module] pairs; a module registered under two names is reported once
   */
  *namedChildren(): Generator<[string, Module]> {
    const seen = new Set<Module>();
    for (const [name, module] of this._modules.entries()) {
      if (seen.has(module)) continue;
      seen.add(module);
      yield [name, module];
    }
  }

  /**
   * Set the module in training mode.
   *
   * This affects certain layers like Dropout and BatchNorm.
   *
   * @param mode - Training mode (true) or evaluation mode (false)
   * @returns this
   */
  train(mode = true): this {
    // Set training mode for this module
    this._training = mode;

    // Recursively propagate training mode to all child modules
    // This ensures layers like Dropout and BatchNorm behave correctly
    for (const module of this._modules.values()) {
      module.train(mode);
    }

    // Return this for method chaining (e.g., model.train().forward(x))
    return this;
  }

  /**
   * Set the module in evaluation mode.
   *
   * This is equivalent to calling `train(false)`. It changes the behavior of layers such as
   * Dropout and BatchNorm only; it does not stop gradient tracking (wrap inference in
   * `noGrad()` for that, as in PyTorch).
   *
   * @returns this
   */
  eval(): this {
    return this.train(false);
  }

  /**
   * Check if the module is in training mode.
   *
   * @returns true if in training mode
   */
  get training(): boolean {
    return this._training;
  }

  /**
   * Zero out the gradients of all parameters.
   *
   * Call this before each training iteration to prevent gradient accumulation
   * from previous iterations.
   *
   * For parameters wrapped in GradTensor, this calls zeroGrad() on each.
   * For regular Tensors, this is a no-op until they are converted to GradTensor.
   *
   * @example
   * ```ts
   * model.zeroGrad();
   * const output = model.forward(input);
   * // ... compute loss and backward
   * optimizer.step();
   * ```
   */
  zeroGrad(): void {
    // Zero out gradients for all parameters in the module
    // This should be called before each backward pass to prevent gradient accumulation
    for (const param of this.parameters()) {
      // parameters() yields GradTensor instances, so zeroGrad is always available
      param.zeroGrad();
    }
  }

  /**
   * Get all buffers of this module and its children.
   *
   * @param recurse - Whether to include buffers of child modules
   * @returns Iterator of buffer tensors (a buffer shared by several modules is yielded once)
   */
  *buffers(recurse = true): Generator<Tensor> {
    for (const [, buffer] of this.namedBuffers("", recurse)) {
      yield buffer;
    }
  }

  /**
   * Get all named buffers of this module and its children.
   *
   * @param prefix - Prefix for buffer names
   * @param recurse - Whether to include buffers of child modules
   * @param removeDuplicate - Skip buffers that were already yielded (default: true)
   * @returns Iterator of [name, buffer] pairs
   */
  *namedBuffers(prefix = "", recurse = true, removeDuplicate = true): Generator<[string, Tensor]> {
    const seen = removeDuplicate ? new Set<Tensor>() : null;
    for (const [mPrefix, module] of this.namedModules(prefix, recurse, removeDuplicate)) {
      for (const [name, buffer] of module._buffers.entries()) {
        if (seen) {
          if (seen.has(buffer)) continue;
          seen.add(buffer);
        }
        yield [mPrefix ? `${mPrefix}.${name}` : name, buffer];
      }
    }
  }

  /**
   * Freeze specific parameters by name (or all if none provided).
   *
   * Frozen parameters have `requiresGrad = false` and their stored gradient is
   * cleared. The parameter objects themselves are kept, so references held by
   * the model, containers and optimizers stay valid. Optimizers skip parameters
   * with `requiresGrad = false` or without a gradient, as PyTorch does. You can
   * also build them from the trainable subset:
   * `new Adam([...model.parameters()].filter((p) => p.requiresGrad))`.
   *
   * @param names - Array of parameter names to freeze (e.g., ['fc1.weight']). If undefined, freezes all parameters.
   * @param recurse - When `names` is omitted, whether to include parameters from child modules (default: true). Explicit names are always resolved through child modules.
   * @throws {InvalidParameterError} If a name in `names` does not match a parameter
   *
   * @example
   * ```ts
   * const model = new MyModel();
   * // Freeze only the first layer's weights
   * model.freezeParameters(['fc1.weight']);
   * ```
   */
  freezeParameters(names?: string[], recurse = true): void {
    this.setRequiresGradForNames(names, false, recurse);
  }

  /**
   * Unfreeze specific parameters by name (or all if none provided).
   *
   * Sets `requiresGrad = true` on the selected parameters in place, so references
   * held by an optimizer stay valid.
   *
   * @param names - Array of parameter names to unfreeze (e.g., ['fc1.weight']). If undefined, unfreezes all parameters.
   * @param recurse - When `names` is omitted, whether to include parameters from child modules (default: true). Explicit names are always resolved through child modules.
   * @throws {InvalidParameterError} If a name in `names` does not match a parameter
   *
   * @example
   * ```ts
   * const model = new MyModel();
   * model.freezeParameters(); // Freeze all
   * model.unfreezeParameters(['fc2.weight']); // Unfreeze only fc2 weights
   * ```
   */
  unfreezeParameters(names?: string[], recurse = true): void {
    this.setRequiresGradForNames(names, true, recurse);
  }

  private setRequiresGradForNames(
    names: string[] | undefined,
    requiresGrad: boolean,
    recurse: boolean
  ): void {
    const targets: GradTensor[] = [];
    if (names === undefined) {
      for (const [, param] of this.namedParameters("", recurse)) {
        targets.push(param);
      }
    } else {
      // Resolve every name first so an unknown name leaves all parameters untouched.
      for (const name of names) {
        const param = this.findParameter(name);
        if (!param) {
          throw new InvalidParameterError(`Unknown parameter name: ${name}`, "names", name);
        }
        targets.push(param);
      }
    }
    for (const param of targets) {
      if (requiresGrad) {
        param.requiresGrad = true;
      } else {
        param.setRequiresGrad(false);
      }
    }
  }

  /**
   * Look up a parameter by its dot-separated path. Child module and parameter names
   * that themselves contain dots (for example `"layers.0"`) are matched as well.
   */
  private findParameter(fullName: string): GradTensor | undefined {
    const parts = fullName.split(".");
    const search = (module: Module, start: number): GradTensor | undefined => {
      const own = module._parameters.get(parts.slice(start).join("."));
      if (own) return own;
      for (let end = start + 1; end < parts.length; end++) {
        const child = module._modules.get(parts.slice(start, end).join("."));
        if (child) {
          const found = search(child, end);
          if (found) return found;
        }
      }
      return undefined;
    };
    return search(this, 0);
  }

  /**
   * Get the state dictionary of the module.
   *
   * Every parameter and buffer is copied in row-major order (views are gathered
   * through their strides), so the result does not alias the live tensors. A
   * parameter shared under several names appears under each name.
   *
   * @returns Plain objects keyed by dot-separated name with `data`, `shape` and `dtype`
   */
  stateDict(): {
    parameters: Record<string, StateEntry>;
    buffers: Record<string, StateEntry>;
  } {
    const parameters: Record<string, StateEntry> = {};
    const buffers: Record<string, StateEntry> = {};

    for (const [name, param] of this.namedParameters("", true, false)) {
      const t = param.tensor;
      parameters[name] = {
        data: cloneTensorData(t),
        shape: [...t.shape],
        dtype: t.dtype,
      };
    }

    for (const [name, buffer] of this.namedBuffers("", true, false)) {
      buffers[name] = {
        data: cloneTensorData(buffer),
        shape: [...buffer.shape],
        dtype: buffer.dtype,
      };
    }

    return { parameters, buffers };
  }

  /**
   * Load state dictionary into the module.
   *
   * Every entry is validated (names, shapes, dtypes, element types) before any
   * tensor is written, so a failed load leaves the module unchanged.
   *
   * @param stateDict - Object produced by {@link Module.stateDict}
   * @throws {InvalidParameterError} If a parameter or buffer is missing or unexpected
   * @throws {ShapeError} If an entry's shape or data length does not match
   * @throws {DTypeError} If an entry's dtype or element type does not match
   */
  loadStateDict(stateDict: {
    parameters?: Record<string, StateEntry>;
    buffers?: Record<string, StateEntry>;
  }): void {
    const parameters = stateDict.parameters ?? {};
    const buffers = stateDict.buffers ?? {};
    const has = (record: object, key: string): boolean => Object.hasOwn(record, key);

    const namedParams = new Map(this.namedParameters("", true, false));
    const namedBuffs = new Map(this.namedBuffers("", true, false));

    for (const name of namedParams.keys()) {
      if (!has(parameters, name)) {
        throw new InvalidParameterError(`missing parameter: ${name}`, "stateDict.parameters", name);
      }
    }

    for (const name of namedBuffs.keys()) {
      if (!has(buffers, name)) {
        throw new InvalidParameterError(`missing buffer: ${name}`, "stateDict.buffers", name);
      }
    }

    for (const name of Object.keys(parameters)) {
      if (!namedParams.has(name)) {
        throw new InvalidParameterError(
          `unexpected parameter: ${name}`,
          "stateDict.parameters",
          name
        );
      }
    }

    for (const name of Object.keys(buffers)) {
      if (!namedBuffs.has(name)) {
        throw new InvalidParameterError(`unexpected buffer: ${name}`, "stateDict.buffers", name);
      }
    }

    const writes: Array<[Tensor, StateEntry]> = [];
    for (const [name, entry] of Object.entries(parameters)) {
      const param = namedParams.get(name);
      if (!param) continue;
      validateStateEntry(name, "parameter", param.tensor, entry);
      writes.push([param.tensor, entry]);
    }
    for (const [name, entry] of Object.entries(buffers)) {
      const buffer = namedBuffs.get(name);
      if (!buffer) continue;
      validateStateEntry(name, "buffer", buffer, entry);
      writes.push([buffer, entry]);
    }
    for (const [target, entry] of writes) {
      writeStateEntry(target, entry);
    }
  }

  /**
   * Move the module's parameters and buffers to a device.
   *
   * Transfers the underlying tensor data (uploading to device memory for
   * kernel devices like `webgpu`, downloading when moving back to `cpu`).
   * Existing parameter gradients move along with their parameters. The
   * device/backend validation happens synchronously (invalid devices throw
   * immediately); the data transfer itself is asynchronous because device
   * readback cannot block the JavaScript thread.
   *
   * Kernel devices execute float32 only, so non-float32 parameters and
   * buffers (e.g. integer bookkeeping buffers) stay in host memory, as they
   * are consumed by host-side code paths. Layers that only have host kernels
   * (`Embedding`, `Conv3d`, `ConvTranspose1d`, `ConvTranspose2d`, the recurrent
   * layers, `PReLU` and `SpectralNorm`) keep their weights on the host as well.
   *
   * @param device - Target device identifier (e.g., 'cpu', 'webgpu', 'wasm')
   * @returns Promise resolving to this module for chaining
   * @throws {InvalidParameterError} If the device identifier is unknown
   * @throws {DeviceError} If no backend is registered/available for the device
   *
   * @example
   * ```ts
   * const model = new Linear(10, 5);
   * await model.to('webgpu'); // parameters now live in GPU memory
   * await model.to('cpu');    // ...and back
   * ```
   */
  to(device: Device): Promise<this> {
    if (!isDevice(device)) {
      throw new InvalidParameterError("device must be one of: cpu, webgpu, wasm", "device", device);
    }

    if (!isBackendAvailable(device)) {
      throw new DeviceError(
        `No backend available for device "${device}". ` +
          "Register one first (e.g. `registerBackend('webgpu', gpu)` after `await gpu.init()`). " +
          "See https://deepbox.dev/docs/devices-and-execution for backends and the accelerated op set."
      );
    }

    return this.moveTo(device);
  }

  /**
   * Whether {@link Module.to} leaves this module's parameters and buffers (and those of its
   * children) in host memory.
   *
   * Layers whose forward pass uses hand-written host kernels, such as `Embedding`, `Conv3d`,
   * `ConvTranspose1d` and `ConvTranspose2d`, override this to return `true`: their weights stay
   * on the CPU so that the layer keeps working with host tensors after `model.to(device)`
   * instead of failing when a kernel reads device memory. The default is `false`.
   */
  protected keepsParametersOnHost(): boolean {
    return false;
  }

  private async moveTo(device: Device): Promise<this> {
    // Kernel devices hold float32 buffers (plus half-precision float16 /
    // bfloat16); other dtypes stay on the host.
    const kernelDevice = getKernelBackend(device) !== null;

    const movable = (t: Tensor): boolean =>
      !kernelDevice || t.dtype === "float32" || t.dtype === "float16" || t.dtype === "bfloat16";

    // Modules whose kernels run on the host keep their whole subtree in host memory.
    const hostOnly = new Set<Module>();
    for (const module of this.modules()) {
      if (module.keepsParametersOnHost()) {
        for (const inner of module.modules()) hostOnly.add(inner);
      }
    }

    for (const module of this.modules()) {
      if (hostOnly.has(module)) continue;
      for (const param of module._parameters.values()) {
        if (!movable(param.tensor)) continue;
        const moved = await param.tensor.to(device);
        if (moved !== param.tensor) {
          Module.replaceGradTensorStorage(param, "tensor", moved);
        }
        const grad = param.grad;
        if (grad && movable(grad)) {
          const movedGrad = await grad.to(device);
          if (movedGrad !== grad) {
            Module.replaceGradTensorStorage(param, "_grad", movedGrad);
          }
        }
      }
      for (const [name, buffer] of module._buffers.entries()) {
        if (!movable(buffer)) continue;
        const moved = await buffer.to(device);
        if (moved !== buffer) {
          module._buffers.set(name, moved);
        }
      }
    }
    return this;
  }

  private static replaceGradTensorStorage(
    target: GradTensor,
    field: "tensor" | "_grad",
    value: Tensor
  ): void {
    if (!Reflect.set(target, field, value)) {
      throw new DeepboxError("Failed to move parameter tensor to the target device");
    }
  }

  /**
   * Apply a function to this module and every descendant, parents before children.
   *
   * @param fn - Callback invoked once per module
   * @returns this
   *
   * @example
   * ```ts
   * model.apply((m) => console.log(m.constructor.name));
   * ```
   */
  apply(fn: (module: Module) => void): this {
    for (const module of this.modules()) {
      fn(module);
    }
    return this;
  }

  /**
   * Register a forward pre-hook, run by {@link Module.call} before `forward`.
   *
   * @param hook - Receives the module and its inputs; may return replacement inputs
   * @returns Function that removes the hook
   */
  registerForwardPreHook(hook: ForwardPreHook): () => void {
    const hookId = this._nextHookId++;
    this._forwardPreHooks.set(hookId, hook);
    return () => {
      this._forwardPreHooks.delete(hookId);
    };
  }

  /**
   * Register a forward hook, run by {@link Module.call} after `forward`.
   *
   * @param hook - Receives the module, its inputs and the output; may return a replacement output
   * @returns Function that removes the hook
   */
  registerForwardHook(hook: ForwardHook): () => void {
    const hookId = this._nextHookId++;
    this._forwardHooks.set(hookId, hook);
    return () => {
      this._forwardHooks.delete(hookId);
    };
  }

  /**
   * Get string representation of the module.
   *
   * @returns Hierarchical string representation showing module structure
   */
  toString(): string {
    const lines = [`${this.constructor.name}(`];

    // Iterate through child modules and format them with indentation
    for (const [name, module] of this._modules.entries()) {
      // Recursively get child module's string representation
      const childLines = module.toString().split("\n");
      // First line goes on the same line as the name; subsequent lines are indented
      const moduleStr = childLines.map((line, i) => (i === 0 ? line : `  ${line}`)).join("\n");
      // Format as: (name): ModuleType(...)
      lines.push(`  (${name}): ${moduleStr}`);
    }

    lines.push(")");
    return lines.join("\n");
  }

  /**
   * Build a text summary of the model: one row per own parameter and per child
   * module with its parameter count, followed by total, trainable and
   * non-trainable counts. Parameters shared between modules are counted once in
   * the totals.
   *
   * @returns Formatted summary string
   */
  summary(): string {
    const rows: { name: string; type: string; params: number }[] = [];

    for (const [pName, param] of this._parameters.entries()) {
      rows.push({ name: pName, type: "(parameter)", params: param.tensor.size });
    }

    for (const [mName, module] of this._modules.entries()) {
      let modParams = 0;
      for (const p of module.parameters(true)) {
        modParams += p.tensor.size;
      }
      rows.push({
        name: mName,
        type: module.constructor.name,
        params: modParams,
      });
    }

    let totalParams = 0;
    let trainableParams = 0;
    for (const p of this.parameters(true)) {
      totalParams += p.tensor.size;
      if (p.requiresGrad) trainableParams += p.tensor.size;
    }

    const sep = "-".repeat(60);
    const lines: string[] = [
      sep,
      `${this.constructor.name} Summary`,
      sep,
      `${"Layer".padEnd(25)} ${"Type".padEnd(20)} ${"Params".padStart(10)}`,
      sep,
    ];

    for (const row of rows) {
      lines.push(
        `${row.name.padEnd(25)} ${row.type.padEnd(20)} ${String(row.params).padStart(10)}`
      );
    }

    lines.push(sep);
    lines.push(`Total params: ${totalParams}`);
    lines.push(`Trainable params: ${trainableParams}`);
    lines.push(`Non-trainable params: ${totalParams - trainableParams}`);
    lines.push(sep);

    return lines.join("\n");
  }
}
