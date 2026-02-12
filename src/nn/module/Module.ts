import {
  DeepboxError,
  type Device,
  type DType,
  DTypeError,
  InvalidParameterError,
  isDevice,
  ShapeError,
} from "../../core";
import { type AnyTensor, GradTensor, type Tensor } from "../../ndarray";
import { offsetFromFlatIndex } from "../../ndarray/tensor/strides";
import { computeStrides } from "../../ndarray/tensor/Tensor";

type StateEntry = {
  data: Array<number | string | bigint>;
  dtype: DType;
  shape: number[];
};

function shapesEqual(a: readonly number[], b: readonly number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if ((a[i] ?? 0) !== (b[i] ?? 0)) return false;
  }
  return true;
}

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

function cloneTensorData(t: Tensor): Array<number | string | bigint> {
  const data = t.data;
  if (Array.isArray(data)) {
    return data.slice();
  }
  if (data instanceof BigInt64Array) {
    return Array.from(data);
  }
  const out = new Array<number>(data.length);
  for (let i = 0; i < data.length; i++) {
    const value = data[i];
    if (value === undefined) {
      throw new DeepboxError("Internal error: tensor data access out of bounds");
    }
    out[i] = value;
  }
  return out;
}

function validateStateEntryShape(
  name: string,
  kind: "parameter" | "buffer",
  entry: StateEntry
): void {
  const size = sizeFromShape(entry.shape, `${kind} ${name} shape`);
  if (entry.data.length !== size) {
    throw new ShapeError(
      `${kind} ${name} data length ${entry.data.length} does not match shape size ${size}`
    );
  }
}

function copyStateEntryIntoTensor(
  name: string,
  kind: "parameter" | "buffer",
  target: Tensor,
  entry: StateEntry
): void {
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

  const size = sizeFromShape(entry.shape, `${kind} ${name} shape`);
  const logicalStrides = computeStrides(target.shape);
  const data = target.data;

  if (target.dtype === "string") {
    if (!Array.isArray(data)) {
      throw new DTypeError(`${kind} ${name} expected string data`);
    }
    for (let i = 0; i < size; i++) {
      const value = entry.data[i];
      if (typeof value !== "string") {
        throw new DTypeError(`${kind} ${name} expects string data`);
      }
      const offset = offsetFromFlatIndex(i, logicalStrides, target.strides, target.offset);
      data[offset] = value;
    }
    return;
  }

  if (data instanceof BigInt64Array) {
    for (let i = 0; i < size; i++) {
      const value = entry.data[i];
      if (typeof value !== "bigint") {
        throw new DTypeError(`${kind} ${name} expects bigint data`);
      }
      const offset = offsetFromFlatIndex(i, logicalStrides, target.strides, target.offset);
      data[offset] = value;
    }
    return;
  }

  if (Array.isArray(data)) {
    throw new DTypeError(`${kind} ${name} expected numeric data`);
  }

  for (let i = 0; i < size; i++) {
    const value = entry.data[i];
    if (typeof value !== "number") {
      throw new DTypeError(`${kind} ${name} expects numeric data`);
    }
    const offset = offsetFromFlatIndex(i, logicalStrides, target.strides, target.offset);
    data[offset] = value;
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
 * This is analogous to PyTorch's nn.Module.
 *
 * @example
 * ```ts
 * import { Module, Linear, ReLU } from 'deepbox/nn';
 * import type { Tensor } from 'deepbox/ndarray';
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
 *   forward(x: Tensor): Tensor {
 *     let out = this.fc1.forward(x);
 *     out = this.relu.forward(out);
 *     out = this.fc2.forward(out);
 *     return out;
 *   }
 * }
 * ```
 *
 * References:
 * - PyTorch nn.Module: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
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
   * @param inputs - Input tensors (Tensor or GradTensor)
   * @returns Output tensor (Tensor or GradTensor depending on input and layer type)
   *
   * @example
   * ```ts
   * // Using with regular Tensor
   * const output = model.forward(inputTensor);
   *
   * // Using with GradTensor for training
   * const gradOutput = model.forward(gradInput);
   * gradOutput.backward();
   * ```
   */
  abstract forward(...inputs: AnyTensor[]): AnyTensor;

  /**
   * Makes the module callable (allows using `module(x)` instead of `module.forward(x)`).
   *
   * @param inputs - Input tensors (Tensor or GradTensor)
   * @returns Output tensor
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
    // Register a trainable parameter (weight or bias) for optimization
    this._parameters.set(name, param);
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
    // Register a non-trainable buffer (e.g., running mean/variance in BatchNorm)
    // Buffers are saved with the model but not updated by optimizers
    this._buffers.set(name, buffer);
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
   * @param recurse - Whether to include parameters of child modules
   * @returns Iterator of GradTensor parameters
   */
  *parameters(recurse = true): Generator<GradTensor> {
    // Yield own parameters first
    for (const param of this._parameters.values()) {
      yield param;
    }

    // Recursively yield child module parameters if requested
    // This allows optimizers to access all trainable parameters in the model
    if (recurse) {
      for (const module of this._modules.values()) {
        yield* module.parameters(true);
      }
    }
  }

  /**
   * Get all named parameters of this module and its children.
   *
   * @param prefix - Prefix for parameter names
   * @param recurse - Whether to include parameters of child modules
   * @returns Iterator of [name, parameter] pairs
   */
  *namedParameters(prefix = "", recurse = true): Generator<[string, GradTensor]> {
    // Yield own parameters with hierarchical naming (e.g., "fc1.weight")
    for (const [name, param] of this._parameters.entries()) {
      // Build full parameter name with dot notation for nested modules
      const fullName = prefix ? `${prefix}.${name}` : name;
      yield [fullName, param];
    }

    // Recursively yield child module parameters with proper prefixing
    if (recurse) {
      for (const [moduleName, module] of this._modules.entries()) {
        // Extend prefix for nested modules (e.g., "encoder.fc1")
        const fullPrefix = prefix ? `${prefix}.${moduleName}` : moduleName;
        yield* module.namedParameters(fullPrefix, true);
      }
    }
  }

  /**
   * Get all child modules.
   *
   * @param recurse - Whether to include nested child modules
   * @returns Iterator of modules
   */
  *modules(recurse = true): Generator<Module> {
    // Always yield self first (root of the module tree)
    yield this;

    // Recursively yield all child modules in depth-first order
    if (recurse) {
      for (const module of this._modules.values()) {
        yield* module.modules(true);
      }
    }
  }

  /**
   * Get all named child modules.
   *
   * @param prefix - Prefix for module names
   * @param recurse - Whether to include nested child modules
   * @returns Iterator of [name, module] pairs
   */
  *namedModules(prefix = "", recurse = true): Generator<[string, Module]> {
    // Yield self with current prefix (empty string for root)
    yield [prefix, this];

    // Recursively yield child modules with hierarchical naming
    if (recurse) {
      for (const [name, module] of this._modules.entries()) {
        // Build full module path (e.g., "encoder.layer1")
        const fullName = prefix ? `${prefix}.${name}` : name;
        yield* module.namedModules(fullName, true);
      }
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
   * This is equivalent to calling `train(false)`.
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
   */
  *buffers(recurse = true): Generator<Tensor> {
    for (const buffer of this._buffers.values()) {
      yield buffer;
    }
    if (recurse) {
      for (const module of this._modules.values()) {
        yield* module.buffers(true);
      }
    }
  }

  /**
   * Get all named buffers of this module and its children.
   */
  *namedBuffers(prefix = "", recurse = true): Generator<[string, Tensor]> {
    for (const [name, buffer] of this._buffers.entries()) {
      const fullName = prefix ? `${prefix}.${name}` : name;
      yield [fullName, buffer];
    }
    if (recurse) {
      for (const [moduleName, module] of this._modules.entries()) {
        const fullPrefix = prefix ? `${prefix}.${moduleName}` : moduleName;
        yield* module.namedBuffers(fullPrefix, true);
      }
    }
  }

  /**
   * Freeze specific parameters by name (or all if none provided).
   *
   * **⚠️ IMPORTANT**: This method creates new GradTensor instances with updated
   * `requiresGrad` flags. Any external references to the old parameter objects
   * will become stale. If you're using an optimizer that holds parameter references,
   * you should recreate the optimizer after freezing/unfreezing parameters.
   *
   * @param names - Array of parameter names to freeze (e.g., ['fc1.weight']). If undefined, freezes all parameters.
   * @param recurse - Whether to include parameters from child modules (default: true)
   *
   * @example
   * ```ts
   * const model = new MyModel();
   * // Freeze only the first layer's weights
   * model.freezeParameters(['fc1.weight']);
   * // Note: Recreate optimizer after freezing
   * const optimizer = new Adam(model.parameters());
   * ```
   */
  freezeParameters(names?: string[], recurse = true): void {
    this.setRequiresGradForNames(names, false, recurse);
  }

  /**
   * Unfreeze specific parameters by name (or all if none provided).
   *
   * **⚠️ IMPORTANT**: This method creates new GradTensor instances with updated
   * `requiresGrad` flags. Any external references to the old parameter objects
   * will become stale. If you're using an optimizer that holds parameter references,
   * you should recreate the optimizer after freezing/unfreezing parameters.
   *
   * @param names - Array of parameter names to unfreeze (e.g., ['fc1.weight']). If undefined, unfreezes all parameters.
   * @param recurse - Whether to include parameters from child modules (default: true)
   *
   * @example
   * ```ts
   * const model = new MyModel();
   * model.freezeParameters(); // Freeze all
   * model.unfreezeParameters(['fc2.weight']); // Unfreeze only fc2 weights
   * // Note: Recreate optimizer after unfreezing
   * const optimizer = new Adam(model.parameters());
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
    const providedNames = names !== undefined;
    const targetNames =
      names ?? Array.from(this.namedParameters("", recurse)).map(([name]) => name);
    for (const name of targetNames) {
      const resolved = this.resolveModuleAndName(name);
      if (!resolved) {
        if (providedNames) {
          throw new InvalidParameterError(`Unknown parameter name: ${name}`, "names", name);
        }
        continue;
      }
      const { module, localName } = resolved;
      const param = module._parameters.get(localName);
      if (!param) {
        if (providedNames) {
          throw new InvalidParameterError(`Unknown parameter name: ${name}`, "names", name);
        }
        continue;
      }
      // Replace parameter to ensure requiresGrad flag change is reflected consistently.
      const nextParam = GradTensor.fromTensor(param.tensor, { requiresGrad });
      module._parameters.set(localName, nextParam);
      for (const [key, value] of Object.entries(module)) {
        if (value === param) {
          Reflect.set(module, key, nextParam);
        }
      }
    }
  }

  private resolveModuleAndName(fullName: string): { module: Module; localName: string } | null {
    const parts = fullName.split(".");
    let module: Module = this;
    for (let i = 0; i < parts.length - 1; i++) {
      const part = parts[i] ?? "";
      const child = module._modules.get(part);
      if (!child) return null;
      module = child;
    }
    const localName = parts[parts.length - 1] ?? "";
    return { module, localName };
  }

  private static setTensorDeviceMetadata(target: Tensor, device: Device): void {
    if (!Reflect.set(target, "device", device)) {
      throw new DeepboxError("Failed to update tensor device metadata");
    }
  }

  /**
   * Get the state dictionary of the module.
   */
  stateDict(): {
    parameters: Record<string, StateEntry>;
    buffers: Record<string, StateEntry>;
  } {
    const parameters: Record<string, StateEntry> = {};
    const buffers: Record<string, StateEntry> = {};

    for (const [name, param] of this.namedParameters()) {
      const t = param.tensor;
      const data = cloneTensorData(t);
      parameters[name] = {
        data,
        shape: [...t.shape],
        dtype: t.dtype,
      };
    }

    for (const [name, buffer] of this.namedBuffers()) {
      const data = cloneTensorData(buffer);
      buffers[name] = {
        data,
        shape: [...buffer.shape],
        dtype: buffer.dtype,
      };
    }

    return { parameters, buffers };
  }

  /**
   * Load state dictionary into the module.
   */
  loadStateDict(stateDict: {
    parameters?: Record<string, StateEntry>;
    buffers?: Record<string, StateEntry>;
  }): void {
    const parameters = stateDict.parameters ?? {};
    const buffers = stateDict.buffers ?? {};

    const namedParams = new Map(this.namedParameters());
    const namedBuffs = new Map(this.namedBuffers());

    for (const name of namedParams.keys()) {
      if (!(name in parameters)) {
        throw new InvalidParameterError(`missing parameter: ${name}`, "stateDict.parameters", name);
      }
    }

    for (const name of namedBuffs.keys()) {
      if (!(name in buffers)) {
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

    for (const [name, entry] of Object.entries(parameters)) {
      const param = namedParams.get(name);
      if (!param) continue;
      validateStateEntryShape(name, "parameter", entry);
      copyStateEntryIntoTensor(name, "parameter", param.tensor, entry);
    }

    for (const [name, entry] of Object.entries(buffers)) {
      const buffer = namedBuffs.get(name);
      if (!buffer) continue;
      validateStateEntryShape(name, "buffer", entry);
      copyStateEntryIntoTensor(name, "buffer", buffer, entry);
    }
  }

  /**
   * Move module to a specific device.
   *
   * **⚠️ WARNING**: This is a metadata-only operation. It updates the device
   * property on parameters and buffers but does NOT actually transfer data
   * between devices. Actual device data transfer requires device-specific
   * memory management which is not yet implemented.
   *
   * This method is provided for API compatibility and future extensibility.
   * Currently, it only updates the `device` metadata field.
   *
   * @param device - Target device identifier (e.g., 'cpu', 'webgpu', 'wasm')
   * @returns this module for method chaining
   *
   * @example
   * ```ts
   * const model = new Linear(10, 5);
   * model.to('webgpu'); // Updates device metadata only
   * ```
   */
  to(device: Device): this {
    if (!isDevice(device)) {
      throw new InvalidParameterError("device must be one of: cpu, webgpu, wasm", "device", device);
    }

    for (const param of this.parameters()) {
      Module.setTensorDeviceMetadata(param.tensor, device);
    }
    for (const buffer of this.buffers()) {
      Module.setTensorDeviceMetadata(buffer, device);
    }
    return this;
  }

  /**
   * Apply a function to all modules recursively.
   */
  apply(fn: (module: Module) => void): this {
    for (const module of this.modules()) {
      fn(module);
    }
    return this;
  }

  /**
   * Register a forward pre-hook.
   */
  registerForwardPreHook(hook: ForwardPreHook): () => void {
    const hookId = this._nextHookId++;
    this._forwardPreHooks.set(hookId, hook);
    return () => {
      this._forwardPreHooks.delete(hookId);
    };
  }

  /**
   * Register a forward hook.
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
}
