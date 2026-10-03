/**
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
import { type AnyTensor, GradTensor } from "../../ndarray";
import { Module } from "../module/Module";

function assertParameter(value: unknown, owner: string): asserts value is GradTensor {
  if (!GradTensor.isGradTensor(value)) {
    throw new InvalidParameterError(`${owner} can only hold GradTensor parameters`, "param", value);
  }
}

function assertKey(key: unknown, owner: string): asserts key is string {
  if (typeof key !== "string" || key.length === 0 || key.includes(".")) {
    throw new InvalidParameterError(
      `${owner} keys must be non-empty strings without "."; received ${JSON.stringify(key)}`,
      "key",
      key
    );
  }
}

/**
 * Holds parameters in a list.
 *
 * ParameterList can be indexed like a regular array and the parameters it
 * contains are properly registered so that all Module methods (parameters,
 * namedParameters, etc.) work correctly.
 *
 * @example
 * ```ts
 * const params = new ParameterList([param1, param2]);
 * params.append(param3);
 * for (const p of params) { ... }
 * ```
 *
 * @category Neural Network Containers
 */
export class ParameterList extends Module {
  /**
   * Parameters live in the base class's registry under the keys "0", "1", ...
   * (which `freezeParameters` may replace), so reads always go through it.
   */
  private count = 0;

  /**
   * @param parameters - Optional parameters to add, in order
   * @throws {InvalidParameterError} If an element is not a GradTensor
   */
  constructor(parameters?: Iterable<GradTensor>) {
    super();
    if (parameters) {
      for (const p of parameters) {
        this.append(p);
      }
    }
  }

  /** Number of parameters in the list. */
  get length(): number {
    return this.count;
  }

  /**
   * Get a parameter by index. Negative indices count from the end.
   *
   * @throws {InvalidParameterError} If the index is not an integer or is out of range
   */
  get(index: number): GradTensor {
    const normalized = index < 0 ? this.count + index : index;
    const item = Number.isInteger(normalized) ? this.getParameter(String(normalized)) : undefined;
    if (item === undefined) {
      throw new InvalidParameterError(
        `ParameterList index ${index} out of range [0, ${this.count})`,
        "index",
        index
      );
    }
    return item;
  }

  /**
   * Append a parameter to the end.
   *
   * @throws {InvalidParameterError} If `param` is not a GradTensor
   */
  append(param: GradTensor): this {
    assertParameter(param, "ParameterList");
    this.registerParameter(String(this.count), param);
    this.count++;
    return this;
  }

  /**
   * Append every parameter of an iterable, in order.
   *
   * @throws {InvalidParameterError} If an element is not a GradTensor
   */
  extend(parameters: Iterable<GradTensor>): this {
    for (const p of parameters) {
      this.append(p);
    }
    return this;
  }

  /** Iterate over parameters. */
  [Symbol.iterator](): Iterator<GradTensor> {
    return this._parameters.values();
  }

  forward(..._inputs: AnyTensor[]): AnyTensor {
    throw new InvalidParameterError(
      "ParameterList is not callable. Access parameters via get() or iteration.",
      "forward",
      undefined
    );
  }

  override toString(): string {
    const lines = ["ParameterList("];
    for (let i = 0; i < this.count; i++) {
      const item = this.getParameter(String(i));
      if (item) {
        lines.push(`  (${i}): Parameter of shape [${item.shape.join(", ")}]`);
      }
    }
    lines.push(")");
    return lines.join("\n");
  }
}

/**
 * Holds parameters in a dictionary.
 *
 * ParameterDict can be indexed by string keys and the parameters it contains
 * are properly registered so that all Module methods work correctly.
 *
 * @example
 * ```ts
 * const params = new ParameterDict({ weight: weightParam, bias: biasParam });
 * const w = params.get('weight');
 * ```
 *
 * @category Neural Network Containers
 */
export class ParameterDict extends Module {
  /**
   * @param parameters - Optional initial parameters keyed by name
   * @throws {InvalidParameterError} If a key is empty or contains ".", or a value is not a GradTensor
   */
  constructor(parameters?: Record<string, GradTensor>) {
    super();
    if (parameters) {
      for (const [key, param] of Object.entries(parameters)) {
        this.set(key, param);
      }
    }
  }

  /** Number of parameters in the dict. */
  get length(): number {
    return this._parameters.size;
  }

  /** Get a parameter by key. */
  get(key: string): GradTensor {
    const item = this.getParameter(key);
    if (item === undefined) {
      throw new InvalidParameterError(`ParameterDict key '${key}' not found`, "key", key);
    }
    return item;
  }

  /** Check if key exists. */
  has(key: string): boolean {
    return this._parameters.has(key);
  }

  /**
   * Set a parameter by key, replacing an existing entry in place.
   *
   * Keys become part of parameter names such as `store.weight`, so they must
   * be non-empty and must not contain ".".
   *
   * @throws {InvalidParameterError} If the key is invalid or `param` is not a GradTensor
   */
  set(key: string, param: GradTensor): this {
    assertKey(key, "ParameterDict");
    assertParameter(param, "ParameterDict");
    this.registerParameter(key, param);
    return this;
  }

  /**
   * Add every `[key, parameter]` pair from a record, a Map or an iterable of pairs.
   *
   * @throws {InvalidParameterError} If a key or parameter is invalid
   */
  update(parameters: Record<string, GradTensor> | Iterable<readonly [string, GradTensor]>): this {
    const entries =
      Symbol.iterator in parameters
        ? (parameters as Iterable<readonly [string, GradTensor]>)
        : Object.entries(parameters as Record<string, GradTensor>);
    for (const [key, param] of entries) {
      this.set(key, param);
    }
    return this;
  }

  /** Delete a parameter by key. */
  delete(key: string): boolean {
    return this._parameters.delete(key);
  }

  /**
   * Remove a parameter by key and return it.
   *
   * @throws {InvalidParameterError} If the key does not exist
   */
  pop(key: string): GradTensor {
    const param = this.get(key);
    this.delete(key);
    return param;
  }

  /** Remove all parameters. */
  clear(): void {
    this._parameters.clear();
  }

  /** Get all keys. */
  keys(): IterableIterator<string> {
    return this._parameters.keys();
  }

  /** Get all values. */
  values(): IterableIterator<GradTensor> {
    return this._parameters.values();
  }

  /** Get all entries. */
  entries(): IterableIterator<[string, GradTensor]> {
    return this._parameters.entries();
  }

  /** Iterate over entries. */
  [Symbol.iterator](): Iterator<[string, GradTensor]> {
    return this._parameters.entries();
  }

  forward(..._inputs: AnyTensor[]): AnyTensor {
    throw new InvalidParameterError(
      "ParameterDict is not callable. Access parameters via get().",
      "forward",
      undefined
    );
  }

  override toString(): string {
    const lines = ["ParameterDict("];
    for (const [key, param] of this._parameters) {
      lines.push(`  (${key}): Parameter of shape [${param.shape.join(", ")}]`);
    }
    lines.push(")");
    return lines.join("\n");
  }
}
