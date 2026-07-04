/**
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
import type { AnyTensor, GradTensor } from "../../ndarray";
import { Module } from "../module/Module";

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
  private readonly items: GradTensor[];

  constructor(parameters?: GradTensor[]) {
    super();
    this.items = [];
    if (parameters) {
      for (const p of parameters) {
        this.append(p);
      }
    }
  }

  /** Number of parameters in the list. */
  get length(): number {
    return this.items.length;
  }

  /** Get a parameter by index. */
  get(index: number): GradTensor {
    const normalized = index < 0 ? this.items.length + index : index;
    const item = this.items[normalized];
    if (item === undefined) {
      throw new InvalidParameterError(
        `ParameterList index ${index} out of range [0, ${this.items.length})`,
        "index",
        index
      );
    }
    return item;
  }

  /** Append a parameter to the end. */
  append(param: GradTensor): this {
    const idx = this.items.length;
    this.items.push(param);
    this.registerParameter(String(idx), param);
    return this;
  }

  /** Iterate over parameters. */
  [Symbol.iterator](): Iterator<GradTensor> {
    return this.items[Symbol.iterator]();
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
    for (let i = 0; i < this.items.length; i++) {
      const item = this.items[i];
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
  private readonly items: Map<string, GradTensor>;

  constructor(parameters?: Record<string, GradTensor>) {
    super();
    this.items = new Map();
    if (parameters) {
      for (const [key, param] of Object.entries(parameters)) {
        this.set(key, param);
      }
    }
  }

  /** Number of parameters in the dict. */
  get length(): number {
    return this.items.size;
  }

  /** Get a parameter by key. */
  get(key: string): GradTensor {
    const item = this.items.get(key);
    if (item === undefined) {
      throw new InvalidParameterError(`ParameterDict key '${key}' not found`, "key", key);
    }
    return item;
  }

  /** Check if key exists. */
  has(key: string): boolean {
    return this.items.has(key);
  }

  /** Set a parameter by key. */
  set(key: string, param: GradTensor): this {
    this.items.set(key, param);
    this.registerParameter(key, param);
    return this;
  }

  /** Delete a parameter by key. */
  delete(key: string): boolean {
    const existed = this.items.delete(key);
    if (existed) {
      this._parameters.delete(key);
    }
    return existed;
  }

  /** Get all keys. */
  keys(): IterableIterator<string> {
    return this.items.keys();
  }

  /** Get all values. */
  values(): IterableIterator<GradTensor> {
    return this.items.values();
  }

  /** Get all entries. */
  entries(): IterableIterator<[string, GradTensor]> {
    return this.items.entries();
  }

  /** Iterate over entries. */
  [Symbol.iterator](): Iterator<[string, GradTensor]> {
    return this.items[Symbol.iterator]();
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
    for (const [key, param] of this.items) {
      lines.push(`  (${key}): Parameter of shape [${param.shape.join(", ")}]`);
    }
    lines.push(")");
    return lines.join("\n");
  }
}
