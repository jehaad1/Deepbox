/**
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
import type { AnyTensor } from "../../ndarray";
import { Module } from "../module/Module";

/**
 * Holds submodules in a list.
 *
 * ModuleList can be indexed like a regular array and the modules it contains
 * are properly registered so that all Module methods (parameters, to, etc.) work.
 *
 * @example
 * ```ts
 * const layers = new ModuleList([new Linear(10, 5), new ReLU()]);
 * layers.append(new Linear(5, 2));
 * for (const layer of layers) { ... }
 * ```
 *
 * @category Neural Network Containers
 */
export class ModuleList extends Module {
  private readonly items: Module[];

  constructor(modules?: Module[]) {
    super();
    this.items = [];
    if (modules) {
      for (const m of modules) {
        this.append(m);
      }
    }
  }

  /** Number of modules in the list. */
  get length(): number {
    return this.items.length;
  }

  /** Get a module by index. */
  get(index: number): Module {
    const normalized = index < 0 ? this.items.length + index : index;
    const item = this.items[normalized];
    if (item === undefined) {
      throw new InvalidParameterError(
        `ModuleList index ${index} out of range [0, ${this.items.length})`,
        "index",
        index
      );
    }
    return item;
  }

  /** Append a module to the end. */
  append(module: Module): this {
    const idx = this.items.length;
    this.items.push(module);
    this.registerModule(String(idx), module);
    return this;
  }

  /** Insert a module at a given index. */
  insert(index: number, module: Module): this {
    if (index < 0 || index > this.items.length) {
      throw new InvalidParameterError(
        `Insert index ${index} out of range [0, ${this.items.length}]`,
        "index",
        index
      );
    }
    this.items.splice(index, 0, module);
    // Re-register all modules to maintain correct index keys
    this.reregisterAll();
    return this;
  }

  /** Iterate over modules. */
  [Symbol.iterator](): Iterator<Module> {
    return this.items[Symbol.iterator]();
  }

  private reregisterAll(): void {
    // Clear existing module registrations and re-register with correct indices
    this._modules.clear();
    for (let i = 0; i < this.items.length; i++) {
      const item = this.items[i];
      if (item) {
        this.registerModule(String(i), item);
      }
    }
  }

  forward(..._inputs: AnyTensor[]): AnyTensor {
    throw new InvalidParameterError(
      "ModuleList is not callable. Use individual modules via get() or iteration.",
      "forward",
      undefined
    );
  }

  override toString(): string {
    const lines = ["ModuleList("];
    for (let i = 0; i < this.items.length; i++) {
      const item = this.items[i];
      if (item) {
        lines.push(`  (${i}): ${item.toString()}`);
      }
    }
    lines.push(")");
    return lines.join("\n");
  }
}

/**
 * Holds submodules in a dictionary.
 *
 * ModuleDict can be indexed by string keys and the modules it contains
 * are properly registered.
 *
 * @example
 * ```ts
 * const modules = new ModuleDict({ encoder: new Linear(10, 5), decoder: new Linear(5, 10) });
 * const enc = modules.get('encoder');
 * ```
 *
 * @category Neural Network Containers
 */
export class ModuleDict extends Module {
  private readonly items: Map<string, Module>;

  constructor(modules?: Record<string, Module>) {
    super();
    this.items = new Map();
    if (modules) {
      for (const [key, mod] of Object.entries(modules)) {
        this.set(key, mod);
      }
    }
  }

  /** Number of modules in the dict. */
  get length(): number {
    return this.items.size;
  }

  /** Get a module by key. */
  get(key: string): Module {
    const item = this.items.get(key);
    if (item === undefined) {
      throw new InvalidParameterError(`ModuleDict key '${key}' not found`, "key", key);
    }
    return item;
  }

  /** Check if key exists. */
  has(key: string): boolean {
    return this.items.has(key);
  }

  /** Set a module by key. */
  set(key: string, module: Module): this {
    this.items.set(key, module);
    this.registerModule(key, module);
    return this;
  }

  /** Delete a module by key. */
  delete(key: string): boolean {
    const existed = this.items.delete(key);
    if (existed) {
      this._modules.delete(key);
    }
    return existed;
  }

  /** Get all keys. */
  keys(): IterableIterator<string> {
    return this.items.keys();
  }

  /** Get all values. */
  values(): IterableIterator<Module> {
    return this.items.values();
  }

  /** Get all entries. */
  entries(): IterableIterator<[string, Module]> {
    return this.items.entries();
  }

  /** Iterate over entries. */
  [Symbol.iterator](): Iterator<[string, Module]> {
    return this.items[Symbol.iterator]();
  }

  forward(..._inputs: AnyTensor[]): AnyTensor {
    throw new InvalidParameterError(
      "ModuleDict is not callable. Use individual modules via get().",
      "forward",
      undefined
    );
  }

  override toString(): string {
    const lines = ["ModuleDict("];
    for (const [key, mod] of this.items) {
      lines.push(`  (${key}): ${mod.toString()}`);
    }
    lines.push(")");
    return lines.join("\n");
  }
}
