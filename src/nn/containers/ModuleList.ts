/**
 * @see {@link https://deepbox.dev/docs/nn-module | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";
import type { AnyTensor } from "../../ndarray";
import { Module } from "../module/Module";

function assertModule(value: unknown, owner: string): asserts value is Module {
  if (
    typeof value !== "object" ||
    value === null ||
    typeof (value as Module).forward !== "function"
  ) {
    throw new InvalidParameterError(`${owner} can only hold Module instances`, "module", value);
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

  /**
   * @param modules - Optional modules to add, in order
   * @throws {InvalidParameterError} If an element is not a Module
   */
  constructor(modules?: Iterable<Module>) {
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

  /**
   * Get a module by index. Negative indices count from the end.
   *
   * @throws {InvalidParameterError} If the index is not an integer or is out of range
   */
  get(index: number): Module {
    const normalized = index < 0 ? this.items.length + index : index;
    const item = Number.isInteger(normalized) ? this.items[normalized] : undefined;
    if (item === undefined) {
      throw new InvalidParameterError(
        `ModuleList index ${index} out of range [0, ${this.items.length})`,
        "index",
        index
      );
    }
    return item;
  }

  /**
   * Append a module to the end.
   *
   * @throws {InvalidParameterError} If `module` is not a Module
   */
  append(module: Module): this {
    assertModule(module, "ModuleList");
    const idx = this.items.length;
    this.items.push(module);
    this.registerModule(String(idx), module);
    return this;
  }

  /**
   * Append every module of an iterable, in order.
   *
   * @throws {InvalidParameterError} If an element is not a Module
   */
  extend(modules: Iterable<Module>): this {
    for (const m of modules) {
      this.append(m);
    }
    return this;
  }

  /**
   * Insert a module at a given index; later modules shift one position up.
   *
   * @throws {InvalidParameterError} If the index is not an integer in `[0, length]`
   *   or `module` is not a Module
   */
  insert(index: number, module: Module): this {
    if (!Number.isInteger(index) || index < 0 || index > this.items.length) {
      throw new InvalidParameterError(
        `Insert index ${index} out of range [0, ${this.items.length}]`,
        "index",
        index
      );
    }
    assertModule(module, "ModuleList");
    this.items.splice(index, 0, module);
    // Re-register all modules to maintain correct index keys
    this.reregisterAll();
    return this;
  }

  /**
   * Remove and return the module at `index` (default: the last one). Later
   * modules shift one position down.
   *
   * @throws {InvalidParameterError} If the index is out of range
   */
  pop(index = -1): Module {
    const removed = this.get(index);
    const normalized = index < 0 ? this.items.length + index : index;
    this.items.splice(normalized, 1);
    this.reregisterAll();
    return removed;
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
        const childLines = item.toString().split("\n");
        const itemStr = childLines.map((line, idx) => (idx === 0 ? line : `  ${line}`)).join("\n");
        lines.push(`  (${i}): ${itemStr}`);
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

  /**
   * @param modules - Optional initial modules keyed by name
   * @throws {InvalidParameterError} If a key is empty or contains ".", or a value is not a Module
   */
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

  /**
   * Set a module by key, replacing an existing entry in place.
   *
   * Keys become part of parameter names such as `encoder.weight`, so they must
   * be non-empty and must not contain ".".
   *
   * @throws {InvalidParameterError} If the key is invalid or `module` is not a Module
   */
  set(key: string, module: Module): this {
    assertKey(key, "ModuleDict");
    assertModule(module, "ModuleDict");
    this.items.set(key, module);
    this.registerModule(key, module);
    return this;
  }

  /**
   * Add every `[key, module]` pair from a record, a Map or an iterable of pairs.
   *
   * @throws {InvalidParameterError} If a key or module is invalid
   */
  update(modules: Record<string, Module> | Iterable<readonly [string, Module]>): this {
    const entries =
      Symbol.iterator in modules
        ? (modules as Iterable<readonly [string, Module]>)
        : Object.entries(modules as Record<string, Module>);
    for (const [key, mod] of entries) {
      this.set(key, mod);
    }
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

  /**
   * Remove a module by key and return it.
   *
   * @throws {InvalidParameterError} If the key does not exist
   */
  pop(key: string): Module {
    const mod = this.get(key);
    this.delete(key);
    return mod;
  }

  /** Remove all modules. */
  clear(): void {
    this.items.clear();
    this._modules.clear();
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
      const childLines = mod.toString().split("\n");
      const modStr = childLines.map((line, idx) => (idx === 0 ? line : `  ${line}`)).join("\n");
      lines.push(`  (${key}): ${modStr}`);
    }
    lines.push(")");
    return lines.join("\n");
  }
}
