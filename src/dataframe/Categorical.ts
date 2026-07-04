import { DataValidationError, InvalidParameterError } from "../core/errors/index";

/**
 * Memory-efficient categorical data type with optional ordering.
 *
 * Stores data as integer codes referencing a compact category list,
 * providing significant memory savings for columns with many repeated values.
 *
 * Supports ordered categories for comparison operations and sorting.
 *
 * @example
 * ```ts
 * import { Categorical } from 'deepbox/dataframe';
 *
 * // Unordered categorical
 * const colors = Categorical.from(['red', 'blue', 'red', 'green', 'blue']);
 * console.log(colors.categories);  // ['blue', 'green', 'red']
 * console.log(colors.codes);       // [2, 0, 2, 1, 0]
 *
 * // Ordered categorical (e.g., size)
 * const sizes = Categorical.from(['M', 'S', 'L', 'M', 'XL'], {
 *   categories: ['S', 'M', 'L', 'XL'],
 *   ordered: true
 * });
 * ```
 *
 * @see {@link https://deepbox.dev/docs/dataframe-series | Deepbox Categorical}
 */
export class Categorical {
  /** Unique category values in canonical order */
  private readonly _categories: string[];
  /** Integer codes mapping each element to its category index (-1 for null/missing) */
  private readonly _codes: Int32Array;
  /** Whether categories have a meaningful ordering */
  private readonly _ordered: boolean;
  /** Fast lookup: category string -> code index */
  private readonly _categoryIndex: Map<string, number>;

  private constructor(categories: string[], codes: Int32Array, ordered: boolean) {
    this._categories = categories;
    this._codes = codes;
    this._ordered = ordered;
    this._categoryIndex = new Map<string, number>();
    for (let i = 0; i < categories.length; i++) {
      const cat = categories[i];
      if (cat !== undefined) {
        this._categoryIndex.set(cat, i);
      }
    }
  }

  /**
   * Create a Categorical from an array of values.
   *
   * @param values - Array of string values (null/undefined treated as missing)
   * @param options - Configuration options
   * @param options.categories - Explicit category list (default: inferred, sorted alphabetically)
   * @param options.ordered - Whether categories are ordered (default: false)
   * @returns New Categorical instance
   *
   * @example
   * ```ts
   * const cat = Categorical.from(['a', 'b', 'a', 'c']);
   * const ordered = Categorical.from(['low', 'mid', 'high'], {
   *   categories: ['low', 'mid', 'high'],
   *   ordered: true,
   * });
   * ```
   */
  static from(
    values: ReadonlyArray<string | null | undefined>,
    options: {
      readonly categories?: readonly string[];
      readonly ordered?: boolean;
    } = {}
  ): Categorical {
    const ordered = options.ordered ?? false;

    // Determine categories
    let categories: string[];
    if (options.categories !== undefined) {
      categories = [...options.categories];
      // Validate no duplicates
      const seen = new Set<string>();
      for (const cat of categories) {
        if (seen.has(cat)) {
          throw new DataValidationError(`Duplicate category '${cat}'`);
        }
        seen.add(cat);
      }
    } else {
      // Infer from data, sorted alphabetically
      const unique = new Set<string>();
      for (const v of values) {
        if (v !== null && v !== undefined) {
          unique.add(v);
        }
      }
      categories = [...unique].sort();
    }

    // Build code lookup
    const catIndex = new Map<string, number>();
    for (let i = 0; i < categories.length; i++) {
      const cat = categories[i];
      if (cat !== undefined) {
        catIndex.set(cat, i);
      }
    }

    // Encode values
    const codes = new Int32Array(values.length);
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v === null || v === undefined) {
        codes[i] = -1;
      } else {
        const code = catIndex.get(v);
        if (code === undefined) {
          throw new DataValidationError(`Value '${v}' is not in the category list`);
        }
        codes[i] = code;
      }
    }

    return new Categorical(categories, codes, ordered);
  }

  /** Get the category list */
  get categories(): readonly string[] {
    return this._categories;
  }

  /** Get the integer codes array */
  get codes(): Int32Array {
    return this._codes;
  }

  /** Whether categories are ordered */
  get ordered(): boolean {
    return this._ordered;
  }

  /** Number of elements */
  get length(): number {
    return this._codes.length;
  }

  /** Number of unique categories (excluding missing) */
  get nCategories(): number {
    return this._categories.length;
  }

  /**
   * Get the string value at a given index.
   *
   * @param index - Zero-based index
   * @returns Category string, or null if missing
   */
  get(index: number): string | null {
    if (index < 0 || index >= this._codes.length) {
      throw new InvalidParameterError(
        `Index ${index} out of bounds [0, ${this._codes.length})`,
        "index",
        index
      );
    }
    const code = this._codes[index] ?? -1;
    if (code === -1) return null;
    return this._categories[code] ?? null;
  }

  /**
   * Convert back to a string array.
   *
   * @returns Array of string values (null for missing)
   */
  toArray(): (string | null)[] {
    const result: (string | null)[] = [];
    for (let i = 0; i < this._codes.length; i++) {
      const code = this._codes[i] ?? -1;
      result.push(code === -1 ? null : (this._categories[code] ?? null));
    }
    return result;
  }

  /**
   * Count occurrences of each category.
   *
   * @returns Map of category -> count
   */
  valueCounts(): Map<string, number> {
    const counts = new Map<string, number>();
    for (const cat of this._categories) {
      counts.set(cat, 0);
    }
    for (let i = 0; i < this._codes.length; i++) {
      const code = this._codes[i] ?? -1;
      if (code >= 0) {
        const cat = this._categories[code];
        if (cat !== undefined) {
          counts.set(cat, (counts.get(cat) ?? 0) + 1);
        }
      }
    }
    return counts;
  }

  /**
   * Add new categories to the category list.
   *
   * @param newCategories - Categories to add
   * @returns New Categorical with added categories (data unchanged)
   */
  addCategories(newCategories: readonly string[]): Categorical {
    const existing = new Set(this._categories);
    const merged = [...this._categories];
    for (const cat of newCategories) {
      if (existing.has(cat)) {
        throw new DataValidationError(`Category '${cat}' already exists`);
      }
      existing.add(cat);
      merged.push(cat);
    }
    // Codes remain the same since new categories have higher indices
    return new Categorical(merged, new Int32Array(this._codes), this._ordered);
  }

  /**
   * Remove categories from the category list.
   * Values in removed categories become missing (-1).
   *
   * @param removeCategories - Categories to remove
   * @returns New Categorical with categories removed
   */
  removeCategories(removeCategories: readonly string[]): Categorical {
    const removeSet = new Set(removeCategories);
    const newCats: string[] = [];
    const oldToNew = new Map<number, number>();
    for (let i = 0; i < this._categories.length; i++) {
      const cat = this._categories[i];
      if (cat !== undefined && !removeSet.has(cat)) {
        oldToNew.set(i, newCats.length);
        newCats.push(cat);
      }
    }
    const newCodes = new Int32Array(this._codes.length);
    for (let i = 0; i < this._codes.length; i++) {
      const old = this._codes[i] ?? -1;
      if (old === -1) {
        newCodes[i] = -1;
      } else {
        const mapped = oldToNew.get(old);
        newCodes[i] = mapped !== undefined ? mapped : -1;
      }
    }
    return new Categorical(newCats, newCodes, this._ordered);
  }

  /**
   * Reorder categories.
   *
   * @param newOrder - New category order (must contain all existing categories)
   * @returns New Categorical with reordered categories
   */
  reorderCategories(newOrder: readonly string[]): Categorical {
    if (newOrder.length !== this._categories.length) {
      throw new InvalidParameterError(
        `newOrder must have ${this._categories.length} categories; got ${newOrder.length}`,
        "newOrder",
        newOrder.length
      );
    }
    const oldSet = new Set(this._categories);
    for (const cat of newOrder) {
      if (!oldSet.has(cat)) {
        throw new DataValidationError(`Category '${cat}' not found in current categories`);
      }
    }

    // Build old -> new mapping
    const newCatIndex = new Map<string, number>();
    const newCats = [...newOrder];
    for (let i = 0; i < newCats.length; i++) {
      const cat = newCats[i];
      if (cat !== undefined) {
        newCatIndex.set(cat, i);
      }
    }

    const newCodes = new Int32Array(this._codes.length);
    for (let i = 0; i < this._codes.length; i++) {
      const old = this._codes[i] ?? -1;
      if (old === -1) {
        newCodes[i] = -1;
      } else {
        const cat = this._categories[old];
        newCodes[i] = cat !== undefined ? (newCatIndex.get(cat) ?? -1) : -1;
      }
    }
    return new Categorical(newCats, newCodes, this._ordered);
  }

  /**
   * Set whether categories are ordered.
   *
   * @param ordered - Whether to set as ordered
   * @returns New Categorical with updated ordering flag
   */
  setOrdered(ordered: boolean): Categorical {
    return new Categorical([...this._categories], new Int32Array(this._codes), ordered);
  }

  /**
   * Rename categories.
   *
   * @param mapping - Map of old name -> new name
   * @returns New Categorical with renamed categories
   */
  renameCategories(mapping: Readonly<Record<string, string>>): Categorical {
    const newCats = this._categories.map((cat) => {
      const renamed = mapping[cat];
      return renamed !== undefined ? renamed : cat;
    });
    // Verify uniqueness
    const seen = new Set<string>();
    for (const cat of newCats) {
      if (seen.has(cat)) {
        throw new DataValidationError(`Renaming would produce duplicate category '${cat}'`);
      }
      seen.add(cat);
    }
    return new Categorical(newCats, new Int32Array(this._codes), this._ordered);
  }

  /**
   * Compare two values by their category order (only for ordered categoricals).
   *
   * @param a - First value
   * @param b - Second value
   * @returns Negative if a < b, 0 if equal, positive if a > b
   */
  compare(a: string, b: string): number {
    if (!this._ordered) {
      throw new DataValidationError("Comparison requires ordered Categorical");
    }
    const codeA = this._categoryIndex.get(a);
    const codeB = this._categoryIndex.get(b);
    if (codeA === undefined) {
      throw new DataValidationError(`'${a}' is not a valid category`);
    }
    if (codeB === undefined) {
      throw new DataValidationError(`'${b}' is not a valid category`);
    }
    return codeA - codeB;
  }

  /**
   * Sort the values by category order.
   *
   * @param ascending - Sort in ascending order (default: true)
   * @returns New Categorical with sorted values
   */
  sort(ascending = true): Categorical {
    const indices = Array.from({ length: this._codes.length }, (_, i) => i);
    indices.sort((a, b) => {
      const ca = this._codes[a] ?? -1;
      const cb = this._codes[b] ?? -1;
      // Missing values go to the end
      if (ca === -1 && cb === -1) return 0;
      if (ca === -1) return 1;
      if (cb === -1) return -1;
      return ascending ? ca - cb : cb - ca;
    });
    const newCodes = new Int32Array(this._codes.length);
    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      if (idx !== undefined) {
        newCodes[i] = this._codes[idx] ?? -1;
      }
    }
    return new Categorical([...this._categories], newCodes, this._ordered);
  }

  /**
   * Estimate memory usage in bytes.
   * Categories are stored once; data uses Int32 codes.
   *
   * @returns Approximate memory usage in bytes
   */
  memoryUsage(): number {
    // Codes: 4 bytes per Int32
    let bytes = this._codes.length * 4;
    // Category strings: ~2 bytes per char (UTF-16)
    for (const cat of this._categories) {
      bytes += cat.length * 2 + 28; // string overhead
    }
    // Map overhead
    bytes += this._categories.length * 64;
    return bytes;
  }

  toString(): string {
    const values = this.toArray();
    const display =
      values.length <= 10
        ? values.map((v) => (v === null ? "NaN" : v)).join(", ")
        : [
            ...values.slice(0, 5).map((v) => (v === null ? "NaN" : v)),
            "...",
            ...values.slice(-2).map((v) => (v === null ? "NaN" : v)),
          ].join(", ");
    return `Categorical([${display}])\nCategories (${this._categories.length}): [${this._categories.join(", ")}]${this._ordered ? " (ordered)" : ""}`;
  }
}
