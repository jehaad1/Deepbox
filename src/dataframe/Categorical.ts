import { DataValidationError, InvalidParameterError } from "../core/errors/index";

/**
 * Categorical data type with optional ordering, modeled on `pandas.Categorical`.
 *
 * Values are stored as integer codes that index into a category list, so a column
 * with many repeated strings keeps each distinct string only once. Missing values
 * (`null` or `undefined` on input) are stored as code `-1`.
 *
 * Instances are immutable: every method that changes categories, order or data
 * returns a new `Categorical`. The `codes` array is returned without copying for speed
 * and must be treated as read-only.
 *
 * Ordered categoricals support {@link Categorical.compare}, {@link Categorical.min}
 * and {@link Categorical.max}.
 *
 * @example
 * ```ts
 * import { Categorical } from 'deepbox/dataframe';
 *
 * // Unordered categorical
 * const colors = Categorical.from(['red', 'blue', 'red', 'green', 'blue']);
 * console.log(colors.categories);  // ['blue', 'green', 'red']
 * console.log(colors.codes);       // Int32Array [2, 0, 2, 1, 0]
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
  /** Unique category values in canonical order (frozen) */
  private readonly _categories: readonly string[];
  /** Integer codes mapping each element to its category index (-1 for null/missing) */
  private readonly _codes: Int32Array;
  /** Whether categories have a meaningful ordering */
  private readonly _ordered: boolean;
  /** Fast lookup: category string -> code index */
  private readonly _categoryIndex: ReadonlyMap<string, number>;

  private constructor(
    categories: string[],
    codes: Int32Array,
    ordered: boolean,
    categoryIndex?: ReadonlyMap<string, number>
  ) {
    this._categories = Object.freeze(categories);
    this._codes = codes;
    this._ordered = ordered;
    this._categoryIndex = categoryIndex ?? buildCategoryIndex(categories);
  }

  /**
   * Create a Categorical from an array of values.
   *
   * @param values - Array of string values (null/undefined treated as missing)
   * @param options - Configuration options
   * @param options.categories - Explicit category list (default: inferred, sorted by Unicode code point)
   * @param options.ordered - Whether categories are ordered (default: false)
   * @returns New Categorical instance
   * @throws {@link DataValidationError} If a value is not a string, a category is not a string
   *   or is duplicated, or a value is missing from an explicit category list
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
      categories = validateCategoryList(options.categories);
    } else {
      // Infer from data, sorted by Unicode code point (same order as pandas)
      const unique = new Set<string>();
      for (let i = 0; i < values.length; i++) {
        const v = values[i];
        if (v !== null && v !== undefined) {
          if (typeof v !== "string") {
            throw new DataValidationError(
              `Categorical values must be strings, null or undefined; got ${typeof v} at index ${i}`
            );
          }
          unique.add(v);
        }
      }
      categories = sortByCodePoint([...unique]);
    }

    // Build code lookup
    const catIndex = buildCategoryIndex(categories);

    // Encode values
    const codes = new Int32Array(values.length);
    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      if (v === null || v === undefined) {
        codes[i] = -1;
      } else {
        const code = catIndex.get(v);
        if (code === undefined) {
          throw new DataValidationError(`Value '${String(v)}' is not in the category list`);
        }
        codes[i] = code;
      }
    }

    return new Categorical(categories, codes, ordered, catIndex);
  }

  /** Get the category list */
  get categories(): readonly string[] {
    return this._categories;
  }

  /**
   * Get the integer codes array (`-1` marks a missing value).
   * The array is shared with the instance and must not be modified.
   */
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
   * @param index - Zero-based integer index
   * @returns Category string, or null if missing
   * @throws {@link InvalidParameterError} If the index is not an integer inside `[0, length)`
   */
  get(index: number): string | null {
    if (!Number.isInteger(index) || index < 0 || index >= this._codes.length) {
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
    const n = this._codes.length;
    const result: (string | null)[] = new Array<string | null>(n);
    for (let i = 0; i < n; i++) {
      const code = this._codes[i] ?? -1;
      result[i] = code === -1 ? null : (this._categories[code] ?? null);
    }
    return result;
  }

  /**
   * Boolean mask marking missing values.
   *
   * @returns Array with `true` where the value is missing
   */
  isNull(): boolean[] {
    const n = this._codes.length;
    const result = new Array<boolean>(n);
    for (let i = 0; i < n; i++) {
      result[i] = (this._codes[i] ?? -1) === -1;
    }
    return result;
  }

  /**
   * Count occurrences of each category. Categories that never occur are included
   * with a count of 0, in category order. Missing values are not counted.
   *
   * @returns Map of category -> count
   */
  valueCounts(): Map<string, number> {
    const tally = this.countCodes();
    const counts = new Map<string, number>();
    for (let i = 0; i < this._categories.length; i++) {
      const cat = this._categories[i];
      if (cat !== undefined) counts.set(cat, tally[i] ?? 0);
    }
    return counts;
  }

  /**
   * Add new categories to the category list.
   *
   * @param newCategories - Categories to add (appended after the existing ones)
   * @returns New Categorical with added categories (data unchanged)
   * @throws {@link DataValidationError} If a category already exists, is repeated, or is not a string
   */
  addCategories(newCategories: readonly string[]): Categorical {
    const existing = new Set(this._categories);
    const merged = [...this._categories];
    for (const cat of newCategories) {
      if (typeof cat !== "string") {
        throw new DataValidationError(`Categories must be strings; got ${typeof cat}`);
      }
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
   * @throws {@link DataValidationError} If a category to remove is not in the current categories
   */
  removeCategories(removeCategories: readonly string[]): Categorical {
    const removeSet = new Set(removeCategories);
    for (const cat of removeSet) {
      if (!this._categoryIndex.has(cat)) {
        throw new DataValidationError(`Category '${String(cat)}' not found in current categories`);
      }
    }
    return this.keepCategories((cat) => !removeSet.has(cat));
  }

  /**
   * Remove categories that do not occur in the data.
   *
   * @returns New Categorical containing only categories that are used
   */
  removeUnusedCategories(): Categorical {
    const tally = this.countCodes();
    const keep = new Set<string>();
    for (let i = 0; i < this._categories.length; i++) {
      const cat = this._categories[i];
      if (cat !== undefined && (tally[i] ?? 0) > 0) keep.add(cat);
    }
    return this.keepCategories((cat) => keep.has(cat));
  }

  /**
   * Reorder categories.
   *
   * @param newOrder - New category order (must contain every existing category exactly once)
   * @param options - Optional settings
   * @param options.ordered - Ordering flag of the result (default: keep the current flag)
   * @returns New Categorical with reordered categories
   * @throws {@link InvalidParameterError} If `newOrder` has the wrong length
   * @throws {@link DataValidationError} If `newOrder` has duplicates or unknown categories
   */
  reorderCategories(
    newOrder: readonly string[],
    options: { readonly ordered?: boolean } = {}
  ): Categorical {
    if (newOrder.length !== this._categories.length) {
      throw new InvalidParameterError(
        `newOrder must have ${this._categories.length} categories; got ${newOrder.length}`,
        "newOrder",
        newOrder.length
      );
    }
    const newCats = validateCategoryList(newOrder);
    for (const cat of newCats) {
      if (!this._categoryIndex.has(cat)) {
        throw new DataValidationError(`Category '${cat}' not found in current categories`);
      }
    }

    const newCatIndex = buildCategoryIndex(newCats);
    const remap = new Int32Array(this._categories.length);
    for (let i = 0; i < this._categories.length; i++) {
      const cat = this._categories[i];
      remap[i] = cat !== undefined ? (newCatIndex.get(cat) ?? -1) : -1;
    }
    const newCodes = this.remapCodes(remap);
    return new Categorical(newCats, newCodes, options.ordered ?? this._ordered, newCatIndex);
  }

  /**
   * Replace the category list. Values that are not in the new list become missing.
   * Unlike {@link Categorical.reorderCategories}, the new list may add or drop categories.
   *
   * @param newCategories - New category list
   * @param options - Optional settings
   * @param options.ordered - Ordering flag of the result (default: keep the current flag)
   * @returns New Categorical using the new categories
   * @throws {@link DataValidationError} If `newCategories` has duplicates or non-string entries
   */
  setCategories(
    newCategories: readonly string[],
    options: { readonly ordered?: boolean } = {}
  ): Categorical {
    const newCats = validateCategoryList(newCategories);
    const newCatIndex = buildCategoryIndex(newCats);
    const remap = new Int32Array(this._categories.length);
    for (let i = 0; i < this._categories.length; i++) {
      const cat = this._categories[i];
      remap[i] = cat !== undefined ? (newCatIndex.get(cat) ?? -1) : -1;
    }
    const newCodes = this.remapCodes(remap);
    return new Categorical(newCats, newCodes, options.ordered ?? this._ordered, newCatIndex);
  }

  /**
   * Set whether categories are ordered.
   *
   * @param ordered - Whether to set as ordered
   * @returns New Categorical with updated ordering flag
   */
  setOrdered(ordered: boolean): Categorical {
    return new Categorical(
      [...this._categories],
      new Int32Array(this._codes),
      ordered,
      this._categoryIndex
    );
  }

  /**
   * Return an ordered copy. Shorthand for `setOrdered(true)`.
   *
   * @returns New ordered Categorical
   */
  asOrdered(): Categorical {
    return this.setOrdered(true);
  }

  /**
   * Return an unordered copy. Shorthand for `setOrdered(false)`.
   *
   * @returns New unordered Categorical
   */
  asUnordered(): Categorical {
    return this.setOrdered(false);
  }

  /**
   * Rename categories. Categories without an entry in the mapping keep their name.
   * Entries for categories that do not exist are ignored.
   *
   * @param mapping - Object or Map of old name -> new name, or a function from old name to new name
   * @returns New Categorical with renamed categories
   * @throws {@link DataValidationError} If the new names are not strings or contain duplicates
   */
  renameCategories(
    mapping:
      | Readonly<Record<string, string>>
      | ReadonlyMap<string, string>
      | ((category: string) => string)
  ): Categorical {
    const lookup = (cat: string): string | undefined => {
      if (typeof mapping === "function") return mapping(cat);
      if (mapping instanceof Map) return mapping.get(cat);
      // Own properties only: names such as "constructor" must not hit Object.prototype.
      return Object.hasOwn(mapping, cat)
        ? (mapping as Readonly<Record<string, string>>)[cat]
        : undefined;
    };
    const newCats = this._categories.map((cat) => {
      const renamed = lookup(cat);
      return renamed !== undefined ? renamed : cat;
    });
    // Verify types and uniqueness
    const seen = new Set<string>();
    for (const cat of newCats) {
      if (typeof cat !== "string") {
        throw new DataValidationError(`Renamed categories must be strings; got ${typeof cat}`);
      }
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
   * @throws {@link DataValidationError} If the categorical is unordered or a value is not a category
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
   * Smallest value by category order. Missing values are skipped.
   *
   * @returns The smallest category present in the data, or null if there is none
   * @throws {@link DataValidationError} If the categorical is unordered
   */
  min(): string | null {
    return this.extreme("min");
  }

  /**
   * Largest value by category order. Missing values are skipped.
   *
   * @returns The largest category present in the data, or null if there is none
   * @throws {@link DataValidationError} If the categorical is unordered
   */
  max(): string | null {
    return this.extreme("max");
  }

  /**
   * Sort the values by category order. Missing values always go last.
   *
   * @param ascending - Sort in ascending order (default: true)
   * @returns New Categorical with sorted values
   */
  sort(ascending = true): Categorical {
    const k = this._categories.length;
    const n = this._codes.length;
    const tally = this.countCodes();
    let missing = 0;
    for (let i = 0; i < n; i++) {
      if ((this._codes[i] ?? -1) === -1) missing++;
    }
    // Counting sort over codes: equal codes are indistinguishable, so the result is
    // identical to a stable comparison sort.
    const newCodes = new Int32Array(n);
    let pos = 0;
    if (ascending) {
      for (let c = 0; c < k; c++) {
        const count = tally[c] ?? 0;
        newCodes.fill(c, pos, pos + count);
        pos += count;
      }
    } else {
      for (let c = k - 1; c >= 0; c--) {
        const count = tally[c] ?? 0;
        newCodes.fill(c, pos, pos + count);
        pos += count;
      }
    }
    newCodes.fill(-1, pos, pos + missing);
    return new Categorical([...this._categories], newCodes, this._ordered, this._categoryIndex);
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

  /**
   * Text summary with the first and last values and the category list.
   * Missing values are shown as `NaN`.
   */
  toString(): string {
    const n = this._codes.length;
    const show = (i: number): string => this.get(i) ?? "NaN";
    const parts: string[] = [];
    if (n <= 10) {
      for (let i = 0; i < n; i++) parts.push(show(i));
    } else {
      for (let i = 0; i < 5; i++) parts.push(show(i));
      parts.push("...");
      parts.push(show(n - 2), show(n - 1));
    }
    return `Categorical([${parts.join(", ")}])\nCategories (${this._categories.length}): [${this._categories.join(", ")}]${this._ordered ? " (ordered)" : ""}`;
  }

  /** Occurrence count per category code. */
  private countCodes(): Int32Array {
    const tally = new Int32Array(this._categories.length);
    for (let i = 0; i < this._codes.length; i++) {
      const code = this._codes[i] ?? -1;
      if (code >= 0) tally[code] = (tally[code] ?? 0) + 1;
    }
    return tally;
  }

  /** Apply an old-code -> new-code table (`-1` for dropped) to every element. */
  private remapCodes(remap: Int32Array): Int32Array {
    const newCodes = new Int32Array(this._codes.length);
    for (let i = 0; i < this._codes.length; i++) {
      const old = this._codes[i] ?? -1;
      newCodes[i] = old === -1 ? -1 : (remap[old] ?? -1);
    }
    return newCodes;
  }

  /** Keep the categories accepted by `keep`; elements of dropped categories become missing. */
  private keepCategories(keep: (category: string) => boolean): Categorical {
    const newCats: string[] = [];
    const remap = new Int32Array(this._categories.length).fill(-1);
    for (let i = 0; i < this._categories.length; i++) {
      const cat = this._categories[i];
      if (cat !== undefined && keep(cat)) {
        remap[i] = newCats.length;
        newCats.push(cat);
      }
    }
    return new Categorical(newCats, this.remapCodes(remap), this._ordered);
  }

  private extreme(kind: "min" | "max"): string | null {
    if (!this._ordered) {
      throw new DataValidationError(`${kind}() requires ordered Categorical`);
    }
    let best = -1;
    for (let i = 0; i < this._codes.length; i++) {
      const code = this._codes[i] ?? -1;
      if (code === -1) continue;
      if (best === -1 || (kind === "min" ? code < best : code > best)) best = code;
    }
    return best === -1 ? null : (this._categories[best] ?? null);
  }
}

function buildCategoryIndex(categories: readonly string[]): Map<string, number> {
  const index = new Map<string, number>();
  for (let i = 0; i < categories.length; i++) {
    const cat = categories[i];
    if (cat !== undefined) index.set(cat, i);
  }
  return index;
}

/** Copy a category list after checking that it holds unique strings. */
function validateCategoryList(categories: readonly string[]): string[] {
  const copy = [...categories];
  const seen = new Set<string>();
  for (const cat of copy) {
    if (typeof cat !== "string") {
      throw new DataValidationError(`Categories must be strings; got ${typeof cat}`);
    }
    if (seen.has(cat)) {
      throw new DataValidationError(`Duplicate category '${cat}'`);
    }
    seen.add(cat);
  }
  return copy;
}

/**
 * Sort strings in place by Unicode code point. The native sort compares UTF-16 code units,
 * which gives the same order unless a string holds a surrogate, so it is used when none does.
 */
function sortByCodePoint(values: string[]): string[] {
  const hasSurrogate = values.some((v) => /[\ud800-\udfff]/.test(v));
  return hasSurrogate ? values.sort(compareCodePoints) : values.sort();
}

/** Order strings by Unicode code point (JavaScript's default sorts by UTF-16 code unit). */
function compareCodePoints(a: string, b: string): number {
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    const ca = a.codePointAt(i) ?? 0;
    const cb = b.codePointAt(j) ?? 0;
    if (ca !== cb) return ca < cb ? -1 : 1;
    i += ca > 0xffff ? 2 : 1;
    j += cb > 0xffff ? 2 : 1;
  }
  if (i < a.length) return 1;
  if (j < b.length) return -1;
  return 0;
}
