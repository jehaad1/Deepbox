import { DataValidationError, InvalidParameterError } from "../core/errors/index";

/**
 * Hierarchical (multi-level) index for DataFrames.
 *
 * A MultiIndex represents a composite key made of multiple levels,
 * enabling hierarchical row or column labeling. Each level is a
 * separate array of labels, and the combination of labels across
 * levels uniquely identifies each row.
 *
 * @example
 * ```ts
 * import { MultiIndex } from 'deepbox/dataframe';
 *
 * // Create from arrays
 * const mi = MultiIndex.fromArrays(
 *   [['US', 'US', 'UK', 'UK'], [2020, 2021, 2020, 2021]],
 *   ['country', 'year']
 * );
 * console.log(mi.length);  // 4
 * console.log(mi.nlevels); // 2
 * console.log(mi.get(0));  // ['US', 2020]
 *
 * // Create from tuples
 * const mi2 = MultiIndex.fromTuples(
 *   [['US', 2020], ['US', 2021], ['UK', 2020], ['UK', 2021]],
 *   ['country', 'year']
 * );
 * ```
 *
 * @see {@link https://deepbox.dev/docs/dataframe-series | Deepbox MultiIndex}
 */
export class MultiIndex {
  /** Level arrays: levels[i] contains the labels for level i */
  private readonly _levels: ReadonlyArray<readonly (string | number)[]>;
  /** Level names */
  private readonly _names: readonly string[];
  /** Number of entries */
  private readonly _length: number;
  /** Flat index for lookup: stringified tuple -> position */
  private readonly _tupleIndex: Map<string, number>;

  private constructor(
    levels: ReadonlyArray<readonly (string | number)[]>,
    names: readonly string[]
  ) {
    this._levels = levels;
    this._names = names;
    this._length = levels.length > 0 ? (levels[0]?.length ?? 0) : 0;

    // Build tuple lookup
    this._tupleIndex = new Map<string, number>();
    for (let i = 0; i < this._length; i++) {
      const key = this.tupleKey(i);
      this._tupleIndex.set(key, i);
    }
  }

  /**
   * Create a MultiIndex from separate arrays for each level.
   *
   * @param arrays - Array of level arrays. All must have the same length.
   * @param names - Names for each level (default: level_0, level_1, ...)
   * @returns New MultiIndex
   *
   * @example
   * ```ts
   * const mi = MultiIndex.fromArrays(
   *   [['a', 'a', 'b', 'b'], [1, 2, 1, 2]],
   *   ['letter', 'number']
   * );
   * ```
   */
  static fromArrays(
    arrays: ReadonlyArray<readonly (string | number)[]>,
    names?: readonly string[]
  ): MultiIndex {
    if (arrays.length === 0) {
      throw new InvalidParameterError(
        "At least one level array is required",
        "arrays",
        arrays.length
      );
    }

    const length = arrays[0]?.length ?? 0;
    for (let i = 1; i < arrays.length; i++) {
      if ((arrays[i]?.length ?? 0) !== length) {
        throw new DataValidationError(
          `All level arrays must have the same length; level 0 has ${length}, level ${i} has ${arrays[i]?.length ?? 0}`
        );
      }
    }

    const levelNames = names ?? Array.from({ length: arrays.length }, (_, i) => `level_${i}`);

    if (levelNames.length !== arrays.length) {
      throw new InvalidParameterError(
        `names length (${levelNames.length}) must match number of levels (${arrays.length})`,
        "names",
        levelNames.length
      );
    }

    return new MultiIndex(
      arrays.map((arr) => [...arr]),
      [...levelNames]
    );
  }

  /**
   * Create a MultiIndex from an array of tuples.
   *
   * @param tuples - Array of label tuples
   * @param names - Names for each level
   * @returns New MultiIndex
   *
   * @example
   * ```ts
   * const mi = MultiIndex.fromTuples(
   *   [['a', 1], ['a', 2], ['b', 1]],
   *   ['letter', 'number']
   * );
   * ```
   */
  static fromTuples(
    tuples: ReadonlyArray<readonly (string | number)[]>,
    names?: readonly string[]
  ): MultiIndex {
    if (tuples.length === 0) {
      throw new InvalidParameterError("At least one tuple is required", "tuples", tuples.length);
    }

    const nlevels = tuples[0]?.length ?? 0;
    for (let i = 1; i < tuples.length; i++) {
      if ((tuples[i]?.length ?? 0) !== nlevels) {
        throw new DataValidationError(
          `All tuples must have the same length; tuple 0 has ${nlevels}, tuple ${i} has ${tuples[i]?.length ?? 0}`
        );
      }
    }

    // Transpose tuples into level arrays
    const arrays: (string | number)[][] = [];
    for (let lvl = 0; lvl < nlevels; lvl++) {
      const level: (string | number)[] = [];
      for (const tuple of tuples) {
        const val = tuple[lvl];
        level.push(val ?? 0);
      }
      arrays.push(level);
    }

    return MultiIndex.fromArrays(arrays, names);
  }

  /**
   * Create a MultiIndex representing all combinations of the given levels
   * (Cartesian product).
   *
   * @param levels - Array of unique values for each level
   * @param names - Names for each level
   * @returns New MultiIndex with all combinations
   *
   * @example
   * ```ts
   * const mi = MultiIndex.fromProduct(
   *   [['a', 'b'], [1, 2, 3]],
   *   ['letter', 'number']
   * );
   * // Results in: (a,1), (a,2), (a,3), (b,1), (b,2), (b,3)
   * ```
   */
  static fromProduct(
    levels: ReadonlyArray<readonly (string | number)[]>,
    names?: readonly string[]
  ): MultiIndex {
    if (levels.length === 0) {
      throw new InvalidParameterError("At least one level is required", "levels", levels.length);
    }

    // Compute Cartesian product
    const tuples: (string | number)[][] = [[]];
    for (const level of levels) {
      const newTuples: (string | number)[][] = [];
      for (const existing of tuples) {
        for (const val of level) {
          newTuples.push([...existing, val]);
        }
      }
      tuples.length = 0;
      tuples.push(...newTuples);
    }

    return MultiIndex.fromTuples(tuples, names);
  }

  /** Number of entries */
  get length(): number {
    return this._length;
  }

  /** Number of levels */
  get nlevels(): number {
    return this._levels.length;
  }

  /** Level names */
  get names(): readonly string[] {
    return this._names;
  }

  /** Level arrays */
  get levels(): ReadonlyArray<readonly (string | number)[]> {
    return this._levels;
  }

  /**
   * Get the label tuple at a given position.
   *
   * @param index - Zero-based position
   * @returns Array of labels for each level
   */
  get(index: number): (string | number)[] {
    if (index < 0 || index >= this._length) {
      throw new InvalidParameterError(
        `Index ${index} out of bounds [0, ${this._length})`,
        "index",
        index
      );
    }
    const result: (string | number)[] = [];
    for (const level of this._levels) {
      result.push(level[index] ?? 0);
    }
    return result;
  }

  /**
   * Get a specific level's values.
   *
   * @param level - Level index or name
   * @returns Array of values at that level
   */
  getLevel(level: number | string): readonly (string | number)[] {
    const idx = typeof level === "string" ? this._names.indexOf(level) : level;
    if (idx < 0 || idx >= this._levels.length) {
      throw new InvalidParameterError(`Level '${String(level)}' not found`, "level", level);
    }
    return this._levels[idx] ?? [];
  }

  /**
   * Get unique values at a specific level.
   *
   * @param level - Level index or name
   * @returns Array of unique values
   */
  getLevelValues(level: number | string): (string | number)[] {
    const data = this.getLevel(level);
    const seen = new Set<string | number>();
    const result: (string | number)[] = [];
    for (const val of data) {
      if (!seen.has(val)) {
        seen.add(val);
        result.push(val);
      }
    }
    return result;
  }

  /**
   * Find the position of a given label tuple.
   *
   * @param labels - Label values for each level
   * @returns Position index, or -1 if not found
   */
  getPosition(labels: readonly (string | number)[]): number {
    const key = labels.map(String).join("\0");
    return this._tupleIndex.get(key) ?? -1;
  }

  /**
   * Select rows matching a partial label specification.
   *
   * @param partialLabels - Object mapping level names to values to match
   * @returns Array of matching position indices
   */
  select(partialLabels: Readonly<Record<string, string | number>>): number[] {
    const result: number[] = [];
    for (let i = 0; i < this._length; i++) {
      let match = true;
      for (const [name, value] of Object.entries(partialLabels)) {
        const lvlIdx = this._names.indexOf(name);
        if (lvlIdx === -1) {
          throw new DataValidationError(`Level '${name}' not found in MultiIndex`);
        }
        if ((this._levels[lvlIdx]?.[i] ?? 0) !== value) {
          match = false;
          break;
        }
      }
      if (match) result.push(i);
    }
    return result;
  }

  /**
   * Drop a level from the MultiIndex.
   *
   * @param level - Level index or name to drop
   * @returns New MultiIndex without the specified level
   */
  droplevel(level: number | string): MultiIndex {
    const idx = typeof level === "string" ? this._names.indexOf(level) : level;
    if (idx < 0 || idx >= this._levels.length) {
      throw new InvalidParameterError(`Level '${String(level)}' not found`, "level", level);
    }
    if (this._levels.length <= 1) {
      throw new DataValidationError("Cannot drop the last level of a MultiIndex");
    }
    const newLevels = this._levels.filter((_, i) => i !== idx);
    const newNames = this._names.filter((_, i) => i !== idx);
    return new MultiIndex(newLevels, newNames);
  }

  /**
   * Swap two levels in the MultiIndex.
   *
   * @param i - First level (index or name)
   * @param j - Second level (index or name)
   * @returns New MultiIndex with swapped levels
   */
  swaplevel(i: number | string, j: number | string): MultiIndex {
    const iIdx = typeof i === "string" ? this._names.indexOf(i) : i;
    const jIdx = typeof j === "string" ? this._names.indexOf(j) : j;
    if (iIdx < 0 || iIdx >= this._levels.length) {
      throw new InvalidParameterError(`Level '${String(i)}' not found`, "i", i);
    }
    if (jIdx < 0 || jIdx >= this._levels.length) {
      throw new InvalidParameterError(`Level '${String(j)}' not found`, "j", j);
    }
    const newLevels = [...this._levels];
    const newNames = [...this._names];
    const tmpLevel = newLevels[iIdx];
    const tmpName = newNames[iIdx];
    if (tmpLevel !== undefined && tmpName !== undefined) {
      newLevels[iIdx] = newLevels[jIdx] ?? [];
      newNames[iIdx] = newNames[jIdx] ?? "";
      newLevels[jIdx] = tmpLevel;
      newNames[jIdx] = tmpName;
    }
    return new MultiIndex(newLevels, newNames);
  }

  /**
   * Convert to a flat array of stringified tuples for use as DataFrame index.
   *
   * @param separator - Separator between level values (default: ', ')
   * @returns Array of string labels
   */
  toFlatIndex(separator = ", "): string[] {
    const result: string[] = [];
    for (let i = 0; i < this._length; i++) {
      const parts: string[] = [];
      for (const level of this._levels) {
        parts.push(String(level[i] ?? ""));
      }
      result.push(`(${parts.join(separator)})`);
    }
    return result;
  }

  /**
   * Stringify a tuple at a given index for internal use.
   */
  private tupleKey(index: number): string {
    const parts: string[] = [];
    for (const level of this._levels) {
      parts.push(String(level[index] ?? ""));
    }
    return parts.join("\0");
  }

  toString(): string {
    const header = `MultiIndex(${this._length} entries, ${this._levels.length} levels)`;
    const nameStr = `Names: [${this._names.join(", ")}]`;
    const maxShow = Math.min(5, this._length);
    const entries: string[] = [];
    for (let i = 0; i < maxShow; i++) {
      entries.push(`(${this.get(i).map(String).join(", ")})`);
    }
    if (this._length > 5) {
      entries.push("...");
    }
    return `${header}\n${nameStr}\n[${entries.join(", ")}]`;
  }
}
