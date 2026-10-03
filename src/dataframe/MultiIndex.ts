import { DataValidationError, InvalidParameterError } from "../core/errors/index";

type Label = string | number;

/**
 * Build an unambiguous lookup key for one label tuple. Strings are length
 * prefixed, so separators inside a label can never make two tuples collide,
 * and the type tag keeps 1 and "1" apart.
 */
function tupleKeyOf(labels: readonly Label[]): string {
  let key = "";
  for (const label of labels) {
    key += typeof label === "number" ? `n${String(label)}|` : `s${label.length}:${String(label)}|`;
  }
  return key;
}

/** Equality that treats NaN as equal to NaN, so NaN labels can be selected. */
function sameLabel(a: unknown, b: unknown): boolean {
  return a === b || (Number.isNaN(a) && Number.isNaN(b));
}

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
  /** Per-level label arrays: `_levels[i][row]` is the label of `row` at level i */
  private readonly _levels: ReadonlyArray<readonly (string | number)[]>;
  /** Level names */
  private readonly _names: readonly string[];
  /** Number of entries */
  private readonly _length: number;
  /** Lookup from tuple key to the first position holding that tuple */
  private readonly _tupleIndex: Map<string, number>;

  private constructor(
    levels: ReadonlyArray<readonly (string | number)[]>,
    names: readonly string[]
  ) {
    this._levels = levels;
    this._names = names;
    this._length = levels.length > 0 ? (levels[0]?.length ?? 0) : 0;

    // Build tuple lookup. Duplicate tuples are allowed; lookups return the first one.
    this._tupleIndex = new Map<string, number>();
    for (let i = 0; i < this._length; i++) {
      const key = this.tupleKey(i);
      if (!this._tupleIndex.has(key)) {
        this._tupleIndex.set(key, i);
      }
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

    for (let i = 0; i < arrays.length; i++) {
      if (!Array.isArray(arrays[i])) {
        throw new InvalidParameterError(
          `Level ${i} must be an array of labels`,
          "arrays",
          arrays[i]
        );
      }
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
   * @param tuples - Array of label tuples. All must have the same length. An empty
   *   array is accepted only together with `names`, which then fixes the number of levels.
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
      if (names !== undefined && names.length > 0) {
        return MultiIndex.fromArrays(
          names.map(() => []),
          names
        );
      }
      throw new InvalidParameterError(
        "At least one tuple is required (or pass names to create an empty MultiIndex)",
        "tuples",
        tuples.length
      );
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
      for (let row = 0; row < tuples.length; row++) {
        const val = tuples[row]?.[lvl];
        if (val === undefined) {
          throw new DataValidationError(`Tuple ${row} has no label at level ${lvl}`);
        }
        level.push(val);
      }
      arrays.push(level);
    }

    return MultiIndex.fromArrays(arrays, names);
  }

  /**
   * Create a MultiIndex representing all combinations of the given levels
   * (Cartesian product). The last level varies fastest.
   *
   * @param levels - Array of unique values for each level
   * @param names - Names for each level
   * @returns New MultiIndex with all combinations (empty if any level is empty)
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

    let total = 1;
    for (const level of levels) total *= level.length;

    // Level k repeats each of its labels `inner` times in a row, and the whole
    // pattern is tiled `total / (inner * size)` times.
    const arrays: (string | number)[][] = [];
    let inner = total;
    for (const level of levels) {
      const size = level.length;
      inner = size === 0 ? 0 : inner / size;
      const column: (string | number)[] = new Array(total);
      for (let row = 0; row < total; row++) {
        column[row] = level[Math.floor(row / inner) % size] as string | number;
      }
      arrays.push(column);
    }

    return MultiIndex.fromArrays(arrays, names);
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

  /**
   * Label arrays, one per level. `levels[i][row]` is the label of entry `row`
   * at level `i`, so every array has `length` entries and may repeat labels.
   * Use {@link MultiIndex.getLevelValues} for the distinct labels of a level.
   */
  get levels(): ReadonlyArray<readonly (string | number)[]> {
    return this._levels;
  }

  /** True when no two entries have the same label tuple. */
  get isUnique(): boolean {
    return this._tupleIndex.size === this._length;
  }

  /**
   * Get the label tuple at a given position.
   *
   * @param index - Zero-based position
   * @returns Array of labels for each level
   * @throws {InvalidParameterError} If index is not an integer inside `[0, length)`
   */
  get(index: number): (string | number)[] {
    if (!Number.isInteger(index) || index < 0 || index >= this._length) {
      throw new InvalidParameterError(
        `Index ${index} out of bounds [0, ${this._length})`,
        "index",
        index
      );
    }
    const result: (string | number)[] = [];
    for (const level of this._levels) {
      result.push(level[index] as string | number);
    }
    return result;
  }

  /**
   * Get all label tuples as arrays.
   *
   * @returns One `[label_0, label_1, ...]` array per entry
   */
  toTuples(): (string | number)[][] {
    const result: (string | number)[][] = new Array(this._length);
    for (let i = 0; i < this._length; i++) {
      result[i] = this.get(i);
    }
    return result;
  }

  /**
   * Resolve a level given by position or name to its position.
   * Negative positions count from the last level.
   */
  private resolveLevel(level: number | string, param: string): number {
    if (typeof level === "string") {
      const first = this._names.indexOf(level);
      if (first === -1) {
        throw new InvalidParameterError(`Level '${level}' not found`, param, level);
      }
      if (this._names.indexOf(level, first + 1) !== -1) {
        throw new DataValidationError(
          `Level name '${level}' occurs more than once; use the level position instead`
        );
      }
      return first;
    }
    const n = this._levels.length;
    const idx = Number.isInteger(level) && level < 0 ? level + n : level;
    if (!Number.isInteger(idx) || idx < 0 || idx >= n) {
      throw new InvalidParameterError(
        `Level ${String(level)} not found; the index has ${n} level${n === 1 ? "" : "s"}`,
        param,
        level
      );
    }
    return idx;
  }

  /**
   * Get a specific level's values.
   *
   * @param level - Level position (negative counts from the end) or name
   * @returns Array of values at that level, one per entry
   * @throws {InvalidParameterError} If the level does not exist
   */
  getLevel(level: number | string): readonly (string | number)[] {
    return this._levels[this.resolveLevel(level, "level")] ?? [];
  }

  /**
   * Get unique values at a specific level, in order of first appearance.
   *
   * @param level - Level position (negative counts from the end) or name
   * @returns Array of unique values
   * @throws {InvalidParameterError} If the level does not exist
   */
  getLevelValues(level: number | string): (string | number)[] {
    return [...new Set(this.getLevel(level))];
  }

  /**
   * Find the position of a given label tuple. With duplicate tuples, the first
   * position is returned.
   *
   * @param labels - Label values for each level
   * @returns Position index, or -1 if not found
   */
  getPosition(labels: readonly (string | number)[]): number {
    if (labels.length !== this._levels.length) {
      return -1;
    }
    return this._tupleIndex.get(tupleKeyOf(labels)) ?? -1;
  }

  /**
   * Whether a full label tuple is present.
   *
   * @param labels - Label values for each level
   */
  has(labels: readonly (string | number)[]): boolean {
    return this.getPosition(labels) !== -1;
  }

  /**
   * Whether another MultiIndex has the same names, levels and labels in the same order.
   */
  equals(other: MultiIndex): boolean {
    if (
      other._length !== this._length ||
      other._levels.length !== this._levels.length ||
      other._names.some((name, i) => name !== this._names[i])
    ) {
      return false;
    }
    return this._levels.every((level, l) =>
      level.every((label, row) => sameLabel(label, other._levels[l]?.[row]))
    );
  }

  /**
   * Select rows matching a partial label specification.
   *
   * @param partialLabels - Object mapping level names to values to match
   * @returns Array of matching position indices
   * @throws {DataValidationError} If a level name does not exist
   */
  select(partialLabels: Readonly<Record<string, string | number>>): number[] {
    const conditions: Array<[readonly Label[], Label]> = [];
    for (const [name, value] of Object.entries(partialLabels)) {
      const lvlIdx = this._names.indexOf(name);
      if (lvlIdx === -1) {
        throw new DataValidationError(`Level '${name}' not found in MultiIndex`);
      }
      conditions.push([this._levels[lvlIdx] ?? [], value]);
    }

    const result: number[] = [];
    for (let i = 0; i < this._length; i++) {
      if (conditions.every(([level, value]) => sameLabel(level[i], value))) {
        result.push(i);
      }
    }
    return result;
  }

  /**
   * Drop a level from the MultiIndex.
   *
   * @param level - Level position (negative counts from the end) or name to drop
   * @returns New MultiIndex without the specified level
   * @throws {InvalidParameterError} If the level does not exist
   * @throws {DataValidationError} If it is the only level
   */
  droplevel(level: number | string): MultiIndex {
    const idx = this.resolveLevel(level, "level");
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
   * @param i - First level (position or name; negative positions count from the end)
   * @param j - Second level (position or name)
   * @returns New MultiIndex with swapped levels
   * @throws {InvalidParameterError} If either level does not exist
   */
  swaplevel(i: number | string, j: number | string): MultiIndex {
    const iIdx = this.resolveLevel(i, "i");
    const jIdx = this.resolveLevel(j, "j");
    const newLevels = [...this._levels];
    const newNames = [...this._names];
    newLevels[iIdx] = this._levels[jIdx] as readonly Label[];
    newLevels[jIdx] = this._levels[iIdx] as readonly Label[];
    newNames[iIdx] = this._names[jIdx] as string;
    newNames[jIdx] = this._names[iIdx] as string;
    return new MultiIndex(newLevels, newNames);
  }

  /**
   * Convert to a flat array of stringified tuples for use as DataFrame index.
   *
   * @param separator - Separator between level values (default: ', ')
   * @returns Array of string labels such as `"(a, 1)"`
   */
  toFlatIndex(separator = ", "): string[] {
    if (typeof separator !== "string") {
      throw new InvalidParameterError("separator must be a string", "separator", separator);
    }
    const result: string[] = [];
    for (let i = 0; i < this._length; i++) {
      const parts: string[] = [];
      for (const level of this._levels) {
        parts.push(String(level[i]));
      }
      result.push(`(${parts.join(separator)})`);
    }
    return result;
  }

  /**
   * Lookup key of the tuple at a given position.
   */
  private tupleKey(index: number): string {
    return tupleKeyOf(this._levels.map((level) => level[index] as Label));
  }

  /**
   * Describe the index: size, level names and the first five entries.
   */
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
