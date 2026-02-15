/**
 * Data structure for DataFrame initialization.
 *
 * Maps column names to arrays of values. All arrays must have the same length.
 *
 * @example
 * ```ts
 * const data: DataFrameData = {
 *   name: ['Alice', 'Bob', 'Charlie'],
 *   age: [25, 30, 35],
 *   score: [85.5, 92.0, 78.5]
 * };
 * ```
 */
export type DataValue = number | string | boolean | null | undefined;

export type DataFrameData = Record<string, unknown[]>;

/**
 * Configuration options for DataFrame construction.
 *
 * @property index - Custom row labels (defaults to 0, 1, 2, ...). Can be strings or numbers.
 * @property columns - Custom column order (defaults to Object.keys order)
 * @property copy - Whether to copy data on construction (default: true). Set to false for performance if data ownership can be transferred.
 */
export type DataFrameOptions = {
  index?: (string | number)[];
  columns?: string[];
  copy?: boolean;
};

/**
 * Configuration options for Series construction.
 *
 * @property name - Optional name for the Series
 * @property index - Custom index labels (defaults to 0, 1, 2, ...). Can be strings or numbers.
 * @property copy - Whether to copy data on construction (default: true). Set to false for performance if data ownership can be transferred.
 */
export type SeriesOptions = {
  name?: string;
  index?: (string | number)[];
  copy?: boolean;
};

/**
 * Supported aggregation functions for DataFrame groupby operations.
 *
 * - `sum`: Sum of values
 * - `mean`: Arithmetic mean
 * - `median`: Median value
 * - `min`: Minimum value
 * - `max`: Maximum value
 * - `std`: Standard deviation
 * - `var`: Variance
 * - `count`: Count of non-null values
 * - `first`: First value in group
 * - `last`: Last value in group
 */
export type AggregateFunction =
  | "sum"
  | "mean"
  | "median"
  | "min"
  | "max"
  | "std"
  | "var"
  | "count"
  | "first"
  | "last";
