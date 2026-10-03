/**
 * Scalar value that can be stored in a DataFrame or Series cell.
 *
 * Columns are not limited to these types (dates, bigints and objects are
 * accepted too), but these are the values every operation understands.
 * `null` and `undefined` mark missing values.
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */
export type DataValue = number | string | boolean | null | undefined;

/**
 * Column-oriented data for DataFrame construction.
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
 * Options for `DataFrame.groupBy`.
 *
 * The defaults keep the behavior of Deepbox 1.0: groups are listed in order of first
 * appearance and missing keys form groups. pandas sorts keys and drops missing keys by
 * default, so pass `{ sort: true, dropna: true }` to match it.
 *
 * @property sort - List groups sorted by key, ascending, column by column (default: false)
 * @property dropna - Drop groups whose key has a null, undefined or NaN part (default: false)
 */
export type GroupByOptions = {
  readonly sort?: boolean;
  readonly dropna?: boolean;
};

/**
 * Supported aggregation functions for DataFrame groupby operations.
 *
 * Numeric aggregations skip `null`, `undefined` and `NaN` and throw on non-numeric values.
 * With no values left, `sum` is 0 and the other numeric aggregations are `NaN`.
 *
 * - `sum`: Sum of values
 * - `mean`: Arithmetic mean
 * - `median`: Median value
 * - `min`: Minimum value
 * - `max`: Maximum value
 * - `std`: Sample standard deviation (ddof = 1; `NaN` for fewer than two values)
 * - `var`: Sample variance (ddof = 1; `NaN` for fewer than two values)
 * - `count`: Number of non-missing values
 * - `first`: First value in the group (missing values included)
 * - `last`: Last value in the group (missing values included)
 * - `nunique`: Number of distinct values, not counting missing ones
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
  | "last"
  | "nunique";

/**
 * Named aggregation for `DataFrameGroupBy.agg`: a `[column, aggregation]` pair that is
 * stored under the output name used as its key, like pandas' `agg(out=("col", "mean"))`.
 * The aggregation is a name from {@link AggregateFunction} or a function that receives
 * the values of the column inside one group (missing values included).
 *
 * @example
 * ```ts
 * df.groupBy("team").agg({ avgScore: ["score", "mean"], best: ["score", (v) => Math.max(...(v as number[]))] });
 * ```
 */
export type NamedAggregation = readonly [
  column: string,
  aggregation: AggregateFunction | ((values: unknown[]) => unknown),
];

/** Direction of a fill: `"ffill"` carries the last valid value forward, `"bfill"` the next one back. */
export type FillMethod = "ffill" | "bfill";

/**
 * Options for filling missing values by propagating neighbours.
 *
 * @property method - `"ffill"` (alias `"pad"`) or `"bfill"` (alias `"backfill"`)
 * @property limit - Maximum number of consecutive missing values to fill per gap (default: no limit)
 * @property axis - 0 fills down each column (default), 1 fills across each row. DataFrame only.
 */
export type FillnaMethodOptions = {
  readonly method: FillMethod | "pad" | "backfill";
  readonly limit?: number;
  readonly axis?: number | "index" | "rows" | "columns";
};

/**
 * Options for `DataFrame.ffill` and `DataFrame.bfill`.
 *
 * @property limit - Maximum number of consecutive missing values to fill per gap (default: no limit)
 * @property axis - 0 fills down each column (default), 1 fills across each row
 */
export type FillOptions = {
  readonly limit?: number;
  readonly axis?: number | "index" | "rows" | "columns";
};

/** Correlation coefficient computed by `DataFrame.corr`. */
export type CorrelationMethod = "pearson" | "spearman" | "kendall";

/**
 * Options for `DataFrame.corr`.
 *
 * @property method - `"pearson"` (default), `"spearman"` (rank correlation) or `"kendall"` (tau-b)
 * @property minPeriods - Minimum number of complete pairs a cell needs, otherwise it is NaN (default: 1)
 */
export type CorrOptions = {
  readonly method?: CorrelationMethod;
  readonly minPeriods?: number;
};

/**
 * Options for `DataFrame.sample`.
 *
 * @property n - Number of rows to draw (default: 1 when `frac` is not given). Exclusive with `frac`.
 * @property frac - Fraction of the rows to draw, rounded half to even like pandas. Exclusive with `n`.
 * @property replace - Draw with replacement (default: false)
 * @property weights - Non-negative sampling weights: one per row, or the name of a numeric column.
 *   Missing weights count as 0. They are normalized to sum to 1.
 * @property randomState - Integer seed. Without it the global generator is used, which follows `setSeed`.
 */
export type SampleOptions = {
  readonly n?: number;
  readonly frac?: number;
  readonly replace?: boolean;
  readonly weights?: readonly number[] | string;
  readonly randomState?: number;
};

/**
 * Options for `DataFrame.rolling`.
 *
 * @property on - Column to compute on (default: every column)
 * @property minPeriods - Minimum number of valid values in a window for a result (default: the window size)
 * @property center - Label the window at its center instead of its right edge (default: false)
 */
export type RollingOptions = {
  readonly on?: string;
  readonly minPeriods?: number;
  readonly center?: boolean;
};

/**
 * Options for `DataFrame.concat`.
 *
 * @property join - `"outer"` keeps the union of the labels on the other axis (missing cells become
 *   null), `"inner"` keeps only the shared labels. Without it, `axis=0` requires identical columns
 *   (as in Deepbox 1.0) and `axis=1` aligns on the union of the row labels.
 * @property ignoreIndex - `true` renumbers the rows 0..n-1 (axis 0) or the columns 0..m-1 (axis 1).
 *   `false` keeps the original row labels and throws when they collide. Without it, axis 0 renumbers
 *   as in Deepbox 1.0.
 */
export type ConcatOptions = {
  readonly join?: "outer" | "inner";
  readonly ignoreIndex?: boolean;
};

/**
 * Options for `Series.valueCounts` and `DataFrame.valueCounts`.
 *
 * @property normalize - Return relative frequencies instead of counts (default: false)
 * @property dropna - Leave out missing values (default: true for a Series, false for a DataFrame)
 * @property sort - Order by count (default: true)
 * @property ascending - Ascending count order when sorting (default: false)
 */
export type ValueCountsOptions = {
  readonly normalize?: boolean;
  readonly dropna?: boolean;
  readonly sort?: boolean;
  readonly ascending?: boolean;
};
