/**
 * DataFrame plotting accessor.
 *
 * Provides a pandas-like `df.plot.line()`, `df.plot.bar()`, etc. API
 * that delegates to the main `deepbox/plot` module.
 *
 * @module dataframe/PlotAccessor
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError } from "../core/errors/index";
import { type Tensor, tensor } from "../ndarray";
import type { Axes } from "../plot/figure/Axes";
import { Figure } from "../plot/figure/Figure";

/** Columnar view of a DataFrame that the accessor reads from. */
type PlotData = {
  readonly columns: string[];
  readonly getColumn: (name: string) => unknown[];
  readonly nRows: number;
};

/** Options shared by every plot method. */
export type PlotBaseOptions = {
  /** Figure size as `[width, height]` in pixels (default: `[640, 480]`). */
  readonly figsize?: readonly [number, number];
  /** Title drawn above the axes. */
  readonly title?: string;
};

/** Values that count as numeric when choosing default columns. */
function isNumericColumn(values: readonly unknown[]): boolean {
  for (const v of values) {
    if (v === null || v === undefined) continue;
    if (typeof v !== "number" && typeof v !== "bigint") return false;
  }
  return true;
}

/**
 * Plotting accessor for DataFrame.
 *
 * Access via `df.plot.line()`, `df.plot.bar()`, etc. Every method returns the
 * {@link Figure}, which can be rendered with `renderSVG()` or saved. Columns that
 * are named in the options must exist; otherwise an InvalidParameterError lists the
 * available columns. When no columns are named, only numeric columns are used.
 * Missing values (null, undefined, NaN) are skipped by the underlying plots.
 *
 * @example
 * ```ts
 * import { DataFrame } from 'deepbox/dataframe';
 *
 * const df = new DataFrame({ x: [1, 2, 3], y: [4, 5, 6] });
 * const fig = df.plot.line({ x: 'x', y: 'y' });
 * ```
 */
export class PlotAccessor {
  private readonly getData: () => PlotData;

  /** @internal */
  constructor(getData: () => PlotData) {
    this.getData = getData;
  }

  private createFigAxes(options?: PlotBaseOptions): { fig: Figure; ax: Axes } {
    const size = options?.figsize;
    if (size !== undefined) {
      if (size.length !== 2 || !size.every((v) => Number.isFinite(v) && v > 0)) {
        throw new InvalidParameterError(
          "figsize must be [width, height] with positive finite numbers",
          "figsize",
          size
        );
      }
    }
    const fig = new Figure({ width: size?.[0] ?? 640, height: size?.[1] ?? 480 });
    const ax = fig.addAxes();
    if (options?.title) ax.setTitle(options.title);
    return { fig, ax };
  }

  /** Read a column by name, rejecting names that are not in the DataFrame. */
  private column(data: PlotData, name: string, param: string): unknown[] {
    if (!data.columns.includes(name)) {
      throw new InvalidParameterError(
        `Column '${String(name)}' not found; available columns: ${data.columns.join(", ")}`,
        param,
        name
      );
    }
    return data.getColumn(name);
  }

  /** Resolve a column option (`undefined`, a name or a list) to a list of names. */
  private resolveColumns(
    data: PlotData,
    value: string | readonly string[] | undefined,
    param: string,
    exclude?: string
  ): string[] {
    if (value === undefined) {
      return data.columns.filter((c) => c !== exclude && isNumericColumn(data.getColumn(c)));
    }
    const names = typeof value === "string" ? [value] : [...value];
    for (const name of names) this.column(data, name, param);
    return names;
  }

  /**
   * Convert a column to numbers. null and undefined become NaN, booleans become
   * 0/1 and Dates become epoch milliseconds. Strings must hold numbers.
   */
  private toNumbers(col: readonly unknown[], name: string): number[] {
    const out: number[] = new Array(col.length);
    for (let i = 0; i < col.length; i++) {
      const v = col[i];
      if (typeof v === "number") {
        out[i] = v;
      } else if (v === null || v === undefined) {
        out[i] = Number.NaN;
      } else if (typeof v === "bigint") {
        out[i] = Number(v);
      } else if (typeof v === "boolean") {
        out[i] = v ? 1 : 0;
      } else if (v instanceof Date) {
        out[i] = v.getTime();
      } else if (typeof v === "string" && v.trim() !== "" && Number.isFinite(Number(v))) {
        out[i] = Number(v);
      } else {
        throw new DataValidationError(
          `Column '${name}' has a non-numeric value at row ${i}: ${String(v)}`
        );
      }
    }
    return out;
  }

  private numeric(data: PlotData, name: string, param: string): number[] {
    return this.toNumbers(this.column(data, name, param), name);
  }

  /**
   * Positions and optional tick labels for a category column. Numeric columns are
   * used as they are; any other column is drawn at 0, 1, 2, ... with its values as labels.
   */
  private categories(
    data: PlotData,
    name: string,
    param: string
  ): { positions: number[]; labels?: string[] } {
    const col = this.column(data, name, param);
    if (isNumericColumn(col)) return { positions: this.toNumbers(col, name) };
    return {
      positions: col.map((_, i) => i),
      labels: col.map((v) => (v instanceof Date ? v.toISOString() : String(v))),
    };
  }

  /**
   * Line plot.
   *
   * @param options - Plot configuration
   * @param options.x - Column for the x values (default: the row number)
   * @param options.y - Column or columns to draw (default: all numeric columns except `x`)
   * @returns The Figure containing the plot
   * @throws {InvalidParameterError} If a named column does not exist
   * @throws {DataValidationError} If a column holds non-numeric values
   */
  line(
    options: PlotBaseOptions & {
      readonly x?: string;
      readonly y?: string | readonly string[];
    } = {}
  ): Figure {
    const { fig, ax } = this.createFigAxes(options);
    const data = this.getData();

    const xCol = options.x;
    const xValues =
      xCol !== undefined
        ? this.numeric(data, xCol, "x")
        : Array.from({ length: data.nRows }, (_, i) => i);
    const yCols = this.resolveColumns(data, options.y, "y", xCol);

    const xT = tensor(xValues);
    for (const yCol of yCols) {
      ax.plot(xT, tensor(this.numeric(data, yCol, "y")), { label: yCol });
    }

    if (yCols.length > 1) ax.legend();
    if (xCol) ax.setXLabel(xCol);
    return fig;
  }

  /**
   * Bar plot. Several `y` columns are drawn as side-by-side groups.
   *
   * @param options - Plot configuration
   * @param options.x - Column for the bar positions (default: the first column).
   *   A non-numeric column is drawn as categories, labelled with its values.
   * @param options.y - Column or columns for the bar heights (default: all numeric columns except `x`)
   * @returns The Figure containing the plot
   * @throws {InvalidParameterError} If a named column does not exist
   * @throws {DataValidationError} If a height column holds non-numeric values
   */
  bar(
    options: PlotBaseOptions & {
      readonly x?: string;
      readonly y?: string | readonly string[];
    } = {}
  ): Figure {
    const { fig, ax } = this.createFigAxes(options);
    const data = this.getData();

    const xCol = options.x ?? data.columns[0];
    if (!xCol) return fig;
    const { positions, labels } = this.categories(data, xCol, "x");
    const yCols = this.resolveColumns(data, options.y, "y", xCol);

    const xT = tensor(positions);
    if (yCols.length === 1) {
      const yCol = yCols[0] as string;
      ax.bar(xT, tensor(this.numeric(data, yCol, "y")), { label: yCol });
    } else if (yCols.length > 1) {
      const heights: Tensor[] = yCols.map((c) => tensor(this.numeric(data, c, "y")));
      ax.groupedBar(xT, heights, { labels: yCols });
      ax.legend();
    }

    if (labels) ax.setXTicks(positions, labels);
    ax.setXLabel(xCol);
    return fig;
  }

  /**
   * Horizontal bar plot. As in matplotlib's `barh`, `y` names the category
   * column drawn along the vertical axis and `x` names the column of bar lengths.
   *
   * @param options - Plot configuration
   * @param options.y - Column for the bar positions (default: the first column).
   *   A non-numeric column is drawn as categories, labelled with its values.
   * @param options.x - Column for the bar lengths (default: the second column)
   * @returns The Figure containing the plot
   * @throws {InvalidParameterError} If a named column does not exist
   * @throws {DataValidationError} If the length column holds non-numeric values
   */
  barh(
    options: PlotBaseOptions & {
      readonly x?: string;
      readonly y?: string;
    } = {}
  ): Figure {
    const { fig, ax } = this.createFigAxes(options);
    const data = this.getData();

    const yCol = options.y ?? data.columns[0];
    const xCol = options.x ?? data.columns[1];
    if (!yCol || !xCol) return fig;

    const { positions, labels } = this.categories(data, yCol, "y");
    ax.barh(tensor(positions), tensor(this.numeric(data, xCol, "x")));
    if (labels) ax.setYTicks(positions, labels);
    ax.setXLabel(xCol);
    ax.setYLabel(yCol);
    return fig;
  }

  /**
   * Histogram.
   *
   * @param options - Plot configuration
   * @param options.column - Column or columns to bin (default: all numeric columns)
   * @param options.bins - Number of bins (default: 10)
   * @returns The Figure containing the plot
   * @throws {InvalidParameterError} If a named column does not exist or bins is not a positive integer
   * @throws {DataValidationError} If a column holds non-numeric values
   */
  hist(
    options: PlotBaseOptions & {
      readonly column?: string | readonly string[];
      readonly bins?: number;
    } = {}
  ): Figure {
    const { fig, ax } = this.createFigAxes(options);
    const data = this.getData();

    const cols = this.resolveColumns(data, options.column, "column");

    for (const col of cols) {
      ax.hist(tensor(this.numeric(data, col, "column")), options.bins, { label: col });
    }

    if (cols.length > 1) ax.legend();
    return fig;
  }

  /**
   * Scatter plot.
   *
   * @param options - Plot configuration
   * @param options.x - Column for the x values
   * @param options.y - Column for the y values
   * @returns The Figure containing the plot
   * @throws {InvalidParameterError} If a named column does not exist
   * @throws {DataValidationError} If a column holds non-numeric values
   */
  scatter(
    options: PlotBaseOptions & {
      readonly x: string;
      readonly y: string;
    }
  ): Figure {
    const { fig, ax } = this.createFigAxes(options);
    const data = this.getData();

    const xValues = this.numeric(data, options.x, "x");
    const yValues = this.numeric(data, options.y, "y");
    ax.scatter(tensor(xValues), tensor(yValues));
    ax.setXLabel(options.x);
    ax.setYLabel(options.y);
    return fig;
  }

  /**
   * Box plot.
   *
   * @param options - Plot configuration
   * @param options.column - Column or columns to summarize (default: all numeric columns)
   * @returns The Figure containing the plot
   * @throws {InvalidParameterError} If a named column does not exist
   * @throws {DataValidationError} If a column holds non-numeric values
   */
  box(
    options: PlotBaseOptions & {
      readonly column?: string | readonly string[];
    } = {}
  ): Figure {
    const { fig, ax } = this.createFigAxes(options);
    const data = this.getData();

    const cols = this.resolveColumns(data, options.column, "column");
    for (const col of cols) {
      ax.boxplot(tensor(this.numeric(data, col, "column")), col ? { label: col } : {});
    }
    return fig;
  }

  /**
   * Pie chart from a single column.
   *
   * @param options - Plot configuration
   * @param options.y - Column of slice sizes (finite and non-negative)
   * @param options.labels - Column of slice labels (values are converted with `String`)
   * @returns The Figure containing the plot
   * @throws {InvalidParameterError} If a named column does not exist
   * @throws {DataValidationError} If the size column holds non-numeric values
   */
  pie(
    options: PlotBaseOptions & {
      readonly y: string;
      readonly labels?: string;
    }
  ): Figure {
    const { fig, ax } = this.createFigAxes(options);
    const data = this.getData();

    const values = this.numeric(data, options.y, "y");
    const labels = options.labels
      ? this.column(data, options.labels, "labels").map(String)
      : undefined;
    ax.pie(tensor(values), labels);
    return fig;
  }

  /**
   * Area plot (filled line plot).
   *
   * @param options - Plot configuration
   * @param options.x - Column for the x values (default: the row number)
   * @param options.y - Column or columns to draw (default: all numeric columns except `x`)
   * @returns The Figure containing the plot
   * @throws {InvalidParameterError} If a named column does not exist
   * @throws {DataValidationError} If a column holds non-numeric values
   */
  area(
    options: PlotBaseOptions & {
      readonly x?: string;
      readonly y?: string | readonly string[];
    } = {}
  ): Figure {
    const { fig, ax } = this.createFigAxes(options);
    const data = this.getData();

    const xCol = options.x;
    const xValues =
      xCol !== undefined
        ? this.numeric(data, xCol, "x")
        : Array.from({ length: data.nRows }, (_, i) => i);
    const yCols = this.resolveColumns(data, options.y, "y", xCol);

    const xT = tensor(xValues);
    for (const yCol of yCols) {
      ax.area(xT, tensor(this.numeric(data, yCol, "y")), { label: yCol });
    }

    if (yCols.length > 1) ax.legend();
    if (xCol) ax.setXLabel(xCol);
    return fig;
  }
}
