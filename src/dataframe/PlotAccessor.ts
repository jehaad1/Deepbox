/**
 * DataFrame plotting accessor.
 *
 * Provides a pandas-like `df.plot.line()`, `df.plot.bar()`, etc. API
 * that delegates to the main `deepbox/plot` module.
 *
 * @module dataframe/PlotAccessor
 * @see {@link https://deepbox.dev/docs/dataframe-overview | Deepbox documentation}
 */

import { tensor } from "../ndarray";
import type { Axes } from "../plot/figure/Axes";
import { Figure } from "../plot/figure/Figure";

/**
 * Plotting accessor for DataFrame.
 *
 * Access via `df.plot.line()`, `df.plot.bar()`, etc.
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
  private readonly getData: () => {
    readonly columns: string[];
    readonly getColumn: (name: string) => unknown[];
    readonly nRows: number;
  };

  /** @internal */
  constructor(
    getData: () => {
      readonly columns: string[];
      readonly getColumn: (name: string) => unknown[];
      readonly nRows: number;
    }
  ) {
    this.getData = getData;
  }

  private createFigAxes(options?: {
    readonly figsize?: readonly [number, number];
    readonly title?: string;
  }): { fig: Figure; ax: Axes } {
    const w = options?.figsize?.[0] ?? 640;
    const h = options?.figsize?.[1] ?? 480;
    const fig = new Figure({ width: w, height: h });
    const ax = fig.addAxes();
    if (options?.title) ax.setTitle(options.title);
    return { fig, ax };
  }

  private toNumbers(col: unknown[]): number[] {
    return col.map((v) => (typeof v === "number" ? v : Number(v)));
  }

  /**
   * Line plot.
   *
   * @param options - Plot configuration
   * @returns The Figure containing the plot
   */
  line(
    options: {
      readonly x?: string;
      readonly y?: string | string[];
      readonly figsize?: readonly [number, number];
      readonly title?: string;
    } = {}
  ): Figure {
    const { fig, ax } = this.createFigAxes(options);
    const data = this.getData();

    const xCol = options.x;
    const xValues = xCol
      ? this.toNumbers(data.getColumn(xCol))
      : Array.from({ length: data.nRows }, (_, i) => i);

    const yCols =
      options.y === undefined
        ? data.columns.filter((c) => c !== xCol)
        : typeof options.y === "string"
          ? [options.y]
          : options.y;

    for (const yCol of yCols) {
      const yValues = this.toNumbers(data.getColumn(yCol));
      ax.plot(tensor(xValues), tensor(yValues), { label: yCol });
    }

    if (yCols.length > 1) ax.legend();
    if (xCol) ax.setXLabel(xCol);
    return fig;
  }

  /**
   * Bar plot.
   */
  bar(
    options: {
      readonly x?: string;
      readonly y?: string | string[];
      readonly figsize?: readonly [number, number];
      readonly title?: string;
    } = {}
  ): Figure {
    const { fig, ax } = this.createFigAxes(options);
    const data = this.getData();

    const xCol = options.x ?? data.columns[0];
    if (!xCol) return fig;
    const xValues = this.toNumbers(data.getColumn(xCol));

    const yCols =
      options.y === undefined
        ? data.columns.filter((c) => c !== xCol)
        : typeof options.y === "string"
          ? [options.y]
          : options.y;

    for (const yCol of yCols) {
      const yValues = this.toNumbers(data.getColumn(yCol));
      ax.bar(tensor(xValues), tensor(yValues), { label: yCol });
    }

    if (yCols.length > 1) ax.legend();
    if (xCol) ax.setXLabel(xCol);
    return fig;
  }

  /**
   * Horizontal bar plot.
   */
  barh(
    options: {
      readonly x?: string;
      readonly y?: string;
      readonly figsize?: readonly [number, number];
      readonly title?: string;
    } = {}
  ): Figure {
    const { fig, ax } = this.createFigAxes(options);
    const data = this.getData();

    const yCol = options.y ?? data.columns[0];
    const xCol = options.x ?? data.columns[1];
    if (!yCol || !xCol) return fig;

    const categories = this.toNumbers(data.getColumn(yCol));
    const values = this.toNumbers(data.getColumn(xCol));
    ax.barh(tensor(categories), tensor(values));
    return fig;
  }

  /**
   * Histogram.
   */
  hist(
    options: {
      readonly column?: string | string[];
      readonly bins?: number;
      readonly figsize?: readonly [number, number];
      readonly title?: string;
    } = {}
  ): Figure {
    const { fig, ax } = this.createFigAxes(options);
    const data = this.getData();

    const cols =
      options.column === undefined
        ? data.columns
        : typeof options.column === "string"
          ? [options.column]
          : options.column;

    for (const col of cols) {
      const values = this.toNumbers(data.getColumn(col));
      ax.hist(tensor(values), options.bins, { label: col });
    }

    if (cols.length > 1) ax.legend();
    return fig;
  }

  /**
   * Scatter plot.
   */
  scatter(options: {
    readonly x: string;
    readonly y: string;
    readonly figsize?: readonly [number, number];
    readonly title?: string;
  }): Figure {
    const { fig, ax } = this.createFigAxes(options);
    const data = this.getData();

    const xValues = this.toNumbers(data.getColumn(options.x));
    const yValues = this.toNumbers(data.getColumn(options.y));
    ax.scatter(tensor(xValues), tensor(yValues));
    ax.setXLabel(options.x);
    ax.setYLabel(options.y);
    return fig;
  }

  /**
   * Box plot.
   */
  box(
    options: {
      readonly column?: string | string[];
      readonly figsize?: readonly [number, number];
      readonly title?: string;
    } = {}
  ): Figure {
    const { fig, ax } = this.createFigAxes(options);
    const data = this.getData();

    const cols =
      options.column === undefined
        ? data.columns
        : typeof options.column === "string"
          ? [options.column]
          : options.column;

    const boxData: number[][] = [];
    for (const col of cols) {
      boxData.push(this.toNumbers(data.getColumn(col)));
    }
    for (let i = 0; i < boxData.length; i++) {
      const lbl = cols[i];
      ax.boxplot(tensor(boxData[i]!), lbl ? { label: lbl } : {});
    }
    return fig;
  }

  /**
   * Pie chart from a single column.
   */
  pie(options: {
    readonly y: string;
    readonly labels?: string;
    readonly figsize?: readonly [number, number];
    readonly title?: string;
  }): Figure {
    const { fig, ax } = this.createFigAxes(options);
    const data = this.getData();

    const values = this.toNumbers(data.getColumn(options.y));
    const labels = options.labels ? data.getColumn(options.labels).map(String) : undefined;
    ax.pie(tensor(values), labels);
    return fig;
  }

  /**
   * Area plot (filled line plot).
   */
  area(
    options: {
      readonly x?: string;
      readonly y?: string | string[];
      readonly figsize?: readonly [number, number];
      readonly title?: string;
    } = {}
  ): Figure {
    const { fig, ax } = this.createFigAxes(options);
    const data = this.getData();

    const xCol = options.x;
    const xValues = xCol
      ? this.toNumbers(data.getColumn(xCol))
      : Array.from({ length: data.nRows }, (_, i) => i);

    const yCols =
      options.y === undefined
        ? data.columns.filter((c) => c !== xCol)
        : typeof options.y === "string"
          ? [options.y]
          : options.y;

    for (const yCol of yCols) {
      const yValues = this.toNumbers(data.getColumn(yCol));
      ax.area(tensor(xValues), tensor(yValues), { label: yCol });
    }

    if (yCols.length > 1) ax.legend();
    return fig;
  }
}
