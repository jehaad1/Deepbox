/**
 * Plot themes: the default colors and font size of new figures and axes.
 *
 * The theme state lives here, in a module that both the figure classes and the
 * high-level helpers of `deepbox/plot` can import without a cycle.
 *
 * @module plot/figure/theme
 * @see {@link https://deepbox.dev/docs/plot-basic | Deepbox documentation}
 */

import { InvalidParameterError } from "../../core";

/**
 * A theme configuration for consistent plot styling.
 *
 * The active theme (see {@link setTheme}) is read when a figure or axes is
 * created: `Figure` takes its default background from `figureFacecolor`, and
 * `Axes` takes its facecolor, grid, text color and font sizes from the other
 * fields (tick labels use `fontSize - 2`, axis labels and the legend
 * `fontSize`, the title `fontSize + 2`). A figure that already exists keeps the
 * look it was created with, so call {@link setTheme} before `figure()`. Options
 * that are passed explicitly always win over the theme.
 *
 * The high-level helpers (the ML plots such as `plotRocCurve`, `kdeplot`,
 * `pairplot`, `jointplot` and `plotDecisionBoundary`) also take their default
 * series colors from `primaryColor` and `colorCycle`. The theme does not change
 * the per-chart default colors of `plot`, `scatter`, `bar`, `barh` and `hist`.
 */
export interface PlotTheme {
  /** Background color for figure. */
  readonly figureFacecolor: string;
  /** Background color for axes. */
  readonly axesFacecolor: string;
  /** Default line/bar color. */
  readonly primaryColor: string;
  /** Color cycle for multiple series. */
  readonly colorCycle: readonly string[];
  /** Default font size: the size of axis labels and legends, in pixels. */
  readonly fontSize: number;
  /** Grid color. */
  readonly gridColor: string;
  /** Whether grid is visible by default. */
  readonly gridVisible: boolean;
  /** Color of the axes frame, ticks, tick labels, titles, axis labels and legend text. */
  readonly textColor: string;
}

function freezeTheme(theme: PlotTheme): PlotTheme {
  return Object.freeze({ ...theme, colorCycle: Object.freeze([...theme.colorCycle]) });
}

const DEFAULT_THEME: PlotTheme = freezeTheme({
  figureFacecolor: "#ffffff",
  axesFacecolor: "#ffffff",
  primaryColor: "#1f77b4",
  colorCycle: [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
  ],
  fontSize: 12,
  gridColor: "#cccccc",
  gridVisible: false,
  textColor: "#000",
});

// A Map, so that names such as "constructor" or "__proto__" are not found through the
// prototype chain.
const themes: ReadonlyMap<string, PlotTheme> = new Map<string, PlotTheme>([
  ["default", DEFAULT_THEME],
  [
    "dark",
    freezeTheme({
      figureFacecolor: "#1e1e1e",
      axesFacecolor: "#2d2d2d",
      primaryColor: "#58a6ff",
      colorCycle: [
        "#58a6ff",
        "#f0883e",
        "#3fb950",
        "#f85149",
        "#bc8cff",
        "#db6d28",
        "#f778ba",
        "#8b949e",
        "#d2a825",
        "#39d0d0",
      ],
      fontSize: 12,
      gridColor: "#444444",
      gridVisible: true,
      textColor: "#e6e6e6",
    }),
  ],
  [
    "paper",
    freezeTheme({
      figureFacecolor: "#ffffff",
      axesFacecolor: "#ffffff",
      primaryColor: "#333333",
      colorCycle: [
        "#333333",
        "#666666",
        "#999999",
        "#bbbbbb",
        "#444444",
        "#777777",
        "#aaaaaa",
        "#555555",
        "#888888",
        "#cccccc",
      ],
      fontSize: 10,
      gridColor: "#e0e0e0",
      gridVisible: true,
      textColor: "#000",
    }),
  ],
  [
    "presentation",
    freezeTheme({
      figureFacecolor: "#ffffff",
      axesFacecolor: "#fafafa",
      primaryColor: "#2563eb",
      colorCycle: [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#ea580c",
        "#9333ea",
        "#0891b2",
        "#db2777",
        "#65a30d",
        "#ca8a04",
        "#4f46e5",
      ],
      fontSize: 16,
      gridColor: "#d4d4d4",
      gridVisible: true,
      textColor: "#111111",
    }),
  ],
]);

let _currentTheme: PlotTheme = DEFAULT_THEME;

/** Primary color of the active theme. @internal */
export function themePrimary(): string {
  return _currentTheme.primaryColor;
}

/** The i-th color of the active theme's color cycle (wrapping around). @internal */
export function themeCycle(i: number): string {
  const cycle = _currentTheme.colorCycle;
  return cycle[i % cycle.length] ?? _currentTheme.primaryColor;
}

/**
 * Set the global plot theme. Figures and axes created afterwards use it for their
 * default colors and font sizes; existing ones are not restyled.
 *
 * @param name - Theme name: "default", "dark", "paper", or "presentation"
 * @throws {InvalidParameterError} If the theme name is unknown.
 */
export function setTheme(name: string): void {
  const theme = themes.get(name);
  if (!theme) {
    throw new InvalidParameterError(
      `Unknown theme "${name}". Available: ${Array.from(themes.keys()).join(", ")}`,
      "name",
      name
    );
  }
  _currentTheme = theme;
}

/**
 * Get the current plot theme. The returned object is frozen.
 */
export function getTheme(): PlotTheme {
  return _currentTheme;
}

/**
 * Reset the theme to default.
 */
export function resetTheme(): void {
  _currentTheme = DEFAULT_THEME;
}

/**
 * List available theme names.
 */
export function listThemes(): readonly string[] {
  return Array.from(themes.keys());
}
