import { beforeEach, describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import type { PlotTheme } from "../src/plot";
import {
  figure,
  getTheme,
  kdeplot,
  listThemes,
  pairplot,
  plotCalibrationCurve,
  plotElbowCurve,
  plotFeatureImportance,
  plotResiduals,
  plotSilhouette,
  resetTheme,
  setTheme,
} from "../src/plot";

beforeEach(() => {
  // Create a fresh figure for each test to avoid state leakage
  figure();
  resetTheme();
});

describe("kdeplot", () => {
  it("plots KDE for simple data", () => {
    figure();
    expect(() => kdeplot(tensor([1, 2, 3, 4, 5]))).not.toThrow();
  });

  it("plots KDE with silverman bandwidth", () => {
    figure();
    expect(() => kdeplot(tensor([1, 2, 3, 4, 5]), { bw_method: "silverman" })).not.toThrow();
  });

  it("plots KDE with explicit bandwidth", () => {
    figure();
    expect(() => kdeplot(tensor([1, 2, 3, 4, 5]), { bw_method: 0.5 })).not.toThrow();
  });

  it("plots filled KDE", () => {
    figure();
    expect(() => kdeplot(tensor([1, 2, 3, 4, 5]), { fill: true })).not.toThrow();
  });

  it("throws on empty data", () => {
    figure();
    expect(() => kdeplot(tensor([]))).toThrow(/at least one/);
  });

  it("accepts custom gridSize", () => {
    figure();
    expect(() => kdeplot(tensor([1, 2, 3, 4, 5]), { gridSize: 50 })).not.toThrow();
  });
});

describe("plotResiduals", () => {
  it("plots residuals for simple regression", () => {
    figure();
    const yTrue = tensor([1, 2, 3, 4, 5]);
    const yPred = tensor([1.1, 2.2, 2.8, 4.1, 4.9]);
    expect(() => plotResiduals(yTrue, yPred)).not.toThrow();
  });

  it("throws on empty data", () => {
    figure();
    expect(() => plotResiduals(tensor([]), tensor([]))).toThrow(/at least one/);
  });

  it("throws on mismatched lengths", () => {
    figure();
    expect(() => plotResiduals(tensor([1, 2, 3]), tensor([1, 2]))).toThrow(/same length/);
  });
});

describe("plotFeatureImportance", () => {
  it("plots feature importances", () => {
    figure();
    const importances = tensor([0.3, 0.5, 0.1, 0.1]);
    expect(() => plotFeatureImportance(importances, ["A", "B", "C", "D"])).not.toThrow();
  });

  it("uses default feature names when not provided", () => {
    figure();
    expect(() => plotFeatureImportance(tensor([0.5, 0.3, 0.2]))).not.toThrow();
  });

  it("throws on empty importances", () => {
    figure();
    expect(() => plotFeatureImportance(tensor([]))).toThrow(/at least one/);
  });
});

describe("plotElbowCurve", () => {
  it("plots elbow curve", () => {
    figure();
    const k = tensor([1, 2, 3, 4, 5]);
    const inertias = tensor([100, 50, 25, 15, 10]);
    expect(() => plotElbowCurve(k, inertias)).not.toThrow();
  });
});

describe("plotSilhouette", () => {
  it("plots silhouette scores", () => {
    figure();
    const k = tensor([2, 3, 4, 5]);
    const scores = tensor([0.5, 0.6, 0.55, 0.45]);
    expect(() => plotSilhouette(k, scores)).not.toThrow();
  });
});

describe("plotCalibrationCurve", () => {
  it("plots calibration curve", () => {
    figure();
    const fractionPos = tensor([0, 0.2, 0.5, 0.8, 1.0]);
    const meanPred = tensor([0, 0.25, 0.5, 0.75, 1.0]);
    expect(() => plotCalibrationCurve(fractionPos, meanPred)).not.toThrow();
  });
});

describe("pairplot", () => {
  it("creates pair plot for 2D data", () => {
    const data = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const fig = pairplot(data);
    expect(fig).toBeDefined();
    expect(fig.axesList.length).toBe(4); // 2x2 grid
  });

  it("creates pair plot for 3 features", () => {
    const data = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const fig = pairplot(data, {
      featureNames: ["X", "Y", "Z"],
    });
    expect(fig).toBeDefined();
    expect(fig.axesList.length).toBe(9); // 3x3 grid
  });

  it("throws on 1D data", () => {
    expect(() => pairplot(tensor([1, 2, 3]))).toThrow(/2D/);
  });

  it("accepts custom color and size", () => {
    const data = tensor([
      [1, 2],
      [3, 4],
    ]);
    const fig = pairplot(data, { color: "#ff0000", size: 5 });
    expect(fig.axesList.length).toBe(4);
  });
});

describe("Themes", () => {
  it("lists available themes", () => {
    const names = listThemes();
    expect(names).toContain("default");
    expect(names).toContain("dark");
    expect(names).toContain("paper");
    expect(names).toContain("presentation");
  });

  it("getTheme returns default theme initially", () => {
    const theme = getTheme();
    expect(theme.primaryColor).toBe("#1f77b4");
    expect(theme.figureFacecolor).toBe("#ffffff");
  });

  it("setTheme changes the current theme", () => {
    setTheme("dark");
    const theme = getTheme();
    expect(theme.primaryColor).toBe("#58a6ff");
    expect(theme.figureFacecolor).toBe("#1e1e1e");
  });

  it("setTheme to paper", () => {
    setTheme("paper");
    const theme = getTheme();
    expect(theme.primaryColor).toBe("#333333");
    expect(theme.fontSize).toBe(10);
  });

  it("setTheme to presentation", () => {
    setTheme("presentation");
    const theme = getTheme();
    expect(theme.primaryColor).toBe("#2563eb");
    expect(theme.fontSize).toBe(16);
  });

  it("resetTheme restores default", () => {
    setTheme("dark");
    resetTheme();
    const theme = getTheme();
    expect(theme.primaryColor).toBe("#1f77b4");
  });

  it("setTheme throws on unknown theme", () => {
    expect(() => setTheme("nonexistent")).toThrow(/Unknown theme/);
  });

  it("theme has expected structure", () => {
    const theme: PlotTheme = getTheme();
    expect(typeof theme.figureFacecolor).toBe("string");
    expect(typeof theme.axesFacecolor).toBe("string");
    expect(typeof theme.primaryColor).toBe("string");
    expect(Array.isArray(theme.colorCycle)).toBe(true);
    expect(theme.colorCycle.length).toBeGreaterThan(0);
    expect(typeof theme.fontSize).toBe("number");
    expect(typeof theme.gridColor).toBe("string");
    expect(typeof theme.gridVisible).toBe("boolean");
  });
});
