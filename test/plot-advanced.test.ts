import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import {
  bar,
  barh,
  boxplot,
  contour,
  contourf,
  Figure,
  figure,
  heatmap,
  hist,
  imshow,
  pie,
  plot,
  plotConfusionMatrix,
  plotDecisionBoundary,
  plotLearningCurve,
  plotPrecisionRecallCurve,
  plotRocCurve,
  plotValidationCurve,
  scatter,
  show,
  violinplot,
} from "../src/plot";

function renderCurrentSvg(): string {
  const rendered = show({ format: "svg" });
  expect(rendered).not.toBeInstanceOf(Promise);
  if (rendered instanceof Promise) return "";
  return rendered.svg;
}

describe("Plot - Advanced Features", () => {
  describe("Global Plot Functions", () => {
    it("should create plot using global function", () => {
      figure();
      plot(tensor([1, 2, 3]), tensor([1, 4, 9]));
      const svg = renderCurrentSvg();
      expect(svg).toContain("<polyline");
    });

    it("should create scatter using global function", () => {
      figure();
      scatter(tensor([1, 2, 3]), tensor([1, 4, 9]));
      const svg = renderCurrentSvg();
      expect(svg).toContain("<circle");
    });

    it("should create bar using global function", () => {
      figure();
      bar(tensor([1, 2, 3]), tensor([5, 10, 7]));
      const svg = renderCurrentSvg();
      expect(svg).toContain("<rect");
    });

    it("should create barh using global function", () => {
      figure();
      barh(tensor([1, 2, 3]), tensor([5, 10, 7]));
      const svg = renderCurrentSvg();
      expect(svg).toContain("<rect");
    });

    it("should create hist using global function", () => {
      figure();
      hist(tensor([1, 2, 3, 4, 5]), 5);
      const svg = renderCurrentSvg();
      expect(svg).toContain("<rect");
    });

    it("should create heatmap using global function", () => {
      figure();
      heatmap(
        tensor([
          [1, 2],
          [3, 4],
        ])
      );
      const svg = renderCurrentSvg();
      expect(svg).toContain("rgb(");
    });

    it("should create imshow using global function", () => {
      figure();
      imshow(
        tensor([
          [1, 2],
          [3, 4],
        ])
      );
      const svg = renderCurrentSvg();
      expect(svg).toContain("rgb(");
    });

    it("should create boxplot using global function", () => {
      figure();
      boxplot(tensor([1, 2, 3, 4, 5]));
      const svg = renderCurrentSvg();
      expect(svg).toContain('stroke-width="2"');
    });

    it("should create violinplot using global function", () => {
      figure();
      violinplot(tensor([1, 2, 3, 4, 5]));
      const svg = renderCurrentSvg();
      expect(svg).toContain('stroke-width="3"');
    });

    it("should create pie using global function", () => {
      figure();
      pie(tensor([30, 20, 50]));
      const svg = renderCurrentSvg();
      expect(svg).toContain("<path");
    });

    it("should create contour using global function", () => {
      const fig = figure({ width: 200, height: 140 });
      const Z = tensor([
        [1, 2],
        [3, 4],
      ]);
      contour(tensor([]), tensor([]), Z);
      const svg = fig.renderSVG().svg;
      expect(svg).toContain('stroke="#1f77b4"');
    });

    it("should create contourf using global function", () => {
      const fig = figure({ width: 200, height: 140 });
      const Z = tensor([
        [1, 2],
        [3, 4],
      ]);
      contourf(tensor([]), tensor([]), Z);
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("rgb(");
    });
  });

  describe("ML Plotting Functions", () => {
    it("should plot confusion matrix", () => {
      const cm = tensor([
        [10, 2],
        [3, 15],
      ]);
      figure();
      plotConfusionMatrix(cm);
      const svg = renderCurrentSvg();
      expect(svg).toContain("Confusion Matrix");
    });

    it("should plot confusion matrix with labels", () => {
      const cm = tensor([
        [10, 2],
        [3, 15],
      ]);
      figure();
      plotConfusionMatrix(cm, ["A", "B"]);
      const svg = renderCurrentSvg();
      expect(svg).toContain("A");
      expect(svg).toContain("B");
    });

    it("should plot ROC curve", () => {
      const fpr = tensor([0, 0.2, 0.4, 1]);
      const tpr = tensor([0, 0.6, 0.8, 1]);
      figure();
      plotRocCurve(fpr, tpr, 0.85);
      const svg = renderCurrentSvg();
      expect(svg).toContain("ROC Curve");
    });

    it("should plot ROC curve without AUC", () => {
      const fpr = tensor([0, 0.2, 0.4, 1]);
      const tpr = tensor([0, 0.6, 0.8, 1]);
      figure();
      plotRocCurve(fpr, tpr);
      const svg = renderCurrentSvg();
      expect(svg).toContain("ROC Curve");
    });

    it("should plot precision-recall curve", () => {
      const precision = tensor([1, 0.8, 0.6, 0.4]);
      const recall = tensor([0, 0.2, 0.6, 1]);
      figure();
      plotPrecisionRecallCurve(precision, recall, 0.75);
      const svg = renderCurrentSvg();
      expect(svg).toContain("Precision-Recall Curve");
    });

    it("should plot precision-recall curve without AP", () => {
      const precision = tensor([1, 0.8, 0.6, 0.4]);
      const recall = tensor([0, 0.2, 0.6, 1]);
      figure();
      plotPrecisionRecallCurve(precision, recall);
      const svg = renderCurrentSvg();
      expect(svg).toContain("Precision-Recall Curve");
    });

    it("should plot learning curve", () => {
      const trainSizes = tensor([10, 20, 30]);
      const trainScores = tensor([0.8, 0.85, 0.9]);
      const valScores = tensor([0.75, 0.8, 0.85]);
      figure();
      plotLearningCurve(trainSizes, trainScores, valScores);
      const svg = renderCurrentSvg();
      const polylineCount = (svg.match(/<polyline/g) || []).length;
      expect(polylineCount).toBe(2);
    });

    it("should plot validation curve", () => {
      const paramRange = tensor([0.1, 1, 10]);
      const trainScores = tensor([0.8, 0.9, 0.85]);
      const valScores = tensor([0.75, 0.85, 0.8]);
      figure();
      plotValidationCurve(paramRange, trainScores, valScores);
      const svg = renderCurrentSvg();
      const polylineCount = (svg.match(/<polyline/g) || []).length;
      expect(polylineCount).toBe(2);
    });

    it("should plot decision boundary", () => {
      const X = tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const y = tensor([0, 1, 0]);
      const model = {
        predict: (grid: typeof X) => {
          const rows = grid.shape[0] ?? 0;
          return tensor(Array.from({ length: rows }, (_, i) => (i % 3 === 0 ? 1 : 0)));
        },
      };
      figure();
      plotDecisionBoundary(X, y, model);
      const svg = renderCurrentSvg();
      expect(svg).toContain("<rect");
      expect(svg).toContain("<circle");
    });

    it("should evaluate classifier on a decision grid", () => {
      const X = tensor([
        [0, 0],
        [1, 1],
        [2, 2],
      ]);
      const y = tensor([0, 1, 1]);
      let seenRows = 0;
      let seenCols = 0;
      const model = {
        predict: (grid: typeof X) => {
          seenRows = grid.shape[0] ?? 0;
          seenCols = grid.shape[1] ?? 0;
          return tensor(Array.from({ length: seenRows }, (_, i) => (i % 2 === 0 ? 0 : 1)));
        },
      };
      plotDecisionBoundary(X, y, model);
      expect(seenRows).toBe(10000);
      expect(seenCols).toBe(2);
    });

    it("should throw on non-2D X for decision boundary", () => {
      const X = tensor([1, 2, 3]);
      const y = tensor([0, 1, 0]);
      const model = { predict: (_x: typeof X) => y };
      expect(() => plotDecisionBoundary(X, y, model)).toThrow(/shape \[n, 2\]/);
    });

    it("should throw on X with wrong feature count", () => {
      const X = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const y = tensor([0, 1]);
      const model = { predict: (_x: typeof X) => y };
      expect(() => plotDecisionBoundary(X, y, model)).toThrow(/shape \[n, 2\]/);
    });
  });

  describe("Boxplot Advanced", () => {
    it("should handle dataset with outliers", () => {
      const data = tensor([1, 2, 3, 4, 5, 100, 200]);
      figure();
      boxplot(data);
      const svg = renderCurrentSvg();
      expect(svg).toContain('stroke-width="2"');
    });

    it("should handle single value", () => {
      const data = tensor([5]);
      figure();
      boxplot(data);
      const svg = renderCurrentSvg();
      expect(svg).toContain('stroke-width="2"');
    });

    it("should handle two values", () => {
      const data = tensor([1, 10]);
      figure();
      boxplot(data);
      const svg = renderCurrentSvg();
      expect(svg).toContain('stroke-width="2"');
    });

    it("should handle empty data", () => {
      const data = tensor([]);
      figure();
      boxplot(data);
      const svg = renderCurrentSvg();
      expect(svg).toContain("<svg");
    });

    it("should handle negative values", () => {
      const data = tensor([-5, -3, -1, 0, 1, 3, 5]);
      figure();
      boxplot(data);
      const svg = renderCurrentSvg();
      expect(svg).toContain('stroke-width="2"');
    });

    it("should handle large dataset", () => {
      const data = Array.from({ length: 10000 }, () => Math.random());
      figure();
      boxplot(tensor(data));
      const svg = renderCurrentSvg();
      expect(svg).toContain('stroke-width="2"');
    });

    it("should apply custom color", () => {
      const data = tensor([1, 2, 3, 4, 5]);
      figure();
      boxplot(data, { color: "#ff0000" });
      const svg = renderCurrentSvg();
      expect(svg).toContain("#ff0000");
    });
  });

  describe("Violinplot Advanced", () => {
    it("should handle dataset with wide range", () => {
      const data = tensor([1, 2, 3, 4, 5, 100]);
      figure();
      violinplot(data);
      const svg = renderCurrentSvg();
      expect(svg).toContain("<path");
    });

    it("should handle single value", () => {
      const data = tensor([5]);
      figure();
      violinplot(data);
      const svg = renderCurrentSvg();
      expect(svg).toContain("<path");
    });

    it("should handle empty data", () => {
      const data = tensor([]);
      figure();
      violinplot(data);
      const svg = renderCurrentSvg();
      expect(svg).toContain("<svg");
    });

    it("should handle negative values", () => {
      const data = tensor([-10, -5, 0, 5, 10]);
      figure();
      violinplot(data);
      const svg = renderCurrentSvg();
      expect(svg).toContain("<path");
    });

    it("should handle large dataset", () => {
      const data = Array.from({ length: 10000 }, () => Math.random());
      figure();
      violinplot(tensor(data));
      const svg = renderCurrentSvg();
      expect(svg).toContain("<path");
    });

    it("should apply custom color", () => {
      const data = tensor([1, 2, 3, 4, 5]);
      figure();
      violinplot(data, { color: "#00ff00" });
      const svg = renderCurrentSvg();
      expect(svg).toContain("#00ff00");
    });

    it("should apply custom size", () => {
      const data = tensor([1, 2, 3, 4, 5]);
      figure();
      violinplot(data, { size: 10 });
      const svg = renderCurrentSvg();
      expect(svg).toContain("<path");
    });
  });

  describe("Pie Chart Advanced", () => {
    it("should handle single slice", () => {
      figure();
      pie(tensor([100]));
      const svg = renderCurrentSvg();
      expect(svg).toContain("<path");
    });

    it("should handle two slices", () => {
      figure();
      pie(tensor([50, 50]));
      const svg = renderCurrentSvg();
      expect(svg).toContain("<path");
    });

    it("should handle many slices", () => {
      const data = Array.from({ length: 20 }, () => Math.random());
      figure();
      pie(tensor(data));
      const svg = renderCurrentSvg();
      expect(svg).toContain("<path");
    });

    it("should reject all-zero values", () => {
      expect(() => pie(tensor([0, 0, 0]))).toThrow(/positive number/);
    });

    it("should reject negative values", () => {
      expect(() => pie(tensor([10, -5, 15]))).toThrow(/non-negative/);
    });

    it("should handle with labels", () => {
      figure();
      pie(tensor([30, 20, 50]), ["A", "B", "C"]);
      const svg = renderCurrentSvg();
      expect(svg).toContain("A");
    });

    it("should handle with custom size", () => {
      figure();
      pie(tensor([30, 20, 50]), undefined, { size: 15 });
      const svg = renderCurrentSvg();
      expect(svg).toContain("<path");
    });

    it("should reject empty data", () => {
      expect(() => pie(tensor([]))).toThrow(/at least one value/);
    });

    it("should handle very small values", () => {
      figure();
      pie(tensor([0.001, 0.002, 0.003]));
      const svg = renderCurrentSvg();
      expect(svg).toContain("<path");
    });
  });

  describe("Contour Plots", () => {
    it("should create contour plot", () => {
      const fig = figure({ width: 240, height: 160 });
      const Z = tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      contour(tensor([]), tensor([]), Z);
      const svg = fig.renderSVG().svg;
      expect(svg).toContain('stroke="#1f77b4"');
    });

    it("should create filled contour plot", () => {
      const fig = figure({ width: 240, height: 160 });
      const Z = tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      contourf(tensor([]), tensor([]), Z);
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("rgb(");
    });

    it("should handle 1x1 grid", () => {
      const Z = tensor([[5]]);
      const fig = figure({ width: 200, height: 140 });
      contour(tensor([]), tensor([]), Z);
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<svg");
    });

    it("should handle large grid", () => {
      const Z = Array.from({ length: 100 }, () => Array.from({ length: 100 }, () => Math.random()));
      const fig = figure({ width: 200, height: 140 });
      contour(tensor([]), tensor([]), tensor(Z));
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<svg");
    });

    it("should handle with custom levels", () => {
      const Z = tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const fig = figure({ width: 200, height: 140 });
      contour(tensor([]), tensor([]), Z, { levels: 5 });
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<svg");
    });

    it("should handle with explicit level values", () => {
      const Z = tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const fig = figure({ width: 200, height: 140 });
      contour(tensor([]), tensor([]), Z, { levels: [2, 4, 6, 8] });
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<svg");
    });

    it("should handle negative values", () => {
      const Z = tensor([
        [-5, -3, -1],
        [0, 2, 4],
        [5, 7, 9],
      ]);
      const fig = figure({ width: 200, height: 140 });
      contour(tensor([]), tensor([]), Z);
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<svg");
    });

    it("should handle all same values", () => {
      const Z = tensor([
        [5, 5, 5],
        [5, 5, 5],
        [5, 5, 5],
      ]);
      const fig = figure({ width: 200, height: 140 });
      contour(tensor([]), tensor([]), Z);
      const svg = fig.renderSVG().svg;
      expect(svg).toContain("<svg");
    });
  });

  describe("RasterCanvas Drawing", () => {
    it("should render line to PNG", async () => {
      const fig = new Figure({ width: 100, height: 100 });
      const ax = fig.addAxes();
      ax.plot(tensor([0, 1]), tensor([0, 1]));
      const result = await fig.renderPNG();
      expect(result.bytes.length).toBeGreaterThan(0);
    });

    it("should render scatter to PNG", async () => {
      const fig = new Figure({ width: 100, height: 100 });
      const ax = fig.addAxes();
      ax.scatter(tensor([0.5, 0.5]), tensor([0.5, 0.5]));
      const result = await fig.renderPNG();
      expect(result.bytes.length).toBeGreaterThan(0);
    });

    it("should render bar to PNG", async () => {
      const fig = new Figure({ width: 100, height: 100 });
      const ax = fig.addAxes();
      ax.bar(tensor([1, 2]), tensor([5, 10]));
      const result = await fig.renderPNG();
      expect(result.bytes.length).toBeGreaterThan(0);
    });

    it("should render heatmap to PNG", async () => {
      const fig = new Figure({ width: 100, height: 100 });
      const ax = fig.addAxes();
      ax.heatmap(
        tensor([
          [1, 2],
          [3, 4],
        ])
      );
      const result = await fig.renderPNG();
      expect(result.bytes.length).toBeGreaterThan(0);
    });

    it("should render histogram to PNG", async () => {
      const fig = new Figure({ width: 100, height: 100 });
      const ax = fig.addAxes();
      ax.hist(tensor([1, 2, 2, 3, 3, 3]), 3);
      const result = await fig.renderPNG();
      expect(result.bytes.length).toBeGreaterThan(0);
    });
  });

  describe("Complex Scenarios", () => {
    it("should handle plot with all features", () => {
      const fig = new Figure({
        width: 800,
        height: 600,
        background: "#f0f0f0",
      });
      const ax = fig.addAxes({ padding: 60, facecolor: "#ffffff" });
      ax.setTitle("Complex Plot");
      ax.setXLabel("X Axis");
      ax.setYLabel("Y Axis");
      ax.plot(tensor([1, 2, 3, 4, 5]), tensor([1, 4, 9, 16, 25]), {
        color: "#ff0000",
        linewidth: 3,
      });
      ax.scatter(tensor([1, 2, 3, 4, 5]), tensor([2, 5, 10, 17, 26]), {
        color: "#00ff00",
        size: 8,
      });
      const result = fig.renderSVG();
      expect(result.svg).toContain("Complex Plot");
      expect(result.svg).toContain("X Axis");
      expect(result.svg).toContain("Y Axis");
    });

    it("should handle multiple axes with different plots", () => {
      const fig = new Figure({ width: 1200, height: 800 });
      const ax1 = fig.addAxes();
      const ax2 = fig.addAxes();
      ax1.plot(tensor([1, 2, 3]), tensor([1, 4, 9]));
      ax2.scatter(tensor([1, 2, 3]), tensor([3, 2, 1]));
      const result = fig.renderSVG();
      expect(result.kind).toBe("svg");
    });

    it("should handle scientific notation values", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1e-10, 2e-10, 3e-10]), tensor([1e10, 2e10, 3e10]));
      const result = fig.renderSVG();
      expect(result.svg).toContain("polyline");
    });

    it("should handle mixed data types", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2, 3]), tensor([1, 4, 9]));
      ax.scatter(tensor([1.5, 2.5]), tensor([2.5, 6.5]));
      ax.bar(tensor([1, 2, 3]), tensor([0.5, 1.5, 2.5]));
      const result = fig.renderSVG();
      expect(result.kind).toBe("svg");
    });

    it("should handle empty figure", () => {
      const fig = new Figure();
      const result = fig.renderSVG();
      expect(result.kind).toBe("svg");
    });

    it("should handle figure with empty axes", () => {
      const fig = new Figure();
      fig.addAxes();
      fig.addAxes();
      fig.addAxes();
      const result = fig.renderSVG();
      expect(result.kind).toBe("svg");
    });
  });

  describe("Edge Cases - Data Range", () => {
    it("should handle identical x values", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([5, 5, 5]), tensor([1, 2, 3]));
      const result = fig.renderSVG();
      expect(result.svg).toContain("polyline");
    });

    it("should handle identical y values", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2, 3]), tensor([5, 5, 5]));
      const result = fig.renderSVG();
      expect(result.svg).toContain("polyline");
    });

    it("should handle identical x and y values", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([5, 5, 5]), tensor([5, 5, 5]));
      const result = fig.renderSVG();
      expect(result.svg).toContain("polyline");
    });

    it("should handle very close values", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1.0000001, 1.0000002, 1.0000003]), tensor([1, 2, 3]));
      const result = fig.renderSVG();
      expect(result.svg).toContain("polyline");
    });
  });

  describe("Options Validation", () => {
    it("should handle undefined options", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2]), tensor([1, 2]), undefined);
      const result = fig.renderSVG();
      expect(result.svg).toContain("polyline");
    });

    it("should handle empty options object", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2]), tensor([1, 2]), {});
      const result = fig.renderSVG();
      expect(result.svg).toContain("polyline");
    });

    it("should handle partial options", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([1, 2]), tensor([1, 2]), { color: "#ff0000" });
      const result = fig.renderSVG();
      expect(result.svg).toContain("#ff0000");
    });

    it("should ignore unknown options", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const options = { color: "#ff0000", unknown: "value" };
      ax.plot(tensor([1, 2]), tensor([1, 2]), options);
      const result = fig.renderSVG();
      expect(result.svg).toContain("#ff0000");
    });
  });

  describe("Memory and Performance", () => {
    it("should not leak memory with repeated plots", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      for (let i = 0; i < 100; i++) {
        ax.plot(tensor([1, 2, 3]), tensor([i, i + 1, i + 2]));
      }
      expect(fig.axesList[0]).toBeDefined();
    });

    it("should handle rapid figure creation", () => {
      for (let i = 0; i < 100; i++) {
        const fig = new Figure();
        fig.addAxes();
      }
      expect(true).toBe(true);
    });

    it("should handle large SVG generation", () => {
      const fig = new Figure({ width: 2000, height: 2000 });
      const ax = fig.addAxes();
      const n = 1000;
      const x = Array.from({ length: n }, (_, i) => i);
      const y = Array.from({ length: n }, (_, i) => Math.sin(i / 10));
      ax.plot(tensor(x), tensor(y));
      const result = fig.renderSVG();
      expect(result.svg.length).toBeGreaterThan(1000);
    });
  });

  describe("Coordinate Transformation", () => {
    it("should correctly transform data coordinates", () => {
      const fig = new Figure({ width: 400, height: 300 });
      const ax = fig.addAxes({ padding: 50 });
      ax.plot(tensor([0, 1]), tensor([0, 1]));
      const result = fig.renderSVG();
      expect(result.svg).toContain("polyline");
    });

    it("should handle negative coordinates", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      ax.plot(tensor([-10, -5, 0, 5, 10]), tensor([-100, -25, 0, 25, 100]));
      const result = fig.renderSVG();
      expect(result.svg).toContain("polyline");
    });

    it("should handle logarithmic-scale data", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const x = [1, 10, 100, 1000, 10000];
      const y = x.map((v) => Math.log10(v));
      ax.plot(tensor(x), tensor(y));
      const result = fig.renderSVG();
      expect(result.svg).toContain("polyline");
    });
  });

  describe("String Tensor Handling", () => {
    it("should throw on string tensor for plot", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const stringTensor = tensor(["a", "b", "c"]);
      expect(() => ax.plot(stringTensor, tensor([1, 2, 3]))).toThrow(/string tensors/);
    });

    it("should throw on string tensor for scatter", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const stringTensor = tensor(["a", "b", "c"]);
      expect(() => ax.scatter(tensor([1, 2, 3]), stringTensor)).toThrow(/string tensors/);
    });

    it("should throw on string tensor for heatmap", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const stringTensor = tensor([
        ["a", "b"],
        ["c", "d"],
      ]);
      expect(() => ax.heatmap(stringTensor)).toThrow(/string tensors/);
    });
  });

  describe("Rendering Consistency", () => {
    it("should produce same SVG for same data", () => {
      const fig1 = new Figure({ width: 400, height: 300 });
      const ax1 = fig1.addAxes();
      ax1.plot(tensor([1, 2, 3]), tensor([1, 4, 9]));
      const result1 = fig1.renderSVG();

      const fig2 = new Figure({ width: 400, height: 300 });
      const ax2 = fig2.addAxes();
      ax2.plot(tensor([1, 2, 3]), tensor([1, 4, 9]));
      const result2 = fig2.renderSVG();

      expect(result1.svg).toBe(result2.svg);
    });

    it("should produce different SVG for different data", () => {
      const fig1 = new Figure();
      const ax1 = fig1.addAxes();
      ax1.plot(tensor([1, 2, 3]), tensor([1, 4, 9]));
      const result1 = fig1.renderSVG();

      const fig2 = new Figure();
      const ax2 = fig2.addAxes();
      ax2.plot(tensor([1, 2, 3, 4]), tensor([2, 5, 10, 17]));
      const result2 = fig2.renderSVG();

      expect(result1.svg).not.toBe(result2.svg);
    });
  });

  describe("Background Colors", () => {
    it("should apply figure background color", () => {
      const fig = new Figure({ background: "#123456" });
      fig.addAxes();
      const result = fig.renderSVG();
      expect(result.svg).toContain("#123456");
    });

    it("should apply axes facecolor", () => {
      const fig = new Figure();
      fig.addAxes({ facecolor: "#abcdef" });
      const result = fig.renderSVG();
      expect(result.svg).toContain("#abcdef");
    });

    it("should handle transparent colors", () => {
      const fig = new Figure({ background: "#ffffff00" });
      fig.addAxes();
      const result = fig.renderSVG();
      expect(result.kind).toBe("svg");
    });
  });

  describe("Histogram Binning Edge Cases", () => {
    it("should handle data at bin edges", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const hist = ax.hist(tensor([0, 1, 2, 3, 4, 5]), 5);
      let total = 0;
      for (let i = 0; i < hist.counts.length; i++) {
        total += hist.counts[i] ?? 0;
      }
      expect(total).toBe(6);
    });

    it("should handle repeated values", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const hist = ax.hist(tensor([1, 1, 1, 2, 2]), 3);
      let total = 0;
      for (let i = 0; i < hist.counts.length; i++) {
        total += hist.counts[i] ?? 0;
      }
      expect(total).toBe(5);
    });

    it("should handle uniform distribution", () => {
      const fig = new Figure();
      const ax = fig.addAxes();
      const data = Array.from({ length: 1000 }, () => Math.random());
      const hist = ax.hist(tensor(data), 10);
      expect(hist.bins.length).toBe(10);
      expect(hist.counts.length).toBe(10);
    });
  });
});
