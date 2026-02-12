import { describe, expect, it } from "vitest";
import { type Tensor, tensor } from "../src/ndarray";
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

describe("Plot Classes", () => {
  it("should create Figure instance", () => {
    const fig = new Figure();
    expect(fig).toBeDefined();
  });

  it("should create Axes instance", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    expect(ax).toBeDefined();
  });
});

describe("Basic Plots", () => {
  const x = tensor([1, 2, 3, 4]);
  const y = tensor([1, 4, 9, 16]);

  it("should create line plot", () => {
    figure();
    plot(x, y);
    const svg = renderCurrentSvg();
    expect(svg).toContain("<polyline");
  });

  it("should create scatter plot", () => {
    figure();
    scatter(x, y);
    const svg = renderCurrentSvg();
    expect(svg).toContain("<circle");
  });

  it("should create bar chart", () => {
    figure();
    bar(x, y);
    const svg = renderCurrentSvg();
    expect(svg).toContain("<rect");
  });

  it("should create horizontal bar chart", () => {
    figure();
    barh(x, y);
    const svg = renderCurrentSvg();
    expect(svg).toContain("<rect");
  });

  it("should create histogram", () => {
    figure();
    hist(x, 10);
    const svg = renderCurrentSvg();
    expect(svg).toContain("<rect");
  });

  it("should create box plot", () => {
    figure();
    boxplot(x);
    const svg = renderCurrentSvg();
    expect(svg).toContain('stroke-width="2"');
  });

  it("should create violin plot", () => {
    figure();
    violinplot(x);
    const svg = renderCurrentSvg();
    expect(svg).toContain('stroke-width="3"');
  });

  it("should create pie chart", () => {
    figure();
    pie(y);
    const svg = renderCurrentSvg();
    expect(svg).toContain("<path");
  });
});

describe("Advanced Plots", () => {
  it("should create heatmap", () => {
    const data = tensor([
      [1, 2],
      [3, 4],
    ]);
    figure();
    heatmap(data);
    const svg = renderCurrentSvg();
    expect(svg).toContain("rgb(");
  });

  it("should display image", () => {
    const img = tensor([
      [1, 2],
      [3, 4],
    ]);
    figure();
    imshow(img);
    const svg = renderCurrentSvg();
    expect(svg).toContain("rgb(");
  });

  it("should create contour plot", () => {
    const Z = tensor([
      [1, 2],
      [3, 4],
    ]);
    const fig = figure({ width: 200, height: 140 });
    contour(tensor([]), tensor([]), Z);
    const svg = fig.renderSVG().svg;
    expect(svg).toContain('stroke="#1f77b4"');
  });

  it("should create filled contour plot", () => {
    const Z = tensor([
      [1, 2],
      [3, 4],
    ]);
    const fig = figure({ width: 200, height: 140 });
    contourf(tensor([]), tensor([]), Z);
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<rect");
  });
});

describe("ML-Specific Plots", () => {
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

  it("should plot ROC curve", () => {
    const fpr = tensor([0, 0.2, 0.4, 1]);
    const tpr = tensor([0, 0.6, 0.8, 1]);
    figure();
    plotRocCurve(fpr, tpr, 0.85);
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
    ]);
    const y = tensor([0, 1]);
    const model = {
      predict: (grid: Tensor) => {
        const rows = grid.shape[0] ?? 0;
        return tensor(Array.from({ length: rows }, (_, i) => (i % 2 === 0 ? 0 : 1)));
      },
    };
    figure();
    plotDecisionBoundary(X, y, model);
    const svg = renderCurrentSvg();
    expect(svg).toContain("<rect");
    expect(svg).toContain("<circle");
  });
});

describe("SVG Rendering", () => {
  it("should render valid SVG for line plot", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    ax.plot(tensor([1, 2, 3]), tensor([1, 4, 9]));
    const result = fig.renderSVG();
    expect(result.kind).toBe("svg");
    expect(result.svg).toContain("<?xml");
    expect(result.svg).toContain("<svg");
    expect(result.svg).toContain("</svg>");
    expect(result.svg).toContain("polyline");
  });

  it("should render valid SVG for scatter plot", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    ax.scatter(tensor([1, 2, 3]), tensor([1, 4, 9]));
    const result = fig.renderSVG();
    expect(result.kind).toBe("svg");
    expect(result.svg).toContain("<circle");
  });

  it("should render valid SVG for bar chart", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    ax.bar(tensor([1, 2, 3]), tensor([5, 10, 7]));
    const result = fig.renderSVG();
    expect(result.kind).toBe("svg");
    expect(result.svg).toContain("<rect");
  });

  it("should render valid SVG for histogram", () => {
    const fig = new Figure({ width: 400, height: 300 });
    const ax = fig.addAxes();
    ax.hist(tensor([1, 2, 2, 3, 3, 3, 4]), 4);
    const result = fig.renderSVG();
    expect(result.kind).toBe("svg");
    expect(result.svg).toContain("<rect");
  });

  it("should include background color in SVG", () => {
    const fig = new Figure({ width: 400, height: 300, background: "#f0f0f0" });
    fig.addAxes();
    const result = fig.renderSVG();
    expect(result.svg).toContain("#f0f0f0");
  });

  it("should set correct SVG dimensions", () => {
    const fig = new Figure({ width: 500, height: 400 });
    fig.addAxes();
    const result = fig.renderSVG();
    expect(result.svg).toContain('width="500"');
    expect(result.svg).toContain('height="400"');
  });
});

describe("PNG Rendering", () => {
  it("should render valid PNG for line plot", async () => {
    const fig = new Figure({ width: 100, height: 100 });
    const ax = fig.addAxes();
    ax.plot(tensor([1, 2, 3]), tensor([1, 4, 9]));
    const result = await fig.renderPNG();
    expect(result.kind).toBe("png");
    expect(result.width).toBe(100);
    expect(result.height).toBe(100);
    expect(result.bytes).toBeInstanceOf(Uint8Array);
    expect(result.bytes.length).toBeGreaterThan(0);
    expect(result.bytes[0]).toBe(137);
    expect(result.bytes[1]).toBe(80);
    expect(result.bytes[2]).toBe(78);
    expect(result.bytes[3]).toBe(71);
  });

  it("should render valid PNG for scatter plot", async () => {
    const fig = new Figure({ width: 100, height: 100 });
    const ax = fig.addAxes();
    ax.scatter(tensor([1, 2, 3]), tensor([1, 4, 9]));
    const result = await fig.renderPNG();
    expect(result.kind).toBe("png");
    expect(result.bytes[0]).toBe(137);
  });

  it("should render valid PNG for bar chart", async () => {
    const fig = new Figure({ width: 100, height: 100 });
    const ax = fig.addAxes();
    ax.bar(tensor([1, 2, 3]), tensor([5, 10, 7]));
    const result = await fig.renderPNG();
    expect(result.kind).toBe("png");
    expect(result.bytes[0]).toBe(137);
  });

  it("should render valid PNG for histogram", async () => {
    const fig = new Figure({ width: 100, height: 100 });
    const ax = fig.addAxes();
    ax.hist(tensor([1, 2, 2, 3, 3, 3, 4]), 4);
    const result = await fig.renderPNG();
    expect(result.kind).toBe("png");
    expect(result.bytes[0]).toBe(137);
  });
});

describe("Figure Configuration", () => {
  it("should create figure with default dimensions", () => {
    const fig = new Figure();
    expect(fig.width).toBe(640);
    expect(fig.height).toBe(480);
  });

  it("should create figure with custom dimensions", () => {
    const fig = new Figure({ width: 800, height: 600 });
    expect(fig.width).toBe(800);
    expect(fig.height).toBe(600);
  });

  it("should throw on invalid dimensions", () => {
    expect(() => new Figure({ width: -1, height: 100 })).toThrow();
    expect(() => new Figure({ width: 100, height: 0 })).toThrow();
    expect(() => new Figure({ width: 1.5, height: 100 })).toThrow();
  });

  it("should create multiple axes", () => {
    const fig = new Figure();
    const ax1 = fig.addAxes();
    const ax2 = fig.addAxes();
    expect(ax1).toBeDefined();
    expect(ax2).toBeDefined();
    expect(fig.axesList.length).toBe(2);
  });
});

describe("Plot Options", () => {
  it("should apply custom color to line plot", () => {
    const fig = new Figure({ width: 200, height: 200 });
    const ax = fig.addAxes();
    ax.plot(tensor([1, 2]), tensor([1, 2]), { color: "#ff0000" });
    const result = fig.renderSVG();
    expect(result.svg).toContain("#ff0000");
  });

  it("should apply custom color to scatter plot", () => {
    const fig = new Figure({ width: 200, height: 200 });
    const ax = fig.addAxes();
    ax.scatter(tensor([1, 2]), tensor([1, 2]), { color: "#00ff00" });
    const result = fig.renderSVG();
    expect(result.svg).toContain("#00ff00");
  });

  it("should apply custom color to bar chart", () => {
    const fig = new Figure({ width: 200, height: 200 });
    const ax = fig.addAxes();
    ax.bar(tensor([1, 2]), tensor([3, 4]), { color: "#0000ff" });
    const result = fig.renderSVG();
    expect(result.svg).toContain("#0000ff");
  });

  it("should apply linewidth option", () => {
    const fig = new Figure({ width: 200, height: 200 });
    const ax = fig.addAxes();
    ax.plot(tensor([1, 2]), tensor([1, 2]), { linewidth: 5 });
    const result = fig.renderSVG();
    expect(result.svg).toContain('stroke-width="5"');
  });

  it("should apply size option to scatter", () => {
    const fig = new Figure({ width: 200, height: 200 });
    const ax = fig.addAxes();
    ax.scatter(tensor([1, 2]), tensor([1, 2]), { size: 10 });
    const result = fig.renderSVG();
    expect(result.svg).toContain('r="10"');
  });
});

describe("Data Validation", () => {
  it("should handle empty tensors", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([]), tensor([]));
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<svg");
  });

  it("should throw on mismatched tensor lengths", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    expect(() => ax.plot(tensor([1, 2]), tensor([1]))).toThrow();
  });

  it("should throw on non-1D tensors for plot", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    expect(() => ax.plot(tensor([[1, 2]]), tensor([1, 2]))).toThrow();
  });

  it("should handle NaN values gracefully", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([1, NaN, 3]), tensor([1, 2, 3]));
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<polyline");
  });

  it("should handle Infinity values gracefully", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([1, Infinity, 3]), tensor([1, 2, 3]));
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<polyline");
  });
});

describe("Histogram Binning", () => {
  it("should create correct number of bins", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    const hist = ax.hist(tensor([1, 2, 3, 4, 5]), 5);
    expect(hist.bins.length).toBe(5);
    expect(hist.counts.length).toBe(5);
  });

  it("should count values correctly", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    const hist = ax.hist(tensor([1, 1, 1, 2, 2, 3]), 3);
    let totalCount = 0;
    for (let i = 0; i < hist.counts.length; i++) {
      totalCount += hist.counts[i] ?? 0;
    }
    expect(totalCount).toBe(6);
  });

  it("should handle single value", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.hist(tensor([5, 5, 5]), 3);
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<rect");
  });
});

describe("Multiple Plots on Same Axes", () => {
  it("should render multiple line plots", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([1, 2, 3]), tensor([1, 2, 3]), { color: "#ff0000" });
    ax.plot(tensor([1, 2, 3]), tensor([3, 2, 1]), { color: "#0000ff" });
    const result = fig.renderSVG();
    expect(result.svg).toContain("#ff0000");
    expect(result.svg).toContain("#0000ff");
  });

  it("should render mixed plot types", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([1, 2, 3]), tensor([1, 2, 3]));
    ax.scatter(tensor([1, 2, 3]), tensor([3, 2, 1]));
    const result = fig.renderSVG();
    expect(result.svg).toContain("polyline");
    expect(result.svg).toContain("circle");
  });
});
