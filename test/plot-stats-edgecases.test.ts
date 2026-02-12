import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { Tensor } from "../src/ndarray/tensor";
import { Figure, figure, heatmap, plotLearningCurve, plotValidationCurve, show } from "../src/plot";
import { corrcoef } from "../src/stats";

const renderCurrentSvg = (): string => {
  const rendered = show({ format: "svg" });
  expect(rendered).not.toBeInstanceOf(Promise);
  if (rendered instanceof Promise) return "";
  return rendered.svg;
};

describe("Plot + Stats Integration", () => {
  it("plots correlation matrix as heatmap", () => {
    const data = tensor([
      [1, 2, 3, 4],
      [2, 4, 6, 8],
      [4, 1, 2, 3],
    ]);
    const corr = corrcoef(data);
    figure({ width: 240, height: 180 });
    heatmap(corr);
    const svg = renderCurrentSvg();
    expect(svg).toContain("<rect");
  });

  it("renders correlation heatmap to SVG", () => {
    const fig = new Figure({ width: 220, height: 180 });
    const ax = fig.addAxes();
    const data = tensor([
      [1, 2, 3],
      [3, 2, 1],
      [2, 3, 1],
    ]);
    ax.heatmap(corrcoef(data));
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<rect");
  });

  it("supports BigInt tensors in line plots", () => {
    const x = Tensor.fromTypedArray({
      data: new BigInt64Array([1n, 2n, 3n]),
      shape: [3],
      dtype: "int64",
      device: "cpu",
    });
    const y = Tensor.fromTypedArray({
      data: new BigInt64Array([1n, 4n, 9n]),
      shape: [3],
      dtype: "int64",
      device: "cpu",
    });
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(x, y);
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("polyline");
  });

  it("uses explicit colors for learning/validation curves", () => {
    figure({ width: 320, height: 220 });
    const trainSizes = tensor([10, 20, 30]);
    const trainScores = tensor([0.8, 0.85, 0.9]);
    const valScores = tensor([0.7, 0.8, 0.88]);
    plotLearningCurve(trainSizes, trainScores, valScores, { color: "#123456" });
    plotValidationCurve(trainSizes, trainScores, valScores, { color: "#654321" });
    const svg = renderCurrentSvg();
    expect(svg).toContain("#123456");
    expect(svg).toContain("#654321");
  });

  it("renders custom figure via show({ figure })", () => {
    const fig = new Figure({ width: 160, height: 120 });
    fig.addAxes().plot(tensor([0, 1]), tensor([1, 0]));
    const rendered = show({ figure: fig, format: "svg" });
    if (!("kind" in rendered)) throw new Error("Expected synchronous SVG result");
    expect(rendered.kind).toBe("svg");
    expect(rendered.svg).toContain("<svg");
  });
});
