import { describe, expect, it } from "vitest";
import { confusionMatrix } from "../src/metrics";
import { LinearRegression, LogisticRegression } from "../src/ml";
import { reshape, tensor } from "../src/ndarray";
import { GradTensor } from "../src/ndarray/autograd";
import { Linear } from "../src/nn";
import {
  Figure,
  figure,
  heatmap,
  hist,
  plot,
  plotConfusionMatrix,
  show,
  subplot,
} from "../src/plot";
import { StandardScaler } from "../src/preprocess";

const readU32BE = (buf: Uint8Array, offset: number): number =>
  ((buf[offset] ?? 0) << 24) |
  ((buf[offset + 1] ?? 0) << 16) |
  ((buf[offset + 2] ?? 0) << 8) |
  (buf[offset + 3] ?? 0);

const renderCurrentSvg = (): string => {
  const rendered = show({ format: "svg" });
  expect(rendered).not.toBeInstanceOf(Promise);
  if (rendered instanceof Promise) return "";
  return rendered.svg;
};

const containsChunkType = (buf: Uint8Array, ascii: string): boolean => {
  const target = new Uint8Array(ascii.split("").map((ch) => ch.charCodeAt(0)));
  outer: for (let i = 0; i + target.length <= buf.length; i += 1) {
    for (let j = 0; j < target.length; j += 1) {
      if (buf[i + j] !== target[j]) continue outer;
    }
    return true;
  }
  return false;
};

describe("Plot - Integrations (ML, NN, preprocess, metrics)", () => {
  it("renders PNG with correct IHDR dimensions", async () => {
    const fig = new Figure({ width: 123, height: 77 });
    fig.addAxes().plot(tensor([0, 1]), tensor([1, 0]));
    const png = await fig.renderPNG();
    expect(png.bytes[0]).toBe(137);
    const width = readU32BE(png.bytes, 16);
    const height = readU32BE(png.bytes, 20);
    expect(width).toBe(123);
    expect(height).toBe(77);
  });

  it("PNG payload contains an IDAT chunk", async () => {
    const fig = new Figure({ width: 120, height: 80 });
    fig.addAxes().scatter(tensor([0.2, 0.8]), tensor([0.2, 0.8]));
    const png = await fig.renderPNG();
    expect(png.bytes.length).toBeGreaterThan(64);
    expect(containsChunkType(png.bytes, "IDAT")).toBe(true);
  });

  it("renders axes viewport with expected padding", () => {
    const fig = new Figure({ width: 200, height: 100 });
    fig.addAxes({ padding: 10 });
    const svg = fig.renderSVG().svg;
    expect(svg).toContain('x="10"');
    expect(svg).toContain('y="10"');
    expect(svg).toContain('width="180"');
    expect(svg).toContain('height="80"');
  });

  it("renders subplot viewports in grid layout", () => {
    figure({ width: 320, height: 240 });
    subplot(2, 2, 1, { padding: 0 });
    subplot(2, 2, 2, { padding: 0 });
    const rendered = show({ format: "svg" });
    expect(rendered).not.toBeInstanceOf(Promise);
    if (rendered instanceof Promise) return;
    const svg = rendered.svg;
    expect(svg).toContain('width="320"');
    expect(svg).toContain('height="240"');
  });

  it("plots ML regression predictions vs actual", () => {
    const X = tensor([[1], [2], [3], [4]]);
    const y = tensor([2, 4, 6, 8]);
    const model = new LinearRegression();
    model.fit(X, y);
    const yPred = model.predict(X);
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(y, yPred);
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("polyline");
  });

  it("plots ML classification confusion matrix", () => {
    const yTrue = tensor([0, 1, 0, 1, 1]);
    const yPred = tensor([0, 1, 1, 1, 0]);
    const cm = confusionMatrix(yTrue, yPred);
    figure({ width: 200, height: 150 });
    plotConfusionMatrix(cm);
    const svg = renderCurrentSvg();
    expect(svg).toContain("Confusion Matrix");
  });

  it("plots LogisticRegression scores with scatter", () => {
    const X = tensor([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
    const y = tensor([0, 1, 1, 0]);
    const model = new LogisticRegression({ maxIter: 50, learningRate: 0.1 });
    model.fit(X, y);
    const preds = model.predict(X);
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.scatter(y, preds);
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("<circle");
  });

  it("plots NN Linear output", () => {
    const layer = new Linear(2, 1);
    const input = GradTensor.fromTensor(
      tensor([
        [1, 2],
        [3, 4],
      ])
    );
    const output = layer.forward(input);
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.plot(tensor([0, 1]), output.reshape([2]));
    const svg = fig.renderSVG().svg;
    expect(svg).toContain("polyline");
  });

  it("plots scaled data histogram", () => {
    // StandardScaler now requires 2D input
    const data = tensor([[1], [2], [3], [4], [5], [6]]);
    const scaler = new StandardScaler();
    scaler.fit(data);
    const scaled = scaler.transform(data);
    const scaled1d = reshape(scaled, [6]);
    figure({ width: 220, height: 160 });
    hist(scaled1d, 4);
    const rendered = show({ format: "svg" });
    expect(rendered).not.toBeInstanceOf(Promise);
    if (rendered instanceof Promise) return;
    expect(rendered.kind).toBe("svg");
    expect(rendered.svg).toContain("<rect");
  });

  it("plots stats output heatmap via global functions", () => {
    const mat = tensor([
      [1, 0, -1],
      [0, 1, 0],
      [-1, 0, 1],
    ]);
    figure({ width: 200, height: 140 });
    heatmap(mat);
    figure({ width: 200, height: 150 });
    plot(tensor([0, 1, 2]), tensor([0, 1, 4]));
    const rendered = show({ format: "svg" });
    expect(rendered).not.toBeInstanceOf(Promise);
    if (rendered instanceof Promise) return;
    expect(rendered.svg).toContain("polyline");
  });
});
