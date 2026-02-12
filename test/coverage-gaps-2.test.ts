import { describe, expect, it } from "vitest";
import { parameter, tensor } from "../src/ndarray";
import { Adam } from "../src/optim/optimizers/adam";
import { AdamW } from "../src/optim/optimizers/adamw";
import { SGD } from "../src/optim/optimizers/sgd";
import { figure, gca } from "../src/plot/figure/state";
import { getParamData } from "./optim-test-helpers";

describe("Axes title/xlabel/ylabel SVG rendering", () => {
  it("should render title in SVG", () => {
    const fig = figure({ width: 320, height: 240 });
    const ax = gca();
    ax.plot(tensor([1, 2, 3]), tensor([4, 5, 6]));
    ax.setTitle("My Title");
    const result = fig.renderSVG();
    expect(result.svg).toContain("My Title");
  });

  it("should render xlabel in SVG", () => {
    const fig = figure({ width: 320, height: 240 });
    const ax = gca();
    ax.plot(tensor([1, 2, 3]), tensor([4, 5, 6]));
    ax.setXLabel("X Axis");
    const result = fig.renderSVG();
    expect(result.svg).toContain("X Axis");
  });

  it("should render ylabel in SVG", () => {
    const fig = figure({ width: 320, height: 240 });
    const ax = gca();
    ax.plot(tensor([1, 2, 3]), tensor([4, 5, 6]));
    ax.setYLabel("Y Axis");
    const result = fig.renderSVG();
    expect(result.svg).toContain("Y Axis");
  });

  it("should render title in raster", async () => {
    const fig = figure({ width: 200, height: 150 });
    const ax = gca();
    ax.plot(tensor([1, 2, 3]), tensor([4, 5, 6]));
    ax.setTitle("Raster Title");
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
    expect(result.bytes.length).toBeGreaterThan(0);
  });

  it("should render xlabel in raster", async () => {
    const fig = figure({ width: 200, height: 150 });
    const ax = gca();
    ax.plot(tensor([1, 2, 3]), tensor([4, 5, 6]));
    ax.setXLabel("Raster X");
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
  });

  it("should render ylabel in raster", async () => {
    const fig = figure({ width: 200, height: 150 });
    const ax = gca();
    ax.plot(tensor([1, 2, 3]), tensor([4, 5, 6]));
    ax.setYLabel("Raster Y");
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
  });

  it("should render all labels together in raster", async () => {
    const fig = figure({ width: 200, height: 150 });
    const ax = gca();
    ax.plot(tensor([1, 2, 3]), tensor([4, 5, 6]));
    ax.setTitle("Title");
    ax.setXLabel("X");
    ax.setYLabel("Y");
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
  });
});

describe("Axes legend rendering", () => {
  it("should render legend with line entries in SVG", () => {
    const fig = figure({ width: 320, height: 240 });
    const ax = gca();
    ax.plot(tensor([1, 2, 3]), tensor([4, 5, 6]), { label: "Line A" });
    ax.plot(tensor([1, 2, 3]), tensor([6, 5, 4]), { label: "Line B" });
    ax.legend();
    const result = fig.renderSVG();
    expect(result.svg).toContain("Line A");
    expect(result.svg).toContain("Line B");
    expect(result.svg).toContain("legend");
  });

  it("should render legend with scatter (marker) entries in SVG", () => {
    const fig = figure({ width: 320, height: 240 });
    const ax = gca();
    ax.scatter(tensor([1, 2, 3]), tensor([4, 5, 6]), { label: "Scatter A" });
    ax.legend();
    const result = fig.renderSVG();
    expect(result.svg).toContain("Scatter A");
  });

  it("should render legend with bar (box) entries in SVG", () => {
    const fig = figure({ width: 320, height: 240 });
    const ax = gca();
    ax.bar(tensor([1, 2, 3]), tensor([4, 5, 6]), { label: "Bar A" });
    ax.legend();
    const result = fig.renderSVG();
    expect(result.svg).toContain("Bar A");
  });

  it("should render legend in raster with line entries", async () => {
    const fig = figure({ width: 200, height: 150 });
    const ax = gca();
    ax.plot(tensor([1, 2, 3]), tensor([4, 5, 6]), { label: "Line" });
    ax.legend();
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
  });

  it("should render legend in raster with scatter entries", async () => {
    const fig = figure({ width: 200, height: 150 });
    const ax = gca();
    ax.scatter(tensor([1, 2, 3]), tensor([4, 5, 6]), { label: "Scatter" });
    ax.legend();
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
  });

  it("should render legend in raster with bar entries", async () => {
    const fig = figure({ width: 200, height: 150 });
    const ax = gca();
    ax.bar(tensor([1, 2, 3]), tensor([4, 5, 6]), { label: "Bar" });
    ax.legend();
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
  });

  it("should support legend location options", () => {
    const fig = figure({ width: 320, height: 240 });
    const ax = gca();
    ax.plot(tensor([1, 2, 3]), tensor([4, 5, 6]), { label: "Data" });
    ax.legend({ location: "lower-left" });
    const result = fig.renderSVG();
    expect(result.svg).toContain("Data");
  });

  it("should support legend location lower-right in raster", async () => {
    const fig = figure({ width: 200, height: 150 });
    const ax = gca();
    ax.plot(tensor([1, 2, 3]), tensor([4, 5, 6]), { label: "Data" });
    ax.legend({ location: "lower-right" });
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
  });

  it("should not render legend when visible is false", () => {
    const fig = figure({ width: 320, height: 240 });
    const ax = gca();
    ax.plot(tensor([1, 2, 3]), tensor([4, 5, 6]), { label: "Data" });
    ax.legend({ visible: false });
    const result = fig.renderSVG();
    expect(result.svg).not.toContain("legend");
  });
});

describe("Axes custom ticks", () => {
  it("should render custom x ticks in SVG", () => {
    const fig = figure({ width: 320, height: 240 });
    const ax = gca();
    ax.plot(tensor([0, 5, 10]), tensor([1, 2, 3]));
    ax.setXTicks([0, 5, 10], ["zero", "five", "ten"]);
    const result = fig.renderSVG();
    expect(result.svg).toContain("zero");
    expect(result.svg).toContain("five");
    expect(result.svg).toContain("ten");
  });

  it("should render custom y ticks in SVG", () => {
    const fig = figure({ width: 320, height: 240 });
    const ax = gca();
    ax.plot(tensor([0, 5, 10]), tensor([0, 50, 100]));
    ax.setYTicks([0, 50, 100], ["low", "mid", "high"]);
    const result = fig.renderSVG();
    expect(result.svg).toContain("low");
    expect(result.svg).toContain("mid");
  });

  it("should render custom ticks in raster", async () => {
    const fig = figure({ width: 200, height: 150 });
    const ax = gca();
    ax.plot(tensor([0, 5, 10]), tensor([1, 2, 3]));
    ax.setXTicks([0, 5, 10]);
    ax.setYTicks([1, 2, 3]);
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
  });
});

describe("Contour2D raster rendering", () => {
  it("should render contour lines as raster", async () => {
    const fig = figure({ width: 100, height: 100 });
    const ax = gca();
    const x = tensor([0, 1, 2]);
    const y = tensor([0, 1, 2]);
    const z = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    ax.contour(x, y, z);
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
  });

  it("should render contour with label in SVG", () => {
    const fig = figure({ width: 200, height: 200 });
    const ax = gca();
    const x = tensor([0, 1, 2, 3]);
    const y = tensor([0, 1, 2, 3]);
    const z = tensor([
      [0, 1, 2, 3],
      [1, 2, 3, 4],
      [2, 3, 4, 5],
      [3, 4, 5, 6],
    ]);
    ax.contour(x, y, z, { label: "Contour" });
    ax.legend();
    const result = fig.renderSVG();
    expect(result.svg).toContain("Contour");
  });
});

describe("Pie raster rendering with labels and legend", () => {
  it("should render pie with labels in raster", async () => {
    const fig = figure({ width: 200, height: 200 });
    const ax = gca();
    ax.pie(tensor([30, 20, 50]), ["A", "B", "C"]);
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
  });

  it("should render pie legend entries with labels", () => {
    const fig = figure({ width: 200, height: 200 });
    const ax = gca();
    ax.pie(tensor([30, 20, 50]), ["Alpha", "Beta", "Gamma"]);
    ax.legend();
    const result = fig.renderSVG();
    expect(result.svg).toContain("Alpha");
  });

  it("should render pie legend entry with label option", () => {
    const fig = figure({ width: 200, height: 200 });
    const ax = gca();
    ax.pie(tensor([30, 70]), undefined, { label: "Distribution" });
    ax.legend();
    const result = fig.renderSVG();
    expect(result.svg).toContain("Distribution");
  });

  it("should render pie with custom colors", () => {
    const fig = figure({ width: 200, height: 200 });
    const ax = gca();
    ax.pie(tensor([30, 70]), undefined, { colors: ["#ff0000", "#00ff00"] });
    const result = fig.renderSVG();
    expect(result.svg).toContain("#ff0000");
  });
});

describe("Bar2D raster rendering", () => {
  it("should render bar as raster", async () => {
    const fig = figure({ width: 200, height: 150 });
    const ax = gca();
    ax.bar(tensor([1, 2, 3]), tensor([10, 20, 15]));
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
  });

  it("should handle NaN in bar data range", () => {
    const fig = figure({ width: 200, height: 150 });
    const ax = gca();
    ax.bar(tensor([Number.NaN, Number.NaN]), tensor([Number.NaN, Number.NaN]));
    const result = fig.renderSVG();
    expect(result.svg).toContain("svg");
  });
});

describe("Optimizer state dict serialization", () => {
  it("SGD should save and load state dict with same params", () => {
    const p = parameter(tensor([1, 2], { dtype: "float64" }));
    p.setGrad(tensor([0.5, -0.3], { dtype: "float64" }));
    const opt = new SGD([p], { lr: 0.01, momentum: 0.9 });
    opt.step();

    const stateDict = opt.stateDict();
    expect(stateDict).toBeDefined();
    expect(stateDict.paramGroups).toBeDefined();
    expect(stateDict.state).toBeDefined();

    // Load state back into same optimizer with same params
    opt.loadStateDict(stateDict);
  });

  it("Adam should save and load state dict with same params", () => {
    const p = parameter(tensor([1, 2], { dtype: "float64" }));
    p.setGrad(tensor([0.5, -0.3], { dtype: "float64" }));
    const opt = new Adam([p], { lr: 0.001 });
    opt.step();

    const stateDict = opt.stateDict();
    expect(stateDict).toBeDefined();

    opt.loadStateDict(stateDict);
  });

  it("AdamW should save and load state dict with same params", () => {
    const p = parameter(tensor([1, 2], { dtype: "float64" }));
    p.setGrad(tensor([0.5, -0.3], { dtype: "float64" }));
    const opt = new AdamW([p], { lr: 0.001 });
    opt.step();

    const stateDict = opt.stateDict();
    expect(stateDict).toBeDefined();

    opt.loadStateDict(stateDict);
  });
});

describe("optim/_internal edge cases", () => {
  it("assertHasGradFloat throws on non-float param", () => {
    // int32 parameter should throw DTypeError
    const p = parameter(tensor([1, 2], { dtype: "int32" }));
    p.setGrad(tensor([0.1, 0.2], { dtype: "float64" }));
    const opt = new SGD([p], { lr: 0.01 });
    expect(() => opt.step()).toThrow();
  });

  it("assertHasGradFloat throws on mismatched dtypes", () => {
    const p = parameter(tensor([1, 2], { dtype: "float32" }));
    p.setGrad(tensor([0.1, 0.2], { dtype: "float64" }));
    const opt = new SGD([p], { lr: 0.01 });
    expect(() => opt.step()).toThrow();
  });
});

describe("Boxplot raster rendering", () => {
  it("should render boxplot as raster", async () => {
    const fig = figure({ width: 200, height: 150 });
    const ax = gca();
    ax.boxplot(tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
  });
});

describe("Violinplot raster rendering", () => {
  it("should render violinplot as raster", async () => {
    const fig = figure({ width: 200, height: 150 });
    const ax = gca();
    ax.violinplot(tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
  });
});

describe("Heatmap raster rendering", () => {
  it("should render heatmap as raster", async () => {
    const fig = figure({ width: 100, height: 100 });
    const ax = gca();
    ax.heatmap(
      tensor([
        [1, 2],
        [3, 4],
      ])
    );
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
  });
});

describe("Histogram raster rendering", () => {
  it("should render histogram as raster", async () => {
    const fig = figure({ width: 200, height: 150 });
    const ax = gca();
    ax.hist(tensor([1, 2, 2, 3, 3, 3, 4, 4, 5]), 5);
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
  });
});

describe("Scatter raster rendering", () => {
  it("should render scatter as raster", async () => {
    const fig = figure({ width: 200, height: 150 });
    const ax = gca();
    ax.scatter(tensor([1, 2, 3]), tensor([4, 5, 6]));
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
  });
});

describe("Adam optimizer getLearningRate/setLearningRate", () => {
  it("should get and set learning rate", () => {
    const p = parameter(tensor([1], { dtype: "float64" }));
    const opt = new Adam([p], { lr: 0.001 });
    expect(opt.getLearningRate()).toBe(0.001);
    opt.setLearningRate(0.01);
    expect(opt.getLearningRate()).toBe(0.01);
  });

  it("should throw on invalid group index", () => {
    const p = parameter(tensor([1], { dtype: "float64" }));
    const opt = new Adam([p]);
    expect(() => opt.getLearningRate(99)).toThrow(/Invalid group index/);
  });

  it("should throw on invalid lr in setLearningRate", () => {
    const p = parameter(tensor([1], { dtype: "float64" }));
    const opt = new Adam([p]);
    expect(() => opt.setLearningRate(-1)).toThrow(/Invalid learning rate/);
  });

  it("should support amsgrad variant", () => {
    const p = parameter(tensor([1, 2], { dtype: "float64" }));
    p.setGrad(tensor([0.5, -0.3], { dtype: "float64" }));
    const opt = new Adam([p], { amsgrad: true, lr: 0.01 });
    opt.step();
    p.setGrad(tensor([0.2, -0.1], { dtype: "float64" }));
    opt.step();
    const after = Array.from(getParamData(p, "Adam param"));
    expect(after[0]).toBeLessThan(1);
  });
});
