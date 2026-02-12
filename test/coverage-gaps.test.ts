import { describe, expect, it } from "vitest";
import { parameter, tensor } from "../src/ndarray";
import { binaryCrossEntropyWithLogitsLoss, crossEntropyLoss } from "../src/nn/losses/crossEntropy";
import { AdaDelta } from "../src/optim/optimizers/adadelta";
import { Adagrad } from "../src/optim/optimizers/adagrad";
import { AdamW } from "../src/optim/optimizers/adamw";
import { Nadam } from "../src/optim/optimizers/nadam";
import { RMSprop } from "../src/optim/optimizers/rmsprop";
import { figure, gca, subplot } from "../src/plot/figure/state";
import { isNodeEnvironment_export, isPNGSupported, pngEncodeRGBA } from "../src/plot/renderers/png";
import { estimateTextWidth } from "../src/plot/utils/text";
import { getParamData } from "./optim-test-helpers";

describe("Optimizer getLearningRate / setLearningRate / stepCount", () => {
  describe("AdaDelta", () => {
    it("should return stepCount and increment on step", () => {
      const p = parameter(tensor([1, 2], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.2], { dtype: "float64" }));
      const opt = new AdaDelta([p]);
      expect(opt.stepCount).toBe(0);
      opt.step();
      expect(opt.stepCount).toBe(1);
      opt.step();
      expect(opt.stepCount).toBe(2);
    });

    it("should get and set learning rate", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const opt = new AdaDelta([p], { lr: 1.0 });
      expect(opt.getLearningRate()).toBe(1.0);
      opt.setLearningRate(0.5);
      expect(opt.getLearningRate()).toBe(0.5);
    });

    it("should throw on invalid group index in getLearningRate", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const opt = new AdaDelta([p]);
      expect(() => opt.getLearningRate(99)).toThrow(/Invalid group index/);
    });

    it("should throw on invalid learning rate in setLearningRate", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const opt = new AdaDelta([p]);
      expect(() => opt.setLearningRate(-1)).toThrow(/Invalid learning rate/);
    });
  });

  describe("Adagrad", () => {
    it("should return stepCount and increment on step", () => {
      const p = parameter(tensor([1, 2], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.2], { dtype: "float64" }));
      const opt = new Adagrad([p]);
      expect(opt.stepCount).toBe(0);
      opt.step();
      expect(opt.stepCount).toBe(1);
    });

    it("should get and set learning rate", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const opt = new Adagrad([p], { lr: 0.01 });
      expect(opt.getLearningRate()).toBe(0.01);
      opt.setLearningRate(0.05);
      expect(opt.getLearningRate()).toBe(0.05);
    });

    it("should throw on invalid group index in getLearningRate", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const opt = new Adagrad([p]);
      expect(() => opt.getLearningRate(99)).toThrow(/Invalid group index/);
    });

    it("should throw on invalid learning rate in setLearningRate", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const opt = new Adagrad([p]);
      expect(() => opt.setLearningRate(-1)).toThrow(/Invalid learning rate/);
    });
  });

  describe("Nadam", () => {
    it("should return stepCount and increment on step", () => {
      const p = parameter(tensor([1, 2], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.2], { dtype: "float64" }));
      const opt = new Nadam([p]);
      expect(opt.stepCount).toBe(0);
      opt.step();
      expect(opt.stepCount).toBe(1);
    });

    it("should get and set learning rate", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const opt = new Nadam([p], { lr: 0.002 });
      expect(opt.getLearningRate()).toBe(0.002);
      opt.setLearningRate(0.01);
      expect(opt.getLearningRate()).toBe(0.01);
    });

    it("should throw on invalid group index in getLearningRate", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const opt = new Nadam([p]);
      expect(() => opt.getLearningRate(99)).toThrow(/Invalid group index/);
    });

    it("should throw on invalid learning rate in setLearningRate", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const opt = new Nadam([p]);
      expect(() => opt.setLearningRate(-1)).toThrow(/Invalid learning rate/);
    });
  });

  describe("RMSprop", () => {
    it("should return stepCount and increment on step", () => {
      const p = parameter(tensor([1, 2], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.2], { dtype: "float64" }));
      const opt = new RMSprop([p]);
      expect(opt.stepCount).toBe(0);
      opt.step();
      expect(opt.stepCount).toBe(1);
    });

    it("should get and set learning rate", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const opt = new RMSprop([p], { lr: 0.01 });
      expect(opt.getLearningRate()).toBe(0.01);
      opt.setLearningRate(0.05);
      expect(opt.getLearningRate()).toBe(0.05);
    });

    it("should throw on invalid group index in getLearningRate", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const opt = new RMSprop([p]);
      expect(() => opt.getLearningRate(99)).toThrow(/Invalid group index/);
    });

    it("should throw on invalid learning rate in setLearningRate", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const opt = new RMSprop([p]);
      expect(() => opt.setLearningRate(-1)).toThrow(/Invalid learning rate/);
    });

    it("should support centered variant", () => {
      const p = parameter(tensor([1, 2], { dtype: "float64" }));
      p.setGrad(tensor([0.5, -0.3], { dtype: "float64" }));
      const opt = new RMSprop([p], { centered: true, lr: 0.01 });
      const before = Array.from(getParamData(p, "RMSprop param"));
      opt.step();
      const after = Array.from(getParamData(p, "RMSprop param"));
      expect(after[0]).not.toBe(before[0]);
      expect(after[1]).not.toBe(before[1]);
    });

    it("should support momentum", () => {
      const p = parameter(tensor([1, 2], { dtype: "float64" }));
      p.setGrad(tensor([0.5, -0.3], { dtype: "float64" }));
      const opt = new RMSprop([p], { momentum: 0.9, lr: 0.01 });
      const before = Array.from(getParamData(p, "RMSprop param"));
      opt.step();
      const after = Array.from(getParamData(p, "RMSprop param"));
      expect(after[0]).not.toBe(before[0]);
    });

    it("should support centered + momentum together", () => {
      const p = parameter(tensor([1, 2], { dtype: "float64" }));
      p.setGrad(tensor([0.5, -0.3], { dtype: "float64" }));
      const opt = new RMSprop([p], { centered: true, momentum: 0.9, lr: 0.01 });
      opt.step();
      // Re-set grad for second step to exercise momentum buffer
      p.setGrad(tensor([0.2, -0.1], { dtype: "float64" }));
      opt.step();
      const after = Array.from(getParamData(p, "RMSprop param"));
      expect(after[0]).toBeLessThan(1);
    });
  });

  describe("AdamW", () => {
    it("should get and set learning rate", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const opt = new AdamW([p], { lr: 0.001 });
      expect(opt.getLearningRate()).toBe(0.001);
      opt.setLearningRate(0.01);
      expect(opt.getLearningRate()).toBe(0.01);
    });

    it("should throw on invalid group index in getLearningRate", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const opt = new AdamW([p]);
      expect(() => opt.getLearningRate(99)).toThrow(/Invalid group index/);
    });

    it("should throw on invalid learning rate in setLearningRate", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const opt = new AdamW([p]);
      expect(() => opt.setLearningRate(-1)).toThrow(/Invalid learning rate/);
    });

    it("should support amsgrad variant", () => {
      const p = parameter(tensor([1, 2], { dtype: "float64" }));
      p.setGrad(tensor([0.5, -0.3], { dtype: "float64" }));
      const opt = new AdamW([p], { amsgrad: true, lr: 0.01 });
      opt.step();
      p.setGrad(tensor([0.2, -0.1], { dtype: "float64" }));
      opt.step();
      const after = Array.from(getParamData(p, "AdamW param"));
      expect(after[0]).toBeLessThan(1);
    });

    it("should support closure in step", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      p.setGrad(tensor([0.1], { dtype: "float64" }));
      const opt = new AdamW([p]);
      const loss = opt.step(() => 42);
      expect(loss).toBe(42);
    });
  });
});

describe("PNG encoder", () => {
  it("should encode a 1x1 RGBA image", async () => {
    const rgba = new Uint8ClampedArray([255, 0, 0, 255]);
    const png = await pngEncodeRGBA(1, 1, rgba);
    // PNG signature: 137 80 78 71 13 10 26 10
    expect(png[0]).toBe(137);
    expect(png[1]).toBe(80);
    expect(png[2]).toBe(78);
    expect(png[3]).toBe(71);
    expect(png.length).toBeGreaterThan(8);
  });

  it("should encode a 2x2 RGBA image", async () => {
    const rgba = new Uint8ClampedArray(2 * 2 * 4);
    for (let i = 0; i < rgba.length; i += 4) {
      rgba[i] = 128;
      rgba[i + 1] = 64;
      rgba[i + 2] = 32;
      rgba[i + 3] = 255;
    }
    const png = await pngEncodeRGBA(2, 2, rgba);
    expect(png[0]).toBe(137);
    expect(png.length).toBeGreaterThan(20);
  });

  it("should throw on invalid dimensions", async () => {
    const rgba = new Uint8ClampedArray(4);
    await expect(pngEncodeRGBA(0, 1, rgba)).rejects.toThrow();
    await expect(pngEncodeRGBA(1, 0, rgba)).rejects.toThrow();
  });

  it("should throw on incorrect buffer length", async () => {
    const rgba = new Uint8ClampedArray(3); // wrong size for 1x1
    await expect(pngEncodeRGBA(1, 1, rgba)).rejects.toThrow(/RGBA buffer/);
  });

  it("should encode a large image spanning multiple deflate blocks", async () => {
    // 200x200 = 40000 pixels, 160000 bytes RGBA
    // raw = 200 * (1 + 800) = 160200 bytes > 65535 (multiple blocks)
    const w = 200;
    const h = 200;
    const rgba = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < rgba.length; i++) rgba[i] = i & 255;
    const png = await pngEncodeRGBA(w, h, rgba);
    expect(png[0]).toBe(137);
    expect(png.length).toBeGreaterThan(1000);
  });

  it("isPNGSupported should return boolean", () => {
    expect(typeof isPNGSupported()).toBe("boolean");
  });

  it("isNodeEnvironment_export should return boolean", () => {
    expect(typeof isNodeEnvironment_export()).toBe("boolean");
  });
});

describe("Text utils", () => {
  it("should estimate text width for non-empty string", () => {
    const width = estimateTextWidth("Hello", 12);
    expect(width).toBeCloseTo(5 * 12 * 0.6);
  });

  it("should return 0 for empty string", () => {
    expect(estimateTextWidth("", 12)).toBe(0);
  });

  it("should scale with font size", () => {
    const w1 = estimateTextWidth("A", 10);
    const w2 = estimateTextWidth("A", 20);
    expect(w2).toBeCloseTo(w1 * 2);
  });
});

describe("Plot state functions", () => {
  it("figure() creates a new figure", () => {
    const fig = figure({ width: 400, height: 300 });
    expect(fig.width).toBe(400);
    expect(fig.height).toBe(300);
  });

  it("gca() returns axes from current figure", () => {
    figure();
    const ax = gca();
    expect(ax).toBeDefined();
  });

  it("subplot() creates subplots", () => {
    figure({ width: 640, height: 480 });
    const ax1 = subplot(2, 2, 1);
    const ax2 = subplot(2, 2, 2);
    expect(ax1).toBeDefined();
    expect(ax2).toBeDefined();
  });

  it("subplot() throws on invalid parameters", () => {
    figure();
    expect(() => subplot(0, 1, 1)).toThrow();
    expect(() => subplot(1, 0, 1)).toThrow();
    expect(() => subplot(1, 1, 0)).toThrow();
    expect(() => subplot(1, 1, 2)).toThrow();
  });

  it("figure() with background color", () => {
    const fig = figure({ background: "#ff0000" });
    expect(fig).toBeDefined();
  });
});

describe("Cross entropy edge cases", () => {
  it("crossEntropyLoss with 2D soft labels", () => {
    const pred = tensor([
      [2.0, 1.0, 0.1],
      [0.1, 2.0, 1.0],
    ]);
    const target = tensor([
      [1, 0, 0],
      [0, 1, 0],
    ]);
    const loss = crossEntropyLoss(pred, target);
    expect(typeof loss).toBe("number");
    expect(loss).toBeGreaterThan(0);
  });

  it("crossEntropyLoss throws on 3D target", () => {
    const pred = tensor([
      [1, 2],
      [3, 4],
    ]);
    const target3D = tensor([[[1, 0]]]);
    expect(() => crossEntropyLoss(pred, target3D)).toThrow();
  });

  it("crossEntropyLoss throws on 1D input", () => {
    const pred = tensor([1, 2, 3]);
    const target = tensor([0]);
    expect(() => crossEntropyLoss(pred, target)).toThrow();
  });

  it("crossEntropyLoss throws on mismatched sample count (1D target)", () => {
    const pred = tensor([
      [1, 2],
      [3, 4],
    ]);
    const target = tensor([0, 1, 2]); // 3 samples vs 2
    expect(() => crossEntropyLoss(pred, target)).toThrow();
  });

  it("crossEntropyLoss throws on mismatched shape (2D target)", () => {
    const pred = tensor([
      [1, 2],
      [3, 4],
    ]);
    const target = tensor([
      [1, 0, 0],
      [0, 1, 0],
    ]); // 3 classes vs 2
    expect(() => crossEntropyLoss(pred, target)).toThrow();
  });

  it("binaryCrossEntropyWithLogitsLoss with 1D inputs", () => {
    const pred = tensor([0.5, -0.5, 1.0]);
    const target = tensor([1, 0, 1]);
    const loss = binaryCrossEntropyWithLogitsLoss(pred, target);
    expect(typeof loss).toBe("number");
    expect(loss).toBeGreaterThan(0);
  });

  it("binaryCrossEntropyWithLogitsLoss with 2D inputs", () => {
    const pred = tensor([[0.5], [-0.5], [1.0]]);
    const target = tensor([[1], [0], [1]]);
    const loss = binaryCrossEntropyWithLogitsLoss(pred, target);
    expect(typeof loss).toBe("number");
    expect(loss).toBeGreaterThan(0);
  });

  it("binaryCrossEntropyWithLogitsLoss throws on 3D input", () => {
    const pred = tensor([[[1]]]);
    const target = tensor([[[1]]]);
    expect(() => binaryCrossEntropyWithLogitsLoss(pred, target)).toThrow();
  });

  it("binaryCrossEntropyWithLogitsLoss throws on batch size mismatch", () => {
    const pred = tensor([0.5, -0.5]);
    const target = tensor([1, 0, 1]);
    expect(() => binaryCrossEntropyWithLogitsLoss(pred, target)).toThrow();
  });

  it("binaryCrossEntropyWithLogitsLoss throws on wrong column count", () => {
    const pred = tensor([
      [0.5, 0.3],
      [-0.5, 0.2],
    ]); // 2 columns, needs 1
    const target = tensor([1, 0]);
    expect(() => binaryCrossEntropyWithLogitsLoss(pred, target)).toThrow();
  });
});

describe("Optimizer closure and weight decay paths", () => {
  it("AdaDelta step with closure", () => {
    const p = parameter(tensor([1], { dtype: "float64" }));
    p.setGrad(tensor([0.1], { dtype: "float64" }));
    const opt = new AdaDelta([p]);
    const loss = opt.step(() => 3.14);
    expect(loss).toBe(3.14);
  });

  it("Adagrad step with closure", () => {
    const p = parameter(tensor([1], { dtype: "float64" }));
    p.setGrad(tensor([0.1], { dtype: "float64" }));
    const opt = new Adagrad([p]);
    const loss = opt.step(() => 2.71);
    expect(loss).toBe(2.71);
  });

  it("Adagrad with weight decay and lr decay", () => {
    const p = parameter(tensor([1, 2], { dtype: "float64" }));
    p.setGrad(tensor([0.5, -0.3], { dtype: "float64" }));
    const opt = new Adagrad([p], { weightDecay: 0.1, lrDecay: 0.01 });
    opt.step();
    p.setGrad(tensor([0.2, -0.1], { dtype: "float64" }));
    opt.step();
    const after = Array.from(getParamData(p, "Adagrad param"));
    expect(after[0]).toBeLessThan(1);
  });

  it("Nadam step with closure", () => {
    const p = parameter(tensor([1], { dtype: "float64" }));
    p.setGrad(tensor([0.1], { dtype: "float64" }));
    const opt = new Nadam([p]);
    const loss = opt.step(() => 1.23);
    expect(loss).toBe(1.23);
  });

  it("Nadam with weight decay", () => {
    const p = parameter(tensor([1, 2], { dtype: "float64" }));
    p.setGrad(tensor([0.5, -0.3], { dtype: "float64" }));
    const opt = new Nadam([p], { weightDecay: 0.1 });
    opt.step();
    const after = Array.from(getParamData(p, "Nadam param"));
    expect(after[0]).toBeLessThan(1);
  });

  it("RMSprop step with closure", () => {
    const p = parameter(tensor([1], { dtype: "float64" }));
    p.setGrad(tensor([0.1], { dtype: "float64" }));
    const opt = new RMSprop([p]);
    const loss = opt.step(() => 0.99);
    expect(loss).toBe(0.99);
  });

  it("RMSprop with weight decay", () => {
    const p = parameter(tensor([1, 2], { dtype: "float64" }));
    p.setGrad(tensor([0.5, -0.3], { dtype: "float64" }));
    const opt = new RMSprop([p], { weightDecay: 0.1, lr: 0.01 });
    opt.step();
    const after = Array.from(getParamData(p, "RMSprop param"));
    expect(after[0]).toBeLessThan(1);
  });
});

describe("HorizontalBar2D via Axes.barh", () => {
  it("should render barh as SVG", () => {
    const fig = figure({ width: 320, height: 240 });
    const ax = gca();
    ax.barh(tensor([0, 1, 2]), tensor([10, 20, 15]));
    const result = fig.renderSVG();
    expect(result.svg).toContain("<rect");
    expect(result.svg).toContain("svg");
  });

  it("should render barh as raster", async () => {
    const fig = figure({ width: 100, height: 80 });
    const ax = gca();
    ax.barh(tensor([0, 1]), tensor([5, 10]));
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
    expect(result.bytes.length).toBeGreaterThan(0);
  });

  it("should handle NaN values in barh", () => {
    const fig = figure({ width: 100, height: 80 });
    const ax = gca();
    ax.barh(tensor([0, Number.NaN, 2]), tensor([10, 20, Number.NaN]));
    const result = fig.renderSVG();
    expect(result.svg).toContain("svg");
  });

  it("should handle barh with custom colors", () => {
    const fig = figure({ width: 100, height: 80 });
    const ax = gca();
    ax.barh(tensor([0, 1]), tensor([5, 10]), {
      color: "#ff0000",
      edgecolor: "#00ff00",
      label: "test",
    });
    const result = fig.renderSVG();
    expect(result.svg).toContain("#ff0000");
  });
});

describe("Pie chart via Axes.pie", () => {
  it("should render pie as SVG", () => {
    const fig = figure({ width: 200, height: 200 });
    const ax = gca();
    ax.pie(tensor([30, 20, 50]));
    const result = fig.renderSVG();
    expect(result.svg).toContain("svg");
  });

  it("should render pie with labels", () => {
    const fig = figure({ width: 200, height: 200 });
    const ax = gca();
    ax.pie(tensor([30, 20, 50]), ["A", "B", "C"]);
    const result = fig.renderSVG();
    expect(result.svg).toContain("A");
  });

  it("should render pie as raster", async () => {
    const fig = figure({ width: 100, height: 100 });
    const ax = gca();
    ax.pie(tensor([40, 60]));
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
  });
});

describe("ContourF2D via Axes.contourf", () => {
  it("should render filled contour as SVG", () => {
    const fig = figure({ width: 200, height: 200 });
    const ax = gca();
    const x = tensor([0, 1, 2]);
    const y = tensor([0, 1, 2]);
    const z = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    ax.contourf(x, y, z);
    const result = fig.renderSVG();
    expect(result.svg).toContain("svg");
  });

  it("should render filled contour as raster", async () => {
    const fig = figure({ width: 100, height: 100 });
    const ax = gca();
    const x = tensor([0, 1, 2]);
    const y = tensor([0, 1, 2]);
    const z = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    ax.contourf(x, y, z);
    const result = await fig.renderPNG();
    expect(result.bytes).toBeInstanceOf(Uint8Array);
  });
});
