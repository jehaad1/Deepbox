import { describe, expect, it } from "vitest";
import { Tensor, tensor } from "../src/ndarray";
import { AvgPool3d, Conv3d, ConvTranspose1d, MaxPool3d } from "../src/nn";

describe("ConvTranspose1d", () => {
  it("constructs with valid params", () => {
    const layer = new ConvTranspose1d(4, 8, 3);
    expect(layer).toBeDefined();
    expect(layer.weight).toBeDefined();
    expect(layer.weight.shape).toEqual([4, 8, 3]);
  });

  it("throws on invalid inChannels", () => {
    expect(() => new ConvTranspose1d(0, 8, 3)).toThrow();
    expect(() => new ConvTranspose1d(-1, 8, 3)).toThrow();
  });

  it("throws on invalid outChannels", () => {
    expect(() => new ConvTranspose1d(4, 0, 3)).toThrow();
  });

  it("throws on invalid kernelSize", () => {
    expect(() => new ConvTranspose1d(4, 8, 0)).toThrow();
  });

  it("produces correct output shape (stride=1, no padding)", () => {
    const layer = new ConvTranspose1d(2, 3, 3);
    // Input: (1, 2, 4) -> outL = (4-1)*1 - 0 + 3 = 6
    const x = tensor([
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ],
    ]);
    const out = layer.forward(x);
    expect(out.shape).toEqual([1, 3, 6]);
  });

  it("produces correct output shape with stride=2", () => {
    const layer = new ConvTranspose1d(2, 3, 3, { stride: 2 });
    // Input: (1, 2, 4) -> outL = (4-1)*2 - 0 + 3 = 9
    const x = tensor([
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ],
    ]);
    const out = layer.forward(x);
    expect(out.shape).toEqual([1, 3, 9]);
  });

  it("produces correct output shape with padding", () => {
    const layer = new ConvTranspose1d(2, 3, 3, { stride: 2, padding: 1 });
    // outL = (4-1)*2 - 2*1 + 3 = 7
    const x = tensor([
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ],
    ]);
    const out = layer.forward(x);
    expect(out.shape).toEqual([1, 3, 7]);
  });

  it("works without bias", () => {
    const layer = new ConvTranspose1d(2, 3, 3, { bias: false });
    const x = tensor([
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ],
    ]);
    const out = layer.forward(x);
    expect(out.shape).toEqual([1, 3, 6]);
  });

  it("rejects wrong ndim input", () => {
    const layer = new ConvTranspose1d(2, 3, 3);
    const x = tensor([[1, 2, 3, 4]]);
    expect(() => layer.forward(x)).toThrow(/3D/);
  });

  it("rejects wrong channel count", () => {
    const layer = new ConvTranspose1d(2, 3, 3);
    const x = tensor([
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
      ],
    ]);
    expect(() => layer.forward(x)).toThrow(/channels/);
  });

  it("toString includes layer info", () => {
    const layer = new ConvTranspose1d(4, 8, 3, { stride: 2 });
    expect(layer.toString()).toContain("ConvTranspose1d");
  });
});

describe("Conv3d", () => {
  it("constructs with scalar kernel size", () => {
    const layer = new Conv3d(1, 4, 3);
    expect(layer.weight.shape).toEqual([4, 1, 3, 3, 3]);
  });

  it("constructs with tuple kernel size", () => {
    const layer = new Conv3d(1, 4, [3, 3, 3]);
    expect(layer.weight.shape).toEqual([4, 1, 3, 3, 3]);
  });

  it("throws on invalid inChannels", () => {
    expect(() => new Conv3d(0, 4, 3)).toThrow();
  });

  it("throws on invalid outChannels", () => {
    expect(() => new Conv3d(1, 0, 3)).toThrow();
  });

  it("produces correct output shape", () => {
    const layer = new Conv3d(1, 2, 2);
    // Input: (1, 1, 4, 4, 4) -> out = floor((4+0-2)/1)+1 = 3 per dim
    const x = Tensor.fromTypedArray({
      data: new Float64Array(64).fill(1),
      shape: [1, 1, 4, 4, 4],
      dtype: "float64",
      device: "cpu",
    });
    const out = layer.forward(x);
    expect(out.shape).toEqual([1, 2, 3, 3, 3]);
  });

  it("produces correct output shape with stride and padding", () => {
    const layer = new Conv3d(1, 2, 3, { stride: 2, padding: 1 });
    // Input: (1, 1, 6, 6, 6) -> out = floor((6+2-3)/2)+1 = 3
    const x = Tensor.fromTypedArray({
      data: new Float64Array(216).fill(1),
      shape: [1, 1, 6, 6, 6],
      dtype: "float64",
      device: "cpu",
    });
    const out = layer.forward(x);
    expect(out.shape).toEqual([1, 2, 3, 3, 3]);
  });

  it("works without bias", () => {
    const layer = new Conv3d(1, 2, 2, { bias: false });
    const x = Tensor.fromTypedArray({
      data: new Float64Array(64).fill(1),
      shape: [1, 1, 4, 4, 4],
      dtype: "float64",
      device: "cpu",
    });
    const out = layer.forward(x);
    expect(out.shape).toEqual([1, 2, 3, 3, 3]);
  });

  it("rejects wrong ndim input", () => {
    const layer = new Conv3d(1, 2, 2);
    const x = tensor([
      [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ],
    ]);
    expect(() => layer.forward(x)).toThrow(/5D/);
  });

  it("rejects wrong channel count", () => {
    const layer = new Conv3d(1, 2, 2);
    const x = Tensor.fromTypedArray({
      data: new Float64Array(128).fill(1),
      shape: [1, 2, 4, 4, 4],
      dtype: "float64",
      device: "cpu",
    });
    expect(() => layer.forward(x)).toThrow(/channels/);
  });

  it("toString includes layer info", () => {
    const layer = new Conv3d(1, 4, 3);
    expect(layer.toString()).toContain("Conv3d");
  });
});

describe("MaxPool3d", () => {
  it("constructs with scalar kernel size", () => {
    const pool = new MaxPool3d(2);
    expect(pool).toBeDefined();
  });

  it("constructs with tuple kernel size", () => {
    const pool = new MaxPool3d([2, 2, 2]);
    expect(pool).toBeDefined();
  });

  it("throws on invalid kernel size", () => {
    expect(() => new MaxPool3d(0)).toThrow();
    expect(() => new MaxPool3d(-1)).toThrow();
  });

  it("produces correct output shape", () => {
    const pool = new MaxPool3d(2);
    // Input: (1, 1, 4, 4, 4) -> out = 4/2 = 2 per dim
    const x = Tensor.fromTypedArray({
      data: Float64Array.from({ length: 64 }, (_, i) => i),
      shape: [1, 1, 4, 4, 4],
      dtype: "float64",
      device: "cpu",
    });
    const out = pool.forward(x);
    expect(out.shape).toEqual([1, 1, 2, 2, 2]);
  });

  it("computes max correctly", () => {
    const pool = new MaxPool3d(2);
    const x = Tensor.fromTypedArray({
      data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]),
      shape: [1, 1, 2, 2, 2],
      dtype: "float64",
      device: "cpu",
    });
    const out = pool.forward(x);
    expect(out.shape).toEqual([1, 1, 1, 1, 1]);
    // Max of [1..8] = 8
    expect(out.toArray()).toEqual([[[[[8]]]]]);
  });

  it("handles stride different from kernel", () => {
    const pool = new MaxPool3d(2, { stride: 1 });
    const x = Tensor.fromTypedArray({
      data: Float64Array.from({ length: 27 }, (_, i) => i),
      shape: [1, 1, 3, 3, 3],
      dtype: "float64",
      device: "cpu",
    });
    const out = pool.forward(x);
    // out = floor((3-2)/1)+1 = 2
    expect(out.shape).toEqual([1, 1, 2, 2, 2]);
  });

  it("rejects wrong ndim input", () => {
    const pool = new MaxPool3d(2);
    const x = tensor([
      [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ],
    ]);
    expect(() => pool.forward(x)).toThrow(/5D/);
  });

  it("toString includes layer info", () => {
    const pool = new MaxPool3d(2);
    expect(pool.toString()).toContain("MaxPool3d");
  });
});

describe("AvgPool3d", () => {
  it("constructs with scalar kernel size", () => {
    const pool = new AvgPool3d(2);
    expect(pool).toBeDefined();
  });

  it("throws on invalid kernel size", () => {
    expect(() => new AvgPool3d(0)).toThrow();
  });

  it("produces correct output shape", () => {
    const pool = new AvgPool3d(2);
    const x = Tensor.fromTypedArray({
      data: Float64Array.from({ length: 64 }, (_, i) => i),
      shape: [1, 1, 4, 4, 4],
      dtype: "float64",
      device: "cpu",
    });
    const out = pool.forward(x);
    expect(out.shape).toEqual([1, 1, 2, 2, 2]);
  });

  it("computes average correctly", () => {
    const pool = new AvgPool3d(2);
    const x = Tensor.fromTypedArray({
      data: new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]),
      shape: [1, 1, 2, 2, 2],
      dtype: "float64",
      device: "cpu",
    });
    const out = pool.forward(x);
    expect(out.shape).toEqual([1, 1, 1, 1, 1]);
    // avg of [1..8] = 4.5
    expect(out.toArray()).toEqual([[[[[4.5]]]]]);
  });

  it("handles multiple channels", () => {
    const pool = new AvgPool3d(2);
    const x = Tensor.fromTypedArray({
      data: new Float64Array(128).fill(1),
      shape: [1, 2, 4, 4, 4],
      dtype: "float64",
      device: "cpu",
    });
    const out = pool.forward(x);
    expect(out.shape).toEqual([1, 2, 2, 2, 2]);
  });

  it("rejects wrong ndim input", () => {
    const pool = new AvgPool3d(2);
    const x = tensor([[1, 2, 3]]);
    expect(() => pool.forward(x)).toThrow(/5D/);
  });

  it("toString includes layer info", () => {
    const pool = new AvgPool3d(2);
    expect(pool.toString()).toContain("AvgPool3d");
  });
});
