import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { AvgPool2d, Conv1d, Conv2d, MaxPool2d } from "../src/nn/layers/conv";
import { expectFloatTypedArray } from "./nn-test-utils";

describe("deepbox/nn - Convolution and Pooling", () => {
  it("Conv1d computes expected output with known kernel", () => {
    const conv = new Conv1d(1, 1, 3, { bias: false });
    const w = conv.weight;
    const wData = expectFloatTypedArray(w.data, "Conv1d weight");
    wData[0] = 1;
    wData[1] = 0;
    wData[2] = -1;

    const x = tensor([[[1, 2, 3, 4, 5]]], { dtype: "float32" });
    const out = conv.forward(x);
    expect(out.shape).toEqual([1, 1, 3]);
    expect(out.tensor.toArray()).toEqual([[[-2, -2, -2]]]);
  });

  it("Conv1d validates input shape", () => {
    const conv = new Conv1d(1, 1, 3);
    const x = tensor([[1, 2, 3]]);
    expect(() => conv.forward(x)).toThrow();
  });

  it("Conv layers reject string inputs", () => {
    const conv1 = new Conv1d(1, 1, 3);
    const conv2 = new Conv2d(1, 1, 1);
    const x1 = tensor([[["a", "b", "c"]]]);
    const x2 = tensor([[[["a"]]]]);
    expect(() => conv1.forward(x1)).toThrow();
    expect(() => conv2.forward(x2)).toThrow();
  });

  it("Conv1d validates parameters", () => {
    expect(() => new Conv1d(0, 1, 3)).toThrow();
    expect(() => new Conv1d(1, 0, 3)).toThrow();
    expect(() => new Conv1d(1, 1, 0)).toThrow();
    expect(() => new Conv1d(1, 1, 3, { stride: 0 })).toThrow();
    expect(() => new Conv1d(1, 1, 3, { padding: -1 })).toThrow();
  });

  it("Conv2d validates parameters", () => {
    expect(() => new Conv2d(0, 1, 3)).toThrow();
    expect(() => new Conv2d(1, 0, 3)).toThrow();
    expect(() => new Conv2d(1, 1, 0)).toThrow();
    expect(() => new Conv2d(1, 1, 3, { stride: 0 })).toThrow();
    expect(() => new Conv2d(1, 1, 3, { padding: -1 })).toThrow();
  });

  it("Conv2d computes expected output with ones kernel", () => {
    const conv = new Conv2d(1, 1, 2, { bias: false });
    const w = conv.weight;
    const wData = expectFloatTypedArray(w.data, "Conv2d weight");
    for (let i = 0; i < wData.length; i++) wData[i] = 1;

    const x = tensor([
      [
        [
          [1, 2],
          [3, 4],
        ],
      ],
    ]);

    const out = conv.forward(x);
    expect(out.shape).toEqual([1, 1, 1, 1]);
    expect(out.tensor.toArray()).toEqual([[[[10]]]]);
  });

  it("MaxPool2d and AvgPool2d compute pooled outputs", () => {
    const x = tensor([
      [
        [
          [1, 2],
          [3, 4],
        ],
      ],
    ]);

    const maxPool = new MaxPool2d(2);
    const maxOut = maxPool.forward(x);
    expect(maxOut.shape).toEqual([1, 1, 1, 1]);
    expect(maxOut.tensor.toArray()).toEqual([[[[4]]]]);

    const avgPool = new AvgPool2d(2);
    const avgOut = avgPool.forward(x);
    expect(avgOut.shape).toEqual([1, 1, 1, 1]);
    expect(avgOut.tensor.toArray()).toEqual([[[[2.5]]]]);
  });

  it("Pool layers validate parameters", () => {
    expect(() => new MaxPool2d(0)).toThrow();
    expect(() => new MaxPool2d(2, { stride: 0 })).toThrow();
    expect(() => new MaxPool2d(2, { padding: -1 })).toThrow();
    expect(() => new AvgPool2d(0)).toThrow();
    expect(() => new AvgPool2d(2, { stride: 0 })).toThrow();
    expect(() => new AvgPool2d(2, { padding: -1 })).toThrow();
  });

  it("Pool layers reject string inputs", () => {
    const maxPool = new MaxPool2d(2);
    const avgPool = new AvgPool2d(2);
    const x = tensor([
      [
        [
          ["a", "b"],
          ["c", "d"],
        ],
      ],
    ]);
    expect(() => maxPool.forward(x)).toThrow();
    expect(() => avgPool.forward(x)).toThrow();
  });
});
