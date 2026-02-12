import { describe, expect, it } from "vitest";
import { tensor, transpose } from "../src/ndarray";
import { BatchNorm1d, LayerNorm } from "../src/nn/layers/normalization";
import { expectNumberArray, expectNumberArray2D, expectNumberArray3D } from "./nn-test-utils";

describe("deepbox/nn - Normalization", () => {
  it("BatchNorm1d normalizes in training mode", () => {
    const bn = new BatchNorm1d(3);
    bn.train();
    const x = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const y = bn.forward(x);
    expect(y.shape).toEqual([2, 3]);

    const arr = expectNumberArray2D(y.tensor.toArray(), "BatchNorm1d output");
    const means = [0, 0, 0];
    for (const row of arr) {
      for (let j = 0; j < 3; j++) {
        means[j] += row[j] ?? 0;
      }
    }
    for (let j = 0; j < 3; j++) {
      means[j] /= 2;
      expect(means[j]).toBeCloseTo(0, 6);
    }
  });

  it("BatchNorm1d validates inputs and options", () => {
    expect(() => new BatchNorm1d(0)).toThrow();
    expect(() => new BatchNorm1d(2, { eps: 0 })).toThrow();
    expect(() => new BatchNorm1d(2, { momentum: -0.1 })).toThrow();
    const bn = new BatchNorm1d(2);
    const bad = tensor([["a", "b"]]);
    expect(() => bn.forward(bad)).toThrow();
  });

  it("BatchNorm1d uses running stats in eval mode", () => {
    const bn = new BatchNorm1d(2);
    const x = tensor([
      [1, 2],
      [3, 4],
    ]);
    bn.train();
    bn.forward(x);
    bn.eval();
    const y = bn.forward(x);
    expect(y.shape).toEqual([2, 2]);
  });

  it("BatchNorm1d updates running stats buffers", () => {
    const bn = new BatchNorm1d(2);
    bn.train();
    const x = tensor([
      [1, 2],
      [3, 4],
    ]);
    bn.forward(x);
    const state = bn.stateDict();
    const runningMean = state.buffers["running_mean"];
    expect(runningMean).toBeDefined();
    if (runningMean) {
      expect(runningMean.data[0]).not.toBe(0);
      expect(runningMean.data[1]).not.toBe(0);
    }
  });

  it("BatchNorm1d uses batch stats when trackRunningStats=false", () => {
    const bn = new BatchNorm1d(2, { trackRunningStats: false });
    const x = tensor([
      [1, 2],
      [3, 4],
    ]);
    bn.eval();
    const y = bn.forward(x);
    const arr = expectNumberArray2D(y.tensor.toArray(), "BatchNorm1d output");
    const means = [0, 0];
    for (const row of arr) {
      means[0] += row[0] ?? 0;
      means[1] += row[1] ?? 0;
    }
    means[0] /= arr.length;
    means[1] /= arr.length;
    expect(means[0]).toBeCloseTo(0, 6);
    expect(means[1]).toBeCloseTo(0, 6);
  });

  it("BatchNorm1d handles non-contiguous input", () => {
    const bn = new BatchNorm1d(2);
    bn.train();
    const base = tensor([
      [1, 2],
      [3, 4],
    ]);
    const x = transpose(base, [1, 0]);
    const y = bn.forward(x);
    expect(y.shape).toEqual([2, 2]);
  });

  it("LayerNorm normalizes per sample", () => {
    const ln = new LayerNorm(3, { elementwiseAffine: false });
    const x = tensor([
      [1, 2, 3],
      [2, 4, 6],
    ]);
    const y = ln.forward(x);
    const rows = expectNumberArray2D(y.tensor.toArray(), "LayerNorm output");
    for (const row of rows) {
      const mean = (row[0] + row[1] + row[2]) / 3;
      const varVal = ((row[0] - mean) ** 2 + (row[1] - mean) ** 2 + (row[2] - mean) ** 2) / 3;
      expect(mean).toBeCloseTo(0, 6);
      expect(varVal).toBeCloseTo(1, 4);
    }
  });

  it("LayerNorm supports 1D input", () => {
    const ln = new LayerNorm(3, { elementwiseAffine: false });
    const x = tensor([1, 2, 3]);
    const y = ln.forward(x);
    expect(y.shape).toEqual([3]);
    const arr = expectNumberArray(y.tensor.toArray(), "LayerNorm output");
    const mean = (arr[0] + arr[1] + arr[2]) / 3;
    const varVal = ((arr[0] - mean) ** 2 + (arr[1] - mean) ** 2 + (arr[2] - mean) ** 2) / 3;
    expect(mean).toBeCloseTo(0, 6);
    expect(varVal).toBeCloseTo(1, 4);
  });

  it("LayerNorm normalizes trailing dimensions for 3D input", () => {
    const ln = new LayerNorm(2, { elementwiseAffine: false });
    const x = tensor([
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ]);
    const y = ln.forward(x);
    expect(y.shape).toEqual([2, 2, 2]);
    const arr = expectNumberArray3D(y.tensor.toArray(), "LayerNorm output");
    for (const batch of arr) {
      for (const row of batch) {
        const mean = (row[0] + row[1]) / 2;
        const varVal = ((row[0] - mean) ** 2 + (row[1] - mean) ** 2) / 2;
        expect(mean).toBeCloseTo(0, 6);
        expect(varVal).toBeCloseTo(1, 4);
      }
    }
  });

  it("LayerNorm validates normalizedShape and input shape", () => {
    expect(() => new LayerNorm(0)).toThrow();
    expect(() => new LayerNorm([2, -1])).toThrow();
    const ln = new LayerNorm([2, 2]);
    const x = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(() => ln.forward(x)).toThrow();
    const s = tensor([["a", "b"]]);
    expect(() => ln.forward(s)).toThrow();
  });

  it("LayerNorm handles non-contiguous input", () => {
    const ln = new LayerNorm(2, { elementwiseAffine: false });
    const base = tensor([
      [1, 2],
      [3, 4],
    ]);
    const x = transpose(base, [1, 0]);
    const y = ln.forward(x);
    expect(y.shape).toEqual([2, 2]);
  });
});
