import { describe, expect, it } from "vitest";
import { tensor, zeros } from "../src/ndarray";
import { BatchNorm1d, LayerNorm } from "../src/nn/layers/normalization";
import type { Module } from "../src/nn/module/Module";

function getParam(module: Module, name: string) {
  for (const [n, p] of module.namedParameters()) {
    if (n === name) return p;
  }
  return undefined;
}

function getBuffer(module: Module, name: string) {
  for (const [n, p] of module.namedBuffers()) {
    if (n === name) return p;
  }
  return undefined;
}

describe("NN Normalization Coverage", () => {
  describe("BatchNorm1d", () => {
    it("should support affine=false", () => {
      const bn = new BatchNorm1d(2, { affine: false });
      const x = tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { dtype: "float32" }
      );
      const y = bn.forward(x);

      expect(y.shape).toEqual([2, 2]);
      // Parameters should not exist
      expect(getParam(bn, "weight")).toBeUndefined();
      expect(getParam(bn, "bias")).toBeUndefined();
    });

    it("should support trackRunningStats=false", () => {
      const bn = new BatchNorm1d(2, { trackRunningStats: false });
      expect(getBuffer(bn, "running_mean")).toBeUndefined();

      const x = tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { dtype: "float32" }
      );
      bn.train();
      const y1 = bn.forward(x);

      bn.eval();
      const y2 = bn.forward(x);

      // Without running stats, eval mode uses batch stats too
      expect(y1.data).toEqual(y2.data);
    });

    it("should handle 3D inputs (batch, features, spatial)", () => {
      // Input: [2, 2, 3] -> 2 samples, 2 features, length 3
      const x = zeros([2, 2, 3], { dtype: "float32" });
      const bn = new BatchNorm1d(2);
      const y = bn.forward(x);
      expect(y.shape).toEqual([2, 2, 3]);
    });

    it("should validate constructor parameters", () => {
      expect(() => new BatchNorm1d(0)).toThrow(/positive integer/);
      expect(() => new BatchNorm1d(1, { eps: 0 })).toThrow(/eps must be/);
      expect(() => new BatchNorm1d(1, { momentum: 2 })).toThrow(/momentum/);
    });

    it("should throw on dimension mismatch", () => {
      const bn = new BatchNorm1d(2);
      const x = tensor([[1, 2, 3]], { dtype: "float32" }); // 3 features
      expect(() => bn.forward(x)).toThrow(/Expected 2 features/);
    });
  });

  describe("LayerNorm", () => {
    it("should support elementwiseAffine=false", () => {
      const ln = new LayerNorm([2], { elementwiseAffine: false });
      const x = tensor(
        [
          [1, 2],
          [3, 4],
        ],
        { dtype: "float32" }
      );
      const y = ln.forward(x);

      expect(y.shape).toEqual([2, 2]);
      expect(getParam(ln, "weight")).toBeUndefined();
    });

    it("should validate normalizedShape", () => {
      expect(() => new LayerNorm([])).toThrow(/at least one dimension/);
    });
  });
});
