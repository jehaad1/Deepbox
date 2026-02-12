import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { Dropout } from "../src/nn";

describe("deepbox/nn - Dropout Layer", () => {
  describe("constructor", () => {
    it("should create Dropout with default probability", () => {
      const dropout = new Dropout();
      expect(dropout).toBeDefined();
      expect(dropout.dropoutRate).toBe(0.5);
      expect(dropout.toString()).toBe("Dropout(p=0.5)");
    });

    it("should create Dropout with custom probability", () => {
      const dropout = new Dropout(0.3);
      expect(dropout.dropoutRate).toBe(0.3);
      expect(dropout.toString()).toBe("Dropout(p=0.3)");
    });

    it("should throw error for invalid probability < 0", () => {
      expect(() => new Dropout(-0.1)).toThrow("Dropout probability must be in [0, 1)");
    });

    it("should throw error for invalid probability >= 1", () => {
      expect(() => new Dropout(1.0)).toThrow("Dropout probability must be in [0, 1)");
      expect(() => new Dropout(1.5)).toThrow("Dropout probability must be in [0, 1)");
    });

    it("should throw error for non-finite probability", () => {
      expect(() => new Dropout(Number.NaN)).toThrow();
      expect(() => new Dropout(Number.POSITIVE_INFINITY)).toThrow();
    });

    it("should accept probability of 0", () => {
      const dropout = new Dropout(0);
      expect(dropout.dropoutRate).toBe(0);
    });

    it("should accept probability close to 1", () => {
      const dropout = new Dropout(0.99);
      expect(dropout.dropoutRate).toBe(0.99);
    });
  });

  describe("forward - training mode", () => {
    it("should be in training mode by default", () => {
      const dropout = new Dropout(0.5);
      expect(dropout.training).toBe(true);
    });

    it("should preserve input shape", () => {
      const dropout = new Dropout(0.5);
      const input = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const output = dropout.forward(input);

      expect(output.shape).toEqual([2, 3]);
    });

    it("should zero out some elements during training", () => {
      const dropout = new Dropout(0.5);
      dropout.train();

      const input = tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
      const output = dropout.forward(input);

      // With p=0.5, we expect roughly half to be zero
      // But due to randomness, we just check that some are zero
      let zeroCount = 0;
      for (let i = 0; i < output.size; i++) {
        if (output.data[i] === 0) zeroCount++;
      }

      // At least one should be zeroed (statistically very likely)
      expect(zeroCount).toBeGreaterThan(0);
    });

    it("should scale non-zero elements by 1/(1-p)", () => {
      const dropout = new Dropout(0.5);
      dropout.train();

      const input = tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]);
      const output = dropout.forward(input);

      // Non-zero elements should be scaled by 1/(1-0.5) = 2
      for (let i = 0; i < output.size; i++) {
        const val = Number(output.data[i]);
        if (val !== 0) {
          expect(val).toBeCloseTo(4, 1); // 2 * 2 = 4
        }
      }
    });

    it("should handle 1D tensors", () => {
      const dropout = new Dropout(0.3);
      const input = tensor([1, 2, 3, 4, 5]);
      const output = dropout.forward(input);

      expect(output.shape).toEqual([5]);
    });

    it("should handle 3D tensors", () => {
      const dropout = new Dropout(0.2);
      const input = tensor([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      const output = dropout.forward(input);

      expect(output.shape).toEqual([2, 2, 2]);
    });

    it("should return input unchanged when p=0", () => {
      const dropout = new Dropout(0);
      dropout.train();

      const input = tensor([1, 2, 3, 4, 5]);
      const output = dropout.forward(input);

      for (let i = 0; i < output.size; i++) {
        expect(output.data[i]).toBe(input.data[i]);
      }
    });

    it("should reject string tensors", () => {
      const dropout = new Dropout(0.5);
      const input = tensor([["a", "b"]]);
      expect(() => dropout.forward(input)).toThrow();
    });
  });

  describe("forward - evaluation mode", () => {
    it("should return input unchanged in eval mode", () => {
      const dropout = new Dropout(0.5);
      dropout.eval();

      const input = tensor([1, 2, 3, 4, 5]);
      const output = dropout.forward(input);

      // In eval mode, dropout is disabled
      for (let i = 0; i < output.size; i++) {
        expect(output.data[i]).toBe(input.data[i]);
      }
    });

    it("should preserve all values in eval mode", () => {
      const dropout = new Dropout(0.9); // High dropout rate
      dropout.eval();

      const input = tensor([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const output = dropout.forward(input);

      // All values should be preserved
      for (let i = 0; i < output.size; i++) {
        expect(output.data[i]).toBe(input.data[i]);
      }
    });

    it("should be deterministic in eval mode", () => {
      const dropout = new Dropout(0.5);
      dropout.eval();

      const input = tensor([1, 2, 3, 4, 5]);
      const output1 = dropout.forward(input);
      const output2 = dropout.forward(input);

      // Both outputs should be identical
      for (let i = 0; i < output1.size; i++) {
        expect(output1.data[i]).toBe(output2.data[i]);
      }
    });
  });

  describe("training mode switching", () => {
    it("should switch between training and eval modes", () => {
      const dropout = new Dropout(0.5);

      expect(dropout.training).toBe(true);

      dropout.eval();
      expect(dropout.training).toBe(false);

      dropout.train();
      expect(dropout.training).toBe(true);
    });

    it("should behave differently in train vs eval", () => {
      const dropout = new Dropout(0.8); // High dropout
      const input = tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);

      // Training mode: some elements should be zero
      dropout.train();
      const trainOutput = dropout.forward(input);
      let trainZeros = 0;
      for (let i = 0; i < trainOutput.size; i++) {
        if (trainOutput.data[i] === 0) trainZeros++;
      }

      // Eval mode: no elements should be zero
      dropout.eval();
      const evalOutput = dropout.forward(input);
      let evalZeros = 0;
      for (let i = 0; i < evalOutput.size; i++) {
        if (evalOutput.data[i] === 0) evalZeros++;
      }

      expect(trainZeros).toBeGreaterThan(0);
      expect(evalZeros).toBe(0);
    });
  });

  describe("edge cases", () => {
    it("should handle very small tensors", () => {
      const dropout = new Dropout(0.5);
      const input = tensor([42]);
      const output = dropout.forward(input);

      expect(output.shape).toEqual([1]);
    });

    it("should handle large tensors", () => {
      const dropout = new Dropout(0.3);
      const input = tensor(new Array(1000).fill(1));
      const output = dropout.forward(input);

      expect(output.shape).toEqual([1000]);
    });

    it("should work with different dtypes", () => {
      const dropout = new Dropout(0.5);
      const input = tensor([1, 2, 3], { dtype: "float64" });
      const output = dropout.forward(input);

      expect(output.shape).toEqual([3]);
    });
  });
});
