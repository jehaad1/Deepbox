import { describe, expect, it } from "vitest";
import { GradTensor, noGrad, parameter, type Tensor, tensor } from "../src/ndarray";
import { numData } from "./_helpers";

/** Asserts that `gt.grad` is not null and returns it for type-safe access. */
function assertGrad(gt: GradTensor): Tensor {
  const g = gt.grad;
  expect(g).not.toBeNull();
  if (g === null) throw new Error("assertGrad: grad is null");

  return g;
}

describe("deepbox/ndarray - Autograd Comprehensive Tests", () => {
  describe("GradTensor creation", () => {
    it("should create GradTensor from tensor", () => {
      const t = tensor([1, 2, 3]);
      const gt = GradTensor.fromTensor(t, { requiresGrad: true });
      expect(gt.requiresGrad).toBe(true);
      expect(gt.shape).toEqual([3]);
    });

    it("should create GradTensor without grad tracking", () => {
      const t = tensor([1, 2, 3]);
      const gt = GradTensor.fromTensor(t, { requiresGrad: false });
      expect(gt.requiresGrad).toBe(false);
    });

    it("should create scalar GradTensor", () => {
      const gt = GradTensor.scalar(5, { requiresGrad: true });
      expect(gt.shape).toEqual([]);
      expect(Number(gt.data[0])).toBe(5);
    });

    it("should create parameter from tensor", () => {
      const t = tensor([1, 2, 3]);
      const p = parameter(t);
      expect(p.requiresGrad).toBe(true);
    });

    it("should create parameter from number", () => {
      const p = parameter(5);
      expect(p.requiresGrad).toBe(true);
      expect(Number(p.data[0])).toBe(5);
    });

    it("should create parameter from array", () => {
      const p = parameter([1, 2, 3]);
      expect(p.requiresGrad).toBe(true);
      expect(p.shape).toEqual([3]);
    });

    it("should throw on dtype mismatch", () => {
      const t = tensor([1, 2, 3], { dtype: "float32" });
      expect(() => GradTensor.fromTensor(t, { dtype: "float64" })).toThrow("dtype mismatch");
    });
  });

  describe("gradient computation - addition", () => {
    it("should compute gradients for simple addition", () => {
      const a = parameter(tensor([1, 2, 3]));
      const b = parameter(tensor([4, 5, 6]));
      const c = a.add(b);
      c.backward(tensor([1, 1, 1]));
      expect(numData(assertGrad(a))).toEqual([1, 1, 1]);
      expect(numData(assertGrad(b))).toEqual([1, 1, 1]);
    });

    it("should accumulate gradients", () => {
      const a = parameter(tensor([1, 2]));
      const b = parameter(tensor([3, 4]));
      const c = a.add(b);
      const d = a.add(c);
      d.backward(tensor([1, 1]));
      expect(numData(assertGrad(a))).toEqual([2, 2]);
    });

    it("should handle scalar addition", () => {
      const a = parameter(5);
      const b = parameter(10);
      const c = a.add(b);
      c.backward();
      expect(Number(assertGrad(a).data[0])).toBe(1);
      expect(Number(assertGrad(b).data[0])).toBe(1);
    });

    it("reduces broadcasted gradients for addition", () => {
      const a = parameter(
        tensor([
          [1, 2, 3],
          [4, 5, 6],
        ])
      );
      const b = parameter(tensor([1, 2, 3]));
      const c = a.add(b);
      c.backward(
        tensor([
          [1, 1, 1],
          [1, 1, 1],
        ])
      );
      expect(assertGrad(a).toArray()).toEqual([
        [1, 1, 1],
        [1, 1, 1],
      ]);
      expect(assertGrad(b).toArray()).toEqual([2, 2, 2]);
    });

    it("should not track gradients when requiresGrad=false", () => {
      const a = GradTensor.fromTensor(tensor([1, 2, 3]), {
        requiresGrad: false,
      });
      const b = parameter(tensor([4, 5, 6]));
      const c = a.add(b);
      c.backward(tensor([1, 1, 1]));
      expect(a.grad).toBeNull();
    });

    it("preserves dtype when broadcast and non-broadcast paths mix", () => {
      const a = parameter(
        tensor(
          [
            [1, 2, 3],
            [4, 5, 6],
          ],
          { dtype: "float32" }
        )
      );
      const b = parameter(tensor([1, 2, 3], { dtype: "float32" }));
      const c = a.add(b);
      const d = b.mul(b);
      const e = c.add(d);
      e.backward(
        tensor(
          [
            [1, 1, 1],
            [1, 1, 1],
          ],
          { dtype: "float32" }
        )
      );
      expect(assertGrad(b).dtype).toBe("float32");
      expect(assertGrad(b).toArray()).toEqual([6, 10, 14]);
    });
  });

  describe("gradient computation - subtraction", () => {
    it("should compute gradients for subtraction", () => {
      const a = parameter(tensor([5, 6, 7]));
      const b = parameter(tensor([1, 2, 3]));
      const c = a.sub(b);
      c.backward(tensor([1, 1, 1]));
      expect(numData(assertGrad(a))).toEqual([1, 1, 1]);
      expect(numData(assertGrad(b))).toEqual([-1, -1, -1]);
    });

    it("should handle negative gradients", () => {
      const a = parameter(tensor([1, 2, 3]));
      const b = parameter(tensor([4, 5, 6]));
      const c = a.sub(b);
      c.backward(tensor([2, 2, 2]));
      expect(numData(assertGrad(a))).toEqual([2, 2, 2]);
      expect(numData(assertGrad(b))).toEqual([-2, -2, -2]);
    });
  });

  describe("gradient computation - multiplication", () => {
    it("should compute gradients for multiplication", () => {
      const a = parameter(tensor([2, 3, 4]));
      const b = parameter(tensor([5, 6, 7]));
      const c = a.mul(b);
      c.backward(tensor([1, 1, 1]));
      expect(numData(assertGrad(a))).toEqual([5, 6, 7]);
      expect(numData(assertGrad(b))).toEqual([2, 3, 4]);
    });

    it("should handle product rule", () => {
      const x = parameter(tensor([2]));
      const y = x.mul(x);
      y.backward();
      expect(Number(assertGrad(x).data[0])).toBe(4);
    });

    it("should compute gradients for chain of multiplications", () => {
      const a = parameter(tensor([2]));
      const b = parameter(tensor([3]));
      const c = parameter(tensor([4]));
      const d = a.mul(b).mul(c);
      d.backward();
      expect(Number(assertGrad(a).data[0])).toBe(12);
      expect(Number(assertGrad(b).data[0])).toBe(8);
      expect(Number(assertGrad(c).data[0])).toBe(6);
    });

    it("reduces broadcasted gradients for multiplication", () => {
      const a = parameter(
        tensor([
          [1, 2, 3],
          [4, 5, 6],
        ])
      );
      const b = parameter(tensor([10, 20, 30]));
      const c = a.mul(b);
      c.backward();
      expect(assertGrad(a).toArray()).toEqual([
        [10, 20, 30],
        [10, 20, 30],
      ]);
      expect(assertGrad(b).toArray()).toEqual([5, 7, 9]);
    });
  });

  describe("gradient computation - negation", () => {
    it("should compute gradients for negation", () => {
      const a = parameter(tensor([1, 2, 3]));
      const b = a.neg();
      b.backward(tensor([1, 1, 1]));
      expect(numData(assertGrad(a))).toEqual([-1, -1, -1]);
    });

    it("should handle double negation", () => {
      const a = parameter(tensor([5]));
      const b = a.neg().neg();
      b.backward();
      expect(Number(assertGrad(a).data[0])).toBe(1);
    });
  });

  describe("gradient computation - sum", () => {
    it("should compute gradients for sum", () => {
      const a = parameter(tensor([1, 2, 3, 4]));
      const b = a.sum();
      b.backward();
      expect(numData(assertGrad(a))).toEqual([1, 1, 1, 1]);
    });

    it("should compute gradients for sum with upstream gradient", () => {
      const a = parameter(tensor([1, 2, 3]));
      const b = a.sum();
      b.backward(tensor(5));
      expect(numData(assertGrad(a))).toEqual([5, 5, 5]);
    });

    it("should compute gradients for sum along axis", () => {
      const a = parameter(
        tensor([
          [1, 2],
          [3, 4],
        ])
      );
      const b = a.sum(0, false);
      b.backward(tensor([1, 1]));
      expect(numData(assertGrad(a))).toEqual([1, 1, 1, 1]);
    });

    it("should compute gradients for sum with keepdims", () => {
      const a = parameter(
        tensor([
          [1, 2],
          [3, 4],
        ])
      );
      const b = a.sum(0, true);
      b.backward(tensor([[1, 1]]));
      expect(numData(assertGrad(a))).toEqual([1, 1, 1, 1]);
    });
  });

  describe("gradient computation - mean", () => {
    it("computes mean on float32 inputs without dtype mismatch", () => {
      const a = parameter(tensor([1, 2, 3, 4], { dtype: "float32" }));
      const m = a.mean();
      m.backward();
      const gradArr = numData(assertGrad(a));
      expect(gradArr).toHaveLength(4);
      for (const v of gradArr) {
        expect(v).toBeCloseTo(0.25, 6);
      }
    });
  });

  describe("complex computational graphs", () => {
    it("should compute gradients for y = x^2 + 2x + 1", () => {
      const x = parameter(tensor([3]));
      const x_squared = x.mul(x);
      const two_x = x.add(x);
      const y = x_squared.add(two_x).add(GradTensor.scalar(1, { requiresGrad: false }));
      y.backward();
      expect(Number(assertGrad(x).data[0])).toBe(8);
    });

    it("should compute gradients for nested operations", () => {
      const a = parameter(tensor([2]));
      const b = parameter(tensor([3]));
      const c = a.add(b);
      const d = c.mul(a);
      const e = d.sub(b);
      e.backward();
      expect(Number(assertGrad(a).data[0])).toBe(7);
      expect(Number(assertGrad(b).data[0])).toBe(1);
    });

    it("should handle diamond-shaped graph", () => {
      const x = parameter(tensor([2]));
      const y = x.add(x);
      const z = x.mul(x);
      const w = y.add(z);
      w.backward();
      expect(Number(assertGrad(x).data[0])).toBe(6);
    });

    it("should handle multiple outputs from same node", () => {
      const x = parameter(tensor([3]));
      const y = x.mul(x);
      const z = y.add(x);
      z.backward();
      expect(Number(assertGrad(x).data[0])).toBe(7);
    });
  });

  describe("gradient manipulation", () => {
    it("should zero gradients", () => {
      const a = parameter(tensor([1, 2, 3]));
      const b = a.sum();
      b.backward();
      expect(a.hasGrad()).toBe(true);
      a.zeroGrad();
      expect(numData(assertGrad(a))).toEqual([0, 0, 0]);
    });

    it("should set custom gradients", () => {
      const a = parameter(tensor([1, 2, 3]));
      a.setGrad(tensor([5, 6, 7]));
      expect(numData(assertGrad(a))).toEqual([5, 6, 7]);
    });

    it("should throw when setting grad on non-grad tensor", () => {
      const a = GradTensor.fromTensor(tensor([1, 2, 3]), {
        requiresGrad: false,
      });
      expect(() => a.setGrad(tensor([1, 1, 1]))).toThrow("requiresGrad=false");
    });

    it("should detach tensor from graph", () => {
      const a = parameter(tensor([1, 2, 3]));
      const b = a.detach();
      expect(b.requiresGrad).toBe(false);
    });
  });

  describe("noGrad context", () => {
    it("should disable gradient tracking in noGrad", () => {
      const result = noGrad(() => {
        const a = parameter(tensor([1, 2, 3]));
        const b = parameter(tensor([4, 5, 6]));
        const c = a.add(b);
        return c.requiresGrad;
      });
      expect(result).toBe(false);
    });

    it("should restore gradient tracking after noGrad", () => {
      noGrad(() => {
        const a = parameter(tensor([1, 2, 3]));
        expect(a.requiresGrad).toBe(false);
      });
      const a = parameter(tensor([1, 2, 3]));
      expect(a.requiresGrad).toBe(true);
    });

    it("should return value from noGrad", () => {
      const result = noGrad(() => {
        return 42;
      });
      expect(result).toBe(42);
    });

    it("should handle exceptions in noGrad", () => {
      expect(() => {
        noGrad(() => {
          throw new TypeError("test error");
        });
      }).toThrow("test error");
      const a = parameter(tensor([1]));
      expect(a.requiresGrad).toBe(true);
    });
  });

  describe("edge cases", () => {
    it("should handle backward on leaf node", () => {
      const a = parameter(tensor([1, 2, 3]));
      a.backward(tensor([1, 1, 1]));
      expect(numData(assertGrad(a))).toEqual([1, 1, 1]);
    });

    it("should handle backward without gradient argument on scalar", () => {
      const a = parameter(tensor(5));
      a.backward();
      expect(Number(assertGrad(a).data[0])).toBe(1);
    });

    it("should handle backward on non-scalar with default gradient", () => {
      const a = parameter(tensor([1, 2, 3]));
      a.backward();
      expect(numData(assertGrad(a))).toEqual([1, 1, 1]);
    });

    it("should handle zero gradients", () => {
      const a = parameter(tensor([1, 2, 3]));
      const b = parameter(tensor([0, 0, 0]));
      const c = a.mul(b);
      c.backward(tensor([1, 1, 1]));
      expect(numData(assertGrad(a))).toEqual([0, 0, 0]);
    });

    it("should handle large computational graphs", () => {
      const leaf = parameter(tensor([2]));
      let x: GradTensor = leaf;
      for (let i = 0; i < 100; i++) {
        x = x.add(GradTensor.scalar(1, { requiresGrad: false }));
      }
      x.backward();
      expect(Number(assertGrad(leaf).data[0])).toBe(1);
    });
  });

  describe("multi-dimensional tensors", () => {
    it("should compute gradients for 2D tensors", () => {
      const a = parameter(
        tensor([
          [1, 2],
          [3, 4],
        ])
      );
      const b = parameter(
        tensor([
          [5, 6],
          [7, 8],
        ])
      );
      const c = a.add(b);
      c.backward(
        tensor([
          [1, 1],
          [1, 1],
        ])
      );
      expect(numData(assertGrad(a))).toEqual([1, 1, 1, 1]);
    });

    it("should compute gradients for 3D tensors", () => {
      const a = parameter(tensor([[[1, 2]], [[3, 4]]]));
      const b = a.sum();
      b.backward();
      expect(numData(assertGrad(a))).toEqual([1, 1, 1, 1]);
    });
  });

  describe("gradient accumulation", () => {
    it("should accumulate gradients from multiple backward passes", () => {
      const a = parameter(tensor([1, 2]));
      const b = a.add(a);
      b.backward(tensor([1, 1]));
      const c = a.mul(a);
      c.backward(tensor([1, 1]));
      expect(numData(assertGrad(a))).toEqual([4, 6]);
    });

    it("should handle gradient accumulation with different operations", () => {
      const x = parameter(tensor([3]));
      const y1 = x.add(x);
      y1.backward();
      const y2 = x.mul(x);
      y2.backward();
      expect(Number(assertGrad(x).data[0])).toBe(8);
    });
  });

  describe("performance tests", () => {
    it("should handle deep computational graphs", () => {
      const leaf = parameter(tensor([1]));
      let x: GradTensor = leaf;
      for (let i = 0; i < 50; i++) {
        x = x.add(GradTensor.scalar(0.1, { requiresGrad: false }));
      }
      x.backward();
      expect(Number(assertGrad(leaf).data[0])).toBe(1);
    });

    it("should handle wide computational graphs", () => {
      const x = parameter(tensor([1]));
      let sum = x;
      for (let i = 0; i < 50; i++) {
        sum = sum.add(x);
      }
      sum.backward();
      expect(Number(assertGrad(x).data[0])).toBe(51);
    });
  });
});
