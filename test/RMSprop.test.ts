import { describe, expect, it } from "vitest";
import { parameter, tensor } from "../src/ndarray";
import { RMSprop } from "../src/optim";
import { getParamData, getParamValue } from "./optim-test-helpers";

describe("deepbox/optim - RMSprop", () => {
  describe("constructor", () => {
    it("should create with default options", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new RMSprop(params);
      expect(optimizer).toBeDefined();
    });

    it("should create with custom learning rate", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new RMSprop(params, { lr: 0.1 });
      expect(optimizer).toBeDefined();
    });

    it("should create with custom alpha", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new RMSprop(params, { alpha: 0.95 });
      expect(optimizer).toBeDefined();
    });

    it("should create with momentum", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new RMSprop(params, { momentum: 0.9 });
      expect(optimizer).toBeDefined();
    });

    it("should create with centered variant", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new RMSprop(params, { centered: true });
      expect(optimizer).toBeDefined();
    });

    it("should validate learning rate is non-negative", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new RMSprop(params, { lr: -0.01 })).toThrow("Invalid learning rate");
    });

    it("should validate alpha is non-negative", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new RMSprop(params, { alpha: -0.1 })).toThrow("Invalid alpha");
    });

    it("should validate epsilon is non-negative", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new RMSprop(params, { eps: -1e-8 })).toThrow("Invalid epsilon");
    });

    it("should validate weight decay is non-negative", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new RMSprop(params, { weightDecay: -0.01 })).toThrow(
        "Invalid weight_decay value"
      );
    });

    it("should validate momentum is non-negative", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new RMSprop(params, { momentum: -0.1 })).toThrow("Invalid momentum value");
    });

    it("should accept zero momentum", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new RMSprop(params, { momentum: 0 })).not.toThrow();
    });

    it("should accept high alpha values", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new RMSprop(params, { alpha: 0.999 })).not.toThrow();
    });
  });

  describe("step", () => {
    it("should throw when gradient is missing", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      const optimizer = new RMSprop([p]);
      expect(() => optimizer.step()).toThrow("Cannot optimize a parameter without a gradient");
    });

    it("should update parameters with positive gradients", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new RMSprop([p], { lr: 0.1 });
      optimizer.step();

      const data = getParamData(p, "RMSprop param");
      expect(data[0]).toBeLessThan(1);
      expect(data[1]).toBeLessThan(2);
      expect(data[2]).toBeLessThan(3);
    });

    it("should update parameters with negative gradients", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([-0.1, -0.1, -0.1], { dtype: "float64" }));

      const optimizer = new RMSprop([p], { lr: 0.1 });
      optimizer.step();

      const data = getParamData(p, "RMSprop param");
      expect(data[0]).toBeGreaterThan(1);
      expect(data[1]).toBeGreaterThan(2);
      expect(data[2]).toBeGreaterThan(3);
    });

    it("should handle zero gradients", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0, 0, 0], { dtype: "float64" }));

      const optimizer = new RMSprop([p], { lr: 0.1 });
      const before = Array.from(getParamData(p, "RMSprop param"));
      optimizer.step();
      const after = Array.from(getParamData(p, "RMSprop param"));

      expect(after[0]).toBeCloseTo(before[0], 5);
      expect(after[1]).toBeCloseTo(before[1], 5);
      expect(after[2]).toBeCloseTo(before[2], 5);
    });

    it("should apply weight decay", () => {
      const p = parameter(tensor([1, 1, 1], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new RMSprop([p], { lr: 0.1, weightDecay: 0.1 });
      optimizer.step();

      const data = getParamData(p, "RMSprop param");
      expect(data[0]).toBeLessThan(1);
    });

    it("should call closure if provided", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0, 0, 0], { dtype: "float64" }));
      const optimizer = new RMSprop([p]);

      let called = false;
      const closure = () => {
        called = true;
        return 0.5;
      };

      const loss = optimizer.step(closure);
      expect(called).toBe(true);
      expect(loss).toBe(0.5);
    });

    it("should accumulate squared gradients", () => {
      const p = parameter(tensor([1, 1, 1], { dtype: "float64" }));
      const optimizer = new RMSprop([p], { lr: 0.1, alpha: 0.9 });

      p.setGrad(tensor([1, 1, 1], { dtype: "float64" }));
      optimizer.step();
      const after1 = getParamValue(p, 0, "RMSprop param");

      p.setGrad(tensor([1, 1, 1], { dtype: "float64" }));
      optimizer.step();
      const after2 = getParamValue(p, 0, "RMSprop param");

      expect(after1).not.toEqual(after2);
    });

    it("should apply momentum correctly", () => {
      const p = parameter(tensor([1, 1, 1], { dtype: "float64" }));
      const optimizer = new RMSprop([p], { lr: 0.1, momentum: 0.9 });

      p.setGrad(tensor([1, 1, 1], { dtype: "float64" }));
      optimizer.step();
      const after1 = getParamValue(p, 0, "RMSprop param");

      p.setGrad(tensor([1, 1, 1], { dtype: "float64" }));
      optimizer.step();
      const after2 = getParamValue(p, 0, "RMSprop param");

      const step1Change = Math.abs(1 - after1);
      const step2Change = Math.abs(after1 - after2);
      expect(step2Change).toBeGreaterThan(step1Change);
    });

    it("should use centered variant correctly", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new RMSprop([p], { lr: 0.1, centered: true });
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should handle multiple parameters", () => {
      const p1 = parameter(tensor([1, 2], { dtype: "float64" }));
      const p2 = parameter(tensor([3, 4], { dtype: "float64" }));
      p1.setGrad(tensor([0.1, 0.1], { dtype: "float64" }));
      p2.setGrad(tensor([0.1, 0.1], { dtype: "float64" }));

      const optimizer = new RMSprop([p1, p2], { lr: 0.1 });
      optimizer.step();

      expect(getParamValue(p1, 0, "RMSprop param")).toBeLessThan(1);
      expect(getParamValue(p2, 0, "RMSprop param")).toBeLessThan(3);
    });

    it("should validate gradients are finite", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([Number.NaN, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new RMSprop([p]);
      expect(() => optimizer.step()).toThrow("Invalid gradient");
    });

    it("should validate parameters are finite", () => {
      const p = parameter(tensor([Number.NaN, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new RMSprop([p]);
      expect(() => optimizer.step()).toThrow("Invalid parameter");
    });

    it("should handle large gradients", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([100, 100, 100], { dtype: "float64" }));

      const optimizer = new RMSprop([p], { lr: 0.001 });
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should handle small gradients", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([1e-10, 1e-10, 1e-10], { dtype: "float64" }));

      const optimizer = new RMSprop([p], { lr: 0.1 });
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should converge on simple quadratic", () => {
      const p = parameter(tensor([5], { dtype: "float64" }));
      const optimizer = new RMSprop([p], { lr: 0.1 });

      for (let i = 0; i < 100; i++) {
        const x = getParamValue(p, 0, "RMSprop param");
        const grad = 2 * x;
        p.setGrad(tensor([grad], { dtype: "float64" }));
        optimizer.step();
      }

      const final = getParamValue(p, 0, "RMSprop param");
      expect(Math.abs(final)).toBeLessThan(0.1);
    });

    it("should work with centered and momentum together", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new RMSprop([p], {
        lr: 0.1,
        centered: true,
        momentum: 0.9,
      });
      expect(() => optimizer.step()).not.toThrow();
    });
  });

  describe("zeroGrad", () => {
    it("should be callable", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new RMSprop(params);
      expect(() => optimizer.zeroGrad()).not.toThrow();
    });
  });

  describe("parameter groups", () => {
    it("should support adding parameter groups", () => {
      const params1 = [parameter(tensor([1, 2], { dtype: "float64" }))];
      const optimizer = new RMSprop(params1);

      const params2 = [parameter(tensor([3, 4], { dtype: "float64" }))];
      expect(() => optimizer.addParamGroup({ params: params2, lr: 0.001 })).not.toThrow();
    });
  });

  describe("edge cases", () => {
    it("should handle single element tensor", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      p.setGrad(tensor([0.1], { dtype: "float64" }));

      const optimizer = new RMSprop([p]);
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should handle large tensors", () => {
      const data = new Array(1000).fill(1);
      const gradData = new Array(1000).fill(0.01);
      const p = parameter(tensor(data, { dtype: "float64" }));
      p.setGrad(tensor(gradData, { dtype: "float64" }));

      const optimizer = new RMSprop([p]);
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should handle alternating gradient signs", () => {
      const p = parameter(tensor([1, 1, 1, 1], { dtype: "float64" }));
      const optimizer = new RMSprop([p], { lr: 0.1 });

      p.setGrad(tensor([0.1, 0.1, 0.1, 0.1], { dtype: "float64" }));
      optimizer.step();

      p.setGrad(tensor([-0.1, -0.1, -0.1, -0.1], { dtype: "float64" }));
      optimizer.step();

      expect(Number.isFinite(getParamValue(p, 0, "RMSprop param"))).toBe(true);
    });

    it("should handle very small epsilon", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new RMSprop([p], { eps: 1e-15 });
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should handle high alpha values", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new RMSprop([p], { alpha: 0.999 });
      expect(() => optimizer.step()).not.toThrow();
    });
  });
});
