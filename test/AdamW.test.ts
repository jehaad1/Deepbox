import { describe, expect, it } from "vitest";
import { parameter, tensor } from "../src/ndarray";
import { AdamW } from "../src/optim";
import { getParamData, getParamValue, getTensorData } from "./optim-test-helpers";

describe("deepbox/optim - AdamW", () => {
  describe("constructor", () => {
    it("should create with default options", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new AdamW(params);
      expect(optimizer).toBeDefined();
      expect(optimizer.stepCount).toBe(0);
    });

    it("should create with custom learning rate", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new AdamW(params, { lr: 0.1 });
      expect(optimizer).toBeDefined();
    });

    it("should create with custom beta1", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new AdamW(params, { beta1: 0.95 });
      expect(optimizer).toBeDefined();
    });

    it("should create with custom beta2", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new AdamW(params, { beta2: 0.9999 });
      expect(optimizer).toBeDefined();
    });

    it("should create with custom weight decay", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new AdamW(params, { weightDecay: 0.01 });
      expect(optimizer).toBeDefined();
    });

    it("should create with amsgrad enabled", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new AdamW(params, { amsgrad: true });
      expect(optimizer).toBeDefined();
    });

    it("should validate learning rate is non-negative", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new AdamW(params, { lr: -0.01 })).toThrow("Invalid learning rate");
    });

    it("should validate learning rate is finite", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new AdamW(params, { lr: Number.NaN })).toThrow("Invalid learning rate");
      expect(() => new AdamW(params, { lr: Number.POSITIVE_INFINITY })).toThrow(
        "Invalid learning rate"
      );
    });

    it("should validate beta1 range [0, 1)", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new AdamW(params, { beta1: -0.1 })).toThrow("Invalid beta1");
      expect(() => new AdamW(params, { beta1: 1 })).toThrow("Invalid beta1");
      expect(() => new AdamW(params, { beta1: 1.1 })).toThrow("Invalid beta1");
    });

    it("should validate beta2 range [0, 1)", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new AdamW(params, { beta2: -0.1 })).toThrow("Invalid beta2");
      expect(() => new AdamW(params, { beta2: 1 })).toThrow("Invalid beta2");
      expect(() => new AdamW(params, { beta2: 1.1 })).toThrow("Invalid beta2");
    });

    it("should validate epsilon is non-negative", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new AdamW(params, { eps: -1e-8 })).toThrow("Invalid epsilon");
    });

    it("should validate weight decay is non-negative", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new AdamW(params, { weightDecay: -0.01 })).toThrow("Invalid weight_decay value");
    });

    it("should accept zero weight decay", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new AdamW(params, { weightDecay: 0 })).not.toThrow();
    });

    it("should accept boundary beta values", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new AdamW(params, { beta1: 0, beta2: 0 })).not.toThrow();
      expect(() => new AdamW(params, { beta1: 0.9999, beta2: 0.9999 })).not.toThrow();
    });
  });

  describe("step", () => {
    it("should throw when gradient is missing", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      const optimizer = new AdamW([p]);
      expect(() => optimizer.step()).toThrow("Cannot optimize a parameter without a gradient");
    });

    it("should update parameters with positive gradients", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new AdamW([p], {
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weightDecay: 0.01,
      });
      optimizer.step();

      const data = getParamData(p, "AdamW param");
      expect(data[0]).toBeLessThan(1);
      expect(data[1]).toBeLessThan(2);
      expect(data[2]).toBeLessThan(3);
    });

    it("should update parameters with negative gradients", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([-0.1, -0.1, -0.1], { dtype: "float64" }));

      const optimizer = new AdamW([p], { lr: 0.1 });
      optimizer.step();

      const data = getParamData(p, "AdamW param");
      expect(data[0]).toBeGreaterThan(1);
      expect(data[1]).toBeGreaterThan(2);
      expect(data[2]).toBeGreaterThan(3);
    });

    it("should handle zero gradients", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0, 0, 0], { dtype: "float64" }));

      const optimizer = new AdamW([p], { lr: 0.1, weightDecay: 0 });
      const before = Array.from(getParamData(p, "AdamW param"));
      optimizer.step();
      const after = Array.from(getParamData(p, "AdamW param"));

      expect(after[0]).toBeCloseTo(before[0], 5);
      expect(after[1]).toBeCloseTo(before[1], 5);
      expect(after[2]).toBeCloseTo(before[2], 5);
    });

    it("should apply weight decay correctly (decoupled)", () => {
      const p = parameter(tensor([1, 1, 1], { dtype: "float64" }));
      p.setGrad(tensor([0, 0, 0], { dtype: "float64" }));

      const optimizer = new AdamW([p], { lr: 0.1, weightDecay: 0.1 });
      optimizer.step();

      const data = getParamData(p, "AdamW param");
      expect(data[0]).toBeLessThan(1);
      expect(data[0]).toBeCloseTo(0.99, 2);
    });

    it("should increment step count", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new AdamW([p]);
      expect(optimizer.stepCount).toBe(0);
      optimizer.step();
      expect(optimizer.stepCount).toBe(1);
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));
      optimizer.step();
      expect(optimizer.stepCount).toBe(2);
    });

    it("should call closure if provided", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0, 0, 0], { dtype: "float64" }));
      const optimizer = new AdamW([p]);

      let called = false;
      const closure = () => {
        called = true;
        return 0.5;
      };

      const loss = optimizer.step(closure);
      expect(called).toBe(true);
      expect(loss).toBe(0.5);
    });

    it("should accumulate momentum over multiple steps", () => {
      const p = parameter(tensor([1, 1, 1], { dtype: "float64" }));
      const optimizer = new AdamW([p], { lr: 0.1, beta1: 0.9, beta2: 0.999 });

      p.setGrad(tensor([1, 1, 1], { dtype: "float64" }));
      optimizer.step();
      const after1 = getParamValue(p, 0, "AdamW param");

      p.setGrad(tensor([1, 1, 1], { dtype: "float64" }));
      optimizer.step();
      const after2 = getParamValue(p, 0, "AdamW param");

      const step1Change = Math.abs(1 - after1);
      const step2Change = Math.abs(after1 - after2);
      expect(step2Change).toBeGreaterThan(step1Change * 0.5);
    });

    it("should handle AMSGrad variant", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new AdamW([p], { amsgrad: true });
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should apply bias correction", () => {
      const p = parameter(tensor([1, 1, 1], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new AdamW([p], { lr: 0.1, beta1: 0.9, beta2: 0.999 });
      optimizer.step();

      const firstStepChange = Math.abs(1 - getParamValue(p, 0, "AdamW param"));
      expect(firstStepChange).toBeGreaterThan(0);
    });

    it("should handle multiple parameters", () => {
      const p1 = parameter(tensor([1, 2], { dtype: "float64" }));
      const p2 = parameter(tensor([3, 4], { dtype: "float64" }));
      p1.setGrad(tensor([0.1, 0.1], { dtype: "float64" }));
      p2.setGrad(tensor([0.1, 0.1], { dtype: "float64" }));

      const optimizer = new AdamW([p1, p2], { lr: 0.1 });
      optimizer.step();

      expect(getParamValue(p1, 0, "AdamW param")).toBeLessThan(1);
      expect(getParamValue(p2, 0, "AdamW param")).toBeLessThan(3);
    });

    it("should validate gradients are finite", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([Number.NaN, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new AdamW([p]);
      expect(() => optimizer.step()).toThrow("Invalid gradient");
    });

    it("should validate parameters are finite", () => {
      const p = parameter(tensor([Number.NaN, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new AdamW([p]);
      expect(() => optimizer.step()).toThrow("Invalid parameter");
    });

    it("should handle large gradients", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([100, 100, 100], { dtype: "float64" }));

      const optimizer = new AdamW([p], { lr: 0.001 });
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should handle small gradients", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([1e-10, 1e-10, 1e-10], { dtype: "float64" }));

      const optimizer = new AdamW([p], { lr: 0.1 });
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should converge on simple quadratic", () => {
      const p = parameter(tensor([5], { dtype: "float64" }));
      const optimizer = new AdamW([p], { lr: 0.1, weightDecay: 0 });

      for (let i = 0; i < 100; i++) {
        const x = getParamValue(p, 0, "AdamW param");
        const grad = 2 * x;
        p.setGrad(tensor([grad], { dtype: "float64" }));
        optimizer.step();
      }

      const final = getParamValue(p, 0, "AdamW param");
      expect(Math.abs(final)).toBeLessThan(0.5);
    });
  });

  describe("zeroGrad", () => {
    it("should be callable", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new AdamW(params);
      expect(() => optimizer.zeroGrad()).not.toThrow();
    });

    it("should zero out gradients", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.2, 0.3], { dtype: "float64" }));

      const optimizer = new AdamW([p]);
      optimizer.zeroGrad();

      const grad = p.grad;
      expect(grad).toBeDefined();
      if (grad) {
        const data = getTensorData(grad, "AdamW grad");
        expect(data[0]).toBe(0);
        expect(data[1]).toBe(0);
        expect(data[2]).toBe(0);
      }
    });
  });

  describe("parameter groups", () => {
    it("should support adding parameter groups", () => {
      const params1 = [parameter(tensor([1, 2], { dtype: "float64" }))];
      const optimizer = new AdamW(params1);

      const params2 = [parameter(tensor([3, 4], { dtype: "float64" }))];
      expect(() => optimizer.addParamGroup({ params: params2, lr: 0.001 })).not.toThrow();
    });
  });

  describe("edge cases", () => {
    it("should handle single element tensor", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      p.setGrad(tensor([0.1], { dtype: "float64" }));

      const optimizer = new AdamW([p]);
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should handle large tensors", () => {
      const data = new Array(1000).fill(1);
      const gradData = new Array(1000).fill(0.01);
      const p = parameter(tensor(data, { dtype: "float64" }));
      p.setGrad(tensor(gradData, { dtype: "float64" }));

      const optimizer = new AdamW([p]);
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should handle alternating gradient signs", () => {
      const p = parameter(tensor([1, 1, 1, 1], { dtype: "float64" }));
      const optimizer = new AdamW([p], { lr: 0.1 });

      p.setGrad(tensor([0.1, 0.1, 0.1, 0.1], { dtype: "float64" }));
      optimizer.step();

      p.setGrad(tensor([-0.1, -0.1, -0.1, -0.1], { dtype: "float64" }));
      optimizer.step();

      expect(() => optimizer.step()).not.toThrow();
    });
  });
});
