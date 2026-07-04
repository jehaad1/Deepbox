import { describe, expect, it } from "vitest";
import { parameter, tensor } from "../src/ndarray";
import { LAMB } from "../src/optim";
import { getParamValue } from "./optim-test-helpers";

describe("deepbox/optim - LAMB", () => {
  describe("constructor", () => {
    it("should create with default options", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new LAMB(params);
      expect(optimizer).toBeDefined();
      expect(optimizer.stepCount).toBe(0);
    });

    it("should create with custom options", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new LAMB(params, {
        lr: 0.01,
        beta1: 0.95,
        beta2: 0.9999,
        weightDecay: 0.05,
      });
      expect(optimizer).toBeDefined();
    });

    it("should validate learning rate is non-negative", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new LAMB(params, { lr: -0.01 })).toThrow("Invalid learning rate");
    });

    it("should validate beta1 range [0, 1)", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new LAMB(params, { beta1: -0.1 })).toThrow("Invalid beta1");
      expect(() => new LAMB(params, { beta1: 1 })).toThrow("Invalid beta1");
    });

    it("should validate beta2 range [0, 1)", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new LAMB(params, { beta2: -0.1 })).toThrow("Invalid beta2");
      expect(() => new LAMB(params, { beta2: 1 })).toThrow("Invalid beta2");
    });

    it("should validate epsilon is positive", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new LAMB(params, { eps: 0 })).toThrow("Invalid epsilon");
      expect(() => new LAMB(params, { eps: -1e-8 })).toThrow("Invalid epsilon");
    });

    it("should validate weight decay is non-negative", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new LAMB(params, { weightDecay: -0.01 })).toThrow("Invalid weight_decay value");
    });
  });

  describe("step", () => {
    it("should update parameters", () => {
      const p = parameter(tensor([5.0, 3.0], { dtype: "float64" }));
      p.setGrad(tensor([1.0, 1.0], { dtype: "float64" }));

      const optimizer = new LAMB([p], { lr: 0.1, weightDecay: 0 });
      const before0 = getParamValue(p, 0, "p");
      optimizer.step();

      expect(optimizer.stepCount).toBe(1);
      const after0 = getParamValue(p, 0, "p");
      expect(after0).not.toBe(before0);
    });

    it("should decrease loss on a quadratic", () => {
      const p = parameter(tensor([10.0], { dtype: "float64" }));
      const optimizer = new LAMB([p], {
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        weightDecay: 0,
      });

      for (let i = 0; i < 100; i++) {
        optimizer.zeroGrad();
        const x = getParamValue(p, 0, "p");
        p.setGrad(tensor([2 * x], { dtype: "float64" }));
        optimizer.step();
      }

      expect(Math.abs(getParamValue(p, 0, "p"))).toBeLessThan(1);
    });

    it("should increment step count", () => {
      const p = parameter(tensor([1.0], { dtype: "float64" }));
      p.setGrad(tensor([0.1], { dtype: "float64" }));
      const optimizer = new LAMB([p]);

      optimizer.step();
      expect(optimizer.stepCount).toBe(1);
      p.setGrad(tensor([0.1], { dtype: "float64" }));
      optimizer.step();
      expect(optimizer.stepCount).toBe(2);
    });

    it("should handle multiple parameters", () => {
      const p1 = parameter(tensor([1, 2], { dtype: "float64" }));
      const p2 = parameter(tensor([3, 4], { dtype: "float64" }));
      p1.setGrad(tensor([0.1, 0.1], { dtype: "float64" }));
      p2.setGrad(tensor([0.1, 0.1], { dtype: "float64" }));

      const optimizer = new LAMB([p1, p2], { lr: 0.1 });
      expect(() => optimizer.step()).not.toThrow();
    });
  });

  describe("learning rate", () => {
    it("getLearningRate returns current lr", () => {
      const params = [parameter(tensor([1], { dtype: "float64" }))];
      const optimizer = new LAMB(params, { lr: 0.05 });
      expect(optimizer.getLearningRate()).toBe(0.05);
    });

    it("setLearningRate updates lr", () => {
      const params = [parameter(tensor([1], { dtype: "float64" }))];
      const optimizer = new LAMB(params, { lr: 0.05 });
      optimizer.setLearningRate(0.01);
      expect(optimizer.getLearningRate()).toBe(0.01);
    });

    it("getLearningRate throws on invalid group index", () => {
      const params = [parameter(tensor([1], { dtype: "float64" }))];
      const optimizer = new LAMB(params);
      expect(() => optimizer.getLearningRate(5)).toThrow();
    });
  });

  describe("closure", () => {
    it("should call closure and return loss", () => {
      const p = parameter(tensor([1.0], { dtype: "float64" }));
      p.setGrad(tensor([0.1], { dtype: "float64" }));
      const optimizer = new LAMB([p]);
      const loss = optimizer.step(() => 42.0);
      expect(loss).toBe(42.0);
    });
  });
});
