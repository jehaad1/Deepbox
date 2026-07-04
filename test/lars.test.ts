import { describe, expect, it } from "vitest";
import { parameter, tensor } from "../src/ndarray";
import { LARS } from "../src/optim";
import { getParamValue } from "./optim-test-helpers";

describe("deepbox/optim - LARS", () => {
  describe("constructor", () => {
    it("should create with default options", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new LARS(params);
      expect(optimizer).toBeDefined();
      expect(optimizer.stepCount).toBe(0);
    });

    it("should create with custom options", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new LARS(params, {
        lr: 0.01,
        momentum: 0.95,
        weightDecay: 1e-3,
        eta: 0.002,
      });
      expect(optimizer).toBeDefined();
    });

    it("should validate learning rate is non-negative", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new LARS(params, { lr: -0.01 })).toThrow("Invalid learning rate");
    });

    it("should validate momentum range [0, 1)", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new LARS(params, { momentum: -0.1 })).toThrow("Invalid momentum");
      expect(() => new LARS(params, { momentum: 1 })).toThrow("Invalid momentum");
    });

    it("should validate weight decay is non-negative", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new LARS(params, { weightDecay: -0.01 })).toThrow("Invalid weight_decay value");
    });

    it("should validate eta is positive", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new LARS(params, { eta: 0 })).toThrow("Invalid eta");
      expect(() => new LARS(params, { eta: -1 })).toThrow("Invalid eta");
    });
  });

  describe("step", () => {
    it("should update parameters", () => {
      const p = parameter(tensor([5.0, 3.0], { dtype: "float64" }));
      p.setGrad(tensor([1.0, 1.0], { dtype: "float64" }));

      const optimizer = new LARS([p], {
        lr: 0.1,
        momentum: 0.0,
        weightDecay: 0,
      });
      const before0 = getParamValue(p, 0, "p");
      optimizer.step();

      expect(optimizer.stepCount).toBe(1);
      const after0 = getParamValue(p, 0, "p");
      expect(after0).not.toBe(before0);
    });

    it("should decrease loss on a quadratic", () => {
      // Minimize f(x) = x^2, gradient = 2x
      const p = parameter(tensor([10.0], { dtype: "float64" }));
      const optimizer = new LARS([p], {
        lr: 0.1,
        momentum: 0.9,
        weightDecay: 0,
        eta: 1.0,
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
      const optimizer = new LARS([p]);

      optimizer.step();
      expect(optimizer.stepCount).toBe(1);
      p.setGrad(tensor([0.1], { dtype: "float64" }));
      optimizer.step();
      expect(optimizer.stepCount).toBe(2);
    });
  });

  describe("learning rate", () => {
    it("getLearningRate returns current lr", () => {
      const params = [parameter(tensor([1], { dtype: "float64" }))];
      const optimizer = new LARS(params, { lr: 0.05 });
      expect(optimizer.getLearningRate()).toBe(0.05);
    });

    it("setLearningRate updates lr", () => {
      const params = [parameter(tensor([1], { dtype: "float64" }))];
      const optimizer = new LARS(params, { lr: 0.05 });
      optimizer.setLearningRate(0.01);
      expect(optimizer.getLearningRate()).toBe(0.01);
    });

    it("getLearningRate throws on invalid group index", () => {
      const params = [parameter(tensor([1], { dtype: "float64" }))];
      const optimizer = new LARS(params);
      expect(() => optimizer.getLearningRate(5)).toThrow();
    });
  });

  describe("closure", () => {
    it("should call closure and return loss", () => {
      const p = parameter(tensor([1.0], { dtype: "float64" }));
      p.setGrad(tensor([0.1], { dtype: "float64" }));
      const optimizer = new LARS([p]);
      const loss = optimizer.step(() => 42.0);
      expect(loss).toBe(42.0);
    });
  });
});
