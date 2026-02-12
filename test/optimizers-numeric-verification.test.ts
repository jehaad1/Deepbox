import { describe, expect, it } from "vitest";
import { parameter, tensor } from "../src/ndarray";
import { Adagrad, Adam, AdamW, Nadam, RMSprop, SGD } from "../src/optim";
import { getParamValue } from "./optim-test-helpers";

describe("deepbox/optim - Numeric Verification Tests", () => {
  describe("Adam Bias Correction Verification", () => {
    it("should compute correct bias-corrected moments on first step", () => {
      const p = parameter(tensor([1.0], { dtype: "float64" }));
      const grad = 0.5;
      p.setGrad(tensor([grad], { dtype: "float64" }));

      const lr = 0.1;
      const beta1 = 0.9;
      const beta2 = 0.999;
      const eps = 1e-8;

      const optimizer = new Adam([p], {
        lr,
        beta1,
        beta2,
        eps,
        weightDecay: 0,
      });
      optimizer.step();

      const newValue = getParamValue(p, 0, "Adam param");
      expect(newValue).toBeCloseTo(0.9, 5);
    });

    it("should compute correct values after multiple steps", () => {
      const p = parameter(tensor([1.0], { dtype: "float64" }));
      const optimizer = new Adam([p], {
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weightDecay: 0,
      });

      p.setGrad(tensor([1.0], { dtype: "float64" }));
      optimizer.step();
      const after1 = getParamValue(p, 0, "Adam param");

      p.setGrad(tensor([1.0], { dtype: "float64" }));
      optimizer.step();
      const after2 = getParamValue(p, 0, "Adam param");

      expect(after1).toBeLessThan(1.0);
      expect(after2).toBeLessThan(after1);
      expect(optimizer.stepCount).toBe(2);
    });
  });

  describe("SGD Momentum Numeric Verification", () => {
    it("should compute correct momentum accumulation", () => {
      const p = parameter(tensor([1.0], { dtype: "float64" }));
      const lr = 0.1;
      const momentum = 0.9;

      const optimizer = new SGD([p], { lr, momentum, dampening: 0 });

      p.setGrad(tensor([1.0], { dtype: "float64" }));
      optimizer.step();
      expect(getParamValue(p, 0, "SGD param")).toBeCloseTo(0.9, 10);

      p.setGrad(tensor([1.0], { dtype: "float64" }));
      optimizer.step();
      expect(getParamValue(p, 0, "SGD param")).toBeCloseTo(0.71, 10);

      p.setGrad(tensor([1.0], { dtype: "float64" }));
      optimizer.step();
      expect(getParamValue(p, 0, "SGD param")).toBeCloseTo(0.439, 10);
    });

    it("should compute correct Nesterov momentum", () => {
      const p = parameter(tensor([1.0], { dtype: "float64" }));
      const optimizer = new SGD([p], {
        lr: 0.1,
        momentum: 0.9,
        dampening: 0,
        nesterov: true,
      });

      p.setGrad(tensor([1.0], { dtype: "float64" }));
      optimizer.step();
      expect(getParamValue(p, 0, "SGD param")).toBeCloseTo(0.81, 10);
    });

    it("should apply dampening correctly", () => {
      const p = parameter(tensor([1.0], { dtype: "float64" }));
      const optimizer = new SGD([p], {
        lr: 0.1,
        momentum: 0.9,
        dampening: 0.5,
      });

      p.setGrad(tensor([1.0], { dtype: "float64" }));
      optimizer.step();
      expect(getParamValue(p, 0, "SGD param")).toBeCloseTo(0.95, 10);

      p.setGrad(tensor([1.0], { dtype: "float64" }));
      optimizer.step();
      expect(getParamValue(p, 0, "SGD param")).toBeCloseTo(0.855, 10);
    });
  });

  describe("Adagrad Numeric Verification", () => {
    it("should compute correct adaptive learning rate", () => {
      const p = parameter(tensor([1.0], { dtype: "float64" }));
      const optimizer = new Adagrad([p], {
        lr: 1.0,
        eps: 1e-10,
        weightDecay: 0,
        lrDecay: 0,
      });

      p.setGrad(tensor([2.0], { dtype: "float64" }));
      optimizer.step();
      expect(getParamValue(p, 0, "Adagrad param")).toBeCloseTo(0.0, 8);

      p.setGrad(tensor([1.0], { dtype: "float64" }));
      optimizer.step();
      expect(getParamValue(p, 0, "Adagrad param")).toBeCloseTo(-1 / Math.sqrt(5), 8);
    });
  });

  describe("RMSprop Numeric Verification", () => {
    it("should compute correct exponential moving average", () => {
      const p = parameter(tensor([1.0], { dtype: "float64" }));
      const eps = 1e-8;
      const optimizer = new RMSprop([p], {
        lr: 0.1,
        alpha: 0.9,
        eps,
        weightDecay: 0,
        momentum: 0,
      });

      p.setGrad(tensor([1.0], { dtype: "float64" }));
      optimizer.step();
      const expected1 = 1.0 - (0.1 * 1.0) / (Math.sqrt(0.1) + eps);
      expect(getParamValue(p, 0, "RMSprop param")).toBeCloseTo(expected1, 8);
    });

    it("should compute correct centered variant denominator", () => {
      const p = parameter(tensor([1.0], { dtype: "float64" }));
      const lr = 0.1;
      const alpha = 0.9;
      const eps = 1.0;
      const optimizer = new RMSprop([p], {
        lr,
        alpha,
        eps,
        weightDecay: 0,
        momentum: 0,
        centered: true,
      });

      p.setGrad(tensor([1.0], { dtype: "float64" }));
      optimizer.step();

      const sqAvg = (1 - alpha) * 1.0;
      const gAvg = (1 - alpha) * 1.0;
      const variance = sqAvg - gAvg * gAvg;
      const expected = 1.0 - lr / Math.sqrt(variance + eps);
      expect(getParamValue(p, 0, "RMSprop param")).toBeCloseTo(expected, 8);
    });
  });

  describe("AdamW Decoupled Weight Decay Verification", () => {
    it("should apply weight decay directly to parameters", () => {
      const p = parameter(tensor([1.0], { dtype: "float64" }));
      const optimizer = new AdamW([p], {
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weightDecay: 0.1,
      });

      p.setGrad(tensor([0.0], { dtype: "float64" }));
      optimizer.step();

      const value = getParamValue(p, 0, "AdamW param");
      expect(value).toBeLessThan(1.0);
      expect(value).toBeCloseTo(0.99, 5);
    });
  });

  describe("Empty Parameter List", () => {
    it("should handle empty parameter list - SGD", () => {
      const optimizer = new SGD([], { lr: 0.01 });
      expect(() => optimizer.step()).not.toThrow();
      expect(() => optimizer.zeroGrad()).not.toThrow();
      optimizer.step();
      expect(optimizer.stepCount).toBe(2);
    });

    it("should handle empty parameter list - Adam", () => {
      const optimizer = new Adam([], { lr: 0.001 });
      expect(() => optimizer.step()).not.toThrow();
      expect(() => optimizer.zeroGrad()).not.toThrow();
    });

    it("should handle empty parameter list - AdamW", () => {
      const optimizer = new AdamW([], { lr: 0.001 });
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should handle empty parameter list - RMSprop", () => {
      const optimizer = new RMSprop([], { lr: 0.01 });
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should handle empty parameter list - Adagrad", () => {
      const optimizer = new Adagrad([], { lr: 0.01 });
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should serialize/deserialize empty optimizer state", () => {
      const optimizer = new SGD([], { lr: 0.01 });
      const state = optimizer.stateDict();
      expect(state.paramGroups).toHaveLength(0);
    });
  });

  describe("Weight Decay Numeric Verification", () => {
    it("SGD weight decay should add to gradient", () => {
      const p = parameter(tensor([2.0], { dtype: "float64" }));
      const optimizer = new SGD([p], { lr: 0.1, weightDecay: 0.5 });

      p.setGrad(tensor([1.0], { dtype: "float64" }));
      optimizer.step();
      expect(getParamValue(p, 0, "SGD param")).toBeCloseTo(1.8, 10);
    });
  });

  describe("Nadam Momentum Decay Verification", () => {
    it("should apply scheduled momentum correction", () => {
      const p = parameter(tensor([1.0], { dtype: "float64" }));
      const lr = 0.1;
      const beta1 = 0.9;
      const beta2 = 0.999;
      const eps = 1e-8;
      const momentumDecay = 0.004;
      const grad = 0.5;

      const optimizer = new Nadam([p], {
        lr,
        beta1,
        beta2,
        eps,
        weightDecay: 0,
        momentumDecay,
      });

      p.setGrad(tensor([grad], { dtype: "float64" }));
      optimizer.step();

      const t = 1;
      const mu = beta1 * (1 - 0.5 * 0.96 ** (t * momentumDecay));
      const muNext = beta1 * (1 - 0.5 * 0.96 ** ((t + 1) * momentumDecay));
      const muProduct = mu;
      const muProductNext = muProduct * muNext;

      const mNew = (1 - beta1) * grad;
      const vNew = (1 - beta2) * grad * grad;
      const denom = Math.sqrt(vNew / (1 - beta2 ** t)) + eps;
      const mHatNext = mNew / (1 - muProductNext);
      const gHat = grad / (1 - muProduct);
      const mNesterov = muNext * mHatNext + (1 - mu) * gHat;
      const expected = 1.0 - (lr * mNesterov) / denom;

      expect(getParamValue(p, 0, "Nadam param")).toBeCloseTo(expected, 8);
    });
  });
});
