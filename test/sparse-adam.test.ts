import { describe, expect, it } from "vitest";
import { parameter, tensor } from "../src/ndarray";
import { SparseAdam } from "../src/optim";
import { getParamData } from "./optim-test-helpers";

describe("deepbox/optim - SparseAdam", () => {
  describe("constructor", () => {
    it("should create with default options", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new SparseAdam(params);
      expect(optimizer).toBeDefined();
      expect(optimizer.stepCount).toBe(0);
    });

    it("should validate beta ranges", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new SparseAdam(params, { beta1: 1 })).toThrow("Invalid beta1");
      expect(() => new SparseAdam(params, { beta2: 1 })).toThrow("Invalid beta2");
    });

    it("should validate learning rate", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new SparseAdam(params, { lr: -1 })).toThrow();
    });

    it("should validate epsilon", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new SparseAdam(params, { eps: 0 })).toThrow();
      expect(() => new SparseAdam(params, { eps: -1 })).toThrow();
    });
  });

  describe("step", () => {
    it("should throw when gradient is missing", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      const optimizer = new SparseAdam([p]);
      expect(() => optimizer.step()).toThrow("Cannot optimize a parameter without a gradient");
    });

    it("should only update parameters with non-zero gradients", () => {
      const p = parameter(tensor([1, 2, 3, 4, 5], { dtype: "float64" }));
      // Sparse gradient: only indices 0 and 3 are non-zero
      p.setGrad(tensor([0.5, 0, 0, 0.5, 0], { dtype: "float64" }));

      const optimizer = new SparseAdam([p], { lr: 0.1 });
      optimizer.step();

      const data = getParamData(p, "SparseAdam param");
      // Indices 0 and 3 should have been updated (decreased, since gradient is positive)
      expect(data[0]).toBeLessThan(1);
      expect(data[3]).toBeLessThan(4);
      // Indices 1, 2, 4 should be unchanged (zero gradient)
      expect(data[1]).toBeCloseTo(2, 10);
      expect(data[2]).toBeCloseTo(3, 10);
      expect(data[4]).toBeCloseTo(5, 10);
    });

    it("should update all parameters with dense gradients", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new SparseAdam([p], { lr: 0.1 });
      optimizer.step();

      const data = getParamData(p, "SparseAdam param");
      expect(data[0]).toBeLessThan(1);
      expect(data[1]).toBeLessThan(2);
      expect(data[2]).toBeLessThan(3);
    });

    it("should converge over multiple steps with sparse gradients", () => {
      const p = parameter(tensor([5, 5, 5], { dtype: "float64" }));
      const optimizer = new SparseAdam([p], { lr: 0.01 });

      // Only update index 0 repeatedly
      for (let i = 0; i < 50; i++) {
        p.setGrad(tensor([1, 0, 0], { dtype: "float64" }));
        optimizer.step();
      }

      const data = getParamData(p, "SparseAdam param");
      // Index 0 should have moved significantly toward lower values
      expect(data[0]).toBeLessThan(5);
      // Indices 1 and 2 should be unchanged
      expect(data[1]).toBeCloseTo(5, 10);
      expect(data[2]).toBeCloseTo(5, 10);
    });

    it("should call closure if provided", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0, 0, 0], { dtype: "float64" }));
      const optimizer = new SparseAdam([p]);

      let called = false;
      const closure = () => {
        called = true;
        return 0.5;
      };

      const loss = optimizer.step(closure);
      expect(called).toBe(true);
      expect(loss).toBe(0.5);
    });

    it("should increment stepCount", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0, 0, 0], { dtype: "float64" }));
      const optimizer = new SparseAdam([p]);

      expect(optimizer.stepCount).toBe(0);
      optimizer.step();
      expect(optimizer.stepCount).toBe(1);
      optimizer.step();
      expect(optimizer.stepCount).toBe(2);
    });
  });

  describe("learning rate management", () => {
    it("getLearningRate returns current LR", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const optimizer = new SparseAdam([p], { lr: 0.01 });
      expect(optimizer.getLearningRate()).toBe(0.01);
    });

    it("setLearningRate updates LR", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const optimizer = new SparseAdam([p], { lr: 0.01 });
      optimizer.setLearningRate(0.001);
      expect(optimizer.getLearningRate()).toBe(0.001);
    });

    it("getLearningRate throws for invalid group index", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const optimizer = new SparseAdam([p]);
      expect(() => optimizer.getLearningRate(5)).toThrow();
    });
  });
});
