import { describe, expect, it } from "vitest";
import { parameter, tensor } from "../src/ndarray";
import { Adagrad } from "../src/optim";
import { getParamData, getParamValue } from "./optim-test-helpers";

describe("deepbox/optim - Adagrad", () => {
  describe("constructor", () => {
    it("should create with default options", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new Adagrad(params);
      expect(optimizer).toBeDefined();
    });

    it("should create with custom learning rate", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new Adagrad(params, { lr: 0.1 });
      expect(optimizer).toBeDefined();
    });

    it("should create with custom epsilon", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new Adagrad(params, { eps: 1e-8 });
      expect(optimizer).toBeDefined();
    });

    it("should create with weight decay", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new Adagrad(params, { weightDecay: 0.01 });
      expect(optimizer).toBeDefined();
    });

    it("should create with learning rate decay", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new Adagrad(params, { lrDecay: 0.001 });
      expect(optimizer).toBeDefined();
    });

    it("should validate learning rate is non-negative", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new Adagrad(params, { lr: -0.01 })).toThrow("Invalid learning rate");
    });

    it("should validate epsilon is non-negative", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new Adagrad(params, { eps: -1e-10 })).toThrow("Invalid epsilon");
    });

    it("should validate weight decay is non-negative", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new Adagrad(params, { weightDecay: -0.01 })).toThrow(
        "Invalid weight_decay value"
      );
    });

    it("should validate lr decay is non-negative", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new Adagrad(params, { lrDecay: -0.001 })).toThrow("Invalid lr_decay");
    });
  });

  describe("step", () => {
    it("should throw when gradient is missing", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      const optimizer = new Adagrad([p]);
      expect(() => optimizer.step()).toThrow("Cannot optimize a parameter without a gradient");
    });

    it("should update parameters with positive gradients", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new Adagrad([p], { lr: 0.1 });
      optimizer.step();

      const data = getParamData(p, "Adagrad param");
      expect(data[0]).toBeLessThan(1);
      expect(data[1]).toBeLessThan(2);
      expect(data[2]).toBeLessThan(3);
    });

    it("should update parameters with negative gradients", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([-0.1, -0.1, -0.1], { dtype: "float64" }));

      const optimizer = new Adagrad([p], { lr: 0.1 });
      optimizer.step();

      const data = getParamData(p, "Adagrad param");
      expect(data[0]).toBeGreaterThan(1);
      expect(data[1]).toBeGreaterThan(2);
      expect(data[2]).toBeGreaterThan(3);
    });

    it("should handle zero gradients", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0, 0, 0], { dtype: "float64" }));

      const optimizer = new Adagrad([p], { lr: 0.1 });
      const before = Array.from(getParamData(p, "Adagrad param"));
      optimizer.step();
      const after = Array.from(getParamData(p, "Adagrad param"));

      expect(after[0]).toBeCloseTo(before[0], 5);
      expect(after[1]).toBeCloseTo(before[1], 5);
      expect(after[2]).toBeCloseTo(before[2], 5);
    });

    it("should apply weight decay", () => {
      const p = parameter(tensor([1, 1, 1], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new Adagrad([p], { lr: 0.1, weightDecay: 0.1 });
      optimizer.step();

      const data = getParamData(p, "Adagrad param");
      expect(data[0]).toBeLessThan(1);
    });

    it("should call closure if provided", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0, 0, 0], { dtype: "float64" }));
      const optimizer = new Adagrad([p]);

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
      const optimizer = new Adagrad([p], { lr: 0.5 });

      p.setGrad(tensor([1, 1, 1], { dtype: "float64" }));
      optimizer.step();
      const after1 = getParamValue(p, 0, "Adagrad param");
      const step1Change = Math.abs(1 - after1);

      p.setGrad(tensor([1, 1, 1], { dtype: "float64" }));
      optimizer.step();
      const after2 = getParamValue(p, 0, "Adagrad param");
      const step2Change = Math.abs(after1 - after2);

      expect(step2Change).toBeLessThan(step1Change);
    });

    it("should apply learning rate decay", () => {
      const p = parameter(tensor([1, 1, 1], { dtype: "float64" }));
      const optimizer = new Adagrad([p], { lr: 1.0, lrDecay: 0.1 });

      p.setGrad(tensor([1, 1, 1], { dtype: "float64" }));
      optimizer.step();
      const after1 = getParamValue(p, 0, "Adagrad param");

      p.setGrad(tensor([1, 1, 1], { dtype: "float64" }));
      optimizer.step();
      const after2 = getParamValue(p, 0, "Adagrad param");

      expect(after1).not.toEqual(after2);
    });

    it("should handle multiple parameters", () => {
      const p1 = parameter(tensor([1, 2], { dtype: "float64" }));
      const p2 = parameter(tensor([3, 4], { dtype: "float64" }));
      p1.setGrad(tensor([0.1, 0.1], { dtype: "float64" }));
      p2.setGrad(tensor([0.1, 0.1], { dtype: "float64" }));

      const optimizer = new Adagrad([p1, p2], { lr: 0.1 });
      optimizer.step();

      expect(getParamValue(p1, 0, "Adagrad param")).toBeLessThan(1);
      expect(getParamValue(p2, 0, "Adagrad param")).toBeLessThan(3);
    });

    it("should validate gradients are finite", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([Number.NaN, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new Adagrad([p]);
      expect(() => optimizer.step()).toThrow("Invalid gradient");
    });

    it("should validate parameters are finite", () => {
      const p = parameter(tensor([Number.NaN, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new Adagrad([p]);
      expect(() => optimizer.step()).toThrow("Invalid parameter");
    });

    it("should handle large gradients", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([100, 100, 100], { dtype: "float64" }));

      const optimizer = new Adagrad([p], { lr: 0.001 });
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should handle small gradients", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([1e-10, 1e-10, 1e-10], { dtype: "float64" }));

      const optimizer = new Adagrad([p], { lr: 0.1 });
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should converge on simple quadratic", () => {
      const p = parameter(tensor([5], { dtype: "float64" }));
      const optimizer = new Adagrad([p], { lr: 0.5 });

      for (let i = 0; i < 100; i++) {
        const x = getParamValue(p, 0, "Adagrad param");
        const grad = 2 * x;
        p.setGrad(tensor([grad], { dtype: "float64" }));
        optimizer.step();
      }

      const final = getParamValue(p, 0, "Adagrad param");
      expect(Math.abs(final)).toBeLessThan(0.5);
    });

    it("should handle sparse gradients well", () => {
      const p = parameter(tensor([1, 1, 1, 1, 1], { dtype: "float64" }));
      const optimizer = new Adagrad([p], { lr: 0.1 });

      p.setGrad(tensor([1, 0, 0, 0, 0], { dtype: "float64" }));
      optimizer.step();

      p.setGrad(tensor([0, 0, 0, 0, 1], { dtype: "float64" }));
      optimizer.step();

      const data = getParamData(p, "Adagrad param");
      expect(data[0]).toBeLessThan(1);
      expect(data[4]).toBeLessThan(1);
    });
  });

  describe("zeroGrad", () => {
    it("should be callable", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new Adagrad(params);
      expect(() => optimizer.zeroGrad()).not.toThrow();
    });
  });

  describe("parameter groups", () => {
    it("should support adding parameter groups", () => {
      const params1 = [parameter(tensor([1, 2], { dtype: "float64" }))];
      const optimizer = new Adagrad(params1);

      const params2 = [parameter(tensor([3, 4], { dtype: "float64" }))];
      expect(() => optimizer.addParamGroup({ params: params2, lr: 0.001 })).not.toThrow();
    });
  });

  describe("edge cases", () => {
    it("should handle single element tensor", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      p.setGrad(tensor([0.1], { dtype: "float64" }));

      const optimizer = new Adagrad([p]);
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should handle large tensors", () => {
      const data = new Array(1000).fill(1);
      const gradData = new Array(1000).fill(0.01);
      const p = parameter(tensor(data, { dtype: "float64" }));
      p.setGrad(tensor(gradData, { dtype: "float64" }));

      const optimizer = new Adagrad([p]);
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should handle very small epsilon", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new Adagrad([p], { eps: 1e-15 });
      expect(() => optimizer.step()).not.toThrow();
    });
  });
});
