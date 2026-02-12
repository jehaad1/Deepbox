import { describe, expect, it } from "vitest";
import { parameter, tensor } from "../src/ndarray";
import { SGD } from "../src/optim";
import { getParamData, getParamValue } from "./optim-test-helpers";

describe("deepbox/optim - SGD", () => {
  describe("constructor", () => {
    it("should create with default options", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new SGD(params);

      expect(optimizer).toBeDefined();
      expect(optimizer.getLearningRate()).toBe(0.01);
    });

    it("should create with custom learning rate", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new SGD(params, { lr: 0.1 });

      expect(optimizer.getLearningRate()).toBe(0.1);
    });

    it("should validate learning rate", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];

      expect(() => new SGD(params, { lr: -0.01 })).toThrow("Invalid learning rate");
    });

    it("should validate momentum", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];

      expect(() => new SGD(params, { momentum: -0.1 })).toThrow("Invalid momentum value");
    });

    it("should validate weight decay", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];

      expect(() => new SGD(params, { weightDecay: -0.01 })).toThrow("Invalid weight_decay value");
    });

    it("should validate nesterov options", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];

      // Nesterov requires momentum and zero dampening
      expect(() => new SGD(params, { nesterov: true, momentum: 0 })).toThrow(
        "Nesterov momentum requires a momentum and zero dampening"
      );

      expect(() => new SGD(params, { nesterov: true, momentum: 0.9, dampening: 0.1 })).toThrow(
        "Nesterov momentum requires a momentum and zero dampening"
      );
    });
  });

  describe("step", () => {
    it("should update parameters for vanilla SGD", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new SGD([p], { lr: 0.1 });
      optimizer.step();

      const data = getParamData(p, "SGD param");
      expect(data[0]).toBeCloseTo(0.99);
      expect(data[1]).toBeCloseTo(1.99);
      expect(data[2]).toBeCloseTo(2.99);
    });

    it("should throw when gradient is missing", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      const optimizer = new SGD([p], { lr: 0.1 });
      expect(() => optimizer.step()).toThrow("Cannot optimize a parameter without a gradient");
    });

    it("should apply momentum over multiple steps", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      const optimizer = new SGD([p], { lr: 0.1, momentum: 0.9 });

      p.setGrad(tensor([1, 1, 1], { dtype: "float64" }));
      optimizer.step();
      // step1: buf=grad=1, p -= 0.1*1
      expect(getParamValue(p, 0, "SGD param")).toBeCloseTo(0.9);

      p.setGrad(tensor([1, 1, 1], { dtype: "float64" }));
      optimizer.step();
      // step2: buf=0.9*1 + 1 = 1.9, p -= 0.1*1.9
      expect(getParamValue(p, 0, "SGD param")).toBeCloseTo(0.71);
    });

    it("should call closure if provided", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      params[0].setGrad(tensor([0, 0, 0], { dtype: "float64" }));
      const optimizer = new SGD(params);

      let called = false;
      const closure = () => {
        called = true;
        return 0.5;
      };

      const loss = optimizer.step(closure);
      expect(called).toBe(true);
      expect(loss).toBe(0.5);
    });
  });

  describe("zeroGrad", () => {
    it("should be callable", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new SGD(params);

      expect(() => optimizer.zeroGrad()).not.toThrow();
    });
  });

  describe("learning rate methods", () => {
    it("should get learning rate", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new SGD(params, { lr: 0.1 });

      expect(optimizer.getLearningRate()).toBe(0.1);
    });

    it("should set learning rate", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new SGD(params);

      optimizer.setLearningRate(0.001);
      expect(optimizer.getLearningRate()).toBe(0.001);
    });
  });

  describe("parameter groups", () => {
    it("should support adding parameter groups", () => {
      const params1 = [parameter(tensor([1, 2], { dtype: "float64" }))];
      const optimizer = new SGD(params1);

      const params2 = [parameter(tensor([3, 4], { dtype: "float64" }))];
      expect(() => optimizer.addParamGroup({ params: params2, lr: 0.001 })).not.toThrow();
    });
  });
});
