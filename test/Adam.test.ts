import { describe, expect, it } from "vitest";
import { parameter, tensor } from "../src/ndarray";
import { Adam } from "../src/optim";
import { getParamData } from "./optim-test-helpers";

describe("deepbox/optim - Adam", () => {
  describe("constructor", () => {
    it("should create with default options", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new Adam(params);
      expect(optimizer).toBeDefined();
      expect(optimizer.stepCount).toBe(0);
    });

    it("should validate beta ranges", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      expect(() => new Adam(params, { beta1: 1 })).toThrow("Invalid beta1");
      expect(() => new Adam(params, { beta2: 1 })).toThrow("Invalid beta2");
    });
  });

  describe("step", () => {
    it("should throw when gradient is missing", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      const optimizer = new Adam([p]);
      expect(() => optimizer.step()).toThrow("Cannot optimize a parameter without a gradient");
    });

    it("should update parameters", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));

      const optimizer = new Adam([p], {
        lr: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
      });
      optimizer.step();

      // Exact numeric values depend on bias correction; just assert it moved in the right direction.
      const data = getParamData(p, "Adam param");
      expect(data[0]).toBeLessThan(1);
      expect(data[1]).toBeLessThan(2);
      expect(data[2]).toBeLessThan(3);
    });

    it("should call closure if provided", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      p.setGrad(tensor([0, 0, 0], { dtype: "float64" }));
      const optimizer = new Adam([p]);

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
});
