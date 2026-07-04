import { describe, expect, it } from "vitest";
import { parameter, tensor } from "../src/ndarray";
import { ASGD, Rprop } from "../src/optim";

function getParamValue(p: ReturnType<typeof parameter>, i: number): number {
  const data = p.tensor.data;
  if (Array.isArray(data)) throw new Error("string tensor");
  return Number(data[i]);
}

describe("Rprop", () => {
  describe("constructor", () => {
    it("creates with default options", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new Rprop(params);
      expect(optimizer).toBeDefined();
      expect(optimizer.stepCount).toBe(0);
    });

    it("creates with custom options", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new Rprop(params, {
        lr: 0.001,
        etaMinus: 0.4,
        etaPlus: 1.3,
        stepMin: 1e-7,
        stepMax: 100,
      });
      expect(optimizer).toBeDefined();
    });

    it("validates etaMinus in (0, 1)", () => {
      const params = [parameter(tensor([1], { dtype: "float64" }))];
      expect(() => new Rprop(params, { etaMinus: 0 })).toThrow();
      expect(() => new Rprop(params, { etaMinus: 1 })).toThrow();
      expect(() => new Rprop(params, { etaMinus: 1.5 })).toThrow();
    });

    it("validates etaPlus > 1", () => {
      const params = [parameter(tensor([1], { dtype: "float64" }))];
      expect(() => new Rprop(params, { etaPlus: 0.5 })).toThrow();
      expect(() => new Rprop(params, { etaPlus: 1 })).toThrow();
    });

    it("validates lr is non-negative", () => {
      const params = [parameter(tensor([1], { dtype: "float64" }))];
      expect(() => new Rprop(params, { lr: -0.01 })).toThrow();
    });
  });

  describe("step", () => {
    it("updates parameters", () => {
      const p = parameter(tensor([5.0, 3.0], { dtype: "float64" }));
      p.setGrad(tensor([1.0, -1.0], { dtype: "float64" }));

      const optimizer = new Rprop([p], { lr: 0.01 });
      const before0 = getParamValue(p, 0);
      optimizer.step();

      expect(getParamValue(p, 0)).not.toBe(before0);
    });

    it("decreases loss on a quadratic", () => {
      const p = parameter(tensor([10.0], { dtype: "float64" }));
      const optimizer = new Rprop([p], { lr: 0.1 });

      let prevLoss = Infinity;
      for (let i = 0; i < 20; i++) {
        const val = getParamValue(p, 0);
        const loss = val * val;
        if (i > 2) {
          expect(loss).toBeLessThan(prevLoss + 1e-6);
        }
        prevLoss = loss;
        p.setGrad(tensor([2 * val], { dtype: "float64" }));
        optimizer.step();
      }
      expect(Math.abs(getParamValue(p, 0))).toBeLessThan(5);
    });

    it("increments step count", () => {
      const p = parameter(tensor([1.0], { dtype: "float64" }));
      p.setGrad(tensor([0.1], { dtype: "float64" }));
      const optimizer = new Rprop([p]);

      optimizer.step();
      expect(optimizer.stepCount).toBe(1);
      optimizer.step();
      expect(optimizer.stepCount).toBe(2);
    });

    it("calls closure and returns loss", () => {
      const p = parameter(tensor([1.0], { dtype: "float64" }));
      p.setGrad(tensor([0.1], { dtype: "float64" }));
      const optimizer = new Rprop([p]);
      const loss = optimizer.step(() => 42.0);
      expect(loss).toBe(42.0);
    });
  });

  describe("learning rate", () => {
    it("getLearningRate returns current lr", () => {
      const params = [parameter(tensor([1], { dtype: "float64" }))];
      const optimizer = new Rprop(params, { lr: 0.05 });
      expect(optimizer.getLearningRate()).toBe(0.05);
    });

    it("setLearningRate updates lr", () => {
      const params = [parameter(tensor([1], { dtype: "float64" }))];
      const optimizer = new Rprop(params, { lr: 0.05 });
      optimizer.setLearningRate(0.01);
      expect(optimizer.getLearningRate()).toBe(0.01);
    });
  });
});

describe("ASGD", () => {
  describe("constructor", () => {
    it("creates with default options", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new ASGD(params);
      expect(optimizer).toBeDefined();
      expect(optimizer.stepCount).toBe(0);
    });

    it("creates with custom options", () => {
      const params = [parameter(tensor([1, 2, 3], { dtype: "float64" }))];
      const optimizer = new ASGD(params, {
        lr: 0.001,
        lambda: 1e-3,
        alpha: 0.5,
        t0: 100,
        weightDecay: 1e-4,
      });
      expect(optimizer).toBeDefined();
    });

    it("validates lr is non-negative", () => {
      const params = [parameter(tensor([1], { dtype: "float64" }))];
      expect(() => new ASGD(params, { lr: -0.01 })).toThrow();
    });

    it("validates weightDecay is non-negative", () => {
      const params = [parameter(tensor([1], { dtype: "float64" }))];
      expect(() => new ASGD(params, { weightDecay: -0.01 })).toThrow();
    });
  });

  describe("step", () => {
    it("updates parameters", () => {
      const p = parameter(tensor([5.0, 3.0], { dtype: "float64" }));
      p.setGrad(tensor([1.0, 1.0], { dtype: "float64" }));

      const optimizer = new ASGD([p], { lr: 0.1 });
      const before0 = getParamValue(p, 0);
      optimizer.step();

      expect(getParamValue(p, 0)).not.toBe(before0);
    });

    it("decreases loss on a quadratic", () => {
      const p = parameter(tensor([10.0], { dtype: "float64" }));
      const optimizer = new ASGD([p], { lr: 0.01, lambda: 0, t0: 1e6 });

      let prevLoss = Infinity;
      for (let i = 0; i < 50; i++) {
        const val = getParamValue(p, 0);
        const loss = val * val;
        if (i > 0) {
          expect(loss).toBeLessThanOrEqual(prevLoss + 1e-6);
        }
        prevLoss = loss;
        p.setGrad(tensor([2 * val], { dtype: "float64" }));
        optimizer.step();
      }
      expect(Math.abs(getParamValue(p, 0))).toBeLessThan(10);
    });

    it("increments step count", () => {
      const p = parameter(tensor([1.0], { dtype: "float64" }));
      p.setGrad(tensor([0.1], { dtype: "float64" }));
      const optimizer = new ASGD([p]);

      optimizer.step();
      expect(optimizer.stepCount).toBe(1);
      optimizer.step();
      expect(optimizer.stepCount).toBe(2);
    });

    it("calls closure and returns loss", () => {
      const p = parameter(tensor([1.0], { dtype: "float64" }));
      p.setGrad(tensor([0.1], { dtype: "float64" }));
      const optimizer = new ASGD([p]);
      const loss = optimizer.step(() => 99.0);
      expect(loss).toBe(99.0);
    });
  });

  describe("learning rate", () => {
    it("getLearningRate returns current lr", () => {
      const params = [parameter(tensor([1], { dtype: "float64" }))];
      const optimizer = new ASGD(params, { lr: 0.05 });
      expect(optimizer.getLearningRate()).toBe(0.05);
    });

    it("setLearningRate updates lr", () => {
      const params = [parameter(tensor([1], { dtype: "float64" }))];
      const optimizer = new ASGD(params, { lr: 0.05 });
      optimizer.setLearningRate(0.01);
      expect(optimizer.getLearningRate()).toBe(0.01);
    });
  });
});
