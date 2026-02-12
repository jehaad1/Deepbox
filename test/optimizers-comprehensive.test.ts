import { describe, expect, it } from "vitest";
import { parameter, tensor } from "../src/ndarray";
import { Adagrad, Adam, AdamW, RMSprop, SGD } from "../src/optim";
import { getParamData, getParamValue, getTensorValue } from "./optim-test-helpers";

describe("deepbox/optim - Comprehensive Integration Tests", () => {
  describe("Optimizer Comparison", () => {
    it("should all optimizers reduce loss on quadratic", () => {
      const sgdParam = parameter(tensor([5], { dtype: "float64" }));
      const adamParam = parameter(tensor([5], { dtype: "float64" }));
      const adamwParam = parameter(tensor([5], { dtype: "float64" }));
      const rmspropParam = parameter(tensor([5], { dtype: "float64" }));
      const adagradParam = parameter(tensor([5], { dtype: "float64" }));
      const optimizers = [
        {
          name: "SGD",
          opt: new SGD([sgdParam], {
            lr: 0.1,
          }),
          params: [sgdParam],
        },
        {
          name: "Adam",
          opt: new Adam([adamParam], {
            lr: 0.1,
          }),
          params: [adamParam],
        },
        {
          name: "AdamW",
          opt: new AdamW([adamwParam], {
            lr: 0.1,
            weightDecay: 0,
          }),
          params: [adamwParam],
        },
        {
          name: "RMSprop",
          opt: new RMSprop([rmspropParam], {
            lr: 0.1,
          }),
          params: [rmspropParam],
        },
        {
          name: "Adagrad",
          opt: new Adagrad([adagradParam], {
            lr: 0.5,
          }),
          params: [adagradParam],
        },
      ];

      for (const { opt, params } of optimizers) {
        const p = params[0];
        expect(p).toBeDefined();
        if (!p) return;
        const initial = getParamValue(p, 0, "optim param");

        for (let i = 0; i < 50; i++) {
          const x = getParamValue(p, 0, "optim param");
          const grad = 2 * x;
          p.setGrad(tensor([grad], { dtype: "float64" }));
          opt.step();
        }

        const final = getParamValue(p, 0, "optim param");
        expect(Math.abs(final)).toBeLessThan(Math.abs(initial));
      }
    });

    it("should all optimizers handle multi-parameter optimization", () => {
      const createParams = () => [
        parameter(tensor([1, 2], { dtype: "float64" })),
        parameter(tensor([3, 4], { dtype: "float64" })),
      ];

      const sgdParams = createParams();
      const adamParams = createParams();
      const adamwParams = createParams();
      const rmspropParams = createParams();
      const adagradParams = createParams();
      const optimizers = [
        {
          name: "SGD",
          opt: new SGD(sgdParams, { lr: 0.01 }),
          params: sgdParams,
        },
        {
          name: "Adam",
          opt: new Adam(adamParams, { lr: 0.01 }),
          params: adamParams,
        },
        {
          name: "AdamW",
          opt: new AdamW(adamwParams, { lr: 0.01 }),
          params: adamwParams,
        },
        {
          name: "RMSprop",
          opt: new RMSprop(rmspropParams, { lr: 0.01 }),
          params: rmspropParams,
        },
        {
          name: "Adagrad",
          opt: new Adagrad(adagradParams, { lr: 0.01 }),
          params: adagradParams,
        },
      ];

      for (const { opt, params } of optimizers) {
        params[0].setGrad(tensor([0.1, 0.1], { dtype: "float64" }));
        params[1].setGrad(tensor([0.1, 0.1], { dtype: "float64" }));

        expect(() => opt.step()).not.toThrow();
      }
    });

    it("should all optimizers support parameter groups", () => {
      const optimizers = [
        {
          name: "SGD",
          opt: new SGD([parameter(tensor([1], { dtype: "float64" }))], {
            lr: 0.01,
          }),
        },
        {
          name: "Adam",
          opt: new Adam([parameter(tensor([1], { dtype: "float64" }))], {
            lr: 0.01,
          }),
        },
        {
          name: "AdamW",
          opt: new AdamW([parameter(tensor([1], { dtype: "float64" }))], {
            lr: 0.01,
          }),
        },
        {
          name: "RMSprop",
          opt: new RMSprop([parameter(tensor([1], { dtype: "float64" }))], {
            lr: 0.01,
          }),
        },
        {
          name: "Adagrad",
          opt: new Adagrad([parameter(tensor([1], { dtype: "float64" }))], {
            lr: 0.01,
          }),
        },
      ];

      for (const { name, opt } of optimizers) {
        const newParams = [parameter(tensor([2], { dtype: "float64" }))];
        expect(
          () => opt.addParamGroup({ params: newParams, lr: 0.001 }),
          `${name} should support adding parameter groups`
        ).not.toThrow();
      }
    });
  });

  describe("State Management", () => {
    it("SGD should maintain momentum state across steps", () => {
      const p = parameter(tensor([1, 1], { dtype: "float64" }));
      const optimizer = new SGD([p], { lr: 0.1, momentum: 0.9 });

      p.setGrad(tensor([1, 1], { dtype: "float64" }));
      optimizer.step();
      const after1 = getParamValue(p, 0, "SGD param");

      p.setGrad(tensor([1, 1], { dtype: "float64" }));
      optimizer.step();
      const after2 = getParamValue(p, 0, "SGD param");

      const change1 = Math.abs(1 - after1);
      const change2 = Math.abs(after1 - after2);
      expect(change2).toBeGreaterThan(change1 * 0.8);
    });

    it("Adam should maintain moment estimates", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const optimizer = new Adam([p], { lr: 0.1 });

      p.setGrad(tensor([1], { dtype: "float64" }));
      optimizer.step();
      p.setGrad(tensor([1], { dtype: "float64" }));
      optimizer.step();

      expect(optimizer.stepCount).toBe(2);
    });

    it("AdamW should maintain separate state per parameter", () => {
      const p1 = parameter(tensor([1], { dtype: "float64" }));
      const p2 = parameter(tensor([2], { dtype: "float64" }));
      const optimizer = new AdamW([p1, p2], { lr: 0.1 });

      p1.setGrad(tensor([0.5], { dtype: "float64" }));
      p2.setGrad(tensor([1.0], { dtype: "float64" }));
      optimizer.step();

      const final1 = getParamValue(p1, 0, "AdamW param");
      const final2 = getParamValue(p2, 0, "AdamW param");
      expect(final1).not.toEqual(final2);
    });

    it("RMSprop should accumulate squared gradients", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const optimizer = new RMSprop([p], { lr: 0.1, alpha: 0.9 });

      p.setGrad(tensor([1], { dtype: "float64" }));
      optimizer.step();
      const after1 = getParamValue(p, 0, "RMSprop param");

      p.setGrad(tensor([1], { dtype: "float64" }));
      optimizer.step();
      const after2 = getParamValue(p, 0, "RMSprop param");

      expect(Math.abs(1 - after1)).toBeGreaterThan(Math.abs(after1 - after2));
    });

    it("Adagrad should accumulate all historical gradients", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const optimizer = new Adagrad([p], { lr: 0.5 });

      const changes: number[] = [];
      let prev = 1;

      for (let i = 0; i < 5; i++) {
        p.setGrad(tensor([1], { dtype: "float64" }));
        optimizer.step();
        const curr = getParamValue(p, 0, "Adagrad param");
        changes.push(Math.abs(prev - curr));
        prev = curr;
      }

      for (let i = 1; i < changes.length; i++) {
        expect(changes[i]).toBeLessThan(changes[i - 1]);
      }
    });
  });

  describe("Gradient Handling", () => {
    it("all optimizers should handle sparse gradients", () => {
      const sgdParam = parameter(tensor([1, 1, 1, 1, 1], { dtype: "float64" }));
      const adamParam = parameter(tensor([1, 1, 1, 1, 1], { dtype: "float64" }));
      const adamwParam = parameter(tensor([1, 1, 1, 1, 1], { dtype: "float64" }));
      const rmspropParam = parameter(tensor([1, 1, 1, 1, 1], { dtype: "float64" }));
      const adagradParam = parameter(tensor([1, 1, 1, 1, 1], { dtype: "float64" }));
      const optimizers = [
        { opt: new SGD([sgdParam], { lr: 0.1 }), param: sgdParam },
        { opt: new Adam([adamParam], { lr: 0.1 }), param: adamParam },
        { opt: new AdamW([adamwParam], { lr: 0.1 }), param: adamwParam },
        { opt: new RMSprop([rmspropParam], { lr: 0.1 }), param: rmspropParam },
        { opt: new Adagrad([adagradParam], { lr: 0.1 }), param: adagradParam },
      ];

      for (const { opt, param: p } of optimizers) {
        p.setGrad(tensor([1, 0, 0, 0, 0], { dtype: "float64" }));
        expect(() => opt.step()).not.toThrow();
      }
    });

    it("all optimizers should handle mixed gradient signs", () => {
      const sgdParam = parameter(tensor([1, 1, 1, 1], { dtype: "float64" }));
      const adamParam = parameter(tensor([1, 1, 1, 1], { dtype: "float64" }));
      const adamwParam = parameter(tensor([1, 1, 1, 1], { dtype: "float64" }));
      const rmspropParam = parameter(tensor([1, 1, 1, 1], { dtype: "float64" }));
      const adagradParam = parameter(tensor([1, 1, 1, 1], { dtype: "float64" }));
      const optimizers = [
        { opt: new SGD([sgdParam], { lr: 0.1 }), param: sgdParam },
        { opt: new Adam([adamParam], { lr: 0.1 }), param: adamParam },
        { opt: new AdamW([adamwParam], { lr: 0.1 }), param: adamwParam },
        { opt: new RMSprop([rmspropParam], { lr: 0.1 }), param: rmspropParam },
        { opt: new Adagrad([adagradParam], { lr: 0.1 }), param: adagradParam },
      ];

      for (const { opt, param: p } of optimizers) {
        p.setGrad(tensor([1, -1, 1, -1], { dtype: "float64" }));
        expect(() => opt.step()).not.toThrow();
      }
    });

    it("all optimizers should handle very small gradients", () => {
      const sgdParam = parameter(tensor([1, 1], { dtype: "float64" }));
      const adamParam = parameter(tensor([1, 1], { dtype: "float64" }));
      const adamwParam = parameter(tensor([1, 1], { dtype: "float64" }));
      const rmspropParam = parameter(tensor([1, 1], { dtype: "float64" }));
      const adagradParam = parameter(tensor([1, 1], { dtype: "float64" }));
      const optimizers = [
        { opt: new SGD([sgdParam], { lr: 0.1 }), param: sgdParam },
        { opt: new Adam([adamParam], { lr: 0.1 }), param: adamParam },
        { opt: new AdamW([adamwParam], { lr: 0.1 }), param: adamwParam },
        { opt: new RMSprop([rmspropParam], { lr: 0.1 }), param: rmspropParam },
        { opt: new Adagrad([adagradParam], { lr: 0.1 }), param: adagradParam },
      ];

      for (const { opt, param: p } of optimizers) {
        p.setGrad(tensor([1e-15, 1e-15], { dtype: "float64" }));
        expect(() => opt.step()).not.toThrow();
      }
    });

    it("all optimizers should handle very large gradients", () => {
      const sgdParam = parameter(tensor([1, 1], { dtype: "float64" }));
      const adamParam = parameter(tensor([1, 1], { dtype: "float64" }));
      const adamwParam = parameter(tensor([1, 1], { dtype: "float64" }));
      const rmspropParam = parameter(tensor([1, 1], { dtype: "float64" }));
      const adagradParam = parameter(tensor([1, 1], { dtype: "float64" }));
      const optimizers = [
        { opt: new SGD([sgdParam], { lr: 0.001 }), param: sgdParam },
        { opt: new Adam([adamParam], { lr: 0.001 }), param: adamParam },
        { opt: new AdamW([adamwParam], { lr: 0.001 }), param: adamwParam },
        {
          opt: new RMSprop([rmspropParam], { lr: 0.001 }),
          param: rmspropParam,
        },
        {
          opt: new Adagrad([adagradParam], { lr: 0.001 }),
          param: adagradParam,
        },
      ];

      for (const { opt, param: p } of optimizers) {
        p.setGrad(tensor([1e6, 1e6], { dtype: "float64" }));
        expect(() => opt.step()).not.toThrow();
      }
    });
  });

  describe("Weight Decay", () => {
    it("SGD should apply L2 weight decay", () => {
      const p = parameter(tensor([1, 1], { dtype: "float64" }));
      const optimizer = new SGD([p], { lr: 0.1, weightDecay: 0.1 });

      p.setGrad(tensor([0, 0], { dtype: "float64" }));
      optimizer.step();

      const data = getParamData(p, "SGD param");
      expect(data[0]).toBeLessThan(1);
    });

    it("Adam should apply coupled weight decay", () => {
      const p = parameter(tensor([1, 1], { dtype: "float64" }));
      const optimizer = new Adam([p], { lr: 0.1, weightDecay: 0.1 });

      p.setGrad(tensor([0.1, 0.1], { dtype: "float64" }));
      optimizer.step();

      const data = getParamData(p, "Adam param");
      expect(data[0]).toBeLessThan(1);
    });

    it("AdamW should apply decoupled weight decay", () => {
      const p = parameter(tensor([1, 1], { dtype: "float64" }));
      const optimizer = new AdamW([p], { lr: 0.1, weightDecay: 0.1 });

      p.setGrad(tensor([0, 0], { dtype: "float64" }));
      optimizer.step();

      const data = getParamData(p, "AdamW param");
      expect(data[0]).toBeLessThan(1);
    });

    it("AdamW weight decay should differ from Adam", () => {
      const p1 = parameter(tensor([1], { dtype: "float64" }));
      const p2 = parameter(tensor([1], { dtype: "float64" }));
      const adam = new Adam([p1], { lr: 0.1, weightDecay: 0.1 });
      const adamw = new AdamW([p2], { lr: 0.1, weightDecay: 0.1 });

      p1.setGrad(tensor([0.1], { dtype: "float64" }));
      p2.setGrad(tensor([0.1], { dtype: "float64" }));

      adam.step();
      adamw.step();

      const adamVal = getParamValue(p1, 0, "Adam param");
      const adamwVal = getParamValue(p2, 0, "AdamW param");
      expect(adamVal).not.toBeCloseTo(adamwVal, 5);
    });
  });

  describe("Learning Rate Variations", () => {
    it("higher learning rate should cause larger updates", () => {
      const p1 = parameter(tensor([1], { dtype: "float64" }));
      const p2 = parameter(tensor([1], { dtype: "float64" }));
      const opt1 = new SGD([p1], { lr: 0.01 });
      const opt2 = new SGD([p2], { lr: 0.1 });

      p1.setGrad(tensor([1], { dtype: "float64" }));
      p2.setGrad(tensor([1], { dtype: "float64" }));

      opt1.step();
      opt2.step();

      const change1 = Math.abs(1 - getParamValue(p1, 0, "SGD param"));
      const change2 = Math.abs(1 - getParamValue(p2, 0, "SGD param"));
      expect(change2).toBeGreaterThan(change1);
    });

    it("parameter groups should allow different learning rates", () => {
      const p1 = parameter(tensor([1], { dtype: "float64" }));
      const p2 = parameter(tensor([1], { dtype: "float64" }));
      const optimizer = new SGD([p1], { lr: 0.01 });
      optimizer.addParamGroup({ params: [p2], lr: 0.1 });

      p1.setGrad(tensor([1], { dtype: "float64" }));
      p2.setGrad(tensor([1], { dtype: "float64" }));
      optimizer.step();

      const change1 = Math.abs(1 - getParamValue(p1, 0, "SGD param"));
      const change2 = Math.abs(1 - getParamValue(p2, 0, "SGD param"));
      expect(change2).toBeGreaterThan(change1);
    });

    it("constructor parameter groups should honor per-group learning rates", () => {
      const p1 = parameter(tensor([1], { dtype: "float64" }));
      const p2 = parameter(tensor([1], { dtype: "float64" }));
      const optimizer = new SGD(
        [
          { params: [p1], lr: 0.01 },
          { params: [p2], lr: 0.1 },
        ],
        { lr: 0.5 }
      );

      p1.setGrad(tensor([1], { dtype: "float64" }));
      p2.setGrad(tensor([1], { dtype: "float64" }));
      optimizer.step();

      const change1 = Math.abs(1 - getParamValue(p1, 0, "SGD param"));
      const change2 = Math.abs(1 - getParamValue(p2, 0, "SGD param"));
      expect(change2).toBeGreaterThan(change1);
    });
  });

  describe("Convergence Behavior", () => {
    it("Adam should converge faster than SGD on non-convex", () => {
      const pSGD = parameter(tensor([5], { dtype: "float64" }));
      const pAdam = parameter(tensor([5], { dtype: "float64" }));
      const sgd = new SGD([pSGD], { lr: 0.01 });
      const adam = new Adam([pAdam], { lr: 0.1 }); // Higher learning rate for Adam

      // Use Rosenbrock-like gradient with varying scales
      for (let i = 0; i < 50; i++) {
        const xSGD = getParamValue(pSGD, 0, "SGD param");
        const xAdam = getParamValue(pAdam, 0, "Adam param");

        // Gradient with varying magnitude (Adam adapts better to this)
        const gradScale = 1 + Math.abs(Math.sin(i * 0.5));
        pSGD.setGrad(tensor([gradScale * 2 * xSGD], { dtype: "float64" }));
        pAdam.setGrad(tensor([gradScale * 2 * xAdam], { dtype: "float64" }));

        sgd.step();
        adam.step();
      }

      const finalSGD = Math.abs(getParamValue(pSGD, 0, "SGD param"));
      const finalAdam = Math.abs(getParamValue(pAdam, 0, "Adam param"));
      expect(finalAdam).toBeLessThan(finalSGD);
    });

    it("RMSprop should adapt to gradient scale", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const optimizer = new RMSprop([p], { lr: 0.1 });

      p.setGrad(tensor([10], { dtype: "float64" }));
      optimizer.step();
      const after1 = getParamValue(p, 0, "RMSprop param");

      p.setGrad(tensor([0.1], { dtype: "float64" }));
      optimizer.step();
      const after2 = getParamValue(p, 0, "RMSprop param");

      expect(after1).not.toEqual(after2);
    });

    it("Adagrad learning rate should decrease over time", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const optimizer = new Adagrad([p], { lr: 1.0 });

      const updates: number[] = [];
      for (let i = 0; i < 10; i++) {
        const before = getParamValue(p, 0, "Adagrad param");
        p.setGrad(tensor([1], { dtype: "float64" }));
        optimizer.step();
        const after = getParamValue(p, 0, "Adagrad param");
        updates.push(Math.abs(after - before));
      }

      for (let i = 1; i < updates.length; i++) {
        expect(updates[i]).toBeLessThanOrEqual(updates[i - 1]);
      }
    });
  });

  describe("Edge Cases and Robustness", () => {
    it("all optimizers should handle single-element tensors", () => {
      const sgdParam = parameter(tensor([1], { dtype: "float64" }));
      const adamParam = parameter(tensor([1], { dtype: "float64" }));
      const adamwParam = parameter(tensor([1], { dtype: "float64" }));
      const rmspropParam = parameter(tensor([1], { dtype: "float64" }));
      const adagradParam = parameter(tensor([1], { dtype: "float64" }));
      const optimizers = [
        { opt: new SGD([sgdParam], { lr: 0.1 }), param: sgdParam },
        { opt: new Adam([adamParam], { lr: 0.1 }), param: adamParam },
        { opt: new AdamW([adamwParam], { lr: 0.1 }), param: adamwParam },
        { opt: new RMSprop([rmspropParam], { lr: 0.1 }), param: rmspropParam },
        { opt: new Adagrad([adagradParam], { lr: 0.1 }), param: adagradParam },
      ];

      for (const { opt, param: p } of optimizers) {
        p.setGrad(tensor([0.1], { dtype: "float64" }));
        expect(() => opt.step()).not.toThrow();
      }
    });

    it("all optimizers should handle large tensors", () => {
      const size = 1000;
      const data = new Array(size).fill(1);
      const gradData = new Array(size).fill(0.01);

      const sgdParam = parameter(tensor(data, { dtype: "float64" }));
      const adamParam = parameter(tensor(data, { dtype: "float64" }));
      const adamwParam = parameter(tensor(data, { dtype: "float64" }));
      const rmspropParam = parameter(tensor(data, { dtype: "float64" }));
      const adagradParam = parameter(tensor(data, { dtype: "float64" }));
      const optimizers = [
        { opt: new SGD([sgdParam], { lr: 0.1 }), param: sgdParam },
        { opt: new Adam([adamParam], { lr: 0.1 }), param: adamParam },
        { opt: new AdamW([adamwParam], { lr: 0.1 }), param: adamwParam },
        { opt: new RMSprop([rmspropParam], { lr: 0.1 }), param: rmspropParam },
        { opt: new Adagrad([adagradParam], { lr: 0.1 }), param: adagradParam },
      ];

      for (const { opt, param: p } of optimizers) {
        p.setGrad(tensor(gradData, { dtype: "float64" }));
        expect(() => opt.step()).not.toThrow();
      }
    });

    it("all optimizers should handle consecutive zero gradients", () => {
      const sgdParam = parameter(tensor([1, 1], { dtype: "float64" }));
      const adamParam = parameter(tensor([1, 1], { dtype: "float64" }));
      const adamwParam = parameter(tensor([1, 1], { dtype: "float64" }));
      const rmspropParam = parameter(tensor([1, 1], { dtype: "float64" }));
      const adagradParam = parameter(tensor([1, 1], { dtype: "float64" }));
      const optimizers = [
        { opt: new SGD([sgdParam], { lr: 0.1 }), param: sgdParam },
        { opt: new Adam([adamParam], { lr: 0.1 }), param: adamParam },
        { opt: new AdamW([adamwParam], { lr: 0.1 }), param: adamwParam },
        { opt: new RMSprop([rmspropParam], { lr: 0.1 }), param: rmspropParam },
        { opt: new Adagrad([adagradParam], { lr: 0.1 }), param: adagradParam },
      ];

      for (const { opt, param: p } of optimizers) {
        for (let i = 0; i < 5; i++) {
          p.setGrad(tensor([0, 0], { dtype: "float64" }));
          expect(() => opt.step()).not.toThrow();
        }
      }
    });

    it("all optimizers should handle alternating gradient directions", () => {
      const sgdParam = parameter(tensor([1], { dtype: "float64" }));
      const adamParam = parameter(tensor([1], { dtype: "float64" }));
      const adamwParam = parameter(tensor([1], { dtype: "float64" }));
      const rmspropParam = parameter(tensor([1], { dtype: "float64" }));
      const adagradParam = parameter(tensor([1], { dtype: "float64" }));
      const optimizers = [
        { opt: new SGD([sgdParam], { lr: 0.1 }), param: sgdParam },
        { opt: new Adam([adamParam], { lr: 0.1 }), param: adamParam },
        { opt: new AdamW([adamwParam], { lr: 0.1 }), param: adamwParam },
        { opt: new RMSprop([rmspropParam], { lr: 0.1 }), param: rmspropParam },
        { opt: new Adagrad([adagradParam], { lr: 0.1 }), param: adagradParam },
      ];

      for (const { opt, param: p } of optimizers) {
        for (let i = 0; i < 10; i++) {
          const grad = i % 2 === 0 ? 0.1 : -0.1;
          p.setGrad(tensor([grad], { dtype: "float64" }));
          expect(() => opt.step()).not.toThrow();
        }
      }
    });
  });

  describe("Closure Support", () => {
    it("all optimizers should support closure functions", () => {
      const sgdParam = parameter(tensor([1], { dtype: "float64" }));
      const adamParam = parameter(tensor([1], { dtype: "float64" }));
      const adamwParam = parameter(tensor([1], { dtype: "float64" }));
      const rmspropParam = parameter(tensor([1], { dtype: "float64" }));
      const adagradParam = parameter(tensor([1], { dtype: "float64" }));
      const optimizers = [
        { opt: new SGD([sgdParam], { lr: 0.1 }), param: sgdParam },
        { opt: new Adam([adamParam], { lr: 0.1 }), param: adamParam },
        { opt: new AdamW([adamwParam], { lr: 0.1 }), param: adamwParam },
        { opt: new RMSprop([rmspropParam], { lr: 0.1 }), param: rmspropParam },
        { opt: new Adagrad([adagradParam], { lr: 0.1 }), param: adagradParam },
      ];

      for (const { opt, param: p } of optimizers) {
        p.setGrad(tensor([0.1], { dtype: "float64" }));

        let closureCalled = false;
        const loss = opt.step(() => {
          closureCalled = true;
          return 0.5;
        });

        expect(closureCalled).toBe(true);
        expect(loss).toBe(0.5);
      }
    });

    it("closure should be called before parameter update", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const optimizer = new SGD([p], { lr: 0.1 });
      p.setGrad(tensor([1], { dtype: "float64" }));

      let paramValueInClosure = 0;
      optimizer.step(() => {
        paramValueInClosure = getParamValue(p, 0, "SGD param");
        return 0.5;
      });

      const paramValueAfter = getParamValue(p, 0, "SGD param");
      expect(paramValueInClosure).toBeCloseTo(1, 5);
      expect(paramValueAfter).toBeLessThan(1);
    });
  });

  describe("Zero Gradient Functionality", () => {
    it("all optimizers should support zeroGrad", () => {
      const optimizers = [
        new SGD([parameter(tensor([1], { dtype: "float64" }))], { lr: 0.1 }),
        new Adam([parameter(tensor([1], { dtype: "float64" }))], { lr: 0.1 }),
        new AdamW([parameter(tensor([1], { dtype: "float64" }))], { lr: 0.1 }),
        new RMSprop([parameter(tensor([1], { dtype: "float64" }))], {
          lr: 0.1,
        }),
        new Adagrad([parameter(tensor([1], { dtype: "float64" }))], {
          lr: 0.1,
        }),
      ];

      for (const opt of optimizers) {
        expect(() => opt.zeroGrad()).not.toThrow();
      }
    });

    it("zeroGrad should clear gradients for all parameters", () => {
      const p1 = parameter(tensor([1], { dtype: "float64" }));
      const p2 = parameter(tensor([2], { dtype: "float64" }));
      const optimizer = new SGD([p1, p2], { lr: 0.1 });

      p1.setGrad(tensor([0.5], { dtype: "float64" }));
      p2.setGrad(tensor([1.0], { dtype: "float64" }));

      optimizer.zeroGrad();

      const grad1 = p1.grad;
      const grad2 = p2.grad;
      expect(grad1).toBeDefined();
      expect(grad2).toBeDefined();
      if (grad1 && grad2) {
        expect(getTensorValue(grad1, 0, "grad1")).toBe(0);
        expect(getTensorValue(grad2, 0, "grad2")).toBe(0);
      }
    });
  });

  describe("Type Safety", () => {
    it("optimizers should only accept float64 parameters", () => {
      const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
      expect(() => new SGD([p], { lr: 0.1 })).not.toThrow();
      expect(() => new Adam([p], { lr: 0.1 })).not.toThrow();
      expect(() => new AdamW([p], { lr: 0.1 })).not.toThrow();
      expect(() => new RMSprop([p], { lr: 0.1 })).not.toThrow();
      expect(() => new Adagrad([p], { lr: 0.1 })).not.toThrow();
    });

    it("optimizers should validate hyperparameter types", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));

      expect(() => new SGD([p], { lr: Number.NaN })).toThrow();
      expect(() => new Adam([p], { lr: Number.POSITIVE_INFINITY })).toThrow();
      expect(() => new AdamW([p], { lr: -1 })).toThrow();
      expect(() => new RMSprop([p], { lr: -0.1 })).toThrow();
      expect(() => new Adagrad([p], { lr: -0.01 })).toThrow();
    });
  });
});
