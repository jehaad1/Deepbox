import { describe, expect, it } from "vitest";
import { DTypeError, InvalidParameterError, NotFittedError } from "../src/core";
import { parameter, tensor } from "../src/ndarray";
import { AdaDelta, Adagrad, Adam, AdamW, Nadam, RMSprop, SGD } from "../src/optim";
import { getParamValue } from "./optim-test-helpers";

describe("deepbox/optim - Edge Cases", () => {
  describe("Missing Gradients", () => {
    it("should throw NotFittedError when gradient is missing - SGD", () => {
      const p = parameter(tensor([1, 2], { dtype: "float64" }));
      const optimizer = new SGD([p], { lr: 0.01 });
      expect(() => optimizer.step()).toThrow(NotFittedError);
    });

    it("should throw NotFittedError when gradient is missing - Adam", () => {
      const p = parameter(tensor([1, 2], { dtype: "float64" }));
      const optimizer = new Adam([p], { lr: 0.01 });
      expect(() => optimizer.step()).toThrow(NotFittedError);
    });

    it("should throw NotFittedError when gradient is missing - AdamW", () => {
      const p = parameter(tensor([1, 2], { dtype: "float64" }));
      const optimizer = new AdamW([p], { lr: 0.01 });
      expect(() => optimizer.step()).toThrow(NotFittedError);
    });

    it("should throw NotFittedError when gradient is missing - RMSprop", () => {
      const p = parameter(tensor([1, 2], { dtype: "float64" }));
      const optimizer = new RMSprop([p], { lr: 0.01 });
      expect(() => optimizer.step()).toThrow(NotFittedError);
    });

    it("should throw NotFittedError when gradient is missing - Adagrad", () => {
      const p = parameter(tensor([1, 2], { dtype: "float64" }));
      const optimizer = new Adagrad([p], { lr: 0.01 });
      expect(() => optimizer.step()).toThrow(NotFittedError);
    });

    it("should throw NotFittedError when gradient is missing - AdaDelta", () => {
      const p = parameter(tensor([1, 2], { dtype: "float64" }));
      const optimizer = new AdaDelta([p], { lr: 1.0 });
      expect(() => optimizer.step()).toThrow(NotFittedError);
    });

    it("should throw NotFittedError when gradient is missing - Nadam", () => {
      const p = parameter(tensor([1, 2], { dtype: "float64" }));
      const optimizer = new Nadam([p], { lr: 0.002 });
      expect(() => optimizer.step()).toThrow(NotFittedError);
    });
  });

  describe("Invalid Hyperparameters", () => {
    it("should throw on negative learning rate", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      expect(() => new SGD([p], { lr: -0.01 })).toThrow(InvalidParameterError);
    });

    it("should throw on NaN learning rate", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      expect(() => new SGD([p], { lr: Number.NaN })).toThrow(InvalidParameterError);
    });

    it("should throw on Infinity learning rate", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      expect(() => new SGD([p], { lr: Number.POSITIVE_INFINITY })).toThrow(InvalidParameterError);
    });

    it("should throw on beta1 >= 1 in Adam", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      expect(() => new Adam([p], { beta1: 1.0 })).toThrow(InvalidParameterError);
    });

    it("should throw on beta1 < 0 in Adam", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      expect(() => new Adam([p], { beta1: -0.1 })).toThrow(InvalidParameterError);
    });

    it("should throw on beta2 >= 1 in AdamW", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      expect(() => new AdamW([p], { beta2: 1.0 })).toThrow(InvalidParameterError);
    });

    it("should throw on epsilon = 0 in Adam", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      expect(() => new Adam([p], { eps: 0 })).toThrow(InvalidParameterError);
    });

    it("should throw on negative momentumDecay in Nadam", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      expect(() => new Nadam([p], { momentumDecay: -0.1 })).toThrow(InvalidParameterError);
    });

    it("should throw on NaN momentumDecay in Nadam", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      expect(() => new Nadam([p], { momentumDecay: Number.NaN })).toThrow(InvalidParameterError);
    });

    it("should throw on Infinity momentumDecay in Nadam", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      expect(() => new Nadam([p], { momentumDecay: Number.POSITIVE_INFINITY })).toThrow(
        InvalidParameterError
      );
    });

    it("should throw on negative epsilon in RMSprop", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      expect(() => new RMSprop([p], { eps: -1e-8 })).toThrow(InvalidParameterError);
    });

    it("should throw on rho >= 1 in AdaDelta", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      expect(() => new AdaDelta([p], { rho: 1.0 })).toThrow(InvalidParameterError);
    });

    it("should throw on rho < 0 in AdaDelta", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      expect(() => new AdaDelta([p], { rho: -0.1 })).toThrow(InvalidParameterError);
    });
  });

  describe("NaN/Inf Gradients", () => {
    it("should throw on NaN gradient - SGD", () => {
      const p = parameter(tensor([1, 2], { dtype: "float64" }));
      p.setGrad(tensor([Number.NaN, 0.5], { dtype: "float64" }));
      const optimizer = new SGD([p], { lr: 0.01 });
      expect(() => optimizer.step()).toThrow(InvalidParameterError);
    });

    it("should throw on Inf gradient - Adam", () => {
      const p = parameter(tensor([1, 2], { dtype: "float64" }));
      p.setGrad(tensor([Number.POSITIVE_INFINITY, 0.5], { dtype: "float64" }));
      const optimizer = new Adam([p], { lr: 0.01 });
      expect(() => optimizer.step()).toThrow(InvalidParameterError);
    });

    it("should throw on -Inf gradient - AdamW", () => {
      const p = parameter(tensor([1, 2], { dtype: "float64" }));
      p.setGrad(tensor([Number.NEGATIVE_INFINITY, 0.5], { dtype: "float64" }));
      const optimizer = new AdamW([p], { lr: 0.01 });
      expect(() => optimizer.step()).toThrow(InvalidParameterError);
    });
  });

  describe("NaN/Inf Parameters", () => {
    it("should throw on NaN parameter - RMSprop", () => {
      const p = parameter(tensor([Number.NaN, 2], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.2], { dtype: "float64" }));
      const optimizer = new RMSprop([p], { lr: 0.01 });
      expect(() => optimizer.step()).toThrow(InvalidParameterError);
    });

    it("should throw on Inf parameter - Adagrad", () => {
      const p = parameter(tensor([Number.POSITIVE_INFINITY, 2], { dtype: "float64" }));
      p.setGrad(tensor([0.1, 0.2], { dtype: "float64" }));
      const optimizer = new Adagrad([p], { lr: 0.01 });
      expect(() => optimizer.step()).toThrow(InvalidParameterError);
    });
  });

  describe("DType Handling", () => {
    it("should support float32 parameters and gradients", () => {
      const createParam = () => parameter(tensor([1, 2], { dtype: "float32" }));
      const optimizers = [
        (() => {
          const p = createParam();
          return { p, opt: new SGD([p], { lr: 0.01 }) };
        })(),
        (() => {
          const p = createParam();
          return { p, opt: new Adam([p], { lr: 0.01 }) };
        })(),
        (() => {
          const p = createParam();
          return { p, opt: new AdamW([p], { lr: 0.01 }) };
        })(),
        (() => {
          const p = createParam();
          return { p, opt: new RMSprop([p], { lr: 0.01 }) };
        })(),
        (() => {
          const p = createParam();
          return { p, opt: new Adagrad([p], { lr: 0.01 }) };
        })(),
        (() => {
          const p = createParam();
          return { p, opt: new AdaDelta([p], { lr: 1.0 }) };
        })(),
        (() => {
          const p = createParam();
          return { p, opt: new Nadam([p], { lr: 0.002 }) };
        })(),
      ];

      for (const { p, opt } of optimizers) {
        p.setGrad(tensor([0.1, -0.2], { dtype: "float32" }));
        expect(() => opt.step()).not.toThrow();
      }
    });

    it("should reject non-float parameters", () => {
      const createParam = () => parameter(tensor([1, 2], { dtype: "int32" }));
      const optimizers = [
        (() => {
          const p = createParam();
          return { p, opt: new SGD([p], { lr: 0.01 }) };
        })(),
        (() => {
          const p = createParam();
          return { p, opt: new Adam([p], { lr: 0.01 }) };
        })(),
        (() => {
          const p = createParam();
          return { p, opt: new AdamW([p], { lr: 0.01 }) };
        })(),
        (() => {
          const p = createParam();
          return { p, opt: new RMSprop([p], { lr: 0.01 }) };
        })(),
        (() => {
          const p = createParam();
          return { p, opt: new Adagrad([p], { lr: 0.01 }) };
        })(),
        (() => {
          const p = createParam();
          return { p, opt: new AdaDelta([p], { lr: 1.0 }) };
        })(),
        (() => {
          const p = createParam();
          return { p, opt: new Nadam([p], { lr: 0.002 }) };
        })(),
      ];

      for (const { p, opt } of optimizers) {
        p.setGrad(tensor([1, 1], { dtype: "int32" }));
        expect(() => opt.step()).toThrow(DTypeError);
      }
    });
  });

  describe("Zero-Sized Tensors", () => {
    it("should handle zero-sized tensor - SGD", () => {
      const p = parameter(tensor([], { dtype: "float64" }));
      p.setGrad(tensor([], { dtype: "float64" }));
      const optimizer = new SGD([p], { lr: 0.01 });
      expect(() => optimizer.step()).not.toThrow();
    });

    it("should handle zero-sized tensor - Adam", () => {
      const p = parameter(tensor([], { dtype: "float64" }));
      p.setGrad(tensor([], { dtype: "float64" }));
      const optimizer = new Adam([p], { lr: 0.01 });
      expect(() => optimizer.step()).not.toThrow();
    });
  });

  describe("Scalar Tensors", () => {
    it("should handle scalar tensor - SGD", () => {
      const p = parameter(tensor([5], { dtype: "float64" }));
      p.setGrad(tensor([1], { dtype: "float64" }));
      const optimizer = new SGD([p], { lr: 0.1 });
      optimizer.step();
      const value = getParamValue(p, 0, "SGD param");
      expect(value).toBe(4.9); // 5 - 0.1 * 1
    });

    it("should handle scalar tensor - Adam", () => {
      const p = parameter(tensor([5], { dtype: "float64" }));
      p.setGrad(tensor([1], { dtype: "float64" }));
      const optimizer = new Adam([p], { lr: 0.1 });
      optimizer.step();
      const value = getParamValue(p, 0, "Adam param");
      expect(value).toBeLessThan(5); // Should decrease
    });
  });

  describe("Closure Parameter", () => {
    it("should call closure and return loss - SGD", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      p.setGrad(tensor([0.1], { dtype: "float64" }));
      const optimizer = new SGD([p], { lr: 0.01 });
      let closureCalled = false;
      const loss = optimizer.step(() => {
        closureCalled = true;
        return 0.5;
      });
      expect(closureCalled).toBe(true);
      expect(loss).toBe(0.5);
    });

    it("should call closure and return loss - Adam", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      p.setGrad(tensor([0.1], { dtype: "float64" }));
      const optimizer = new Adam([p], { lr: 0.01 });
      let closureCalled = false;
      const loss = optimizer.step(() => {
        closureCalled = true;
        return 1.5;
      });
      expect(closureCalled).toBe(true);
      expect(loss).toBe(1.5);
    });

    it("should call closure and return loss - AdaDelta", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      p.setGrad(tensor([0.1], { dtype: "float64" }));
      const optimizer = new AdaDelta([p], { lr: 1.0 });
      let closureCalled = false;
      const loss = optimizer.step(() => {
        closureCalled = true;
        return 2.5;
      });
      expect(closureCalled).toBe(true);
      expect(loss).toBe(2.5);
    });

    it("should call closure and return loss - Nadam", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      p.setGrad(tensor([0.1], { dtype: "float64" }));
      const optimizer = new Nadam([p], { lr: 0.002 });
      let closureCalled = false;
      const loss = optimizer.step(() => {
        closureCalled = true;
        return 3.5;
      });
      expect(closureCalled).toBe(true);
      expect(loss).toBe(3.5);
    });
  });

  describe("Learning Rate Getters/Setters", () => {
    it("should get and set learning rate - SGD", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const optimizer = new SGD([p], { lr: 0.01 });
      expect(optimizer.getLearningRate()).toBe(0.01);
      optimizer.setLearningRate(0.05);
      expect(optimizer.getLearningRate()).toBe(0.05);
    });

    it("should get and set learning rate - Adam", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const optimizer = new Adam([p], { lr: 0.001 });
      expect(optimizer.getLearningRate()).toBe(0.001);
      optimizer.setLearningRate(0.002);
      expect(optimizer.getLearningRate()).toBe(0.002);
    });

    it("should throw on invalid group index", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const optimizer = new SGD([p], { lr: 0.01 });
      expect(() => optimizer.getLearningRate(5)).toThrow(InvalidParameterError);
    });

    it("should throw on negative learning rate in setter", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      const optimizer = new SGD([p], { lr: 0.01 });
      expect(() => optimizer.setLearningRate(-0.01)).toThrow(InvalidParameterError);
    });
  });

  describe("Step Counter", () => {
    it("should track step count - SGD", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      p.setGrad(tensor([0.1], { dtype: "float64" }));
      const optimizer = new SGD([p], { lr: 0.01 });
      expect(optimizer.stepCount).toBe(0);
      optimizer.step();
      expect(optimizer.stepCount).toBe(1);
      optimizer.step();
      expect(optimizer.stepCount).toBe(2);
    });

    it("should track step count - Adam", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      p.setGrad(tensor([0.1], { dtype: "float64" }));
      const optimizer = new Adam([p], { lr: 0.01 });
      expect(optimizer.stepCount).toBe(0);
      optimizer.step();
      expect(optimizer.stepCount).toBe(1);
    });

    it("should track step count - RMSprop", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      p.setGrad(tensor([0.1], { dtype: "float64" }));
      const optimizer = new RMSprop([p], { lr: 0.01 });
      expect(optimizer.stepCount).toBe(0);
      optimizer.step();
      expect(optimizer.stepCount).toBe(1);
    });
  });

  describe("AMSGrad Variant", () => {
    it("should work with AMSGrad enabled - Adam", () => {
      const p = parameter(tensor([5], { dtype: "float64" }));
      p.setGrad(tensor([1], { dtype: "float64" }));
      const optimizer = new Adam([p], { lr: 0.1, amsgrad: true });
      optimizer.step();
      const value = getParamValue(p, 0, "Adam param");
      expect(value).toBeLessThan(5);
    });

    it("should work with AMSGrad enabled - AdamW", () => {
      const p = parameter(tensor([5], { dtype: "float64" }));
      p.setGrad(tensor([1], { dtype: "float64" }));
      const optimizer = new AdamW([p], { lr: 0.1, amsgrad: true });
      optimizer.step();
      const value = getParamValue(p, 0, "AdamW param");
      expect(value).toBeLessThan(5);
    });
  });

  describe("Nesterov Momentum", () => {
    it("should work with Nesterov momentum - SGD", () => {
      const p = parameter(tensor([5], { dtype: "float64" }));
      p.setGrad(tensor([1], { dtype: "float64" }));
      const optimizer = new SGD([p], {
        lr: 0.1,
        momentum: 0.9,
        nesterov: true,
      });
      optimizer.step();
      const value = getParamValue(p, 0, "RMSprop param");
      expect(value).toBeLessThan(5);
    });

    it("should throw on Nesterov without momentum", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      expect(() => new SGD([p], { lr: 0.01, momentum: 0, nesterov: true })).toThrow(
        InvalidParameterError
      );
    });

    it("should throw on Nesterov with dampening", () => {
      const p = parameter(tensor([1], { dtype: "float64" }));
      expect(
        () =>
          new SGD([p], {
            lr: 0.01,
            momentum: 0.9,
            dampening: 0.1,
            nesterov: true,
          })
      ).toThrow(InvalidParameterError);
    });
  });

  describe("Centered RMSprop", () => {
    it("should work with centered variant", () => {
      const p = parameter(tensor([5], { dtype: "float64" }));
      p.setGrad(tensor([1], { dtype: "float64" }));
      const optimizer = new RMSprop([p], { lr: 0.1, centered: true });
      optimizer.step();
      const value = getParamValue(p, 0, "RMSprop param");
      expect(value).toBeLessThan(5);
    });
  });

  describe("RMSprop with Momentum", () => {
    it("should work with momentum", () => {
      const p = parameter(tensor([5], { dtype: "float64" }));
      p.setGrad(tensor([1], { dtype: "float64" }));
      const optimizer = new RMSprop([p], { lr: 0.1, momentum: 0.9 });
      optimizer.step();
      const value = getParamValue(p, 0, "RMSprop param");
      expect(value).toBeLessThan(5);
    });
  });

  describe("Weight Decay", () => {
    it("should apply weight decay - SGD", () => {
      const p = parameter(tensor([5], { dtype: "float64" }));
      p.setGrad(tensor([1], { dtype: "float64" }));
      const pNoDecay = parameter(tensor([5], { dtype: "float64" }));
      const optimizerNoDecay = new SGD([pNoDecay], {
        lr: 0.1,
        weightDecay: 0,
      });
      const optimizerWithDecay = new SGD([p], { lr: 0.1, weightDecay: 0.01 });

      pNoDecay.setGrad(tensor([1], { dtype: "float64" }));
      p.setGrad(tensor([1], { dtype: "float64" }));

      optimizerNoDecay.step();
      optimizerWithDecay.step();

      const valueNoDecay = getParamValue(pNoDecay, 0, "SGD param");
      const valueWithDecay = getParamValue(p, 0, "SGD param");

      // With weight decay, parameter should be smaller (more regularization)
      expect(valueWithDecay).toBeLessThan(valueNoDecay);
    });

    it("should apply weight decay - Adam", () => {
      const p = parameter(tensor([5], { dtype: "float64" }));
      p.setGrad(tensor([1], { dtype: "float64" }));
      const optimizer = new Adam([p], { lr: 0.1, weightDecay: 0.01 });
      optimizer.step();
      const value = getParamValue(p, 0, "Adam param");
      expect(value).toBeLessThan(5);
    });
  });
});
