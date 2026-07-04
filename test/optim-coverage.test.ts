/**
 * Optimizer coverage tests: Lion (hand-verified update rule), LBFGS
 * strong-Wolfe / tolerance paths, and the state / learning-rate surfaces of
 * Adamax, RAdam, ASGD, RMSprop (centered) and Rprop.
 */

import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../src/core";
import { parameter, tensor } from "../src/ndarray";
import { Adamax, ASGD, LBFGS, Lion, RAdam, RMSprop, Rprop } from "../src/optim";
import { getParamValue } from "./optim-test-helpers";

function scalarParam(value: number) {
  return parameter(tensor([value], { dtype: "float64" }));
}

function setGrad(p: ReturnType<typeof scalarParam>, g: number) {
  p.setGrad(tensor([g], { dtype: "float64" }));
}

function value(p: ReturnType<typeof scalarParam>): number {
  return getParamValue(p, 0, "param");
}

/** Drive an optimizer down f(x) = (x - target)^2 and return the final x. */
function descend(
  opt: { step(): unknown; zeroGrad(): void },
  p: ReturnType<typeof scalarParam>,
  target: number,
  steps: number
): number {
  for (let i = 0; i < steps; i++) {
    opt.zeroGrad();
    setGrad(p, 2 * (value(p) - target));
    opt.step();
  }
  return value(p);
}

describe("Lion", () => {
  it("matches the hand-computed sign-momentum update", () => {
    // p0 = 1, constant grad 0.5, lr = 0.1, beta1 = 0.9, beta2 = 0.99:
    //   step 1: u = sign(0.9*0 + 0.1*0.5) = 1      -> p = 0.9;  m = 0.005
    //   step 2: u = sign(0.9*0.005 + 0.1*0.5) = 1  -> p = 0.8;  m = 0.00995
    const p = scalarParam(1);
    const opt = new Lion([p], { lr: 0.1, beta1: 0.9, beta2: 0.99 });

    setGrad(p, 0.5);
    opt.step();
    expect(value(p)).toBeCloseTo(0.9, 12);

    setGrad(p, 0.5);
    opt.step();
    expect(value(p)).toBeCloseTo(0.8, 12);
    expect(opt.stepCount).toBe(2);
  });

  it("applies decoupled weight decay inside the update", () => {
    // p -= lr * (sign(...) + wd * p) = 1 - 0.1 * (1 + 0.1 * 1) = 0.89
    const p = scalarParam(1);
    const opt = new Lion([p], { lr: 0.1, weightDecay: 0.1 });
    setGrad(p, 0.5);
    opt.step();
    expect(value(p)).toBeCloseTo(0.89, 12);
  });

  it("moves opposite to the gradient sign and minimizes a quadratic", () => {
    const p = scalarParam(-2);
    const opt = new Lion([p], { lr: 0.05 });
    const x = descend(opt, p, 3, 200);
    expect(Math.abs(x - 3)).toBeLessThan(0.2);
  });

  it("supports the learning-rate surface and state round-trip", () => {
    const p = scalarParam(1);
    const opt = new Lion([p], { lr: 1e-4 });
    expect(opt.getLearningRate()).toBe(1e-4);
    opt.setLearningRate(3e-4);
    expect(opt.getLearningRate()).toBe(3e-4);
    expect(() => opt.getLearningRate(5)).toThrow(InvalidParameterError);

    setGrad(p, 1);
    opt.step();
    const sd = opt.stateDict();
    const opt2 = new Lion([p], { lr: 3e-4 });
    opt2.loadStateDict(sd);
    expect(opt2.stateDict()).toEqual(sd);
  });

  it("validates hyperparameters", () => {
    const p = scalarParam(1);
    expect(() => new Lion([p], { lr: -1 })).toThrow(InvalidParameterError);
    expect(() => new Lion([p], { beta1: 1.5 })).toThrow(InvalidParameterError);
    expect(() => new Lion([p], { beta2: -0.1 })).toThrow(InvalidParameterError);
    expect(() => new Lion([p], { weightDecay: -1 })).toThrow(InvalidParameterError);
  });
});

describe("LBFGS extended paths", () => {
  function quadClosure(opt: LBFGS, p: ReturnType<typeof scalarParam>) {
    return () => {
      opt.zeroGrad();
      const x = value(p);
      setGrad(p, 2 * (x - 3));
      return (x - 3) ** 2;
    };
  }

  it("converges with strong_wolfe line search", () => {
    const p = scalarParam(-5);
    const opt = new LBFGS([p], { lr: 1, maxIter: 25, lineSearchFn: "strong_wolfe" });
    for (let i = 0; i < 5; i++) opt.step(quadClosure(opt, p));
    expect(value(p)).toBeCloseTo(3, 4);
  });

  it("stops early when the gradient tolerance is met", () => {
    const p = scalarParam(3); // already at the optimum
    const opt = new LBFGS([p], { toleranceGrad: 1e-7 });
    let evals = 0;
    opt.step(() => {
      evals++;
      opt.zeroGrad();
      setGrad(p, 2 * (value(p) - 3));
      return (value(p) - 3) ** 2;
    });
    expect(evals).toBe(1);
    expect(value(p)).toBeCloseTo(3, 10);
  });

  it("works with a small history size on a 2-D quadratic", () => {
    const p = parameter(tensor([4, -2], { dtype: "float64" }));
    const opt = new LBFGS([p], { historySize: 2, maxIter: 30 });
    for (let i = 0; i < 5; i++) {
      opt.step(() => {
        opt.zeroGrad();
        const d = p.tensor.data as Float64Array;
        const [x, y] = [d[0]!, d[1]!];
        p.setGrad(tensor([2 * (x - 1), 8 * (y - 2)], { dtype: "float64" }));
        return (x - 1) ** 2 + 4 * (y - 2) ** 2;
      });
    }
    const d = p.tensor.data as Float64Array;
    expect(d[0]).toBeCloseTo(1, 4);
    expect(d[1]).toBeCloseTo(2, 4);
  });
});

describe("Adamax", () => {
  it("matches the hand-computed update for constant gradients", () => {
    // lr = 0.002, beta1 = 0.9, g = 1 each step, p0 = 1:
    //   step 1: m = 0.1, u = 1, p -= 0.002/(1-0.9) * 0.1/1  = 0.002 -> 0.998
    //   step 2: m = 0.19, u = 1, p -= 0.002/0.19 * 0.19     = 0.002 -> 0.996
    const p = scalarParam(1);
    const opt = new Adamax([p], { lr: 0.002 });
    setGrad(p, 1);
    opt.step();
    expect(value(p)).toBeCloseTo(0.998, 9);
    setGrad(p, 1);
    opt.step();
    expect(value(p)).toBeCloseTo(0.996, 9);
  });

  it("exposes lr accessors and state round-trip", () => {
    const p = scalarParam(1);
    const opt = new Adamax([p]);
    expect(opt.getLearningRate()).toBe(0.002);
    opt.setLearningRate(0.01);
    expect(opt.getLearningRate()).toBe(0.01);
    expect(() => opt.getLearningRate(3)).toThrow(InvalidParameterError);

    setGrad(p, 0.5);
    opt.step();
    const sd = opt.stateDict();
    const opt2 = new Adamax([p]);
    opt2.loadStateDict(sd);
    expect(opt2.stateDict()).toEqual(sd);
  });
});

describe("RAdam", () => {
  it("minimizes a quadratic and round-trips state", () => {
    const p = scalarParam(-4);
    const opt = new RAdam([p], { lr: 0.5 });
    const x = descend(opt, p, 2, 100);
    expect(Math.abs(x - 2)).toBeLessThan(0.15);

    expect(opt.getLearningRate()).toBe(0.5);
    opt.setLearningRate(0.1);
    expect(opt.getLearningRate()).toBe(0.1);
    expect(() => opt.getLearningRate(9)).toThrow(InvalidParameterError);

    const sd = opt.stateDict();
    const opt2 = new RAdam([p]);
    opt2.loadStateDict(sd);
    expect(opt2.stateDict()).toEqual(sd);
  });
});

describe("ASGD", () => {
  it("averages iterates after t0 and converges", () => {
    const p = scalarParam(5);
    const opt = new ASGD([p], { lr: 0.05, t0: 5, lambda: 1e-4 });
    const x = descend(opt, p, 1, 150);
    expect(Math.abs(x - 1)).toBeLessThan(0.2);

    const sd = opt.stateDict();
    const opt2 = new ASGD([p], { lr: 0.05, t0: 5 });
    opt2.loadStateDict(sd);
    expect(opt2.stateDict()).toEqual(sd);
  });
});

describe("RMSprop centered + momentum", () => {
  it("converges with the centered variant and momentum buffer", () => {
    const p = scalarParam(4);
    const opt = new RMSprop([p], { lr: 0.02, momentum: 0.9, centered: true });
    const x = descend(opt, p, -1, 300);
    expect(Math.abs(x - -1)).toBeLessThan(0.2);

    const sd = opt.stateDict();
    const opt2 = new RMSprop([p], { lr: 0.02, momentum: 0.9, centered: true });
    opt2.loadStateDict(sd);
    expect(opt2.stateDict()).toEqual(sd);
  });
});

describe("Rprop", () => {
  it("adapts step sizes through gradient sign changes", () => {
    const p = scalarParam(2.5);
    const opt = new Rprop([p], { lr: 0.1 });
    // Quadratic descent inherently flips gradient signs near the optimum,
    // exercising the eta-minus shrink path.
    const x = descend(opt, p, 0, 100);
    expect(Math.abs(x)).toBeLessThan(0.1);

    const sd = opt.stateDict();
    const opt2 = new Rprop([p], { lr: 0.1 });
    opt2.loadStateDict(sd);
    expect(opt2.stateDict()).toEqual(sd);
  });
});
