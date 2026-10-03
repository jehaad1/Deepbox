/**
 * v1.5.0 regression tests for src/optim/optimizers (AdaDelta, Adagrad, Adam, Adamax,
 * AdamW, ASGD, LAMB, LARS, LBFGS, Lion, Nadam).
 *
 * Reference values come from PyTorch 2.12 (torch.optim.*) and from NumPy 2.4
 * implementations of the published update rules (Lion, LAMB, LARS).
 */
import { describe, expect, it } from "vitest";
import { DataValidationError, InvalidParameterError } from "../../src/core";
import { type GradTensor, parameter, tensor } from "../../src/ndarray";
import {
  AdaDelta,
  Adagrad,
  Adam,
  Adamax,
  AdamW,
  ASGD,
  LAMB,
  LARS,
  LBFGS,
  Lion,
  Nadam,
} from "../../src/optim";

const P0 = [1.0, -2.0, 0.5];
const GRADS = [
  [0.3, -0.4, 0.1],
  [0.25, -0.1, -0.2],
  [-0.1, 0.5, 0.3],
  [0.0, 0.2, -0.6],
];

// PyTorch 2.12 trajectories (float64) for the gradient sequence above.
const TORCH_ADAGRAD_INIT = [
  [0.9804716633554188, -1.9753817018071638, 0.49299859958084036],
  [0.9649970518425802, -1.9692732295900948, 0.506482596828287],
  [0.971140003010165, -1.9953375313445116, 0.48773259683063075],
  [0.971140003010165, -2.0055437386050667, 0.5177325968276307],
];
const TORCH_ADAMAX = [
  [0.99000000025, -1.9900000001666667, 0.4900000006666666],
  [0.9806617148290527, -1.982632984150358, 0.49055768623239665],
  [0.9747812683441932, -1.9798391872293166, 0.4870198787787191],
  [0.9698958236341886, -1.9778655933875204, 0.48833907275984223],
];
const TORCH_NADAM_DECOUPLED = [
  [0.9884354825685967, -1.9874354824805591, 0.4889354832728978],
  [0.9802708449092972, -1.982300542962858, 0.49751212214315227],
  [0.9811478665773785, -1.9885094212675907, 0.4881789433743981],
  [0.9794123478106136, -1.9902773568670462, 0.4979574390339992],
];
const TORCH_ASGD = [
  [0.9795, -1.969, 0.49225],
  [0.9616194547854675, -1.9531764334666624, 0.49953989146152317],
  [0.9613307642749768, -1.9674232792833188, 0.4818057210244264],
  [0.9560493855092307, -1.9666033734684614, 0.5091250610603264],
];
const TORCH_ASGD_AX = [0.9560493855092307, -1.9666033734684614, 0.5091250610603264];
const TORCH_ADAM_AMSGRAD = [
  [0.9900000003333334, -1.99000000025, 0.4900000009999999],
  [0.9800882719464279, -1.9816940251982005, 0.49366103603884887],
  [0.9742522160543945, -1.9825420695138274, 0.49022862539477424],
  [0.9694740650810962, -1.9849208463548087, 0.4936736439397883],
];

// NumPy reference implementations of the published update rules.
const NUMPY_LION = [
  [0.89, -1.88, 0.395],
  [0.7811, -1.7611999999999999, 0.49105000000000004],
  [0.873289, -1.8435879999999998, 0.3861395],
  [0.76455611, -1.9251521199999997, 0.482278105],
];
const NUMPY_LAMB = [
  [0.9868810765457328, -1.9862563535100288, 0.4871935161515824],
  [0.9704354170263267, -1.971558856045129, 0.4925948548346634],
  [0.9509750391132643, -1.9711348225831051, 0.4812697597643456],
  [0.9323140103852439, -1.9760812766473022, 0.49264663361223404],
];
const NUMPY_LARS = [
  [0.9865348158500923, -1.9820165979127478, 0.49550414947818694],
  [0.957553318932906, -1.9589797520101107, 0.5048618326948839],
  [0.9352028176887539, -1.9570174357572265, 0.5019579130265734],
  [0.9150545618331156, -1.9621982375346767, 0.5203733798620488],
];

type Stepper = { step(): unknown };

function makeParam(dtype: "float32" | "float64" = "float64"): GradTensor {
  return parameter(tensor(P0, { dtype }));
}

function values(p: GradTensor): number[] {
  return Array.from(p.tensor.data as Float64Array | Float32Array);
}

function runSequence(p: GradTensor, opt: Stepper, n = GRADS.length): number[][] {
  const out: number[][] = [];
  for (let t = 0; t < n; t++) {
    p.setGrad(tensor(GRADS[t] as number[], { dtype: p.tensor.dtype as "float32" | "float64" }));
    opt.step();
    out.push(values(p));
  }
  return out;
}

function expectTrajectory(actual: number[][], expected: number[][], digits = 12): void {
  expect(actual.length).toBe(expected.length);
  for (let t = 0; t < expected.length; t++) {
    for (let i = 0; i < 3; i++) {
      expect(actual[t]?.[i]).toBeCloseTo((expected[t] as number[])[i] as number, digits);
    }
  }
}

describe("Adagrad initialAccumulatorValue", () => {
  it("matches torch.optim.Adagrad(initial_accumulator_value=0.5)", () => {
    const p = makeParam();
    const opt = new Adagrad([p], { lr: 0.05, initialAccumulatorValue: 0.5 });
    expectTrajectory(runSequence(p, opt), TORCH_ADAGRAD_INIT);
  });

  it("defaults to zero and validates the value", () => {
    const withDefault = makeParam();
    const explicit = makeParam();
    const a = runSequence(withDefault, new Adagrad([withDefault], { lr: 0.05 }));
    const b = runSequence(
      explicit,
      new Adagrad([explicit], { lr: 0.05, initialAccumulatorValue: 0 })
    );
    expect(a).toEqual(b);
    expect(() => new Adagrad([makeParam()], { initialAccumulatorValue: -1 })).toThrow(
      InvalidParameterError
    );
    expect(() => new Adagrad([makeParam()], { initialAccumulatorValue: Number.NaN })).toThrow(
      InvalidParameterError
    );
  });
});

describe("Adamax", () => {
  it("matches torch.optim.Adamax exactly (eps folded into the infinity norm)", () => {
    const p = makeParam();
    const opt = new Adamax([p], { lr: 0.01, weightDecay: 0.1 });
    // The previous u + eps formulation differed from torch at ~1e-12; require 15 digits now.
    expectTrajectory(runSequence(p, opt), TORCH_ADAMAX, 15);
  });

  it("re-validates hyperparameters changed after construction", () => {
    const p = makeParam();
    const opt = new Adamax([p]);
    p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));
    const group = opt.paramGroups[0];
    if (!group) throw new Error("missing group");
    group.options.beta2 = 1.5;
    expect(() => opt.step()).toThrow(InvalidParameterError);
    group.options.beta2 = 0.999;
    group.options.eps = 0;
    expect(() => opt.step()).toThrow(InvalidParameterError);
  });

  it("reports the valid range for an invalid group index", () => {
    const opt = new Adamax([makeParam()]);
    expect(() => opt.getLearningRate(3)).toThrow(/valid range: \[0, 1\)/);
  });
});

describe("Nadam decoupledWeightDecay", () => {
  it("matches torch.optim.NAdam(decoupled_weight_decay=True)", () => {
    const p = makeParam();
    const opt = new Nadam([p], { lr: 0.01, weightDecay: 0.1, decoupledWeightDecay: true });
    expectTrajectory(runSequence(p, opt), TORCH_NADAM_DECOUPLED, 14);
  });

  it("differs from the L2 formulation when weight decay is non-zero", () => {
    const decoupled = makeParam();
    const l2 = makeParam();
    const a = runSequence(
      decoupled,
      new Nadam([decoupled], { lr: 0.01, weightDecay: 0.1, decoupledWeightDecay: true })
    );
    const b = runSequence(l2, new Nadam([l2], { lr: 0.01, weightDecay: 0.1 }));
    expect(Math.abs((a[3]?.[0] ?? 0) - (b[3]?.[0] ?? 0))).toBeGreaterThan(1e-6);
  });
});

describe("Adam / AdamW amsgrad", () => {
  it("matches torch.optim.Adam(amsgrad=True)", () => {
    const p = makeParam();
    expectTrajectory(
      runSequence(p, new Adam([p], { lr: 0.01, amsgrad: true })),
      TORCH_ADAM_AMSGRAD
    );
  });

  it("Adam: enabling amsgrad after the first step creates the max buffer instead of throwing", () => {
    const toggled = makeParam();
    const opt = new Adam([toggled], { lr: 0.01 });
    const plain = makeParam();
    const plainOpt = new Adam([plain], { lr: 0.01 });
    const a = runSequence(toggled, opt, 1);
    const b = runSequence(plain, plainOpt, 1);
    expect(a).toEqual(b);

    const group = opt.paramGroups[0];
    if (!group) throw new Error("missing group");
    group.options.amsgrad = true;
    // With a zero-initialised max buffer the second step equals the plain Adam step.
    toggled.setGrad(tensor(GRADS[1] as number[], { dtype: "float64" }));
    expect(() => opt.step()).not.toThrow();
    plain.setGrad(tensor(GRADS[1] as number[], { dtype: "float64" }));
    plainOpt.step();
    expectTrajectory([values(toggled)], [values(plain)], 15);
  });

  it("AdamW: enabling amsgrad after the first step creates the max buffer instead of throwing", () => {
    const toggled = makeParam();
    const opt = new AdamW([toggled], { lr: 0.01, weightDecay: 0.1 });
    const plain = makeParam();
    const plainOpt = new AdamW([plain], { lr: 0.01, weightDecay: 0.1 });
    runSequence(toggled, opt, 1);
    runSequence(plain, plainOpt, 1);

    const group = opt.paramGroups[0];
    if (!group) throw new Error("missing group");
    group.options.amsgrad = true;
    toggled.setGrad(tensor(GRADS[1] as number[], { dtype: "float64" }));
    expect(() => opt.step()).not.toThrow();
    plain.setGrad(tensor(GRADS[1] as number[], { dtype: "float64" }));
    plainOpt.step();
    expectTrajectory([values(toggled)], [values(plain)], 15);
  });
});

describe("ASGD", () => {
  it("matches torch.optim.ASGD including the running average", () => {
    const p = makeParam();
    const opt = new ASGD([p], { lr: 0.05, lambda: 0.01, alpha: 0.75, t0: 2, weightDecay: 0.1 });
    expectTrajectory(runSequence(p, opt), TORCH_ASGD, 14);
    const averaged = opt.averagedParameters();
    expect(averaged).toHaveLength(1);
    const avg = averaged[0];
    if (!avg) throw new Error("missing averaged parameter");
    expect(avg.shape).toEqual([3]);
    const data = Array.from(avg.data as Float64Array);
    for (let i = 0; i < 3; i++) {
      expect(data[i]).toBeCloseTo(TORCH_ASGD_AX[i] as number, 14);
    }
  });

  it("averagedParameters returns copies with the parameter's shape and dtype", () => {
    const w = parameter(
      tensor(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        { dtype: "float32" }
      )
    );
    const opt = new ASGD([w], { lr: 0.1 });
    const before = opt.averagedParameters()[0];
    expect(before?.shape).toEqual([2, 3]);
    expect(before?.dtype).toBe("float32");
    w.setGrad(
      tensor(
        [
          [1, 1, 1],
          [1, 1, 1],
        ],
        { dtype: "float32" }
      )
    );
    opt.step();
    const after = opt.averagedParameters()[0];
    // t0 defaults to 1e6, so the average equals the latest iterate.
    expect(Array.from(after?.data as Float32Array)).toEqual(
      Array.from(w.tensor.data as Float32Array)
    );
    // Mutating the copy must not touch the live parameter.
    (after?.data as Float32Array)[0] = 99;
    expect((w.tensor.data as Float32Array)[0]).not.toBe(99);
  });

  it("re-validates hyperparameters at step time", () => {
    const p = makeParam();
    const opt = new ASGD([p]);
    p.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));
    const group = opt.paramGroups[0];
    if (!group) throw new Error("missing group");
    group.options.lambda = -1;
    expect(() => opt.step()).toThrow(InvalidParameterError);
  });

  it("rejects state with non-numeric counters", () => {
    const p = makeParam();
    const opt = new ASGD([p]);
    const saved = opt.stateDict();
    const bad = {
      ...saved,
      state: [{ paramId: 0, state: { step: "x", eta: 0.1, mu: 1 } }],
    };
    expect(() => opt.loadStateDict(bad)).toThrow(DataValidationError);
  });
});

describe("Lion", () => {
  it("matches the NumPy reference of the Lion update rule", () => {
    const p = makeParam();
    const opt = new Lion([p], { lr: 0.1, beta1: 0.9, beta2: 0.99, weightDecay: 0.1 });
    expectTrajectory(runSequence(p, opt), NUMPY_LION, 14);
  });

  it("accepts lr = 0, like setLearningRate and every other optimizer", () => {
    const p = makeParam();
    const opt = new Lion([p], { lr: 0 });
    p.setGrad(tensor([1, 1, 1], { dtype: "float64" }));
    opt.step();
    expect(values(p)).toEqual(P0);
    expect(() => new Lion([makeParam()], { lr: -1e-3 })).toThrow(InvalidParameterError);
  });

  it("throws on a non-finite gradient instead of writing NaN into the parameter", () => {
    const p = makeParam();
    const opt = new Lion([p], { lr: 0.1 });
    p.setGrad(tensor([0.1, Number.NaN, 0.1], { dtype: "float64" }));
    expect(() => opt.step()).toThrow(InvalidParameterError);
    expect(values(p).every(Number.isFinite)).toBe(true);
  });

  it("works on float32 parameters", () => {
    const p = makeParam("float32");
    const opt = new Lion([p], { lr: 0.1, weightDecay: 0.1 });
    expectTrajectory(runSequence(p, opt), NUMPY_LION, 5);
  });
});

describe("LAMB", () => {
  it("matches the NumPy reference of the LAMB update rule", () => {
    const p = makeParam();
    const opt = new LAMB([p], { lr: 0.01, weightDecay: 0.05 });
    expectTrajectory(runSequence(p, opt), NUMPY_LAMB, 14);
  });

  it("works on float32 parameters", () => {
    const p = makeParam("float32");
    const opt = new LAMB([p], { lr: 0.01, weightDecay: 0.05 });
    expectTrajectory(runSequence(p, opt), NUMPY_LAMB, 5);
  });
});

describe("LARS", () => {
  it("matches the NumPy reference of the LARS update rule", () => {
    const p = makeParam();
    const opt = new LARS([p], { lr: 0.5, momentum: 0.9, weightDecay: 1e-3, eta: 0.02 });
    expectTrajectory(runSequence(p, opt), NUMPY_LARS, 14);
  });
});

describe("AdaDelta / Adagrad / Adam non-finite input", () => {
  it.each([
    ["AdaDelta", (p: GradTensor) => new AdaDelta([p])],
    ["Adagrad", (p: GradTensor) => new Adagrad([p])],
    ["Adam", (p: GradTensor) => new Adam([p])],
    ["AdamW", (p: GradTensor) => new AdamW([p])],
    ["Adamax", (p: GradTensor) => new Adamax([p])],
    ["Nadam", (p: GradTensor) => new Nadam([p])],
    ["ASGD", (p: GradTensor) => new ASGD([p])],
    ["LAMB", (p: GradTensor) => new LAMB([p])],
    ["LARS", (p: GradTensor) => new LARS([p])],
    ["Lion", (p: GradTensor) => new Lion([p])],
  ])("%s rejects NaN gradients and infinite parameters", (_name, make) => {
    const p = makeParam();
    const opt = make(p);
    p.setGrad(tensor([0.1, Number.POSITIVE_INFINITY, 0.1], { dtype: "float64" }));
    expect(() => opt.step()).toThrow(InvalidParameterError);

    const q = parameter(tensor([1, Number.NaN, 3], { dtype: "float64" }));
    const opt2 = make(q);
    q.setGrad(tensor([0.1, 0.1, 0.1], { dtype: "float64" }));
    expect(() => opt2.step()).toThrow(InvalidParameterError);
  });
});

describe("LBFGS", () => {
  function rosenbrock(p: GradTensor) {
    return () => {
      p.zeroGrad();
      const [x, y] = Array.from(p.tensor.data as Float64Array) as [number, number];
      p.setGrad(
        tensor([-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)], { dtype: "float64" })
      );
      return (1 - x) ** 2 + 100 * (y - x * x) ** 2;
    };
  }

  it("strong_wolfe follows torch.optim.LBFGS step for step (iterates and evaluation counts)", () => {
    const p = parameter(tensor([-1.5, 2.0], { dtype: "float64" }));
    const opt = new LBFGS([p], {
      lr: 1,
      maxIter: 20,
      historySize: 100,
      lineSearchFn: "strong_wolfe",
    });
    let evals = 0;
    const closure = rosenbrock(p);
    const counted = () => {
      evals++;
      return closure();
    };
    opt.step(counted);
    expect(evals).toBe(25);
    expect(values(p)[0]).toBeCloseTo(-0.08511855589860293, 10);
    expect(values(p)[1]).toBeCloseTo(-0.028757058413769698, 10);
    opt.step(counted);
    expect(evals).toBe(50);
    expect(values(p)[0]).toBeCloseTo(0.9995835472529905, 10);
    expect(values(p)[1]).toBeCloseTo(0.9991550319577943, 10);
  });

  it("strong_wolfe really changes the search (it is no longer ignored)", () => {
    const run = (lineSearchFn: "strong_wolfe" | null) => {
      const p = parameter(tensor([-1.5, 2.0], { dtype: "float64" }));
      const opt = new LBFGS([p], { maxIter: 20, historySize: 100, lineSearchFn });
      let evals = 0;
      const closure = rosenbrock(p);
      opt.step(() => {
        evals++;
        return closure();
      });
      return { evals, x: values(p) };
    };
    const wolfe = run("strong_wolfe");
    const plain = run(null);
    expect(wolfe.evals).not.toBe(plain.evals);
    expect(wolfe.x[0]).not.toBeCloseTo(plain.x[0] as number, 6);
  });

  it("backtracking never increases the loss across steps", () => {
    const p = parameter(tensor([-1.5, 2.0], { dtype: "float64" }));
    const opt = new LBFGS([p], { historySize: 100 });
    const closure = rosenbrock(p);
    let last = closure();
    for (let i = 0; i < 4; i++) {
      const loss = opt.step(closure) as number;
      expect(loss).toBeLessThanOrEqual(last + 1e-12);
      last = loss;
    }
    expect(last).toBeLessThan(1e-6);
  });

  it("scales the first step so a steep objective still makes progress", () => {
    // f(x) = 1e8 * x^2 -> g(0.5) = 1e8; an unscaled lr = 1 step needs >10 halvings.
    const p = parameter(tensor([0.5], { dtype: "float64" }));
    const opt = new LBFGS([p], { maxIter: 5 });
    const loss = opt.step(() => {
      p.zeroGrad();
      const x = values(p)[0] as number;
      p.setGrad(tensor([2e8 * x], { dtype: "float64" }));
      return 1e8 * x * x;
    }) as number;
    expect(loss).toBeLessThan(1e8 * 0.25);
  });

  it("defaults maxEval to floor(maxIter * 5 / 4) like torch", () => {
    const p = parameter(tensor([0], { dtype: "float64" }));
    const opt = new LBFGS([p], { maxIter: 100 });
    const group = opt.paramGroups[0];
    expect(group?.options.maxEval).toBe(125);
    const explicit = new LBFGS([p], { maxIter: 100, maxEval: 7 });
    expect(explicit.paramGroups[0]?.options.maxEval).toBe(7);
  });

  it("validates integer iteration budgets", () => {
    const p = () => parameter(tensor([0], { dtype: "float64" }));
    expect(() => new LBFGS([p()], { maxIter: 1.5 })).toThrow(InvalidParameterError);
    expect(() => new LBFGS([p()], { maxEval: 0 })).toThrow(InvalidParameterError);
    expect(() => new LBFGS([p()], { maxEval: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new LBFGS([p()], { historySize: 2.5 })).toThrow(InvalidParameterError);
    expect(() => new LBFGS([p()], { maxIter: Number.NaN })).toThrow(InvalidParameterError);
  });

  it("accepts Infinity as an unlimited iteration or evaluation budget", () => {
    const p = parameter(tensor([3, -4], { dtype: "float64" }));
    const opt = new LBFGS([p], { maxIter: Number.POSITIVE_INFINITY, lineSearchFn: "strong_wolfe" });
    expect(opt.paramGroups[0]?.options.maxEval).toBe(Number.POSITIVE_INFINITY);
    const loss = opt.step(() => {
      const [x, y] = values(p) as [number, number];
      p.setGrad(tensor([2 * x, 2 * y], { dtype: "float64" }));
      return x * x + y * y;
    }) as number;
    expect(loss).toBeLessThan(1e-12);
    expect(() => new LBFGS([p], { maxEval: Number.POSITIVE_INFINITY })).not.toThrow();
  });

  it("treats a NaN gradient as an error, not as convergence", () => {
    const p = parameter(tensor([1, 2], { dtype: "float64" }));
    const opt = new LBFGS([p]);
    expect(() =>
      opt.step(() => {
        p.zeroGrad();
        p.setGrad(tensor([Number.NaN, 1], { dtype: "float64" }));
        return 1;
      })
    ).toThrow(InvalidParameterError);
  });

  it("backs off from a trial point where the loss overflows (strong_wolfe)", () => {
    // f(x) = exp(x) for x < 5, +Infinity beyond; the minimum is at x -> -inf.
    const p = parameter(tensor([4], { dtype: "float64" }));
    const opt = new LBFGS([p], { lineSearchFn: "strong_wolfe", maxIter: 5 });
    const loss = opt.step(() => {
      p.zeroGrad();
      const x = values(p)[0] as number;
      if (x >= 5) {
        p.setGrad(tensor([Number.POSITIVE_INFINITY], { dtype: "float64" }));
        return Number.POSITIVE_INFINITY;
      }
      p.setGrad(tensor([Math.exp(x)], { dtype: "float64" }));
      return Math.exp(x);
    }) as number;
    expect(Number.isFinite(loss)).toBe(true);
    expect(loss).toBeLessThan(Math.exp(4));
    expect(Number.isFinite(values(p)[0])).toBe(true);
  });

  it("rejects parameter groups whose options differ instead of silently using group 0", () => {
    const a = parameter(tensor([1], { dtype: "float64" }));
    const b = parameter(tensor([2], { dtype: "float64" }));
    const opt = new LBFGS([{ params: [a] }, { params: [b], lr: 0.1 }]);
    const closure = () => {
      a.setGrad(tensor([1], { dtype: "float64" }));
      b.setGrad(tensor([1], { dtype: "float64" }));
      return 1;
    };
    expect(() => opt.step(closure)).toThrow(/per-group options/);
  });

  it("still optimizes several parameter groups that share options", () => {
    const a = parameter(tensor([4], { dtype: "float64" }));
    const b = parameter(tensor([-3], { dtype: "float64" }));
    const opt = new LBFGS([{ params: [a] }, { params: [b] }], { maxIter: 20 });
    const closure = () => {
      const x = values(a)[0] as number;
      const y = values(b)[0] as number;
      a.setGrad(tensor([2 * (x - 1)], { dtype: "float64" }));
      b.setGrad(tensor([2 * (y + 2)], { dtype: "float64" }));
      return (x - 1) ** 2 + (y + 2) ** 2;
    };
    for (let i = 0; i < 3; i++) opt.step(closure);
    expect(values(a)[0]).toBeCloseTo(1, 5);
    expect(values(b)[0]).toBeCloseTo(-2, 5);
  });

  it("round-trips the curvature history through stateDict/loadStateDict", () => {
    const options = { maxIter: 4, historySize: 100, lineSearchFn: "strong_wolfe" as const };
    // Reference: two consecutive steps without interruption.
    const ref = parameter(tensor([-1.5, 2.0], { dtype: "float64" }));
    const refOpt = new LBFGS([ref], options);
    refOpt.step(rosenbrock(ref));
    const snapshot = refOpt.stateDict();
    refOpt.step(rosenbrock(ref));

    // Resume: a fresh optimizer on a copy of the parameters after step one.
    const resumedParam = parameter(tensor([-1.5, 2.0], { dtype: "float64" }));
    const firstOpt = new LBFGS([resumedParam], options);
    firstOpt.step(rosenbrock(resumedParam));
    const resumed = new LBFGS([resumedParam], options);
    resumed.loadStateDict(snapshot as unknown as Record<string, unknown>);
    resumed.step(rosenbrock(resumedParam));
    expect(values(resumedParam)).toEqual(values(ref));

    // Without the history the second step differs.
    const coldParam = parameter(tensor([-1.5, 2.0], { dtype: "float64" }));
    const coldFirst = new LBFGS([coldParam], options);
    coldFirst.step(rosenbrock(coldParam));
    const cold = new LBFGS([coldParam], options);
    cold.step(rosenbrock(coldParam));
    expect(values(coldParam)).not.toEqual(values(ref));
  });

  it("rejects a malformed lbfgs state entry", () => {
    const p = parameter(tensor([1, 2], { dtype: "float64" }));
    const opt = new LBFGS([p]);
    const good = opt.stateDict() as unknown as Record<string, unknown>;
    expect(() => opt.loadStateDict({ ...good, lbfgs: 3 })).toThrow(DataValidationError);
    expect(() =>
      opt.loadStateDict({
        ...good,
        lbfgs: {
          stepCount: 1,
          sHistory: [new Float64Array(5)],
          yHistory: [new Float64Array(5)],
          rhoHistory: [1],
          prevFlatGrad: null,
          prevFlatParams: null,
        },
      })
    ).toThrow(DataValidationError);
  });

  it("a rejected loadStateDict leaves the optimizer untouched", () => {
    const options = { maxIter: 4, historySize: 100, lineSearchFn: "strong_wolfe" as const };
    const p = parameter(tensor([-1.5, 2.0], { dtype: "float64" }));
    const opt = new LBFGS([p], options);
    opt.step(rosenbrock(p));
    const before = opt.stateDict().lbfgs;
    expect(before.sHistory.length).toBeGreaterThan(0);
    const good = opt.stateDict() as unknown as Record<string, unknown>;
    expect(() => opt.loadStateDict({ ...good, lbfgs: { stepCount: -1 } })).toThrow(
      DataValidationError
    );
    expect(opt.stateDict().lbfgs).toEqual(before);
  });
});
