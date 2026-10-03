/**
 * Regression tests for the v1.5.0 review of RAdam, RMSprop, Rprop, SGD, SparseAdam and the
 * learning rate schedulers.
 *
 * Reference trajectories come from PyTorch 2.12 (float64): a three element parameter
 * starting at [1, -2, 0.5] receives the fixed gradient sequence GRADS below.
 * Scheduler references come from torch.optim.lr_scheduler.
 */
import { describe, expect, it } from "vitest";
import { DataValidationError, InvalidParameterError } from "../../src/core";
import { type GradTensor, parameter, tensor } from "../../src/ndarray";
import {
  CosineAnnealingWarmRestarts,
  CyclicLR,
  ExponentialLR,
  LambdaLR,
  LinearLR,
  OneCycleLR,
  RAdam,
  ReduceLROnPlateau,
  RMSprop,
  Rprop,
  SequentialLR,
  SGD,
  SparseAdam,
  StepLR,
  WarmupLR,
} from "../../src/optim";

const GRADS = [
  [0.5, -0.3, 0.0],
  [0.4, 0.2, 0.1],
  [-0.2, 0.25, 0.0],
  [0.3, -0.1, -0.05],
  [0.1, 0.1, 0.02],
  [-0.4, 0.05, 0.0],
  [0.2, -0.2, 0.3],
  [0.05, 0.3, -0.1],
  [0.6, -0.5, 0.2],
  [-0.1, 0.4, 0.0],
];

type Stepper = { step(): unknown };

function trajectory(make: (p: GradTensor[]) => Stepper, steps: number): number[][] {
  const p = parameter(tensor([1, -2, 0.5], { dtype: "float64" }));
  const opt = make([p]);
  const out: number[][] = [];
  for (let t = 0; t < steps; t++) {
    p.setGrad(tensor(GRADS[t] ?? [], { dtype: "float64" }));
    opt.step();
    out.push(Array.from(p.tensor.data as Float64Array));
  }
  return out;
}

function expectTrajectory(actual: number[][], expected: number[][]): void {
  expect(actual.length).toBe(expected.length);
  for (let t = 0; t < expected.length; t++) {
    for (let j = 0; j < 3; j++) {
      expect(actual[t]?.[j]).toBeCloseTo(expected[t]?.[j] ?? Number.NaN, 9);
    }
  }
}

function mockOptimizer(...lrs: number[]) {
  return { paramGroups: lrs.map((lr) => ({ params: [{}] as unknown[], lr })) };
}

function lrSequence(
  opt: { paramGroups: Array<{ lr: number }> },
  sched: { step(): void },
  n: number
): number[] {
  const out = [opt.paramGroups[0]?.lr ?? Number.NaN];
  for (let i = 1; i < n; i++) {
    sched.step();
    out.push(opt.paramGroups[0]?.lr ?? Number.NaN);
  }
  return out;
}

function expectClose(actual: number[], expected: number[], digits = 10): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i] ?? Number.NaN, digits);
  }
}

describe("RMSprop centered variant matches PyTorch", () => {
  it("uses sqrt(variance) + eps as the denominator", () => {
    expectTrajectory(
      trajectory((p) => new RMSprop(p, { lr: 0.05, alpha: 0.9, eps: 1e-3, centered: true }), 6),
      [
        [0.834437086093, -1.835164835165, 0.5],
        [0.721981121497, -1.925436243282, 0.338709677419],
        [0.774806686044, -2.021164885935, 0.338709677419],
        [0.697486352405, -1.982399738524, 0.413538198665],
        [0.670339617073, -2.022211817471, 0.382445382419],
        [0.758359262986, -2.043109237817, 0.382445382419],
      ]
    );
  });

  it("matches with momentum and weight decay", () => {
    expectTrajectory(
      trajectory(
        (p) =>
          new RMSprop(p, {
            lr: 0.02,
            alpha: 0.95,
            momentum: 0.9,
            centered: true,
            weightDecay: 0.1,
          }),
        8
      ),
      [
        [0.908233713469, -1.908233714873, 0.408233790662],
        [0.765149934922, -1.827367787964, 0.237528319615],
        [0.651547044934, -1.767323109025, 0.068720809874],
        [0.507184508648, -1.666104646297, -0.056788043072],
        [0.359541630109, -1.563382809707, -0.17873642745],
        [0.264911404816, -1.452082207381, -0.277160587569],
        [0.155875901442, -1.298328917696, -0.448782981117],
        [0.050653421177, -1.185229205835, -0.563716653039],
      ]
    );
  });

  it("supports maximize", () => {
    expectTrajectory(
      trajectory((p) => new RMSprop(p, { lr: 0.02, maximize: true }), 5),
      [
        [1.19999996, -2.199999933333, 0.5],
        [1.325321612366, -2.088673896319, 0.6999998],
        [1.265252690281, -1.974059217646, 0.6999998],
        [1.347747600018, -2.018959864321, 0.609836582928],
        [1.375124293407, -1.97493965877, 0.645502548132],
      ]
    );
  });
});

describe("RAdam matches PyTorch", () => {
  it("places eps next to sqrt(v) in the rectified branch", () => {
    expectTrajectory(
      trajectory((p) => new RAdam(p, { lr: 0.05, beta2: 0.9, eps: 1e-3 }), 10),
      [
        [0.975, -1.985, 0.5],
        [0.952631578947, -1.983157894737, 0.497368421053],
        [0.942207224704, -1.986608079239, 0.495707904447],
        [0.930452354102, -1.987601100466, 0.495257192031],
        [0.92034698427, -1.989572602993, 0.494672346728],
        [0.917550028622, -1.99257574048, 0.492072726063],
        [0.912414684561, -1.992154789453, 0.48433230856],
        [0.906288915276, -1.996762701172, 0.479013059944],
        [0.89551125026, -1.99349501352, 0.469437958119],
        [0.885772528795, -1.99570471152, 0.459705117749],
      ]
    );
  });

  it("matches with L2 weight decay", () => {
    expectTrajectory(
      trajectory((p) => new RAdam(p, { lr: 0.05, beta2: 0.9, eps: 1e-3, weightDecay: 0.1 }), 10),
      [
        [0.97, -1.975, 0.4975],
        [0.942710526316, -1.963223684211, 0.492375],
        [0.927441687706, -1.956783234609, 0.488232702952],
        [0.91090260489, -1.947916653407, 0.485312020773],
        [0.896069114071, -1.940057861499, 0.482267806097],
        [0.89030900286, -1.931574228363, 0.472207102789],
        [0.881482596848, -1.919381042343, 0.460962455556],
        [0.870663404575, -1.907934719023, 0.450181651965],
        [0.856306028105, -1.893448037035, 0.435857582189],
        [0.841853508326, -1.881478303135, 0.420345015393],
      ]
    );
  });

  it("supports decoupled weight decay", () => {
    expectTrajectory(
      trajectory(
        (p) =>
          new RAdam(p, {
            lr: 0.05,
            beta2: 0.9,
            eps: 1e-3,
            weightDecay: 0.1,
            decoupledWeightDecay: true,
          }),
        10
      ),
      [
        [0.97, -1.975, 0.4975],
        [0.942781578947, -1.963282894737, 0.492380921053],
        [0.927643316809, -1.956916664765, 0.488258499842],
        [0.911250229623, -1.948125102668, 0.485366494927],
        [0.896588608643, -1.940355979682, 0.482354817149],
        [0.889308709952, -1.933657337271, 0.477343422398],
        [0.879726822341, -1.923568099557, 0.467216287783],
        [0.869202418944, -1.918558170778, 0.459560957728],
        [0.854078741834, -1.905697692273, 0.447688051115],
        [0.84006962666, -1.898378901811, 0.435716770489],
      ]
    );
  });

  it("supports maximize", () => {
    expectTrajectory(
      trajectory((p) => new RAdam(p, { lr: 0.05, beta2: 0.9, maximize: true }), 10),
      [
        [1.025, -2.015, 0.5],
        [1.047368421053, -2.016842105263, 0.502631578947],
        [1.057792775296, -2.013391920761, 0.504292095553],
        [1.069547645898, -2.012398899534, 0.504742807969],
        [1.07965301573, -2.010427397007, 0.505327653272],
        [1.082462193796, -2.007399281334, 0.508014188199],
        [1.087620248551, -2.00782345997, 0.51583277444],
        [1.093774496769, -2.003185994009, 0.521205687655],
        [1.104590841065, -2.00646877387, 0.530865282322],
        [1.114366142959, -2.00424991568, 0.540688653182],
      ]
    );
  });

  it("uses the SGD fallback for the first steps with default betas", () => {
    expectTrajectory(
      trajectory((p) => new RAdam(p, { lr: 0.01 }), 6),
      [
        [0.995, -1.997, 0.5],
        [0.990526315789, -1.996631578947, 0.499473684211],
        [0.988441444941, -1.997321615848, 0.499141580889],
        [0.98609047082, -1.997520220093, 0.499051438406],
        [0.984069396854, -1.997914520599, 0.498934469346],
        [0.984014130449, -1.997971420741, 0.498883208007],
      ]
    );
  });

  it("validates group options at construction and again at step time", () => {
    const p = parameter(tensor([1], { dtype: "float64" }));
    p.setGrad(tensor([1], { dtype: "float64" }));
    expect(() => new RAdam([{ params: [p], beta1: 1.5 }])).toThrow(InvalidParameterError);
    const opt = new RAdam([{ params: [p] }]);
    opt.paramGroups[0]!.options.beta1 = 1.5;
    expect(() => opt.step()).toThrow(InvalidParameterError);
  });
});

describe("Rprop matches PyTorch", () => {
  it("clamps the initial step size into [stepMin, stepMax]", () => {
    expectTrajectory(
      trajectory((p) => new Rprop(p, { lr: 100 }), 4),
      [
        [-49.0, 48.0, 0.5],
        [-99.0, 48.0, -49.5],
        [-99.0, 23.0, -49.5],
        [-124.0, 23.0, 0.5],
      ]
    );
  });

  it("follows the sign-change rule with custom bounds", () => {
    expectTrajectory(
      trajectory(
        (p) => new Rprop(p, { lr: 0.1, etaMinus: 0.5, etaPlus: 1.2, stepMin: 0.01, stepMax: 0.3 }),
        10
      ),
      [
        [0.9, -1.9, 0.5],
        [0.78, -1.9, 0.4],
        [0.78, -1.95, 0.4],
        [0.72, -1.95, 0.5],
        [0.648, -1.975, 0.5],
        [0.648, -2.005, 0.5],
        [0.612, -2.005, 0.45],
        [0.5688, -2.02, 0.45],
        [0.51696, -2.02, 0.425],
        [0.51696, -2.03, 0.425],
      ]
    );
  });

  it("supports maximize", () => {
    expectTrajectory(
      trajectory((p) => new Rprop(p, { lr: 0.1, maximize: true }), 10),
      [
        [1.1, -2.1, 0.5],
        [1.22, -2.1, 0.6],
        [1.22, -2.05, 0.6],
        [1.28, -2.05, 0.5],
        [1.352, -2.025, 0.5],
        [1.352, -1.995, 0.5],
        [1.388, -1.995, 0.55],
        [1.4312, -1.98, 0.55],
        [1.48304, -1.98, 0.575],
        [1.48304, -1.9725, 0.575],
      ]
    );
  });

  it("clamps a growing step into updated bounds like torch", () => {
    const p = parameter(tensor([1], { dtype: "float64" }));
    const opt = new Rprop([p], { lr: 0.01 });
    p.setGrad(tensor([1], { dtype: "float64" }));
    opt.step();
    const group = opt.paramGroups[0];
    if (!group) throw new Error("missing group");
    group.options.stepMin = 0.5;
    group.options.stepMax = 1;
    opt.step();
    expect(Array.from(p.tensor.data as Float64Array)[0]).toBeCloseTo(0.49, 12);
  });

  it("rejects NaN and non-finite eta values", () => {
    const p = parameter(tensor([1], { dtype: "float64" }));
    expect(() => new Rprop([p], { etaMinus: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new Rprop([p], { etaPlus: Number.NaN })).toThrow(InvalidParameterError);
    expect(() => new Rprop([p], { etaPlus: Number.POSITIVE_INFINITY })).toThrow(
      InvalidParameterError
    );
  });

  it("rejects stepMin above stepMax", () => {
    const p = parameter(tensor([1], { dtype: "float64" }));
    expect(() => new Rprop([p], { stepMin: 1, stepMax: 0.5 })).toThrow(/stepMin/);
  });

  it("rejects a loaded state whose buffers have the wrong size", () => {
    const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
    p.setGrad(tensor([1, 1, 1], { dtype: "float64" }));
    const opt = new Rprop([p]);
    opt.loadStateDict({
      state: [
        { paramId: 0, state: { prevGrad: new Float64Array(5), stepSizes: new Float64Array(5) } },
      ],
    });
    expect(() => opt.step()).toThrow(/size mismatch/);
  });
});

describe("SparseAdam matches PyTorch", () => {
  it("folds the bias correction into the step size like torch.optim.SparseAdam", () => {
    expectTrajectory(
      trajectory((p) => new SparseAdam(p, { lr: 0.05, beta1: 0.9, beta2: 0.99, eps: 1e-3 }), 10),
      [
        [0.950980392157, -1.951612903226, 0.5],
        [0.902311945568, -1.944576415574, 0.4662518278],
        [0.875732177064, -1.957901049113, 0.4662518278],
        [0.844085114233, -1.962229266958, 0.456739009154],
        [0.813875794653, -1.971628803561, 0.444391237234],
        [0.803273023168, -1.982476302143, 0.444391237234],
        [0.788437438997, -1.98127821564, 0.417885752107],
        [0.773928041491, -1.992853450507, 0.402607172104],
        [0.74959807751, -1.985081727857, 0.37884522617],
        [0.730210233706, -1.989886708631, 0.37884522617],
      ]
    );
  });

  it("supports maximize", () => {
    expectTrajectory(
      trajectory((p) => new SparseAdam(p, { lr: 0.05, maximize: true }), 10),
      [
        [1.049999968377, -2.049999947295, 0.5],
        [1.099406234176, -2.057225967441, 0.53720672352],
        [1.126328074187, -2.043604675776, 0.53720672352],
        [1.158329658857, -2.039190065034, 0.547606394979],
        [1.188813733138, -2.029622599615, 0.561065837418],
        [1.199515519117, -2.018604456631, 0.561065837418],
        [1.214466291752, -2.019820765471, 0.588727285003],
        [1.229056496032, -2.008057764924, 0.604624466602],
        [1.2536278686, -2.015976605848, 0.629257354487],
        [1.273167093202, -2.011081341297, 0.629257354487],
      ]
    );
  });
});

describe("SGD", () => {
  it("supports maximize with momentum, dampening and weight decay", () => {
    expectTrajectory(
      trajectory(
        (p) =>
          new SGD(p, {
            lr: 0.1,
            momentum: 0.9,
            dampening: 0.1,
            weightDecay: 0.01,
            maximize: true,
          }),
        6
      ),
      [
        [1.049, -2.028, 0.4995],
        [1.1281559, -2.0333748, 0.50760045],
        [1.18038086969, -2.01388208268, 0.514434014595],
        [1.253320999628, -2.003526143218, 0.515621232117],
        [1.326839127673, -1.983402624173, 0.518025668779],
        [1.355811287698, -1.95900639467, 0.519723438672],
      ]
    );
  });

  it("supports maximize with Nesterov momentum", () => {
    expectTrajectory(
      trajectory((p) => new SGD(p, { lr: 0.1, momentum: 0.9, nesterov: true, maximize: true }), 6),
      [
        [1.095, -2.057, 0.5],
        [1.2115, -2.0433, 0.519],
        [1.24235, -2.00147, 0.5271],
        [1.345115, -2.005323, 0.52489],
        [1.4296035, -1.9807907, 0.531201],
        [1.42064315, -1.95821163, 0.5350809],
      ]
    );
  });

  it("rejects a loaded momentum buffer of the wrong size", () => {
    const p = parameter(tensor([1, 2, 3], { dtype: "float64" }));
    p.setGrad(tensor([1, 1, 1], { dtype: "float64" }));
    const opt = new SGD([p], { momentum: 0.9 });
    opt.loadStateDict({ state: [{ paramId: 0, state: { momentumBuffer: new Float64Array(5) } }] });
    expect(() => opt.step()).toThrow(/size mismatch/);
  });
});

describe("OneCycleLR matches PyTorch", () => {
  it("cosine annealing with torch phase boundaries and minimum lr", () => {
    const opt = mockOptimizer(0.1);
    const s = new OneCycleLR(opt, {
      maxLr: 1,
      totalSteps: 10,
      pctStart: 0.3,
      divFactor: 10,
      finalDivFactor: 100,
    });
    expectClose(
      lrSequence(opt, s, 10),
      [
        0.09999999999999998, 0.55, 1.0, 0.9505339495172583, 0.8119331560284374, 0.611649206511179,
        0.389350793488821, 0.18906684397156262, 0.05046605048274169, 0.001,
      ]
    );
  });

  it("linear annealing", () => {
    const opt = mockOptimizer(0.1);
    const s = new OneCycleLR(opt, {
      maxLr: 1,
      totalSteps: 10,
      pctStart: 0.3,
      divFactor: 10,
      finalDivFactor: 100,
      annealStrategy: "linear",
    });
    expectClose(
      lrSequence(opt, s, 10),
      [
        0.1, 0.55, 1.0, 0.8572857142857143, 0.7145714285714286, 0.5718571428571428,
        0.42914285714285716, 0.28642857142857137, 0.1437142857142858, 0.001,
      ]
    );
  });

  it("uses a cosine rise in the first phase and defaults of 25 and 1e4", () => {
    const opt = mockOptimizer(0.1);
    const s = new OneCycleLR(opt, { maxLr: 0.1, totalSteps: 20 });
    expectClose(
      lrSequence(opt, s, 20),
      [
        0.0040000000000000036, 0.013167184270002533, 0.037167184270002526, 0.06683281572999747,
        0.09083281572999748, 0.1, 0.09874640062350874, 0.09504846320134738, 0.089091617757105,
        0.0811745653949763, 0.07169430017913009, 0.06112620219362893, 0.0500002,
        0.03887419780637107, 0.028306099820869922, 0.0188258346050237, 0.010908782242895008,
        0.004951936798652629, 0.0012539993764912555, 4e-7,
      ]
    );
  });

  it("holds the minimum after the last step", () => {
    const opt = mockOptimizer(0.1);
    const s = new OneCycleLR(opt, { maxLr: 1, totalSteps: 5 });
    for (let i = 0; i < 12; i++) s.step();
    expect(opt.paramGroups[0]?.lr).toBeCloseTo(1 / 25 / 1e4, 12);
  });
});

describe("SequentialLR matches PyTorch", () => {
  it("starts at the first scheduler's epoch 0 and restarts the next one at the milestone", () => {
    const opt = mockOptimizer(1);
    const a = new LinearLR(opt, { startFactor: 0.1, totalIters: 5 });
    const b = new ExponentialLR(opt, { gamma: 0.9 });
    const s = new SequentialLR(opt, { schedulers: [a, b], milestones: [5] });
    expectClose(
      lrSequence(opt, s, 12),
      [0.1, 0.28, 0.46, 0.64, 0.82, 1.0, 0.9, 0.81, 0.729, 0.6561, 0.59049, 0.531441]
    );
  });

  it("works when the first scheduler is constant", () => {
    const opt = mockOptimizer(0.1);
    const a = new StepLR(opt, { stepSize: 100, gamma: 1 });
    const b = new ExponentialLR(opt, { gamma: 0.5 });
    const s = new SequentialLR(opt, { schedulers: [a, b], milestones: [3] });
    expectClose(lrSequence(opt, s, 8), [0.1, 0.1, 0.1, 0.1, 0.05, 0.025, 0.0125, 0.00625]);
  });

  it("rejects entries that are not schedulers", () => {
    const opt = mockOptimizer(0.1);
    const a = new StepLR(opt, { stepSize: 1 });
    expect(() => new SequentialLR(opt, { schedulers: [a, {} as StepLR], milestones: [2] })).toThrow(
      InvalidParameterError
    );
  });
});

describe("scheduler base learning rates", () => {
  it("a second scheduler starts from the original lr, not the value left by the first", () => {
    const opt = mockOptimizer(1);
    new LinearLR(opt, { startFactor: 0.1, totalIters: 5 });
    expect(opt.paramGroups[0]?.lr).toBeCloseTo(0.1, 12);
    const b = new ExponentialLR(opt, { gamma: 0.5 });
    // Base lr is 1 (the optimizer's lr), so epoch 0 of the exponential schedule is 1.
    expect(opt.paramGroups[0]?.lr).toBeCloseTo(1, 12);
    b.step();
    expect(opt.paramGroups[0]?.lr).toBeCloseTo(0.5, 12);
  });

  it("a rate changed by hand becomes the new base", () => {
    const opt = mockOptimizer(1);
    new StepLR(opt, { stepSize: 1 });
    opt.paramGroups[0]!.lr = 0.5;
    const s = new ExponentialLR(opt, { gamma: 0.5 });
    s.step();
    expect(opt.paramGroups[0]?.lr).toBeCloseTo(0.25, 12);
  });

  it("WarmupLR warms up to the original lr when the wrapped scheduler runs first", () => {
    const opt = mockOptimizer(1);
    const after = new LinearLR(opt, { startFactor: 0.5, totalIters: 4 });
    const s = new WarmupLR(opt, after, { warmupEpochs: 2 });
    expect(opt.paramGroups[0]?.lr).toBeCloseTo(0.5, 12);
    s.step();
    expect(opt.paramGroups[0]?.lr).toBeCloseTo(1, 12);
  });
});

describe("LinearLR", () => {
  it("allows endFactor 0 to decay to zero", () => {
    const opt = mockOptimizer(1);
    const s = new LinearLR(opt, { startFactor: 1, endFactor: 0, totalIters: 4 });
    expectClose(lrSequence(opt, s, 6), [1, 0.75, 0.5, 0.25, 0, 0]);
  });

  it("still rejects a negative endFactor", () => {
    const opt = mockOptimizer(1);
    expect(() => new LinearLR(opt, { endFactor: -0.1, totalIters: 4 })).toThrow(
      InvalidParameterError
    );
  });
});

describe("CyclicLR", () => {
  it("supports exp_range with gamma", () => {
    const opt = mockOptimizer(0.01);
    const s = new CyclicLR(opt, {
      baseLr: 0.01,
      maxLr: 0.1,
      stepSizeUp: 3,
      mode: "exp_range",
      gamma: 0.9,
    });
    expectClose(
      lrSequence(opt, s, 10),
      [
        0.01, 0.037, 0.0586, 0.07561, 0.049366, 0.0277147, 0.01, 0.024348907, 0.0358280326,
        0.04486784401,
      ]
    );
  });
});

describe("CosineAnnealingWarmRestarts aliases", () => {
  it("accepts t0 and tMult", () => {
    const a = mockOptimizer(0.1);
    const b = mockOptimizer(0.1);
    const s1 = new CosineAnnealingWarmRestarts(a, { T_0: 3, T_mult: 2 });
    const s2 = new CosineAnnealingWarmRestarts(b, { t0: 3, tMult: 2 });
    expectClose(lrSequence(a, s1, 12), lrSequence(b, s2, 12), 14);
  });

  it("requires a period", () => {
    expect(() => new CosineAnnealingWarmRestarts(mockOptimizer(0.1), {})).toThrow(
      InvalidParameterError
    );
  });
});

describe("LambdaLR validation", () => {
  it("rejects a negative factor at the epoch it occurs", () => {
    const opt = mockOptimizer(1);
    const s = new LambdaLR(opt, { lrLambda: (epoch) => 1 - epoch });
    s.step();
    expect(() => s.step()).toThrow(InvalidParameterError);
  });

  it("rejects non-function entries in the array form", () => {
    const opt = mockOptimizer(1);
    expect(() => new LambdaLR(opt, { lrLambda: [1 as unknown as (e: number) => number] })).toThrow(
      InvalidParameterError
    );
  });
});

describe("ReduceLROnPlateau", () => {
  it("supports one minLr per group", () => {
    const opt = mockOptimizer(1, 1);
    const s = new ReduceLROnPlateau(opt, { patience: 0, factor: 0.1, minLr: [0.5, 0.01] });
    s.step(1);
    s.step(1);
    expect(s.getLastLr()).toEqual([0.5, 0.1]);
    expect(() => new ReduceLROnPlateau(opt, { minLr: [0.1] })).toThrow(InvalidParameterError);
  });

  it("skips reductions smaller than eps", () => {
    const opt = mockOptimizer(1e-9);
    const s = new ReduceLROnPlateau(opt, { patience: 0, factor: 0.5, eps: 1e-8 });
    s.step(1);
    s.step(1);
    expect(s.getLastLr()[0]).toBe(1e-9);
  });

  it("treats the first metric as an improvement even with threshold 1", () => {
    const opt = mockOptimizer(1);
    const s = new ReduceLROnPlateau(opt, { patience: 0, threshold: 1, factor: 0.5 });
    s.step(5);
    expect(s.getLastLr()[0]).toBe(1);
  });

  it("restores its state from stateDict", () => {
    const opt = mockOptimizer(1);
    const s = new ReduceLROnPlateau(opt, { patience: 1, factor: 0.5 });
    s.step(1);
    s.step(2);
    const state = s.stateDict();
    const opt2 = mockOptimizer(1);
    const s2 = new ReduceLROnPlateau(opt2, { patience: 1, factor: 0.5 });
    s2.loadStateDict(state);
    s.step(2);
    s2.step(2);
    expect(opt2.paramGroups[0]?.lr).toBe(opt.paramGroups[0]?.lr);
    expect(opt.paramGroups[0]?.lr).toBe(0.5);
    expect(() => s2.loadStateDict({ ...state, lastLr: [1, 1] })).toThrow(DataValidationError);
  });
});

describe("WarmupLR stateDict", () => {
  it("round-trips through the wrapped scheduler", () => {
    const make = () => {
      const opt = mockOptimizer(1);
      const after = new ExponentialLR(opt, { gamma: 0.9 });
      return { opt, s: new WarmupLR(opt, after, { warmupEpochs: 3 }) };
    };
    const a = make();
    for (let i = 0; i < 6; i++) a.s.step();
    const b = make();
    b.s.loadStateDict(a.s.stateDict());
    expect(b.opt.paramGroups[0]?.lr).toBe(a.opt.paramGroups[0]?.lr);
    a.s.step();
    b.s.step();
    expect(b.opt.paramGroups[0]?.lr).toBeCloseTo(a.opt.paramGroups[0]?.lr ?? Number.NaN, 12);
  });
});

describe("LRScheduler stateDict", () => {
  it("round-trips the epoch and learning rates", () => {
    const opt = mockOptimizer(0.1);
    const s = new StepLR(opt, { stepSize: 2, gamma: 0.5 });
    for (let i = 0; i < 5; i++) s.step();
    const state = s.stateDict();
    expect(state.lastEpoch).toBe(5);

    const opt2 = mockOptimizer(0.1);
    const s2 = new StepLR(opt2, { stepSize: 2, gamma: 0.5 });
    s2.loadStateDict(state);
    expect(s2.epoch).toBe(5);
    expect(opt2.paramGroups[0]?.lr).toBeCloseTo(0.025, 12);
    s.step();
    s2.step();
    expect(opt2.paramGroups[0]?.lr).toBeCloseTo(opt.paramGroups[0]?.lr ?? Number.NaN, 12);
  });

  it("includes wrapped schedulers and validates the shape", () => {
    const opt = mockOptimizer(1);
    const a = new LinearLR(opt, { startFactor: 0.1, totalIters: 5 });
    const b = new ExponentialLR(opt, { gamma: 0.9 });
    const s = new SequentialLR(opt, { schedulers: [a, b], milestones: [5] });
    for (let i = 0; i < 7; i++) s.step();
    const state = s.stateDict();
    expect(state.children?.length).toBe(2);

    const opt2 = mockOptimizer(1);
    const a2 = new LinearLR(opt2, { startFactor: 0.1, totalIters: 5 });
    const b2 = new ExponentialLR(opt2, { gamma: 0.9 });
    const s2 = new SequentialLR(opt2, { schedulers: [a2, b2], milestones: [5] });
    s2.loadStateDict(state);
    s.step();
    s2.step();
    expect(opt2.paramGroups[0]?.lr).toBeCloseTo(opt.paramGroups[0]?.lr ?? Number.NaN, 12);

    expect(() => s2.loadStateDict({ ...state, lastLr: [1, 2] })).toThrow(DataValidationError);
    const { children: _unused, ...withoutChildren } = state;
    expect(() => s2.loadStateDict(withoutChildren)).toThrow(DataValidationError);
  });
});
