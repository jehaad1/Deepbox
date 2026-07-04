import { describe, expect, it } from "vitest";
import {
  CosineAnnealingWarmRestarts,
  CyclicLR,
  ExponentialLR,
  LambdaLR,
  PolynomialLR,
  SequentialLR,
  StepLR,
} from "../src/optim";

function makeOptimizer(lr: number) {
  return {
    paramGroups: [{ params: [{}], lr }],
  };
}

describe("CosineAnnealingWarmRestarts", () => {
  it("starts at base lr", () => {
    const opt = makeOptimizer(0.1);
    new CosineAnnealingWarmRestarts(opt, { T_0: 10 });
    // At construction (epoch 0), lr should be at max (base lr)
    expect(opt.paramGroups[0]!.lr).toBeCloseTo(0.1, 5);
  });

  it("reaches etaMin at half period", () => {
    const opt = makeOptimizer(0.1);
    const s = new CosineAnnealingWarmRestarts(opt, { T_0: 10, etaMin: 0 });
    // Construction is epoch 0; step to epoch 5 (mid-period), lr should be ~0.05
    for (let i = 0; i < 5; i++) s.step();
    const lr = opt.paramGroups[0]!.lr!;
    expect(lr).toBeCloseTo(0.05, 5);
    expect(lr).toBeLessThan(0.1);
    expect(lr).toBeGreaterThan(0);
  });

  it("reaches etaMin at end of period", () => {
    const opt = makeOptimizer(0.1);
    const s = new CosineAnnealingWarmRestarts(opt, { T_0: 10, etaMin: 0 });
    // Construction is epoch 0; step to epoch 9 (just before restart)
    for (let i = 0; i < 9; i++) s.step();
    const lr = opt.paramGroups[0]!.lr!;
    // Should be near etaMin=0 (not exact due to discrete cosine)
    expect(lr).toBeLessThan(0.01);
  });

  it("restarts after T_0 epochs", () => {
    const opt = makeOptimizer(0.1);
    const s = new CosineAnnealingWarmRestarts(opt, { T_0: 5, etaMin: 0 });
    // Construction is epoch 0; record lr per epoch (lrs[i] === epoch i)
    const lrs: number[] = [opt.paramGroups[0]!.lr!];
    for (let i = 0; i < 9; i++) {
      s.step();
      lrs.push(opt.paramGroups[0]!.lr!);
    }
    // At restart (epoch 5), lr should jump back up near base
    expect(lrs[4]!).toBeLessThan(lrs[0]!); // lr decreased during period
    expect(lrs[5]!).toBeGreaterThan(lrs[4]!); // restart jumps up
  });

  it("T_mult doubles period after restart", () => {
    const opt = makeOptimizer(0.1);
    const s = new CosineAnnealingWarmRestarts(opt, {
      T_0: 4,
      T_mult: 2,
      etaMin: 0,
    });
    // Construction is epoch 0; record lr per epoch (lrs[i] === epoch i)
    const lrs: number[] = [opt.paramGroups[0]!.lr!];
    for (let i = 0; i < 12; i++) {
      s.step();
      lrs.push(opt.paramGroups[0]!.lr!);
    }
    // After restart at epoch 4, lr should jump back up
    expect(lrs[4]!).toBeGreaterThan(lrs[3]!); // restart at epoch 4
    // Second period is 8 epochs, so next restart at epoch 12
    expect(lrs[12]!).toBeGreaterThan(lrs[11]!); // restart at epoch 12
  });
});

describe("CyclicLR", () => {
  it("starts at baseLr", () => {
    const opt = makeOptimizer(0.001);
    new CyclicLR(opt, { baseLr: 0.001, maxLr: 0.01, stepSizeUp: 5 });
    // At step 0 (construction), should be at baseLr
    expect(opt.paramGroups[0]!.lr!).toBeCloseTo(0.001, 5);
  });

  it("reaches maxLr at stepSizeUp", () => {
    const opt = makeOptimizer(0.001);
    const s = new CyclicLR(opt, {
      baseLr: 0.001,
      maxLr: 0.01,
      stepSizeUp: 5,
      stepSizeDown: 5,
    });
    for (let i = 0; i < 5; i++) s.step(); // construction=epoch 0, step to epoch 5 = peak
    const lr = opt.paramGroups[0]!.lr!;
    expect(lr).toBeCloseTo(0.01, 5);
  });

  it("returns to baseLr after full cycle", () => {
    const opt = makeOptimizer(0.001);
    const s = new CyclicLR(opt, {
      baseLr: 0.001,
      maxLr: 0.01,
      stepSizeUp: 5,
      stepSizeDown: 5,
    });
    for (let i = 0; i < 10; i++) s.step(); // construction=epoch 0, step to epoch 10 = full cycle
    const lr = opt.paramGroups[0]!.lr!;
    expect(lr).toBeCloseTo(0.001, 5);
  });

  it("triangular2 mode halves amplitude each cycle", () => {
    const opt = makeOptimizer(0.001);
    const s = new CyclicLR(opt, {
      baseLr: 0.001,
      maxLr: 0.01,
      stepSizeUp: 5,
      stepSizeDown: 5,
      mode: "triangular2",
    });

    // Construction=epoch 0; peak of first cycle (epoch 5)
    for (let i = 0; i < 5; i++) s.step();
    const peak1 = opt.paramGroups[0]!.lr!;

    // Full cycle + peak of second (epoch 15)
    for (let i = 0; i < 10; i++) s.step();
    const peak2 = opt.paramGroups[0]!.lr!;

    // Second cycle peak should be roughly half the amplitude
    const amp1 = peak1 - 0.001;
    const amp2 = peak2 - 0.001;
    expect(amp2).toBeCloseTo(amp1 / 2, 3);
  });

  it("cycles repeatedly", () => {
    const opt = makeOptimizer(0.001);
    const s = new CyclicLR(opt, {
      baseLr: 0.001,
      maxLr: 0.01,
      stepSizeUp: 3,
      stepSizeDown: 3,
    });
    const lrs: number[] = [];
    for (let i = 0; i < 13; i++) {
      s.step();
      lrs.push(opt.paramGroups[0]!.lr!);
    }
    // Should see two peaks
    expect(lrs.length).toBe(13);
    // At least one value should be close to maxLr
    expect(lrs.some((lr) => lr > 0.008)).toBe(true);
  });
});

describe("LambdaLR", () => {
  it("applies lambda to base lr", () => {
    const opt = makeOptimizer(0.1);
    const s = new LambdaLR(opt, {
      lrLambda: (epoch) => 0.5 ** epoch,
    });
    // epoch 0 at construction: factor=1
    expect(opt.paramGroups[0]!.lr).toBeCloseTo(0.1, 10);
    s.step(); // epoch 1: factor=0.5
    expect(opt.paramGroups[0]!.lr).toBeCloseTo(0.05, 10);
    s.step(); // epoch 2: factor=0.25
    expect(opt.paramGroups[0]!.lr).toBeCloseTo(0.025, 10);
  });

  it("supports constant lambda", () => {
    const opt = makeOptimizer(0.1);
    const s = new LambdaLR(opt, { lrLambda: () => 0.5 });
    s.step();
    expect(opt.paramGroups[0]!.lr).toBeCloseTo(0.05, 10);
    s.step();
    expect(opt.paramGroups[0]!.lr).toBeCloseTo(0.05, 10);
  });

  it("supports array of lambdas for multiple param groups", () => {
    const opt = {
      paramGroups: [
        { params: [{}], lr: 0.1 },
        { params: [{}], lr: 0.01 },
      ],
    };
    const s = new LambdaLR(opt, {
      lrLambda: [(epoch) => 1.0 / (1 + epoch), () => 1.0],
    });
    // epoch 0 at construction
    expect(opt.paramGroups[0]!.lr).toBeCloseTo(0.1, 10);
    expect(opt.paramGroups[1]!.lr).toBeCloseTo(0.01, 10);
    s.step(); // epoch 1
    expect(opt.paramGroups[0]!.lr).toBeCloseTo(0.05, 10);
    expect(opt.paramGroups[1]!.lr).toBeCloseTo(0.01, 10);
  });

  it("throws for mismatched array length", () => {
    const opt = makeOptimizer(0.1);
    expect(() => new LambdaLR(opt, { lrLambda: [() => 1, () => 1] })).toThrow();
  });
});

describe("SequentialLR", () => {
  it("switches schedulers at milestone", () => {
    const opt = makeOptimizer(0.1);
    const s1 = new StepLR(opt, { stepSize: 100, gamma: 1.0 }); // constant
    const s2 = new ExponentialLR(opt, { gamma: 0.5 });
    const seq = new SequentialLR(opt, {
      schedulers: [s1, s2],
      milestones: [3],
    });

    // First 3 epochs: s1 (constant at base lr)
    seq.step(); // epoch 0
    const lr0 = opt.paramGroups[0]!.lr!;
    seq.step(); // epoch 1
    seq.step(); // epoch 2
    const lr2 = opt.paramGroups[0]!.lr!;
    expect(lr2).toBeCloseTo(lr0, 5);

    // After milestone 3: s2 (exponential decay)
    // s2 epoch 0 → factor=1, epoch 1 → factor=0.5, epoch 2 → factor=0.25
    seq.step(); // epoch 3 → s2 epoch 0
    seq.step(); // epoch 4 → s2 epoch 1
    seq.step(); // epoch 5 → s2 epoch 2
    const lr5 = opt.paramGroups[0]!.lr!;
    // s2 has been stepped 3 times, its internal epoch should produce decay
    expect(lr5).toBeLessThan(lr0);
  });

  it("throws for fewer than 2 schedulers", () => {
    const opt = makeOptimizer(0.1);
    const s1 = new StepLR(opt, { stepSize: 10 });
    expect(() => new SequentialLR(opt, { schedulers: [s1], milestones: [] })).toThrow();
  });

  it("throws for wrong milestones count", () => {
    const opt = makeOptimizer(0.1);
    const s1 = new StepLR(opt, { stepSize: 10 });
    const s2 = new StepLR(opt, { stepSize: 10 });
    expect(() => new SequentialLR(opt, { schedulers: [s1, s2], milestones: [5, 10] })).toThrow();
  });
});

describe("PolynomialLR", () => {
  it("decays linearly with power=1", () => {
    const opt = makeOptimizer(0.1);
    const s = new PolynomialLR(opt, { totalIters: 10, power: 1 });

    // epoch 0 at construction: (1-0/10)^1 = 1.0 → lr = 0.1
    expect(opt.paramGroups[0]!.lr).toBeCloseTo(0.1, 10);

    for (let i = 0; i < 4; i++) s.step(); // epoch 4
    // (1-4/10)^1 = 0.6 → lr = 0.06
    expect(opt.paramGroups[0]!.lr).toBeCloseTo(0.06, 5);
  });

  it("decays quadratically with power=2", () => {
    const opt = makeOptimizer(0.1);
    const s = new PolynomialLR(opt, { totalIters: 10, power: 2 });

    for (let i = 0; i < 5; i++) s.step(); // construction=epoch 0, step to epoch 5
    // (1-5/10)^2 = 0.25 → lr = 0.025
    expect(opt.paramGroups[0]!.lr).toBeCloseTo(0.025, 5);
  });

  it("stays at endLr after totalIters", () => {
    const opt = makeOptimizer(0.1);
    const s = new PolynomialLR(opt, { totalIters: 5, endLr: 0.001 });

    for (let i = 0; i < 10; i++) s.step();
    expect(opt.paramGroups[0]!.lr).toBeCloseTo(0.001, 10);
  });

  it("respects custom endLr", () => {
    const opt = makeOptimizer(0.1);
    const s = new PolynomialLR(opt, { totalIters: 10, power: 1, endLr: 0.01 });

    for (let i = 0; i < 5; i++) s.step(); // construction=epoch 0, step to epoch 5
    // (0.1-0.01) * (1-5/10)^1 + 0.01 = 0.09*0.5 + 0.01 = 0.055
    expect(opt.paramGroups[0]!.lr).toBeCloseTo(0.055, 5);
  });

  it("throws for invalid totalIters", () => {
    const opt = makeOptimizer(0.1);
    expect(() => new PolynomialLR(opt, { totalIters: 0 })).toThrow();
    expect(() => new PolynomialLR(opt, { totalIters: -5 })).toThrow();
  });
});
