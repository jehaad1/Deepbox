/**
 * Benchmark 08 — Optimizers & LR Schedulers
 * Deepbox vs PyTorch
 */

import { GradTensor, parameter, randn } from "deepbox/ndarray";
import { Linear, ReLU, Sequential } from "deepbox/nn";
import {
  AdaDelta,
  Adagrad,
  Adam,
  AdamW,
  CosineAnnealingLR,
  ExponentialLR,
  LinearLR,
  MultiStepLR,
  Nadam,
  OneCycleLR,
  ReduceLROnPlateau,
  RMSprop,
  SGD,
  StepLR,
  WarmupLR,
} from "deepbox/optim";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("optim");
header("Benchmark 08 — Optimizers & LR Schedulers");

// ── Helper ──────────────────────────────────────────────

function makeModel(inF: number, hidden: number) {
  return new Sequential(new Linear(inF, hidden), new ReLU(), new Linear(hidden, 1));
}

function trainLoop(
  optimizerFn: (params: Iterable<GradTensor>) => {
    zeroGrad(): void;
    step(): void;
  },
  epochs: number,
  batchSize: number,
  inF: number
) {
  const m = makeModel(inF, 32);
  const opt = optimizerFn(m.parameters());
  const tX = parameter(randn([batchSize, inF]));
  const tY = parameter(randn([batchSize, 1]));
  for (let i = 0; i < epochs; i++) {
    opt.zeroGrad();
    const out = m.forward(tX);
    if (!(out instanceof GradTensor)) break;
    const diff = out.sub(tY);
    const loss = diff.mul(diff).mean();
    loss.backward();
    opt.step();
  }
}

// ── Optimizer Creation ──────────────────────────────────

const baseModel = makeModel(10, 32);
const params = () => baseModel.parameters();

run(suite, "SGD create", "—", () => new SGD(params(), { lr: 0.01 }));
run(suite, "SGD create (momentum)", "—", () => new SGD(params(), { lr: 0.01, momentum: 0.9 }));
run(suite, "Adam create", "—", () => new Adam(params(), { lr: 0.001 }));
run(suite, "AdamW create", "—", () => new AdamW(params(), { lr: 0.001 }));
run(suite, "Adagrad create", "—", () => new Adagrad(params(), { lr: 0.01 }));
run(suite, "AdaDelta create", "—", () => new AdaDelta(params(), { lr: 1.0 }));
run(suite, "Nadam create", "—", () => new Nadam(params(), { lr: 0.002 }));
run(suite, "RMSprop create", "—", () => new RMSprop(params(), { lr: 0.01 }));

// ── Optimizer Step ──────────────────────────────────────

function makeStep(
  OptimizerClass: new (
    p: Iterable<GradTensor>,
    o: Record<string, number>
  ) => { zeroGrad(): void; step(): void },
  opts: Record<string, number>
) {
  const m = makeModel(10, 32);
  const opt = new OptimizerClass(m.parameters(), opts);
  const tX = parameter(randn([16, 10]));
  const tY = parameter(randn([16, 1]));
  const out = m.forward(tX);
  if (out instanceof GradTensor) {
    const diff = out.sub(tY);
    const loss = diff.mul(diff).mean();
    loss.backward();
  }
  return () => opt.step();
}

run(suite, "SGD step", "16x10→1", makeStep(SGD as never, { lr: 0.01 }));
run(suite, "Adam step", "16x10→1", makeStep(Adam as never, { lr: 0.001 }));
run(suite, "AdamW step", "16x10→1", makeStep(AdamW as never, { lr: 0.001 }));
run(suite, "Adagrad step", "16x10→1", makeStep(Adagrad as never, { lr: 0.01 }));
run(suite, "RMSprop step", "16x10→1", makeStep(RMSprop as never, { lr: 0.01 }));

// ── Training Loops (per optimizer) ──────────────────────

run(
  suite,
  "SGD train 50 epochs",
  "32x10→1",
  () => trainLoop((p) => new SGD(p, { lr: 0.01 }), 50, 32, 10),
  { warmup: 2, iterations: 5 }
);
run(
  suite,
  "SGD+momentum train 50 epochs",
  "32x10→1",
  () => trainLoop((p) => new SGD(p, { lr: 0.01, momentum: 0.9 }), 50, 32, 10),
  { warmup: 2, iterations: 5 }
);
run(
  suite,
  "Adam train 50 epochs",
  "32x10→1",
  () => trainLoop((p) => new Adam(p, { lr: 0.01 }), 50, 32, 10),
  { warmup: 2, iterations: 5 }
);
run(
  suite,
  "AdamW train 50 epochs",
  "32x10→1",
  () => trainLoop((p) => new AdamW(p, { lr: 0.01 }), 50, 32, 10),
  { warmup: 2, iterations: 5 }
);
run(
  suite,
  "Adagrad train 50 epochs",
  "32x10→1",
  () => trainLoop((p) => new Adagrad(p, { lr: 0.01 }), 50, 32, 10),
  { warmup: 2, iterations: 5 }
);
run(
  suite,
  "AdaDelta train 50 epochs",
  "32x10→1",
  () => trainLoop((p) => new AdaDelta(p, { lr: 1.0 }), 50, 32, 10),
  { warmup: 2, iterations: 5 }
);
run(
  suite,
  "Nadam train 50 epochs",
  "32x10→1",
  () => trainLoop((p) => new Nadam(p, { lr: 0.002 }), 50, 32, 10),
  { warmup: 2, iterations: 5 }
);
run(
  suite,
  "RMSprop train 50 epochs",
  "32x10→1",
  () => trainLoop((p) => new RMSprop(p, { lr: 0.01 }), 50, 32, 10),
  { warmup: 2, iterations: 5 }
);

run(
  suite,
  "Adam train 100 epochs",
  "64x50→1",
  () => trainLoop((p) => new Adam(p, { lr: 0.001 }), 100, 64, 50),
  { warmup: 1, iterations: 3 }
);
run(
  suite,
  "SGD train 100 epochs",
  "64x50→1",
  () => trainLoop((p) => new SGD(p, { lr: 0.01 }), 100, 64, 50),
  { warmup: 1, iterations: 3 }
);

// ── LR Schedulers ───────────────────────────────────────

function schedStep(makeSched: (opt: SGD) => { step(metric?: number): void }, steps: number) {
  const m = makeModel(10, 16);
  const opt = new SGD(m.parameters(), { lr: 0.1 });
  const sched = makeSched(opt);
  return () => {
    for (let i = 0; i < steps; i++) sched.step();
  };
}

run(
  suite,
  "StepLR (100 steps)",
  "—",
  schedStep((o) => new StepLR(o, { stepSize: 10, gamma: 0.1 }), 100)
);
run(
  suite,
  "MultiStepLR (100 steps)",
  "—",
  schedStep((o) => new MultiStepLR(o, { milestones: [30, 60, 80], gamma: 0.1 }), 100)
);
run(
  suite,
  "ExponentialLR (100 steps)",
  "—",
  schedStep((o) => new ExponentialLR(o, { gamma: 0.95 }), 100)
);
run(
  suite,
  "CosineAnnealingLR (100 steps)",
  "—",
  schedStep((o) => new CosineAnnealingLR(o, { T_max: 100 }), 100)
);
run(
  suite,
  "LinearLR (100 steps)",
  "—",
  schedStep((o) => new LinearLR(o, { startFactor: 0.1, totalIters: 100 }), 100)
);
run(
  suite,
  "OneCycleLR (100 steps)",
  "—",
  schedStep((o) => new OneCycleLR(o, { maxLr: 0.1, totalSteps: 100 }), 100)
);
run(suite, "ReduceLROnPlateau (100 steps)", "—", () => {
  const m = makeModel(10, 16);
  const opt = new SGD(m.parameters(), { lr: 0.1 });
  const sched = new ReduceLROnPlateau(opt, { patience: 10 });
  for (let i = 0; i < 100; i++) sched.step(1.0);
});
run(
  suite,
  "WarmupLR (100 steps)",
  "—",
  schedStep((o) => new WarmupLR(o, null, { warmupEpochs: 50 }), 100)
);

// ── State Dict ──────────────────────────────────────────

const adamOpt = new Adam(baseModel.parameters(), { lr: 0.001 });
run(suite, "optimizer stateDict()", "Adam", () => adamOpt.stateDict());

const sgdOpt = new SGD(baseModel.parameters(), { lr: 0.01 });
run(suite, "optimizer stateDict()", "SGD", () => sgdOpt.stateDict());

// ── zeroGrad ────────────────────────────────────────────

run(suite, "zeroGrad", "3-layer", () => adamOpt.zeroGrad());

footer(suite, "deepbox-optim.json");
