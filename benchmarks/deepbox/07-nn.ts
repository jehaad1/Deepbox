/**
 * Benchmark 07 — Neural Networks
 * Deepbox vs PyTorch
 */

import { GradTensor, noGrad, parameter, randn, tensor } from "deepbox/ndarray";
import {
  BatchNorm1d,
  binaryCrossEntropyWithLogitsLoss,
  Conv1d,
  Conv2d,
  crossEntropyLoss,
  ELU,
  GELU,
  GRU,
  huberLoss,
  LayerNorm,
  LeakyReLU,
  Linear,
  LSTM,
  Mish,
  maeLoss,
  mseLoss,
  ReLU,
  RNN,
  rmseLoss,
  Sequential,
  Sigmoid,
  Softmax,
  Swish,
  Tanh,
} from "deepbox/nn";
import { Adam, SGD } from "deepbox/optim";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("nn");
header("Benchmark 07 — Neural Networks");

// ── Layer Creation ──────────────────────────────────────

run(suite, "Linear create", "10→64", () => new Linear(10, 64));
run(suite, "Linear create", "64→128", () => new Linear(64, 128));
run(suite, "Linear create", "128→256", () => new Linear(128, 256));
run(
  suite,
  "Sequential create",
  "10→64→1",
  () => new Sequential(new Linear(10, 64), new ReLU(), new Linear(64, 1))
);
run(
  suite,
  "Sequential create",
  "50→128→64→1",
  () =>
    new Sequential(
      new Linear(50, 128),
      new ReLU(),
      new Linear(128, 64),
      new ReLU(),
      new Linear(64, 1)
    )
);
run(suite, "Conv1d create", "1→16 k=3", () => new Conv1d(1, 16, 3));
run(suite, "Conv2d create", "1→16 k=3", () => new Conv2d(1, 16, 3));
run(suite, "RNN create", "10→32", () => new RNN(10, 32));
run(suite, "LSTM create", "10→32", () => new LSTM(10, 32));
run(suite, "GRU create", "10→32", () => new GRU(10, 32));
run(suite, "BatchNorm1d create", "64", () => new BatchNorm1d(64));
run(suite, "LayerNorm create", "[64]", () => new LayerNorm([64]));

// ── Forward Pass ────────────────────────────────────────

const model1 = new Sequential(new Linear(10, 64), new ReLU(), new Linear(64, 1));
const model2 = new Sequential(
  new Linear(50, 128),
  new ReLU(),
  new Linear(128, 64),
  new ReLU(),
  new Linear(64, 1)
);
const model3 = new Sequential(
  new Linear(10, 256),
  new ReLU(),
  new Linear(256, 128),
  new ReLU(),
  new Linear(128, 64),
  new ReLU(),
  new Linear(64, 1)
);

const x32_10 = randn([32, 10]);
const x128_10 = randn([128, 10]);
const x64_50 = randn([64, 50]);
const x32_10g = parameter(randn([32, 10]));
const x64_50g = parameter(randn([64, 50]));

run(suite, "forward (10→64→1)", "batch=32", () => model1.forward(x32_10));
run(suite, "forward (10→64→1)", "batch=128", () => model1.forward(x128_10));
run(suite, "forward (50→128→64→1)", "batch=64", () => model2.forward(x64_50));
run(suite, "forward (10→256→128→64→1)", "batch=32", () => model3.forward(x32_10));

// ── Activation Layers ───────────────────────────────────

const act_in = randn([32, 64]);

run(suite, "ReLU forward", "32x64", () => new ReLU().forward(act_in));
run(suite, "Sigmoid forward", "32x64", () => new Sigmoid().forward(act_in));
run(suite, "Tanh forward", "32x64", () => new Tanh().forward(act_in));
run(suite, "LeakyReLU forward", "32x64", () => new LeakyReLU().forward(act_in));
run(suite, "ELU forward", "32x64", () => new ELU().forward(act_in));
run(suite, "GELU forward", "32x64", () => new GELU().forward(act_in));
run(suite, "Swish forward", "32x64", () => new Swish().forward(act_in));
run(suite, "Mish forward", "32x64", () => new Mish().forward(act_in));
run(suite, "Softmax forward", "32x64", () => new Softmax().forward(act_in));

// ── Forward + Backward ─────────────────────────────────

function fwdBwdSmall() {
  const out = model1.forward(x32_10g);
  if (!(out instanceof GradTensor)) return;
  const target = parameter(randn([32, 1]));
  const diff = out.sub(target);
  const loss = diff.mul(diff).mean();
  loss.backward();
}

function fwdBwdLarge() {
  const out = model2.forward(x64_50g);
  if (!(out instanceof GradTensor)) return;
  const target = parameter(randn([64, 1]));
  const diff = out.sub(target);
  const loss = diff.mul(diff).mean();
  loss.backward();
}

run(suite, "forward+backward (10→64→1)", "batch=32", fwdBwdSmall);
run(suite, "forward+backward (50→128→64→1)", "batch=64", fwdBwdLarge, {
  iterations: 10,
});

// ── Loss Functions ──────────────────────────────────────

const pred32 = randn([32, 1]);
const target32 = randn([32, 1]);
const pred32_10 = randn([32, 10]);
const target32_cls = tensor(Array.from({ length: 32 }, (_, i) => i % 10));
const pred32_bin = randn([32, 1]);
const target32_bin = tensor(
  Array.from({ length: 32 }, () => (Math.random() > 0.5 ? 1 : 0))
).reshape([32, 1]);

run(suite, "mseLoss", "32x1", () => mseLoss(pred32, target32));
run(suite, "maeLoss", "32x1", () => maeLoss(pred32, target32));
run(suite, "rmseLoss", "32x1", () => rmseLoss(pred32, target32));
run(suite, "huberLoss", "32x1", () => huberLoss(pred32, target32));
run(suite, "crossEntropyLoss", "32x10", () => crossEntropyLoss(pred32_10, target32_cls));
run(suite, "binaryCrossEntropyLoss", "32x1", () =>
  binaryCrossEntropyWithLogitsLoss(pred32_bin, target32_bin)
);

// ── Training Loops ──────────────────────────────────────

function trainAdam50() {
  const m = new Sequential(new Linear(10, 32), new ReLU(), new Linear(32, 1));
  const opt = new Adam(m.parameters(), { lr: 0.01 });
  const tX = parameter(randn([32, 10]));
  const tY = parameter(randn([32, 1]));
  for (let i = 0; i < 50; i++) {
    opt.zeroGrad();
    const out = m.forward(tX);
    if (!(out instanceof GradTensor)) break;
    const diff = out.sub(tY);
    const loss = diff.mul(diff).mean();
    loss.backward();
    opt.step();
  }
}

function trainSGD50() {
  const m = new Sequential(new Linear(10, 32), new ReLU(), new Linear(32, 1));
  const opt = new SGD(m.parameters(), { lr: 0.01 });
  const tX = parameter(randn([32, 10]));
  const tY = parameter(randn([32, 1]));
  for (let i = 0; i < 50; i++) {
    opt.zeroGrad();
    const out = m.forward(tX);
    if (!(out instanceof GradTensor)) break;
    const diff = out.sub(tY);
    const loss = diff.mul(diff).mean();
    loss.backward();
    opt.step();
  }
}

function trainAdam100() {
  const m = new Sequential(
    new Linear(50, 64),
    new ReLU(),
    new Linear(64, 32),
    new ReLU(),
    new Linear(32, 1)
  );
  const opt = new Adam(m.parameters(), { lr: 0.001 });
  const tX = parameter(randn([64, 50]));
  const tY = parameter(randn([64, 1]));
  for (let i = 0; i < 100; i++) {
    opt.zeroGrad();
    const out = m.forward(tX);
    if (!(out instanceof GradTensor)) break;
    const diff = out.sub(tY);
    const loss = diff.mul(diff).mean();
    loss.backward();
    opt.step();
  }
}

run(suite, "train Adam 50 epochs", "32x10→1", trainAdam50, {
  warmup: 2,
  iterations: 5,
});
run(suite, "train SGD 50 epochs", "32x10→1", trainSGD50, {
  warmup: 2,
  iterations: 5,
});
run(suite, "train Adam 100 epochs", "64x50→1", trainAdam100, {
  warmup: 1,
  iterations: 3,
});

// ── Inference (noGrad) ──────────────────────────────────

const inferModel = new Sequential(new Linear(10, 64), new ReLU(), new Linear(64, 1));
const inferX = randn([256, 10]);

run(suite, "inference (noGrad)", "batch=256", () => noGrad(() => inferModel.forward(inferX)));
run(suite, "inference (noGrad)", "batch=32", () => noGrad(() => inferModel.forward(x32_10)));

// ── Module Operations ───────────────────────────────────

run(suite, "parameters()", "3-layer", () => model1.parameters());
run(suite, "stateDict()", "3-layer", () => model1.stateDict());
run(suite, "train/eval toggle", "—", () => {
  model1.train();
  model1.eval();
});

footer(suite, "deepbox-nn.json");
