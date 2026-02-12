/**
 * Example 16: Learning Rate Schedulers
 *
 * Control the learning rate during training for better convergence.
 * Deepbox provides 8 schedulers inspired by PyTorch.
 */

import { Linear, ReLU, Sequential } from "deepbox/nn";
import {
  Adam,
  CosineAnnealingLR,
  ExponentialLR,
  LinearLR,
  MultiStepLR,
  OneCycleLR,
  ReduceLROnPlateau,
  StepLR,
  WarmupLR,
} from "deepbox/optim";

console.log("=== Learning Rate Schedulers ===\n");

// Create a small model and optimizer for demonstration
const createOptimizer = () => {
  const model = new Sequential(new Linear(4, 8), new ReLU(), new Linear(8, 1));
  return new Adam(model.parameters(), { lr: 0.1 });
};

// ---------------------------------------------------------------------------
// Part 1: StepLR — decay every N steps
// ---------------------------------------------------------------------------
console.log("--- Part 1: StepLR ---");

const opt1 = createOptimizer();
const stepLR = new StepLR(opt1, { stepSize: 3, gamma: 0.5 });

for (let epoch = 0; epoch < 10; epoch++) {
  const lr = stepLR.getLastLr()[0] ?? 0;
  console.log(`  Epoch ${epoch}: lr = ${lr.toFixed(6)}`);
  stepLR.step();
}

// ---------------------------------------------------------------------------
// Part 2: MultiStepLR — decay at specific milestones
// ---------------------------------------------------------------------------
console.log("\n--- Part 2: MultiStepLR ---");

const opt2 = createOptimizer();
const multiStepLR = new MultiStepLR(opt2, {
  milestones: [3, 6, 8],
  gamma: 0.5,
});

for (let epoch = 0; epoch < 10; epoch++) {
  const lr = multiStepLR.getLastLr()[0] ?? 0;
  console.log(`  Epoch ${epoch}: lr = ${lr.toFixed(6)}`);
  multiStepLR.step();
}

// ---------------------------------------------------------------------------
// Part 3: ExponentialLR — exponential decay each epoch
// ---------------------------------------------------------------------------
console.log("\n--- Part 3: ExponentialLR ---");

const opt3 = createOptimizer();
const expLR = new ExponentialLR(opt3, { gamma: 0.9 });

for (let epoch = 0; epoch < 10; epoch++) {
  const lr = expLR.getLastLr()[0] ?? 0;
  console.log(`  Epoch ${epoch}: lr = ${lr.toFixed(6)}`);
  expLR.step();
}

// ---------------------------------------------------------------------------
// Part 4: CosineAnnealingLR — cosine annealing
// ---------------------------------------------------------------------------
console.log("\n--- Part 4: CosineAnnealingLR ---");

const opt4 = createOptimizer();
const cosineLR = new CosineAnnealingLR(opt4, { T_max: 10, etaMin: 0.001 });

for (let epoch = 0; epoch < 10; epoch++) {
  const lr = cosineLR.getLastLr()[0] ?? 0;
  console.log(`  Epoch ${epoch}: lr = ${lr.toFixed(6)}`);
  cosineLR.step();
}

// ---------------------------------------------------------------------------
// Part 5: LinearLR — linear warmup / decay
// ---------------------------------------------------------------------------
console.log("\n--- Part 5: LinearLR ---");

const opt5 = createOptimizer();
const linearLR = new LinearLR(opt5, {
  startFactor: 0.1,
  endFactor: 1.0,
  totalIters: 5,
});

for (let epoch = 0; epoch < 8; epoch++) {
  const lr = linearLR.getLastLr()[0] ?? 0;
  console.log(`  Epoch ${epoch}: lr = ${lr.toFixed(6)}`);
  linearLR.step();
}

// ---------------------------------------------------------------------------
// Part 6: ReduceLROnPlateau — reduce when metric stops improving
// ---------------------------------------------------------------------------
console.log("\n--- Part 6: ReduceLROnPlateau ---");

const opt6 = createOptimizer();
const plateauLR = new ReduceLROnPlateau(opt6, { factor: 0.5, patience: 2 });

// Simulate a training loop where loss plateaus
const fakeLosses = [1.0, 0.8, 0.6, 0.59, 0.58, 0.58, 0.58, 0.3, 0.29, 0.29];
for (let epoch = 0; epoch < fakeLosses.length; epoch++) {
  const loss = fakeLosses[epoch];
  plateauLR.step(loss);
  console.log(
    `  Epoch ${epoch}: loss = ${loss.toFixed(2)}, lr = ${plateauLR.getLastLr()[0]?.toFixed(6)}`
  );
}

// ---------------------------------------------------------------------------
// Part 7: WarmupLR — linear warmup then constant
// ---------------------------------------------------------------------------
console.log("\n--- Part 7: WarmupLR ---");

const opt7 = createOptimizer();
const warmupLR = new WarmupLR(opt7, null, { warmupEpochs: 5 });

for (let epoch = 0; epoch < 8; epoch++) {
  const lr = warmupLR.getLastLr()[0] ?? 0;
  console.log(`  Epoch ${epoch}: lr = ${lr.toFixed(6)}`);
  warmupLR.step();
}

// ---------------------------------------------------------------------------
// Part 8: OneCycleLR — super-convergence schedule
// ---------------------------------------------------------------------------
console.log("\n--- Part 8: OneCycleLR ---");

const opt8 = createOptimizer();
const oneCycleLR = new OneCycleLR(opt8, { maxLr: 0.1, totalSteps: 10 });

for (let epoch = 0; epoch < 10; epoch++) {
  const lr = oneCycleLR.getLastLr()[0] ?? 0;
  console.log(`  Epoch ${epoch}: lr = ${lr.toFixed(6)}`);
  oneCycleLR.step();
}

console.log("\n=== Learning Rate Schedulers Complete ===");
