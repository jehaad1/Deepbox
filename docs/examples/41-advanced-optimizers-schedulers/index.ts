/**
 * Example 41: Advanced Optimizers & Schedulers
 *
 * New in v1.0.0: RAdam, LAMB, LARS optimizers and CyclicLR,
 * CosineAnnealingWarmRestarts, PolynomialLR, LambdaLR, SequentialLR schedulers.
 */

import { GradTensor, type Tensor, tensor } from "deepbox/ndarray";
import { Linear, mseLoss, ReLU, Sequential } from "deepbox/nn";
import {
  Adam,
  CosineAnnealingWarmRestarts,
  CyclicLR,
  LAMB,
  LARS,
  LambdaLR,
  PolynomialLR,
  RAdam,
  SequentialLR,
  StepLR,
} from "deepbox/optim";

console.log("=".repeat(60));
console.log("Example 41: Advanced Optimizers & Schedulers");
console.log("=".repeat(60));

// ============================================================================
// Helper: simple training demo
// ============================================================================

function trainDemo(
  model: Sequential,
  optimizerName: string,
  optimizer: { step: () => void; zeroGrad: () => void; lr: number },
  epochs = 10
): void {
  const xTrain = tensor([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
  ]);
  const yTrain = tensor([[3], [7], [11], [15]]);

  console.log(`\n  ${optimizerName} (initial lr=${optimizer.lr.toFixed(6)}):`);

  for (let epoch = 1; epoch <= epochs; epoch++) {
    optimizer.zeroGrad();
    const pred = model.forward(xTrain);
    const loss = mseLoss(pred, yTrain);
    const scalarLoss: Tensor = GradTensor.isGradTensor(loss) ? loss.tensor : loss;
    const lossVal = Number(scalarLoss.data[scalarLoss.offset]);

    if (GradTensor.isGradTensor(loss)) {
      loss.backward();
    }
    optimizer.step();

    if (epoch === 1 || epoch === epochs || epoch % 5 === 0) {
      console.log(
        `    Epoch ${String(epoch).padStart(3)}: loss=${lossVal.toFixed(6)}, lr=${optimizer.lr.toFixed(6)}`
      );
    }
  }
}

// ============================================================================
// Part 1: RAdam (Rectified Adam)
// ============================================================================
console.log("\n🚀 Part 1: RAdam (Rectified Adam)");
console.log("-".repeat(60));

// RAdam auto-adjusts the adaptive learning rate based on variance of gradients
// No need for learning rate warmup
const model1 = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const radam = new RAdam(model1.parameters(), { lr: 0.01 });

console.log("RAdam: Rectified Adam — no warmup needed");
console.log("  Auto-adjusts adaptive LR based on gradient variance");
trainDemo(model1, "RAdam", radam);

// ============================================================================
// Part 2: LAMB (Layer-wise Adaptive Moments)
// ============================================================================
console.log("\n🐑 Part 2: LAMB");
console.log("-".repeat(60));

// LAMB scales gradients layer-wise — great for large batch training
const model2 = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const lamb = new LAMB(model2.parameters(), { lr: 0.01 });

console.log("LAMB: Layer-wise Adaptive Moments — ideal for large batch training");
trainDemo(model2, "LAMB", lamb);

// ============================================================================
// Part 3: LARS (Layer-wise Adaptive Rate Scaling)
// ============================================================================
console.log("\n🏔️  Part 3: LARS");
console.log("-".repeat(60));

// LARS adjusts learning rate per layer based on weight/gradient norms
const model3 = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const lars = new LARS(model3.parameters(), { lr: 0.01 });

console.log("LARS: Layer-wise Adaptive Rate Scaling — for very large batches");
trainDemo(model3, "LARS", lars);

// ============================================================================
// Part 4: CyclicLR Scheduler
// ============================================================================
console.log("\n🔄 Part 4: CyclicLR Scheduler");
console.log("-".repeat(60));

// CyclicLR cycles the learning rate between a base and max value
const model4 = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const adam4 = new Adam(model4.parameters(), { lr: 0.001 });
const cyclicLr = new CyclicLR(adam4, {
  baseLr: 0.001,
  maxLr: 0.01,
  stepSizeUp: 5,
  mode: "triangular",
});

console.log("CyclicLR: triangular cycling between 0.001 and 0.01");
console.log("  LR schedule over 20 steps:");
for (let i = 0; i < 20; i++) {
  const lrs = cyclicLr.getLr();
  if (i % 4 === 0 || i === 19) {
    console.log(`    Step ${String(i + 1).padStart(3)}: lr=${lrs[0]?.toFixed(6)}`);
  }
  cyclicLr.step();
}

// ============================================================================
// Part 5: CosineAnnealingWarmRestarts
// ============================================================================
console.log("\n🌊 Part 5: CosineAnnealingWarmRestarts");
console.log("-".repeat(60));

// Cosine annealing with periodic warm restarts
const model5 = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const adam5 = new Adam(model5.parameters(), { lr: 0.01 });
const cosineWR = new CosineAnnealingWarmRestarts(adam5, {
  T_0: 5, // restart every 5 epochs
  T_mult: 2, // double the period after each restart
  etaMin: 0.001,
});

console.log("CosineAnnealingWarmRestarts: T_0=5, T_mult=2, etaMin=0.001");
console.log("  LR schedule (restarts at epoch 5, then 15, ...):");
for (let i = 0; i < 20; i++) {
  const lrs = cosineWR.getLr();
  if (i % 3 === 0 || i === 19) {
    console.log(`    Epoch ${String(i + 1).padStart(3)}: lr=${lrs[0]?.toFixed(6)}`);
  }
  cosineWR.step();
}

// ============================================================================
// Part 6: PolynomialLR
// ============================================================================
console.log("\n📉 Part 6: PolynomialLR");
console.log("-".repeat(60));

// Polynomial decay from initial LR to end LR
const model6 = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const adam6 = new Adam(model6.parameters(), { lr: 0.01 });
const polyLr = new PolynomialLR(adam6, {
  totalIters: 20,
  power: 2.0,
});

console.log("PolynomialLR: power=2.0 decay over 20 iterations");
console.log("  LR schedule:");
for (let i = 0; i < 20; i++) {
  const lrs = polyLr.getLr();
  if (i % 4 === 0 || i === 19) {
    console.log(`    Step ${String(i + 1).padStart(3)}: lr=${lrs[0]?.toFixed(6)}`);
  }
  polyLr.step();
}

// ============================================================================
// Part 7: LambdaLR
// ============================================================================
console.log("\n🔧 Part 7: LambdaLR");
console.log("-".repeat(60));

// LambdaLR uses a custom function to compute the LR multiplier
const model7 = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const adam7 = new Adam(model7.parameters(), { lr: 0.01 });
const lambdaLr = new LambdaLR(adam7, {
  lrLambda: (epoch: number) => 0.95 ** epoch, // exponential decay
});

console.log("LambdaLR: lr_lambda = 0.95^epoch (exponential decay)");
console.log("  LR schedule:");
for (let i = 0; i < 20; i++) {
  const lrs = lambdaLr.getLr();
  if (i % 4 === 0 || i === 19) {
    console.log(`    Epoch ${String(i + 1).padStart(3)}: lr=${lrs[0]?.toFixed(6)}`);
  }
  lambdaLr.step();
}

// ============================================================================
// Part 8: SequentialLR
// ============================================================================
console.log("\n📋 Part 8: SequentialLR");
console.log("-".repeat(60));

// SequentialLR chains multiple schedulers at specified milestones
const model8 = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const adam8 = new Adam(model8.parameters(), { lr: 0.01 });

// Phase 1: Warmup with LambdaLR (epochs 0-4)
const warmup = new LambdaLR(adam8, {
  lrLambda: (epoch: number) => Math.min(1.0, (epoch + 1) / 5),
});

// Phase 2: StepLR decay (epochs 5+)
const decay = new StepLR(adam8, { stepSize: 3, gamma: 0.5 });

const seqLr = new SequentialLR(adam8, {
  schedulers: [warmup, decay],
  milestones: [5],
});

console.log("SequentialLR: LambdaLR warmup (0-4) → StepLR decay (5+)");
console.log("  LR schedule:");
for (let i = 0; i < 20; i++) {
  const lrs = seqLr.getLr();
  if (i % 3 === 0 || i === 19) {
    console.log(`    Epoch ${String(i + 1).padStart(3)}: lr=${lrs[0]?.toFixed(6)}`);
  }
  seqLr.step();
}

// ============================================================================
// Part 9: Optimizer Comparison
// ============================================================================
console.log("\n📊 Part 9: Optimizer Comparison");
console.log("-".repeat(60));

console.log("┌─────────────────────────────┬──────────────────────────────────────┐");
console.log("│ Optimizer                   │ Best For                             │");
console.log("├─────────────────────────────┼──────────────────────────────────────┤");
console.log("│ Adam                        │ General purpose, default choice      │");
console.log("│ RAdam                       │ No warmup needed, stable convergence │");
console.log("│ LAMB                        │ Large batch distributed training     │");
console.log("│ LARS                        │ Very large batch SGD-style training  │");
console.log("└─────────────────────────────┴──────────────────────────────────────┘");

console.log("\n┌─────────────────────────────┬──────────────────────────────────────┐");
console.log("│ Scheduler                   │ Strategy                             │");
console.log("├─────────────────────────────┼──────────────────────────────────────┤");
console.log("│ CyclicLR                    │ Triangular LR cycling                │");
console.log("│ CosineAnnealingWarmRestarts │ Cosine decay with periodic restarts  │");
console.log("│ PolynomialLR                │ Polynomial decay to end LR           │");
console.log("│ LambdaLR                    │ Custom function-based scheduling     │");
console.log("│ SequentialLR                │ Chain multiple schedulers at epochs   │");
console.log("└─────────────────────────────┴──────────────────────────────────────┘");

// ============================================================================
// Summary
// ============================================================================
console.log("\n💡 Key Takeaways");
console.log("-".repeat(60));
console.log("• RAdam: no warmup needed, automatically adjusts adaptive learning rate");
console.log("• LAMB: layer-wise scaling for large batch training (keeps per-layer LR)");
console.log("• LARS: layer-wise rate scaling for SGD-style very large batch training");
console.log("• CyclicLR: avoids local minima by cycling LR between base and max");
console.log("• CosineWarmRestarts: periodic warm restarts explore new loss basins");
console.log("• PolynomialLR: smooth polynomial decay over fixed iterations");
console.log("• LambdaLR: fully custom scheduling via user-defined functions");
console.log("• SequentialLR: combine warmup + decay phases at milestone epochs");

console.log("\n✅ Advanced Optimizers & Schedulers Example Complete!");
console.log("=".repeat(60));
