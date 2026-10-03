/**
 * Example 41: Advanced Optimizers & Schedulers
 *
 * The RAdam, LAMB and LARS optimizers, and the CyclicLR, CosineAnnealingWarmRestarts,
 * PolynomialLR, LambdaLR and SequentialLR learning rate schedulers.
 *
 * Training uses plain tensors: model.forward(x) returns a tensor that tracks the
 * weights, so loss.backward() and loss.item() work without wrapping the data.
 */

import { tensor } from "deepbox/ndarray";
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
import { setSeed } from "deepbox/random";

console.log("=".repeat(60));
console.log("Example 41: Advanced Optimizers & Schedulers");
console.log("=".repeat(60));

// ============================================================================
// Helper: a short training run
// ============================================================================

function trainDemo(
  model: Sequential,
  optimizerName: string,
  optimizer: { step: () => void; zeroGrad: () => void; lr: number },
  epochs = 30
): void {
  // The task is y = x1 + x2. The data is scaled down so that the three optimizers
  // train steadily at these learning rates.
  const xTrain = tensor([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
  ]).div(4);
  const yTrain = tensor([[3], [7], [11], [15]]).div(4);

  console.log(`\n  ${optimizerName} (initial lr=${optimizer.lr.toFixed(6)}):`);

  for (let epoch = 1; epoch <= epochs; epoch++) {
    optimizer.zeroGrad();
    const loss = mseLoss(model.forward(xTrain), yTrain);
    loss.backward();
    optimizer.step();

    if (epoch === 1 || epoch % 10 === 0) {
      console.log(
        `    Epoch ${String(epoch).padStart(3)}: loss=${Number(loss.item()).toFixed(6)}, lr=${optimizer.lr.toFixed(6)}`
      );
    }
  }
}

// ============================================================================
// Part 1: RAdam (Rectified Adam)
// ============================================================================
console.log("\nPart 1: RAdam (Rectified Adam)");
console.log("-".repeat(60));

// RAdam corrects the variance of Adam's adaptive step in the first iterations,
// which is the problem a learning rate warmup is normally used for
setSeed(42); // the same starting weights for each optimizer
const model1 = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const radam = new RAdam(model1.parameters(), { lr: 0.01 });

console.log("RAdam: Rectified Adam, a variant of Adam that needs no warmup");
trainDemo(model1, "RAdam", radam);

// ============================================================================
// Part 2: LAMB (Layer-wise Adaptive Moments)
// ============================================================================
console.log("\nPart 2: LAMB");
console.log("-".repeat(60));

// LAMB scales each layer's update by the ratio of its weight norm to its update norm
setSeed(42);
const model2 = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const lamb = new LAMB(model2.parameters(), { lr: 0.01 });

console.log("LAMB: layer-wise adaptive moments, designed for large batches");
trainDemo(model2, "LAMB", lamb);

// ============================================================================
// Part 3: LARS (Layer-wise Adaptive Rate Scaling)
// ============================================================================
console.log("\nPart 3: LARS");
console.log("-".repeat(60));

// LARS scales each layer's learning rate by the ratio of its weight norm to its gradient norm
setSeed(42);
const model3 = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const lars = new LARS(model3.parameters(), { lr: 3 }); // LARS shrinks each step by a trust ratio, so it needs a large lr

console.log("LARS: layer-wise adaptive rate scaling, designed for very large batches");
trainDemo(model3, "LARS", lars);

// ============================================================================
// Part 4: CyclicLR Scheduler
// ============================================================================
console.log("\nPart 4: CyclicLR Scheduler");
console.log("-".repeat(60));

// CyclicLR moves the learning rate up from baseLr to maxLr and back down, in a repeating cycle
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
console.log("\nPart 5: CosineAnnealingWarmRestarts");
console.log("-".repeat(60));

// Cosine annealing from the initial rate down to etaMin, then a restart at the initial rate
const model5 = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const adam5 = new Adam(model5.parameters(), { lr: 0.01 });
const cosineWR = new CosineAnnealingWarmRestarts(adam5, {
  t0: 5, // first cycle lasts 5 epochs (T_0 is accepted as well)
  tMult: 2, // each cycle is twice as long as the one before (T_mult is accepted as well)
  etaMin: 0.001,
});

console.log("CosineAnnealingWarmRestarts: t0=5, tMult=2, etaMin=0.001");
console.log("  LR schedule (the rate returns to 0.01 after 5 steps, then after 15, ...):");
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
console.log("\nPart 6: PolynomialLR");
console.log("-".repeat(60));

// Polynomial decay from the initial rate to the end rate
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
console.log("\nPart 7: LambdaLR");
console.log("-".repeat(60));

// LambdaLR multiplies the initial rate by the value of your own function of the epoch
const model7 = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const adam7 = new Adam(model7.parameters(), { lr: 0.01 });
const lambdaLr = new LambdaLR(adam7, {
  lrLambda: (epoch: number) => 0.95 ** epoch,
});

console.log("LambdaLR: lrLambda = 0.95^epoch (exponential decay)");
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
console.log("\nPart 8: SequentialLR");
console.log("-".repeat(60));

// SequentialLR switches from one scheduler to the next at the given milestones
const model8 = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
const adam8 = new Adam(model8.parameters(), { lr: 0.01 });

// Phase 1: warmup with LambdaLR (epochs 0 to 4)
const warmup = new LambdaLR(adam8, {
  lrLambda: (epoch: number) => Math.min(1.0, (epoch + 1) / 5),
});

// Phase 2: StepLR decay (epoch 5 onward)
const decay = new StepLR(adam8, { stepSize: 3, gamma: 0.5 });

const seqLr = new SequentialLR(adam8, {
  schedulers: [warmup, decay],
  milestones: [5],
});

console.log("SequentialLR: LambdaLR warmup for epochs 0 to 4, then StepLR decay");
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
console.log("\nPart 9: Which One to Use");
console.log("-".repeat(60));

console.log("Optimizers:");
console.log("• Adam: the usual default");
console.log("• RAdam: Adam with a built-in correction, so no warmup schedule");
console.log("• LAMB: Adam-style updates scaled per layer, for large-batch training");
console.log("• LARS: SGD-style updates scaled per layer, for very large batches");

console.log("\nSchedulers:");
console.log("• CyclicLR: the rate moves between baseLr and maxLr in a repeating cycle");
console.log("• CosineAnnealingWarmRestarts: cosine decay, restarted at the initial rate");
console.log("• PolynomialLR: polynomial decay to an end rate");
console.log("• LambdaLR: any function of the epoch");
console.log("• SequentialLR: a different scheduler for each stretch of epochs");

// ============================================================================
// Summary
// ============================================================================
console.log("\nKey Takeaways");
console.log("-".repeat(60));
console.log("• RAdam: Adam with a variance correction instead of a warmup schedule");
console.log("• LAMB, LARS: per-layer scaling for large batches");
console.log("• CyclicLR: the learning rate cycles between a lower and an upper bound");
console.log("• CosineAnnealingWarmRestarts: cosine decay with periodic restarts");
console.log("• PolynomialLR: smooth decay over a fixed number of iterations");
console.log("• LambdaLR: a schedule defined by your own function");
console.log("• SequentialLR: chain a warmup and a decay phase at a milestone epoch");
console.log("• Optimizers skip parameters that have no gradient, so frozen layers are safe");

console.log("\nAdvanced Optimizers & Schedulers Example Complete!");
console.log("=".repeat(60));
