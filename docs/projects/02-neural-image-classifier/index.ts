/**
 * Neural Network Image Classifier
 *
 * A demonstration of neural network training for image classification
 * using the Deepbox Framework's deep learning modules.
 *
 * Deepbox Modules Used:
 * - deepbox/nn: Neural network layers, Sequential, loss functions
 * - deepbox/ndarray: GradTensor, autograd, tensor operations
 * - deepbox/datasets: loadDigits dataset
 * - deepbox/metrics: Classification metrics
 * - deepbox/preprocess: train/test split
 */

import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { isNumericTypedArray, isTypedArray } from "deepbox/core";
import { loadDigits } from "deepbox/datasets";
import { confusionMatrix, f1Score, precision, recall } from "deepbox/metrics";
import { GradTensor, type Tensor, tensor } from "deepbox/ndarray";
import { crossEntropyLoss } from "deepbox/nn";
import { Adam } from "deepbox/optim";
import { Figure } from "deepbox/plot";
import { StandardScaler, trainTestSplit } from "deepbox/preprocess";
import { setSeed } from "deepbox/random";

import { createModel, createSimpleMLP, getModelSummary } from "./src/models";

// ============================================================================
// Configuration
// ============================================================================

const OUTPUT_DIR = "docs/projects/02-neural-image-classifier/output";
const NUM_CLASSES = 10;
const LEARNING_RATE = 0.001;
const NUM_EPOCHS = 40;
const BATCH_SIZE = 32;

// ============================================================================
// Main Execution
// ============================================================================

console.log("═".repeat(70));
console.log("  NEURAL NETWORK IMAGE CLASSIFIER");
console.log("  Built with Deepbox — TypeScript toolkit for AI & numerical computing");
console.log("═".repeat(70));

setSeed(42);

// Create output directory
if (!existsSync(OUTPUT_DIR)) {
  mkdirSync(OUTPUT_DIR, { recursive: true });
}

// ============================================================================
// Step 1: Load and Prepare Data
// ============================================================================

console.log("\n📊 STEP 1: Loading Digits Dataset");
console.log("─".repeat(70));

const digits = loadDigits();
console.log(`✓ Loaded digits dataset`);
console.log(`  Samples: ${digits.data.shape[0]}`);
console.log(`  Features: ${digits.data.shape[1]} (8x8 pixel images)`);
console.log(`  Classes: ${NUM_CLASSES} (digits 0-9)`);

// Split data
const [XTrain, XTest, yTrain, yTest] = trainTestSplit(digits.data, digits.target, {
  testSize: 0.2,
  randomState: 42,
  shuffle: true,
});

console.log(`\n✓ Split data:`);
console.log(`  Training samples: ${XTrain.shape[0]}`);
console.log(`  Test samples: ${XTest.shape[0]}`);

// Scale features
const scaler = new StandardScaler();
scaler.fit(XTrain);
const XTrainScaled = scaler.transform(XTrain);
const XTestScaled = scaler.transform(XTest);

console.log(`✓ Scaled features using StandardScaler`);

// Convert to arrays for neural network
const trainSize = XTrainScaled.shape[0];
const testSize = XTestScaled.shape[0];
const numFeatures = XTrainScaled.shape[1];

// ============================================================================
// Step 2: Build Neural Network Model
// ============================================================================

console.log("\n🏗️  STEP 2: Building Neural Network");
console.log("─".repeat(70));

// Create a simple MLP
const model = createSimpleMLP(numFeatures, 128, NUM_CLASSES);

console.log("\nModel Architecture:");
console.log(model.toString());

const summary = getModelSummary(model);
console.log(`\nModel Summary:`);
console.log(`  Total Layers: ${summary.numLayers}`);
console.log(`  Total Parameters: ${summary.numParameters}`);

// ============================================================================
// Step 3: Training Setup
// ============================================================================

console.log("\n⚙️  STEP 3: Training Setup");
console.log("─".repeat(70));

// Get model parameters for optimizer
const modelParams = Array.from(model.parameters());
console.log(`  Trainable parameters collected: ${modelParams.length} tensors`);

// Create optimizer
const optimizer = new Adam(model.parameters(), { lr: LEARNING_RATE });
console.log(`  Optimizer: Adam (lr=${LEARNING_RATE})`);
console.log(`  Epochs: ${NUM_EPOCHS}`);
console.log(`  Batch Size: ${BATCH_SIZE}`);

// ============================================================================
// Step 4: Training Loop (Demonstration)
// ============================================================================

console.log("\n🚀 STEP 4: Training Neural Network");
console.log("─".repeat(70));

// Training history
const history: { epochs: number[]; trainLoss: number[]; trainAcc: number[] } = {
  epochs: [],
  trainLoss: [],
  trainAcc: [],
};

const expectNumericTypedArray = (
  value: unknown
): Float32Array | Float64Array | Int32Array | Uint8Array => {
  if (!isTypedArray(value) || !isNumericTypedArray(value)) {
    throw new Error("Expected numeric typed array");
  }
  return value;
};

const scalarFromTensor = (value: Tensor): number => {
  const data = expectNumericTypedArray(value.data);
  return Number(data[value.offset] ?? 0);
};

// Helper to extract data as arrays
function extractData(X: Tensor, y: Tensor): { XArr: number[][]; yArr: number[] } {
  const n = X.shape[0];
  const f = X.shape[1] || 1;

  const XArr: number[][] = [];
  const yArr: number[] = [];

  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < f; j++) {
      row.push(Number(X.at(i, j)));
    }
    XArr.push(row);
    yArr.push(Number(y.at(i)));
  }

  return { XArr, yArr };
}

// Training with mini-batches
console.log("\nTraining Progress:");
console.log("─".repeat(50));

const { XArr: XTrainArr, yArr: yTrainArr } = extractData(XTrainScaled, yTrain);
const createEpochOrder = (epoch: number): number[] => {
  const order = Array.from({ length: trainSize }, (_, index) => index);
  let state = 42 + epoch * 9973;

  for (let i = order.length - 1; i > 0; i--) {
    state = (state * 1664525 + 1013904223) >>> 0;
    const j = state % (i + 1);
    const current = order[i];
    order[i] = order[j] ?? current;
    order[j] = current ?? order[j] ?? 0;
  }

  return order;
};

// Simplified training demonstration
for (let epoch = 0; epoch < NUM_EPOCHS; epoch++) {
  model.train(true);
  let epochLoss = 0;
  let epochCorrect = 0;
  const epochOrder = createEpochOrder(epoch);
  const numBatches = Math.ceil(epochOrder.length / BATCH_SIZE);

  // Process in batches
  for (let batch = 0; batch < numBatches; batch++) {
    const startIdx = batch * BATCH_SIZE;
    const endIdx = Math.min(startIdx + BATCH_SIZE, epochOrder.length);
    const batchSize = endIdx - startIdx;

    // Extract batch
    const batchIndices = epochOrder.slice(startIdx, endIdx);
    const XBatch = batchIndices.map((index) => XTrainArr[index] ?? []);
    const yBatch = batchIndices.map((index) => yTrainArr[index] ?? 0);

    // Create GradTensor input
    const input = GradTensor.fromTensor(tensor(XBatch, { dtype: "float32" }), {
      requiresGrad: false,
    });

    optimizer.zeroGrad();

    // Forward pass
    const output = model.forward(input);
    if (!(output instanceof GradTensor)) {
      throw new Error("Expected GradTensor output during training");
    }

    // Create targets - crossEntropyLoss expects 1D class labels, not one-hot
    const targetsTensor = tensor(yBatch, { dtype: "int32" });
    const loss = crossEntropyLoss(output, targetsTensor);
    epochLoss += scalarFromTensor(loss.tensor);
    loss.backward();
    optimizer.step();

    // Calculate accuracy
    const outData = expectNumericTypedArray(output.tensor.data);
    for (let i = 0; i < batchSize; i++) {
      let maxVal = -Infinity;
      let predClass = 0;
      for (let j = 0; j < NUM_CLASSES; j++) {
        const val = outData[i * NUM_CLASSES + j];
        if (val > maxVal) {
          maxVal = val;
          predClass = j;
        }
      }
      if (predClass === Math.round(yBatch[i])) {
        epochCorrect++;
      }
    }
  }

  const avgLoss = epochLoss / numBatches;
  const trainAcc = epochCorrect / trainSize;

  history.epochs.push(epoch + 1);
  history.trainLoss.push(avgLoss);
  history.trainAcc.push(trainAcc);

  if ((epoch + 1) % 5 === 0 || epoch === 0) {
    console.log(
      `  Epoch ${String(epoch + 1).padStart(2)}: Loss = ${avgLoss.toFixed(4)}, ` +
        `Accuracy = ${(trainAcc * 100).toFixed(2)}%`
    );
  }
}

console.log("─".repeat(50));
console.log("✓ Training Complete!");

// ============================================================================
// Step 5: Model Evaluation
// ============================================================================

console.log("\n📈 STEP 5: Model Evaluation");
console.log("─".repeat(70));

model.train(false);

// Evaluate on test set
const { XArr: XTestArr, yArr: yTestArr } = extractData(XTestScaled, yTest);
const testInput = GradTensor.fromTensor(tensor(XTestArr, { dtype: "float32" }), {
  requiresGrad: false,
});
const testOutputRaw = model.forward(testInput);
const testOutput = testOutputRaw instanceof GradTensor ? testOutputRaw.tensor : testOutputRaw;

// Get predictions
const testOutData = expectNumericTypedArray(testOutput.data);
const predictions: number[] = [];
let testCorrect = 0;

for (let i = 0; i < testSize; i++) {
  let maxVal = -Infinity;
  let predClass = 0;
  for (let j = 0; j < NUM_CLASSES; j++) {
    const val = testOutData[i * NUM_CLASSES + j];
    if (val > maxVal) {
      maxVal = val;
      predClass = j;
    }
  }
  predictions.push(predClass);
  if (predClass === Math.round(yTestArr[i])) {
    testCorrect++;
  }
}

const testAccuracy = testCorrect / testSize;
const predTensor = tensor(predictions);
const yTestTensor = tensor(yTestArr);

console.log("\nTest Set Performance:");
console.log(`  Accuracy:  ${(testAccuracy * 100).toFixed(2)}%`);
console.log(
  `  Precision: ${(Number(precision(yTestTensor, predTensor, "macro")) * 100).toFixed(2)}%`
);
console.log(`  Recall:    ${(Number(recall(yTestTensor, predTensor, "macro")) * 100).toFixed(2)}%`);
console.log(
  `  F1 Score:  ${(Number(f1Score(yTestTensor, predTensor, "macro")) * 100).toFixed(2)}%`
);

// Confusion matrix
console.log("\nConfusion Matrix:");
const cm = confusionMatrix(yTestTensor, predTensor);
const cmData = expectNumericTypedArray(cm.data);

// Print confusion matrix
console.log("        Predicted");
console.log(
  `        ${Array.from({ length: NUM_CLASSES }, (_, i) => i.toString().padStart(3)).join("")}`
);
console.log(`      +${"-".repeat(NUM_CLASSES * 3 + 1)}`);
for (let i = 0; i < NUM_CLASSES; i++) {
  let row = `    ${i} |`;
  for (let j = 0; j < NUM_CLASSES; j++) {
    row += String(cmData[i * NUM_CLASSES + j]).padStart(3);
  }
  console.log(row);
}

// ============================================================================
// Step 6: Model Comparison
// ============================================================================

console.log("\n🔬 STEP 6: Architecture Overview");
console.log("─".repeat(70));

const architectureConfigs = [
  { name: "simple", label: "Simple + ReLU" },
  { name: "gelu", label: "GELU" },
  { name: "leaky", label: "LeakyReLU" },
] as const;

for (const config of architectureConfigs) {
  const candidate = createModel({
    inputSize: numFeatures,
    hiddenSize: 128,
    numClasses: NUM_CLASSES,
    architecture: config.name,
  });
  const candidateSummary = getModelSummary(candidate);
  console.log(
    `  ${config.label.padEnd(16)}: ${candidateSummary.numLayers} layers, ${candidateSummary.numParameters} parameters`
  );
}

// ============================================================================
// Step 7: Visualizations
// ============================================================================

console.log("\n📊 STEP 7: Generating Visualizations");
console.log("─".repeat(70));

// Learning curve plot
try {
  const fig = new Figure({ width: 800, height: 400 });
  const ax = fig.addAxes();

  ax.plot(tensor(history.epochs), tensor(history.trainLoss), {
    color: "#2196F3",
    linewidth: 2,
  });
  ax.setTitle("Training Loss Curve");
  ax.setXLabel("Epoch");
  ax.setYLabel("Loss");

  const svg = fig.renderSVG();
  writeFileSync(`${OUTPUT_DIR}/loss-curve.svg`, svg.svg);
  console.log(`  ✓ Saved: ${OUTPUT_DIR}/loss-curve.svg`);
} catch (e) {
  console.log(`  ⚠ Could not generate loss curve: ${e}`);
}

// Accuracy curve
try {
  const fig = new Figure({ width: 800, height: 400 });
  const ax = fig.addAxes();

  ax.plot(tensor(history.epochs), tensor(history.trainAcc.map((a) => a * 100)), {
    color: "#4CAF50",
    linewidth: 2,
  });
  ax.setTitle("Training Accuracy Curve");
  ax.setXLabel("Epoch");
  ax.setYLabel("Accuracy (%)");

  const svg = fig.renderSVG();
  writeFileSync(`${OUTPUT_DIR}/accuracy-curve.svg`, svg.svg);
  console.log(`  ✓ Saved: ${OUTPUT_DIR}/accuracy-curve.svg`);
} catch (e) {
  console.log(`  ⚠ Could not generate accuracy curve: ${e}`);
}

// ============================================================================
// Step 8: Summary
// ============================================================================

console.log(`\n${"═".repeat(70)}`);
console.log("  TRAINING COMPLETE - SUMMARY");
console.log("═".repeat(70));

console.log("\n📌 Results Summary:\n");
console.log("  Dataset:");
console.log(`    • ${trainSize} training samples`);
console.log(`    • ${testSize} test samples`);
console.log(`    • ${numFeatures} features per sample`);

console.log("\n  Model:");
console.log(`    • Architecture: Simple MLP`);
console.log(`    • Hidden Size: 128`);
console.log(`    • Parameters: ${summary.numParameters}`);

console.log("\n  Training:");
console.log(`    • Epochs: ${NUM_EPOCHS}`);
console.log(`    • Final Loss: ${history.trainLoss[history.trainLoss.length - 1].toFixed(4)}`);
console.log(
  `    • Final Train Acc: ${(history.trainAcc[history.trainAcc.length - 1] * 100).toFixed(2)}%`
);

console.log("\n  Evaluation:");
console.log(`    • Test Accuracy: ${(testAccuracy * 100).toFixed(2)}%`);

console.log("\n📁 Output Files:");
console.log(`   • ${OUTPUT_DIR}/loss-curve.svg`);
console.log(`   • ${OUTPUT_DIR}/accuracy-curve.svg`);

console.log(`\n${"═".repeat(70)}`);
console.log("  ✅ Neural Network Image Classifier Complete!");
console.log("═".repeat(70));
