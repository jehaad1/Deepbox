/**
 * Customer Churn Prediction System
 *
 * A comprehensive ML pipeline for predicting customer churn using
 * classical machine learning algorithms.
 *
 * Deepbox Modules Used:
 * - deepbox/ml: Classical ML models
 * - deepbox/preprocess: Data preprocessing, train/test split, cross-validation
 * - deepbox/metrics: Classification metrics
 * - deepbox/dataframe: Data manipulation
 * - deepbox/stats: Statistical analysis
 * - deepbox/plot: Visualization
 */

import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { isNumericTypedArray, isTypedArray } from "deepbox/core";
import { DataFrame } from "deepbox/dataframe";
import { accuracy, confusionMatrix, f1Score, precision, recall } from "deepbox/metrics";
import {
  DecisionTreeClassifier,
  GaussianNB,
  GradientBoostingClassifier,
  KNeighborsClassifier,
  LogisticRegression,
  RandomForestClassifier,
} from "deepbox/ml";
import { type Tensor, tensor } from "deepbox/ndarray";
import { Figure } from "deepbox/plot";
import { KFold, StandardScaler, trainTestSplit } from "deepbox/preprocess";

// ============================================================================
// Configuration
// ============================================================================

const OUTPUT_DIR = "docs/projects/03-customer-churn-prediction/output";
const NUM_SAMPLES = 1000;
const NUM_FEATURES = 10;
const TEST_SIZE = 0.2;
const RANDOM_STATE = 42;

const expectNumericTypedArray = (
  value: unknown
): Float32Array | Float64Array | Int32Array | Uint8Array => {
  if (!isTypedArray(value) || !isNumericTypedArray(value)) {
    throw new Error("Expected numeric typed array");
  }
  return value;
};

// ============================================================================
// Data Generation
// ============================================================================

/**
 * Generate synthetic customer churn dataset
 */
function generateChurnData(
  numSamples: number,
  seed = 42
): {
  X: Tensor;
  y: Tensor;
  featureNames: string[];
} {
  // Seeded random for reproducibility
  let randomSeed = seed;
  const seededRandom = () => {
    randomSeed = (randomSeed * 1103515245 + 12345) & 0x7fffffff;
    return randomSeed / 0x7fffffff;
  };

  const randomNormal = (mean: number, std: number) => {
    const u1 = seededRandom();
    const u2 = seededRandom();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return mean + std * z;
  };

  const featureNames = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_products",
    "has_contract",
    "support_calls",
    "payment_delay_days",
    "age",
    "satisfaction_score",
    "usage_frequency",
  ];

  const X: number[][] = [];
  const y: number[] = [];

  for (let i = 0; i < numSamples; i++) {
    // Generate features
    const tenure = Math.max(1, Math.round(randomNormal(24, 18))); // months
    const monthlyCharges = Math.max(20, randomNormal(65, 30));
    const totalCharges = tenure * monthlyCharges * (0.8 + seededRandom() * 0.4);
    const numProducts = Math.round(Math.max(1, Math.min(5, randomNormal(2, 1))));
    const hasContract = seededRandom() > 0.4 ? 1 : 0;
    const supportCalls = Math.round(Math.max(0, randomNormal(2, 3)));
    const paymentDelay = Math.max(0, Math.round(randomNormal(5, 10)));
    const age = Math.round(Math.max(18, Math.min(80, randomNormal(42, 15))));
    const satisfaction = Math.max(1, Math.min(10, randomNormal(6, 2)));
    const usageFreq = Math.max(0, randomNormal(15, 8)); // days per month

    X.push([
      tenure,
      monthlyCharges,
      totalCharges,
      numProducts,
      hasContract,
      supportCalls,
      paymentDelay,
      age,
      satisfaction,
      usageFreq,
    ]);

    // Churn probability based on features
    let churnProb = 0.2; // base probability

    // Higher charges increase churn
    if (monthlyCharges > 80) churnProb += 0.15;
    // Short tenure increases churn
    if (tenure < 12) churnProb += 0.2;
    // No contract increases churn
    if (hasContract === 0) churnProb += 0.15;
    // Many support calls increase churn
    if (supportCalls > 4) churnProb += 0.2;
    // Payment delays increase churn
    if (paymentDelay > 10) churnProb += 0.15;
    // Low satisfaction increases churn
    if (satisfaction < 4) churnProb += 0.25;
    // Low usage increases churn
    if (usageFreq < 5) churnProb += 0.1;

    // Cap probability
    churnProb = Math.min(0.9, Math.max(0.1, churnProb));

    // Generate label
    y.push(seededRandom() < churnProb ? 1 : 0);
  }

  return {
    X: tensor(X),
    y: tensor(y),
    featureNames,
  };
}

// ============================================================================
// Main Execution
// ============================================================================

console.log("═".repeat(70));
console.log("  CUSTOMER CHURN PREDICTION SYSTEM");
console.log("  Built with Deepbox - TypeScript Data Science & ML Library");
console.log("═".repeat(70));

// Create output directory
if (!existsSync(OUTPUT_DIR)) {
  mkdirSync(OUTPUT_DIR, { recursive: true });
}

// ============================================================================
// Step 1: Generate and Explore Data
// ============================================================================

console.log("\n📊 STEP 1: Data Generation and Exploration");
console.log("─".repeat(70));

const { X, y, featureNames } = generateChurnData(NUM_SAMPLES, RANDOM_STATE);

console.log(`\n✓ Generated synthetic customer data`);
console.log(`  Samples: ${NUM_SAMPLES}`);
console.log(`  Features: ${NUM_FEATURES}`);

// Class distribution
const yData = expectNumericTypedArray(y.data);
const numChurned = Array.from(yData).filter((v) => v === 1).length;
const numRetained = NUM_SAMPLES - numChurned;

console.log(`\nClass Distribution:`);
console.log(`  Churned (1):  ${numChurned} (${((numChurned / NUM_SAMPLES) * 100).toFixed(1)}%)`);
console.log(`  Retained (0): ${numRetained} (${((numRetained / NUM_SAMPLES) * 100).toFixed(1)}%)`);

// Feature statistics
console.log(`\nFeature Statistics:`);
const XData = expectNumericTypedArray(X.data);

const statsDF = new DataFrame({
  Feature: featureNames,
  Mean: featureNames.map((_, i) => {
    let sum = 0;
    for (let j = 0; j < NUM_SAMPLES; j++) {
      sum += XData[j * NUM_FEATURES + i];
    }
    return (sum / NUM_SAMPLES).toFixed(2);
  }),
  Std: featureNames.map((_, i) => {
    let sum = 0;
    let sumSq = 0;
    for (let j = 0; j < NUM_SAMPLES; j++) {
      const val = XData[j * NUM_FEATURES + i];
      sum += val;
      sumSq += val * val;
    }
    const mean = sum / NUM_SAMPLES;
    const variance = sumSq / NUM_SAMPLES - mean * mean;
    return Math.sqrt(variance).toFixed(2);
  }),
  Min: featureNames.map((_, i) => {
    let min = Infinity;
    for (let j = 0; j < NUM_SAMPLES; j++) {
      min = Math.min(min, XData[j * NUM_FEATURES + i]);
    }
    return min.toFixed(2);
  }),
  Max: featureNames.map((_, i) => {
    let max = -Infinity;
    for (let j = 0; j < NUM_SAMPLES; j++) {
      max = Math.max(max, XData[j * NUM_FEATURES + i]);
    }
    return max.toFixed(2);
  }),
});

console.log(statsDF.toString());

// ============================================================================
// Step 2: Data Preprocessing
// ============================================================================

console.log("\n🔄 STEP 2: Data Preprocessing");
console.log("─".repeat(70));

// Train/test split
const [XTrain, XTest, yTrain, yTest] = trainTestSplit(X, y, {
  testSize: TEST_SIZE,
  randomState: RANDOM_STATE,
  shuffle: true,
});

console.log(`\n✓ Train/Test Split:`);
console.log(`  Training samples: ${XTrain.shape[0]}`);
console.log(`  Test samples: ${XTest.shape[0]}`);

// Feature scaling
const scaler = new StandardScaler();
scaler.fit(XTrain);
const XTrainScaled = scaler.transform(XTrain);
const XTestScaled = scaler.transform(XTest);

console.log(`✓ Applied StandardScaler`);

// ============================================================================
// Step 3: Model Training and Evaluation
// ============================================================================

console.log("\n🤖 STEP 3: Model Training and Evaluation");
console.log("─".repeat(70));

// Define models to compare
const models: {
  name: string;
  model:
    | LogisticRegression
    | DecisionTreeClassifier
    | RandomForestClassifier
    | GradientBoostingClassifier
    | KNeighborsClassifier
    | GaussianNB;
  params: string;
}[] = [
  {
    name: "Logistic Regression",
    model: new LogisticRegression({ maxIter: 100, learningRate: 0.1 }),
    params: "maxIter=100, lr=0.1",
  },
  {
    name: "Decision Tree",
    model: new DecisionTreeClassifier({ maxDepth: 5 }),
    params: "maxDepth=5",
  },
  {
    name: "Random Forest",
    model: new RandomForestClassifier({
      nEstimators: 50,
      maxDepth: 5,
      randomState: RANDOM_STATE,
    }),
    params: "nEstimators=50, maxDepth=5",
  },
  {
    name: "Gradient Boosting",
    model: new GradientBoostingClassifier({
      nEstimators: 50,
      maxDepth: 3,
      learningRate: 0.1,
    }),
    params: "nEstimators=50, maxDepth=3, lr=0.1",
  },
  {
    name: "KNN",
    model: new KNeighborsClassifier({ nNeighbors: 5 }),
    params: "k=5",
  },
  {
    name: "Naive Bayes",
    model: new GaussianNB(),
    params: "default",
  },
];

const results: {
  name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  trainTime: number;
}[] = [];

console.log("\nTraining models...\n");

for (const { name, model, params: _params } of models) {
  const startTime = Date.now();

  try {
    // Train model
    model.fit(XTrainScaled, yTrain);

    // Predict
    const yPred = model.predict(XTestScaled);

    // Calculate metrics
    const acc = accuracy(yTest, yPred);
    const prec = precision(yTest, yPred, "binary");
    const rec = recall(yTest, yPred, "binary");
    const f1 = f1Score(yTest, yPred, "binary");

    const trainTime = Date.now() - startTime;

    results.push({
      name,
      accuracy: Number(acc),
      precision: Number(prec),
      recall: Number(rec),
      f1: Number(f1),
      trainTime,
    });

    console.log(
      `  ✓ ${name.padEnd(20)} - Accuracy: ${(Number(acc) * 100).toFixed(2)}% (${trainTime}ms)`
    );
  } catch (error) {
    console.log(`  ✗ ${name.padEnd(20)} - Error: ${error}`);
  }
}

// ============================================================================
// Step 4: Model Comparison
// ============================================================================

console.log("\n📈 STEP 4: Model Comparison");
console.log("─".repeat(70));

// Sort by F1 score
results.sort((a, b) => b.f1 - a.f1);

const comparisonDF = new DataFrame({
  Model: results.map((r) => r.name),
  "Accuracy (%)": results.map((r) => (r.accuracy * 100).toFixed(2)),
  "Precision (%)": results.map((r) => (r.precision * 100).toFixed(2)),
  "Recall (%)": results.map((r) => (r.recall * 100).toFixed(2)),
  "F1 Score (%)": results.map((r) => (r.f1 * 100).toFixed(2)),
  "Time (ms)": results.map((r) => r.trainTime.toString()),
});

console.log("\nModel Performance Comparison (sorted by F1 Score):\n");
console.log(comparisonDF.toString());

// Best model
const bestModel = results[0];
console.log(`\n🏆 Best Model: ${bestModel.name}`);
console.log(`   F1 Score: ${(bestModel.f1 * 100).toFixed(2)}%`);

// ============================================================================
// Step 5: Cross-Validation
// ============================================================================

console.log("\n🔄 STEP 5: Cross-Validation (Best Model)");
console.log("─".repeat(70));

// Re-train best model type for cross-validation
const bestModelType = results[0].name;
console.log(`\nPerforming 5-Fold Cross-Validation on ${bestModelType}...`);

const kfold = new KFold({
  nSplits: 5,
  shuffle: true,
  randomState: RANDOM_STATE,
});
const cvScores: number[] = [];

let foldNum = 1;
for (const [trainIdx, valIdx] of kfold.split(X)) {
  // Extract fold data
  const XTrainFold: number[][] = [];
  const yTrainFold: number[] = [];
  const XValFold: number[][] = [];
  const yValFold: number[] = [];

  for (const idx of trainIdx) {
    const row: number[] = [];
    for (let j = 0; j < NUM_FEATURES; j++) {
      row.push(XData[idx * NUM_FEATURES + j]);
    }
    XTrainFold.push(row);
    yTrainFold.push(yData[idx]);
  }

  for (const idx of valIdx) {
    const row: number[] = [];
    for (let j = 0; j < NUM_FEATURES; j++) {
      row.push(XData[idx * NUM_FEATURES + j]);
    }
    XValFold.push(row);
    yValFold.push(yData[idx]);
  }

  // Scale
  const foldScaler = new StandardScaler();
  foldScaler.fit(tensor(XTrainFold));
  const XTrainFoldScaled = foldScaler.transform(tensor(XTrainFold));
  const XValFoldScaled = foldScaler.transform(tensor(XValFold));

  // Train and evaluate
  let cvModel:
    | LogisticRegression
    | DecisionTreeClassifier
    | RandomForestClassifier
    | GradientBoostingClassifier
    | KNeighborsClassifier
    | GaussianNB;
  if (bestModelType === "Random Forest") {
    cvModel = new RandomForestClassifier({
      nEstimators: 50,
      maxDepth: 5,
      randomState: RANDOM_STATE,
    });
  } else if (bestModelType === "Gradient Boosting") {
    cvModel = new GradientBoostingClassifier({
      nEstimators: 50,
      maxDepth: 3,
      learningRate: 0.1,
    });
  } else if (bestModelType === "Logistic Regression") {
    cvModel = new LogisticRegression({ maxIter: 100, learningRate: 0.1 });
  } else {
    cvModel = new DecisionTreeClassifier({ maxDepth: 5 });
  }

  cvModel.fit(XTrainFoldScaled, tensor(yTrainFold));
  const yPredFold = cvModel.predict(XValFoldScaled);
  const foldAcc = Number(accuracy(tensor(yValFold), yPredFold));

  cvScores.push(foldAcc);
  console.log(`  Fold ${foldNum}: Accuracy = ${(foldAcc * 100).toFixed(2)}%`);
  foldNum++;
}

const cvMean = cvScores.reduce((a, b) => a + b, 0) / cvScores.length;
const cvStd = Math.sqrt(cvScores.reduce((sum, s) => sum + (s - cvMean) ** 2, 0) / cvScores.length);

console.log(`\n  CV Mean Accuracy: ${(cvMean * 100).toFixed(2)}% ± ${(cvStd * 100).toFixed(2)}%`);

// ============================================================================
// Step 6: Confusion Matrix Analysis
// ============================================================================

console.log("\n📊 STEP 6: Confusion Matrix Analysis");
console.log("─".repeat(70));

// Re-train best model for detailed analysis
let analysisModel:
  | LogisticRegression
  | DecisionTreeClassifier
  | RandomForestClassifier
  | GradientBoostingClassifier
  | KNeighborsClassifier
  | GaussianNB;
if (bestModelType === "Random Forest") {
  analysisModel = new RandomForestClassifier({
    nEstimators: 50,
    maxDepth: 5,
    randomState: RANDOM_STATE,
  });
} else if (bestModelType === "Gradient Boosting") {
  analysisModel = new GradientBoostingClassifier({
    nEstimators: 50,
    maxDepth: 3,
    learningRate: 0.1,
  });
} else {
  analysisModel = new LogisticRegression({ maxIter: 100, learningRate: 0.1 });
}

analysisModel.fit(XTrainScaled, yTrain);
const yPredFinal = analysisModel.predict(XTestScaled);

const cm = confusionMatrix(yTest, yPredFinal);
const cmData = expectNumericTypedArray(cm.data);

console.log("\nConfusion Matrix:");
console.log("                  Predicted");
console.log("                  Retained  Churned");
console.log(
  `  Actual Retained    ${String(cmData[0]).padStart(4)}     ${String(cmData[1]).padStart(4)}`
);
console.log(
  `  Actual Churned     ${String(cmData[2]).padStart(4)}     ${String(cmData[3]).padStart(4)}`
);

const tn = cmData[0];
const fp = cmData[1];
const fn = cmData[2];
const tp = cmData[3];

console.log(`\n  True Negatives:  ${tn} (correctly predicted retained)`);
console.log(`  False Positives: ${fp} (incorrectly predicted churned)`);
console.log(`  False Negatives: ${fn} (missed churns)`);
console.log(`  True Positives:  ${tp} (correctly predicted churned)`);

// Business metrics
const detectionRate = tp / (tp + fn);
const falseAlarmRate = fp / (fp + tn);

console.log(`\nBusiness Metrics:`);
console.log(`  Churn Detection Rate: ${(detectionRate * 100).toFixed(1)}%`);
console.log(`  False Alarm Rate:     ${(falseAlarmRate * 100).toFixed(1)}%`);

// ============================================================================
// Step 7: Feature Importance (for tree-based models)
// ============================================================================

console.log("\n🔍 STEP 7: Feature Importance Analysis");
console.log("─".repeat(70));

// Train Random Forest for feature importance
const rfForImportance = new RandomForestClassifier({
  nEstimators: 100,
});
rfForImportance.fit(XTrainScaled, yTrain);

// Note: Feature importance analysis would require the model to expose featureImportances
// This is a limitation of the current Deepbox ML implementation
console.log("\n  Note: Feature importance requires tree model internals access");
console.log("  Key predictive features based on domain knowledge:");
console.log("    1. satisfaction_score - Low satisfaction strongly predicts churn");
console.log("    2. has_contract - No contract increases churn risk");
console.log("    3. support_calls - High support calls indicate dissatisfaction");
console.log("    4. tenure_months - Short tenure correlates with higher churn");
console.log("    5. payment_delay_days - Payment issues signal disengagement");

// ============================================================================
// Step 8: Visualizations
// ============================================================================

console.log("\n📊 STEP 8: Generating Visualizations");
console.log("─".repeat(70));

// Model comparison bar chart
try {
  const fig = new Figure({ width: 800, height: 500 });
  const ax = fig.addAxes();

  const modelNames = results.map((_, i) => i);
  const f1Scores = results.map((r) => r.f1 * 100);

  ax.bar(tensor(modelNames), tensor(f1Scores), { color: "#4CAF50" });
  ax.setTitle("Model Comparison (F1 Score)");
  ax.setXLabel("Model");
  ax.setYLabel("F1 Score (%)");

  const svg = fig.renderSVG();
  writeFileSync(`${OUTPUT_DIR}/model-comparison.svg`, svg.svg);
  console.log(`  ✓ Saved: ${OUTPUT_DIR}/model-comparison.svg`);
} catch (e) {
  console.log(`  ⚠ Could not generate model comparison plot: ${e}`);
}

// Cross-validation scores plot
try {
  const fig = new Figure({ width: 800, height: 400 });
  const ax = fig.addAxes();

  const folds = cvScores.map((_, i) => i + 1);
  ax.bar(tensor(folds), tensor(cvScores.map((s) => s * 100)), {
    color: "#2196F3",
  });
  ax.setTitle("Cross-Validation Scores");
  ax.setXLabel("Fold");
  ax.setYLabel("Accuracy (%)");

  const svg = fig.renderSVG();
  writeFileSync(`${OUTPUT_DIR}/cv-scores.svg`, svg.svg);
  console.log(`  ✓ Saved: ${OUTPUT_DIR}/cv-scores.svg`);
} catch (e) {
  console.log(`  ⚠ Could not generate CV scores plot: ${e}`);
}

// ============================================================================
// Step 9: Summary and Recommendations
// ============================================================================

console.log(`\n${"═".repeat(70)}`);
console.log("  ANALYSIS COMPLETE - SUMMARY");
console.log("═".repeat(70));

console.log("\n📌 Key Findings:\n");
console.log("  1. Dataset Overview:");
console.log(`     • ${NUM_SAMPLES} customers analyzed`);
console.log(`     • ${((numChurned / NUM_SAMPLES) * 100).toFixed(1)}% churn rate`);

console.log("\n  2. Best Performing Model:");
console.log(`     • ${bestModel.name}`);
console.log(`     • Accuracy: ${(bestModel.accuracy * 100).toFixed(2)}%`);
console.log(`     • F1 Score: ${(bestModel.f1 * 100).toFixed(2)}%`);
console.log(`     • CV Score: ${(cvMean * 100).toFixed(2)}% ± ${(cvStd * 100).toFixed(2)}%`);

console.log("\n  3. Business Impact:");
console.log(`     • Can detect ${(detectionRate * 100).toFixed(1)}% of churning customers`);
console.log(`     • False alarm rate: ${(falseAlarmRate * 100).toFixed(1)}%`);

console.log("\n💡 Recommendations:");
console.log("   • Focus retention efforts on customers with:");
console.log("     - Low satisfaction scores");
console.log("     - No contract");
console.log("     - High support call frequency");
console.log("   • Consider ensemble methods for production deployment");
console.log("   • Implement model monitoring for drift detection");

console.log("\n📁 Output Files:");
console.log(`   • ${OUTPUT_DIR}/model-comparison.svg`);
console.log(`   • ${OUTPUT_DIR}/cv-scores.svg`);

console.log(`\n${"═".repeat(70)}`);
console.log("  ✅ Customer Churn Prediction Complete!");
console.log("═".repeat(70));
