/**
 * Customer Churn Prediction System
 *
 * Predicts customer churn on a synthetic dataset. Six classical models are
 * trained and compared, then the best one is cross-validated and inspected.
 *
 * Deepbox Modules Used:
 * - deepbox/ml: Classifiers, Pipeline, crossValScore
 * - deepbox/preprocess: StandardScaler, trainTestSplit
 * - deepbox/metrics: Classification metrics
 * - deepbox/dataframe: Tables for console output
 * - deepbox/plot: SVG charts
 */

import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { DataFrame } from "deepbox/dataframe";
import { accuracy, confusionMatrix, f1Score, precision, recall } from "deepbox/metrics";
import {
  crossValScore,
  DecisionTreeClassifier,
  GaussianNB,
  GradientBoostingClassifier,
  KNeighborsClassifier,
  LogisticRegression,
  Pipeline,
  RandomForestClassifier,
} from "deepbox/ml";
import { type Tensor, tensor } from "deepbox/ndarray";
import { Figure } from "deepbox/plot";
import { StandardScaler, trainTestSplit } from "deepbox/preprocess";

// ============================================================================
// Configuration
// ============================================================================

const OUTPUT_DIR = "docs/projects/03-customer-churn-prediction/output";
const NUM_SAMPLES = 1000;
const NUM_FEATURES = 10;
const TEST_SIZE = 0.2;
const RANDOM_STATE = 42;

type ChurnClassifier =
  | LogisticRegression
  | DecisionTreeClassifier
  | RandomForestClassifier
  | GradientBoostingClassifier
  | KNeighborsClassifier
  | GaussianNB;

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
console.log("  Built with Deepbox: TypeScript toolkit for AI & numerical computing");
console.log("═".repeat(70));

// Create output directory
if (!existsSync(OUTPUT_DIR)) {
  mkdirSync(OUTPUT_DIR, { recursive: true });
}

// ============================================================================
// Step 1: Generate and Explore Data
// ============================================================================

console.log("\nSTEP 1: Data Generation and Exploration");
console.log("─".repeat(70));

const { X, y, featureNames } = generateChurnData(NUM_SAMPLES, RANDOM_STATE);

console.log(`\nGenerated synthetic customer data`);
console.log(`  Samples: ${NUM_SAMPLES}`);
console.log(`  Features: ${NUM_FEATURES}`);

// Class distribution
const yData = y.toArray() as number[];
const numChurned = yData.filter((v) => v === 1).length;
const numRetained = NUM_SAMPLES - numChurned;

console.log(`\nClass Distribution:`);
console.log(`  Churned (1):  ${numChurned} (${((numChurned / NUM_SAMPLES) * 100).toFixed(1)}%)`);
console.log(`  Retained (0): ${numRetained} (${((numRetained / NUM_SAMPLES) * 100).toFixed(1)}%)`);

// Feature statistics
console.log(`\nFeature Statistics:`);
const XRows = X.toArray() as number[][];

const statsDF = new DataFrame({
  Feature: featureNames,
  Mean: featureNames.map((_, i) => {
    let sum = 0;
    for (let j = 0; j < NUM_SAMPLES; j++) {
      sum += XRows[j][i];
    }
    return (sum / NUM_SAMPLES).toFixed(2);
  }),
  Std: featureNames.map((_, i) => {
    let sum = 0;
    let sumSq = 0;
    for (let j = 0; j < NUM_SAMPLES; j++) {
      const val = XRows[j][i];
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
      min = Math.min(min, XRows[j][i]);
    }
    return min.toFixed(2);
  }),
  Max: featureNames.map((_, i) => {
    let max = -Infinity;
    for (let j = 0; j < NUM_SAMPLES; j++) {
      max = Math.max(max, XRows[j][i]);
    }
    return max.toFixed(2);
  }),
});

console.log(statsDF.toString());

// ============================================================================
// Step 2: Data Preprocessing
// ============================================================================

console.log("\nSTEP 2: Data Preprocessing");
console.log("─".repeat(70));

// Train/test split
const [XTrain, XTest, yTrain, yTest] = trainTestSplit(X, y, {
  testSize: TEST_SIZE,
  randomState: RANDOM_STATE,
  shuffle: true,
});

console.log(`\nTrain/Test Split:`);
console.log(`  Training samples: ${XTrain.shape[0]}`);
console.log(`  Test samples: ${XTest.shape[0]}`);

// Feature scaling
const scaler = new StandardScaler();
scaler.fit(XTrain);
const XTrainScaled = scaler.transform(XTrain);
const XTestScaled = scaler.transform(XTest);

console.log(`Applied StandardScaler`);

// ============================================================================
// Step 3: Model Training and Evaluation
// ============================================================================

console.log("\nSTEP 3: Model Training and Evaluation");
console.log("─".repeat(70));

// Each entry builds a fresh, unfitted model, so the same settings can be reused
// for cross-validation and for the final analysis.
const models: { name: string; create: () => ChurnClassifier }[] = [
  {
    name: "Logistic Regression",
    create: () => new LogisticRegression({ maxIter: 100, learningRate: 0.1 }),
  },
  {
    name: "Decision Tree",
    create: () => new DecisionTreeClassifier({ maxDepth: 5 }),
  },
  {
    name: "Random Forest",
    create: () =>
      new RandomForestClassifier({ nEstimators: 50, maxDepth: 5, randomState: RANDOM_STATE }),
  },
  {
    name: "Gradient Boosting",
    create: () =>
      new GradientBoostingClassifier({ nEstimators: 50, maxDepth: 3, learningRate: 0.1 }),
  },
  {
    name: "KNN",
    create: () => new KNeighborsClassifier({ nNeighbors: 5 }),
  },
  {
    name: "Naive Bayes",
    create: () => new GaussianNB(),
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

for (const { name, create } of models) {
  const startTime = Date.now();
  const model = create();

  try {
    model.fit(XTrainScaled, yTrain);
    const yPred = model.predict(XTestScaled);

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
      `  ${name.padEnd(20)} - Accuracy: ${(Number(acc) * 100).toFixed(2)}% (${trainTime}ms)`
    );
  } catch (error) {
    console.log(`  ${name.padEnd(20)} - Error: ${error}`);
  }
}

// ============================================================================
// Step 4: Model Comparison
// ============================================================================

console.log("\nSTEP 4: Model Comparison");
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
console.log(`\nBest Model: ${bestModel.name}`);
console.log(`  F1 Score: ${(bestModel.f1 * 100).toFixed(2)}%`);

// ============================================================================
// Step 5: Cross-Validation
// ============================================================================

console.log("\nSTEP 5: Cross-Validation (Best Model)");
console.log("─".repeat(70));

const bestModelType = bestModel.name;
const createBest = (models.find((m) => m.name === bestModelType) ?? models[0]).create;
console.log(`\nPerforming 5-fold cross-validation on ${bestModelType}...`);

// The scaler sits inside the pipeline, so each fold fits it on its own training rows only.
// crossValScore stratifies the folds by class for classifiers.
const cvScores = crossValScore(
  new Pipeline([
    ["scaler", new StandardScaler()],
    ["clf", createBest()],
  ]),
  X,
  y,
  5
);

cvScores.forEach((score, i) => {
  console.log(`  Fold ${i + 1}: Accuracy = ${(score * 100).toFixed(2)}%`);
});

const cvMean = cvScores.reduce((a, b) => a + b, 0) / cvScores.length;
const cvStd = Math.sqrt(cvScores.reduce((sum, s) => sum + (s - cvMean) ** 2, 0) / cvScores.length);

console.log(`\n  CV Mean Accuracy: ${(cvMean * 100).toFixed(2)}% ± ${(cvStd * 100).toFixed(2)}%`);

// ============================================================================
// Step 6: Confusion Matrix Analysis
// ============================================================================

console.log("\nSTEP 6: Confusion Matrix Analysis");
console.log("─".repeat(70));

// Fit the best model again on the training split for the detailed breakdown
const analysisModel = createBest();
analysisModel.fit(XTrainScaled, yTrain);
const yPredFinal = analysisModel.predict(XTestScaled);

const cm = confusionMatrix(yTest, yPredFinal);
const [[tn, fp], [fn, tp]] = cm.toArray() as number[][];

console.log("\nConfusion Matrix:");
console.log("                  Predicted");
console.log("                  Retained  Churned");
console.log(`  Actual Retained    ${String(tn).padStart(4)}     ${String(fp).padStart(4)}`);
console.log(`  Actual Churned     ${String(fn).padStart(4)}     ${String(tp).padStart(4)}`);

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
// Step 7: Feature Importance (random forest)
// ============================================================================

console.log("\nSTEP 7: Feature Importance Analysis");
console.log("─".repeat(70));

// Train Random Forest for feature importance
const rfForImportance = new RandomForestClassifier({
  nEstimators: 100,
});
rfForImportance.fit(XTrainScaled, yTrain);

const featureImportances = rfForImportance.featureImportances.toArray() as number[];
const rankedFeatures = featureNames
  .map((name, index) => ({ name, importance: featureImportances[index] }))
  .sort((left, right) => right.importance - left.importance);

console.log("\n  Random Forest feature importances:");
for (const [index, feature] of rankedFeatures.slice(0, 5).entries()) {
  console.log(`    ${index + 1}. ${feature.name} - importance=${feature.importance.toFixed(4)}`);
}

// ============================================================================
// Step 8: Visualizations
// ============================================================================

console.log("\nSTEP 8: Generating Visualizations");
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
  console.log(`  Saved: ${OUTPUT_DIR}/model-comparison.svg`);
} catch (e) {
  console.log(`  Warning: could not generate model comparison plot: ${e}`);
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
  console.log(`  Saved: ${OUTPUT_DIR}/cv-scores.svg`);
} catch (e) {
  console.log(`  Warning: could not generate CV scores plot: ${e}`);
}

// ============================================================================
// Step 9: Summary and Recommendations
// ============================================================================

console.log(`\n${"═".repeat(70)}`);
console.log("  ANALYSIS COMPLETE - SUMMARY");
console.log("═".repeat(70));

console.log("\nKey Findings:\n");
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

console.log("\nNotes:");
console.log("   • The synthetic data raises churn probability for customers with:");
console.log("     - Low satisfaction scores");
console.log("     - No contract");
console.log("     - High support call frequency");
console.log("   • Compare the models again on real data before choosing one");
console.log("   • Monitor the churn rate after deployment to catch drift");

console.log("\nOutput Files:");
console.log(`   • ${OUTPUT_DIR}/model-comparison.svg`);
console.log(`   • ${OUTPUT_DIR}/cv-scores.svg`);

console.log(`\n${"═".repeat(70)}`);
console.log("  Customer Churn Prediction Complete!");
console.log("═".repeat(70));
