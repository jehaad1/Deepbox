/**
 * Example 37: Model Selection & Pipeline
 *
 * New in v1.0.0: GridSearchCV, RandomizedSearchCV, Pipeline,
 * ColumnTransformer, and cross-validation utilities.
 */

import { makeClassification } from "deepbox/datasets";
import { accuracy } from "deepbox/metrics";
import {
  cross_validate,
  GridSearchCV,
  KNeighborsClassifier,
  LogisticRegression,
  Pipeline,
  RandomForestClassifier,
  RandomizedSearchCV,
  SVC,
} from "deepbox/ml";
import { StandardScaler, trainTestSplit } from "deepbox/preprocess";

console.log("=".repeat(60));
console.log("Example 37: Model Selection & Pipeline");
console.log("=".repeat(60));

// ============================================================================
// Generate dataset
// ============================================================================

const [X, y] = makeClassification({
  nSamples: 200,
  nFeatures: 8,
  nInformative: 5,
  nClasses: 2,
  randomState: 42,
});

const [XTrain, XTest, yTrain, yTest] = trainTestSplit(X, y, {
  testSize: 0.25,
  randomState: 42,
});

console.log(`Dataset: ${X.shape[0]} samples, ${X.shape[1]} features`);
console.log(`Train: ${XTrain.shape[0]}, Test: ${XTest.shape[0]}`);

// ============================================================================
// Part 1: Pipeline — Chain Preprocessing + Model
// ============================================================================
console.log("\n🔗 Part 1: Pipeline");
console.log("-".repeat(60));

// A Pipeline chains transformers and a final estimator into a single object
// Steps: [name, estimator] tuples — intermediate steps must be transformers
const pipe = new Pipeline([
  ["scaler", new StandardScaler()],
  ["classifier", new LogisticRegression({ maxIter: 200 })],
]);

// fit() applies fitTransform to intermediate steps, then fit to final step
pipe.fit(XTrain, yTrain);

// predict() applies transform to intermediate steps, then predict on final step
const pipePred = pipe.predict(XTest);
const pipeAcc = accuracy(yTest, pipePred);

console.log("Pipeline (StandardScaler → LogisticRegression):");
console.log(`  Accuracy: ${(Number(pipeAcc) * 100).toFixed(2)}%`);

// Try a different pipeline with StandardScaler + KNN
const pipe2 = new Pipeline([
  ["scaler", new StandardScaler()],
  ["classifier", new KNeighborsClassifier({ nNeighbors: 5 })],
]);
pipe2.fit(XTrain, yTrain);

const pipe2Pred = pipe2.predict(XTest);
const pipe2Acc = accuracy(yTest, pipe2Pred);

console.log("Pipeline (StandardScaler → KNN):");
console.log(`  Accuracy: ${(Number(pipe2Acc) * 100).toFixed(2)}%`);

// ============================================================================
// Part 2: Cross-Validation
// ============================================================================
console.log("\n📊 Part 2: Cross-Validation");
console.log("-".repeat(60));

// cross_validate evaluates a model using k-fold cross-validation
const lr = new LogisticRegression({ maxIter: 200 });
const cvResult = cross_validate(lr, XTrain, yTrain, { cv: 5 });

console.log("LogisticRegression 5-fold Cross-Validation:");
console.log(
  `  Test scores: [${cvResult.testScores["score"]?.map((s) => s.toFixed(4)).join(", ")}]`
);
const meanScore =
  (cvResult.testScores["score"]?.reduce((a, b) => a + b, 0) ?? 0) /
  (cvResult.testScores["score"]?.length ?? 1);
console.log(`  Mean score: ${meanScore.toFixed(4)}`);

// Cross-validate different models for comparison
const models = [
  {
    name: "LogisticRegression",
    model: new LogisticRegression({ maxIter: 200 }),
  },
  { name: "KNN (k=5)", model: new KNeighborsClassifier({ nNeighbors: 5 }) },
  {
    name: "RandomForest (20)",
    model: new RandomForestClassifier({ nEstimators: 20, randomState: 42 }),
  },
  { name: "SVC (RBF)", model: new SVC({ kernel: "rbf", gamma: "scale" }) },
];

console.log("\nModel Comparison (5-fold CV):");
for (const { name, model } of models) {
  const cv = cross_validate(model, XTrain, yTrain, { cv: 5 });
  const scores = cv.testScores["score"] ?? [];
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const std = Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / scores.length);
  console.log(`  ${name.padEnd(22)} — Mean: ${mean.toFixed(4)} ± ${std.toFixed(4)}`);
}

// ============================================================================
// Part 3: GridSearchCV — Exhaustive Hyperparameter Search
// ============================================================================
console.log("\n🔍 Part 3: GridSearchCV");
console.log("-".repeat(60));

// GridSearchCV searches over all combinations of hyperparameters
const knnGrid = new GridSearchCV(
  new KNeighborsClassifier(),
  {
    nNeighbors: [3, 5, 7, 9, 11],
  },
  { cv: 5 }
);

knnGrid.fit(XTrain, yTrain);

console.log("GridSearchCV for KNeighborsClassifier:");
console.log(`  Best params: nNeighbors=${knnGrid.bestParams["nNeighbors"]}`);
console.log(`  Best CV score: ${knnGrid.bestScore.toFixed(4)}`);

// Evaluate best model on test set
if (knnGrid.bestEstimator) {
  const best = knnGrid.bestEstimator as KNeighborsClassifier;
  const bestPred = best.predict(XTest);
  const bestAcc = accuracy(yTest, bestPred);
  console.log(`  Test accuracy (best model): ${(Number(bestAcc) * 100).toFixed(2)}%`);
}

// Show all CV results
console.log("\n  All GridSearch Results:");
for (const result of knnGrid.cvResults) {
  const params = JSON.stringify(result.params);
  const scores = result.scores;
  const mean = result.meanScore;
  const std = Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / scores.length);
  console.log(`    ${params.padEnd(20)} → mean: ${mean.toFixed(4)}, std: ${std.toFixed(4)}`);
}

// ============================================================================
// Part 4: RandomizedSearchCV — Random Hyperparameter Sampling
// ============================================================================
console.log("\n🎲 Part 4: RandomizedSearchCV");
console.log("-".repeat(60));

// RandomizedSearchCV samples random combinations — faster than exhaustive search
const rfRandomSearch = new RandomizedSearchCV(
  new RandomForestClassifier({ randomState: 42 }),
  {
    nEstimators: [10, 20, 50, 100],
    maxDepth: [3, 5, 10, 15],
  },
  { cv: 3, nIter: 8 }
);

rfRandomSearch.fit(XTrain, yTrain);

console.log("RandomizedSearchCV for RandomForestClassifier (8 iterations):");
console.log(`  Best params: ${JSON.stringify(rfRandomSearch.bestParams)}`);
console.log(`  Best CV score: ${rfRandomSearch.bestScore.toFixed(4)}`);

if (rfRandomSearch.bestEstimator) {
  const best = rfRandomSearch.bestEstimator as RandomForestClassifier;
  const bestPred = best.predict(XTest);
  const bestAcc = accuracy(yTest, bestPred);
  console.log(`  Test accuracy (best model): ${(Number(bestAcc) * 100).toFixed(2)}%`);
}

// ============================================================================
// Part 5: GridSearchCV with Pipeline
// ============================================================================
console.log("\n🔗🔍 Part 5: Combining Pipeline with GridSearchCV");
console.log("-".repeat(60));

// You can use GridSearchCV on individual models within a pipeline workflow
// First scale, then search over different KNN params
const scalerForSearch = new StandardScaler();
const XTrainScaled = scalerForSearch.fitTransform(XTrain);
const XTestScaled = scalerForSearch.transform(XTest);

const knnGridScaled = new GridSearchCV(
  new KNeighborsClassifier(),
  {
    nNeighbors: [3, 5, 7, 9],
  },
  { cv: 5 }
);

knnGridScaled.fit(XTrainScaled, yTrain);

console.log("GridSearchCV on scaled data:");
console.log(`  Best nNeighbors: ${knnGridScaled.bestParams["nNeighbors"]}`);
console.log(`  Best CV score: ${knnGridScaled.bestScore.toFixed(4)}`);

if (knnGridScaled.bestEstimator) {
  const best = knnGridScaled.bestEstimator as KNeighborsClassifier;
  const bestPred = best.predict(XTestScaled);
  const bestAcc = accuracy(yTest, bestPred);
  console.log(`  Test accuracy: ${(Number(bestAcc) * 100).toFixed(2)}%`);
}

// ============================================================================
// Summary
// ============================================================================
console.log("\n💡 Key Takeaways");
console.log("-".repeat(60));
console.log("• Pipeline: chain transformers + estimator into a single fit/predict call");
console.log("• cross_validate: evaluate models with k-fold CV for robust estimates");
console.log("• GridSearchCV: exhaustive search over all hyperparameter combinations");
console.log("• RandomizedSearchCV: random sampling — faster for large param spaces");
console.log("• Always use CV scores (not single train/test) for model selection");
console.log("• Scale features before using distance-based models (KNN, SVM)");

console.log("\n✅ Model Selection & Pipeline Example Complete!");
console.log("=".repeat(60));
