/**
 * Example 37: Model Selection & Pipeline
 *
 * Pipeline, crossValidate, GridSearchCV and RandomizedSearchCV: chain a scaler
 * and a model, score models with k-fold cross-validation, and tune
 * hyperparameters, including those of a pipeline step.
 */

import { makeClassification } from "deepbox/datasets";
import { accuracy } from "deepbox/metrics";
import {
  crossValidate,
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
// Part 1: Pipeline (chain Preprocessing + Model)
// ============================================================================
console.log("\nPart 1: Pipeline");
console.log("-".repeat(60));

// A Pipeline chains transformers and a final estimator into one object.
// Steps are [name, estimator] pairs. Every step except the last must be a transformer.
const pipe = new Pipeline([
  ["scaler", new StandardScaler()],
  ["classifier", new LogisticRegression({ maxIter: 200 })],
]);

// fit() calls fitTransform on each intermediate step, then fit on the final step
pipe.fit(XTrain, yTrain);

// predict() calls transform on each intermediate step, then predict on the final step
const pipePred = pipe.predict(XTest);
const pipeAcc = accuracy(yTest, pipePred);

console.log("Pipeline (StandardScaler, LogisticRegression):");
console.log(`  Accuracy: ${(Number(pipeAcc) * 100).toFixed(2)}%`);

// The same scaler in front of a different model
const pipe2 = new Pipeline([
  ["scaler", new StandardScaler()],
  ["classifier", new KNeighborsClassifier({ nNeighbors: 5 })],
]);
pipe2.fit(XTrain, yTrain);

const pipe2Pred = pipe2.predict(XTest);
const pipe2Acc = accuracy(yTest, pipe2Pred);

console.log("Pipeline (StandardScaler, KNN):");
console.log(`  Accuracy: ${(Number(pipe2Acc) * 100).toFixed(2)}%`);

// ============================================================================
// Part 2: Cross-Validation
// ============================================================================
console.log("\nPart 2: Cross-Validation");
console.log("-".repeat(60));

// crossValidate fits a fresh copy of the model on each fold and scores it on the held-out part
const lr = new LogisticRegression({ maxIter: 200 });
const cvResult = crossValidate(lr, XTrain, yTrain, { cv: 5 });

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
  const cv = crossValidate(model, XTrain, yTrain, { cv: 5 });
  const scores = cv.testScores["score"] ?? [];
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  const std = Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / scores.length);
  console.log(`  ${name.padEnd(22)} Mean: ${mean.toFixed(4)} ± ${std.toFixed(4)}`);
}

// A Pipeline can be cross-validated directly. The scaler is refit on each training fold,
// so no statistics from the held-out fold leak into the scaling.
const pipeCv = crossValidate(pipe, XTrain, yTrain, { cv: 5 });
const pipeScores = pipeCv.testScores["score"] ?? [];
const pipeMean = pipeScores.reduce((a, b) => a + b, 0) / pipeScores.length;
console.log(`  ${"Scaler + LogReg".padEnd(22)} Mean: ${pipeMean.toFixed(4)}`);

// ============================================================================
// Part 3: GridSearchCV (exhaustive hyperparameter search)
// ============================================================================
console.log("\nPart 3: GridSearchCV");
console.log("-".repeat(60));

// GridSearchCV scores every combination in the grid with k-fold cross-validation
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
  console.log(`    ${params.padEnd(20)} mean: ${mean.toFixed(4)}, std: ${std.toFixed(4)}`);
}

// ============================================================================
// Part 4: RandomizedSearchCV (random hyperparameter sampling)
// ============================================================================
console.log("\nPart 4: RandomizedSearchCV");
console.log("-".repeat(60));

// RandomizedSearchCV tries nIter random combinations from the grid, which costs less than the full grid
const rfRandomSearch = new RandomizedSearchCV(
  new RandomForestClassifier({ randomState: 42 }),
  {
    nEstimators: [10, 20, 50, 100],
    maxDepth: [3, 5, 10, 15],
  },
  { cv: 3, nIter: 8, randomState: 42 }
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
console.log("\nPart 5: GridSearchCV with a Pipeline");
console.log("-".repeat(60));

// A parameter of a pipeline step is addressed as stepName__parameterName.
// The scaler is refit inside every fold, so the search is free of scaling leakage.
const pipeForSearch = new Pipeline([
  ["scaler", new StandardScaler()],
  ["classifier", new KNeighborsClassifier()],
]);

const pipeGrid = new GridSearchCV(
  pipeForSearch,
  {
    classifier__nNeighbors: [3, 5, 7, 9],
  },
  { cv: 5 }
);

pipeGrid.fit(XTrain, yTrain);

console.log("GridSearchCV over a Pipeline (scaler + KNN):");
console.log(`  Best params: ${JSON.stringify(pipeGrid.bestParams)}`);
console.log(`  Best CV score: ${pipeGrid.bestScore.toFixed(4)}`);

if (pipeGrid.bestEstimator) {
  const best = pipeGrid.bestEstimator as Pipeline;
  const bestPred = best.predict(XTest);
  const bestAcc = accuracy(yTest, bestPred);
  console.log(`  Test accuracy: ${(Number(bestAcc) * 100).toFixed(2)}%`);
}

// ============================================================================
// Summary
// ============================================================================
console.log("\nKey Takeaways");
console.log("-".repeat(60));
console.log("• Pipeline: transformers and an estimator behind one fit/predict");
console.log("• crossValidate: k-fold scores for a model or a whole pipeline");
console.log("• GridSearchCV: tries every combination in the grid");
console.log("• RandomizedSearchCV: tries nIter random combinations, cheaper for large grids");
console.log("• Pipeline parameters are tuned as stepName__parameterName");
console.log("• Choose models by CV score, and keep the test set for a final check");
console.log("• Scale features before distance-based models (KNN, SVM)");

console.log("\nModel Selection & Pipeline Example Complete!");
console.log("=".repeat(60));
