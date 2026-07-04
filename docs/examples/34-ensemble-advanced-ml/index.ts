/**
 * Example 34: Ensemble & Advanced ML Models
 *
 * New in v1.0.0: AdaBoost, Bagging, Voting, Stacking, ExtraTrees,
 * Gaussian Processes, Discriminant Analysis, and Semi-supervised Learning.
 */

import { makeClassification, makeRegression } from "deepbox/datasets";
import { accuracy, f1Score, r2Score } from "deepbox/metrics";
import {
  AdaBoostClassifier,
  BaggingClassifier,
  DecisionTreeClassifier,
  ExtraTreesClassifier,
  GaussianProcessRegressor,
  KNeighborsClassifier,
  LinearDiscriminantAnalysis,
  LogisticRegression,
  RandomForestClassifier,
  StackingClassifier,
  VotingClassifier,
} from "deepbox/ml";
import { trainTestSplit } from "deepbox/preprocess";

console.log("=".repeat(60));
console.log("Example 34: Ensemble & Advanced ML Models");
console.log("=".repeat(60));

// ============================================================================
// Generate datasets for classification and regression
// ============================================================================

const [XClass, yClass] = makeClassification({
  nSamples: 200,
  nFeatures: 10,
  nInformative: 6,
  nClasses: 2,
  randomState: 42,
});

const [XTrain, XTest, yTrain, yTest] = trainTestSplit(XClass, yClass, {
  testSize: 0.25,
  randomState: 42,
});

const [XReg, yReg] = makeRegression({
  nSamples: 100,
  nFeatures: 5,
  noise: 10,
  randomState: 42,
});

const [XTrainReg, XTestReg, yTrainReg, yTestReg] = trainTestSplit(XReg, yReg, {
  testSize: 0.25,
  randomState: 42,
});

// ============================================================================
// Part 1: AdaBoost Classifier
// ============================================================================
console.log("\n🚀 Part 1: AdaBoost Classifier");
console.log("-".repeat(60));

// AdaBoost builds an ensemble of weak learners, focusing on misclassified samples
const adaboost = new AdaBoostClassifier({
  nEstimators: 50,
  learningRate: 1.0,
});
adaboost.fit(XTrain, yTrain);

const adaPred = adaboost.predict(XTest);
const adaAcc = accuracy(yTest, adaPred);
const adaF1 = f1Score(yTest, adaPred);

console.log("AdaBoost Classifier (50 estimators):");
console.log(`  Accuracy: ${(Number(adaAcc) * 100).toFixed(2)}%`);
console.log(`  F1 Score: ${Number(adaF1).toFixed(4)}`);

// ============================================================================
// Part 2: Bagging Classifier
// ============================================================================
console.log("\n🎒 Part 2: Bagging Classifier");
console.log("-".repeat(60));

// Bagging trains multiple models on random subsets of the data (bootstrap)
const bagging = new BaggingClassifier({
  nEstimators: 20,
  maxSamples: 0.8,
  maxFeatures: 0.8,
  randomState: 42,
});
bagging.fit(XTrain, yTrain);

const bagPred = bagging.predict(XTest);
const bagAcc = accuracy(yTest, bagPred);
const bagF1 = f1Score(yTest, bagPred);

console.log("Bagging Classifier (20 estimators, 80% samples/features):");
console.log(`  Accuracy: ${(Number(bagAcc) * 100).toFixed(2)}%`);
console.log(`  F1 Score: ${Number(bagF1).toFixed(4)}`);

// ============================================================================
// Part 3: Voting Classifier
// ============================================================================
console.log("\n🗳️  Part 3: Voting Classifier");
console.log("-".repeat(60));

// Voting combines multiple diverse classifiers for better predictions
const voting = new VotingClassifier({
  estimators: [
    new LogisticRegression({ maxIter: 200 }),
    new RandomForestClassifier({ nEstimators: 20, randomState: 42 }),
    new KNeighborsClassifier({ nNeighbors: 5 }),
  ],
  voting: "hard",
});
voting.fit(XTrain, yTrain);

const votePred = voting.predict(XTest);
const voteAcc = accuracy(yTest, votePred);
const voteF1 = f1Score(yTest, votePred);

console.log("Voting Classifier (LogReg + RandomForest + KNN, hard voting):");
console.log(`  Accuracy: ${(Number(voteAcc) * 100).toFixed(2)}%`);
console.log(`  F1 Score: ${Number(voteF1).toFixed(4)}`);

// ============================================================================
// Part 4: Stacking Classifier
// ============================================================================
console.log("\n📚 Part 4: Stacking Classifier");
console.log("-".repeat(60));

// Stacking uses a meta-learner to combine base estimator predictions
const stacking = new StackingClassifier({
  estimators: [
    new DecisionTreeClassifier({ maxDepth: 5 }),
    new KNeighborsClassifier({ nNeighbors: 5 }),
  ],
  finalEstimator: new LogisticRegression({ maxIter: 200 }),
});
stacking.fit(XTrain, yTrain);

const stackPred = stacking.predict(XTest);
const stackAcc = accuracy(yTest, stackPred);
const stackF1 = f1Score(yTest, stackPred);

console.log("Stacking Classifier (DecisionTree + KNN → LogReg meta-learner):");
console.log(`  Accuracy: ${(Number(stackAcc) * 100).toFixed(2)}%`);
console.log(`  F1 Score: ${Number(stackF1).toFixed(4)}`);

// ============================================================================
// Part 5: ExtraTrees Classifier
// ============================================================================
console.log("\n🌳 Part 5: ExtraTrees Classifier");
console.log("-".repeat(60));

// ExtraTrees: like Random Forest but with random split thresholds (more randomized)
const extraTrees = new ExtraTreesClassifier({
  nEstimators: 50,
  maxDepth: 10,
  randomState: 42,
});
extraTrees.fit(XTrain, yTrain);

const etPred = extraTrees.predict(XTest);
const etAcc = accuracy(yTest, etPred);
const etF1 = f1Score(yTest, etPred);

console.log("ExtraTrees Classifier (50 estimators, maxDepth=10):");
console.log(`  Accuracy: ${(Number(etAcc) * 100).toFixed(2)}%`);
console.log(`  F1 Score: ${Number(etF1).toFixed(4)}`);

// Feature importances
const importances = extraTrees.featureImportances;
console.log("  Feature importances:", importances.toString());

// ============================================================================
// Part 6: Gaussian Process Regressor
// ============================================================================
console.log("\n📐 Part 6: Gaussian Process Regressor");
console.log("-".repeat(60));

// Gaussian Processes provide probabilistic predictions with uncertainty estimates
const gpr = new GaussianProcessRegressor({
  alpha: 1e-2,
});
gpr.fit(XTrainReg, yTrainReg);

const gprPred = gpr.predict(XTestReg);
const gprR2 = r2Score(yTestReg, gprPred);

console.log("Gaussian Process Regressor:");
console.log(`  R² Score: ${Number(gprR2).toFixed(4)}`);
console.log(`  Predictions shape: ${gprPred.shape}`);

// ============================================================================
// Part 7: Linear Discriminant Analysis
// ============================================================================
console.log("\n📏 Part 7: Linear Discriminant Analysis");
console.log("-".repeat(60));

// LDA finds linear combinations of features that best separate classes
const lda = new LinearDiscriminantAnalysis();
lda.fit(XTrain, yTrain);

const ldaPred = lda.predict(XTest);
const ldaAcc = accuracy(yTest, ldaPred);

console.log("Linear Discriminant Analysis:");
console.log(`  Accuracy: ${(Number(ldaAcc) * 100).toFixed(2)}%`);

// LDA can also transform data for dimensionality reduction
const ldaTransformed = lda.transform(XTest);
console.log(`  Transformed shape: ${XTest.shape} → ${ldaTransformed.shape}`);

// ============================================================================
// Part 8: Model Comparison Summary
// ============================================================================
console.log("\n📊 Part 8: Model Comparison");
console.log("-".repeat(60));

const results = [
  { name: "AdaBoost", acc: Number(adaAcc), f1: Number(adaF1) },
  { name: "Bagging", acc: Number(bagAcc), f1: Number(bagF1) },
  { name: "Voting", acc: Number(voteAcc), f1: Number(voteF1) },
  { name: "Stacking", acc: Number(stackAcc), f1: Number(stackF1) },
  { name: "ExtraTrees", acc: Number(etAcc), f1: Number(etF1) },
  { name: "LDA", acc: Number(ldaAcc), f1: 0 },
];

console.log("\nClassification Model Comparison:");
console.log("┌─────────────┬──────────┬──────────┐");
console.log("│ Model       │ Accuracy │ F1 Score │");
console.log("├─────────────┼──────────┼──────────┤");
for (const r of results) {
  const name = r.name.padEnd(11);
  const acc = `${(r.acc * 100).toFixed(2).padStart(6)}%`;
  const f1 = r.f1 > 0 ? r.f1.toFixed(4).padStart(8) : "     N/A";
  console.log(`│ ${name} │ ${acc}  │ ${f1} │`);
}
console.log("└─────────────┴──────────┴──────────┘");

// ============================================================================
// Summary
// ============================================================================
console.log("\n💡 Key Takeaways");
console.log("-".repeat(60));
console.log("• AdaBoost: boosting weak learners sequentially, focuses on hard examples");
console.log("• Bagging: reduces variance by training on random data subsets");
console.log("• Voting: combines diverse models via majority vote or probability averaging");
console.log("• Stacking: uses a meta-learner to combine base model predictions");
console.log("• ExtraTrees: extremely randomized trees for faster training");
console.log("• Gaussian Processes: probabilistic regression with uncertainty estimates");
console.log("• LDA: simultaneous classification and dimensionality reduction");
console.log("• All models follow the unified fit/predict/score API");

console.log("\n✅ Ensemble & Advanced ML Example Complete!");
console.log("=".repeat(60));
