/**
 * Example 34: Ensemble & Advanced ML Models
 *
 * AdaBoost, Bagging, Voting, Stacking, ExtraTrees, a Gaussian Process regressor
 * and Linear Discriminant Analysis, compared on one classification dataset.
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
  noise: 0.5,
  randomState: 42,
});

const [XTrainReg, XTestReg, yTrainReg, yTestReg] = trainTestSplit(XReg, yReg, {
  testSize: 0.25,
  randomState: 42,
});

// ============================================================================
// Part 1: AdaBoost Classifier
// ============================================================================
console.log("\nPart 1: AdaBoost Classifier");
console.log("-".repeat(60));

// AdaBoost fits weak learners one after another and gives more weight to the samples the earlier ones got wrong
const adaboost = new AdaBoostClassifier({
  nEstimators: 50,
  learningRate: 1.0,
  randomState: 42,
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
console.log("\nPart 2: Bagging Classifier");
console.log("-".repeat(60));

// Bagging trains each model on a random bootstrap subset of the samples and features
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
console.log("\nPart 3: Voting Classifier");
console.log("-".repeat(60));

// Voting combines different classifiers: majority vote (hard) or averaged probabilities (soft)
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
console.log("\nPart 4: Stacking Classifier");
console.log("-".repeat(60));

// Stacking trains a final estimator on the predictions of the base estimators
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

console.log("Stacking Classifier (DecisionTree + KNN, LogReg meta-learner):");
console.log(`  Accuracy: ${(Number(stackAcc) * 100).toFixed(2)}%`);
console.log(`  F1 Score: ${Number(stackF1).toFixed(4)}`);

// ============================================================================
// Part 5: ExtraTrees Classifier
// ============================================================================
console.log("\nPart 5: ExtraTrees Classifier");
console.log("-".repeat(60));

// ExtraTrees is like a random forest, but picks split thresholds at random
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

// Class weights and sample weights change how much each sample counts in a split.
// classWeight: "balanced" weights classes inversely to their frequency.
const weighted = new ExtraTreesClassifier({
  nEstimators: 50,
  maxDepth: 10,
  classWeight: "balanced",
  randomState: 42,
});
weighted.fit(XTrain, yTrain);
console.log(
  `  With classWeight "balanced": accuracy ${(Number(accuracy(yTest, weighted.predict(XTest))) * 100).toFixed(2)}%`
);

// clone() returns an unfitted copy with the same hyperparameters
const fresh = extraTrees.clone();
fresh.fit(XTrain, yTrain);
console.log(
  `  clone() refit: accuracy ${(Number(accuracy(yTest, fresh.predict(XTest))) * 100).toFixed(2)}%`
);

// ============================================================================
// Part 6: Gaussian Process Regressor
// ============================================================================
console.log("\nPart 6: Gaussian Process Regressor");
console.log("-".repeat(60));

// A Gaussian process regressor predicts a mean for each sample and can also report its uncertainty
// alpha is the noise variance, lengthScale the RBF kernel width. The kernel has a zero
// prior mean, so normalizeY centers and scales the targets before fitting.
const gpr = new GaussianProcessRegressor({
  alpha: 0.1,
  lengthScale: 10,
  normalizeY: true,
});
gpr.fit(XTrainReg, yTrainReg);

const gprPred = gpr.predict(XTestReg);
const gprR2 = r2Score(yTestReg, gprPred);

console.log("Gaussian Process Regressor:");
console.log(`  R² Score: ${Number(gprR2).toFixed(4)}`);

// predictWithStd also returns the standard deviation of each prediction
const { std } = gpr.predictWithStd(XTestReg);
console.log(`  Mean predictive std: ${Number(std.mean().item()).toFixed(4)}`);
console.log(`  Predictions shape: [${gprPred.shape.join(", ")}]`);

// ============================================================================
// Part 7: Linear Discriminant Analysis
// ============================================================================
console.log("\nPart 7: Linear Discriminant Analysis");
console.log("-".repeat(60));

// LDA finds linear combinations of the features that best separate the classes
const lda = new LinearDiscriminantAnalysis();
lda.fit(XTrain, yTrain);

const ldaPred = lda.predict(XTest);
const ldaAcc = accuracy(yTest, ldaPred);
const ldaF1 = f1Score(yTest, ldaPred);

console.log("Linear Discriminant Analysis:");
console.log(`  Accuracy: ${(Number(ldaAcc) * 100).toFixed(2)}%`);
console.log(`  F1 Score: ${Number(ldaF1).toFixed(4)}`);

// LDA can also project the data onto those combinations, which reduces the dimension
const ldaTransformed = lda.transform(XTest);
console.log(
  `  Transformed shape: [${XTest.shape.join(", ")}] to [${ldaTransformed.shape.join(", ")}]`
);

// ============================================================================
// Part 8: Model Comparison Summary
// ============================================================================
console.log("\nPart 8: Model Comparison");
console.log("-".repeat(60));

const results = [
  { name: "AdaBoost", acc: Number(adaAcc), f1: Number(adaF1) },
  { name: "Bagging", acc: Number(bagAcc), f1: Number(bagF1) },
  { name: "Voting", acc: Number(voteAcc), f1: Number(voteF1) },
  { name: "Stacking", acc: Number(stackAcc), f1: Number(stackF1) },
  { name: "ExtraTrees", acc: Number(etAcc), f1: Number(etF1) },
  { name: "LDA", acc: Number(ldaAcc), f1: Number(ldaF1) },
];

console.log("\nClassification Model Comparison:");
console.log("┌─────────────┬──────────┬──────────┐");
console.log("│ Model       │ Accuracy │ F1 Score │");
console.log("├─────────────┼──────────┼──────────┤");
for (const r of results) {
  const name = r.name.padEnd(11);
  const acc = `${(r.acc * 100).toFixed(2).padStart(6)}%`;
  const f1 = r.f1.toFixed(4).padStart(8);
  console.log(`│ ${name} │ ${acc}  │ ${f1} │`);
}
console.log("└─────────────┴──────────┴──────────┘");

// ============================================================================
// Summary
// ============================================================================
console.log("\nKey Takeaways");
console.log("-".repeat(60));
console.log("• AdaBoost: weak learners in sequence, each focused on the previous mistakes");
console.log("• Bagging: averages models trained on bootstrap subsets, which lowers variance");
console.log("• Voting: majority vote or averaged probabilities of different models");
console.log("• Stacking: a final estimator learns from the base models' predictions");
console.log("• ExtraTrees: random split thresholds, so each tree trains faster");
console.log("• Gaussian process: regression with a built-in uncertainty estimate");
console.log("• LDA: a classifier that can also reduce the dimension");
console.log("• Every model uses the same fit/predict API, and clone() gives an unfitted copy");

console.log("\nEnsemble & Advanced ML Example Complete!");
console.log("=".repeat(60));
