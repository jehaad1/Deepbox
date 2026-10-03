/**
 * Example 38: Feature Engineering & Preprocessing
 *
 * Imputation (SimpleImputer, KNNImputer), feature selection (SelectKBest,
 * VarianceThreshold), text vectorizers (Count, TF-IDF), PolynomialFeatures,
 * SplineTransformer, RobustScaler, PowerTransformer and cross-validation splitters.
 */

import { makeClassification } from "deepbox/datasets";
import { tensor } from "deepbox/ndarray";
import {
  CountVectorizer,
  fClassif,
  KFold,
  KNNImputer,
  PolynomialFeatures,
  PowerTransformer,
  RobustScaler,
  SelectKBest,
  SimpleImputer,
  SplineTransformer,
  StratifiedKFold,
  TfidfVectorizer,
  TimeSeriesSplit,
  VarianceThreshold,
} from "deepbox/preprocess";

console.log("=".repeat(60));
console.log("Example 38: Feature Engineering & Preprocessing");
console.log("=".repeat(60));

// ============================================================================
// Part 1: SimpleImputer (fill missing values)
// ============================================================================
console.log("\nPart 1: SimpleImputer");
console.log("-".repeat(60));

// SimpleImputer replaces missing values (NaN) using a column statistic or a constant
const XMissing = tensor([
  [1, 2, NaN],
  [3, NaN, 6],
  [7, 8, 9],
  [NaN, 5, 3],
  [4, 6, 7],
]);

console.log("Data with missing values:");
console.log(XMissing.toString());

// Strategy: mean (default)
const impMean = new SimpleImputer({ strategy: "mean" });
const XFilledMean = impMean.fitTransform(XMissing);
console.log("\nSimpleImputer (mean):");
console.log(XFilledMean.toString());

// Strategy: median
const impMedian = new SimpleImputer({ strategy: "median" });
const XFilledMedian = impMedian.fitTransform(XMissing);
console.log("SimpleImputer (median):");
console.log(XFilledMedian.toString());

// Strategy: constant
const impConst = new SimpleImputer({ strategy: "constant", fillValue: -1 });
const XFilledConst = impConst.fitTransform(XMissing);
console.log("SimpleImputer (constant=-1):");
console.log(XFilledConst.toString());

// ============================================================================
// Part 2: KNNImputer (k-nearest neighbors imputation)
// ============================================================================
console.log("\nPart 2: KNNImputer");
console.log("-".repeat(60));

// KNNImputer fills a missing value with the mean of that feature over the k nearest rows
const knnImp = new KNNImputer({ nNeighbors: 2 });
const XFilledKNN = knnImp.fitTransform(XMissing);
console.log("KNNImputer (k=2):");
console.log(XFilledKNN.toString());

// ============================================================================
// Part 3: VarianceThreshold (remove low-variance features)
// ============================================================================
console.log("\nPart 3: VarianceThreshold");
console.log("-".repeat(60));

// VarianceThreshold drops features whose variance is below the threshold
const XVar = tensor([
  [0, 2, 0.1],
  [0, 4, 0.2],
  [0, 6, 0.1],
  [0, 8, 0.3],
  [0, 10, 0.2],
]);

console.log("Original data (feature 0 has zero variance):");
console.log(XVar.toString());

const varThresh = new VarianceThreshold({ threshold: 0.01 });
const XVarReduced = varThresh.fitTransform(XVar);
console.log("\nAfter VarianceThreshold (threshold=0.01):");
console.log(XVarReduced.toString());
console.log(`  Removed ${XVar.shape[1]! - XVarReduced.shape[1]!} low-variance feature(s)`);

// ============================================================================
// Part 4: SelectKBest (univariate feature selection)
// ============================================================================
console.log("\nPart 4: SelectKBest");
console.log("-".repeat(60));

// SelectKBest keeps the k features with the highest score. fClassif is the ANOVA F-value.
const [XFeat, yFeat] = makeClassification({
  nSamples: 100,
  nFeatures: 10,
  nInformative: 3,
  nClasses: 2,
  randomState: 42,
});

console.log(`Original features: ${XFeat.shape[1]}`);

const selector = new SelectKBest({ scoreFunc: fClassif, k: 5 });
const XSelected = selector.fitTransform(XFeat, yFeat);
console.log(`After SelectKBest (k=5): ${XSelected.shape[1]} features`);
console.log(`  Selected feature scores: ${selector.scores.toString()}`);

// ============================================================================
// Part 5: Text Vectorization
// ============================================================================
console.log("\nPart 5: Text Vectorization");
console.log("-".repeat(60));

// CountVectorizer turns documents into word-count vectors
const documents = ["the cat sat on the mat", "the dog sat on the log", "cats and dogs are friends"];

console.log("Documents:");
for (const doc of documents) {
  console.log(`  "${doc}"`);
}

const countVec = new CountVectorizer();
const XCount = countVec.fitTransformText(documents);
console.log(`\nCountVectorizer output shape: [${XCount.shape.join(", ")}]`);
console.log(`  Vocabulary size: ${countVec.vocabulary.size}`);

// TfidfVectorizer weights each count by how rare the word is across documents
const tfidfVec = new TfidfVectorizer();
const XTfidf = tfidfVec.fitTransformText(documents);
console.log(`\nTfidfVectorizer output shape: [${XTfidf.shape.join(", ")}]`);
console.log("  Words that appear in few documents get higher weights");

// ============================================================================
// Part 6: PolynomialFeatures
// ============================================================================
console.log("\nPart 6: PolynomialFeatures");
console.log("-".repeat(60));

// PolynomialFeatures adds powers and products of the input features
const XPoly = tensor([
  [1, 2],
  [3, 4],
  [5, 6],
]);

console.log("Original features:");
console.log(XPoly.toString());

const poly = new PolynomialFeatures({ degree: 2, includeBias: false });
const XPolyTransformed = poly.fitTransform(XPoly);
console.log("\nPolynomialFeatures (degree=2, no bias):");
console.log(XPolyTransformed.toString());
console.log(
  `  Original: ${XPoly.shape[1]} features, expanded: ${XPolyTransformed.shape[1]} features`
);

// ============================================================================
// Part 7: SplineTransformer
// ============================================================================
console.log("\nPart 7: SplineTransformer");
console.log("-".repeat(60));

// SplineTransformer expands one feature into B-spline basis functions
const XSpline = tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]);

console.log("Original 1D feature:");
console.log(XSpline.toString());

const spline = new SplineTransformer({ nKnots: 4, degree: 3 });
const XSplineTransformed = spline.fitTransform(XSpline);
console.log("\nSplineTransformer (4 knots, degree 3):");
console.log(`  Output shape: [${XSplineTransformed.shape.join(", ")}]`);
console.log(`  1 feature becomes ${XSplineTransformed.shape[1]} spline basis functions`);

// ============================================================================
// Part 8: RobustScaler & PowerTransformer
// ============================================================================
console.log("\nPart 8: Advanced Scalers");
console.log("-".repeat(60));

const XSkewed = tensor([
  [1, 100],
  [2, 200],
  [3, 300],
  [100, 400], // outlier in the first feature
  [4, 500],
]);

// RobustScaler centers on the median and scales by the interquartile range, so one outlier barely changes it
const robust = new RobustScaler();
const XRobust = robust.fitTransform(XSkewed);
console.log("RobustScaler (median and IQR):");
console.log(XRobust.toString());

// PowerTransformer applies a power transform that makes each feature more Gaussian.
// standardize defaults to false in Deepbox. scikit-learn defaults to true, so pass
// { standardize: true } to get zero mean and unit variance as scikit-learn does.
const XPositive = tensor([
  [1, 10],
  [2, 20],
  [3, 30],
  [4, 40],
  [5, 50],
]);

const power = new PowerTransformer({ method: "yeo-johnson" });
const XPower = power.fitTransform(XPositive);
console.log("\nPowerTransformer (Yeo-Johnson, standardize=false):");
console.log(XPower.toString());

const powerStd = new PowerTransformer({ method: "yeo-johnson", standardize: true });
console.log("PowerTransformer (Yeo-Johnson, standardize=true):");
console.log(powerStd.fitTransform(XPositive).toString());

// ============================================================================
// Part 9: Cross-Validation Splitters
// ============================================================================
console.log("\nPart 9: Cross-Validation Splitters");
console.log("-".repeat(60));

const [XSplit, ySplit] = makeClassification({
  nSamples: 20,
  nFeatures: 2,
  nInformative: 2,
  nRedundant: 0,
  nClasses: 2,
  randomState: 42,
});

// KFold: k folds, each used once as the test set
const kfold = new KFold({ nSplits: 5, shuffle: true, randomState: 42 });
console.log("KFold (5 splits, shuffled):");
let foldNum = 1;
for (const { trainIndex, testIndex } of kfold.split(XSplit)) {
  console.log(
    `  Fold ${foldNum}: train=${trainIndex.length} samples, test=${testIndex.length} samples`
  );
  foldNum++;
}

// StratifiedKFold: keeps the class proportions in each fold
const stratKfold = new StratifiedKFold({ nSplits: 5 });
console.log("\nStratifiedKFold (5 splits, preserves class balance):");
foldNum = 1;
for (const { trainIndex, testIndex } of stratKfold.split(XSplit, ySplit)) {
  console.log(`  Fold ${foldNum}: train=${trainIndex.length}, test=${testIndex.length}`);
  foldNum++;
}

// TimeSeriesSplit: each test fold comes after its training rows in time
const tsSplit = new TimeSeriesSplit({ nSplits: 4 });
console.log("\nTimeSeriesSplit (4 splits, expanding window):");
foldNum = 1;
for (const { trainIndex, testIndex } of tsSplit.split(XSplit)) {
  console.log(`  Fold ${foldNum}: train=${trainIndex.length}, test=${testIndex.length}`);
  foldNum++;
}

// ============================================================================
// Summary
// ============================================================================
console.log("\nKey Takeaways");
console.log("-".repeat(60));
console.log("• SimpleImputer: fill NaN with the mean, median, most_frequent value or a constant");
console.log("• KNNImputer: fill NaN from the nearest rows");
console.log("• VarianceThreshold: drop near-constant features");
console.log("• SelectKBest: keep the k features with the best test score");
console.log("• CountVectorizer: word counts per document");
console.log("• TfidfVectorizer: word counts weighted by rarity");
console.log("• SplineTransformer: nonlinear feature expansion with B-splines");
console.log("• RobustScaler: scaling that outliers barely affect");
console.log("• PowerTransformer: more Gaussian features (standardize is false by default)");
console.log("• KFold, StratifiedKFold, TimeSeriesSplit: splitters for different kinds of data");

console.log("\nFeature Engineering Example Complete!");
console.log("=".repeat(60));
