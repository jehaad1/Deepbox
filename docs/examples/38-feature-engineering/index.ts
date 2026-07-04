/**
 * Example 38: Feature Engineering & Preprocessing
 *
 * New in v1.0.0: Imputation (SimpleImputer, KNNImputer), feature selection
 * (SelectKBest, VarianceThreshold), text vectorizers (TF-IDF, Count),
 * SplineTransformer, advanced scalers, and cross-validation splitters.
 */

import { makeClassification } from "deepbox/datasets";
import { tensor } from "deepbox/ndarray";
import {
  CountVectorizer,
  f_classif,
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
// Part 1: SimpleImputer — Fill Missing Values
// ============================================================================
console.log("\n🔧 Part 1: SimpleImputer");
console.log("-".repeat(60));

// SimpleImputer fills missing values (NaN) with a chosen strategy
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
// Part 2: KNNImputer — k-Nearest Neighbors Imputation
// ============================================================================
console.log("\n🔍 Part 2: KNNImputer");
console.log("-".repeat(60));

// KNNImputer fills missing values using the mean of k-nearest neighbors
const knnImp = new KNNImputer({ nNeighbors: 2 });
const XFilledKNN = knnImp.fitTransform(XMissing);
console.log("KNNImputer (k=2):");
console.log(XFilledKNN.toString());

// ============================================================================
// Part 3: VarianceThreshold — Remove Low-Variance Features
// ============================================================================
console.log("\n📉 Part 3: VarianceThreshold");
console.log("-".repeat(60));

// VarianceThreshold removes features with variance below a threshold
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
// Part 4: SelectKBest — Univariate Feature Selection
// ============================================================================
console.log("\n⭐ Part 4: SelectKBest");
console.log("-".repeat(60));

// SelectKBest selects the top k features based on a scoring function
const [XFeat, yFeat] = makeClassification({
  nSamples: 100,
  nFeatures: 10,
  nInformative: 3,
  nClasses: 2,
  randomState: 42,
});

console.log(`Original features: ${XFeat.shape[1]}`);

const selector = new SelectKBest({ scoreFunc: f_classif, k: 5 });
const XSelected = selector.fitTransform(XFeat, yFeat);
console.log(`After SelectKBest (k=5): ${XSelected.shape[1]} features`);
console.log(`  Selected feature scores: ${selector.scores.toString()}`);

// ============================================================================
// Part 5: Text Vectorization
// ============================================================================
console.log("\n📝 Part 5: Text Vectorization");
console.log("-".repeat(60));

// CountVectorizer converts text documents to bag-of-words representation
const documents = ["the cat sat on the mat", "the dog sat on the log", "cats and dogs are friends"];

console.log("Documents:");
for (const doc of documents) {
  console.log(`  "${doc}"`);
}

const countVec = new CountVectorizer();
const XCount = countVec.fitTransformText(documents);
console.log(`\nCountVectorizer output shape: ${XCount.shape}`);
console.log(`  Vocabulary size: ${countVec.vocabulary.size}`);

// TfidfVectorizer weights terms by importance (TF-IDF)
const tfidfVec = new TfidfVectorizer();
const XTfidf = tfidfVec.fitTransformText(documents);
console.log(`\nTfidfVectorizer output shape: ${XTfidf.shape}`);
console.log("  TF-IDF values emphasize unique/important words");

// ============================================================================
// Part 6: PolynomialFeatures
// ============================================================================
console.log("\n📐 Part 6: PolynomialFeatures");
console.log("-".repeat(60));

// PolynomialFeatures generates polynomial and interaction features
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
  `  Original: ${XPoly.shape[1]} features → Expanded: ${XPolyTransformed.shape[1]} features`
);

// ============================================================================
// Part 7: SplineTransformer
// ============================================================================
console.log("\n🌊 Part 7: SplineTransformer");
console.log("-".repeat(60));

// SplineTransformer generates B-spline basis functions for flexible modeling
const XSpline = tensor([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]);

console.log("Original 1D feature:");
console.log(XSpline.toString());

const spline = new SplineTransformer({ nKnots: 4, degree: 3 });
const XSplineTransformed = spline.fitTransform(XSpline);
console.log(`\nSplineTransformer (4 knots, degree 3):`);
console.log(`  Output shape: ${XSplineTransformed.shape}`);
console.log(`  1 feature → ${XSplineTransformed.shape[1]} spline basis functions`);

// ============================================================================
// Part 8: RobustScaler & PowerTransformer
// ============================================================================
console.log("\n📊 Part 8: Advanced Scalers");
console.log("-".repeat(60));

const XSkewed = tensor([
  [1, 100],
  [2, 200],
  [3, 300],
  [100, 400], // outlier in first feature
  [4, 500],
]);

// RobustScaler is resistant to outliers (uses median and IQR)
const robust = new RobustScaler();
const XRobust = robust.fitTransform(XSkewed);
console.log("RobustScaler (uses median & IQR, robust to outliers):");
console.log(XRobust.toString());

// PowerTransformer maps data to a Gaussian distribution
const XPositive = tensor([
  [1, 10],
  [2, 20],
  [3, 30],
  [4, 40],
  [5, 50],
]);

const power = new PowerTransformer({ method: "yeo-johnson" });
const XPower = power.fitTransform(XPositive);
console.log("\nPowerTransformer (Yeo-Johnson):");
console.log(XPower.toString());

// ============================================================================
// Part 9: Cross-Validation Splitters
// ============================================================================
console.log("\n✂️  Part 9: Cross-Validation Splitters");
console.log("-".repeat(60));

const [XSplit, ySplit] = makeClassification({
  nSamples: 20,
  nFeatures: 2,
  nInformative: 2,
  nRedundant: 0,
  nClasses: 2,
  randomState: 42,
});

// KFold — standard k-fold cross-validation
const kfold = new KFold({ nSplits: 5, shuffle: true, randomState: 42 });
console.log("KFold (5 splits, shuffled):");
let foldNum = 1;
for (const { trainIndex, testIndex } of kfold.split(XSplit)) {
  console.log(
    `  Fold ${foldNum}: train=${trainIndex.length} samples, test=${testIndex.length} samples`
  );
  foldNum++;
}

// StratifiedKFold — preserves class distribution in each fold
const stratKfold = new StratifiedKFold({ nSplits: 5 });
console.log("\nStratifiedKFold (5 splits, preserves class balance):");
foldNum = 1;
for (const { trainIndex, testIndex } of stratKfold.split(XSplit, ySplit)) {
  console.log(`  Fold ${foldNum}: train=${trainIndex.length}, test=${testIndex.length}`);
  foldNum++;
}

// TimeSeriesSplit — forward-chaining for temporal data
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
console.log("\n💡 Key Takeaways");
console.log("-".repeat(60));
console.log("• SimpleImputer: fill NaN with mean, median, most_frequent, or constant");
console.log("• KNNImputer: fill NaN using k-nearest neighbors (context-aware)");
console.log("• VarianceThreshold: remove features with low variance (near-constant)");
console.log("• SelectKBest: select top-k features by statistical test score");
console.log("• CountVectorizer: bag-of-words text representation");
console.log("• TfidfVectorizer: importance-weighted text representation");
console.log("• SplineTransformer: flexible nonlinear feature expansion via B-splines");
console.log("• RobustScaler: outlier-resistant scaling using median and IQR");
console.log("• PowerTransformer: map data to approximate Gaussian distribution");
console.log("• KFold/StratifiedKFold/TimeSeriesSplit: proper CV for different data types");

console.log("\n✅ Feature Engineering Example Complete!");
console.log("=".repeat(60));
