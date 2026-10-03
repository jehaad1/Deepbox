/**
 * Benchmark 05: Machine Learning
 * Deepbox vs scikit-learn
 */

import {
  AdaBoostClassifier,
  BaggingClassifier,
  BayesianRidge,
  BernoulliNB,
  Birch,
  DBSCAN,
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  ElasticNet,
  ExtraTreesClassifier,
  GaussianMixture,
  GaussianNB,
  GaussianRandomProjection,
  GradientBoostingClassifier,
  GradientBoostingRegressor,
  IsolationForest,
  KMeans,
  KNeighborsClassifier,
  KNeighborsRegressor,
  Lasso,
  LinearDiscriminantAnalysis,
  LinearRegression,
  LinearSVC,
  LinearSVR,
  LocalOutlierFactor,
  LogisticRegression,
  MeanShift,
  MiniBatchKMeans,
  MultinomialNB,
  NearestCentroid,
  NuSVC,
  OPTICS,
  PCA,
  RadiusNeighborsClassifier,
  RandomForestClassifier,
  RandomForestRegressor,
  Ridge,
  SGDClassifier,
  SGDRegressor,
  SpectralClustering,
  StackingClassifier,
  SVC,
  TSNE,
  VotingClassifier,
} from "deepbox/ml";
import { tensor } from "deepbox/ndarray";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("ml");
header("Benchmark 05: Machine Learning");

// ── Data generators ──────────────────────────────────────

function seededRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

function makeRegData(n: number, f: number, seed: number) {
  const rand = seededRng(seed);
  const X: number[][] = [],
    y: number[] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    let v = 0;
    for (let j = 0; j < f; j++) {
      const x = rand() * 10;
      row.push(x);
      v += x * (j + 1);
    }
    X.push(row);
    y.push(v + (rand() - 0.5) * 2);
  }
  return { X: tensor(X), y: tensor(y) };
}

function makeClsData(n: number, f: number, seed: number) {
  const rand = seededRng(seed);
  const X: number[][] = [],
    y: number[] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    let s = 0;
    for (let j = 0; j < f; j++) {
      const x = rand() * 10;
      row.push(x);
      s += x;
    }
    X.push(row);
    y.push(s > f * 5 ? 1 : 0);
  }
  return { X: tensor(X), y: tensor(y) };
}

function makeMultiClsData(n: number, f: number, k: number, seed: number) {
  const rand = seededRng(seed);
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const cls = i % k;
    const row: number[] = [];
    for (let j = 0; j < f; j++) {
      row.push(cls * 2.5 + rand() * 1.25);
    }
    X.push(row);
    y.push(cls);
  }
  return { X: tensor(X), y: tensor(y) };
}

function makeCountClsData(n: number, f: number, k: number, seed: number) {
  const rand = seededRng(seed);
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const cls = i % k;
    const row: number[] = [];
    for (let j = 0; j < f; j++) {
      row.push(cls + Math.floor(rand() * 4));
    }
    X.push(row);
    y.push(cls);
  }
  return { X: tensor(X), y: tensor(y) };
}

function makeBinaryFeatureClsData(n: number, f: number, k: number, seed: number) {
  const rand = seededRng(seed);
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 0; i < n; i++) {
    const cls = i % k;
    const row: number[] = [];
    for (let j = 0; j < f; j++) {
      const threshold = (cls + j) % 3 === 0 ? 0.25 : 0.55;
      row.push(rand() > threshold ? 1 : 0);
    }
    X.push(row);
    y.push(cls);
  }
  return { X: tensor(X), y: tensor(y) };
}

function makeClustData(n: number, f: number, seed: number) {
  const rand = seededRng(seed);
  const X: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    const c = i % 3;
    for (let j = 0; j < f; j++) row.push(c * 5 + rand() * 2);
    X.push(row);
  }
  return tensor(X);
}

const reg200 = makeRegData(200, 5, 42);
const reg500 = makeRegData(500, 10, 42);
const cls200 = makeClsData(200, 5, 42);
const cls500 = makeClsData(500, 10, 42);
const clsMulti200 = makeMultiClsData(200, 5, 3, 42);
const _clsMulti120 = makeMultiClsData(120, 2, 3, 84);
const countCls200 = makeCountClsData(200, 8, 3, 17);
const countCls500 = makeCountClsData(500, 12, 3, 23);
const binaryCls200 = makeBinaryFeatureClsData(200, 8, 3, 29);
const clust200 = makeClustData(200, 5, 42);
const clust500 = makeClustData(500, 5, 42);
const clust120_2d = makeClustData(120, 2, 64);

// ── Linear Regression ───────────────────────────────────

run(suite, "LinearRegression fit", "200x5", () => new LinearRegression().fit(reg200.X, reg200.y));
run(suite, "LinearRegression fit", "500x10", () => new LinearRegression().fit(reg500.X, reg500.y));
const lr = new LinearRegression().fit(reg200.X, reg200.y);
run(suite, "LinearRegression predict", "200x5", () => lr.predict(reg200.X));

// ── Ridge ───────────────────────────────────────────────

run(suite, "Ridge fit", "200x5", () => new Ridge({ alpha: 1.0 }).fit(reg200.X, reg200.y));
run(suite, "Ridge fit", "500x10", () => new Ridge({ alpha: 1.0 }).fit(reg500.X, reg500.y));
const ridge = new Ridge({ alpha: 1.0 }).fit(reg200.X, reg200.y);
run(suite, "Ridge predict", "200x5", () => ridge.predict(reg200.X));

// ── Bayesian Ridge ─────────────────────────────────────

run(suite, "BayesianRidge fit", "200x5", () => new BayesianRidge().fit(reg200.X, reg200.y));
const bayesRidge = new BayesianRidge().fit(reg200.X, reg200.y);
run(suite, "BayesianRidge predict", "200x5", () => bayesRidge.predict(reg200.X));

// ── Lasso ───────────────────────────────────────────────

run(suite, "Lasso fit", "200x5", () => new Lasso({ alpha: 0.1 }).fit(reg200.X, reg200.y));
run(suite, "Lasso fit", "500x10", () => new Lasso({ alpha: 0.1 }).fit(reg500.X, reg500.y));
const lasso = new Lasso({ alpha: 0.1 }).fit(reg200.X, reg200.y);
run(suite, "Lasso predict", "200x5", () => lasso.predict(reg200.X));

// ── ElasticNet ─────────────────────────────────────────

run(suite, "ElasticNet fit", "200x5", () =>
  new ElasticNet({ alpha: 0.05, l1Ratio: 0.5, randomState: 42 }).fit(reg200.X, reg200.y)
);
const elasticNet = new ElasticNet({ alpha: 0.05, l1Ratio: 0.5, randomState: 42 }).fit(
  reg200.X,
  reg200.y
);
run(suite, "ElasticNet predict", "200x5", () => elasticNet.predict(reg200.X));

// ── Logistic Regression ─────────────────────────────────

run(suite, "LogisticRegression fit", "200x5", () =>
  new LogisticRegression().fit(cls200.X, cls200.y)
);
run(suite, "LogisticRegression fit", "500x10", () =>
  new LogisticRegression().fit(cls500.X, cls500.y)
);
const logReg = new LogisticRegression().fit(cls200.X, cls200.y);
run(suite, "LogisticRegression predict", "200x5", () => logReg.predict(cls200.X));

// ── Linear Discriminant Analysis ───────────────────────

run(suite, "LinearDiscriminantAnalysis fit", "200x5", () =>
  new LinearDiscriminantAnalysis({ nComponents: 2 }).fit(clsMulti200.X, clsMulti200.y)
);
const lda = new LinearDiscriminantAnalysis({ nComponents: 2 }).fit(clsMulti200.X, clsMulti200.y);
run(suite, "LinearDiscriminantAnalysis transform", "200x5→2", () => lda.transform(clsMulti200.X));

// ── GaussianNB ──────────────────────────────────────────

run(suite, "GaussianNB fit", "200x5", () => new GaussianNB().fit(cls200.X, cls200.y));
run(suite, "GaussianNB fit", "500x10", () => new GaussianNB().fit(cls500.X, cls500.y));
const gnb = new GaussianNB().fit(cls200.X, cls200.y);
run(suite, "GaussianNB predict", "200x5", () => gnb.predict(cls200.X));

// ── BernoulliNB ────────────────────────────────────────

run(suite, "BernoulliNB fit", "200x8", () => new BernoulliNB().fit(binaryCls200.X, binaryCls200.y));
const bernoulli = new BernoulliNB().fit(binaryCls200.X, binaryCls200.y);
run(suite, "BernoulliNB predict", "200x8", () => bernoulli.predict(binaryCls200.X));

// ── MultinomialNB ──────────────────────────────────────

run(suite, "MultinomialNB fit", "200x8", () =>
  new MultinomialNB().fit(countCls200.X, countCls200.y)
);
run(suite, "MultinomialNB fit", "500x12", () =>
  new MultinomialNB().fit(countCls500.X, countCls500.y)
);
const multinomial = new MultinomialNB().fit(countCls200.X, countCls200.y);
run(suite, "MultinomialNB predict", "200x8", () => multinomial.predict(countCls200.X));

// ── KNN Classifier ──────────────────────────────────────

run(suite, "KNeighborsClassifier fit", "200x5", () =>
  new KNeighborsClassifier({ nNeighbors: 5 }).fit(cls200.X, cls200.y)
);
run(suite, "KNeighborsClassifier fit", "500x10", () =>
  new KNeighborsClassifier({ nNeighbors: 5 }).fit(cls500.X, cls500.y)
);
const knnc = new KNeighborsClassifier({ nNeighbors: 5 }).fit(cls200.X, cls200.y);
run(suite, "KNeighborsClassifier predict", "200x5", () => knnc.predict(cls200.X));

// ── Radius-based Neighbors ─────────────────────────────

run(suite, "RadiusNeighborsClassifier fit", "200x5", () =>
  new RadiusNeighborsClassifier({ radius: 2.5 }).fit(clsMulti200.X, clsMulti200.y)
);
const radiusCls = new RadiusNeighborsClassifier({ radius: 2.5 }).fit(clsMulti200.X, clsMulti200.y);
run(suite, "RadiusNeighborsClassifier predict", "200x5", () => radiusCls.predict(clsMulti200.X));

// ── Nearest Centroid ───────────────────────────────────

run(suite, "NearestCentroid fit", "200x5", () =>
  new NearestCentroid().fit(clsMulti200.X, clsMulti200.y)
);
const nearestCentroid = new NearestCentroid().fit(clsMulti200.X, clsMulti200.y);
run(suite, "NearestCentroid predict", "200x5", () => nearestCentroid.predict(clsMulti200.X));

// ── KNN Regressor ───────────────────────────────────────

run(suite, "KNeighborsRegressor fit", "200x5", () =>
  new KNeighborsRegressor({ nNeighbors: 5 }).fit(reg200.X, reg200.y)
);
const knnr = new KNeighborsRegressor({ nNeighbors: 5 }).fit(reg200.X, reg200.y);
run(suite, "KNeighborsRegressor predict", "200x5", () => knnr.predict(reg200.X));

// ── LinearSVC ───────────────────────────────────────────

run(suite, "LinearSVC fit", "200x5", () => new LinearSVC().fit(cls200.X, cls200.y));
run(suite, "LinearSVC fit", "500x10", () => new LinearSVC().fit(cls500.X, cls500.y));
const svc = new LinearSVC().fit(cls200.X, cls200.y);
run(suite, "LinearSVC predict", "200x5", () => svc.predict(cls200.X));

// ── LinearSVR ───────────────────────────────────────────

run(suite, "LinearSVR fit", "200x5", () => new LinearSVR().fit(reg200.X, reg200.y));
const svr = new LinearSVR().fit(reg200.X, reg200.y);
run(suite, "LinearSVR predict", "200x5", () => svr.predict(reg200.X));

// ── SGD Models ─────────────────────────────────────────

run(suite, "SGDClassifier fit", "200x5", () =>
  new SGDClassifier({ loss: "log_loss", maxIter: 200, randomState: 42 }).fit(cls200.X, cls200.y)
);
const sgdClassifier = new SGDClassifier({
  loss: "log_loss",
  maxIter: 200,
  randomState: 42,
}).fit(cls200.X, cls200.y);
run(suite, "SGDClassifier predict", "200x5", () => sgdClassifier.predict(cls200.X));

run(suite, "SGDRegressor fit", "200x5", () =>
  new SGDRegressor({ maxIter: 200, randomState: 42 }).fit(reg200.X, reg200.y)
);
const sgdRegressor = new SGDRegressor({ maxIter: 200, randomState: 42 }).fit(reg200.X, reg200.y);
run(suite, "SGDRegressor predict", "200x5", () => sgdRegressor.predict(reg200.X));

// ── Decision Tree ───────────────────────────────────────

run(suite, "DecisionTreeClassifier fit", "200x5", () =>
  new DecisionTreeClassifier({ maxDepth: 5 }).fit(cls200.X, cls200.y)
);
run(suite, "DecisionTreeClassifier fit", "500x10", () =>
  new DecisionTreeClassifier({ maxDepth: 5 }).fit(cls500.X, cls500.y)
);
const dtc = new DecisionTreeClassifier({ maxDepth: 5 }).fit(cls200.X, cls200.y);
run(suite, "DecisionTreeClassifier predict", "200x5", () => dtc.predict(cls200.X));

run(suite, "DecisionTreeRegressor fit", "200x5", () =>
  new DecisionTreeRegressor({ maxDepth: 5 }).fit(reg200.X, reg200.y)
);
const dtr = new DecisionTreeRegressor({ maxDepth: 5 }).fit(reg200.X, reg200.y);
run(suite, "DecisionTreeRegressor predict", "200x5", () => dtr.predict(reg200.X));

// ── Random Forest ───────────────────────────────────────

run(
  suite,
  "RandomForestClassifier fit",
  "200x5",
  () => new RandomForestClassifier({ nEstimators: 10, maxDepth: 5 }).fit(cls200.X, cls200.y),
  { iterations: 5 }
);
run(
  suite,
  "RandomForestClassifier fit",
  "500x10",
  () => new RandomForestClassifier({ nEstimators: 10, maxDepth: 5 }).fit(cls500.X, cls500.y),
  { iterations: 5 }
);
const rfc = new RandomForestClassifier({ nEstimators: 10, maxDepth: 5 }).fit(cls200.X, cls200.y);
run(suite, "RandomForestClassifier predict", "200x5", () => rfc.predict(cls200.X));

run(
  suite,
  "RandomForestRegressor fit",
  "200x5",
  () => new RandomForestRegressor({ nEstimators: 10, maxDepth: 5 }).fit(reg200.X, reg200.y),
  { iterations: 5 }
);
const rfr = new RandomForestRegressor({ nEstimators: 10, maxDepth: 5 }).fit(reg200.X, reg200.y);
run(suite, "RandomForestRegressor predict", "200x5", () => rfr.predict(reg200.X));

// ── Gradient Boosting ───────────────────────────────────

run(
  suite,
  "GradientBoostingClassifier fit",
  "200x5",
  () => new GradientBoostingClassifier({ nEstimators: 10, maxDepth: 3 }).fit(cls200.X, cls200.y),
  { iterations: 5 }
);
const gbc = new GradientBoostingClassifier({
  nEstimators: 10,
  maxDepth: 3,
}).fit(cls200.X, cls200.y);
run(suite, "GradientBoostingClassifier predict", "200x5", () => gbc.predict(cls200.X));

run(
  suite,
  "GradientBoostingRegressor fit",
  "200x5",
  () => new GradientBoostingRegressor({ nEstimators: 10, maxDepth: 3 }).fit(reg200.X, reg200.y),
  { iterations: 5 }
);
const gbr = new GradientBoostingRegressor({ nEstimators: 10, maxDepth: 3 }).fit(reg200.X, reg200.y);
run(suite, "GradientBoostingRegressor predict", "200x5", () => gbr.predict(reg200.X));

// ── KMeans ──────────────────────────────────────────────

run(suite, "KMeans fit", "200x5 k=3", () =>
  new KMeans({ nClusters: 3, maxIter: 50 }).fit(clust200)
);
run(suite, "KMeans fit", "500x5 k=3", () =>
  new KMeans({ nClusters: 3, maxIter: 50 }).fit(clust500)
);
const km = new KMeans({ nClusters: 3, maxIter: 50 }).fit(clust200);
run(suite, "KMeans predict", "200x5", () => km.predict(clust200));

// ── DBSCAN ──────────────────────────────────────────────

run(suite, "DBSCAN fit", "200x5", () => new DBSCAN({ eps: 2.0, minSamples: 5 }).fit(clust200));
run(suite, "DBSCAN fit", "500x5", () => new DBSCAN({ eps: 2.0, minSamples: 5 }).fit(clust500));

// ── PCA ─────────────────────────────────────────────────

run(suite, "PCA fit", "200x5 k=2", () => new PCA({ nComponents: 2 }).fit(clust200));
run(suite, "PCA fit", "500x5 k=3", () => new PCA({ nComponents: 3 }).fit(clust500));
const pca = new PCA({ nComponents: 2 }).fit(clust200);
run(suite, "PCA transform", "200x5", () => pca.transform(clust200));

// ── TSNE ────────────────────────────────────────────────

run(suite, "TSNE fit", "200x5", () => new TSNE({ nComponents: 2, perplexity: 30 }).fit(clust200), {
  iterations: 3,
  warmup: 1,
});

// ── Additional v1.0.0 estimators ───────────────────────

run(
  suite,
  "AdaBoostClassifier fit",
  "200x5",
  () => new AdaBoostClassifier({ nEstimators: 20, learningRate: 1.0 }).fit(cls200.X, cls200.y),
  { iterations: 5 }
);
run(
  suite,
  "BaggingClassifier fit",
  "200x5",
  () => new BaggingClassifier({ nEstimators: 10, randomState: 42 }).fit(cls200.X, cls200.y),
  { iterations: 5 }
);
run(
  suite,
  "VotingClassifier fit",
  "200x5",
  () =>
    new VotingClassifier({
      estimators: [
        new LogisticRegression({ maxIter: 100 }),
        new RandomForestClassifier({ nEstimators: 5, randomState: 42 }),
        new KNeighborsClassifier({ nNeighbors: 5 }),
      ],
      voting: "hard",
    }).fit(cls200.X, cls200.y),
  { iterations: 5 }
);
run(
  suite,
  "StackingClassifier fit",
  "200x5",
  () =>
    new StackingClassifier({
      estimators: [
        new DecisionTreeClassifier({ maxDepth: 5 }),
        new KNeighborsClassifier({ nNeighbors: 5 }),
      ],
      finalEstimator: new LogisticRegression({ maxIter: 100 }),
    }).fit(cls200.X, cls200.y),
  { iterations: 5 }
);
run(
  suite,
  "ExtraTreesClassifier fit",
  "200x5",
  () =>
    new ExtraTreesClassifier({ nEstimators: 20, maxDepth: 5, randomState: 42 }).fit(
      cls200.X,
      cls200.y
    ),
  { iterations: 5 }
);
run(
  suite,
  "SVC fit",
  "200x5",
  () => new SVC({ kernel: "rbf", C: 10, maxIter: 200 }).fit(cls200.X, cls200.y),
  {
    iterations: 3,
    warmup: 1,
  }
);
const svcKernel = new SVC({ kernel: "rbf", C: 10, maxIter: 200 }).fit(cls200.X, cls200.y);
run(suite, "SVC predict", "200x5", () => svcKernel.predict(cls200.X));
run(suite, "NuSVC fit", "200x5", () => new NuSVC({ nu: 0.5 }).fit(cls200.X, cls200.y), {
  iterations: 3,
  warmup: 1,
});
run(
  suite,
  "IsolationForest fit",
  "200x5",
  () => new IsolationForest({ nEstimators: 50, randomState: 42 }).fit(clust200),
  { iterations: 5 }
);
const iforest = new IsolationForest({ nEstimators: 50, randomState: 42 }).fit(clust200);
run(suite, "IsolationForest predict", "200x5", () => iforest.predict(clust200));
run(
  suite,
  "LocalOutlierFactor fit",
  "200x5",
  () => new LocalOutlierFactor({ nNeighbors: 10 }).fit(clust200),
  { iterations: 5 }
);
const lof = new LocalOutlierFactor({ nNeighbors: 10 }).fit(clust200);
run(suite, "LocalOutlierFactor predict", "200x5", () => lof.predict(clust200));
run(
  suite,
  "GaussianMixture fit",
  "200x5 k=3",
  () => new GaussianMixture({ nComponents: 3, randomState: 42 }).fit(clust200),
  { iterations: 5 }
);
run(
  suite,
  "MiniBatchKMeans fit",
  "200x5 k=3",
  () => new MiniBatchKMeans({ nClusters: 3, batchSize: 32, maxIter: 50 }).fit(clust200),
  { iterations: 5 }
);
run(
  suite,
  "SpectralClustering fit",
  "200x5 k=3",
  () => new SpectralClustering({ nClusters: 3, randomState: 42 }).fit(clust200),
  { iterations: 3, warmup: 1 }
);
run(
  suite,
  "Birch fit",
  "120x2 k=3",
  () => new Birch({ nClusters: 3, threshold: 0.8 }).fit(clust120_2d),
  { iterations: 5 }
);
const birch = new Birch({ nClusters: 3, threshold: 0.8 }).fit(clust120_2d);
run(suite, "Birch predict", "120x2", () => birch.predict(clust120_2d));
run(
  suite,
  "MeanShift fit",
  "120x2",
  () => new MeanShift({ bandwidth: 2.5, maxIter: 60 }).fit(clust120_2d),
  { iterations: 3, warmup: 1 }
);
const meanShift = new MeanShift({ bandwidth: 2.5, maxIter: 60 }).fit(clust120_2d);
run(suite, "MeanShift predict", "120x2", () => meanShift.predict(clust120_2d));
run(
  suite,
  "OPTICS fit",
  "120x2",
  () => new OPTICS({ minSamples: 5, maxEps: 3.5 }).fit(clust120_2d),
  { iterations: 3, warmup: 1 }
);
run(suite, "GaussianRandomProjection fit+transform", "500x10→4", () =>
  new GaussianRandomProjection({ nComponents: 4, seed: 42 }).fitTransform(reg500.X)
);

footer(suite, "deepbox-ml.json");
