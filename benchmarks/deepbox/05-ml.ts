/**
 * Benchmark 05 — Machine Learning
 * Deepbox vs scikit-learn
 */

import {
  DBSCAN,
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  GaussianNB,
  GradientBoostingClassifier,
  GradientBoostingRegressor,
  KMeans,
  KNeighborsClassifier,
  KNeighborsRegressor,
  Lasso,
  LinearRegression,
  LinearSVC,
  LinearSVR,
  LogisticRegression,
  PCA,
  RandomForestClassifier,
  RandomForestRegressor,
  Ridge,
  TSNE,
} from "deepbox/ml";
import { tensor } from "deepbox/ndarray";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("ml");
header("Benchmark 05 — Machine Learning");

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
const clust200 = makeClustData(200, 5, 42);
const clust500 = makeClustData(500, 5, 42);

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

// ── Lasso ───────────────────────────────────────────────

run(suite, "Lasso fit", "200x5", () => new Lasso({ alpha: 0.1 }).fit(reg200.X, reg200.y));
run(suite, "Lasso fit", "500x10", () => new Lasso({ alpha: 0.1 }).fit(reg500.X, reg500.y));
const lasso = new Lasso({ alpha: 0.1 }).fit(reg200.X, reg200.y);
run(suite, "Lasso predict", "200x5", () => lasso.predict(reg200.X));

// ── Logistic Regression ─────────────────────────────────

run(suite, "LogisticRegression fit", "200x5", () =>
  new LogisticRegression().fit(cls200.X, cls200.y)
);
run(suite, "LogisticRegression fit", "500x10", () =>
  new LogisticRegression().fit(cls500.X, cls500.y)
);
const logReg = new LogisticRegression().fit(cls200.X, cls200.y);
run(suite, "LogisticRegression predict", "200x5", () => logReg.predict(cls200.X));

// ── GaussianNB ──────────────────────────────────────────

run(suite, "GaussianNB fit", "200x5", () => new GaussianNB().fit(cls200.X, cls200.y));
run(suite, "GaussianNB fit", "500x10", () => new GaussianNB().fit(cls500.X, cls500.y));
const gnb = new GaussianNB().fit(cls200.X, cls200.y);
run(suite, "GaussianNB predict", "200x5", () => gnb.predict(cls200.X));

// ── KNN Classifier ──────────────────────────────────────

run(suite, "KNeighborsClassifier fit", "200x5", () =>
  new KNeighborsClassifier({ nNeighbors: 5 }).fit(cls200.X, cls200.y)
);
run(suite, "KNeighborsClassifier fit", "500x10", () =>
  new KNeighborsClassifier({ nNeighbors: 5 }).fit(cls500.X, cls500.y)
);
const knnc = new KNeighborsClassifier({ nNeighbors: 5 }).fit(cls200.X, cls200.y);
run(suite, "KNeighborsClassifier predict", "200x5", () => knnc.predict(cls200.X));

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

footer(suite, "deepbox-ml.json");
