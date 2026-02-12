/**
 * Benchmark 03: ML Training & Inference — Deepbox vs scikit-learn
 *
 * Compares classical ML model training and prediction speed:
 * Linear/Logistic Regression, Ridge, Lasso, KMeans, PCA,
 * Decision Tree, Random Forest, Gradient Boosting, KNN.
 */

import {
  DecisionTreeClassifier,
  GaussianNB,
  GradientBoostingClassifier,
  GradientBoostingRegressor,
  KMeans,
  KNeighborsClassifier,
  KNeighborsRegressor,
  Lasso,
  LinearRegression,
  LogisticRegression,
  PCA,
  RandomForestClassifier,
  Ridge,
} from "deepbox/ml";
import { tensor } from "deepbox/ndarray";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("ml-training");
header("Benchmark 03: ML Training & Inference");

// ── Data Generation ───────────────────────────────────────

function makeDataset(n: number, f: number, seed: number) {
  let state = seed >>> 0;
  const rand = () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  const weights = Array.from({ length: f }, () => rand() * 2 - 1);
  const X: number[][] = [];
  const y: number[] = [];
  const yBin: number[] = [];
  for (let i = 0; i < n; i++) {
    const row = Array.from({ length: f }, () => rand() * 2 - 1);
    const dot = row.reduce((s, v, j) => s + v * (weights[j] ?? 0), 0);
    X.push(row);
    y.push(dot + (rand() - 0.5) * 0.1);
    yBin.push(dot >= 0 ? 1 : 0);
  }
  return { X: tensor(X), y: tensor(y), yBin: tensor(yBin, { dtype: "int32" }) };
}

const small = makeDataset(500, 10, 42);
const medium = makeDataset(2000, 20, 123);
const large = makeDataset(5000, 15, 456);

// ── Linear Regression ─────────────────────────────────────

run(suite, "LinearRegression fit", "500x10", () => {
  new LinearRegression().fit(small.X, small.y);
});

run(suite, "LinearRegression fit", "2000x20", () => {
  new LinearRegression().fit(medium.X, medium.y);
});

const lrModel = new LinearRegression();
lrModel.fit(medium.X, medium.y);
run(suite, "LinearRegression predict", "2000x20", () => {
  lrModel.predict(medium.X);
});

// ── Ridge & Lasso ─────────────────────────────────────────

run(suite, "Ridge fit (alpha=1.0)", "2000x20", () => {
  new Ridge({ alpha: 1.0 }).fit(medium.X, medium.y);
});

run(suite, "Lasso fit (alpha=0.1)", "500x10", () => {
  new Lasso({ alpha: 0.1 }).fit(small.X, small.y);
});

// ── Logistic Regression ───────────────────────────────────

run(suite, "LogisticRegression fit", "500x10", () => {
  new LogisticRegression({ maxIter: 100, learningRate: 0.1 }).fit(small.X, small.yBin);
});

run(
  suite,
  "LogisticRegression fit",
  "2000x20",
  () => {
    new LogisticRegression({ maxIter: 100, learningRate: 0.1 }).fit(medium.X, medium.yBin);
  },
  { iterations: 10 }
);

// ── KMeans ────────────────────────────────────────────────

run(suite, "KMeans fit (k=4)", "500x10", () => {
  new KMeans({ nClusters: 4, maxIter: 20, randomState: 7 }).fit(small.X);
});

run(
  suite,
  "KMeans fit (k=8)",
  "2000x20",
  () => {
    new KMeans({ nClusters: 8, maxIter: 20, randomState: 7 }).fit(medium.X);
  },
  { iterations: 10 }
);

run(
  suite,
  "KMeans fit (k=8)",
  "5000x15",
  () => {
    new KMeans({ nClusters: 8, maxIter: 15, randomState: 7 }).fit(large.X);
  },
  { iterations: 5 }
);

// ── PCA ───────────────────────────────────────────────────

run(suite, "PCA fit (n=5)", "500x10", () => {
  new PCA({ nComponents: 5 }).fit(small.X);
});

run(suite, "PCA fit (n=10)", "2000x20", () => {
  new PCA({ nComponents: 10 }).fit(medium.X);
});

const pcaModel = new PCA({ nComponents: 5 });
pcaModel.fit(medium.X);
run(suite, "PCA transform", "2000x20 -> 5", () => {
  pcaModel.transform(medium.X);
});

// ── Decision Tree ─────────────────────────────────────────

run(suite, "DecisionTreeClassifier fit", "500x10", () => {
  new DecisionTreeClassifier({ maxDepth: 6 }).fit(small.X, small.yBin);
});

run(
  suite,
  "DecisionTreeClassifier fit",
  "2000x20",
  () => {
    new DecisionTreeClassifier({ maxDepth: 8 }).fit(medium.X, medium.yBin);
  },
  { iterations: 10 }
);

// ── Random Forest ─────────────────────────────────────────

run(suite, "RandomForestClassifier fit (5 trees)", "500x10", () => {
  new RandomForestClassifier({
    nEstimators: 5,
    maxDepth: 6,
    randomState: 42,
  }).fit(small.X, small.yBin);
});

run(
  suite,
  "RandomForestClassifier fit (10 trees)",
  "2000x20",
  () => {
    new RandomForestClassifier({
      nEstimators: 10,
      maxDepth: 6,
      randomState: 42,
    }).fit(medium.X, medium.yBin);
  },
  { iterations: 5 }
);

// ── Gradient Boosting ─────────────────────────────────────

run(suite, "GradientBoostingClassifier (20 trees)", "500x10", () => {
  new GradientBoostingClassifier({
    nEstimators: 20,
    learningRate: 0.1,
    maxDepth: 3,
  }).fit(small.X, small.yBin);
});

run(suite, "GradientBoostingRegressor (20 trees)", "500x10", () => {
  new GradientBoostingRegressor({
    nEstimators: 20,
    learningRate: 0.1,
    maxDepth: 3,
  }).fit(small.X, small.y);
});

// ── KNN ───────────────────────────────────────────────────

const knnC = new KNeighborsClassifier({ nNeighbors: 5 });
knnC.fit(small.X, small.yBin);
run(suite, "KNeighborsClassifier predict", "100 queries", () => {
  knnC.predict(
    tensor(
      Array.from({ length: 100 }, (_, i) =>
        Array.from({ length: 10 }, (_, j) => (i * 10 + j) / 1000)
      )
    )
  );
});

const knnR = new KNeighborsRegressor({ nNeighbors: 5 });
knnR.fit(small.X, small.y);
run(suite, "KNeighborsRegressor predict", "100 queries", () => {
  knnR.predict(
    tensor(
      Array.from({ length: 100 }, (_, i) =>
        Array.from({ length: 10 }, (_, j) => (i * 10 + j) / 1000)
      )
    )
  );
});

// ── Gaussian Naive Bayes ──────────────────────────────────

run(suite, "GaussianNB fit", "500x10", () => {
  new GaussianNB().fit(small.X, small.yBin);
});

run(suite, "GaussianNB fit", "2000x20", () => {
  new GaussianNB().fit(medium.X, medium.yBin);
});

footer(suite, "deepbox-ml-training.json");
