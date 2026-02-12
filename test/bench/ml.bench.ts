import { bench, describe } from "vitest";
import {
  DecisionTreeClassifier,
  GradientBoostingRegressor,
  KMeans,
  KNeighborsRegressor,
  LinearRegression,
  LogisticRegression,
  PCA,
  RandomForestClassifier,
  TSNE,
} from "../../src/ml";
import { tensor } from "../../src/ndarray";

type Dataset = {
  readonly X: number[][];
  readonly y: number[];
  readonly yBinary: number[];
};

function makeDataset(nSamples: number, nFeatures: number, seed: number): Dataset {
  let state = seed >>> 0;
  const rand = (): number => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 32;
  };

  const weights: number[] = new Array(nFeatures).fill(0).map(() => rand() * 2 - 1);
  const X: number[][] = [];
  const y: number[] = [];
  const yBinary: number[] = [];

  for (let i = 0; i < nSamples; i++) {
    const row: number[] = new Array(nFeatures);
    let dot = 0;
    for (let j = 0; j < nFeatures; j++) {
      const v = rand() * 2 - 1;
      row[j] = v;
      dot += v * (weights[j] ?? 0);
    }
    X.push(row);
    const noise = (rand() - 0.5) * 0.1;
    const target = dot + noise;
    y.push(target);
    yBinary.push(target >= 0 ? 1 : 0);
  }

  return { X, y, yBinary };
}

const small = makeDataset(200, 10, 123);
const medium = makeDataset(1000, 20, 456);
const large = makeDataset(5000, 30, 789);
const huge = makeDataset(20000, 10, 101112);

const smallX = tensor(small.X);
const smallY = tensor(small.y);
const smallYBinary = tensor(small.yBinary, { dtype: "int32" });

const mediumX = tensor(medium.X);
const mediumY = tensor(medium.y);

const largeX = tensor(large.X);
const largeY = tensor(large.y);
const largeYBinary = tensor(large.yBinary, { dtype: "int32" });

const hugeX = tensor(huge.X);

describe("ML performance benchmarks", () => {
  bench("LinearRegression fit (200x10)", () => {
    const model = new LinearRegression();
    model.fit(smallX, smallY);
  });

  const lr = new LinearRegression();
  lr.fit(mediumX, mediumY);
  bench("LinearRegression predict (1000x20)", () => {
    lr.predict(mediumX);
  });

  bench("LogisticRegression fit (200x10)", () => {
    const model = new LogisticRegression({ maxIter: 200, learningRate: 0.1 });
    model.fit(smallX, smallYBinary);
  });

  bench("KMeans fit (200x10, k=4)", () => {
    const model = new KMeans({ nClusters: 4, maxIter: 20, randomState: 7 });
    model.fit(smallX);
  });

  const pca = new PCA({ nComponents: 5 });
  pca.fit(mediumX);
  bench("PCA transform (1000x20 -> 5)", () => {
    pca.transform(mediumX);
  });

  bench("DecisionTreeClassifier fit (200x10)", () => {
    const model = new DecisionTreeClassifier({ maxDepth: 6, minSamplesSplit: 2 });
    model.fit(smallX, smallYBinary);
  });

  bench("RandomForestClassifier fit (200x10, 5 trees)", () => {
    const model = new RandomForestClassifier({
      nEstimators: 5,
      maxDepth: 6,
      maxFeatures: "sqrt",
      bootstrap: true,
      randomState: 42,
    });
    model.fit(smallX, smallYBinary);
  });

  bench("GradientBoostingRegressor fit (200x10, 20 trees)", () => {
    const model = new GradientBoostingRegressor({
      nEstimators: 20,
      learningRate: 0.1,
      maxDepth: 3,
    });
    model.fit(smallX, smallY);
  });

  const knn = new KNeighborsRegressor({ nNeighbors: 5 });
  knn.fit(mediumX, mediumY);
  const knnQuery = tensor(medium.X.slice(0, 200));
  bench("KNeighborsRegressor predict (200 queries, 1k train)", () => {
    knn.predict(knnQuery);
  });
});

describe("ML performance benchmarks - scaling and memory pressure", () => {
  bench("PCA fit (5000x30)", () => {
    const model = new PCA({ nComponents: 10 });
    model.fit(largeX);
  });

  bench("PCA transform (20000x10 -> 5) [memory]", () => {
    const model = new PCA({ nComponents: 5 });
    model.fit(hugeX);
    model.transform(hugeX);
  });

  bench("KMeans fit (5000x30, k=8)", () => {
    const model = new KMeans({ nClusters: 8, maxIter: 15, randomState: 11 });
    model.fit(largeX);
  });

  bench("RandomForestClassifier fit (5000x30, 10 trees)", () => {
    const model = new RandomForestClassifier({
      nEstimators: 10,
      maxDepth: 8,
      maxFeatures: "sqrt",
      bootstrap: true,
      randomState: 7,
    });
    model.fit(largeX, largeYBinary);
  });

  bench("KNeighborsRegressor predict (1000 queries, 5k train)", () => {
    const model = new KNeighborsRegressor({ nNeighbors: 7 });
    model.fit(largeX, largeY);
    const queries = tensor(large.X.slice(0, 1000));
    model.predict(queries);
  });

  bench("TSNE approximate fit (1000x30)", () => {
    const sample = large.X.slice(0, 1000);
    const X = tensor(sample);
    const model = new TSNE({
      nComponents: 2,
      nIter: 250,
      perplexity: 30,
      method: "approximate",
      approximateNeighbors: 90,
      negativeSamples: 50,
      randomState: 13,
    });
    model.fit(X);
  });
});
