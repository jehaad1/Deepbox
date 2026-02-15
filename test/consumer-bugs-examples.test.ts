import { describe, expect, it } from "vitest";
import { loadIris, makeBlobs, makeRegression } from "../src/datasets";
import { accuracy, f1Score, r2Score, silhouetteScore } from "../src/metrics";
import {
  DecisionTreeClassifier,
  KMeans,
  KNeighborsClassifier,
  LinearRegression,
  LogisticRegression,
  PCA,
  RandomForestClassifier,
  TSNE,
} from "../src/ml";
import {
  dot,
  flatten,
  gather,
  reshape,
  slice,
  squeeze,
  tensor,
  transpose,
  unsqueeze,
} from "../src/ndarray";
import { Linear, ReLU, Sequential, TransformerEncoderLayer } from "../src/nn";
import { Adam, CosineAnnealingLR, SGD } from "../src/optim";
import { heatmap, hist, plot, scatter } from "../src/plot";
import {
  KFold,
  LabelEncoder,
  MinMaxScaler,
  OneHotEncoder,
  StandardScaler,
  StratifiedKFold,
  trainTestSplit,
} from "../src/preprocess";

describe("bug fixes: docs-shapeindexing", () => {
  it("reshape supports -1 dimension inference", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const r = reshape(t, [-1]);
    expect(r.shape).toEqual([6]);
    expect(r.at(0)).toBe(1);
  });

  it("all shape/index operations work", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(reshape(t, [3, 2]).shape).toEqual([3, 2]);
    expect(reshape(t, [6]).shape).toEqual([6]);
    expect(transpose(t).shape).toEqual([3, 2]);
    expect(flatten(t).shape).toEqual([6]);
    expect(squeeze(tensor([[[1, 2, 3]]])).shape).toEqual([3]);
    expect(unsqueeze(t, 0).shape).toEqual([1, 2, 3]);

    const sliced = slice(t, { start: 0, end: 1 });
    expect(sliced.shape[0]).toBe(1);

    const idx = tensor([0, 2], { dtype: "int32" });
    const gathered = gather(t, idx, 1);
    expect(gathered.shape[1]).toBe(2);
  });
});

describe("bug fixes: quickstart-plotting", () => {
  it("hist accepts { bins: N } object as second argument", () => {
    expect(() => hist(tensor([1, 2, 2, 3, 3, 3, 4, 4, 5]), { bins: 5 })).not.toThrow();
  });

  it("scatter, plot, heatmap work without error", () => {
    scatter(tensor([1, 2, 3, 4, 5]), tensor([2, 4, 5, 4, 6]), {
      color: "#1f77b4",
    });
    plot(tensor([1, 2, 3, 4, 5]), tensor([2, 4, 5, 4, 6]), {
      color: "#ff7f0e",
    });
    heatmap(
      tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ])
    );
  });
});

describe("bug fixes: project1 (dot 1D x 2D)", () => {
  it("dot supports 1D x 2D vector-matrix multiplication", () => {
    const v = tensor([1, 0, 0, 0, 0]);
    const M = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
      [9, 10],
    ]);
    const result = dot(v, M);
    expect(result.shape).toEqual([2]);
    expect(result.at(0)).toBe(1);
    expect(result.at(1)).toBe(2);
  });
});

describe("bug fixes: docs-manifold-learning (TSNE.transform)", () => {
  it("TSNE has transform() method after fit()", () => {
    const X = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
      [13, 14, 15],
    ]);
    const tsne = new TSNE({ nComponents: 2, perplexity: 2, randomState: 42 });
    tsne.fit(X);
    const embedding = tsne.transform(X);
    expect(embedding.shape[0]).toBe(5);
    expect(embedding.shape[1]).toBe(2);
  });
});

describe("bug fixes: example29 (TransformerEncoderLayer options)", () => {
  it("accepts options object as 3rd arg with dimFeedforward", () => {
    const layer = new TransformerEncoderLayer(64, 8, { dimFeedforward: 256 });
    const input = tensor([[[...Array(64).fill(0.1)]]]);
    const output = layer.forward(input);
    expect(output.shape[2]).toBe(64);
  });

  it("accepts full options object as single arg", () => {
    const layer = new TransformerEncoderLayer({ dModel: 32, nHead: 4 });
    const input = tensor([[[...Array(32).fill(0.1)]]]);
    const output = layer.forward(input);
    expect(output.shape[2]).toBe(32);
  });
});

describe("bug fixes: CosineAnnealingLR tMax alias", () => {
  it("accepts tMax option", () => {
    const model = new Sequential(new Linear(2, 1));
    const optimizer = new Adam(model.parameters(), { lr: 0.01 });
    const scheduler = new CosineAnnealingLR(optimizer, { tMax: 100 });
    scheduler.step();
    expect(optimizer.lr).toBeGreaterThan(0);
  });
});

describe("bug fixes: optimizer .lr getter", () => {
  it("exposes current learning rate", () => {
    const model = new Sequential(new Linear(2, 1));
    const optimizer = new SGD(model.parameters(), { lr: 0.05 });
    expect(optimizer.lr).toBe(0.05);
  });
});

describe("bug fixes: f1Score object form", () => {
  it("accepts { average: 'macro' } object", () => {
    const yTrue = tensor([0, 1, 0, 1]);
    const yPred = tensor([0, 1, 1, 1]);
    const score = f1Score(yTrue, yPred, { average: "macro" });
    expect(score).toBeGreaterThan(0);
  });
});

describe("bug fixes: SplitResult objects", () => {
  it("KFold returns objects with trainIndex/testIndex", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6]]);
    const kf = new KFold({ nSplits: 3 });
    const splits = kf.split(X);
    expect(splits.length).toBe(3);
    for (const split of splits) {
      expect(split).toHaveProperty("trainIndex");
      expect(split).toHaveProperty("testIndex");
      expect(split.trainIndex.length + split.testIndex.length).toBe(6);
    }
  });

  it("StratifiedKFold returns objects with trainIndex/testIndex", () => {
    const X = tensor([[1], [2], [3], [4]]);
    const y = tensor([0, 0, 1, 1]);
    const skf = new StratifiedKFold({ nSplits: 2 });
    const splits = skf.split(X, y);
    expect(splits.length).toBe(2);
    for (const split of splits) {
      expect(split).toHaveProperty("trainIndex");
      expect(split).toHaveProperty("testIndex");
    }
  });
});

describe("bug fixes: encoders accept plain arrays", () => {
  it("LabelEncoder accepts plain array", () => {
    const enc = new LabelEncoder();
    enc.fit(["cat", "dog", "cat"]);
    const result = enc.transform(["dog", "cat"]);
    expect(result.size).toBe(2);
  });

  it("OneHotEncoder accepts plain 2D array", () => {
    const enc = new OneHotEncoder();
    enc.fit([["red"], ["blue"], ["green"]]);
    const result = enc.transform([["red"], ["blue"]]);
    expect(result).toBeDefined();
  });
});

describe("bug fixes: tensor accepts booleans", () => {
  it("tensor([true, false, true]) creates tensor", () => {
    const t = tensor([true, false, true]);
    expect(t.size).toBe(3);
    expect(t.at(0)).toBe(1);
    expect(t.at(1)).toBe(0);
  });
});

describe("examples: KMeans clustering", () => {
  it("fits and predicts clusters", () => {
    const X = tensor([
      [1, 2],
      [1.5, 1.8],
      [5, 8],
      [8, 8],
      [1, 0.6],
      [9, 11],
    ]);
    const kmeans = new KMeans({ nClusters: 2, randomState: 42 });
    kmeans.fit(X);
    const labels = kmeans.predict(X);
    expect(labels.size).toBe(6);
    expect(kmeans.inertia).toBeGreaterThan(0);
  });
});

describe("examples: PCA dimensionality reduction", () => {
  it("reduces dimensions", () => {
    const iris = loadIris();
    const pca = new PCA({ nComponents: 2 });
    pca.fit(iris.data);
    const reduced = pca.transform(iris.data);
    expect(reduced.shape[0]).toBe(150);
    expect(reduced.shape[1]).toBe(2);
    expect(pca.explainedVarianceRatio.size).toBe(2);
  });
});

describe("examples: classification pipeline", () => {
  it("train-test split + logistic regression", () => {
    const iris = loadIris();
    const [XTrain, XTest, yTrain, yTest] = trainTestSplit(iris.data, iris.target, {
      testSize: 0.3,
      randomState: 42,
    });
    expect(XTrain.shape[0] + XTest.shape[0]).toBe(150);

    const scaler = new StandardScaler();
    scaler.fit(XTrain);
    const XTrainScaled = scaler.transform(XTrain);
    const XTestScaled = scaler.transform(XTest);

    const lr = new LogisticRegression({ maxIter: 200 });
    lr.fit(XTrainScaled, yTrain);
    const preds = lr.predict(XTestScaled);
    const acc = accuracy(yTest, preds);
    expect(acc).toBeGreaterThan(0.5);
  });
});

describe("examples: neural network forward pass", () => {
  it("Sequential model forward pass produces correct shape", () => {
    const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));

    const X = tensor([
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ]);
    const output = model.forward(X);
    expect(output.shape).toEqual([4, 1]);
  });

  it("model has trainable parameters", () => {
    const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
    const params = [...model.parameters()];
    expect(params.length).toBeGreaterThan(0);
  });
});

describe("docs: clustering", () => {
  it("KMeans + silhouette score", () => {
    const [X] = makeBlobs({ nSamples: 100, centers: 3, randomState: 42 });
    const kmeans = new KMeans({ nClusters: 3, randomState: 42 });
    kmeans.fit(X);
    const labels = kmeans.predict(X);
    const score = silhouetteScore(X, labels);
    expect(score).toBeGreaterThan(0);
  });
});

describe("docs: data preprocessing", () => {
  it("StandardScaler + MinMaxScaler", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const scaler = new StandardScaler();
    scaler.fit(X);
    const scaled = scaler.transform(X);
    expect(scaled.shape).toEqual([3, 2]);

    const mm = new MinMaxScaler();
    mm.fit(X);
    const mmScaled = mm.transform(X);
    expect(mmScaled.shape).toEqual([3, 2]);
  });
});

describe("projects: ML models on Iris", () => {
  it("multiple classifiers run without error", () => {
    const iris = loadIris();
    const [XTrain, XTest, yTrain, yTest] = trainTestSplit(iris.data, iris.target, {
      testSize: 0.3,
      randomState: 42,
    });
    const scaler = new StandardScaler();
    scaler.fit(XTrain);
    const XTs = scaler.transform(XTrain);
    const XTe = scaler.transform(XTest);

    const knn = new KNeighborsClassifier({ nNeighbors: 3 });
    knn.fit(XTs, yTrain);
    const knnPreds = knn.predict(XTe);
    expect(accuracy(yTest, knnPreds)).toBeGreaterThan(0.5);

    const dt = new DecisionTreeClassifier({ maxDepth: 3 });
    dt.fit(XTs, yTrain);
    expect(accuracy(yTest, dt.predict(XTe))).toBeGreaterThan(0.5);

    const rf = new RandomForestClassifier({
      nEstimators: 10,
      maxDepth: 3,
      randomState: 42,
    });
    rf.fit(XTs, yTrain);
    expect(accuracy(yTest, rf.predict(XTe))).toBeGreaterThan(0.5);
  });
});

describe("projects: regression", () => {
  it("LinearRegression on synthetic data", () => {
    const [X, y] = makeRegression({
      nSamples: 100,
      nFeatures: 3,
      noise: 0.1,
      randomState: 42,
    });
    const [XTrain, XTest, yTrain, yTest] = trainTestSplit(X, y, {
      testSize: 0.2,
      randomState: 42,
    });
    const lr = new LinearRegression();
    lr.fit(XTrain, yTrain);
    const preds = lr.predict(XTest);
    const r2 = r2Score(yTest, preds);
    expect(r2).toBeGreaterThan(0.5);
  });
});
