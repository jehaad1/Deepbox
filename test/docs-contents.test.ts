/**
 * Tests derived from Deepbox-Docs/src/app/docs/contents/*.json code snippets.
 * Covers API patterns documented in the official docs JSON files.
 */
import { describe, expect, it } from "vitest";
import {
  DataLoader,
  loadBreastCancer,
  loadIris,
  makeBlobs,
  makeCircles,
  makeMoons,
  makeRegression,
} from "../src/datasets";
import {
  accuracy,
  adjustedRandScore,
  confusionMatrix,
  daviesBouldinScore,
  f1Score,
  mae,
  mse,
  precision,
  r2Score,
  recall,
  rmse,
  silhouetteScore,
} from "../src/metrics";
import {
  DBSCAN,
  DecisionTreeClassifier,
  DecisionTreeRegressor,
  KMeans,
  Lasso,
  LinearRegression,
  LogisticRegression,
  PCA,
  Ridge,
} from "../src/ml";
import {
  add,
  arange,
  CSRMatrix,
  concatenate,
  dot,
  elu,
  eye,
  flatten,
  full,
  gather,
  gelu,
  leakyRelu,
  linspace,
  mean,
  mish,
  mul,
  noGrad,
  ones,
  parameter,
  randn,
  relu,
  reshape,
  sigmoid,
  sort,
  squeeze,
  stack,
  sum,
  tensor,
  transpose,
  unsqueeze,
  zeros,
} from "../src/ndarray";
import {
  BatchNorm1d,
  Dropout,
  GRU,
  huberLoss,
  LayerNorm,
  Linear,
  LSTM,
  MultiheadAttention,
  maeLoss,
  mseLoss,
  ReLU as ReLULayer,
  RNN,
  Sequential,
  TransformerEncoderLayer,
} from "../src/nn";
import { Adam, SGD, StepLR } from "../src/optim";
import { Figure } from "../src/plot";
import {
  KFold,
  LabelBinarizer,
  LabelEncoder,
  MinMaxScaler,
  OneHotEncoder,
  StandardScaler,
  trainTestSplit,
} from "../src/preprocess";
import {
  binomial,
  choice,
  clearSeed,
  normal,
  permutation,
  poisson,
  rand,
  randint,
  setSeed,
  uniform,
} from "../src/random";
import {
  corrcoef,
  cov,
  geometricMean,
  kendalltau,
  kurtosis,
  median,
  pearsonr,
  percentile,
  shapiro,
  skewness,
  spearmanr,
  mean as statsMean,
  std,
  trimMean,
  ttest_1samp,
  ttest_ind,
  variance,
} from "../src/stats";

describe("docs/contents ndarray: tensor creation", () => {
  it("creates tensors from various inputs", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(a.shape).toEqual([2, 3]);
    expect(a.dtype).toBe("float32");

    const b = tensor([1, 2, 3], { dtype: "int32" });
    expect(b.dtype).toBe("int32");

    const c = tensor(new Float32Array([1, 2, 3]));
    expect(c.shape).toEqual([3]);

    const d = tensor(42);
    expect(d.shape).toEqual([]);
    expect(d.size).toBe(1);
  });

  it("factory functions", () => {
    expect(zeros([3, 3]).at(0, 0)).toBe(0);
    expect(ones([2, 2]).at(1, 1)).toBe(1);
    expect(eye(3).at(0, 0)).toBe(1);
    expect(eye(3).at(0, 1)).toBe(0);
    expect(full([2, 2], 7).at(0, 0)).toBe(7);
    expect(arange(0, 5).size).toBe(5);
    expect(linspace(0, 1, 5).size).toBe(5);
    expect(randn([2, 3]).shape).toEqual([2, 3]);
  });
});

describe("docs/contents ndarray: operations", () => {
  it("arithmetic with broadcasting", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    const b = tensor([
      [5, 6],
      [7, 8],
    ]);
    const c = add(a, b);
    expect(c.at(0, 0)).toBe(6);
    expect(c.at(1, 1)).toBe(12);

    const d = mul(a, tensor([10, 100]));
    expect(d.at(0, 0)).toBe(10);
    expect(d.at(0, 1)).toBe(200);
  });

  it("reductions", () => {
    const a = tensor([
      [1, 2],
      [3, 4],
    ]);
    expect(sum(a).at()).toBe(10);
    expect(mean(a).at()).toBeCloseTo(2.5);

    const rowSum = sum(a, 1);
    expect(rowSum.at(0)).toBe(3);
    expect(rowSum.at(1)).toBe(7);
  });

  it("matrix operations", () => {
    const A = tensor([
      [1, 2],
      [3, 4],
    ]);
    const B = tensor([
      [5, 6],
      [7, 8],
    ]);
    const C = dot(A, B);
    expect(C.at(0, 0)).toBe(19);
  });

  it("concatenate and stack", () => {
    const a = tensor([1, 2, 3]);
    const b = tensor([4, 5, 6]);
    const c = concatenate([a, b]);
    expect(c.size).toBe(6);

    const s = stack([a, b]);
    expect(s.shape).toEqual([2, 3]);
  });

  it("sort", () => {
    const sorted = sort(tensor([3, 1, 4, 1, 5]));
    expect(sorted.at(0)).toBe(1);
    expect(sorted.at(4)).toBe(5);
  });
});

describe("docs/contents ndarray: activations", () => {
  it("all activation functions", () => {
    const t = tensor([-2, -1, 0, 1, 2]);
    expect(relu(t).at(0)).toBe(0);
    expect(relu(t).at(3)).toBe(1);
    expect(sigmoid(t).at(2)).toBeCloseTo(0.5, 1);
    expect(gelu(t).size).toBe(5);
    expect(mish(t).size).toBe(5);
    expect(elu(t, 1.0).size).toBe(5);
    expect(leakyRelu(t, 0.01).size).toBe(5);
  });
});

describe("docs/contents ndarray: shape & indexing", () => {
  it("reshape, transpose, flatten", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(reshape(t, [3, 2]).shape).toEqual([3, 2]);
    expect(reshape(t, [6]).shape).toEqual([6]);
    expect(reshape(t, [-1]).shape).toEqual([6]);
    expect(transpose(t).shape).toEqual([3, 2]);
    expect(flatten(t).shape).toEqual([6]);
  });

  it("squeeze and unsqueeze", () => {
    const t = tensor([[1, 2, 3]]);
    expect(squeeze(t).shape).toEqual([3]);
    const u = unsqueeze(tensor([1, 2, 3]), 0);
    expect(u.shape).toEqual([1, 3]);
  });

  it("gather", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    const idx = tensor([0, 2], { dtype: "int32" });
    const g = gather(t, idx, 0);
    expect(g.shape[0]).toBe(2);
  });
});

describe("docs/contents ndarray: autograd", () => {
  it("gradient computation", () => {
    const x = parameter([2, 3]);
    const y = x.mul(x).sum();
    y.backward();
    expect(Number(y.tensor.data[0])).toBe(13);
    expect(x.grad).toBeDefined();
  });

  it("noGrad block", () => {
    const x = parameter([1, 2, 3]);
    const result = noGrad(() => x.mul(x).sum());
    expect(Number(result.tensor.data[0])).toBe(14);
  });
});

describe("docs/contents ndarray: sparse matrices", () => {
  it("CSRMatrix operations", () => {
    const sparse = CSRMatrix.fromCOO({
      rows: 3,
      cols: 3,
      rowIndices: new Int32Array([0, 1, 2]),
      colIndices: new Int32Array([0, 2, 1]),
      values: new Float64Array([1, 2, 3]),
    });
    expect(sparse.shape).toEqual([3, 3]);
    expect(sparse.nnz).toBe(3);

    const dense = sparse.toDense();
    expect(dense.at(0, 0)).toBe(1);
    expect(dense.at(1, 2)).toBe(2);

    const scaled = sparse.scale(2);
    expect(scaled.toDense().at(0, 0)).toBe(2);

    const v = tensor([1, 1, 1]);
    const result = sparse.matvec(v);
    expect(result.size).toBe(3);
  });
});

describe("docs/contents ml: supervised learning", () => {
  it("LinearRegression fit/predict", () => {
    const X = tensor([[1], [2], [3], [4], [5]]);
    const y = tensor([2, 4, 6, 8, 10]);
    const model = new LinearRegression();
    model.fit(X, y);
    const preds = model.predict(X);
    expect(r2Score(y, preds)).toBeGreaterThan(0.99);
  });

  it("LogisticRegression on Iris", () => {
    const iris = loadIris();
    const [X_tr, X_te, y_tr, y_te] = trainTestSplit(iris.data, iris.target, {
      testSize: 0.2,
      randomState: 42,
    });
    const scaler = new StandardScaler();
    scaler.fit(X_tr);
    const model = new LogisticRegression();
    model.fit(scaler.transform(X_tr), y_tr);
    const preds = model.predict(scaler.transform(X_te));
    expect(accuracy(y_te, preds)).toBeGreaterThan(0.9);
  });

  it("DecisionTree classifier and regressor", () => {
    const iris = loadIris();
    const dt = new DecisionTreeClassifier({ maxDepth: 5 });
    dt.fit(iris.data, iris.target);
    const preds = dt.predict(iris.data);
    expect(accuracy(iris.target, preds)).toBeGreaterThan(0.9);

    const X = tensor([[1], [2], [3], [4], [5]]);
    const y = tensor([1, 4, 9, 16, 25]);
    const dtr = new DecisionTreeRegressor({ maxDepth: 5 });
    dtr.fit(X, y);
    const rPreds = dtr.predict(X);
    expect(rPreds.shape[0]).toBe(5);
  });

  it("Ridge and Lasso", () => {
    const X = tensor([
      [1, 0],
      [0, 1],
      [1, 1],
      [2, 1],
      [1, 2],
    ]);
    const y = tensor([1, 2, 3, 4, 5]);
    const ridge = new Ridge({ alpha: 1.0 });
    ridge.fit(X, y);
    expect(ridge.predict(X).shape[0]).toBe(5);

    const lasso = new Lasso({ alpha: 0.01 });
    lasso.fit(X, y);
    expect(lasso.predict(X).shape[0]).toBe(5);
  });
});

describe("docs/contents ml: unsupervised learning", () => {
  it("KMeans clustering", () => {
    const [X] = makeBlobs({ nSamples: 100, centers: 3, randomState: 42 });
    const kmeans = new KMeans({ nClusters: 3, randomState: 42 });
    kmeans.fit(X);
    const labels = kmeans.predict(X);
    expect(labels.shape).toEqual([100]);
    expect(kmeans.inertia).toBeGreaterThan(0);
  });

  it("DBSCAN clustering", () => {
    const [X] = makeBlobs({ nSamples: 50, centers: 2, randomState: 42 });
    const dbscan = new DBSCAN({ eps: 1.5, minSamples: 3 });
    dbscan.fit(X);
    const labels = dbscan.labels;
    expect(labels.shape).toEqual([50]);
  });

  it("PCA dimensionality reduction", () => {
    const iris = loadIris();
    const pca = new PCA({ nComponents: 2 });
    pca.fit(iris.data);
    const projected = pca.transform(iris.data);
    expect(projected.shape).toEqual([150, 2]);
    expect(pca.explainedVarianceRatio.size).toBe(2);
  });
});

describe("docs/contents nn: layers and losses", () => {
  it("Sequential model", () => {
    const model = new Sequential(
      new Linear(4, 16),
      new ReLULayer(),
      new Dropout(0.1),
      new Linear(16, 8),
      new ReLULayer(),
      new Linear(8, 1)
    );
    expect([...model.parameters()].length).toBe(6);
    const out = model.forward(tensor([[1, 2, 3, 4]]));
    expect(out.shape).toEqual([1, 1]);
  });

  it("loss functions", () => {
    const pred = tensor([2.5, 0.0, 2.1]);
    const target = tensor([3.0, -0.5, 2.0]);
    const mseVal = mseLoss(pred, target);
    expect(mseVal.size).toBe(1);
    const maeVal = maeLoss(pred, target);
    expect(maeVal.size).toBe(1);
    const huberVal = huberLoss(pred, target, 1.0);
    expect(huberVal.size).toBe(1);
  });

  it("normalization layers", () => {
    const ln = new LayerNorm(4);
    const out = ln.forward(tensor([[1, 2, 3, 4]]));
    expect(out.shape).toEqual([1, 4]);

    const bn = new BatchNorm1d(4);
    const bnOut = bn.forward(
      tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ])
    );
    expect(bnOut.shape).toEqual([2, 4]);
  });

  it("recurrent layers", () => {
    const rnn = new RNN(4, 8);
    const rnnOut = rnn.forward(
      tensor([
        [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
        ],
      ])
    );
    expect(rnnOut.shape[2]).toBe(8);

    const lstm = new LSTM(4, 8);
    const lstmOut = lstm.forward(
      tensor([
        [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
        ],
      ])
    );
    expect(lstmOut.shape[2]).toBe(8);

    const gru = new GRU(4, 8);
    const gruOut = gru.forward(
      tensor([
        [
          [1, 2, 3, 4],
          [5, 6, 7, 8],
        ],
      ])
    );
    expect(gruOut.shape[2]).toBe(8);
  });

  it("attention and transformer", () => {
    const mha = new MultiheadAttention(8, 2);
    const q = tensor([[[1, 2, 3, 4, 5, 6, 7, 8]]]);
    const out = mha.forward(q, q, q);
    expect(out.shape).toEqual([1, 1, 8]);

    const tel = new TransformerEncoderLayer(8, 2, 16);
    const telOut = tel.forward(q);
    expect(telOut.shape).toEqual([1, 1, 8]);
  });
});

describe("docs/contents optim: optimizers and schedulers", () => {
  it("Adam and SGD optimizers", () => {
    const model = new Sequential(new Linear(2, 4), new ReLULayer(), new Linear(4, 1));
    const adam = new Adam(model.parameters(), { lr: 0.001 });
    expect(adam.lr).toBeCloseTo(0.001);

    const sgd = new SGD(model.parameters(), { lr: 0.01, momentum: 0.9 });
    expect(sgd.lr).toBeCloseTo(0.01);
  });

  it("schedulers", () => {
    const model = new Sequential(new Linear(2, 1));
    const opt = new SGD(model.parameters(), { lr: 0.1 });
    const sched = new StepLR(opt, { stepSize: 10, gamma: 0.1 });
    expect(opt.lr).toBeCloseTo(0.1);
    for (let i = 0; i < 11; i++) sched.step();
    expect(opt.lr).toBeCloseTo(0.01);
  });
});

describe("docs/contents preprocess: scalers", () => {
  it("StandardScaler fit/transform/inverseTransform", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const ss = new StandardScaler();
    ss.fit(X);
    const XStd = ss.transform(X);
    expect(XStd.shape).toEqual([4, 2]);
    const XOrig = ss.inverseTransform(XStd);
    expect(XOrig.at(0, 0)).toBeCloseTo(1, 0);
  });

  it("MinMaxScaler fitTransform", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
    ]);
    const mms = new MinMaxScaler();
    const XNorm = mms.fitTransform(X);
    expect(XNorm.shape).toEqual([4, 2]);
  });
});

describe("docs/contents preprocess: encoders", () => {
  it("LabelEncoder with plain arrays", () => {
    const le = new LabelEncoder();
    le.fit(["cat", "dog", "fish"]);
    const encoded = le.transform(["dog", "cat", "fish"]);
    expect(encoded.size).toBe(3);
    const decoded = le.inverseTransform(encoded);
    expect(decoded.size).toBe(3);
  });

  it("OneHotEncoder with plain arrays", () => {
    const ohe = new OneHotEncoder();
    ohe.fit([["red"], ["green"], ["blue"]]);
    const encoded = ohe.transform([["red"], ["blue"]]);
    expect(encoded.shape[0]).toBe(2);
    expect(encoded.shape[1]).toBe(3);
  });

  it("LabelBinarizer", () => {
    const lb = new LabelBinarizer();
    lb.fit(tensor(["cat", "dog", "fish"]));
    const binarized = lb.transform(tensor(["cat", "fish"]));
    expect(binarized.shape[0]).toBe(2);
  });
});

describe("docs/contents preprocess: splitting", () => {
  it("trainTestSplit", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
      [9, 10],
    ]);
    const y = tensor([0, 0, 1, 1, 1]);
    const [XTrain, XTest, _yTrain, _yTest] = trainTestSplit(X, y, {
      testSize: 0.2,
      randomState: 42,
    });
    expect(XTrain.shape[0] + XTest.shape[0]).toBe(5);
  });

  it("KFold with SplitResult objects", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
      [9, 10],
    ]);
    const kf = new KFold({ nSplits: 5, shuffle: true, randomState: 42 });
    let count = 0;
    for (const { trainIndex, testIndex } of kf.split(X)) {
      expect(trainIndex.length + testIndex.length).toBe(5);
      count++;
    }
    expect(count).toBe(5);
  });
});

describe("docs/contents stats: descriptive", () => {
  it("all descriptive stats", () => {
    const data = tensor([2, 4, 4, 4, 5, 5, 7, 9]);
    expect(statsMean(data).size).toBe(1);
    expect(median(data).size).toBe(1);
    expect(std(data).size).toBe(1);
    expect(variance(data).size).toBe(1);
    expect(skewness(data).size).toBe(1);
    expect(kurtosis(data).size).toBe(1);
    expect(percentile(data, 75).size).toBe(1);
    expect(geometricMean(tensor([1, 2, 4, 8])).size).toBe(1);
    expect(trimMean(data, 0.1).size).toBe(1);
  });
});

describe("docs/contents stats: correlations", () => {
  it("pearsonr, spearmanr, kendalltau", () => {
    const x = tensor([1, 2, 3, 4, 5]);
    const y = tensor([2, 4, 5, 4, 5]);
    const [r, _pval] = pearsonr(x, y);
    expect(r).toBeGreaterThan(0);
    const [rho] = spearmanr(x, y);
    expect(typeof rho).toBe("number");
    const [tau] = kendalltau(x, y);
    expect(typeof tau).toBe("number");
  });

  it("corrcoef and cov", () => {
    const x = tensor([1, 2, 3, 4, 5]);
    const y = tensor([2, 4, 5, 4, 5]);
    const corrMatrix = corrcoef(x, y);
    expect(corrMatrix.shape).toEqual([2, 2]);
    const covMatrix = cov(x, y);
    expect(covMatrix.shape).toEqual([2, 2]);
  });
});

describe("docs/contents stats: hypothesis tests", () => {
  it("ttest_1samp", () => {
    const sample = tensor([2.3, 1.9, 2.5, 2.1, 2.7]);
    const { statistic, pvalue } = ttest_1samp(sample, 0);
    expect(statistic).toBeGreaterThan(0);
    expect(pvalue).toBeLessThan(0.05);
  });

  it("ttest_ind", () => {
    const g1 = tensor([5.1, 4.9, 5.0, 5.2]);
    const g2 = tensor([4.5, 4.3, 4.6, 4.4]);
    const result = ttest_ind(g1, g2);
    expect(result.pvalue).toBeLessThan(0.05);
  });

  it("shapiro normality", () => {
    const data = tensor([1.2, 2.3, 1.8, 2.1, 1.9, 2.5, 2.0]);
    const result = shapiro(data);
    expect(result.pvalue).toBeGreaterThan(0.05);
  });
});

describe("docs/contents metrics: classification", () => {
  it("binary classification metrics", () => {
    const yTrue = tensor([0, 1, 1, 0, 1, 0, 1, 1]);
    const yPred = tensor([0, 1, 0, 0, 1, 1, 1, 1]);
    expect(accuracy(yTrue, yPred)).toBeCloseTo(0.75, 2);
    expect(typeof precision(yTrue, yPred)).toBe("number");
    expect(typeof recall(yTrue, yPred)).toBe("number");
    expect(typeof f1Score(yTrue, yPred)).toBe("number");
    expect(confusionMatrix(yTrue, yPred).shape).toEqual([2, 2]);
  });
});

describe("docs/contents metrics: regression", () => {
  it("regression metrics", () => {
    const yTrue = tensor([3.0, -0.5, 2.0, 7.0]);
    const yPred = tensor([2.5, 0.0, 2.1, 7.8]);
    expect(typeof mse(yTrue, yPred)).toBe("number");
    expect(typeof rmse(yTrue, yPred)).toBe("number");
    expect(typeof mae(yTrue, yPred)).toBe("number");
    expect(typeof r2Score(yTrue, yPred)).toBe("number");
  });
});

describe("docs/contents metrics: clustering", () => {
  it("clustering metrics", () => {
    const X = tensor([
      [1, 2],
      [1.5, 1.8],
      [5, 8],
      [8, 8],
      [1, 0.6],
      [9, 11],
    ]);
    const labels = tensor([0, 0, 1, 1, 0, 1]);
    expect(typeof silhouetteScore(X, labels)).toBe("number");
    expect(typeof daviesBouldinScore(X, labels)).toBe("number");

    const trueLabels = tensor([0, 0, 1, 1, 0, 1]);
    const predLabels = tensor([0, 0, 1, 1, 0, 2]);
    expect(typeof adjustedRandScore(trueLabels, predLabels)).toBe("number");
  });
});

describe("docs/contents datasets: built-in and synthetic", () => {
  it("built-in datasets", () => {
    const iris = loadIris();
    expect(iris.data.shape).toEqual([150, 4]);
    expect(iris.target.shape).toEqual([150]);

    const cancer = loadBreastCancer();
    expect(cancer.data.shape[0]).toBe(569);
  });

  it("synthetic generators", () => {
    const [X1, y1] = makeBlobs({ nSamples: 100, centers: 3, randomState: 42 });
    expect(X1.shape).toEqual([100, 2]);
    expect(y1.shape).toEqual([100]);

    const [X2, _y2] = makeCircles({ nSamples: 100, noise: 0.05, factor: 0.5 });
    expect(X2.shape).toEqual([100, 2]);

    const [X3, _y3] = makeMoons({ nSamples: 100, noise: 0.1 });
    expect(X3.shape).toEqual([100, 2]);

    const [X4, _y4] = makeRegression({ nSamples: 50, nFeatures: 3, noise: 0.1 });
    expect(X4.shape).toEqual([50, 3]);
  });

  it("DataLoader iteration", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
      [9, 10],
    ]);
    const y = tensor([0, 1, 0, 1, 0]);
    const loader = new DataLoader(X, y, { batchSize: 2, shuffle: false });
    let count = 0;
    for (const batch of loader) {
      const [batchX, batchY] = batch as [typeof X, typeof y];
      expect(batchX.shape[1]).toBe(2);
      expect(batchY.ndim).toBe(1);
      count++;
    }
    expect(count).toBe(3);
  });
});

describe("docs/contents random: distributions", () => {
  it("seeded random", () => {
    setSeed(42);
    const r = rand([3, 3]);
    expect(r.shape).toEqual([3, 3]);
    clearSeed();
  });

  it("parametric distributions", () => {
    setSeed(42);
    expect(uniform(0, 10, [5]).size).toBe(5);
    expect(normal(0, 1, [5]).size).toBe(5);
    expect(binomial(10, 0.5, [5]).size).toBe(5);
    expect(poisson(3, [5]).size).toBe(5);
    expect(randint(0, 10, [5]).size).toBe(5);
    clearSeed();
  });

  it("sampling utilities", () => {
    setSeed(42);
    const data = tensor([10, 20, 30, 40, 50]);
    expect(choice(data, 3).size).toBe(3);
    expect(permutation(5).size).toBe(5);
    clearSeed();
  });
});

describe("docs/contents plot: visualization", () => {
  it("line plot with axes", () => {
    const fig = new Figure({ width: 640, height: 480 });
    const ax = fig.addAxes();
    const x = linspace(0, 10, 50);
    ax.plot(x, mul(x, tensor(2)), { color: "blue" });
    ax.setTitle("y = 2x");
    ax.setXLabel("x");
    ax.setYLabel("y");
    const svg = fig.renderSVG();
    expect(svg.svg.length).toBeGreaterThan(0);
  });

  it("heatmap", () => {
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.heatmap(
      tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ])
    );
    ax.setTitle("Heatmap");
    const svg = fig.renderSVG();
    expect(svg.svg.length).toBeGreaterThan(0);
  });
});
