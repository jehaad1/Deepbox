/**
 * Tests derived from real Deepbox-Docs/src/data/examples.ts code snippets.
 * Each test corresponds to a documented example on the Deepbox website.
 */
import { describe, expect, it } from "vitest";
import { DataFrame } from "../src/dataframe";
import {
  loadBreastCancer,
  loadDiabetes,
  loadDigits,
  loadIris,
  makeBlobs,
  makeCircles,
  makeMoons,
  makeRegression,
} from "../src/datasets";
import { cond, det, eig, inv, lstsq, matrixRank, norm, solve, svd } from "../src/linalg";
import {
  accuracy,
  confusionMatrix,
  f1Score,
  mae,
  mse,
  precision,
  r2Score,
  recall,
  rmse,
} from "../src/metrics";
import {
  DecisionTreeClassifier,
  GaussianNB,
  KMeans,
  KNeighborsClassifier,
  Lasso,
  LinearRegression,
  LogisticRegression,
  PCA,
  RandomForestClassifier,
  Ridge,
} from "../src/ml";
import {
  add,
  arange,
  CSRMatrix,
  cos,
  dot,
  elu,
  eye,
  full,
  gelu,
  greater,
  leakyRelu,
  linspace,
  logSoftmax,
  max,
  mean,
  mish,
  noGrad,
  ones,
  parameter,
  randn,
  relu,
  sigmoid,
  sin,
  softmax,
  softplus,
  sort,
  sqrt,
  sum,
  swish,
  tensor,
  zeros,
} from "../src/ndarray";
import type { GradTensor } from "../src/ndarray/autograd";
import { Linear, ReLU, Sequential } from "../src/nn";
import { Adam, CosineAnnealingLR, SGD, StepLR } from "../src/optim";
import { Figure } from "../src/plot";
import {
  KFold,
  LabelEncoder,
  LeaveOneOut,
  MinMaxScaler,
  OneHotEncoder,
  OrdinalEncoder,
  RobustScaler,
  StandardScaler,
  StratifiedKFold,
  trainTestSplit,
} from "../src/preprocess";
import {
  binomial,
  choice,
  clearSeed,
  getSeed,
  normal,
  permutation,
  poisson,
  rand,
  randint,
  setSeed,
  uniform,
} from "../src/random";
import {
  kurtosis,
  median,
  pearsonr,
  percentile,
  shapiro,
  skewness,
  spearmanr,
  mean as statsMean,
  std as statsStd,
  ttest_1samp,
  variance,
} from "../src/stats";

describe("docs example 00: Quick Start Guide", () => {
  it("tensor arithmetic + mean", () => {
    const a = tensor([1, 2, 3, 4, 5]);
    const b = tensor([10, 20, 30, 40, 50]);
    const c = add(a, b);
    expect(c.shape).toEqual([5]);
    expect(c.at(0)).toBe(11);
    const m = mean(a);
    expect(m.size).toBe(1);
  });

  it("DataFrame creation", () => {
    const df = new DataFrame({
      name: ["Alice", "Bob", "Charlie"],
      age: [25, 30, 35],
      score: [85, 90, 78],
    });
    expect(df.shape).toEqual([3, 3]);
  });

  it("LinearRegression y = 2x + 1", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8]]);
    const y = tensor([3, 5, 7, 9, 11, 13, 15, 17]);
    const [X_train, X_test, y_train, _y_test] = trainTestSplit(X, y, {
      testSize: 0.25,
      randomState: 42,
    });
    const model = new LinearRegression();
    model.fit(X_train, y_train);
    const predictions = model.predict(X_test);
    expect(predictions.shape[0]).toBe(X_test.shape[0]);
  });
});

describe("docs example 01: Tensor Basics", () => {
  it("tensor properties", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(a.shape).toEqual([2, 3]);
    expect(a.dtype).toBe("float32");
    expect(a.size).toBe(6);
    expect(a.ndim).toBe(2);
    expect(a.at(0, 1)).toBe(2);
  });

  it("factory functions", () => {
    expect(zeros([3, 3]).size).toBe(9);
    expect(ones([2, 4]).size).toBe(8);
    expect(eye(3).shape).toEqual([3, 3]);
    expect(full([2, 2], 7).at(0, 0)).toBe(7);
    expect(arange(0, 10, 2).size).toBe(5);
    expect(linspace(0, 1, 5).size).toBe(5);
    expect(randn([2, 3]).shape).toEqual([2, 3]);
  });

  it("explicit dtypes", () => {
    const ints = tensor([1, 2, 3], { dtype: "int32" });
    expect(ints.dtype).toBe("int32");
    const bools = tensor([true, false, true]);
    expect(bools.dtype).toBe("bool");
  });
});

describe("docs example 02: Tensor Operations", () => {
  it("arithmetic and broadcasting", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = tensor([
      [7, 8, 9],
      [10, 11, 12],
    ]);
    const c = add(a, b);
    expect(c.at(0, 0)).toBe(8);

    const row = tensor([10, 20, 30]);
    const bc = add(a, row);
    expect(bc.at(0, 0)).toBe(11);
  });

  it("math functions", () => {
    const s = sqrt(tensor([4, 9, 16, 25]));
    expect(s.at(0)).toBeCloseTo(2);
  });

  it("reductions", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(sum(a).at()).toBe(21);
    expect(max(a).at()).toBe(6);
  });

  it("comparisons", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const g = greater(a, tensor(3));
    expect(g.at(0, 0)).toBe(0);
    expect(g.at(1, 0)).toBe(1);
  });

  it("matrix dot", () => {
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
    expect(C.at(0, 1)).toBe(22);
  });

  it("sort", () => {
    const s = sort(tensor([3, 1, 4, 1, 5]));
    expect(s.at(0)).toBe(1);
    expect(s.at(4)).toBe(5);
  });
});

describe("docs example 03: Data Analysis", () => {
  it("DataFrame groupBy and filter", () => {
    const df = new DataFrame({
      name: [
        "Alice",
        "Bob",
        "Charlie",
        "David",
        "Eve",
        "Frank",
        "Grace",
        "Henry",
        "Ivy",
        "Jack",
        "Kate",
        "Leo",
        "Mia",
        "Noah",
        "Olivia",
        "Paul",
        "Quinn",
        "Rachel",
        "Sam",
        "Tina",
      ],
      department: [
        "Engineering",
        "Sales",
        "Engineering",
        "HR",
        "Engineering",
        "Sales",
        "Marketing",
        "Engineering",
        "HR",
        "Sales",
        "Engineering",
        "Marketing",
        "Sales",
        "Engineering",
        "HR",
        "Sales",
        "Engineering",
        "Marketing",
        "Engineering",
        "Sales",
      ],
      salary: [
        95000, 65000, 105000, 55000, 98000, 72000, 68000, 110000, 58000, 70000, 102000, 71000,
        67000, 115000, 60000, 69000, 108000, 73000, 112000, 66000,
      ],
      experience: [5, 3, 8, 2, 6, 4, 3, 10, 2, 5, 7, 4, 3, 12, 3, 4, 9, 5, 11, 3],
    });

    const salaryTensor = tensor(df.get("salary").toArray() as number[]);
    const m = statsMean(salaryTensor);
    expect(Number(m.data[0])).toBeCloseTo(81950, -1);

    const highEarners = df.filter(
      (row: Record<string, unknown>) => (row.salary as number) > 100000
    );
    expect(highEarners.shape[0]).toBe(6);
  });
});

describe("docs example 04: DataFrame Basics", () => {
  it("select, filter, sort, head", () => {
    const df = new DataFrame({
      name: ["Alice", "Bob", "Charlie", "David", "Eve"],
      age: [25, 30, 35, 28, 32],
      salary: [50000, 60000, 75000, 55000, 70000],
      dept: ["Engineering", "Sales", "Engineering", "HR", "Sales"],
    });
    expect(df.shape).toEqual([5, 4]);
    expect(df.columns).toEqual(["name", "age", "salary", "dept"]);

    const selected = df.select(["name", "salary"]);
    expect(selected.columns).toEqual(["name", "salary"]);

    const engineers = df.filter((row: Record<string, unknown>) => row.dept === "Engineering");
    expect(engineers.shape[0]).toBe(2);

    const sorted = df.sort("salary", false);
    expect(sorted.shape).toEqual([5, 4]);

    const head = df.head(2);
    expect(head.shape[0]).toBe(2);
  });
});

describe("docs example 05: DataFrame GroupBy", () => {
  it("groupBy aggregation", () => {
    const df = new DataFrame({
      dept: ["Eng", "Sales", "Eng", "HR", "Sales", "Eng", "HR", "Sales"],
      salary: [90000, 60000, 95000, 50000, 65000, 100000, 55000, 70000],
      years: [5, 3, 8, 2, 4, 10, 3, 6],
    });
    const grouped = df.groupBy("dept").agg({
      salary: "mean",
      years: "max",
    });
    expect(grouped.shape[0]).toBe(3);

    const counts = df.groupBy("dept").agg({ salary: "count" });
    expect(counts.shape[0]).toBe(3);
  });
});

describe("docs example 06: ML Pipeline (Iris)", () => {
  it("multi-model comparison with f1Score macro", () => {
    const iris = loadIris();
    expect(iris.data.shape).toEqual([150, 4]);

    const [X_tr, X_te, y_tr, y_te] = trainTestSplit(iris.data, iris.target, {
      testSize: 0.2,
      randomState: 42,
    });
    const scaler = new StandardScaler();
    scaler.fit(X_tr);
    const X_train = scaler.transform(X_tr);
    const X_test = scaler.transform(X_te);

    const models = [
      { name: "LogReg", model: new LogisticRegression() },
      { name: "DT", model: new DecisionTreeClassifier() },
      { name: "RF", model: new RandomForestClassifier() },
      { name: "KNN", model: new KNeighborsClassifier({ nNeighbors: 5 }) },
      { name: "GNB", model: new GaussianNB() },
    ];

    for (const { model } of models) {
      model.fit(X_train, y_tr);
      const preds = model.predict(X_test);
      const acc = accuracy(y_te, preds);
      const f1 = f1Score(y_te, preds, "macro");
      expect(acc).toBeGreaterThan(0.8);
      expect(f1).toBeGreaterThan(0.8);
    }
  });
});

describe("docs example 07: Linear Regression", () => {
  it("y = 3x + 2 regression", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);
    const y = tensor([5.1, 8.0, 10.9, 14.1, 17.0, 19.8, 23.1, 25.9, 29.0, 32.1]);
    const [X_train, X_test, y_train, y_test] = trainTestSplit(X, y, {
      testSize: 0.3,
      randomState: 42,
    });
    const model = new LinearRegression();
    model.fit(X_train, y_train);
    const preds = model.predict(X_test);
    const r2 = r2Score(y_test, preds);
    expect(r2).toBeGreaterThan(0.99);
  });
});

describe("docs example 09: Ridge & Lasso", () => {
  it("Ridge and Lasso on diabetes", () => {
    const data = loadDiabetes();
    const [X_tr, X_te, y_tr, y_te] = trainTestSplit(data.data, data.target, {
      testSize: 0.2,
      randomState: 42,
    });
    const scaler = new StandardScaler();
    scaler.fit(X_tr);
    const X_train = scaler.transform(X_tr);
    const X_test = scaler.transform(X_te);

    const ridge = new Ridge({ alpha: 1.0 });
    ridge.fit(X_train, y_tr);
    const ridgePreds = ridge.predict(X_test);
    expect(typeof r2Score(y_te, ridgePreds)).toBe("number");

    const lasso = new Lasso({ alpha: 1.0 });
    lasso.fit(X_train, y_tr);
    const lassoPreds = lasso.predict(X_test);
    expect(typeof r2Score(y_te, lassoPreds)).toBe("number");
  });
});

describe("docs example 10: Advanced ML Models", () => {
  it("KMeans clustering", () => {
    const X_cluster = tensor([
      [1, 2],
      [1.5, 1.8],
      [5, 8],
      [8, 8],
      [1, 0.6],
      [9, 11],
      [8, 2],
      [10, 2],
      [9, 3],
    ]);
    const kmeans = new KMeans({ nClusters: 3, randomState: 42 });
    kmeans.fit(X_cluster);
    const labels = kmeans.predict(X_cluster);
    expect(labels.shape).toEqual([9]);
    expect(kmeans.inertia).toBeGreaterThan(0);
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

describe("docs example 11: Tree & Ensemble Models", () => {
  it("4 classifiers on Iris", () => {
    const iris = loadIris();
    const [X_tr, X_te, y_tr, y_te] = trainTestSplit(iris.data, iris.target, {
      testSize: 0.2,
      randomState: 42,
    });
    const scaler = new StandardScaler();
    scaler.fit(X_tr);
    const X_train = scaler.transform(X_tr);
    const X_test = scaler.transform(X_te);

    const models = [
      new DecisionTreeClassifier({ maxDepth: 5 }),
      new RandomForestClassifier({ nEstimators: 50 }),
    ];
    for (const m of models) {
      m.fit(X_train, y_tr);
      const preds = m.predict(X_test);
      expect(accuracy(y_te, preds)).toBeGreaterThan(0.8);
      expect(f1Score(y_te, preds, { average: "macro" })).toBeGreaterThan(0.8);
    }
  });
});

describe("docs example 13: Neural Network Training", () => {
  it("Sequential model forward pass", () => {
    const model = new Sequential(new Linear(2, 16), new ReLU(), new Linear(16, 1));
    expect([...model.parameters()].length).toBe(4);

    const X = tensor([
      [1, 0],
      [0, 1],
      [1, 1],
    ]);
    const out = model.forward(X);
    expect(out.shape).toEqual([3, 1]);
  });

  it("training with autograd", () => {
    const model = new Sequential(new Linear(2, 16), new ReLU(), new Linear(16, 1));
    const X = parameter([
      [1, 0],
      [0, 1],
      [1, 1],
      [2, 1],
    ]);
    const yTargets = parameter([[1], [2], [3], [4]]);
    const optimizer = new Adam(model.parameters(), { lr: 0.01 });

    let lastLoss = Infinity;
    for (let epoch = 0; epoch < 50; epoch++) {
      const pred = model.forward(X) as GradTensor;
      const diff = pred.sub(yTargets);
      const loss = diff.mul(diff).mean();
      optimizer.zeroGrad();
      loss.backward();
      optimizer.step();
      lastLoss = Number(loss.tensor.data[0]);
    }
    expect(lastLoss).toBeLessThan(10);
  });
});

describe("docs example 14: Autograd", () => {
  it("simple gradient y = x^2", () => {
    const x = parameter([2, 3]);
    const y = x.mul(x).sum();
    y.backward();
    expect(Number(y.tensor.data[0])).toBe(13);
    expect(x.grad).toBeDefined();
  });

  it("noGrad disables tracking", () => {
    const x = parameter([2, 3]);
    const result = noGrad(() => x.mul(x).sum());
    expect(Number(result.tensor.data[0])).toBe(13);
  });
});

describe("docs example 15: Activation Functions", () => {
  it("all activations", () => {
    const t = tensor([-2, -1, 0, 1, 2]);
    expect(relu(t).at(0)).toBe(0);
    expect(relu(t).at(4)).toBe(2);
    expect(sigmoid(t).at(2)).toBeCloseTo(0.5, 1);
    expect(gelu(t).size).toBe(5);
    expect(mish(t).size).toBe(5);
    expect(swish(t).size).toBe(5);
    expect(elu(t, 1.0).size).toBe(5);
    expect(leakyRelu(t, 0.1).at(0)).toBeCloseTo(-0.2, 1);
    expect(softplus(t).size).toBe(5);
  });

  it("softmax and logSoftmax", () => {
    const logits = tensor([[2.0, 1.0, 0.1]]);
    const sm = softmax(logits, 1);
    expect(sm.shape).toEqual([1, 3]);
    const ls = logSoftmax(logits, 1);
    expect(ls.shape).toEqual([1, 3]);
  });
});

describe("docs example 16: LR Schedulers", () => {
  it("StepLR decay", () => {
    const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
    const opt = new SGD(model.parameters(), { lr: 0.1 });
    const sched = new StepLR(opt, { stepSize: 5, gamma: 0.5 });
    expect(opt.lr).toBeCloseTo(0.1);
    for (let i = 0; i < 6; i++) sched.step();
    expect(opt.lr).toBeCloseTo(0.05);
  });

  it("CosineAnnealingLR", () => {
    const model = new Sequential(new Linear(2, 8), new ReLU(), new Linear(8, 1));
    const opt = new SGD(model.parameters(), { lr: 0.1 });
    const sched = new CosineAnnealingLR(opt, { tMax: 20, etaMin: 0.001 });
    expect(opt.lr).toBeCloseTo(0.1);
    for (let i = 0; i < 10; i++) sched.step();
    expect(opt.lr).toBeLessThan(0.1);
    expect(opt.lr).toBeGreaterThan(0.001);
  });
});

describe("docs example 17: Preprocessing Encoders", () => {
  it("LabelEncoder", () => {
    const le = new LabelEncoder();
    le.fit(tensor(["cat", "dog", "bird", "cat", "dog"]));
    const encoded = le.transform(tensor(["cat", "bird", "dog"]));
    expect(encoded.size).toBe(3);
    const decoded = le.inverseTransform(encoded);
    expect(decoded.size).toBe(3);
  });

  it("OneHotEncoder", () => {
    const ohe = new OneHotEncoder();
    ohe.fit(tensor([["red"], ["green"], ["blue"], ["red"]]));
    const onehot = ohe.transform(tensor([["red"], ["blue"], ["green"]]));
    expect(onehot.shape[0]).toBe(3);
    expect(onehot.shape[1]).toBe(3);
  });

  it("OrdinalEncoder", () => {
    const oe = new OrdinalEncoder();
    oe.fit(tensor([["small"], ["medium"], ["large"]]));
    const out = oe.transform(tensor([["large"], ["small"]]));
    expect(out.shape[0]).toBe(2);
  });
});

describe("docs example 18: Preprocessing Scalers", () => {
  it("StandardScaler zero mean unit var", () => {
    const X = tensor([
      [1, 100],
      [2, 200],
      [3, 300],
      [4, 400],
      [5, 500],
    ]);
    const ss = new StandardScaler();
    ss.fit(X);
    const X_std = ss.transform(X);
    const m = statsMean(X_std, 0);
    expect(Math.abs(Number(m.data[0]))).toBeLessThan(0.01);
  });

  it("MinMaxScaler", () => {
    const X = tensor([
      [1, 100],
      [2, 200],
      [3, 300],
      [4, 400],
      [5, 500],
    ]);
    const mm = new MinMaxScaler();
    mm.fit(X);
    const X_mm = mm.transform(X);
    expect(X_mm.shape).toEqual([5, 2]);
  });

  it("RobustScaler", () => {
    const X = tensor([
      [1, 100],
      [2, 200],
      [3, 300],
      [4, 400],
      [5, 500],
    ]);
    const rs = new RobustScaler();
    rs.fit(X);
    const X_rs = rs.transform(X);
    expect(X_rs.shape).toEqual([5, 2]);
  });
});

describe("docs example 19: Statistics", () => {
  it("descriptive stats", () => {
    const data = tensor([2, 4, 4, 4, 5, 5, 7, 9]);
    expect(statsMean(data).size).toBe(1);
    expect(median(data).size).toBe(1);
    expect(statsStd(data).size).toBe(1);
    expect(variance(data).size).toBe(1);
    expect(skewness(data).size).toBe(1);
    expect(kurtosis(data).size).toBe(1);
    expect(percentile(data, 75).size).toBe(1);
  });

  it("correlation", () => {
    const x = tensor([1, 2, 3, 4, 5]);
    const y = tensor([2, 4, 5, 4, 5]);
    const [r, pval] = pearsonr(x, y);
    expect(r).toBeGreaterThan(0);
    expect(pval).toBeGreaterThan(0);
    const [rho] = spearmanr(x, y);
    expect(typeof rho).toBe("number");
  });

  it("hypothesis tests", () => {
    const sample = tensor([2.3, 1.9, 2.5, 2.1, 2.7]);
    const { statistic, pvalue } = ttest_1samp(sample, 0);
    expect(statistic).toBeGreaterThan(0);
    expect(pvalue).toBeLessThan(0.05);
  });

  it("Shapiro normality test", () => {
    const norm = shapiro(tensor([1.2, 2.3, 1.8, 2.1, 1.9, 2.5, 2.0]));
    expect(norm.pvalue).toBeGreaterThan(0.05);
  });
});

describe("docs example 20: Linear Algebra", () => {
  it("solve Ax = b", () => {
    const A = tensor([
      [2, 1],
      [1, 3],
    ]);
    const b = tensor([5, 7]);
    const x = solve(A, b);
    expect(x.at(0)).toBeCloseTo(1.6, 1);
    expect(x.at(1)).toBeCloseTo(1.8, 1);
  });

  it("inverse", () => {
    const A = tensor([
      [2, 1],
      [1, 3],
    ]);
    const Ainv = inv(A);
    const I = dot(A, Ainv);
    expect(I.at(0, 0)).toBeCloseTo(1, 5);
    expect(I.at(0, 1)).toBeCloseTo(0, 5);
  });

  it("determinant and properties", () => {
    const A = tensor([
      [2, 1],
      [1, 3],
    ]);
    expect(det(A)).toBeCloseTo(5, 1);
    expect(matrixRank(A)).toBe(2);
    expect(typeof cond(A)).toBe("number");
    expect(typeof norm(A)).not.toBe("undefined");
  });

  it("SVD", () => {
    const A = tensor([
      [2, 1],
      [1, 3],
    ]);
    const [_U, S, _Vt] = svd(A);
    expect(S.size).toBe(2);
  });

  it("eigenvalues", () => {
    const A = tensor([
      [2, 1],
      [1, 3],
    ]);
    const [eigenvalues] = eig(A);
    expect(eigenvalues.size).toBe(2);
  });

  it("least squares", () => {
    const Aover = tensor([
      [1, 1],
      [1, 2],
      [1, 3],
    ]);
    const bover = tensor([1, 2, 2]);
    const { x: lsSolution } = lstsq(Aover, bover);
    expect(lsSolution.size).toBe(2);
  });
});

describe("docs example 21: Random Sampling", () => {
  it("seeded reproducibility", () => {
    setSeed(42);
    expect(getSeed()).toBe(42);
    const r1 = rand([3]);
    expect(r1.size).toBe(3);
    clearSeed();
    expect(getSeed()).toBeUndefined();
  });

  it("distributions", () => {
    setSeed(42);
    expect(uniform(0, 10, [3]).size).toBe(3);
    expect(normal(5, 2, [3]).size).toBe(3);
    expect(binomial(10, 0.5, [5]).size).toBe(5);
    expect(poisson(3, [5]).size).toBe(5);
    expect(randint(0, 10, [5]).size).toBe(5);
    clearSeed();
  });

  it("choice and permutation", () => {
    setSeed(42);
    const data = tensor([10, 20, 30, 40, 50]);
    const c = choice(data, 3);
    expect(c.size).toBe(3);
    const p = permutation(5);
    expect(p.size).toBe(5);
    clearSeed();
  });
});

describe("docs example 22: Built-in Datasets", () => {
  it("classic datasets", () => {
    const iris = loadIris();
    expect(iris.data.shape).toEqual([150, 4]);
    expect(iris.featureNames.length).toBe(4);
    expect(iris.targetNames?.length).toBe(3);

    const digits = loadDigits();
    expect(digits.data.shape).toEqual([1797, 64]);

    const cancer = loadBreastCancer();
    expect(cancer.data.shape[0]).toBe(569);

    const diabetes = loadDiabetes();
    expect(diabetes.data.shape[1]).toBe(10);
  });

  it("synthetic generators", () => {
    const [blobsX, blobsY] = makeBlobs({
      nSamples: 300,
      centers: 3,
      randomState: 42,
    });
    expect(blobsX.shape).toEqual([300, 2]);
    expect(blobsY.shape).toEqual([300]);

    const [circlesX] = makeCircles({
      nSamples: 200,
      noise: 0.05,
      factor: 0.5,
    });
    expect(circlesX.shape).toEqual([200, 2]);

    const [moonsX] = makeMoons({ nSamples: 200, noise: 0.1 });
    expect(moonsX.shape).toEqual([200, 2]);

    const [regX, regY] = makeRegression({
      nSamples: 100,
      nFeatures: 5,
      noise: 0.1,
    });
    expect(regX.shape).toEqual([100, 5]);
    expect(regY.shape).toEqual([100]);
  });
});

describe("docs example 23: Cross-Validation", () => {
  it("KFold split", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);
    const kf = new KFold({ nSplits: 5, shuffle: true, randomState: 42 });
    let count = 0;
    for (const { trainIndex, testIndex } of kf.split(X)) {
      expect(trainIndex.length).toBe(8);
      expect(testIndex.length).toBe(2);
      count++;
    }
    expect(count).toBe(5);
  });

  it("StratifiedKFold split", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);
    const y = tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
    const skf = new StratifiedKFold({
      nSplits: 2,
      shuffle: true,
      randomState: 42,
    });
    let count = 0;
    for (const { trainIndex, testIndex } of skf.split(X, y)) {
      expect(trainIndex.length + testIndex.length).toBe(10);
      count++;
    }
    expect(count).toBe(2);
  });

  it("LeaveOneOut", () => {
    const X = tensor([[1], [2], [3], [4], [5]]);
    const loo = new LeaveOneOut();
    let count = 0;
    for (const { testIndex } of loo.split(X)) {
      expect(testIndex.length).toBe(1);
      count++;
    }
    expect(count).toBe(5);
  });
});

describe("docs example 24: Metrics", () => {
  it("classification metrics", () => {
    const yTrue = tensor([0, 1, 1, 0, 1, 0, 1, 1]);
    const yPred = tensor([0, 1, 0, 0, 1, 1, 1, 1]);
    expect(accuracy(yTrue, yPred)).toBeCloseTo(0.75, 2);
    expect(precision(yTrue, yPred)).toBeCloseTo(0.8, 1);
    expect(recall(yTrue, yPred)).toBeCloseTo(0.8, 1);
    expect(f1Score(yTrue, yPred)).toBeCloseTo(0.8, 1);
    expect(confusionMatrix(yTrue, yPred).shape).toEqual([2, 2]);
  });

  it("regression metrics", () => {
    const yTrueReg = tensor([3.0, -0.5, 2.0, 7.0]);
    const yPredReg = tensor([2.5, 0.0, 2.1, 7.8]);
    expect(mse(yTrueReg, yPredReg)).toBeCloseTo(0.2875, 3);
    expect(rmse(yTrueReg, yPredReg)).toBeCloseTo(0.5362, 3);
    expect(mae(yTrueReg, yPredReg)).toBeCloseTo(0.475, 3);
    expect(r2Score(yTrueReg, yPredReg)).toBeGreaterThan(0.95);
  });
});

describe("docs example 25: Plotting", () => {
  it("Figure + Axes creation", () => {
    const x = linspace(0, 2 * Math.PI, 100);
    const fig = new Figure({ width: 640, height: 480 });
    const ax = fig.addAxes();
    ax.plot(x, sin(x), { color: "#1f77b4", linewidth: 2 });
    ax.plot(x, cos(x), { color: "#ff7f0e", linewidth: 2 });
    ax.setTitle("Sine and Cosine Functions");
    const svg = fig.renderSVG();
    expect(svg.svg.length).toBeGreaterThan(0);
  });

  it("scatter, bar, hist, heatmap", () => {
    const fig2 = new Figure();
    const ax2 = fig2.addAxes();
    ax2.scatter(tensor([1, 2, 3, 4, 5]), tensor([2, 4, 6, 8, 10]), {
      color: "#2ca02c",
    });
    ax2.setTitle("Test Scatter");

    const fig3 = new Figure();
    const ax3 = fig3.addAxes();
    ax3.bar(tensor([0, 1, 2]), tensor([10, 20, 30]), { color: "#d62728" });

    const fig4 = new Figure();
    const ax4 = fig4.addAxes();
    ax4.hist(tensor([1, 2, 2, 3, 3, 3, 4, 4, 5]), 5);

    const fig5 = new Figure();
    const ax5 = fig5.addAxes();
    ax5.heatmap(
      tensor([
        [1, 2, 3],
        [4, 5, 6],
      ])
    );
  });
});

describe("docs example 26: Sparse Matrices", () => {
  it("CSRMatrix from COO", () => {
    const sparse = CSRMatrix.fromCOO({
      rows: 4,
      cols: 4,
      rowIndices: new Int32Array([0, 0, 1, 2, 3, 3]),
      colIndices: new Int32Array([0, 2, 1, 2, 0, 3]),
      values: new Float64Array([1, 2, 3, 4, 5, 6]),
    });
    expect(sparse.shape).toEqual([4, 4]);
    expect(sparse.nnz).toBe(6);

    const dense = sparse.toDense();
    expect(dense.shape).toEqual([4, 4]);

    const scaled = sparse.scale(2);
    expect(scaled.nnz).toBe(6);

    const v = tensor([1, 2, 3, 4]);
    const result = sparse.matvec(v);
    expect(result.size).toBe(4);

    expect(sparse.transpose().nnz).toBe(6);
  });
});

describe("docs example 12: Complete ML Pipeline", () => {
  it("Iris classification with f1Score macro + KFold", () => {
    const iris = loadIris();
    const [X_tr, X_te, y_tr, y_te] = trainTestSplit(iris.data, iris.target, {
      testSize: 0.2,
      randomState: 42,
    });
    const scaler = new StandardScaler();
    scaler.fit(X_tr);
    const X_train = scaler.transform(X_tr);
    const X_test = scaler.transform(X_te);

    const model = new LogisticRegression();
    model.fit(X_train, y_tr);
    const preds = model.predict(X_test);

    expect(accuracy(y_te, preds)).toBeGreaterThan(0.9);
    expect(f1Score(y_te, preds, "macro")).toBeGreaterThan(0.9);
    expect(confusionMatrix(y_te, preds).shape).toEqual([3, 3]);

    const kf = new KFold({ nSplits: 5, shuffle: true, randomState: 42 });
    let foldCount = 0;
    for (const { trainIndex, testIndex } of kf.split(iris.data)) {
      expect(trainIndex.length + testIndex.length).toBe(150);
      foldCount++;
    }
    expect(foldCount).toBe(5);
  });
});
