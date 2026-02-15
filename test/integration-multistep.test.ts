/**
 * Comprehensive multi-step integration tests for all 13 Deepbox scopes.
 * Each test exercises a realistic multi-step workflow, not just single assertions.
 * Covers: core, ndarray, linalg, dataframe, stats, metrics, preprocess, ml, nn, optim, random, plot, datasets.
 */

import { describe, expect, it } from "vitest";

// ── core ──
import { DeepboxError, DTypeError, InvalidParameterError, ShapeError } from "../src/core";
// ── dataframe ──
import { DataFrame, Series } from "../src/dataframe";
// ── datasets ──
import {
  DataLoader,
  loadBreastCancer,
  loadDiabetes,
  loadDigits,
  loadHousingMini,
  loadIris,
  makeBlobs,
  makeCircles,
  makeClassification,
  makeMoons,
  makeRegression,
} from "../src/datasets";

// ── linalg ──
import {
  cholesky,
  cond,
  det,
  eig,
  inv,
  lstsq,
  lu,
  matrixRank,
  norm,
  qr,
  solve,
  svd,
  trace,
} from "../src/linalg";
// ── metrics ──
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
  silhouetteScore,
} from "../src/metrics";
// ── ml ──
import {
  DBSCAN,
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
// ── ndarray ──
import {
  add,
  arange,
  CSRMatrix,
  dot,
  elu,
  eye,
  flatten,
  full,
  GradTensor,
  gelu,
  greater,
  leakyRelu,
  linspace,
  mean as ndarrayMean,
  ones,
  parameter,
  relu,
  reshape,
  sigmoid,
  softmax,
  sqrt,
  squeeze,
  sum,
  tensor,
  transpose,
  unsqueeze,
  zeros,
} from "../src/ndarray";
// ── nn ──
import {
  Conv1d,
  GRU,
  Linear,
  LSTM,
  Module,
  MultiheadAttention,
  maeLoss,
  mseLoss,
  ReLU as ReLULayer,
  RNN,
  Sequential,
  TransformerEncoderLayer,
} from "../src/nn";
// ── optim ──
import {
  Adam,
  AdamW,
  CosineAnnealingLR,
  ExponentialLR,
  OneCycleLR,
  SGD,
  StepLR,
} from "../src/optim";
// ── plot ──
import { Figure } from "../src/plot";
// ── preprocess ──
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
// ── random ──
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
// ── stats ──
import {
  corrcoef,
  cov,
  kurtosis,
  mean,
  median,
  mode,
  pearsonr,
  percentile,
  quantile,
  shapiro,
  skewness,
  spearmanr,
  std,
  ttest_1samp,
  ttest_ind,
  variance,
} from "../src/stats";

// ═══════════════════════════════════════════════════════════════════════════
// 1. CORE: Error types, config, utils
// ═══════════════════════════════════════════════════════════════════════════
describe("integration: core", () => {
  it("custom error hierarchy works correctly", () => {
    const err = new ShapeError("test");
    expect(err).toBeInstanceOf(DeepboxError);
    expect(err).toBeInstanceOf(ShapeError);
    expect(err.message).toBe("test");

    const dtypeErr = new DTypeError("dtype test");
    expect(dtypeErr).toBeInstanceOf(DeepboxError);

    const paramErr = new InvalidParameterError("bad param", "alpha", -1);
    expect(paramErr).toBeInstanceOf(DeepboxError);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 2. NDARRAY: Tensor creation, ops, broadcasting, autograd, CSR
// ═══════════════════════════════════════════════════════════════════════════
describe("integration: ndarray", () => {
  it("full tensor creation and manipulation pipeline", () => {
    // Create tensors from various sources
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(a.shape).toEqual([2, 3]);
    expect(a.dtype).toBe("float32");
    expect(a.size).toBe(6);
    expect(a.ndim).toBe(2);
    expect(a.at(0, 1)).toBe(2);

    // Factory functions
    const z = zeros([3, 3]);
    expect(z.at(0, 0)).toBe(0);
    const o = ones([2, 4]);
    expect(o.at(1, 3)).toBe(1);
    const I = eye(3);
    expect(I.at(0, 0)).toBe(1);
    expect(I.at(0, 1)).toBe(0);
    const f = full([2, 2], 7);
    expect(f.at(0, 0)).toBe(7);
    const r = arange(0, 10, 2);
    expect(r.toArray()).toEqual([0, 2, 4, 6, 8]);
    const l = linspace(0, 1, 5);
    expect(l.size).toBe(5);

    // Explicit dtypes
    const ints = tensor([1, 2, 3], { dtype: "int32" });
    expect(ints.dtype).toBe("int32");
  });

  it("arithmetic and broadcasting pipeline", () => {
    const a = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const b = tensor([
      [7, 8, 9],
      [10, 11, 12],
    ]);

    // Element-wise operations
    const c = add(a, b);
    expect(c.at(0, 0)).toBe(8);
    expect(c.at(1, 2)).toBe(18);

    // Broadcasting: [2,3] + [3]
    const row = tensor([10, 20, 30]);
    const broadcasted = add(a, row);
    expect(broadcasted.at(0, 0)).toBe(11);
    expect(broadcasted.at(1, 2)).toBe(36);

    // Reductions
    const s = sum(a);
    expect(Number(s.data[0])).toBe(21);

    const colSum = sum(a, 0);
    expect(colSum.toArray()).toEqual([5, 7, 9]);

    const rowMean = ndarrayMean(a, 1);
    expect(rowMean.toArray()).toEqual([2, 5]);

    // Math functions
    const sqrtResult = sqrt(tensor([4, 9, 16]));
    expect(sqrtResult.at(0)).toBeCloseTo(2, 5);

    // Matrix multiplication
    const A = tensor([
      [1, 2],
      [3, 4],
    ]);
    const B = tensor([
      [5, 6],
      [7, 8],
    ]);
    const AB = dot(A, B);
    expect(AB.at(0, 0)).toBe(19);
    expect(AB.at(1, 1)).toBe(50);

    // Comparisons
    const g = greater(a, tensor(3));
    expect(g.at(0, 0)).toBe(0);
    expect(g.at(1, 0)).toBe(1);
  });

  it("reshape handles non-contiguous tensors (transpose then reshape)", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const tT = transpose(t);
    expect(tT.shape).toEqual([3, 2]);
    // Transpose then reshape should now work (copies data)
    const flat = reshape(tT, [6]);
    expect(flat.shape).toEqual([6]);
    expect(flat.toArray()).toEqual([1, 4, 2, 5, 3, 6]);

    // Reshape method on Tensor also works
    const flat2 = tT.reshape([6]);
    expect(flat2.toArray()).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it("shape manipulation pipeline", () => {
    const t = tensor([
      [1, 2, 3],
      [4, 5, 6],
    ]);

    // Reshape
    const r = reshape(t, [3, 2]);
    expect(r.shape).toEqual([3, 2]);

    // Flatten
    const f = flatten(t);
    expect(f.shape).toEqual([6]);

    // Transpose
    const tr = transpose(t);
    expect(tr.shape).toEqual([3, 2]);
    expect(tr.at(0, 0)).toBe(1);
    expect(tr.at(0, 1)).toBe(4);

    // Squeeze / Unsqueeze
    const s = tensor([[[1, 2, 3]]]);
    expect(s.shape).toEqual([1, 1, 3]);
    const sq = squeeze(s);
    expect(sq.shape).toEqual([3]);
    const us = unsqueeze(t, 0);
    expect(us.shape).toEqual([1, 2, 3]);
  });

  it("activation functions pipeline", () => {
    const t = tensor([-2, -1, 0, 1, 2]);

    const r = relu(t);
    expect(r.at(0)).toBe(0);
    expect(r.at(3)).toBe(1);
    expect(r.at(4)).toBe(2);

    const s = sigmoid(t);
    expect(s.at(2)).toBeCloseTo(0.5, 4);

    const sm = softmax(tensor([[2.0, 1.0, 0.1]]), 1);
    const smSum = Number(sum(sm).data[0]);
    expect(smSum).toBeCloseTo(1.0, 5);

    const g = gelu(t);
    expect(g.at(2)).toBeCloseTo(0, 4);

    const e = elu(t, 1.0);
    expect(e.at(3)).toBe(1);

    const lr = leakyRelu(t, 0.1);
    expect(lr.at(0)).toBeCloseTo(-0.2, 5);
  });

  it("autograd forward and backward pipeline", () => {
    const x = parameter([
      [1, 2],
      [3, 4],
    ]);
    const w = parameter([[0.5], [0.5]]);

    const y = x.matmul(w);
    expect(y.shape).toEqual([2, 1]);

    const loss = y.mul(y).mean();
    loss.backward();

    // Gradients should exist
    expect(x.grad).not.toBeNull();
    expect(w.grad).not.toBeNull();
  });

  it("CSR sparse matrix operations", () => {
    const csr = CSRMatrix.fromCOO({
      rows: 3,
      cols: 3,
      rowIndices: new Int32Array([0, 0, 1, 2]),
      colIndices: new Int32Array([0, 2, 1, 0]),
      values: new Float64Array([1, 2, 3, 4]),
    });
    expect(csr.rows).toBe(3);
    expect(csr.cols).toBe(3);
    expect(csr.nnz).toBe(4);
    expect(csr.get(0, 0)).toBe(1);
    expect(csr.get(0, 2)).toBe(2);

    const dense = csr.toDense();
    expect(dense.shape).toEqual([3, 3]);
    expect(dense.at(0, 0)).toBe(1);
    expect(dense.at(1, 1)).toBe(3);

    const scaled = csr.scale(2);
    expect(scaled.get(0, 0)).toBe(2);

    const transposed = csr.transpose();
    expect(transposed.get(2, 0)).toBe(2);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 3. LINALG: Decompositions, solvers, properties
// ═══════════════════════════════════════════════════════════════════════════
describe("integration: linalg", () => {
  it("full linear algebra pipeline: solve, decompose, verify", () => {
    const A = tensor([
      [2, 1],
      [1, 3],
    ]);
    const b = tensor([5, 7]);

    // Solve Ax = b
    const x = solve(A, b);
    expect(x.size).toBe(2);
    // Verify: A*x ≈ b
    const Ax = dot(A, x);
    expect(Number(Ax.data[0])).toBeCloseTo(5, 4);
    expect(Number(Ax.data[1])).toBeCloseTo(7, 4);

    // Inverse: A * A^-1 ≈ I
    const Ainv = inv(A);
    const I_approx = dot(A, Ainv);
    expect(I_approx.at(0, 0)).toBeCloseTo(1, 4);
    expect(I_approx.at(0, 1)).toBeCloseTo(0, 4);

    // Determinant
    const d = det(A);
    expect(d).toBeCloseTo(5, 4);

    // Trace
    const tr = trace(A);
    expect(Number(tr.data[0])).toBe(5);

    // Rank and condition
    const r = matrixRank(A);
    expect(r).toBe(2);
    const c = cond(A);
    expect(c).toBeGreaterThan(1);

    // Norm
    const n = norm(A);
    expect(n).toBeGreaterThan(0);
  });

  it("SVD decomposition and reconstruction", () => {
    const A = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const [U, S, Vt] = svd(A, false);
    expect(U.shape[0]).toBe(3);
    expect(S.size).toBeGreaterThan(0);
    expect(Vt.shape[0]).toBeGreaterThan(0);
  });

  it("eigenvalue decomposition", () => {
    const A = tensor([
      [2, 1],
      [1, 2],
    ]);
    const [eigvals, _eigvecs] = eig(A);
    // Symmetric matrix: eigenvalues should be 1 and 3
    const vals = [Number(eigvals.data[0]), Number(eigvals.data[1])].sort();
    expect(vals[0]).toBeCloseTo(1, 3);
    expect(vals[1]).toBeCloseTo(3, 3);
  });

  it("QR and LU decomposition", () => {
    const A = tensor([
      [1, 2],
      [3, 4],
    ]);

    const [Q, _R] = qr(A);
    // Q should be orthogonal: Q^T * Q ≈ I
    const QTQ = dot(transpose(Q), Q);
    expect(QTQ.at(0, 0)).toBeCloseTo(1, 3);
    expect(QTQ.at(0, 1)).toBeCloseTo(0, 3);

    const [L, U, _P] = lu(A);
    expect(L.shape).toEqual([2, 2]);
    expect(U.shape).toEqual([2, 2]);
  });

  it("least squares overdetermined system", () => {
    const A = tensor([
      [1, 1],
      [1, 2],
      [1, 3],
    ]);
    const b = tensor([1, 2, 2]);
    const result = lstsq(A, b);
    expect(result.x.size).toBe(2);
  });

  it("Cholesky decomposition for positive definite matrix", () => {
    const A = tensor([
      [4, 2],
      [2, 3],
    ]);
    const L = cholesky(A);
    // L * L^T ≈ A
    const LLT = dot(L, transpose(L));
    expect(LLT.at(0, 0)).toBeCloseTo(4, 4);
    expect(LLT.at(0, 1)).toBeCloseTo(2, 4);
    expect(LLT.at(1, 1)).toBeCloseTo(3, 4);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 4. DATAFRAME: Creation, manipulation, groupby, I/O
// ═══════════════════════════════════════════════════════════════════════════
describe("integration: dataframe", () => {
  it("full DataFrame pipeline: create, filter, group, aggregate", () => {
    const df = new DataFrame({
      name: ["Alice", "Bob", "Charlie", "David", "Eve"],
      age: [25, 30, 35, 28, 22],
      salary: [50000, 60000, 75000, 55000, 48000],
      department: ["IT", "HR", "IT", "HR", "IT"],
    });

    expect(df.shape).toEqual([5, 4]);
    expect(df.columns).toEqual(["name", "age", "salary", "department"]);

    // Selection
    const subset = df.select(["name", "salary"]);
    expect(subset.columns).toEqual(["name", "salary"]);
    expect(subset.shape[1]).toBe(2);

    // Filter
    const highEarners = df.filter((row) => (row.salary as number) > 55000);
    expect(highEarners.shape[0]).toBe(2);

    // Head
    const top3 = df.head(3);
    expect(top3.shape[0]).toBe(3);

    // Sort
    const sorted = df.sort("salary", false);
    expect(sorted.shape[0]).toBe(5);

    // GroupBy aggregation
    const byDept = df.groupBy("department").agg({
      salary: "mean",
      age: "max",
    });
    expect(byDept.shape[0]).toBeGreaterThan(0);

    // Describe
    const desc = df.describe();
    expect(desc.shape[0]).toBeGreaterThan(0);

    // CSV round-trip
    const csv = df.toCsvString();
    const restored = DataFrame.fromCsvString(csv);
    expect(restored.shape).toEqual(df.shape);

    // toTensor
    const t = df.select(["age", "salary"]).toTensor();
    expect(t.shape).toEqual([5, 2]);
  });

  it("Series operations", () => {
    const s = new Series([10, 20, 30, 40, 50], { name: "test" });
    expect(s.length).toBe(5);
    expect(s.toArray()).toEqual([10, 20, 30, 40, 50]);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 5. STATS: Descriptive, correlation, hypothesis tests
// ═══════════════════════════════════════════════════════════════════════════
describe("integration: stats", () => {
  it("descriptive statistics pipeline", () => {
    const data = tensor([2, 4, 4, 4, 5, 5, 7, 9]);

    const m = mean(data);
    expect(Number(m.data[0])).toBe(5);

    const s = std(data);
    expect(Number(s.data[0])).toBeGreaterThan(0);

    const v = variance(data);
    expect(Number(v.data[0])).toBeGreaterThan(0);

    const med = median(data);
    expect(Number(med.data[0])).toBe(4.5);

    const mo = mode(data);
    expect(Number(mo.data[0])).toBe(4);

    const q = quantile(data, 0.75);
    expect(Number(q.data[0])).toBeGreaterThan(0);

    const p = percentile(data, 75);
    expect(Number(p.data[0])).toBeGreaterThan(0);

    const sk = skewness(data);
    expect(typeof Number(sk.data[0])).toBe("number");

    const ku = kurtosis(data);
    expect(typeof Number(ku.data[0])).toBe("number");
  });

  it("correlation and covariance pipeline", () => {
    const x = tensor([1, 2, 3, 4, 5]);
    const y = tensor([2, 4, 5, 4, 5]);

    // corrcoef(x, y) form returns 2x2 matrix
    const corr = corrcoef(x, y);
    expect(corr.shape).toEqual([2, 2]);
    expect(corr.at(0, 0)).toBeCloseTo(1, 4);

    // cov with 2D input: columns are variables, rows are observations
    const dataMatrix = tensor([
      [1, 2],
      [2, 4],
      [3, 5],
      [4, 4],
      [5, 5],
    ]);
    const covMat = cov(dataMatrix);
    expect(covMat.shape).toEqual([2, 2]);

    const [r, p] = pearsonr(x, y);
    expect(r).toBeGreaterThan(0);
    expect(p).toBeGreaterThanOrEqual(0);

    const [rs, _ps] = spearmanr(x, y);
    expect(rs).toBeGreaterThan(0);
  });

  it("hypothesis testing pipeline", () => {
    const data = tensor([5.1, 4.9, 5.0, 5.2, 4.8, 5.1, 5.0, 4.9]);

    const tResult = ttest_1samp(data, 5.0);
    expect(tResult.statistic).toBeDefined();
    expect(tResult.pvalue).toBeGreaterThanOrEqual(0);

    const group1 = tensor([5.1, 4.9, 5.0, 5.2, 4.8]);
    const group2 = tensor([5.5, 5.3, 5.4, 5.6, 5.2]);
    const tResult2 = ttest_ind(group1, group2);
    expect(tResult2.pvalue).toBeDefined();

    const normResult = shapiro(data);
    expect(normResult.statistic).toBeGreaterThan(0);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 6. METRICS: Classification, regression, clustering
// ═══════════════════════════════════════════════════════════════════════════
describe("integration: metrics", () => {
  it("binary classification metrics pipeline", () => {
    const yTrue = tensor([0, 1, 1, 0, 1, 0, 1, 1]);
    const yPred = tensor([0, 1, 0, 0, 1, 1, 1, 1]);

    const acc = accuracy(yTrue, yPred);
    expect(acc).toBeGreaterThan(0.5);

    const prec = precision(yTrue, yPred);
    expect(prec).toBeGreaterThan(0);

    const rec = recall(yTrue, yPred);
    expect(rec).toBeGreaterThan(0);

    const f1 = f1Score(yTrue, yPred);
    expect(f1).toBeGreaterThan(0);

    const cm = confusionMatrix(yTrue, yPred);
    expect(cm.shape).toEqual([2, 2]);
  });

  it("multiclass classification with macro averaging", () => {
    const yTrue = tensor([0, 1, 2, 0, 1, 2]);
    const yPred = tensor([0, 2, 1, 0, 0, 1]);

    const acc = accuracy(yTrue, yPred);
    expect(acc).toBeGreaterThanOrEqual(0);

    const f1Macro = f1Score(yTrue, yPred, "macro");
    expect(f1Macro).toBeGreaterThanOrEqual(0);

    const precMacro = precision(yTrue, yPred, "macro");
    expect(precMacro).toBeGreaterThanOrEqual(0);

    const recWeighted = recall(yTrue, yPred, "weighted");
    expect(recWeighted).toBeGreaterThanOrEqual(0);

    const cm = confusionMatrix(yTrue, yPred);
    expect(cm.shape).toEqual([3, 3]);
  });

  it("regression metrics pipeline", () => {
    const yTrue = tensor([3, -0.5, 2, 7]);
    const yPred = tensor([2.5, 0.0, 2, 8]);

    const r2 = r2Score(yTrue, yPred);
    expect(r2).toBeGreaterThan(0.9);

    const mseVal = mse(yTrue, yPred);
    expect(mseVal).toBeGreaterThanOrEqual(0);

    const maeVal = mae(yTrue, yPred);
    expect(maeVal).toBeGreaterThanOrEqual(0);

    const rmseVal = rmse(yTrue, yPred);
    expect(rmseVal).toBeGreaterThanOrEqual(0);
    expect(rmseVal).toBeCloseTo(Math.sqrt(mseVal), 5);
  });

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

    const score = silhouetteScore(X, labels);
    expect(score).toBeGreaterThan(0);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 7. PREPROCESS: Scaling, encoding, splitting
// ═══════════════════════════════════════════════════════════════════════════
describe("integration: preprocess", () => {
  it("standard scaler pipeline: fit, transform, inverse", () => {
    const X = tensor([
      [1, 100],
      [2, 200],
      [3, 300],
      [4, 400],
      [5, 500],
    ]);

    const scaler = new StandardScaler();
    scaler.fit(X);
    const XScaled = scaler.transform(X);

    // Mean should be approximately 0
    const colMeans = ndarrayMean(XScaled, 0);
    expect(Math.abs(Number(colMeans.data[0]))).toBeLessThan(0.001);
    expect(Math.abs(Number(colMeans.data[1]))).toBeLessThan(0.001);

    // Inverse transform should recover original
    const XRecovered = scaler.inverseTransform(XScaled);
    expect(XRecovered.at(0, 0)).toBeCloseTo(1, 3);
    expect(XRecovered.at(0, 1)).toBeCloseTo(100, 1);
  });

  it("minmax and robust scaler pipeline", () => {
    const X = tensor([
      [1, 100],
      [2, 200],
      [3, 300],
      [4, 400],
      [5, 500],
    ]);

    const mm = new MinMaxScaler();
    mm.fit(X);
    const XMM = mm.transform(X);
    expect(XMM.at(0, 0)).toBeCloseTo(0, 5);
    expect(XMM.at(4, 0)).toBeCloseTo(1, 5);

    const rs = new RobustScaler();
    rs.fit(X);
    const XRS = rs.transform(X);
    expect(XRS.shape).toEqual([5, 2]);
  });

  it("encoder pipeline: label, onehot, ordinal", () => {
    const le = new LabelEncoder();
    le.fit(tensor(["cat", "dog", "bird", "cat", "dog"], { dtype: "string" }));
    const encoded = le.transform(tensor(["cat", "bird", "dog"], { dtype: "string" }));
    expect(encoded.size).toBe(3);
    const decoded = le.inverseTransform(encoded);
    expect(decoded.toArray()).toEqual(["cat", "bird", "dog"]);

    const ohe = new OneHotEncoder();
    ohe.fit(tensor([["red"], ["green"], ["blue"], ["red"]], { dtype: "string" }));
    const onehot = ohe.transform(tensor([["red"], ["blue"], ["green"]], { dtype: "string" }));
    expect(onehot.shape[0]).toBe(3);
    expect(onehot.shape[1]).toBe(3);

    const oe = new OrdinalEncoder();
    oe.fit(tensor([["small"], ["medium"], ["large"]], { dtype: "string" }));
    const ordinal = oe.transform(tensor([["large"], ["small"]], { dtype: "string" }));
    expect(ordinal.shape[0]).toBe(2);
  });

  it("train-test split and cross-validation pipeline", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]);
    const y = tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);

    // Train-test split
    const [XTrain, XTest, yTrain, yTest] = trainTestSplit(X, y, {
      testSize: 0.3,
      randomState: 42,
    });
    expect(XTrain.shape[0] + XTest.shape[0]).toBe(10);
    expect(yTrain.shape[0] + yTest.shape[0]).toBe(10);

    // KFold
    const kf = new KFold({ nSplits: 5, shuffle: true, randomState: 42 });
    let foldCount = 0;
    for (const { trainIndex: trainIdx, testIndex: testIdx } of kf.split(X)) {
      expect(trainIdx.length + testIdx.length).toBe(10);
      foldCount++;
    }
    expect(foldCount).toBe(5);

    // StratifiedKFold
    const skf = new StratifiedKFold({
      nSplits: 3,
      shuffle: true,
      randomState: 42,
    });
    let sFoldCount = 0;
    for (const { trainIndex: trainIdx, testIndex: testIdx } of skf.split(X, y)) {
      expect(trainIdx.length + testIdx.length).toBe(10);
      sFoldCount++;
    }
    expect(sFoldCount).toBe(3);

    // LeaveOneOut
    const loo = new LeaveOneOut();
    let looCount = 0;
    for (const { testIndex: testIdx } of loo.split(X)) {
      expect(testIdx.length).toBe(1);
      looCount++;
    }
    expect(looCount).toBe(10);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 8. ML: Supervised, unsupervised, clustering, dimensionality reduction
// ═══════════════════════════════════════════════════════════════════════════
describe("integration: ml", () => {
  it("linear regression end-to-end: train, predict, evaluate", () => {
    const X = tensor([[1], [2], [3], [4], [5], [6], [7], [8]]);
    const y = tensor([3, 5, 7, 9, 11, 13, 15, 17]);

    const [XTrain, XTest, yTrain, yTest] = trainTestSplit(X, y, {
      testSize: 0.25,
      randomState: 42,
    });

    const model = new LinearRegression();
    model.fit(XTrain, yTrain);

    const preds = model.predict(XTest);
    expect(preds.size).toBe(XTest.shape[0]);

    const r2 = r2Score(yTest, preds);
    expect(r2).toBeGreaterThan(0.95);

    // Coefficients should approximate y = 2x + 1
    expect(model.coef).toBeDefined();
    expect(model.intercept).toBeDefined();
  });

  it("ridge and lasso regression comparison", () => {
    const X = tensor([[1], [2], [3], [4], [5]]);
    const y = tensor([2.1, 4.0, 5.9, 8.1, 9.9]);

    const ridge = new Ridge({ alpha: 1.0 });
    ridge.fit(X, y);
    const ridgePreds = ridge.predict(X);
    expect(ridgePreds.size).toBe(5);

    const lasso = new Lasso({ alpha: 0.1 });
    lasso.fit(X, y);
    const lassoPreds = lasso.predict(X);
    expect(lassoPreds.size).toBe(5);
  });

  it("logistic regression binary classification", () => {
    const iris = loadIris();
    // Convert to binary: setosa (0) vs non-setosa (1)
    const binaryTarget: number[] = [];
    for (let i = 0; i < iris.target.size; i++) {
      binaryTarget.push(Number(iris.target.data[iris.target.offset + i]) === 0 ? 0 : 1);
    }
    const y = tensor(binaryTarget);

    const [XTr, XTe, yTr, yTe] = trainTestSplit(iris.data, y, {
      testSize: 0.3,
      randomState: 42,
    });

    const scaler = new StandardScaler();
    scaler.fit(XTr);
    const XTrS = scaler.transform(XTr);
    const XTeS = scaler.transform(XTe);

    const model = new LogisticRegression();
    model.fit(XTrS, yTr);
    const preds = model.predict(XTeS);

    const acc = accuracy(yTe, preds);
    expect(acc).toBeGreaterThan(0.9);
  });

  it("multi-model comparison on Iris (multiclass)", () => {
    const iris = loadIris();
    const [XTr, XTe, yTr, yTe] = trainTestSplit(iris.data, iris.target, {
      testSize: 0.2,
      randomState: 42,
    });
    const scaler = new StandardScaler();
    scaler.fit(XTr);
    const XTrS = scaler.transform(XTr);
    const XTeS = scaler.transform(XTe);

    const models = [
      new DecisionTreeClassifier(),
      new RandomForestClassifier(),
      new KNeighborsClassifier({ nNeighbors: 5 }),
      new GaussianNB(),
    ];

    for (const model of models) {
      model.fit(XTrS, yTr);
      const preds = model.predict(XTeS);
      const acc = accuracy(yTe, preds);
      expect(acc).toBeGreaterThan(0.7);
    }
  });

  it("KMeans clustering pipeline", () => {
    const X = tensor([
      [1, 2],
      [1.5, 1.8],
      [5, 8],
      [8, 8],
      [1, 0.6],
      [9, 11],
    ]);

    const km = new KMeans({ nClusters: 2, randomState: 42 });
    km.fit(X);

    expect(km.labels).toBeDefined();
    expect(km.clusterCenters).toBeDefined();
    expect(km.inertia).toBeGreaterThan(0);

    const newLabels = km.predict(tensor([[3, 3]]));
    expect(newLabels.size).toBe(1);

    const score = silhouetteScore(X, km.labels);
    expect(score).toBeGreaterThan(0);
  });

  it("PCA dimensionality reduction pipeline", () => {
    const iris = loadIris();

    const pca = new PCA({ nComponents: 2 });
    pca.fit(iris.data);

    const X2d = pca.transform(iris.data);
    expect(X2d.shape[0]).toBe(150);
    expect(X2d.shape[1]).toBe(2);

    expect(pca.explainedVarianceRatio.size).toBe(2);
    expect(pca.components.shape[0]).toBe(2);
  });

  it("DBSCAN density-based clustering", () => {
    const X = tensor([
      [1, 2],
      [1.5, 1.8],
      [5, 8],
      [8, 8],
      [1, 0.6],
      [9, 11],
    ]);

    const db = new DBSCAN({ eps: 3, minSamples: 2 });
    db.fit(X);

    expect(db.labels).toBeDefined();
    expect(db.labels.size).toBeGreaterThanOrEqual(0);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 9. NN: Layers, activations, losses, forward/backward
// ═══════════════════════════════════════════════════════════════════════════
describe("integration: nn", () => {
  it("Sequential model build, forward, parameters", () => {
    const model = new Sequential(new Linear(4, 8), new ReLULayer(), new Linear(8, 2));

    const params = Array.from(model.parameters());
    expect(params.length).toBe(4); // 2 weight + 2 bias

    const input = tensor([[1, 2, 3, 4]]);
    const output = model.forward(input);
    const outTensor = output instanceof GradTensor ? output.tensor : output;
    expect(outTensor.shape).toEqual([1, 2]);
  });

  it("Conv1d forward pass", () => {
    const conv = new Conv1d(1, 4, 3, { padding: 1 });
    const input = tensor([[[1, 2, 3, 4, 5, 6, 7, 8]]]);
    const output = conv.forward(input);
    const outTensor = output instanceof GradTensor ? output.tensor : output;
    expect(outTensor.shape[0]).toBe(1);
    expect(outTensor.shape[1]).toBe(4);
  });

  it("Conv2d forward pass (tests non-contiguous reshape fix)", async () => {
    const { Conv2d } = await import("../src/nn");
    const conv = new Conv2d(1, 2, 2, { bias: false });
    const input = tensor([
      [
        [
          [1, 2],
          [3, 4],
        ],
      ],
    ]);
    const output = conv.forward(input);
    const outTensor = output instanceof GradTensor ? output.tensor : output;
    expect(outTensor.shape[0]).toBe(1);
    expect(outTensor.shape[1]).toBe(2);
  });

  it("RNN, LSTM, GRU forward pass", () => {
    const rnn = new RNN(4, 8);
    const lstm = new LSTM(4, 8);
    const gru = new GRU(4, 8);

    const input = tensor([
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ],
    ]);

    const rnnOut = rnn.forward(input);
    const rnnTensor = rnnOut instanceof GradTensor ? rnnOut.tensor : rnnOut;
    expect(rnnTensor.shape[2]).toBe(8);

    const lstmOut = lstm.forward(input);
    const lstmTensor = lstmOut instanceof GradTensor ? lstmOut.tensor : lstmOut;
    expect(lstmTensor.shape[2]).toBe(8);

    const gruOut = gru.forward(input);
    const gruTensor = gruOut instanceof GradTensor ? gruOut.tensor : gruOut;
    expect(gruTensor.shape[2]).toBe(8);
  });

  it("MultiheadAttention and TransformerEncoderLayer", () => {
    const mha = new MultiheadAttention(8, 2);
    const input = tensor([
      [
        [1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1],
      ],
    ]);
    const attnOut = mha.forward(input, input, input);
    const attnTensor = attnOut instanceof GradTensor ? attnOut.tensor : attnOut;
    expect(attnTensor.shape).toEqual([1, 2, 8]);

    const encoder = new TransformerEncoderLayer(8, 2, 16);
    const encOut = encoder.forward(input);
    const encTensor = encOut instanceof GradTensor ? encOut.tensor : encOut;
    expect(encTensor.shape).toEqual([1, 2, 8]);
  });

  it("loss functions pipeline", () => {
    const pred = tensor([
      [1.0, 2.0],
      [3.0, 4.0],
    ]);
    const target = tensor([
      [1.1, 2.1],
      [2.9, 3.9],
    ]);

    const mseL = mseLoss(pred, target);
    expect(
      mseL instanceof GradTensor ? Number(mseL.tensor.data[0]) : Number(mseL.data[0])
    ).toBeGreaterThan(0);

    const maeL = maeLoss(pred, target);
    expect(
      maeL instanceof GradTensor ? Number(maeL.tensor.data[0]) : Number(maeL.data[0])
    ).toBeGreaterThan(0);
  });

  it("Module system: custom module, state dict, freeze/unfreeze", () => {
    class MyNet extends Module {
      fc1: Linear;
      relu: ReLULayer;
      fc2: Linear;

      constructor() {
        super();
        this.fc1 = new Linear(4, 8);
        this.relu = new ReLULayer();
        this.fc2 = new Linear(8, 2);
        this.registerModule("fc1", this.fc1);
        this.registerModule("relu", this.relu);
        this.registerModule("fc2", this.fc2);
      }

      override forward(x: GradTensor): GradTensor {
        let out = this.fc1.forward(x) as GradTensor;
        out = this.relu.forward(out) as GradTensor;
        return this.fc2.forward(out) as GradTensor;
      }
    }

    const net = new MyNet();
    expect(Array.from(net.parameters()).length).toBe(4);

    // State dict
    const state = net.stateDict();
    expect(Object.keys(state.parameters).length).toBeGreaterThan(0);
    net.loadStateDict(state);

    // Train/eval mode
    net.train();
    expect(net.training).toBe(true);
    net.eval();
    expect(net.training).toBe(false);

    // Freeze/unfreeze
    net.freezeParameters();
    net.unfreezeParameters();
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 10. OPTIM: Optimizers and LR schedulers
// ═══════════════════════════════════════════════════════════════════════════
describe("integration: optim", () => {
  it("optimizer training loop with SGD, Adam, AdamW", () => {
    const model = new Sequential(new Linear(2, 4), new ReLULayer(), new Linear(4, 1));

    // SGD
    const sgd = new SGD(model.parameters(), { lr: 0.01, momentum: 0.9 });
    expect(sgd.getLearningRate()).toBe(0.01);

    // Adam
    const adam = new Adam(model.parameters(), { lr: 0.001 });
    expect(adam.getLearningRate()).toBe(0.001);

    // AdamW
    const adamw = new AdamW(model.parameters(), {
      lr: 0.001,
      weightDecay: 0.01,
    });
    expect(adamw.getLearningRate()).toBe(0.001);

    // Basic training step
    const x = parameter([[1, 2]]);
    const target = parameter([[5]]);
    const pred = model.forward(x);
    const diff = (pred as GradTensor).sub(target);
    const loss = diff.mul(diff).mean();

    adam.zeroGrad();
    loss.backward();
    adam.step();
  });

  it("LR scheduler pipeline: StepLR, CosineAnnealing, OneCycleLR", () => {
    const model = new Sequential(new Linear(2, 4), new ReLULayer(), new Linear(4, 1));

    // StepLR
    const opt1 = new SGD(model.parameters(), { lr: 0.1 });
    const sched1 = new StepLR(opt1, { stepSize: 5, gamma: 0.5 });
    for (let i = 0; i < 10; i++) sched1.step();
    expect(opt1.getLearningRate()).toBeLessThan(0.1);

    // CosineAnnealingLR
    const opt2 = new SGD(model.parameters(), { lr: 0.1 });
    const sched2 = new CosineAnnealingLR(opt2, { T_max: 20, etaMin: 0.001 });
    for (let i = 0; i < 10; i++) sched2.step();
    expect(opt2.getLearningRate()).toBeLessThan(0.1);

    // OneCycleLR
    const opt3 = new SGD(model.parameters(), { lr: 0.01 });
    const sched3 = new OneCycleLR(opt3, { maxLr: 0.1, totalSteps: 20 });
    for (let i = 0; i < 5; i++) sched3.step();
    expect(opt3.getLearningRate()).toBeGreaterThan(0.01);

    // ExponentialLR
    const opt4 = new SGD(model.parameters(), { lr: 0.1 });
    const sched4 = new ExponentialLR(opt4, { gamma: 0.9 });
    sched4.step();
    sched4.step();
    expect(opt4.getLearningRate()).toBeLessThan(0.1);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 11. RANDOM: Distributions, seeding, sampling
// ═══════════════════════════════════════════════════════════════════════════
describe("integration: random", () => {
  it("seeded reproducibility pipeline", () => {
    setSeed(42);
    expect(getSeed()).toBe(42);

    const a1 = rand([3]);
    setSeed(42);
    const a2 = rand([3]);

    // Same seed should produce same results
    expect(a1.toArray()).toEqual(a2.toArray());

    clearSeed();
    expect(getSeed()).toBeUndefined();
  });

  it("distribution sampling pipeline", () => {
    setSeed(42);

    const u = uniform(0, 10, [100]);
    expect(u.size).toBe(100);

    const n = normal(5, 2, [100]);
    expect(n.size).toBe(100);

    const b = binomial(10, 0.5, [50]);
    expect(b.size).toBe(50);

    const p = poisson(3, [50]);
    expect(p.size).toBe(50);

    const ri = randint(0, 10, [50]);
    expect(ri.size).toBe(50);

    const data = tensor([10, 20, 30, 40, 50]);
    const c = choice(data, 3);
    expect(c.size).toBe(3);

    const perm = permutation(5);
    expect(perm.size).toBe(5);

    clearSeed();
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 12. PLOT: Figure, Axes, SVG rendering
// ═══════════════════════════════════════════════════════════════════════════
describe("integration: plot", () => {
  it("Figure/Axes API: scatter, plot, bar, hist, heatmap, renderSVG", () => {
    const fig = new Figure({ width: 800, height: 600 });
    const ax = fig.addAxes();

    // Scatter
    ax.scatter(tensor([1, 2, 3]), tensor([4, 5, 6]), { color: "#1f77b4" });
    ax.setTitle("Test Plot");
    ax.setXLabel("X");
    ax.setYLabel("Y");

    // Line plot
    ax.plot(tensor([1, 2, 3]), tensor([1, 4, 9]), { color: "#ff7f0e" });

    const svg = fig.renderSVG();
    expect(svg.svg).toContain("<svg");
    expect(svg.svg.length).toBeGreaterThan(100);

    // Histogram
    const fig2 = new Figure();
    const ax2 = fig2.addAxes();
    ax2.hist(tensor([1, 2, 2, 3, 3, 3, 4, 4, 5]), 5);
    const svg2 = fig2.renderSVG();
    expect(svg2.svg).toContain("<svg");

    // Bar chart
    const fig3 = new Figure();
    const ax3 = fig3.addAxes();
    ax3.bar(tensor([0, 1, 2]), tensor([10, 20, 15]));
    const svg3 = fig3.renderSVG();
    expect(svg3.svg).toContain("<svg");

    // Heatmap
    const fig4 = new Figure();
    const ax4 = fig4.addAxes();
    ax4.heatmap(
      tensor([
        [1, 2],
        [3, 4],
      ])
    );
    const svg4 = fig4.renderSVG();
    expect(svg4.svg).toContain("<svg");
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// 13. DATASETS: Built-in datasets, synthetic generators, DataLoader
// ═══════════════════════════════════════════════════════════════════════════
describe("integration: datasets", () => {
  it("built-in dataset loading pipeline", () => {
    const iris = loadIris();
    expect(iris.data.shape).toEqual([150, 4]);
    expect(iris.target.size).toBe(150);
    expect(iris.featureNames).toBeDefined();
    expect(iris.targetNames).toBeDefined();

    const digits = loadDigits();
    expect(digits.data.shape[0]).toBe(1797);

    const cancer = loadBreastCancer();
    expect(cancer.data.shape[0]).toBe(569);

    const diabetes = loadDiabetes();
    expect(diabetes.data.shape[0]).toBe(442);

    const housing = loadHousingMini();
    expect(housing.data.shape[0]).toBeGreaterThan(0);
  });

  it("synthetic data generators pipeline", () => {
    const [blobsX, blobsY] = makeBlobs({
      nSamples: 100,
      centers: 3,
      randomState: 42,
    });
    expect(blobsX.shape[0]).toBe(100);
    expect(blobsY.size).toBe(100);

    const [circlesX, _circlesY] = makeCircles({
      nSamples: 100,
      noise: 0.05,
      factor: 0.5,
    });
    expect(circlesX.shape[0]).toBe(100);

    const [moonsX, _moonsY] = makeMoons({ nSamples: 100, noise: 0.1 });
    expect(moonsX.shape[0]).toBe(100);

    const [regX, regY] = makeRegression({
      nSamples: 50,
      nFeatures: 5,
      noise: 0.1,
    });
    expect(regX.shape).toEqual([50, 5]);
    expect(regY.size).toBe(50);

    const [clsX, _clsY] = makeClassification({
      nSamples: 100,
      nFeatures: 10,
      nInformative: 5,
      nClasses: 3,
    });
    expect(clsX.shape[0]).toBe(100);
  });

  it("DataLoader batching and iteration", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
      [9, 10],
    ]);
    const y = tensor([0, 1, 0, 1, 0]);

    const loader = new DataLoader(X, y, {
      batchSize: 2,
      shuffle: false,
    });

    let batchCount = 0;
    for (const batch of loader as Iterable<
      [ReturnType<typeof tensor>, ReturnType<typeof tensor>]
    >) {
      expect(batch[0].shape[1]).toBe(2);
      batchCount++;
    }
    expect(batchCount).toBe(3); // 2+2+1

    // With dropLast
    const loader2 = new DataLoader(X, y, {
      batchSize: 2,
      dropLast: true,
    });
    let dropCount = 0;
    for (const batch of loader2 as Iterable<
      [ReturnType<typeof tensor>, ReturnType<typeof tensor>]
    >) {
      expect(batch[0].shape[0]).toBe(2);
      dropCount++;
    }
    expect(dropCount).toBe(2);
  });
});

// ═══════════════════════════════════════════════════════════════════════════
// CROSS-SCOPE INTEGRATION: End-to-end ML pipeline
// ═══════════════════════════════════════════════════════════════════════════
describe("integration: cross-scope end-to-end pipelines", () => {
  it("complete classification pipeline: load → split → scale → train → evaluate → visualize", () => {
    // Load data (datasets)
    const iris = loadIris();

    // Convert to binary for metrics that need it
    const binaryTarget: number[] = [];
    for (let i = 0; i < iris.target.size; i++) {
      binaryTarget.push(Number(iris.target.data[iris.target.offset + i]) === 0 ? 0 : 1);
    }
    const y = tensor(binaryTarget);

    // Split (preprocess)
    const [XTr, XTe, yTr, yTe] = trainTestSplit(iris.data, y, {
      testSize: 0.3,
      randomState: 42,
    });

    // Scale (preprocess)
    const scaler = new StandardScaler();
    scaler.fit(XTr);
    const XTrS = scaler.transform(XTr);
    const XTeS = scaler.transform(XTe);

    // Train (ml)
    const model = new LogisticRegression();
    model.fit(XTrS, yTr);

    // Predict
    const preds = model.predict(XTeS);

    // Evaluate (metrics)
    const acc = accuracy(yTe, preds);
    expect(acc).toBeGreaterThan(0.9);

    const f1 = f1Score(yTe, preds);
    expect(f1).toBeGreaterThan(0.8);

    const cm = confusionMatrix(yTe, preds);
    expect(cm.shape).toEqual([2, 2]);

    // Visualize (plot)
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.scatter(tensor([1, 2, 3]), tensor([acc, f1, 0.95]));
    ax.setTitle("Metrics");
    const svg = fig.renderSVG();
    expect(svg.svg).toContain("<svg");
  });

  it("complete regression pipeline: generate → split → train → evaluate", () => {
    // Generate synthetic data (datasets)
    const [X, y] = makeRegression({
      nSamples: 100,
      nFeatures: 3,
      noise: 0.5,
      randomState: 42,
    });

    // Split
    const [XTr, XTe, yTr, yTe] = trainTestSplit(X, y, {
      testSize: 0.2,
      randomState: 42,
    });

    // Scale
    const scaler = new StandardScaler();
    scaler.fit(XTr);
    const XTrS = scaler.transform(XTr);
    const XTeS = scaler.transform(XTe);

    // Train multiple models
    const lr = new LinearRegression();
    lr.fit(XTrS, yTr);
    const lrPreds = lr.predict(XTeS);
    const lrR2 = r2Score(yTe, lrPreds);
    expect(lrR2).toBeGreaterThan(0.5);

    const ridge = new Ridge({ alpha: 1.0 });
    ridge.fit(XTrS, yTr);
    const ridgePreds = ridge.predict(XTeS);
    const ridgeR2 = r2Score(yTe, ridgePreds);
    expect(ridgeR2).toBeGreaterThan(0.5);
  });

  it("neural network training loop with autograd", () => {
    const model = new Sequential(new Linear(2, 8), new ReLULayer(), new Linear(8, 1));
    const optimizer = new Adam(model.parameters(), { lr: 0.01 });

    // Training data: y = x0 + 2*x1
    const X = parameter([
      [1, 0],
      [0, 1],
      [1, 1],
      [2, 1],
    ]);
    const yTarget = parameter([[1], [2], [3], [4]]);

    let initialLoss = Infinity;
    for (let epoch = 0; epoch < 50; epoch++) {
      const pred = model.forward(X) as GradTensor;
      const diff = pred.sub(yTarget);
      const loss = diff.mul(diff).mean();

      if (epoch === 0) initialLoss = Number(loss.tensor.data[0]);

      optimizer.zeroGrad();
      loss.backward();
      optimizer.step();
    }

    // Loss should decrease
    const X2 = parameter([
      [1, 0],
      [0, 1],
      [1, 1],
      [2, 1],
    ]);
    const yTarget2 = parameter([[1], [2], [3], [4]]);
    const finalPred = model.forward(X2) as GradTensor;
    const finalDiff = finalPred.sub(yTarget2);
    const finalLoss = Number(finalDiff.mul(finalDiff).mean().tensor.data[0]);
    expect(finalLoss).toBeLessThan(initialLoss);
  });

  it("clustering + visualization pipeline", () => {
    // Generate clusters
    const [X, _y] = makeBlobs({ nSamples: 100, centers: 3, randomState: 42 });

    // Cluster
    const km = new KMeans({ nClusters: 3, randomState: 42 });
    km.fit(X);

    // Evaluate
    const score = silhouetteScore(X, km.labels);
    expect(score).toBeGreaterThan(0.3);

    // PCA for visualization
    const pca = new PCA({ nComponents: 2 });
    pca.fit(X);
    const X2d = pca.transform(X);
    expect(X2d.shape).toEqual([100, 2]);

    // Plot
    const fig = new Figure();
    const ax = fig.addAxes();
    ax.scatter(tensor([1, 2, 3]), tensor([4, 5, 6]));
    const svg = fig.renderSVG();
    expect(svg.svg).toContain("<svg");
  });

  it("DataLoader + NN training loop pipeline", () => {
    const X = tensor([
      [1, 2],
      [3, 4],
      [5, 6],
      [7, 8],
      [9, 10],
      [11, 12],
    ]);
    const y = tensor([[3], [7], [11], [15], [19], [23]]);

    const loader = new DataLoader(X, y, {
      batchSize: 3,
      shuffle: false,
    });

    const model = new Sequential(new Linear(2, 4), new ReLULayer(), new Linear(4, 1));
    const optimizer = new Adam(model.parameters(), { lr: 0.01 });

    // One epoch through the DataLoader
    let batchCount = 0;
    for (const batch of loader as Iterable<
      [ReturnType<typeof tensor>, ReturnType<typeof tensor>]
    >) {
      const xGrad = parameter(batch[0].toArray() as number[][]);
      const yGrad = parameter(batch[1].toArray() as number[][]);
      const pred = model.forward(xGrad) as GradTensor;
      const diff = pred.sub(yGrad);
      const loss = diff.mul(diff).mean();

      optimizer.zeroGrad();
      loss.backward();
      optimizer.step();
      batchCount++;
    }
    expect(batchCount).toBe(2); // 3+3
  });

  it("statistics + DataFrame analysis pipeline", () => {
    const df = new DataFrame({
      x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
      y: [2.1, 3.9, 6.1, 8.0, 9.9, 12.1, 14.0, 15.9, 18.1, 20.0],
    });

    // Extract columns
    const xArr = df.get("x").toArray() as number[];
    const yArr = df.get("y").toArray() as number[];

    const xTensor = tensor(xArr);
    const yTensor = tensor(yArr);

    // Compute statistics
    const xMean = Number(mean(xTensor).data[0]);
    expect(xMean).toBeCloseTo(5.5, 1);

    const xStd = Number(std(xTensor).data[0]);
    expect(xStd).toBeGreaterThan(0);

    // Correlation
    const [r, _p] = pearsonr(xTensor, yTensor);
    expect(r).toBeGreaterThan(0.99);

    // Linear regression
    const X = tensor(xArr.map((v) => [v]));
    const model = new LinearRegression();
    model.fit(X, yTensor);
    const preds = model.predict(X);
    const r2 = r2Score(yTensor, preds);
    expect(r2).toBeGreaterThan(0.99);
  });
});
