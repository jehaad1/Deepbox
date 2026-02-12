/**
 * Benchmark 10 — Preprocessing
 * Deepbox vs scikit-learn
 */

import { tensor } from "deepbox/ndarray";
import {
  KFold,
  LabelBinarizer,
  LabelEncoder,
  LeaveOneOut,
  MaxAbsScaler,
  MinMaxScaler,
  Normalizer,
  OneHotEncoder,
  OrdinalEncoder,
  PowerTransformer,
  QuantileTransformer,
  RobustScaler,
  StandardScaler,
  StratifiedKFold,
  trainTestSplit,
} from "deepbox/preprocess";
import { createSuite, footer, header, run } from "../utils";

const suite = createSuite("preprocess");
header("Benchmark 10 — Preprocessing");

// ── Data generators ──────────────────────────────────────

function seededRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}

function makeNumeric(n: number, f: number, seed: number) {
  const rand = seededRng(seed);
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < f; j++) row.push(rand() * 100 - 50);
    data.push(row);
  }
  return tensor(data);
}

function makePositive(n: number, f: number, seed: number) {
  const rand = seededRng(seed);
  const data: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < f; j++) row.push(rand() * 100 + 1);
    data.push(row);
  }
  return tensor(data);
}

const X200 = makeNumeric(200, 5, 42);
const X500 = makeNumeric(500, 10, 42);
const X1k = makeNumeric(1000, 10, 42);
const X5k = makeNumeric(5000, 10, 42);
const Xpos500 = makePositive(500, 10, 42);
const _Xpos1k = makePositive(1000, 10, 42);

function makeLabels(n: number, k: number, seed: number) {
  const rand = seededRng(seed);
  const labels: number[] = [];
  for (let i = 0; i < n; i++) labels.push(Math.floor(rand() * k));
  return tensor(labels);
}

const y200 = makeLabels(200, 3, 42);
const y500 = makeLabels(500, 5, 42);
const y1k = makeLabels(1000, 5, 42);

// ── StandardScaler ──────────────────────────────────────

run(suite, "StandardScaler fit", "200x5", () => new StandardScaler().fit(X200));
run(suite, "StandardScaler fit", "500x10", () => new StandardScaler().fit(X500));
run(suite, "StandardScaler fit", "1Kx10", () => new StandardScaler().fit(X1k));
const ss = new StandardScaler().fit(X500);
run(suite, "StandardScaler transform", "500x10", () => ss.transform(X500));
run(suite, "StandardScaler transform", "1Kx10", () => ss.transform(X1k));
run(suite, "StandardScaler fit+transform", "5Kx10", () =>
  new StandardScaler().fit(X5k).transform(X5k)
);

// ── MinMaxScaler ────────────────────────────────────────

run(suite, "MinMaxScaler fit", "200x5", () => new MinMaxScaler().fit(X200));
run(suite, "MinMaxScaler fit", "500x10", () => new MinMaxScaler().fit(X500));
run(suite, "MinMaxScaler fit", "1Kx10", () => new MinMaxScaler().fit(X1k));
const mms = new MinMaxScaler().fit(X500);
run(suite, "MinMaxScaler transform", "500x10", () => mms.transform(X500));
run(suite, "MinMaxScaler transform", "1Kx10", () => mms.transform(X1k));

// ── RobustScaler ────────────────────────────────────────

run(suite, "RobustScaler fit", "500x10", () => new RobustScaler().fit(X500));
run(suite, "RobustScaler fit", "1Kx10", () => new RobustScaler().fit(X1k));
const rs = new RobustScaler().fit(X500);
run(suite, "RobustScaler transform", "500x10", () => rs.transform(X500));

// ── MaxAbsScaler ────────────────────────────────────────

run(suite, "MaxAbsScaler fit", "500x10", () => new MaxAbsScaler().fit(X500));
run(suite, "MaxAbsScaler fit", "1Kx10", () => new MaxAbsScaler().fit(X1k));
const mas = new MaxAbsScaler().fit(X500);
run(suite, "MaxAbsScaler transform", "500x10", () => mas.transform(X500));

// ── Normalizer ──────────────────────────────────────────

run(suite, "Normalizer fit+transform", "500x10", () => new Normalizer().fit(X500).transform(X500));
run(suite, "Normalizer fit+transform", "1Kx10", () => new Normalizer().fit(X1k).transform(X1k));

// ── PowerTransformer ────────────────────────────────────

run(suite, "PowerTransformer fit", "500x10", () => new PowerTransformer().fit(Xpos500));
const pt = new PowerTransformer().fit(Xpos500);
run(suite, "PowerTransformer transform", "500x10", () => pt.transform(Xpos500));

// ── QuantileTransformer ─────────────────────────────────

run(suite, "QuantileTransformer fit", "500x10", () => new QuantileTransformer().fit(X500));
const qt = new QuantileTransformer().fit(X500);
run(suite, "QuantileTransformer transform", "500x10", () => qt.transform(X500));

// ── LabelEncoder ────────────────────────────────────────

const strLabels500 = tensor(
  Array.from({ length: 500 }, (_, i) => ["cat", "dog", "fish", "bird", "snake"][i % 5])
);
const strLabels1k = tensor(
  Array.from({ length: 1000 }, (_, i) => ["cat", "dog", "fish", "bird", "snake"][i % 5])
);

run(suite, "LabelEncoder fit", "500 labels", () => new LabelEncoder().fit(strLabels500));
run(suite, "LabelEncoder fit", "1K labels", () => new LabelEncoder().fit(strLabels1k));
const le = new LabelEncoder().fit(strLabels500);
run(suite, "LabelEncoder transform", "500 labels", () => le.transform(strLabels500));
run(suite, "LabelEncoder transform", "1K labels", () => le.transform(strLabels1k));

// ── OneHotEncoder ───────────────────────────────────────

const strLabels500_2d = strLabels500.reshape([500, 1]);
const strLabels1k_2d = strLabels1k.reshape([1000, 1]);

run(suite, "OneHotEncoder fit", "500 samples", () => new OneHotEncoder().fit(strLabels500_2d));
run(suite, "OneHotEncoder fit", "1K samples", () => new OneHotEncoder().fit(strLabels1k_2d));
const ohe = new OneHotEncoder().fit(strLabels500_2d);
run(suite, "OneHotEncoder transform", "500 samples", () => ohe.transform(strLabels500_2d));

// ── OrdinalEncoder ──────────────────────────────────────

run(suite, "OrdinalEncoder fit", "500 samples", () => new OrdinalEncoder().fit(strLabels500_2d));
const oe = new OrdinalEncoder().fit(strLabels500_2d);
run(suite, "OrdinalEncoder transform", "500 samples", () => oe.transform(strLabels500_2d));

// ── LabelBinarizer ──────────────────────────────────────

run(suite, "LabelBinarizer fit", "500 samples", () => new LabelBinarizer().fit(strLabels500));
const lb = new LabelBinarizer().fit(strLabels500);
run(suite, "LabelBinarizer transform", "500 samples", () => lb.transform(strLabels500));

// ── trainTestSplit ──────────────────────────────────────

run(suite, "trainTestSplit", "200x5", () => trainTestSplit(X200, y200, { testSize: 0.2 }));
run(suite, "trainTestSplit", "500x10", () => trainTestSplit(X500, y500, { testSize: 0.2 }));
run(suite, "trainTestSplit", "1Kx10", () => trainTestSplit(X1k, y1k, { testSize: 0.2 }));

// ── KFold ───────────────────────────────────────────────

run(suite, "KFold (k=5)", "500 samples", () => {
  const kf = new KFold({ nSplits: 5 });
  for (const _ of kf.split(X500)) {
    /* iterate */
  }
});
run(suite, "KFold (k=5)", "1K samples", () => {
  const kf = new KFold({ nSplits: 5 });
  for (const _ of kf.split(X1k)) {
    /* iterate */
  }
});
run(suite, "KFold (k=10)", "1K samples", () => {
  const kf = new KFold({ nSplits: 10 });
  for (const _ of kf.split(X1k)) {
    /* iterate */
  }
});

// ── StratifiedKFold ─────────────────────────────────────

run(suite, "StratifiedKFold (k=5)", "500 samples", () => {
  const sf = new StratifiedKFold({ nSplits: 5 });
  for (const _ of sf.split(X500, y500)) {
    /* iterate */
  }
});
run(suite, "StratifiedKFold (k=5)", "1K samples", () => {
  const sf = new StratifiedKFold({ nSplits: 5 });
  for (const _ of sf.split(X1k, y1k)) {
    /* iterate */
  }
});

// ── LeaveOneOut ─────────────────────────────────────────

const Xsmall = makeNumeric(50, 3, 42);
run(suite, "LeaveOneOut", "50 samples", () => {
  const loo = new LeaveOneOut();
  for (const _ of loo.split(Xsmall)) {
    /* iterate */
  }
});

footer(suite, "deepbox-preprocess.json");
