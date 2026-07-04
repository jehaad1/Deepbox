/**
 * Benchmark 10 — Preprocessing
 * Deepbox vs scikit-learn
 */

import { tensor } from "deepbox/ndarray";
import {
  Binarizer,
  CountVectorizer,
  f_classif,
  f_regression,
  GroupKFold,
  GroupShuffleSplit,
  HashingVectorizer,
  KBinsDiscretizer,
  KFold,
  KNNImputer,
  LabelBinarizer,
  LabelEncoder,
  LeaveOneOut,
  MaxAbsScaler,
  MinMaxScaler,
  MissingIndicator,
  MultiLabelBinarizer,
  mutual_info_classif,
  mutual_info_regression,
  Normalizer,
  OneHotEncoder,
  OrdinalEncoder,
  PolynomialFeatures,
  PowerTransformer,
  QuantileTransformer,
  RepeatedKFold,
  RobustScaler,
  SelectKBest,
  ShuffleSplit,
  SimpleImputer,
  SplineTransformer,
  StandardScaler,
  StratifiedKFold,
  StratifiedShuffleSplit,
  TfidfVectorizer,
  TimeSeriesSplit,
  trainTestSplit,
  VarianceThreshold,
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
const XFeat1k = makeNumeric(1000, 20, 84);
const Xpos500 = makePositive(500, 10, 42);
const _Xpos1k = makePositive(1000, 10, 42);
const XSpline1k = makeNumeric(1000, 1, 21);

function makeLabels(n: number, k: number, seed: number) {
  const rand = seededRng(seed);
  const labels: number[] = [];
  for (let i = 0; i < n; i++) labels.push(Math.floor(rand() * k));
  return tensor(labels);
}

const y200 = makeLabels(200, 3, 42);
const y500 = makeLabels(500, 5, 42);
const y1k = makeLabels(1000, 5, 42);
const yFeat1k = makeLabels(1000, 3, 84);
const groups1k = tensor(Array.from({ length: 1000 }, (_, i) => Math.floor(i / 10)));
const yReg1k = tensor(Array.from({ length: 1000 }, (_, i) => ((i * 7) % 31) / 3));
const multiLabelTargets = Array.from({ length: 1000 }, (_, i) => {
  const labels = [`label_${i % 5}`];
  if (i % 3 === 0) labels.push(`group_${i % 4}`);
  return labels;
});

const Xmissing500 = tensor(
  Array.from({ length: 500 }, (_, i) =>
    Array.from({ length: 10 }, (_, j) =>
      (i + j) % 7 === 0 ? Number.NaN : ((i * 13 + j * 7) % 97) / 10
    )
  )
);

const documents1k = Array.from({ length: 1000 }, (_, i) => {
  const animals = ["cat", "dog", "fox", "owl", "yak"];
  const verbs = ["jumps", "runs", "sleeps", "eats", "looks"];
  const places = ["garden", "forest", "city", "desert", "river"];
  return `${animals[i % 5]} ${verbs[(i * 3) % 5]} in the ${places[(i * 7) % 5]}`;
});

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

// ── Advanced v1.0.0 Preprocessing ───────────────────────

run(suite, "KBinsDiscretizer fit+transform", "1Kx10", () =>
  new KBinsDiscretizer({ nBins: 8, strategy: "quantile" }).fitTransform(X1k)
);

run(suite, "SplineTransformer fit+transform", "1Kx1", () =>
  new SplineTransformer({ nKnots: 5, degree: 3 }).fitTransform(XSpline1k)
);

run(suite, "KNNImputer fit+transform", "500x10", () =>
  new KNNImputer({ nNeighbors: 3 }).fitTransform(Xmissing500)
);

run(suite, "MissingIndicator fit+transform", "500x10", () =>
  new MissingIndicator().fitTransform(Xmissing500)
);

run(suite, "SelectKBest fit+transform", "1Kx20", () =>
  new SelectKBest({ scoreFunc: f_classif, k: 10 }).fitTransform(XFeat1k, yFeat1k)
);

run(suite, "mutual_info_classif", "1Kx20", () => mutual_info_classif(XFeat1k, yFeat1k));
run(suite, "f_regression", "1Kx20", () => f_regression(XFeat1k, yReg1k));
run(suite, "mutual_info_regression", "1Kx20", () => mutual_info_regression(XFeat1k, yReg1k));

run(suite, "SimpleImputer fit+transform", "500x10", () =>
  new SimpleImputer({ strategy: "mean" }).fitTransform(Xmissing500)
);

run(suite, "Binarizer fit+transform", "1Kx10", () =>
  new Binarizer({ threshold: 0 }).fitTransform(X1k)
);

run(suite, "PolynomialFeatures fit+transform", "200x5→deg2", () =>
  new PolynomialFeatures({ degree: 2, includeBias: false }).fitTransform(X200)
);

run(suite, "VarianceThreshold fit+transform", "1Kx20", () =>
  new VarianceThreshold({ threshold: 0.1 }).fitTransform(XFeat1k)
);

run(suite, "MultiLabelBinarizer fit+transform", "1K label-sets", () =>
  new MultiLabelBinarizer().fitTransform(multiLabelTargets)
);

run(suite, "CountVectorizer fit+transform", "1K docs", () =>
  new CountVectorizer().fitTransformText(documents1k)
);

run(suite, "TfidfVectorizer fit+transform", "1K docs", () =>
  new TfidfVectorizer().fitTransformText(documents1k)
);

run(suite, "HashingVectorizer transform", "1K docs", () =>
  new HashingVectorizer({ nFeatures: 1024 }).transformText(documents1k)
);

run(suite, "ShuffleSplit (5)", "1K samples", () => {
  const splitter = new ShuffleSplit({ nSplits: 5, testSize: 0.2, randomState: 42 });
  for (const _ of splitter.split(X1k)) {
    /* iterate */
  }
});

run(suite, "StratifiedShuffleSplit (5)", "1K samples", () => {
  const splitter = new StratifiedShuffleSplit({ nSplits: 5, testSize: 0.2, randomState: 42 });
  for (const _ of splitter.split(X1k, y1k)) {
    /* iterate */
  }
});

run(suite, "TimeSeriesSplit (5)", "1K samples", () => {
  const splitter = new TimeSeriesSplit({ nSplits: 5 });
  for (const _ of splitter.split(X1k)) {
    /* iterate */
  }
});
run(suite, "GroupKFold (5)", "1K samples", () => {
  const splitter = new GroupKFold({ nSplits: 5 });
  for (const _ of splitter.split(X1k, undefined, groups1k)) {
    /* iterate */
  }
});
run(suite, "RepeatedKFold (5x3)", "1K samples", () => {
  const splitter = new RepeatedKFold({ nSplits: 5, nRepeats: 3, randomState: 42 });
  for (const _ of splitter.split(X1k)) {
    /* iterate */
  }
});

run(
  suite,
  "GroupShuffleSplit (5)",
  "1K samples",
  () => {
    const splitter = new GroupShuffleSplit({ nSplits: 5, testSize: 0.2, randomState: 42 });
    for (const _ of splitter.split(X1k, undefined, Array.from(groups1k.data, Number))) {
      /* iterate */
    }
  },
  { comparable: false, tags: ["deepbox-only"] }
);

footer(suite, "deepbox-preprocess.json");
