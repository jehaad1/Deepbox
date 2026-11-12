import { describe, expect, it } from "vitest";
import { LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis } from "../../src/ml";
import { tensor } from "../../src/ndarray";

// Reference values: scikit-learn 1.9.1 LinearDiscriminantAnalysis() and
// QuadraticDiscriminantAnalysis() fitted on makeData(36), evaluated on the first 5 rows.
const LDA_PROBA = [
  [0.979687849680908, 0.020311133451555916, 1.0168675360325304e-6],
  [0.0004790849700434409, 0.3800244614886829, 0.6194964535412736],
  [2.0426303770219153e-9, 0.0009812465950500495, 0.9990187513623195],
  [0.943732594233714, 0.056258354815812026, 9.050950473905996e-6],
  [0.6183263187614301, 0.38131484409559074, 0.0003588371429791508],
];
const LDA_TRANSFORM = [
  [2.772068119966927, 0.8877753400739896],
  [-1.4546056814764985, -0.47400636203417484],
  [-4.046692174646778, -0.8446664916955144],
  [2.323229698030437, 1.2025560787949838],
  [1.4955866193257277, -0.9923884301446674],
];
const LDA_EVR = [0.9989418367204802, 0.0010581632795196778];
const QDA_PROBA = [
  [0.9767078128447433, 0.023292128915220298, 5.824003638930909e-8],
  [0.0004333420710164117, 0.41516298280198194, 0.5844036751270016],
  [1.3339607145475903e-9, 0.0009689402241765274, 0.9990310584418628],
  [0.9038409581041996, 0.09615824545603158, 7.96439768794076e-7],
  [0.5318797661011094, 0.4680996404723736, 2.0593426517033332e-5],
];

function makeData(n: number) {
  const rows: number[][] = [];
  const labels: number[] = [];
  for (let i = 0; i < n; i++) {
    const y = i % 3;
    rows.push([
      Math.sin(i * 0.9) + y * 1.1,
      Math.cos(i * 1.7) * 0.8 - y * 0.4,
      Math.sin(i * 2.3 + 1) + 0.3 * y,
      (i % 7) * 0.2 + Math.cos(i) * 0.5,
    ]);
    labels.push(y);
  }
  return { X: tensor(rows, { dtype: "float64" }), y: tensor(labels) };
}

function maxAbsDiff(a: number[][], b: number[][], flipColumns = false): number {
  let worst = 0;
  const cols = a[0]?.length ?? 0;
  for (let c = 0; c < cols; c++) {
    // A discriminant axis is defined up to its sign.
    const sign = flipColumns && (a[0]?.[c] ?? 0) * (b[0]?.[c] ?? 0) < 0 ? -1 : 1;
    for (let r = 0; r < a.length; r++) {
      worst = Math.max(worst, Math.abs((a[r]?.[c] ?? 0) - sign * (b[r]?.[c] ?? 0)));
    }
  }
  return worst;
}

describe("ML-5: LDA and QDA use scikit-learn's divisors", () => {
  const { X, y } = makeData(36);
  const head = (t: ReturnType<typeof tensor>): ReturnType<typeof tensor> =>
    tensor((t.toArray() as number[][]).slice(0, 5), { dtype: "float64" });

  it("LDA predictProba matches scikit-learn to 1e-10", () => {
    const m = new LinearDiscriminantAnalysis().fit(X, y);
    const p = m.predictProba(head(X)).toArray() as number[][];
    expect(maxAbsDiff(p, LDA_PROBA)).toBeLessThan(1e-10);
  });

  it("LDA transform and explainedVarianceRatio match scikit-learn to 1e-10", () => {
    const m = new LinearDiscriminantAnalysis().fit(X, y);
    const t = m.transform(head(X)).toArray() as number[][];
    expect(maxAbsDiff(t, LDA_TRANSFORM, true)).toBeLessThan(1e-10);
    const evr = Array.from(m.explainedVarianceRatio ?? []);
    expect(Math.abs((evr[0] ?? 0) - LDA_EVR[0]!)).toBeLessThan(1e-10);
    expect(Math.abs((evr[1] ?? 0) - LDA_EVR[1]!)).toBeLessThan(1e-10);
  });

  it("QDA predictProba matches scikit-learn to 1e-10", () => {
    const m = new QuadraticDiscriminantAnalysis().fit(X, y);
    const p = m.predictProba(head(X)).toArray() as number[][];
    expect(maxAbsDiff(p, QDA_PROBA)).toBeLessThan(1e-10);
  });
});

// Reference values: scikit-learn 1.9.1 on makeWide() (600 samples, 100 features, 3 classes),
// evaluated on the first 3 rows.
const WIDE_LDA_PROBA = [
  [0.0013523280886971617, 0.998646756030161, 9.158811417070679e-7],
  [1.6361509869560515e-10, 0.7131798909551794, 0.2868201088812055],
  [5.325222731130655e-21, 7.71681025013e-8, 0.9999999228318974],
];
const WIDE_LDA_TRANSFORM_ABS = [
  [0.8272360420582603, 0.7440424720542437],
  [2.4611812719899318, 1.5975897481995014],
  [5.387530333907887, 1.0466790384522173],
];
const WIDE_QDA_PROBA = [
  [0.9999999999710338, 2.637513505127178e-25, 2.8966220252481e-11],
  [8.199585901709026e-35, 0.9999998996962552, 1.0030374484563777e-7],
  [2.365370874950055e-64, 3.203276708555997e-94, 1.0],
];

function makeWide(n: number, f: number) {
  const d = new Float64Array(n * f);
  const labels = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    labels[i] = i % 3;
    for (let j = 0; j < f; j++) {
      d[i * f + j] =
        Math.sin(i * 0.37 + j * 1.1) +
        0.5 * Math.cos(i * j * 0.013 + 1) +
        0.4 * Math.sin(i * 0.91 * (j + 1) * 0.1) +
        (j < 5 ? (labels[i] ?? 0) * 0.6 : 0);
    }
  }
  return { X: tensor(d, { dtype: "float64" }).reshape([n, f]), y: tensor(labels) };
}

describe("ML-7: LDA and QDA eigen-decomposition on wide data", () => {
  const { X, y } = makeWide(600, 100);
  const head = tensor((X.toArray() as number[][]).slice(0, 3), { dtype: "float64" });

  it("LDA matches scikit-learn to 1e-8 with 100 features", () => {
    const m = new LinearDiscriminantAnalysis().fit(X, y);
    expect(maxAbsDiff(m.predictProba(head).toArray() as number[][], WIDE_LDA_PROBA)).toBeLessThan(
      1e-8
    );
    const t = (m.transform(head).toArray() as number[][]).map((r) => r.map(Math.abs));
    expect(maxAbsDiff(t, WIDE_LDA_TRANSFORM_ABS)).toBeLessThan(1e-8);
  });

  it("QDA matches scikit-learn to 1e-8 with 100 features", () => {
    const m = new QuadraticDiscriminantAnalysis().fit(X, y);
    expect(maxAbsDiff(m.predictProba(head).toArray() as number[][], WIDE_QDA_PROBA)).toBeLessThan(
      1e-8
    );
  });

  it("fits wide data with the symmetric eigensolver", () => {
    // Cyclic Jacobi took 18 s (LDA) and 45 s (QDA) at 1000 x 400; benchmarks track the speed.
    const big = makeWide(300, 120);
    const lda = new LinearDiscriminantAnalysis().fit(big.X, big.y);
    const qda = new QuadraticDiscriminantAnalysis({ regParam: 0.1 }).fit(big.X, big.y);
    for (const p of lda.predictProba(big.X).toArray() as number[][]) {
      expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    }
    expect(qda.predict(big.X).shape).toEqual([300]);
  });
});
