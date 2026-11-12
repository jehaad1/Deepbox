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
