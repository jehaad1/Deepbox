import { describe, expect, it } from "vitest";
import { Ridge } from "../../src/ml";
import { tensor } from "../../src/ndarray";

// Reference values: scikit-learn 1.9.1 Ridge(alpha=..., solver="auto") on makeWide(30, 80).
const REF = {
  a1: {
    coefHead: [
      0.012420801445138064, 0.010292456499087336, 0.005374480364350717, 0.004781944030540788,
      0.013532024595529105, 0.03134118975597821,
    ],
    coefNorm: 0.8503141281301592,
    intercept: 0.19432847030399347,
    pred: [-0.10395796159831483, 0.7169663398483405, 1.1256565970007624, 0.7356607717901295],
  },
  a01: {
    coefHead: [
      0.012136013934813172, 0.0019508267447274075, -0.007845751152193992, -0.0072181693683072146,
      0.008715977869692582, 0.03665218119687906,
    ],
    coefNorm: 0.9121930817146271,
    intercept: 0.20830219944348907,
    pred: [-0.09880835088571746, 0.7530670489833745, 1.1664879015994707, 0.7468887906191888],
  },
  a1noint: {
    coefHead: [
      0.15459800565244777, 0.0966965393452172, 0.03362117645729165, -0.018363032257847148,
      -0.047546422299352305, -0.05139913928496943,
    ],
    coefNorm: 0.8943569570091588,
    intercept: 0,
    pred: [-0.16957586692053433, 0.6781569709240487, 1.1028837417245714, 0.7233559406317738],
  },
};

function makeWide(n: number, p: number) {
  const X = new Float64Array(n * p);
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      X[i * p + j] = Math.sin(i * 0.31 + j * 0.77) + 0.5 * Math.cos(i * j * 0.021 + 1);
    }
    y[i] = Math.sin(i * 0.9) + 0.3 * (X[i * p] ?? 0) - 0.2 * (X[i * p + 3] ?? 0);
  }
  return { X: tensor(X, { dtype: "float64" }).reshape([n, p]), y: tensor(y, { dtype: "float64" }) };
}

function norm(v: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += (v[i] as number) ** 2;
  return Math.sqrt(s);
}

describe("ML-6: Ridge with more features than samples", () => {
  const { X, y } = makeWide(30, 80);
  const cases: Array<[keyof typeof REF, { alpha: number; fitIntercept?: boolean }]> = [
    ["a1", { alpha: 1 }],
    ["a01", { alpha: 0.1 }],
    ["a1noint", { alpha: 1, fitIntercept: false }],
  ];

  for (const [name, opts] of cases) {
    it(`matches scikit-learn (${name})`, () => {
      const m = new Ridge(opts).fit(X, y);
      const coef = m.coef.data as Float64Array;
      const ref = REF[name];
      ref.coefHead.forEach((v, i) => {
        expect(coef[i]).toBeCloseTo(v, 10);
      });
      expect(norm(coef)).toBeCloseTo(ref.coefNorm, 10);
      expect(m.intercept).toBeCloseTo(ref.intercept, 10);
      const pred = m
        .predict(tensor((X.toArray() as number[][]).slice(0, 4), { dtype: "float64" }))
        .toArray() as number[];
      pred.forEach((v, i) => {
        expect(v).toBeCloseTo(ref.pred[i] as number, 10);
      });
    });

    it(`equals the SVD solution to 1e-8 (${name})`, () => {
      const a = new Ridge(opts).fit(X, y);
      const b = new Ridge({ ...opts, solver: "svd" }).fit(X, y);
      const ca = a.coef.data as Float64Array;
      const cb = b.coef.data as Float64Array;
      for (let j = 0; j < ca.length; j++)
        expect(Math.abs((ca[j] ?? 0) - (cb[j] ?? 0))).toBeLessThan(1e-8);
      expect(Math.abs(a.intercept - b.intercept)).toBeLessThan(1e-8);
    });
  }

  it("normalize still agrees with the SVD solution", () => {
    const a = new Ridge({ alpha: 0.5, normalize: true }).fit(X, y);
    const b = new Ridge({ alpha: 0.5, normalize: true, solver: "svd" }).fit(X, y);
    const ca = a.coef.data as Float64Array;
    const cb = b.coef.data as Float64Array;
    for (let j = 0; j < ca.length; j++)
      expect(Math.abs((ca[j] ?? 0) - (cb[j] ?? 0))).toBeLessThan(1e-8);
  });

  it("alpha = 0 with p > n returns the minimum-norm interpolating solution", () => {
    const small = makeWide(10, 25);
    const a = new Ridge({ alpha: 0 }).fit(small.X, small.y);
    const b = new Ridge({ alpha: 0, solver: "svd" }).fit(small.X, small.y);
    const ca = a.coef.data as Float64Array;
    const cb = b.coef.data as Float64Array;
    for (let j = 0; j < ca.length; j++)
      expect(Math.abs((ca[j] ?? 0) - (cb[j] ?? 0))).toBeLessThan(1e-6);
  });

  it("fits n = 100, p = 3000 in about a second (the primal system took 17 s)", () => {
    const big = makeWide(100, 3000);
    const start = performance.now();
    new Ridge().fit(big.X, big.y);
    expect(performance.now() - start).toBeLessThan(5000);
  }, 60000);
});
