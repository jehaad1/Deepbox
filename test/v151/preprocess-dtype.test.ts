import { describe, expect, it } from "vitest";
import { tensor } from "../../src/ndarray";
import * as P from "../../src/preprocess";

// scikit-learn keeps float32 input as float32 for the scalers, PolynomialFeatures,
// SplineTransformer and the imputers, and converts integer input to float64.

const rows = [
  [1.5, 2.5, 10],
  [2.5, 1.5, 20],
  [3.5, 4.5, 15],
  [4.5, 3.5, 30],
  [5.5, 6.5, 25],
  [6.5, 5.5, 5],
];

type Case = {
  make: () => {
    fitTransform(X: ReturnType<typeof tensor>): ReturnType<typeof tensor>;
    transform(X: ReturnType<typeof tensor>): ReturnType<typeof tensor>;
    inverseTransform?(X: ReturnType<typeof tensor>): ReturnType<typeof tensor>;
  };
  data?: number[][];
};

const cases: Record<string, Case> = {
  StandardScaler: { make: () => new P.StandardScaler() },
  MinMaxScaler: { make: () => new P.MinMaxScaler() },
  MaxAbsScaler: { make: () => new P.MaxAbsScaler() },
  RobustScaler: { make: () => new P.RobustScaler() },
  Normalizer: { make: () => new P.Normalizer() },
  QuantileTransformer: { make: () => new P.QuantileTransformer({ nQuantiles: 6 }) },
  PowerTransformer: { make: () => new P.PowerTransformer() },
  PolynomialFeatures: { make: () => new P.PolynomialFeatures({ degree: 2 }) },
  SplineTransformer: { make: () => new P.SplineTransformer({ nKnots: 3 }) },
  SimpleImputer: {
    make: () => new P.SimpleImputer(),
    data: [
      [1.5, Number.NaN, 10],
      [2.5, 1.5, 20],
      [3.5, 4.5, Number.NaN],
      [4.5, 3.5, 30],
    ],
  },
  KNNImputer: {
    make: () => new P.KNNImputer({ nNeighbors: 2 }),
    data: [
      [1.5, Number.NaN, 10],
      [2.5, 1.5, 20],
      [3.5, 4.5, Number.NaN],
      [4.5, 3.5, 30],
    ],
  },
};

describe("float32 input stays float32", () => {
  for (const [name, c] of Object.entries(cases)) {
    describe(name, () => {
      const data = c.data ?? rows;
      it("returns float32 for float32 input and agrees with the float64 result", () => {
        const f32 = c.make().fitTransform(tensor(data, { dtype: "float32" }));
        const f64 = c.make().fitTransform(tensor(data, { dtype: "float64" }));
        expect(f32.dtype).toBe("float32");
        expect(f64.dtype).toBe("float64");
        const a = f32.toArray() as number[][];
        const b = f64.toArray() as number[][];
        for (const [i, row] of b.entries()) {
          for (const [j, value] of row.entries()) {
            expect((a[i] as number[])[j]).toBeCloseTo(value, 4);
          }
        }
      });

      it("transform follows the dtype of the data it is given", () => {
        const est = c.make();
        est.fitTransform(tensor(data, { dtype: "float32" }));
        expect(est.transform(tensor(data, { dtype: "float32" })).dtype).toBe("float32");
        expect(est.transform(tensor(data, { dtype: "float64" })).dtype).toBe("float64");
      });

      if (c.make().inverseTransform !== undefined && name !== "SplineTransformer") {
        it("inverseTransform keeps float32", () => {
          const est = c.make();
          const x32 = tensor(data, { dtype: "float32" });
          const encoded = est.fitTransform(x32);
          expect(est.inverseTransform?.(encoded).dtype).toBe("float32");
        });
      }
    });
  }

  it("integer input becomes float64", () => {
    const ints = tensor(
      rows.map((r) => r.map((v) => Math.round(v))),
      { dtype: "int32" }
    );
    for (const name of ["StandardScaler", "MinMaxScaler", "RobustScaler", "PolynomialFeatures"]) {
      expect((cases[name] as Case).make().fitTransform(ints).dtype).toBe("float64");
    }
  });
});
