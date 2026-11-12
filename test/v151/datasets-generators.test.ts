import { describe, expect, it } from "vitest";
import { makeClassification } from "../../src/datasets";

// Reference counts come from scikit-learn 1.9 make_classification with
// n_clusters_per_class=1, flip_y=0, random_state=0 (np.bincount(y)).
function counts(y: ArrayLike<number>, k: number): number[] {
  const out = new Array<number>(k).fill(0);
  for (let i = 0; i < y.length; i++) out[y[i] as number] = (out[y[i] as number] as number) + 1;
  return out;
}

describe("DF-17 makeClassification class balance", () => {
  const cases: Array<[number, number, number[]]> = [
    [100, 2, [50, 50]],
    [100, 3, [34, 33, 33]],
    [10, 4, [3, 3, 2, 2]],
    [7, 3, [3, 2, 2]],
    [101, 5, [21, 20, 20, 20, 20]],
    [30, 3, [10, 10, 10]],
  ];
  for (const [n, k, expected] of cases) {
    it(`n=${n}, classes=${k} gives ${expected.join("/")} for every seed`, () => {
      for (let seed = 0; seed < 20; seed++) {
        const [, y] = makeClassification({
          nSamples: n,
          nClasses: k,
          nFeatures: 5,
          nInformative: 3,
          nRedundant: 0,
          flipY: 0,
          randomState: seed,
        });
        expect(counts(y.data as Int32Array, k)).toEqual(expected);
      }
    });
  }

  it("shuffles the labels instead of grouping them by class", () => {
    const [, y] = makeClassification({ nSamples: 100, flipY: 0, randomState: 3 });
    const labels = Array.from(y.data as Int32Array);
    expect(labels.slice(0, 50).some((v) => v === 1)).toBe(true);
    expect(labels.slice(50).some((v) => v === 0)).toBe(true);
  });

  it("stays deterministic for a given randomState", () => {
    const a = makeClassification({ nSamples: 40, randomState: 9 });
    const b = makeClassification({ nSamples: 40, randomState: 9 });
    expect(Array.from(a[1].data as Int32Array)).toEqual(Array.from(b[1].data as Int32Array));
    expect(Array.from(a[0].data as Float64Array)).toEqual(Array.from(b[0].data as Float64Array));
  });
});
