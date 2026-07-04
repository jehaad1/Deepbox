import { describe, expect, it } from "vitest";
import {
  adjustedMutualInfoScore,
  adjustedRandScore,
  calinskiHarabaszScore,
  completenessScore,
  daviesBouldinScore,
  fowlkesMallowsScore,
  homogeneityScore,
  normalizedMutualInfoScore,
  silhouetteSamples,
  silhouetteScore,
  vMeasureScore,
} from "../src/metrics/clustering";
import { tensor } from "../src/ndarray";

const X = tensor([
  [1, 0],
  [1.1, 0],
  [0, 1],
  [0, 1.1],
  [5, 5],
  [5.1, 5],
]);
const labels = tensor([0, 0, 1, 1, 2, 2]);
const labelsTrue = tensor([0, 0, 1, 1, 2, 2]);
const labelsPred = tensor([0, 0, 1, 1, 2, 2]);

// ────── silhouetteScore ──────
describe("silhouetteScore", () => {
  it("perfect clusters", () => {
    const s = silhouetteScore(X, labels);
    expect(s).toBeGreaterThan(0);
    expect(s).toBeLessThanOrEqual(1);
  });

  it("two clusters three samples", () => {
    const X2 = tensor([
      [0, 0],
      [0.1, 0],
      [5, 5],
    ]);
    const l2 = tensor([0, 0, 1]);
    const s = silhouetteScore(X2, l2);
    expect(typeof s).toBe("number");
  });
});

// ────── silhouetteSamples ──────
describe("silhouetteSamples", () => {
  it("returns per-sample scores", () => {
    const samples = silhouetteSamples(X, labels);
    expect(samples.shape).toEqual([6]);
  });
});

// ────── daviesBouldinScore ──────
describe("daviesBouldinScore", () => {
  it("well-separated clusters", () => {
    const s = daviesBouldinScore(X, labels);
    expect(s).toBeGreaterThanOrEqual(0);
  });
});

// ────── calinskiHarabaszScore ──────
describe("calinskiHarabaszScore", () => {
  it("well-separated clusters", () => {
    const s = calinskiHarabaszScore(X, labels);
    expect(s).toBeGreaterThan(0);
  });
});

// ────── adjustedRandScore ──────
describe("adjustedRandScore", () => {
  it("perfect agreement", () => {
    const s = adjustedRandScore(labelsTrue, labelsPred);
    expect(s).toBeCloseTo(1, 1);
  });

  it("permuted labels still perfect", () => {
    const pred = tensor([1, 1, 0, 0, 2, 2]);
    const s = adjustedRandScore(labelsTrue, pred);
    expect(s).toBeCloseTo(1, 1);
  });

  it("random labels low score", () => {
    const pred = tensor([0, 1, 0, 1, 0, 1]);
    const s = adjustedRandScore(labelsTrue, pred);
    expect(s).toBeLessThan(1);
  });
});

// ────── adjustedMutualInfoScore ──────
describe("adjustedMutualInfoScore", () => {
  it("perfect agreement", () => {
    const s = adjustedMutualInfoScore(labelsTrue, labelsPred);
    expect(s).toBeCloseTo(1, 1);
  });

  it("imperfect agreement", () => {
    const pred = tensor([0, 1, 0, 1, 0, 1]);
    const s = adjustedMutualInfoScore(labelsTrue, pred);
    expect(s).toBeLessThan(1);
  });
});

// ────── normalizedMutualInfoScore ──────
describe("normalizedMutualInfoScore", () => {
  it("perfect agreement", () => {
    const s = normalizedMutualInfoScore(labelsTrue, labelsPred);
    expect(s).toBeCloseTo(1, 1);
  });

  it("imperfect", () => {
    const pred = tensor([0, 0, 0, 1, 1, 1]);
    const s = normalizedMutualInfoScore(labelsTrue, pred);
    expect(s).toBeGreaterThanOrEqual(0);
    expect(s).toBeLessThanOrEqual(1);
  });
});

// ────── fowlkesMallowsScore ──────
describe("fowlkesMallowsScore", () => {
  it("perfect agreement", () => {
    const s = fowlkesMallowsScore(labelsTrue, labelsPred);
    expect(s).toBeCloseTo(1, 1);
  });

  it("imperfect", () => {
    const pred = tensor([0, 1, 0, 1, 0, 1]);
    const s = fowlkesMallowsScore(labelsTrue, pred);
    expect(s).toBeGreaterThanOrEqual(0);
    expect(s).toBeLessThanOrEqual(1);
  });
});

// ────── homogeneityScore ──────
describe("homogeneityScore", () => {
  it("perfect homogeneity", () => {
    const s = homogeneityScore(labelsTrue, labelsPred);
    expect(s).toBeCloseTo(1, 1);
  });

  it("imperfect", () => {
    const pred = tensor([0, 0, 0, 1, 1, 1]);
    const s = homogeneityScore(labelsTrue, pred);
    expect(s).toBeGreaterThanOrEqual(0);
  });
});

// ────── completenessScore ──────
describe("completenessScore", () => {
  it("perfect completeness", () => {
    const s = completenessScore(labelsTrue, labelsPred);
    expect(s).toBeCloseTo(1, 1);
  });

  it("imperfect", () => {
    const pred = tensor([0, 0, 0, 1, 1, 1]);
    const s = completenessScore(labelsTrue, pred);
    expect(s).toBeGreaterThanOrEqual(0);
  });
});

// ────── vMeasureScore ──────
describe("vMeasureScore", () => {
  it("perfect v-measure", () => {
    const s = vMeasureScore(labelsTrue, labelsPred);
    expect(s).toBeCloseTo(1, 1);
  });

  it("custom beta", () => {
    const s = vMeasureScore(labelsTrue, labelsPred, 2.0);
    expect(s).toBeGreaterThan(0);
  });

  it("throws for invalid beta", () => {
    expect(() => vMeasureScore(labelsTrue, labelsPred, -1)).toThrow();
    expect(() => vMeasureScore(labelsTrue, labelsPred, 0)).toThrow();
    expect(() => vMeasureScore(labelsTrue, labelsPred, Infinity)).toThrow();
  });

  it("all same cluster", () => {
    const pred = tensor([0, 0, 0, 0, 0, 0]);
    const s = vMeasureScore(labelsTrue, pred);
    expect(typeof s).toBe("number");
  });
});
