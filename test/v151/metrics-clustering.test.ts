import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { calinskiHarabaszScore, daviesBouldinScore } from "../../src/metrics";
import { tensor } from "../../src/ndarray";

// scikit-learn 1.9.1 check_number_of_labels: with n_labels outside 2..n_samples-1 both
// davies_bouldin_score and calinski_harabasz_score raise ValueError("Number of labels is
// {n}. Valid values are 2 to n_samples - 1 (inclusive)").

const X = tensor([
  [1, 2],
  [1, 4],
  [1, 0],
  [4, 2],
  [4, 4],
  [4, 0],
]);

describe("clustering indices that need 2 to n_samples - 1 labels", () => {
  for (const [name, score] of [
    ["daviesBouldinScore", daviesBouldinScore],
    ["calinskiHarabaszScore", calinskiHarabaszScore],
  ] as const) {
    describe(name, () => {
      it("rejects a single label", () => {
        expect(() => score(X, tensor([0, 0, 0, 0, 0, 0]))).toThrow(InvalidParameterError);
        expect(() => score(X, tensor([0, 0, 0, 0, 0, 0]))).toThrow(
          "Number of labels is 1. Valid values are 2 to n_samples - 1 (inclusive)"
        );
      });

      it("rejects one label per sample", () => {
        expect(() => score(X, tensor([0, 1, 2, 3, 4, 5]))).toThrow(
          "Number of labels is 6. Valid values are 2 to n_samples - 1 (inclusive)"
        );
      });

      it("rejects empty input", () => {
        expect(() => score(tensor([]).reshape([0, 2]), tensor([]))).toThrow(InvalidParameterError);
      });

      it("accepts the boundary values 2 and n_samples - 1", () => {
        const two = name === "daviesBouldinScore" ? 0.8888888888888888 : 3.375;
        const most = name === "daviesBouldinScore" ? 0.3069648918191717 : 3.4375;
        expect(score(X, tensor([0, 0, 0, 1, 1, 1]))).toBeCloseTo(two, 12);
        expect(score(X, tensor([0, 0, 1, 2, 3, 4]))).toBeCloseTo(most, 12);
      });
    });
  }
});
