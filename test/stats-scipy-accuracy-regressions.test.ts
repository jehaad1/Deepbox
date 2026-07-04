import { describe, expect, it } from "vitest";
import { tensor } from "../src/ndarray";
import { erf, erfc, normalCdf, normalPpf } from "../src/stats/_internal";
import { beta, chi2, f, gamma, norm, t } from "../src/stats/distributions";
import { fligner, shapiro } from "../src/stats/tests";

// These tests pin behaviors that were previously incorrect or low-accuracy,
// validated against scipy.stats (reference values in comments).
describe("stats accuracy regressions (vs scipy)", () => {
  describe("normalCdf / erf high-accuracy", () => {
    it("matches scipy norm.cdf to ~1e-7", () => {
      // scipy.stats.norm.cdf
      expect(normalCdf(0)).toBeCloseTo(0.5, 12);
      expect(normalCdf(1.96)).toBeCloseTo(0.9750021048517795, 6);
      expect(normalCdf(2)).toBeCloseTo(0.9772498680518208, 6);
      expect(normalCdf(-1.5)).toBeCloseTo(0.06680720126885807, 6);
      expect(normalCdf(3)).toBeCloseTo(0.9986501019683699, 6);
    });

    it("erf and erfc are consistent and accurate", () => {
      expect(erf(0)).toBeCloseTo(0, 12);
      // math.erf(1) = 0.8427007929497149
      expect(erf(1)).toBeCloseTo(0.8427007929497149, 6);
      expect(erfc(1)).toBeCloseTo(1 - 0.8427007929497149, 6);
      for (const x of [-2, -0.3, 0.7, 2.5]) {
        expect(erf(x) + erfc(x)).toBeCloseTo(1, 10);
      }
    });
  });

  describe("normalPpf is the accurate inverse of normalCdf", () => {
    it("round-trips against normalCdf", () => {
      // normalPpf refines against normalCdf, so the round-trip is consistent
      // to ~1e-9 even though absolute accuracy is bounded by erfc (~1e-7).
      for (const p of [0.01, 0.25, 0.5, 0.7, 0.975, 0.999]) {
        expect(normalCdf(normalPpf(p))).toBeCloseTo(p, 7);
      }
      // scipy.stats.norm.ppf(0.975) = 1.959963984540054 (matched to ~1e-7)
      expect(normalPpf(0.975)).toBeCloseTo(1.959963984540054, 6);
    });
  });

  describe("Shapiro-Wilk p-value direction (was inverted)", () => {
    it("reports a small p-value for clearly non-normal data", () => {
      // scipy.stats.shapiro -> W=0.78881, p=0.0067038
      const res = shapiro(tensor([148, 154, 158, 160, 161, 162, 166, 170, 182, 195, 236]));
      expect(res.statistic).toBeCloseTo(0.7888146948631714, 4);
      expect(res.pvalue).toBeCloseTo(0.006703814061898779, 4);
      expect(res.pvalue).toBeLessThan(0.05);
    });
  });

  describe("Fligner-Killeen matches scipy", () => {
    it("uses half-normal scores", () => {
      // scipy.stats.fligner(...) -> statistic=9.563383, pvalue=0.0083818
      const res = fligner([
        tensor([1, 2, 3, 4, 5, 6, 7, 8]),
        tensor([2, 4, 6, 8, 10, 12, 14, 16]),
        tensor([1, 1, 1, 2, 2, 3, 3, 4]),
      ]);
      expect(res.statistic).toBeCloseTo(9.563383016928093, 5);
      expect(res.pvalue).toBeCloseTo(0.008381809068724117, 6);
    });
  });

  describe("differential entropies use digamma (were approximations)", () => {
    it("matches scipy .entropy()", () => {
      expect(norm(0, 1).entropy()).toBeCloseTo(1.4189385332046727, 9);
      expect(t(5).entropy()).toBeCloseTo(1.627502672414396, 6);
      expect(chi2(4).entropy()).toBeCloseTo(2.270362845461478, 6);
      expect(gamma(2, 1).entropy()).toBeCloseTo(1.5772156649015328, 6);
      expect(beta(2, 3).entropy()).toBeCloseTo(-0.2349066497880008, 6);
      expect(f(5, 10).entropy()).toBeCloseTo(1.1307598049090615, 5);
    });
  });
});
