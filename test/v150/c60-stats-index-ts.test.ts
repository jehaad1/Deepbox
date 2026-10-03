import { describe, expect, it } from "vitest";
import { InvalidParameterError } from "../../src/core";
import { tensor } from "../../src/ndarray";
import * as stats from "../../src/stats";

// Regression tests for the v1.5.0 audit of src/stats/tests.ts, kde.ts, multiple.ts, power.ts
// and index.ts. Reference values were computed with SciPy 1.17 (scipy.stats, scipy.special).

const T = (a: number[]) => tensor(a, { dtype: "float64" });

function expectClose(actual: number, expected: number, rtol = 1e-9): void {
  if (Number.isNaN(expected)) {
    expect(actual).toBeNaN();
    return;
  }
  if (!Number.isFinite(expected)) {
    expect(actual).toBe(expected);
    return;
  }
  const tol = Math.max(rtol * Math.abs(expected), 1e-300);
  expect(Math.abs(actual - expected)).toBeLessThanOrEqual(tol);
}

describe("t tests (SciPy references)", () => {
  it("ttest_1samp alternative two-sided", () => {
    const res = stats.ttest_1samp(T([2.1, 1.9, 2.4, 2.2, 2, 2.6, 1.7, 2.3]), 2.4, "two-sided");
    expectClose(res.statistic, -2.4565184222025875);
    expectClose(res.pvalue, 0.04368772120886764);
  });
  it("ttest_ind pooled alternative two-sided", () => {
    const res = stats.ttest_ind(
      T([2.1, 1.9, 2.4, 2.2, 2, 2.6, 1.7, 2.3]),
      T([1.8, 2.5, 2.9, 3.1, 2.7, 3.3, 2.2, 3, 2.8]),
      true,
      "two-sided"
    );
    expectClose(res.statistic, -2.8657755814539856);
    expectClose(res.pvalue, 0.01178419165770738);
  });
  it("ttest_ind Welch alternative two-sided", () => {
    const res = stats.ttest_ind(
      T([2.1, 1.9, 2.4, 2.2, 2, 2.6, 1.7, 2.3]),
      T([1.8, 2.5, 2.9, 3.1, 2.7, 3.3, 2.2, 3, 2.8]),
      false,
      "two-sided"
    );
    expectClose(res.statistic, -2.948242233789349);
    expectClose(res.pvalue, 0.010963189151821777);
  });
  it("ttest_rel alternative two-sided", () => {
    const res = stats.ttest_rel(
      T([10.2, 11.5, 9.8, 12.1, 10.9, 11.3, 10.1]),
      T([9.9, 10.8, 10.1, 11, 10.2, 10.9, 9.5]),
      "two-sided"
    );
    expectClose(res.statistic, 3.034884893334422);
    expectClose(res.pvalue, 0.022953277643220527);
  });
  it("ttest_1samp alternative less", () => {
    const res = stats.ttest_1samp(T([2.1, 1.9, 2.4, 2.2, 2, 2.6, 1.7, 2.3]), 2.4, "less");
    expectClose(res.statistic, -2.4565184222025875);
    expectClose(res.pvalue, 0.02184386060443382);
  });
  it("ttest_ind pooled alternative less", () => {
    const res = stats.ttest_ind(
      T([2.1, 1.9, 2.4, 2.2, 2, 2.6, 1.7, 2.3]),
      T([1.8, 2.5, 2.9, 3.1, 2.7, 3.3, 2.2, 3, 2.8]),
      true,
      "less"
    );
    expectClose(res.statistic, -2.8657755814539856);
    expectClose(res.pvalue, 0.00589209582885369);
  });
  it("ttest_ind Welch alternative less", () => {
    const res = stats.ttest_ind(
      T([2.1, 1.9, 2.4, 2.2, 2, 2.6, 1.7, 2.3]),
      T([1.8, 2.5, 2.9, 3.1, 2.7, 3.3, 2.2, 3, 2.8]),
      false,
      "less"
    );
    expectClose(res.statistic, -2.948242233789349);
    expectClose(res.pvalue, 0.005481594575910889);
  });
  it("ttest_rel alternative less", () => {
    const res = stats.ttest_rel(
      T([10.2, 11.5, 9.8, 12.1, 10.9, 11.3, 10.1]),
      T([9.9, 10.8, 10.1, 11, 10.2, 10.9, 9.5]),
      "less"
    );
    expectClose(res.statistic, 3.034884893334422);
    expectClose(res.pvalue, 0.9885233611783897);
  });
  it("ttest_1samp alternative greater", () => {
    const res = stats.ttest_1samp(T([2.1, 1.9, 2.4, 2.2, 2, 2.6, 1.7, 2.3]), 2.4, "greater");
    expectClose(res.statistic, -2.4565184222025875);
    expectClose(res.pvalue, 0.9781561393955662);
  });
  it("ttest_ind pooled alternative greater", () => {
    const res = stats.ttest_ind(
      T([2.1, 1.9, 2.4, 2.2, 2, 2.6, 1.7, 2.3]),
      T([1.8, 2.5, 2.9, 3.1, 2.7, 3.3, 2.2, 3, 2.8]),
      true,
      "greater"
    );
    expectClose(res.statistic, -2.8657755814539856);
    expectClose(res.pvalue, 0.9941079041711464);
  });
  it("ttest_ind Welch alternative greater", () => {
    const res = stats.ttest_ind(
      T([2.1, 1.9, 2.4, 2.2, 2, 2.6, 1.7, 2.3]),
      T([1.8, 2.5, 2.9, 3.1, 2.7, 3.3, 2.2, 3, 2.8]),
      false,
      "greater"
    );
    expectClose(res.statistic, -2.948242233789349);
    expectClose(res.pvalue, 0.9945184054240891);
  });
  it("ttest_rel alternative greater", () => {
    const res = stats.ttest_rel(
      T([10.2, 11.5, 9.8, 12.1, 10.9, 11.3, 10.1]),
      T([9.9, 10.8, 10.1, 11, 10.2, 10.9, 9.5]),
      "greater"
    );
    expectClose(res.statistic, 3.034884893334422);
    expectClose(res.pvalue, 0.011476638821610263);
  });
  it("ttest_1samp keeps precision for a tiny p-value", () => {
    const res = stats.ttest_1samp(T([100.1, 100.2, 99.9, 100, 100.3]), 0);
    expectClose(res.statistic, 1415.6277759354882);
    expectClose(res.pvalue, 1.494009999960631e-12);
  });
  it("ttest_1samp one-sided p-value near 1", () => {
    const res = stats.ttest_1samp(T([100.1, 100.2, 99.9, 100, 100.3]), 0, "less");
    expectClose(res.statistic, 1415.6277759354882);
    expectClose(res.pvalue, 0.999999999999253);
  });
});

describe("chisquare (SciPy references)", () => {
  it("chisquare uniform expectation", () => {
    const res = stats.chisquare(T([16, 18, 16, 14, 12, 12]));
    expectClose(res.statistic, 2.0);
    expectClose(res.pvalue, 0.8491450360846096);
  });
  it("chisquare ddof reduces the degrees of freedom", () => {
    const res = stats.chisquare(T([16, 18, 16, 14, 12, 12]), undefined, 2);
    expectClose(res.statistic, 2.0);
    expectClose(res.pvalue, 0.5724067044708798);
  });
  it("chisquare p-value in the far tail", () => {
    const res = stats.chisquare(T([500, 3, 1, 2]));
    expectClose(res.statistic, 1470.3952569169962);
    expectClose(res.pvalue, 0.0);
  });
  it("chisquare with expected frequencies", () => {
    const res = stats.chisquare(T([300, 50, 2]), T([250, 90, 12]));
    expectClose(res.statistic, 36.111111111111114);
    expectClose(res.pvalue, 1.4406943550769141e-8);
  });
});

describe("kstest (SciPy references)", () => {
  it("kstest exact p-value (documented example)", () => {
    const res = stats.kstest(T([0.1, -0.3, 1.2, 0.8, -1.5, 0.4, 2.2]), "norm");
    expectClose(res.statistic, 0.2541135515627433);
    expectClose(res.pvalue, 0.6692838264713744);
  });
  it("kstest one-sided less uses the exact Smirnov distribution", () => {
    const res = stats.kstest(T([0.1, -0.3, 1.2, 0.8, -1.5, 0.4, 2.2]), "norm", {
      alternative: "less",
    });
    expectClose(res.statistic, 0.2541135515627433);
    expectClose(res.pvalue, 0.3467039071233188);
  });
  it("kstest one-sided greater uses the exact Smirnov distribution", () => {
    const res = stats.kstest(T([0.1, -0.3, 1.2, 0.8, -1.5, 0.4, 2.2]), "norm", {
      alternative: "greater",
    });
    expectClose(res.statistic, 0.07604994158828479);
    expectClose(res.pvalue, 0.8819425347811334);
  });
  it("kstest asymptotic method", () => {
    const res = stats.kstest(T([0.1, -0.3, 1.2, 0.8, -1.5, 0.4, 2.2]), "norm", {
      method: "asymptotic",
    });
    expectClose(res.statistic, 0.2541135515627433);
    expectClose(res.pvalue, 0.7566787255708709);
  });
  it("kstest single observation", () => {
    const res = stats.kstest(T([0.3]), "norm");
    expectClose(res.statistic, 0.6179114221889526);
    expectClose(res.pvalue, 0.7641771556220949);
  });
  it("kstest two-sided exact p-value, n = 30", () => {
    const res = stats.kstest(
      T(Array.from({ length: 30 }, (_, i) => (((i * 7) % 31) / 31) * 4 - 2 + 0.0)),
      "norm"
    );
    expectClose(res.statistic, 0.13341336552052474);
    expectClose(res.pvalue, 0.6123288228127332);
  });
  it("kstest two-sided exact p-value, n = 500", () => {
    const res = stats.kstest(
      T(Array.from({ length: 500 }, (_, i) => (((i * 7919) % 10007) / 10007) * 4 - 2 + 0.0)),
      "norm"
    );
    expectClose(res.statistic, 0.0949931635142266);
    expectClose(res.pvalue, 0.00022303077791962204);
  });
  it("kstest two-sided exact p-value, n = 2000", () => {
    const res = stats.kstest(
      T(Array.from({ length: 2000 }, (_, i) => (((i * 7919) % 10007) / 10007) * 4 - 2 + 0.05)),
      "norm"
    );
    expectClose(res.statistic, 0.1056673993517);
    expectClose(res.pvalue, 6.72123366964778e-20);
  });
  it("kstest two-sided exact p-value, n = 12000", () => {
    const res = stats.kstest(
      T(Array.from({ length: 12000 }, (_, i) => (((i * 7919) % 10007) / 10007) * 4 - 2 + 0.02)),
      "norm"
    );
    expectClose(res.statistic, 0.0966956580146755);
    expectClose(res.pvalue, 4.1214432803309706e-98);
  });
  it("kstest with a custom distribution function", () => {
    const res = stats.kstest(T([0.05, 0.2, 0.33, 0.4, 0.52, 0.7, 0.81, 0.9, 0.95, 0.99]), (v) =>
      Math.min(1, Math.max(0, v))
    );
    expectClose(res.statistic, 0.21000000000000008);
    expectClose(res.pvalue, 0.6963025889106401);
  });
});

describe("ks_2samp (SciPy references)", () => {
  it("ks_2samp exact unequal sizes, two-sided", () => {
    const res = stats.ks_2samp(
      T([-0.989, -0.368, 1.288, 0.194, 0.92, 0.577, -0.636]),
      T([0.942, 0.083, 0.078, 0.497, -1.126, 1.592, -0.271, 1.4, 0.536]),
      { alternative: "two-sided" }
    );
    expectClose(res.statistic, 0.31746031746031744);
    expectClose(res.pvalue, 0.7136363636363634);
  });
  it("ks_2samp exact unequal sizes, less", () => {
    const res = stats.ks_2samp(
      T([-0.989, -0.368, 1.288, 0.194, 0.92, 0.577, -0.636]),
      T([0.942, 0.083, 0.078, 0.497, -1.126, 1.592, -0.271, 1.4, 0.536]),
      { alternative: "less" }
    );
    expectClose(res.statistic, 0.1111111111111111);
    expectClose(res.pvalue, 0.8585664335664336);
  });
  it("ks_2samp exact unequal sizes, greater", () => {
    const res = stats.ks_2samp(
      T([-0.989, -0.368, 1.288, 0.194, 0.92, 0.577, -0.636]),
      T([0.942, 0.083, 0.078, 0.497, -1.126, 1.592, -0.271, 1.4, 0.536]),
      { alternative: "greater" }
    );
    expectClose(res.statistic, 0.31746031746031744);
    expectClose(res.pvalue, 0.3729895104895105);
  });
  it("ks_2samp exact equal sizes, two-sided", () => {
    const res = stats.ks_2samp(
      T([1.532, -0.66, -0.312, 0.338, -2.207, 0.828, 1.542, 1.127]),
      T([1.155, 0.254, 1.682, 1.474, 0.793, 0.405, 0.038, -0.83]),
      { alternative: "two-sided" }
    );
    expectClose(res.statistic, 0.25);
    expectClose(res.pvalue, 0.98010878010878);
  });
  it("ks_2samp exact equal sizes, less", () => {
    const res = stats.ks_2samp(
      T([1.532, -0.66, -0.312, 0.338, -2.207, 0.828, 1.542, 1.127]),
      T([1.155, 0.254, 1.682, 1.474, 0.793, 0.405, 0.038, -0.83]),
      { alternative: "less" }
    );
    expectClose(res.statistic, 0.125);
    expectClose(res.pvalue, 0.8888888888888888);
  });
  it("ks_2samp exact equal sizes, greater", () => {
    const res = stats.ks_2samp(
      T([1.532, -0.66, -0.312, 0.338, -2.207, 0.828, 1.542, 1.127]),
      T([1.155, 0.254, 1.682, 1.474, 0.793, 0.405, 0.038, -0.83]),
      { alternative: "greater" }
    );
    expectClose(res.statistic, 0.25);
    expectClose(res.pvalue, 0.6222222222222221);
  });
  it("ks_2samp with ties", () => {
    const res = stats.ks_2samp(
      T([1, 2, 2, 3, 3, 3, 4, 5, 5, 6]),
      T([2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9])
    );
    expectClose(res.statistic, 0.44545454545454544);
    expectClose(res.pvalue, 0.16434185010036403);
  });
  it("ks_2samp asymptotic method", () => {
    const res = stats.ks_2samp(
      T([1, 2, 2, 3, 3, 3, 4, 5, 5, 6]),
      T([2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9]),
      { method: "asymptotic" }
    );
    expectClose(res.statistic, 0.4454545454545455);
    expectClose(res.pvalue, 0.20305994250268555);
  });
  it("ks_2samp asymptotic one-sided (Hodges)", () => {
    const res = stats.ks_2samp(
      T([1, 2, 2, 3, 3, 3, 4, 5, 5, 6]),
      T([2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 9]),
      { method: "asymptotic", alternative: "greater" }
    );
    expectClose(res.statistic, 0.4454545454545455);
    expectClose(res.pvalue, 0.0806874913828625);
  });
  it("ks_2samp switches to the rounded Kolmogorov distribution above 10000 values", () => {
    const res = stats.ks_2samp(
      T(Array.from({ length: 12000 }, (_, i) => (((i * 7919) % 10007) / 10007) * 4 - 2)),
      T(Array.from({ length: 50 }, (_, i) => (((i * 104729) % 10007) / 10007) * 3 - 1.2))
    );
    expectClose(res.statistic, 0.20008333333333334);
    expectClose(res.pvalue, 0.03133089905853881);
  });
});

describe("normality tests (SciPy references)", () => {
  it("normaltest tiny p-value", () => {
    const res = stats.normaltest(
      T(Array.from({ length: 200 }, (_, i) => -Math.log((i + 0.5) / 200)))
    );
    expectClose(res.statistic, 86.64675468696021);
    expectClose(res.pvalue, 1.530721852874994e-19);
  });
  it("shapiro tiny p-value", () => {
    const res = stats.shapiro(T(Array.from({ length: 200 }, (_, i) => -Math.log((i + 0.5) / 200))));
    expectClose(res.statistic, 0.8237525605585383, 1e-7);
    expectClose(res.pvalue, 2.6924168019466744e-14, 1e-7);
  });
  it("shapiro n = 3", () => {
    const res = stats.shapiro(T([1, 2, 3.5]));
    expectClose(res.statistic, 0.9868421052631577, 1e-7);
    expectClose(res.pvalue, 0.780440814879016, 1e-7);
  });
  it("shapiro n = 5", () => {
    const res = stats.shapiro(T([1, 2, 3, 4, 5.5]));
    expectClose(res.statistic, 0.9890060614617024, 1e-7);
    expectClose(res.pvalue, 0.9760970940327423, 1e-7);
  });
  it("shapiro n = 12", () => {
    const res = stats.shapiro(T([0.2, 1.3, 1.1, 2.5, 3.9, 0.7, 1.9, 2.2, 6.1, 0.4, 1.2, 1.7]));
    expectClose(res.statistic, 0.850738105290165, 1e-7);
    expectClose(res.pvalue, 0.037491924387261726, 1e-7);
  });
  it("normaltest small sample", () => {
    const res = stats.normaltest(T([2.1, 1.9, 2.4, 2.2, 2, 2.6, 1.7, 2.3, 2.5, 3.9]));
    expectClose(res.statistic, 14.005557930495529);
    expectClose(res.pvalue, 0.0009093513950744462);
  });
});

describe("anderson (SciPy references)", () => {
  it("anderson small sample uses the sample standard deviation", () => {
    const res = stats.anderson(T([1, 2, 3, 4, 5.5]));
    expectClose(res.statistic, 0.14361959866123986);
    expect(res.critical_values).toEqual([0.452, 0.509, 0.606, 0.704, 0.835]);
    expect(res.significance_level).toEqual([0.15, 0.1, 0.05, 0.025, 0.01]);
  });
  it("anderson skewed sample", () => {
    const res = stats.anderson(T([0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.7, 1.1, 1.9, 3.8, 7.2, 14.5]));
    expectClose(res.statistic, 1.825092723364298);
    expect(res.critical_values).toEqual([0.52, 0.585, 0.698, 0.81, 0.96]);
    expect(res.significance_level).toEqual([0.15, 0.1, 0.05, 0.025, 0.01]);
  });
  it("anderson n = 3 critical values", () => {
    const res = stats.anderson(T([0.4, 0.7, 1.9]));
    expectClose(res.statistic, 0.31221114599275746);
    expect(res.critical_values).toEqual([0.374, 0.421, 0.501, 0.582, 0.69]);
    expect(res.significance_level).toEqual([0.15, 0.1, 0.05, 0.025, 0.01]);
  });
});

describe("mannwhitneyu (SciPy references)", () => {
  it("mannwhitneyu documented example (exact, statistic is U1)", () => {
    const res = stats.mannwhitneyu(T([19, 22, 16, 29, 24]), T([20, 11, 17, 12]));
    expectClose(res.statistic, 17.0);
    expectClose(res.pvalue, 0.1111111111111111);
  });
  it("mannwhitneyu statistic is U1 even when it is the larger one", () => {
    const res = stats.mannwhitneyu(T([20, 11, 17, 12]), T([19, 22, 16, 29, 24]));
    expectClose(res.statistic, 3.0);
    expectClose(res.pvalue, 0.1111111111111111);
  });
  it("mannwhitneyu asymptotic with continuity correction", () => {
    const res = stats.mannwhitneyu(T([19, 22, 16, 29, 24]), T([20, 11, 17, 12]), {
      method: "asymptotic",
    });
    expectClose(res.statistic, 17.0);
    expectClose(res.pvalue, 0.11134688653314039);
  });
  it("mannwhitneyu asymptotic without continuity correction", () => {
    const res = stats.mannwhitneyu(T([19, 22, 16, 29, 24]), T([20, 11, 17, 12]), {
      method: "asymptotic",
      correction: false,
    });
    expectClose(res.statistic, 17.0);
    expectClose(res.pvalue, 0.08641073297370001);
  });
  it("mannwhitneyu less", () => {
    const res = stats.mannwhitneyu(T([19, 22, 16, 29, 24]), T([20, 11, 17, 12]), {
      alternative: "less",
    });
    expectClose(res.statistic, 17.0);
    expectClose(res.pvalue, 0.9682539682539683);
  });
  it("mannwhitneyu greater", () => {
    const res = stats.mannwhitneyu(T([19, 22, 16, 29, 24]), T([20, 11, 17, 12]), {
      alternative: "greater",
    });
    expectClose(res.statistic, 17.0);
    expectClose(res.pvalue, 0.05555555555555555);
  });
  it("mannwhitneyu exact greater", () => {
    const res = stats.mannwhitneyu(T([20, 11, 17, 12]), T([19, 22, 16, 29, 24]), {
      alternative: "greater",
      method: "exact",
    });
    expectClose(res.statistic, 3.0);
    expectClose(res.pvalue, 0.9682539682539683);
  });
  it("mannwhitneyu large samples use the normal approximation", () => {
    const res = stats.mannwhitneyu(
      T([1.25, 1.761, -0.182, 2.998, 0.556, -0.068, 1.928, 0.64, 0.124]),
      T([-0.542, 2.102, -0.994, 1.126, -0.812, -0.623, -0.082, 0.465, -1.645, -0.544, 1.029, 1.025])
    );
    expectClose(res.statistic, 81.0);
    expectClose(res.pvalue, 0.05966338125040271);
  });
  it("mannwhitneyu large samples, greater", () => {
    const res = stats.mannwhitneyu(
      T([1.25, 1.761, -0.182, 2.998, 0.556, -0.068, 1.928, 0.64, 0.124]),
      T([
        -0.542, 2.102, -0.994, 1.126, -0.812, -0.623, -0.082, 0.465, -1.645, -0.544, 1.029, 1.025,
      ]),
      { alternative: "greater" }
    );
    expectClose(res.statistic, 81.0);
    expectClose(res.pvalue, 0.029831690625201353);
  });
  it("mannwhitneyu with ties uses the tie-corrected normal approximation", () => {
    const res = stats.mannwhitneyu(T([4, 2, 0, 3, 5, 3]), T([2, 1, 1, 4, 2, 3, 3]));
    expectClose(res.statistic, 26.5);
    expectClose(res.pvalue, 0.4650714430272246);
  });
  it("mannwhitneyu with ties, no continuity correction", () => {
    const res = stats.mannwhitneyu(T([4, 2, 0, 3, 5, 3]), T([2, 1, 1, 4, 2, 3, 3]), {
      correction: false,
    });
    expectClose(res.statistic, 26.5);
    expectClose(res.pvalue, 0.421643211839287);
  });
});

describe("wilcoxon (SciPy references)", () => {
  it("wilcoxon documented example (exact)", () => {
    const res = stats.wilcoxon(T([1.5, 2.5, 3.1, -0.5, 4.2, 0.7, 2.2]), undefined);
    expectClose(res.statistic, 1.0);
    expectClose(res.pvalue, 0.03125);
  });
  it("wilcoxon exact p-value is not the normal approximation", () => {
    const res = stats.wilcoxon(T([1.5, 2.5, 3.1, -0.5, 4.2, 0.7, 2.2]), undefined, {
      method: "asymptotic",
    });
    expectClose(res.statistic, 1.0);
    expectClose(res.pvalue, 0.02799181548566574);
  });
  it("wilcoxon continuity correction is off by default and optional", () => {
    const res = stats.wilcoxon(T([1.5, 2.5, 3.1, -0.5, 4.2, 0.7, 2.2]), undefined, {
      method: "asymptotic",
      correction: true,
    });
    expectClose(res.statistic, 1.0);
    expectClose(res.pvalue, 0.034610557515707366);
  });
  it("wilcoxon less (statistic is W+)", () => {
    const res = stats.wilcoxon(T([1.5, 2.5, 3.1, -0.5, 4.2, 0.7, 2.2]), undefined, {
      alternative: "less",
    });
    expectClose(res.statistic, 27.0);
    expectClose(res.pvalue, 0.9921875);
  });
  it("wilcoxon greater (statistic is W+)", () => {
    const res = stats.wilcoxon(T([1.5, 2.5, 3.1, -0.5, 4.2, 0.7, 2.2]), undefined, {
      alternative: "greater",
    });
    expectClose(res.statistic, 27.0);
    expectClose(res.pvalue, 0.015625);
  });
  it("wilcoxon paired samples", () => {
    const res = stats.wilcoxon(
      T([-0.163, -1.166, 0.063, 1.164, 0.278, 0.555, 2.421, 0.84, -0.038]),
      T([-0.298, -0.001, -0.144, 0.541, 0.171, 0.426, 0.063, 0.374, -0.3])
    );
    expectClose(res.statistic, 8.0);
    expectClose(res.pvalue, 0.09765625);
  });
  it("wilcoxon with ties and zeros uses the exact sign-flip distribution (n <= 13)", () => {
    const res = stats.wilcoxon(T([1, -1, 2, 4, 4, 3, -1, 3, -1, -3, 3, -1]), undefined);
    expectClose(res.statistic, 20.5);
    expectClose(res.pvalue, 0.17529296875);
  });
  it("wilcoxon exact method with ties rounds the statistic conservatively", () => {
    const res = stats.wilcoxon(T([1, -1, 2, 4, 4, 3, -1, 3, -1, -3, 3, -1]), undefined, {
      method: "exact",
    });
    expectClose(res.statistic, 20.5);
    expectClose(res.pvalue, 0.17626953125);
  });
  it("wilcoxon zero_method pratt", () => {
    const res = stats.wilcoxon(T([4, 1, 2, 1, 0, 4, -3, -2, 1, 4, 3, 2, -3]), undefined, {
      zeroMethod: "pratt",
    });
    expectClose(res.statistic, 24.0);
    expectClose(res.pvalue, 0.17578125);
  });
  it("wilcoxon zero_method zsplit", () => {
    const res = stats.wilcoxon(T([4, 1, 2, 1, 0, 4, -3, -2, 1, 4, 3, 2, -3]), undefined, {
      zeroMethod: "zsplit",
    });
    expectClose(res.statistic, 24.5);
    expectClose(res.pvalue, 0.17578125);
  });
  it("wilcoxon zero_method pratt, asymptotic", () => {
    const res = stats.wilcoxon(T([4, 1, 2, 1, 0, 4, -3, -2, 1, 4, 3, 2, -3]), undefined, {
      method: "asymptotic",
      zeroMethod: "pratt",
    });
    expectClose(res.statistic, 24.0);
    expectClose(res.pvalue, 0.1400165031971689);
  });
  it("wilcoxon zero_method zsplit, greater", () => {
    const res = stats.wilcoxon(T([4, 1, 2, 1, 0, 4, -3, -2, 1, 4, 3, 2, -3]), undefined, {
      alternative: "greater",
      zeroMethod: "zsplit",
    });
    expectClose(res.statistic, 66.5);
    expectClose(res.pvalue, 0.087890625);
  });
  it("wilcoxon more than 50 differences use the normal approximation", () => {
    const res = stats.wilcoxon(
      T([
        -0.502, -1.024, 0.052, 0.72, 1.436, 0.41, -0.253, -0.485, 1.049, 1.935, 0.573, -0.933,
        -0.658, 1.9, 0.503, -1.432, 0.216, -0.863, -0.329, -0.188, -0.413, 0.853, 0.237, -0.289,
        0.71, 1.13, -1.343, 0.043, -0.681, 0.127, -0.989, 0.321, 0.262, -0.004, -0.748, -0.096,
        -0.791, -1.055, 0.525, -0.809, 1.47, 1.017, -1.698, 0.572, -0.802, 0.333, 0.344, -1.688,
        0.067, 0.044, 1.262, -0.881, 1.038, -0.799, -0.031, -0.54, 1.749, 0.868, 2.732, 0.942,
      ]),
      undefined
    );
    expectClose(res.statistic, 845.0);
    expectClose(res.pvalue, 0.606334867988235);
  });
  it("wilcoxon with ties and n > 13 falls back to the normal approximation", () => {
    const res = stats.wilcoxon(T([3, 1, 2, 1, 0, 4, -3, -2, 1, 4, 3, 2, -3, 1, 5, 2]), undefined);
    expectClose(res.statistic, 27.5);
    expectClose(res.pvalue, 0.06323692674670552);
  });
});

describe("variance tests (SciPy references)", () => {
  it("levene center median", () => {
    const res = stats.levene(
      "median",
      T([8.88, 9.12, 9.04, 8.98, 9, 9.08, 9.01, 8.85, 9.06, 8.99]),
      T([8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]),
      T([8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98])
    );
    expectClose(res.statistic, 7.584952754501659);
    expectClose(res.pvalue, 0.002431505967249677);
  });
  it("levene center mean", () => {
    const res = stats.levene(
      "mean",
      T([8.88, 9.12, 9.04, 8.98, 9, 9.08, 9.01, 8.85, 9.06, 8.99]),
      T([8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]),
      T([8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98])
    );
    expectClose(res.statistic, 7.905194483442054);
    expectClose(res.pvalue, 0.001983795817472729);
  });
  it("levene center trimmed", () => {
    const res = stats.levene(
      "trimmed",
      T([8.88, 9.12, 9.04, 8.98, 9, 9.08, 9.01, 8.85, 9.06, 8.99]),
      T([8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]),
      T([8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98])
    );
    expectClose(res.statistic, 7.905194483442053);
    expectClose(res.pvalue, 0.0019837958174727323);
  });
  it("levene without a center argument uses the median (SciPy signature)", () => {
    const res = stats.levene(
      T([8.88, 9.12, 9.04, 8.98, 9, 9.08, 9.01, 8.85, 9.06, 8.99]),
      T([8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]),
      T([8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98])
    );
    expectClose(res.statistic, 7.584952754501659);
    expectClose(res.pvalue, 0.002431505967249677);
  });
  it("levene options object with proportiontocut", () => {
    const res = stats.levene(
      { center: "trimmed", proportiontocut: 0.2 },
      T([8.88, 9.12, 9.04, 8.98, 9, 9.08, 9.01, 8.85, 9.06, 8.99]),
      T([8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]),
      T([8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98])
    );
    expectClose(res.statistic, 7.734626490501758);
    expectClose(res.pvalue, 0.0022100567186374032);
  });
  it("bartlett", () => {
    const res = stats.bartlett(
      T([8.88, 9.12, 9.04, 8.98, 9, 9.08, 9.01, 8.85, 9.06, 8.99]),
      T([8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]),
      T([8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98])
    );
    expectClose(res.statistic, 22.789434813726768);
    expectClose(res.pvalue, 1.1254782518834626e-5);
  });
  it("fligner center median", () => {
    const res = stats.fligner(
      [
        T([8.88, 9.12, 9.04, 8.98, 9, 9.08, 9.01, 8.85, 9.06, 8.99]),
        T([8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]),
        T([8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98]),
      ],
      { center: "median" }
    );
    expectClose(res.statistic, 10.803687663522247);
    expectClose(res.pvalue, 0.004508260800047729);
  });
  it("fligner center mean", () => {
    const res = stats.fligner(
      [
        T([8.88, 9.12, 9.04, 8.98, 9, 9.08, 9.01, 8.85, 9.06, 8.99]),
        T([8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]),
        T([8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98]),
      ],
      { center: "mean" }
    );
    expectClose(res.statistic, 10.138447776942241);
    expectClose(res.pvalue, 0.006287297892096322);
  });
  it("fligner center trimmed", () => {
    const res = stats.fligner(
      [
        T([8.88, 9.12, 9.04, 8.98, 9, 9.08, 9.01, 8.85, 9.06, 8.99]),
        T([8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]),
        T([8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98]),
      ],
      { center: "trimmed" }
    );
    expectClose(res.statistic, 10.14113256518551);
    expectClose(res.pvalue, 0.006278863522755499);
  });
  it("fligner accepts separate samples", () => {
    const res = stats.fligner(
      T([8.88, 9.12, 9.04, 8.98, 9, 9.08, 9.01, 8.85, 9.06, 8.99]),
      T([8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]),
      T([8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98])
    );
    expectClose(res.statistic, 10.803687663522247);
    expectClose(res.pvalue, 0.004508260800047729);
  });
  it("fligner proportiontocut", () => {
    const res = stats.fligner(
      [
        T([8.88, 9.12, 9.04, 8.98, 9, 9.08, 9.01, 8.85, 9.06, 8.99]),
        T([8.88, 8.95, 9.29, 9.44, 9.15, 9.58, 8.36, 9.18, 8.67, 9.05]),
        T([8.95, 9.12, 8.95, 8.85, 9.03, 8.84, 9.07, 8.98, 8.86, 8.98]),
      ],
      { center: "trimmed", proportiontocut: 0.2 }
    );
    expectClose(res.statistic, 10.905968124915766);
    expectClose(res.pvalue, 0.0042835033587034855);
  });
});

describe("median_test (SciPy references)", () => {
  it("median_test applies Yates' correction for two samples", () => {
    const res = stats.median_test(T([1, 2, 3, 4, 5]), T([3, 5, 7, 8, 9]));
    expectClose(res.statistic, 1.6);
    expectClose(res.pvalue, 0.2059032107320647);
  });
  it("median_test without correction", () => {
    const res = stats.median_test(T([1, 2, 3, 4, 5]), T([3, 5, 7, 8, 9]), { correction: false });
    expectClose(res.statistic, 3.6);
    expectClose(res.pvalue, 0.05777957112359719);
  });
  it("median_test with three samples", () => {
    const res = stats.median_test(T([1, 2, 3, 4, 5]), T([3, 5, 7, 8, 9]), T([2, 6, 6, 9, 10, 11]));
    expectClose(res.statistic, 7.866666666666666);
    expectClose(res.pvalue, 0.01957830265491302);
  });
  it("median_test ties above", () => {
    const res = stats.median_test(T([1, 2, 3, 4, 5]), T([3, 5, 7, 8, 9]), T([2, 6, 6, 9, 10, 11]), {
      ties: "above",
    });
    expectClose(res.statistic, 7.866666666666666);
    expectClose(res.pvalue, 0.01957830265491302);
  });
  it("median_test ties ignore", () => {
    const res = stats.median_test(T([1, 2, 3, 4, 5]), T([3, 5, 7, 8, 9]), T([2, 6, 6, 9, 10, 11]), {
      ties: "ignore",
    });
    expectClose(res.statistic, 7.866666666666666);
    expectClose(res.pvalue, 0.01957830265491302);
  });
});

describe("chi2_contingency (SciPy references)", () => {
  it("chi2_contingency [[12, 5], [3, 9]] correction true", () => {
    const res = stats.chi2_contingency(
      [
        [12, 5],
        [3, 9],
      ],
      true
    );
    expectClose(res.statistic, 4.171457749766574);
    expectClose(res.pvalue, 0.041110414194307);
    expect(res.dof).toBe(1);
    expectClose(res.expected[0]?.[0] ?? Number.NaN, 8.793103448275861);
  });
  it("chi2_contingency [[12, 5], [3, 9]] correction false", () => {
    const res = stats.chi2_contingency(
      [
        [12, 5],
        [3, 9],
      ],
      false
    );
    expectClose(res.statistic, 5.85483193277311);
    expectClose(res.pvalue, 0.015534341414683487);
    expect(res.dof).toBe(1);
    expectClose(res.expected[0]?.[0] ?? Number.NaN, 8.793103448275861);
  });
  it("chi2_contingency [[10, 20, 30], [6, 9, 17]] correction true", () => {
    const res = stats.chi2_contingency(
      [
        [10, 20, 30],
        [6, 9, 17],
      ],
      true
    );
    expectClose(res.statistic, 0.27157465150403504);
    expectClose(res.pvalue, 0.873028283380073);
    expect(res.dof).toBe(2);
    expectClose(res.expected[0]?.[0] ?? Number.NaN, 10.434782608695652);
  });
  it("chi2_contingency [[10, 20, 30], [6, 9, 17]] correction false", () => {
    const res = stats.chi2_contingency(
      [
        [10, 20, 30],
        [6, 9, 17],
      ],
      false
    );
    expectClose(res.statistic, 0.27157465150403504);
    expectClose(res.pvalue, 0.873028283380073);
    expect(res.dof).toBe(2);
    expectClose(res.expected[0]?.[0] ?? Number.NaN, 10.434782608695652);
  });
});

describe("fisher_exact (SciPy references)", () => {
  it("fisher_exact [[1, 9], [11, 3]] two-sided", () => {
    const res = stats.fisher_exact(
      [
        [1, 9],
        [11, 3],
      ] as [[number, number], [number, number]],
      "two-sided"
    );
    expectClose(res.oddsRatio, 0.030303030303030304);
    expectClose(res.pvalue, 0.0027594561852200836, 1e-9);
  });
  it("fisher_exact [[1, 9], [11, 3]] less", () => {
    const res = stats.fisher_exact(
      [
        [1, 9],
        [11, 3],
      ] as [[number, number], [number, number]],
      "less"
    );
    expectClose(res.oddsRatio, 0.030303030303030304);
    expectClose(res.pvalue, 0.0013797280926100418, 1e-9);
  });
  it("fisher_exact [[1, 9], [11, 3]] greater", () => {
    const res = stats.fisher_exact(
      [
        [1, 9],
        [11, 3],
      ] as [[number, number], [number, number]],
      "greater"
    );
    expectClose(res.oddsRatio, 0.030303030303030304);
    expectClose(res.pvalue, 0.9999663480953022, 1e-9);
  });
  it("fisher_exact [[8, 2], [1, 5]] two-sided", () => {
    const res = stats.fisher_exact(
      [
        [8, 2],
        [1, 5],
      ] as [[number, number], [number, number]],
      "two-sided"
    );
    expectClose(res.oddsRatio, 20.0);
    expectClose(res.pvalue, 0.034965034965034975, 1e-9);
  });
  it("fisher_exact [[8, 2], [1, 5]] less", () => {
    const res = stats.fisher_exact(
      [
        [8, 2],
        [1, 5],
      ] as [[number, number], [number, number]],
      "less"
    );
    expectClose(res.oddsRatio, 20.0);
    expectClose(res.pvalue, 0.9991258741258742, 1e-9);
  });
  it("fisher_exact [[8, 2], [1, 5]] greater", () => {
    const res = stats.fisher_exact(
      [
        [8, 2],
        [1, 5],
      ] as [[number, number], [number, number]],
      "greater"
    );
    expectClose(res.oddsRatio, 20.0);
    expectClose(res.pvalue, 0.024475524475524483, 1e-9);
  });
  it("fisher_exact [[10, 0], [0, 10]] two-sided", () => {
    const res = stats.fisher_exact(
      [
        [10, 0],
        [0, 10],
      ] as [[number, number], [number, number]],
      "two-sided"
    );
    expectClose(res.oddsRatio, Number.POSITIVE_INFINITY);
    expectClose(res.pvalue, 1.082508822446903e-5, 1e-9);
  });
  it("fisher_exact [[10, 0], [0, 10]] less", () => {
    const res = stats.fisher_exact(
      [
        [10, 0],
        [0, 10],
      ] as [[number, number], [number, number]],
      "less"
    );
    expectClose(res.oddsRatio, Number.POSITIVE_INFINITY);
    expectClose(res.pvalue, 1.0, 1e-9);
  });
  it("fisher_exact [[10, 0], [0, 10]] greater", () => {
    const res = stats.fisher_exact(
      [
        [10, 0],
        [0, 10],
      ] as [[number, number], [number, number]],
      "greater"
    );
    expectClose(res.oddsRatio, Number.POSITIVE_INFINITY);
    expectClose(res.pvalue, 5.412544112234515e-6, 1e-9);
  });
  it("fisher_exact [[0, 5], [5, 0]] two-sided", () => {
    const res = stats.fisher_exact(
      [
        [0, 5],
        [5, 0],
      ] as [[number, number], [number, number]],
      "two-sided"
    );
    expectClose(res.oddsRatio, 0.0);
    expectClose(res.pvalue, 0.007936507936507938, 1e-9);
  });
  it("fisher_exact [[0, 5], [5, 0]] less", () => {
    const res = stats.fisher_exact(
      [
        [0, 5],
        [5, 0],
      ] as [[number, number], [number, number]],
      "less"
    );
    expectClose(res.oddsRatio, 0.0);
    expectClose(res.pvalue, 0.003968253968253969, 1e-9);
  });
  it("fisher_exact [[0, 5], [5, 0]] greater", () => {
    const res = stats.fisher_exact(
      [
        [0, 5],
        [5, 0],
      ] as [[number, number], [number, number]],
      "greater"
    );
    expectClose(res.oddsRatio, 0.0);
    expectClose(res.pvalue, 1.0, 1e-9);
  });
  it("fisher_exact [[3, 0], [0, 0]] two-sided", () => {
    const res = stats.fisher_exact(
      [
        [3, 0],
        [0, 0],
      ] as [[number, number], [number, number]],
      "two-sided"
    );
    expectClose(res.oddsRatio, Number.NaN);
    expectClose(res.pvalue, 1.0, 1e-9);
  });
  it("fisher_exact [[3, 0], [0, 0]] less", () => {
    const res = stats.fisher_exact(
      [
        [3, 0],
        [0, 0],
      ] as [[number, number], [number, number]],
      "less"
    );
    expectClose(res.oddsRatio, Number.NaN);
    expectClose(res.pvalue, 1.0, 1e-9);
  });
  it("fisher_exact [[3, 0], [0, 0]] greater", () => {
    const res = stats.fisher_exact(
      [
        [3, 0],
        [0, 0],
      ] as [[number, number], [number, number]],
      "greater"
    );
    expectClose(res.oddsRatio, Number.NaN);
    expectClose(res.pvalue, 1.0, 1e-9);
  });
  it("fisher_exact [[100, 200], [150, 260]] two-sided", () => {
    const res = stats.fisher_exact(
      [
        [100, 200],
        [150, 260],
      ] as [[number, number], [number, number]],
      "two-sided"
    );
    expectClose(res.oddsRatio, 0.8666666666666667);
    expectClose(res.pvalue, 0.3825406765261436, 1e-9);
  });
  it("fisher_exact [[100, 200], [150, 260]] less", () => {
    const res = stats.fisher_exact(
      [
        [100, 200],
        [150, 260],
      ] as [[number, number], [number, number]],
      "less"
    );
    expectClose(res.oddsRatio, 0.8666666666666667);
    expectClose(res.pvalue, 0.20720039186301248, 1e-9);
  });
  it("fisher_exact [[100, 200], [150, 260]] greater", () => {
    const res = stats.fisher_exact(
      [
        [100, 200],
        [150, 260],
      ] as [[number, number], [number, number]],
      "greater"
    );
    expectClose(res.oddsRatio, 0.8666666666666667);
    expectClose(res.pvalue, 0.8353760596409885, 1e-9);
  });
});

describe("kruskal, f_oneway and friedmanchisquare (SciPy references)", () => {
  it("kruskal", () => {
    const res = stats.kruskal(
      T([2.9, 3, 2.5, 3.6, 2.8]),
      T([3.8, 2.7, 4, 2.4, 3.3]),
      T([2.8, 3.4, 3.7, 2.2, 2, 3.5])
    );
    expectClose(res.statistic, 0.5578055964653953);
    expectClose(res.pvalue, 0.7566134438047544);
  });
  it("f_oneway", () => {
    const res = stats.f_oneway(
      T([2.9, 3, 2.5, 3.6, 2.8]),
      T([3.8, 2.7, 4, 2.4, 3.3]),
      T([2.8, 3.4, 3.7, 2.2, 2, 3.5])
    );
    expectClose(res.statistic, 0.3827654982997645);
    expectClose(res.pvalue, 0.6894098649256987);
  });
  it("friedmanchisquare", () => {
    const res = stats.friedmanchisquare(
      T([7, 9.9, 8.5, 5.1, 10.3]),
      T([5.3, 5.7, 4.7, 3.5, 7.7]),
      T([4.9, 7.6, 5.5, 2.8, 8.4])
    );
    expectClose(res.statistic, 7.6000000000000085);
    expectClose(res.pvalue, 0.022370771856165508);
  });
});

describe("kernel density (SciPy references)", () => {
  it("bandwidth, density and log-density match SciPy (scott)", () => {
    const kde = stats.gaussian_kde([1.2, 2.3, 2.9, 3.1, 4.8, 5, 5.5, 7.1], { bwMethod: "scott" });
    expectClose(kde.bandwidth, 1.2775776213018786, 1e-12);
    const pts = [-2, 1, 3, 4.9, 7, 12];
    const pdf = [
      0.001868664929029625, 0.08568126181984319, 0.1577969180138252, 0.15308660752787262,
      0.07947310321725151, 2.506539671264057e-5,
    ];
    const logpdf = [
      -6.28253104478877, -2.4571211258287335, -1.8464464017254472, -1.8767514554688383,
      -2.532336638877734, -10.594022280019558,
    ];
    const density = kde.evaluate(pts);
    const logDensity = kde.logDensity(pts);
    for (let i = 0; i < pts.length; i++) {
      expectClose(density[i] ?? Number.NaN, pdf[i] ?? Number.NaN, 1e-10);
      expectClose(logDensity[i] ?? Number.NaN, logpdf[i] ?? Number.NaN, 1e-10);
    }
    expectClose(kde.integrateBox1d(2, 5), 0.4649018388360582, 1e-10);
    expectClose(kde.integrateBox1d(6.5, Number.POSITIVE_INFINITY), 0.13953112392617528, 1e-10);
    expectClose(kde.integrateBox1d(Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY), 1, 1e-12);
  });
  it("bandwidth, density and log-density match SciPy (scott, weighted)", () => {
    const kde = stats.gaussian_kde([1.2, 2.3, 2.9, 3.1, 4.8, 5, 5.5, 7.1], {
      bwMethod: "scott",
      weights: [1, 2, 1, 1, 3, 1, 2, 1],
    });
    expectClose(kde.bandwidth, 1.228454892389084, 1e-12);
    const pts = [-2, 1, 3, 4.9, 7, 12];
    const pdf = [
      0.0010424141464599667, 0.07296582826239961, 0.15108101901026993, 0.1838713075407612,
      0.07649447301309602, 9.546044951283188e-6,
    ];
    const logpdf = [
      -6.866215961198175, -2.6177640534240854, -1.8899390363216682, -1.6935191816176485,
      -2.5705367889544966, -11.559383630478688,
    ];
    const density = kde.evaluate(pts);
    const logDensity = kde.logDensity(pts);
    for (let i = 0; i < pts.length; i++) {
      expectClose(density[i] ?? Number.NaN, pdf[i] ?? Number.NaN, 1e-10);
      expectClose(logDensity[i] ?? Number.NaN, logpdf[i] ?? Number.NaN, 1e-10);
    }
    expectClose(kde.integrateBox1d(2, 5), 0.48130039051644313, 1e-10);
    expectClose(kde.integrateBox1d(6.5, Number.POSITIVE_INFINITY), 0.12239931874182208, 1e-10);
    expectClose(kde.integrateBox1d(Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY), 1, 1e-12);
  });
  it("bandwidth, density and log-density match SciPy (silverman)", () => {
    const kde = stats.gaussian_kde([1.2, 2.3, 2.9, 3.1, 4.8, 5, 5.5, 7.1], {
      bwMethod: "silverman",
    });
    expectClose(kde.bandwidth, 1.3532406752733808, 1e-12);
    const pts = [-2, 1, 3, 4.9, 7, 12];
    const pdf = [
      0.0025696272965655526, 0.0858160022837938, 0.15559104097264775, 0.15100445368579044,
      0.07992492432918752, 5.2850167918775446e-5,
    ];
    const logpdf = [
      -5.963994411383152, -2.4555497830743285, -1.8605242461920901, -1.8904459479942863,
      -2.5266675308094353, -9.848049668382016,
    ];
    const density = kde.evaluate(pts);
    const logDensity = kde.logDensity(pts);
    for (let i = 0; i < pts.length; i++) {
      expectClose(density[i] ?? Number.NaN, pdf[i] ?? Number.NaN, 1e-10);
      expectClose(logDensity[i] ?? Number.NaN, logpdf[i] ?? Number.NaN, 1e-10);
    }
    expectClose(kde.integrateBox1d(2, 5), 0.4594042607554355, 1e-10);
    expectClose(kde.integrateBox1d(6.5, Number.POSITIVE_INFINITY), 0.14380803204451564, 1e-10);
    expectClose(kde.integrateBox1d(Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY), 1, 1e-12);
  });
  it("bandwidth, density and log-density match SciPy (silverman, weighted)", () => {
    const kde = stats.gaussian_kde([1.2, 2.3, 2.9, 3.1, 4.8, 5, 5.5, 7.1], {
      bwMethod: "silverman",
      weights: [1, 2, 1, 1, 3, 1, 2, 1],
    });
    expectClose(kde.bandwidth, 1.301208709671571, 1e-12);
    const pts = [-2, 1, 3, 4.9, 7, 12];
    const pdf = [
      0.0014923973110860375, 0.07345068512276483, 0.15050845714698155, 0.17900517029402677,
      0.07850523423072195, 2.151060752498414e-5,
    ];
    const logpdf = [
      -6.507371518357009, -2.611141048662189, -1.8937360027043608, -1.720340589234354,
      -2.544589978318291, -10.746964371244148,
    ];
    const density = kde.evaluate(pts);
    const logDensity = kde.logDensity(pts);
    for (let i = 0; i < pts.length; i++) {
      expectClose(density[i] ?? Number.NaN, pdf[i] ?? Number.NaN, 1e-10);
      expectClose(logDensity[i] ?? Number.NaN, logpdf[i] ?? Number.NaN, 1e-10);
    }
    expectClose(kde.integrateBox1d(2, 5), 0.47628682755775664, 1e-10);
    expectClose(kde.integrateBox1d(6.5, Number.POSITIVE_INFINITY), 0.12833317328496288, 1e-10);
    expectClose(kde.integrateBox1d(Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY), 1, 1e-12);
  });
});

describe("multiple comparisons (SciPy references)", () => {
  it("benjaminiHochberg matches scipy.stats.false_discovery_control", () => {
    const res = stats.benjaminiHochberg([0.01, 0.04, 0.03, 0.005, 0.2, 0.0001, 0.5, 0.03], 0.05);
    const expected = [
      0.026666666666666665, 0.05333333333333333, 0.048, 0.02, 0.22857142857142856, 0.0008, 0.5,
      0.048,
    ];
    for (let i = 0; i < expected.length; i++)
      expectClose(res.corrected[i] ?? Number.NaN, expected[i] ?? Number.NaN, 1e-12);
    expect(res.rejected).toEqual([true, false, true, true, false, true, false, true]);
  });
  it("benjaminiYekutieli matches scipy.stats.false_discovery_control", () => {
    const res = stats.benjaminiYekutieli([0.01, 0.04, 0.03, 0.005, 0.2, 0.0001, 0.5, 0.03], 0.05);
    const expected = [
      0.07247619047619046, 0.14495238095238092, 0.13045714285714285, 0.054357142857142854,
      0.6212244897959183, 0.002174285714285714, 1, 0.13045714285714285,
    ];
    for (let i = 0; i < expected.length; i++)
      expectClose(res.corrected[i] ?? Number.NaN, expected[i] ?? Number.NaN, 1e-12);
    expect(res.rejected).toEqual([false, false, false, false, false, true, false, false]);
  });
  it("holm adjusted p-values (ties keep a monotone sequence)", () => {
    const res = stats.holm([0.01, 0.04, 0.03, 0.005, 0.2, 0.0001, 0.5, 0.03], 0.05);
    const expected = [0.06, 0.15, 0.15, 0.035, 0.4, 0.0008, 0.5, 0.15];
    for (let i = 0; i < expected.length; i++)
      expectClose(res.corrected[i] ?? Number.NaN, expected[i] ?? Number.NaN, 1e-12);
  });
  it("hochberg adjusted p-values", () => {
    const res = stats.hochberg([0.01, 0.04, 0.03, 0.005, 0.2, 0.0001, 0.5, 0.03], 0.05);
    const expected = [0.06, 0.12, 0.12, 0.035, 0.4, 0.0008, 0.5, 0.12];
    for (let i = 0; i < expected.length; i++)
      expectClose(res.corrected[i] ?? Number.NaN, expected[i] ?? Number.NaN, 1e-12);
    expect(res.rejected).toEqual([false, false, false, true, false, true, false, false]);
  });
  it("sidak keeps full precision for tiny p-values", () => {
    const res = stats.sidak([1e-12, 1e-10, 1e-5, 0.3, 1], 0.05);
    const expected = [4.99999999999e-12, 4.999999999e-10, 4.999900000999996e-5, 0.83193, 1];
    for (let i = 0; i < expected.length; i++)
      expectClose(res.corrected[i] ?? Number.NaN, expected[i] ?? Number.NaN, 1e-12);
  });
});

describe("power analysis (SciPy references)", () => {
  it("power matches the non-central t distribution of SciPy", () => {
    expectClose(
      stats.tTestPower({ effectSize: 0.5, nObs: 64, alpha: 0.05, alternative: 2 }).power,
      0.8014595579222543,
      1e-9
    );
    expectClose(
      stats.tTestPower({ effectSize: 0.5, nObs: 50, alpha: 0.05, alternative: 2 }).power,
      0.696893405533384,
      1e-9
    );
    expectClose(
      stats.tTestPower({ effectSize: 0.2, nObs: 20, alpha: 0.01, alternative: 2 }).power,
      0.025131563164977437,
      1e-9
    );
    expectClose(
      stats.tTestPower({ effectSize: 0.8, nObs: 10, alpha: 0.05, alternative: 1 }).power,
      0.5303874685238488,
      1e-9
    );
    expectClose(
      stats.tTestPower({ effectSize: 1.2, nObs: 5, alpha: 0.001, alternative: 2, ratio: 2.5 })
        .power,
      0.08168079066942616,
      1e-9
    );
    expectClose(
      stats.tTestPower({ effectSize: 0.3, nObs: 500, alpha: 0.05, alternative: 1, ratio: 0.5 })
        .power,
      0.9869467563191168,
      1e-9
    );
    expectClose(
      stats.tTestPower({ effectSize: 0.05, nObs: 3, alpha: 0.2, alternative: 2 }).power,
      0.20067857750195203,
      1e-9
    );
  });
  it("solves for the smallest integer sample size", () => {
    expect(stats.tTestPower({ effectSize: 0.5, alpha: 0.05, power: 0.8 }).nObs).toBe(64);
    expect(stats.tTestPower({ effectSize: 0.2, alpha: 0.05, power: 0.9 }).nObs).toBe(527);
    expect(
      stats.tTestPower({ effectSize: 0.8, alpha: 0.01, power: 0.95, alternative: 1 }).nObs
    ).toBe(51);
    expect(stats.tTestPower({ effectSize: 0.5, alpha: 0.05, power: 0.8, ratio: 2 }).nObs).toBe(48);
  });
  it("solves for the effect size", () => {
    expectClose(
      stats.tTestPower({ nObs: 64, alpha: 0.05, power: 0.8 }).effectSize,
      0.49906917796582073,
      1e-9
    );
    expectClose(
      stats.tTestPower({ nObs: 12, alpha: 0.01, power: 0.9, alternative: 1 }).effectSize,
      1.5717108920622027,
      1e-9
    );
  });
  it("solves for alpha", () => {
    expectClose(
      stats.tTestPower({ effectSize: 0.5, nObs: 64, power: 0.8 }).alpha,
      0.04940542050566953,
      1e-8
    );
    expectClose(
      stats.tTestPower({ effectSize: 0.3, nObs: 40, power: 0.6, alternative: 1 }).alpha,
      0.13935271033052796,
      1e-8
    );
  });
});

describe("public API surface", () => {
  it("exposes camelCase aliases that are the same functions as the snake_case names", () => {
    expect(stats.ttest1samp).toBe(stats.ttest_1samp);
    expect(stats.ttestInd).toBe(stats.ttest_ind);
    expect(stats.ttestRel).toBe(stats.ttest_rel);
    expect(stats.fOneway).toBe(stats.f_oneway);
    expect(stats.fTwoway).toBe(stats.f_twoway);
    expect(stats.chi2Contingency).toBe(stats.chi2_contingency);
    expect(stats.fisherExact).toBe(stats.fisher_exact);
    expect(stats.ks2samp).toBe(stats.ks_2samp);
    expect(stats.medianTest).toBe(stats.median_test);
    expect(stats.runsTest).toBe(stats.runs_test);
    expect(stats.gaussianKde).toBe(stats.gaussian_kde);
  });

  it("exports the added multiple comparison procedures", () => {
    expect(typeof stats.benjaminiYekutieli).toBe("function");
    expect(typeof stats.hochberg).toBe("function");
  });

  it("anderson reports camelCase fields next to the deprecated snake_case ones", () => {
    const res = stats.anderson(T([1, 2, 3, 4, 5.5]));
    expect(res.criticalValues).toEqual(res.critical_values);
    expect(res.significanceLevel).toEqual([0.15, 0.1, 0.05, 0.025, 0.01]);
    expect(res.significanceLevel).toEqual(res.significance_level);
  });

  it("ttest_ind accepts an options object as well as positional arguments", () => {
    const a = T([2.1, 1.9, 2.4, 2.2, 2.0, 2.6]);
    const b = T([2.8, 3.1, 2.6, 3.0, 2.9]);
    const positional = stats.ttest_ind(a, b, false, "less");
    const named = stats.ttest_ind(a, b, { equalVar: false, alternative: "less" });
    expect(named).toEqual(positional);
    expect(stats.ttest_ind(a, b, {})).toEqual(stats.ttest_ind(a, b));
  });

  it("accepts the camelCase bwMethod option and the deprecated bw_method option", () => {
    const a = stats.gaussianKde([1, 2, 3, 4], { bwMethod: 0.7 });
    const b = stats.gaussian_kde([1, 2, 3, 4], { bw_method: 0.7 });
    expect(a.bandwidth).toBe(0.7);
    expect(b.bandwidth).toBe(0.7);
  });
});

describe("alternative and option validation", () => {
  const x = T([1, 2, 3, 4, 5]);
  const y = T([2, 3, 5, 7, 11]);

  it("rejects an unknown alternative with InvalidParameterError", () => {
    const bad = "sideways" as "two-sided";
    expect(() => stats.ttest_1samp(x, 1, bad)).toThrow(InvalidParameterError);
    expect(() => stats.ttest_ind(x, y, true, bad)).toThrow(InvalidParameterError);
    expect(() => stats.ttest_rel(x, y, bad)).toThrow(InvalidParameterError);
    expect(() => stats.mannwhitneyu(x, y, { alternative: bad })).toThrow(InvalidParameterError);
    expect(() => stats.wilcoxon(x, y, { alternative: bad })).toThrow(InvalidParameterError);
    expect(() => stats.kstest(x, "norm", { alternative: bad })).toThrow(InvalidParameterError);
    expect(() => stats.ks_2samp(x, y, { alternative: bad })).toThrow(InvalidParameterError);
    expect(() =>
      stats.fisher_exact(
        [
          [1, 2],
          [3, 4],
        ],
        bad
      )
    ).toThrow(InvalidParameterError);
  });

  it("rejects an unknown method", () => {
    const bad = "magic" as "auto";
    expect(() => stats.mannwhitneyu(x, y, { method: bad })).toThrow(InvalidParameterError);
    expect(() => stats.wilcoxon(x, y, { method: bad })).toThrow(InvalidParameterError);
    expect(() => stats.kstest(x, "norm", { method: bad })).toThrow(InvalidParameterError);
    expect(() => stats.ks_2samp(x, y, { method: bad })).toThrow(InvalidParameterError);
  });

  it("rejects an unknown wilcoxon zeroMethod", () => {
    const bad = "drop" as "wilcox";
    expect(() => stats.wilcoxon(x, y, { zeroMethod: bad })).toThrow(InvalidParameterError);
  });

  it("rejects invalid levene and fligner centers", () => {
    const bad = "mode" as "mean";
    expect(() => stats.levene(bad, x, y)).toThrow(InvalidParameterError);
    expect(() => stats.fligner([x, y], { center: bad })).toThrow(InvalidParameterError);
    expect(() => stats.levene({ center: "trimmed", proportiontocut: 0.5 }, x, y)).toThrow(
      InvalidParameterError
    );
    expect(() => stats.levene({ proportiontocut: Number.NaN }, x, y)).toThrow(
      InvalidParameterError
    );
  });

  it("rejects non-tensor samples for levene and median_test", () => {
    const notTensor = [1, 2, 3] as unknown as ReturnType<typeof T>;
    expect(() => stats.levene("mean", x, notTensor)).toThrow(InvalidParameterError);
    expect(() => stats.median_test(x, notTensor)).toThrow(InvalidParameterError);
  });

  it("requires at least two groups for levene and fligner", () => {
    expect(() => stats.levene("median", x)).toThrow(InvalidParameterError);
    expect(() => stats.fligner([x])).toThrow(InvalidParameterError);
  });

  it("validates chisquare ddof", () => {
    expect(() => stats.chisquare(T([5, 6, 7]), undefined, -1)).toThrow(InvalidParameterError);
    expect(() => stats.chisquare(T([5, 6, 7]), undefined, 1.5)).toThrow(InvalidParameterError);
    expect(() => stats.chisquare(T([5, 6, 7]), undefined, 2)).toThrow(InvalidParameterError);
  });

  it("validates the median_test ties option", () => {
    expect(() => stats.median_test(x, y, { ties: "middle" as "below" })).toThrow(
      InvalidParameterError
    );
  });
});

describe("NaN propagation", () => {
  const withNaN = T([1, 2, Number.NaN, 4, 5, 6, 7, 8, 9, 10]);
  const clean = T([2.5, 3.1, 1.7, 4.2, 6.6, 5.1, 7.3, 9.9, 8.4, 3.3]);

  it("returns NaN results instead of ranking NaN values", () => {
    const tests: (() => { statistic: number; pvalue: number })[] = [
      () => stats.mannwhitneyu(withNaN, clean),
      () => stats.wilcoxon(withNaN, clean),
      () => stats.kruskal(withNaN, clean),
      () => stats.friedmanchisquare(withNaN, clean, clean),
      () => stats.ks_2samp(withNaN, clean),
      () => stats.kstest(withNaN, "norm"),
      () => stats.normaltest(withNaN),
      () => stats.shapiro(withNaN),
      () => stats.anderson(withNaN),
      () => stats.lilliefors(withNaN),
      () => stats.levene("median", withNaN, clean),
      () => stats.bartlett(withNaN, clean),
      () => stats.fligner([withNaN, clean]),
      () => stats.f_oneway(withNaN, clean),
      () => stats.median_test(withNaN, clean),
      () => stats.runs_test(withNaN),
      () => stats.ttest_1samp(withNaN, 1),
      () => stats.ttest_ind(withNaN, clean),
      () => stats.ttest_rel(withNaN, clean),
    ];
    for (const run of tests) {
      const res = run();
      expect(res.statistic).toBeNaN();
      expect(res.pvalue).toBeNaN();
    }
  });

  it("still reports the critical values for anderson with NaN data", () => {
    const res = stats.anderson(withNaN);
    expect(res.critical_values).toHaveLength(5);
  });
});

describe("edge cases", () => {
  it("mannwhitneyu returns NaN when all values are tied and U1 is reported", () => {
    const res = stats.mannwhitneyu(T([1, 1, 1]), T([1, 1]));
    expect(res.statistic).toBe(3);
    expect(res.pvalue).toBeNaN();
  });

  it("mannwhitneyu refuses an exact p-value with ties", () => {
    expect(() => stats.mannwhitneyu(T([1, 2, 2]), T([2, 3, 4]), { method: "exact" })).toThrow(
      InvalidParameterError
    );
  });

  it("mannwhitneyu refuses an exact p-value for huge samples instead of exhausting memory", () => {
    const big = tensor(new Float64Array(20000).map((_, i) => i));
    const big2 = tensor(new Float64Array(20000).map((_, i) => i + 0.5));
    expect(() => stats.mannwhitneyu(big, big2, { method: "exact" })).toThrow(InvalidParameterError);
  });

  it("wilcoxon rejects empty input and all-zero differences with typed errors", () => {
    expect(() => stats.wilcoxon(T([]))).toThrow(InvalidParameterError);
    expect(() => stats.wilcoxon(T([0, 0, 0]))).toThrow(/all differences are zero/);
    expect(() => stats.wilcoxon(T([0, 0, 0]), undefined, { zeroMethod: "pratt" })).toThrow(
      /all differences are zero/
    );
  });

  it("wilcoxon flattens tensors in row-major order", () => {
    const x = tensor(
      [
        [1.5, 2.5, 3.1],
        [-0.5, 4.2, 0.7],
      ],
      { dtype: "float64" }
    );
    const flat = T([1.5, 2.5, 3.1, -0.5, 4.2, 0.7]);
    const a = stats.wilcoxon(x);
    const b = stats.wilcoxon(flat);
    expect(a.statistic).toBe(b.statistic);
    expect(a.pvalue).toBe(b.pvalue);
  });

  it("anderson needs two distinct finite values", () => {
    expect(() => stats.anderson(T([1]))).toThrow(InvalidParameterError);
    expect(() => stats.anderson(T([2, 2, 2]))).toThrow(InvalidParameterError);
    expect(() => stats.anderson(T([1, 2, Number.POSITIVE_INFINITY]))).toThrow(
      InvalidParameterError
    );
  });

  it("fligner returns NaN when every absolute deviation is equal", () => {
    const res = stats.fligner([T([3, 3, 3]), T([5, 5, 5])]);
    expect(res.statistic).toBeNaN();
    expect(res.pvalue).toBeNaN();
  });

  it("levene is infinite when only the group means of the deviations differ", () => {
    const res = stats.levene("median", T([0, 0, 0, 0]), T([-1, 1, -1, 1]));
    expect(res.statistic).toBe(Number.POSITIVE_INFINITY);
    expect(res.pvalue).toBe(0);
  });

  it("median_test reports one-sided data with typed errors like SciPy", () => {
    expect(() => stats.median_test(T([1, 1, 1]), T([1, 1, 1]), { ties: "above" })).toThrow(
      InvalidParameterError
    );
    expect(() => stats.median_test(T([1, 1, 1]), T([1, 1, 1]))).toThrow(InvalidParameterError);
    expect(() =>
      stats.median_test(T([1, 2, 3]), T([2, 2, 2]), T([4, 5, 6]), { ties: "ignore" })
    ).toThrow(InvalidParameterError);
  });

  it("fisher_exact requires a 2x2 table of non-negative integers", () => {
    expect(() =>
      stats.fisher_exact([[1, 2]] as unknown as [[number, number], [number, number]])
    ).toThrow(InvalidParameterError);
    expect(() =>
      stats.fisher_exact([
        [1, 2],
        [3, -4],
      ])
    ).toThrow(InvalidParameterError);
    expect(() =>
      stats.fisher_exact([
        [1, 2],
        [3, 4.5],
      ])
    ).toThrow(InvalidParameterError);
  });

  it("fisher_exact handles an all-zero table and a very large table", () => {
    const empty = stats.fisher_exact([
      [0, 0],
      [0, 0],
    ]);
    expect(empty.oddsRatio).toBeNaN();
    expect(empty.pvalue).toBe(1);
    // scipy 1.17: fisher_exact([[500500, 500000], [500000, 500000]])
    const big = stats.fisher_exact([
      [500500, 500000],
      [500000, 500000],
    ]);
    expectClose(big.oddsRatio, 1.001, 1e-12);
    expectClose(big.pvalue, 0.7247668862194241, 1e-8);
    expectClose(
      stats.fisher_exact(
        [
          [500500, 500000],
          [500000, 500000],
        ],
        "less"
      ).pvalue,
      0.6386433578353569,
      1e-8
    );
    expectClose(
      stats.fisher_exact(
        [
          [500500, 500000],
          [500000, 500000],
        ],
        "greater"
      ).pvalue,
      0.362416573200693,
      1e-8
    );
    // scipy 1.17: fisher_exact([[100, 2], [3, 200]]).pvalue = 2.871561172512235e-74
    expectClose(
      stats.fisher_exact([
        [100, 2],
        [3, 200],
      ]).pvalue,
      2.871561172512235e-74,
      1e-9
    );
  });

  it("kstest rejects an unknown distribution name and empty data", () => {
    expect(() => stats.kstest(T([0.1, 0.2]), "gamma")).toThrow(InvalidParameterError);
    expect(() => stats.kstest(T([]), "norm")).toThrow(InvalidParameterError);
  });

  it("ks_2samp rejects an empty sample", () => {
    expect(() => stats.ks_2samp(T([]), T([1]))).toThrow(InvalidParameterError);
    expect(() => stats.ks_2samp(T([1]), T([]))).toThrow(InvalidParameterError);
  });

  it("ks_2samp of identical samples has statistic 0 and p-value 1", () => {
    const res = stats.ks_2samp(T([1, 2, 3, 4]), T([1, 2, 3, 4]));
    expect(res.statistic).toBe(0);
    expect(res.pvalue).toBe(1);
  });

  it("ks_2samp with fully separated samples equals 2 / C(7, 3)", () => {
    // scipy 1.17: ks_2samp([1, 2, 3], [4, 5, 6, 7]) -> (1.0, 0.05714285714285715)
    const res = stats.ks_2samp(T([1, 2, 3]), T([4, 5, 6, 7]));
    expect(res.statistic).toBe(1);
    expectClose(res.pvalue, 0.05714285714285715, 1e-12);
  });
});

describe("kernel density estimation (validation)", () => {
  it("rejects non-finite data and non-finite bandwidths", () => {
    expect(() => stats.gaussian_kde([1, 2, Number.NaN])).toThrow(InvalidParameterError);
    expect(() => stats.gaussian_kde([1, 2, Number.POSITIVE_INFINITY])).toThrow(
      InvalidParameterError
    );
    expect(() => stats.gaussian_kde([1, 2, 3], { bwMethod: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => stats.gaussian_kde([1, 2, 3], { bwMethod: Number.POSITIVE_INFINITY })).toThrow(
      InvalidParameterError
    );
    expect(() => stats.gaussian_kde([1, 2, 3], { bwMethod: "plug-in" as "scott" })).toThrow(
      InvalidParameterError
    );
  });

  it("rejects invalid weights", () => {
    expect(() => stats.gaussian_kde([1, 2, 3], { weights: [1, 2] })).toThrow(InvalidParameterError);
    expect(() => stats.gaussian_kde([1, 2, 3], { weights: [1, -1, 1] })).toThrow(
      InvalidParameterError
    );
    expect(() => stats.gaussian_kde([1, 2, 3], { weights: [0, 0, 0] })).toThrow(
      InvalidParameterError
    );
  });

  it("validates the GaussianKDE constructor", () => {
    expect(() => new stats.GaussianKDE(new Float64Array(0), 1)).toThrow(InvalidParameterError);
    expect(() => new stats.GaussianKDE(new Float64Array([1, 2]), 0)).toThrow(InvalidParameterError);
    expect(() => new stats.GaussianKDE(new Float64Array([1, 2]), Number.NaN)).toThrow(
      InvalidParameterError
    );
  });

  it("copies the data so later changes to the input do not alter the estimate", () => {
    const data = new Float64Array([1, 2, 3, 4]);
    const kde = stats.gaussian_kde(data, { bwMethod: 1 });
    const before = kde.evaluate([2.5])[0] ?? Number.NaN;
    data[0] = 100;
    expect(kde.evaluate([2.5])[0]).toBe(before);
  });

  it("uses a bandwidth factor with unit scale when all values are equal or there is one point", () => {
    // scott factor n^(-1/5) with the standard deviation replaced by 1
    expectClose(stats.gaussian_kde([5, 5, 5, 5]).bandwidth, 4 ** -0.2, 1e-14);
    expect(stats.gaussian_kde([5]).bandwidth).toBe(1);
  });

  it("logDensity is -Infinity far outside any floating point range and NaN for NaN", () => {
    const kde = stats.gaussian_kde([0, 1, 2], { bwMethod: 0.5 });
    const out = kde.logDensity([Number.POSITIVE_INFINITY, Number.NaN, 1e3]);
    expect(out[0]).toBe(Number.NEGATIVE_INFINITY);
    expect(out[1]).toBeNaN();
    // log of the Gaussian tail: log(1/3) - log(0.5 sqrt(2 pi)) - 0.5 (1e3 - 2)^2 / 0.25 plus the sum over points
    expect(Number.isFinite(out[2])).toBe(true);
    expect(kde.evaluate([1e3])[0]).toBe(0);
  });

  it("pdf and logpdf are aliases of evaluate and logDensity", () => {
    const kde = stats.gaussian_kde([1, 2, 3, 4]);
    expect(Array.from(kde.pdf([2, 3]))).toEqual(Array.from(kde.evaluate([2, 3])));
    expect(Array.from(kde.logpdf([2, 3]))).toEqual(Array.from(kde.logDensity([2, 3])));
    expect(kde.integrate()).toBe(1);
  });

  it("silverman uses the exact factor (3 n / 4)^(-1/5), not 1.06 n^(-1/5)", () => {
    const data = [1.2, 2.3, 2.9, 3.1, 4.8, 5.0, 5.5, 7.1];
    const n = data.length;
    const mean = data.reduce((a, b) => a + b, 0) / n;
    const sd = Math.sqrt(data.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1));
    expectClose(
      stats.gaussian_kde(data, { bwMethod: "silverman" }).bandwidth,
      ((3 * n) / 4) ** -0.2 * sd,
      1e-13
    );
  });
});

describe("multiple comparisons (validation)", () => {
  it("rejects empty input, out-of-range p-values and invalid alpha for every procedure", () => {
    const procs = [
      stats.bonferroni,
      stats.holm,
      stats.hochberg,
      stats.benjaminiHochberg,
      stats.benjaminiYekutieli,
      stats.sidak,
    ];
    for (const proc of procs) {
      expect(() => proc([])).toThrow(InvalidParameterError);
      expect(() => proc([0.5, 1.5])).toThrow(InvalidParameterError);
      expect(() => proc([0.5, Number.NaN])).toThrow(InvalidParameterError);
      expect(() => proc([0.5], 0)).toThrow(InvalidParameterError);
      expect(() => proc([0.5], Number.NaN)).toThrow(InvalidParameterError);
    }
  });

  it("sidak maps p = 1 to 1 and p = 0 to 0", () => {
    const res = stats.sidak([0, 1], 0.05);
    expect(res.corrected).toEqual([0, 1]);
  });

  it("every procedure keeps the order of the input and never exceeds 1", () => {
    const p = [0.9, 0.001, 0.4, 0.04];
    for (const proc of [
      stats.holm,
      stats.hochberg,
      stats.benjaminiHochberg,
      stats.benjaminiYekutieli,
    ]) {
      const res = proc(p, 0.05);
      expect(res.pvalues).toEqual(p);
      for (const c of res.corrected) {
        expect(c).toBeLessThanOrEqual(1);
        expect(c).toBeGreaterThanOrEqual(0);
      }
      expect(res.corrected[1]).toBeLessThan(res.corrected[0] ?? 0);
    }
  });

  it("hochberg rejects whenever holm does", () => {
    const p = [0.012, 0.02, 0.019, 0.3, 0.001, 0.045];
    const h = stats.holm(p, 0.05);
    const hb = stats.hochberg(p, 0.05);
    for (let i = 0; i < p.length; i++) {
      if (h.rejected[i]) expect(hb.rejected[i]).toBe(true);
      expect(hb.corrected[i] ?? 0).toBeLessThanOrEqual((h.corrected[i] ?? 0) + 1e-15);
    }
  });
});

describe("power analysis (validation)", () => {
  it("rejects non-finite and out-of-range inputs", () => {
    expect(() => stats.tTestPower({ effectSize: Number.NaN, nObs: 10, alpha: 0.05 })).toThrow(
      InvalidParameterError
    );
    expect(() => stats.tTestPower({ effectSize: 0.5, nObs: 10, alpha: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => stats.tTestPower({ effectSize: 0.5, nObs: 10, power: Number.NaN })).toThrow(
      InvalidParameterError
    );
    expect(() => stats.tTestPower({ effectSize: 0.5, nObs: 10.5, alpha: 0.05 })).toThrow(
      InvalidParameterError
    );
    expect(() => stats.tTestPower({ effectSize: 0.5, nObs: 10, alpha: 0.05, ratio: 0 })).toThrow(
      InvalidParameterError
    );
    expect(() =>
      stats.tTestPower({ effectSize: 0.5, nObs: 10, alpha: 0.05, alternative: 3 as 1 })
    ).toThrow(InvalidParameterError);
  });

  it("reports unreachable targets instead of returning a wrong value", () => {
    expect(() => stats.tTestPower({ effectSize: 0, alpha: 0.05, power: 0.8 })).toThrow(
      InvalidParameterError
    );
    expect(() => stats.tTestPower({ effectSize: 1e-9, alpha: 0.05, power: 0.99 })).toThrow(
      InvalidParameterError
    );
    // The power at effectSize = 0 equals alpha, so a smaller target has no solution.
    expect(() => stats.tTestPower({ nObs: 20, alpha: 0.2, power: 0.1 })).toThrow(
      InvalidParameterError
    );
  });

  it("returns exactly the inputs it was given and a power in [0, 1]", () => {
    const res = stats.tTestPower({ effectSize: 3, nObs: 200, alpha: 0.001 });
    expect(res.effectSize).toBe(3);
    expect(res.nObs).toBe(200);
    expect(res.alpha).toBe(0.001);
    expect(res.power).toBeLessThanOrEqual(1);
    expect(res.power).toBeGreaterThan(0.999);
  });

  it("the power at effectSize 0 is alpha for the two-sided and the one-sided test", () => {
    expectClose(stats.tTestPower({ effectSize: 0, nObs: 15, alpha: 0.05 }).power, 0.05, 1e-9);
    expectClose(
      stats.tTestPower({ effectSize: 0, nObs: 15, alpha: 0.05, alternative: 1 }).power,
      0.05,
      1e-9
    );
  });

  it("solves for nObs with the documented example", () => {
    expect(stats.tTestPower({ effectSize: 0.5, alpha: 0.05, power: 0.8 }).nObs).toBe(64);
    expectClose(
      stats.tTestPower({ effectSize: 0.5, nObs: 50, alpha: 0.05 }).power,
      0.696893405533384,
      1e-9
    );
  });
});

describe("f_twoway and lilliefors (independent references)", () => {
  it("f_twoway matches the analysis of variance decomposition (numpy, scipy.stats.f)", () => {
    const data = [
      [
        [-1.738, -1.337, -1.361, -0.352],
        [-2.113, 0.011, -0.757, 1.094],
        [1.357, 1.792, 1.167, 0.347],
      ],
      [
        [1.76, 2.405, 0.246, 1.51],
        [1.057, 2.54, 0.263, 0.798],
        [1.662, 1.558, -0.339, 1.66],
      ],
    ];
    const res = stats.f_twoway(data);
    expectClose(res.factorA.statistic, 13.715825796652783, 1e-9);
    expectClose(res.factorA.pvalue, 0.0016259776206391074, 1e-9);
    expectClose(res.factorB.statistic, 2.5614028465045378, 1e-9);
    expectClose(res.factorB.pvalue, 0.1049754581087256, 1e-9);
    expectClose(res.interaction.statistic, 4.231202852951457, 1e-9);
    expectClose(res.interaction.pvalue, 0.031173720298187234, 1e-9);
  });

  it("f_twoway rejects unbalanced designs and non-finite data", () => {
    expect(() =>
      stats.f_twoway([
        [
          [1, 2],
          [3, 4],
        ],
        [[5, 6], [7]],
      ])
    ).toThrow(InvalidParameterError);
    expect(() =>
      stats.f_twoway([
        [
          [1, 2],
          [3, Number.NaN],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ])
    ).toThrow(InvalidParameterError);
    expect(() =>
      stats.f_twoway([
        [[1], [3]],
        [[5], [7]],
      ])
    ).toThrow(InvalidParameterError);
  });

  it("lilliefors follows the Dallal-Wilkinson formula with Stephens' large p-value fits (R nortest::lillie.test)", () => {
    const small = stats.lilliefors(T([0.2, 1.3, 1.1, 2.5, 3.9, 0.7, 1.9, 2.2, 6.1, 0.4, 1.2, 1.7]));
    expectClose(small.statistic, 0.19948201277312927, 1e-12);
    expectClose(small.pvalue, 0.20759800383398996, 1e-9);
    const mid = stats.lilliefors(T([2.1, 1.9, 2.4, 2.2, 2.0, 2.6, 1.7, 2.3, 2.5, 3.9]));
    expectClose(mid.statistic, 0.24645336036902077, 1e-12);
    expectClose(mid.pvalue, 0.08629676218820981, 1e-9);
    const large = stats.lilliefors(
      T(Array.from({ length: 150 }, (_, i) => -Math.log((i + 0.5) / 150)))
    );
    expectClose(large.statistic, 0.157319789058393, 1e-12);
    expectClose(large.pvalue, 1.0024634228301703e-9, 1e-8);
  });
});

describe("review regressions", () => {
  it("fisher_exact stays fast for huge counts and rejects totals above 2^40", () => {
    // Reference values from scipy.stats.fisher_exact([[3000, 2000], [2000, 1000]]).
    const res = stats.fisher_exact([
      [3000, 2000],
      [2000, 1000],
    ]);
    expectClose(res.pvalue, 2.4312963376688072e-9, 1e-9);
    const less = stats.fisher_exact(
      [
        [3000, 2000],
        [2000, 1000],
      ],
      "less"
    );
    expectClose(less.pvalue, 1.2648669652609989e-9, 1e-9);
    // The observed table is hundreds of standard deviations from the mode: the p-value
    // underflows to zero and the call must return immediately.
    const start = Date.now();
    const far = stats.fisher_exact(
      [
        [3e11, 2e11],
        [2e11, 1e11],
      ],
      "less"
    );
    expect(far.pvalue).toBe(0);
    expect(Date.now() - start).toBeLessThan(5000);
    expect(() =>
      stats.fisher_exact([
        [1e300, 5],
        [1e300, 1e300],
      ])
    ).toThrow(InvalidParameterError);
  });

  it("tTestPower accepts a target power equal to alpha when solving for the effect size", () => {
    const res = stats.tTestPower({ nObs: 20, alpha: 0.05, power: 0.05 });
    expect(res.effectSize).toBeLessThan(1e-6);
    expect(() => stats.tTestPower({ nObs: 20, alpha: 0.05, power: 0.04 })).toThrow(
      InvalidParameterError
    );
  });
});
