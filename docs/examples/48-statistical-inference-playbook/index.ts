/**
 * Example 48: Statistical Inference Playbook
 *
 * Covers the v1.0.0 inference layer that was not represented in the earlier
 * stats examples: confidence intervals, bootstrap uncertainty, Gaussian KDE,
 * multiple-comparison correction, and power analysis.
 */

import { mkdir } from "node:fs/promises";
import { tensor } from "deepbox/ndarray";
import { axhline, figure, groupedBar, kdeplot, legend, saveFig } from "deepbox/plot";
import {
  benjaminiHochberg,
  bonferroni,
  bootstrap,
  cohenD,
  gaussian_kde,
  meanConfidenceInterval,
  meanConfidenceIntervalZ,
  meanDiffConfidenceInterval,
  proportionConfidenceInterval,
  tTestPower,
  ttest_ind,
} from "deepbox/stats";

const OUTPUT_DIR = "docs/examples/48-statistical-inference-playbook/output";

console.log("=".repeat(72));
console.log("Example 48: Statistical Inference Playbook");
console.log("=".repeat(72));

await mkdir(OUTPUT_DIR, { recursive: true });

// A compact A/B rollout dataset for a checkout experiment.
const controlRevenuePerSession = [86, 91, 88, 94, 90, 96, 84, 89, 92, 87, 95, 90];
const treatmentRevenuePerSession = [94, 101, 99, 104, 100, 107, 92, 98, 102, 97, 105, 100];

const controlLatencyMs = [1210, 1185, 1224, 1178, 1202, 1194, 1218, 1189];
const treatmentLatencyMs = [1124, 1098, 1116, 1108, 1121, 1095, 1113, 1104];

const controlCsat = [4.1, 4.2, 4.0, 4.3, 4.1, 4.2, 4.0, 4.1];
const treatmentCsat = [4.3, 4.4, 4.2, 4.5, 4.4, 4.3, 4.2, 4.4];

const controlResolutionRate = [0.74, 0.71, 0.76, 0.73, 0.75, 0.72, 0.74, 0.73];
const treatmentResolutionRate = [0.81, 0.79, 0.83, 0.82, 0.8, 0.81, 0.84, 0.8];

const controlConversions = { successes: 158, total: 200 };
const treatmentConversions = { successes: 186, total: 205 };

// ============================================================================
// Part 1: Mean confidence intervals and uplift intervals
// ============================================================================
console.log("\n📊 Part 1: Confidence Intervals");
console.log("-".repeat(72));

const controlRevenueCi = meanConfidenceInterval(controlRevenuePerSession, 0.95);
const treatmentRevenueCi = meanConfidenceInterval(treatmentRevenuePerSession, 0.95);
const revenueUpliftCi = meanDiffConfidenceInterval(
  treatmentRevenuePerSession,
  controlRevenuePerSession,
  0.95
);

console.log(
  `Control mean revenue/session:   ${controlRevenueCi.mean.toFixed(2)}  | 95% CI [${controlRevenueCi.lower.toFixed(2)}, ${controlRevenueCi.upper.toFixed(2)}]`
);
console.log(
  `Treatment mean revenue/session: ${treatmentRevenueCi.mean.toFixed(2)} | 95% CI [${treatmentRevenueCi.lower.toFixed(2)}, ${treatmentRevenueCi.upper.toFixed(2)}]`
);
console.log(
  `Treatment - control uplift:     ${revenueUpliftCi.mean.toFixed(2)}  | 95% CI [${revenueUpliftCi.lower.toFixed(2)}, ${revenueUpliftCi.upper.toFixed(2)}]`
);

const treatmentLatencyCiZ = meanConfidenceIntervalZ(treatmentLatencyMs, 36, 0.99);
console.log(
  `Known-noise latency z-interval: ${treatmentLatencyCiZ.mean.toFixed(1)} ms | 99% CI [${treatmentLatencyCiZ.lower.toFixed(1)}, ${treatmentLatencyCiZ.upper.toFixed(1)}]`
);

// ============================================================================
// Part 2: Conversion-rate intervals
// ============================================================================
console.log("\n✅ Part 2: Proportion Intervals");
console.log("-".repeat(72));

const controlConversionCi = proportionConfidenceInterval(
  controlConversions.successes,
  controlConversions.total,
  0.95
);
const treatmentConversionCi = proportionConfidenceInterval(
  treatmentConversions.successes,
  treatmentConversions.total,
  0.95
);

console.log(
  `Control conversion rate:   ${(controlConversionCi.mean * 100).toFixed(2)}% | 95% CI [${(controlConversionCi.lower * 100).toFixed(2)}%, ${(controlConversionCi.upper * 100).toFixed(2)}%]`
);
console.log(
  `Treatment conversion rate: ${(treatmentConversionCi.mean * 100).toFixed(2)}% | 95% CI [${(treatmentConversionCi.lower * 100).toFixed(2)}%, ${(treatmentConversionCi.upper * 100).toFixed(2)}%]`
);

// ============================================================================
// Part 3: Bootstrap the mean uplift
// ============================================================================
console.log("\n♻️  Part 3: Bootstrap Uncertainty");
console.log("-".repeat(72));

const upliftByMatchedCell = treatmentRevenuePerSession.map(
  (value, index) => value - (controlRevenuePerSession[index] ?? 0)
);
const upliftBootstrap = bootstrap(
  upliftByMatchedCell,
  (sample) => sample.reduce((sum, value) => sum + value, 0) / sample.length,
  {
    nResamples: 4000,
    seed: 48,
    confidenceLevel: 0.95,
  }
);

console.log(
  `Bootstrap mean uplift estimate: ${upliftBootstrap.estimate.toFixed(2)} | percentile CI [${upliftBootstrap.ci[0].toFixed(2)}, ${upliftBootstrap.ci[1].toFixed(2)}]`
);
console.log(`Bootstrap resamples generated:  ${upliftBootstrap.samples.length}`);

// ============================================================================
// Part 4: Gaussian KDE and density diagnostics
// ============================================================================
console.log("\n📈 Part 4: Gaussian KDE");
console.log("-".repeat(72));

const controlKde = gaussian_kde(controlRevenuePerSession, { bw_method: "silverman" });
const treatmentKde = gaussian_kde(treatmentRevenuePerSession, { bw_method: "silverman" });
const probePoints = [90, 95, 100];
const controlDensity = Array.from(controlKde.evaluate(probePoints), (value) => value.toFixed(5));
const treatmentDensity = Array.from(treatmentKde.evaluate(probePoints), (value) =>
  value.toFixed(5)
);

console.log(
  `KDE bandwidths -> control=${controlKde.bandwidth.toFixed(3)}, treatment=${treatmentKde.bandwidth.toFixed(3)}`
);
console.log(`Control KDE at [90, 95, 100]:   ${controlDensity.join(", ")}`);
console.log(`Treatment KDE at [90, 95, 100]: ${treatmentDensity.join(", ")}`);

const densityFigure = figure({ width: 840, height: 520 });
kdeplot(tensor(controlRevenuePerSession), {
  color: "#1d4ed8",
  label: "control revenue/session",
  bw_method: "silverman",
});
kdeplot(tensor(treatmentRevenuePerSession), {
  color: "#059669",
  label: "treatment revenue/session",
  bw_method: "silverman",
});
legend();
await saveFig(`${OUTPUT_DIR}/revenue-density.svg`, { figure: densityFigure });
console.log(`Saved revenue density plot: ${OUTPUT_DIR}/revenue-density.svg`);

// ============================================================================
// Part 5: Correct for multiple comparisons
// ============================================================================
console.log("\n🧪 Part 5: Multiple Comparisons");
console.log("-".repeat(72));

const metricTests = [
  {
    metric: "revenue_per_session",
    pvalue: ttest_ind(tensor(controlRevenuePerSession), tensor(treatmentRevenuePerSession)).pvalue,
  },
  {
    metric: "latency_ms",
    pvalue: ttest_ind(tensor(controlLatencyMs), tensor(treatmentLatencyMs)).pvalue,
  },
  {
    metric: "csat",
    pvalue: ttest_ind(tensor(controlCsat), tensor(treatmentCsat)).pvalue,
  },
  {
    metric: "resolution_rate",
    pvalue: ttest_ind(tensor(controlResolutionRate), tensor(treatmentResolutionRate)).pvalue,
  },
];

const rawPvalues = metricTests.map((test) => test.pvalue);
const bhCorrection = benjaminiHochberg(rawPvalues, 0.05);
const bonfCorrection = bonferroni(rawPvalues, 0.05);

for (const [index, test] of metricTests.entries()) {
  console.log(
    `${test.metric.padEnd(20)} raw=${test.pvalue.toFixed(6)} | BH=${bhCorrection.corrected[index]?.toFixed(6)} | Bonferroni=${bonfCorrection.corrected[index]?.toFixed(6)}`
  );
}

const correctionFigure = figure({ width: 840, height: 520 });
groupedBar(tensor([1, 2, 3, 4]), [tensor(rawPvalues), tensor(Array.from(bhCorrection.corrected))], {
  colors: ["#2563eb", "#f97316"],
  labels: ["raw p-values", "BH corrected"],
});
axhline(0.05, { color: "#991b1b", linewidth: 2, label: "alpha = 0.05" });
legend();
await saveFig(`${OUTPUT_DIR}/multiple-comparisons.svg`, { figure: correctionFigure });
console.log(`Saved multiple-comparison plot: ${OUTPUT_DIR}/multiple-comparisons.svg`);

// ============================================================================
// Part 6: Power planning
// ============================================================================
console.log("\n🎯 Part 6: Power Analysis");
console.log("-".repeat(72));

const observedEffectSize = Math.abs(cohenD(controlRevenuePerSession, treatmentRevenuePerSession));
const currentPower = tTestPower({
  effectSize: observedEffectSize,
  nObs: controlRevenuePerSession.length,
  alpha: 0.05,
});
const requiredSample = tTestPower({
  effectSize: observedEffectSize,
  alpha: 0.05,
  power: 0.9,
});

console.log(`Observed effect size (|Cohen's d|): ${observedEffectSize.toFixed(3)}`);
console.log(
  `Current power at n=${controlRevenuePerSession.length}: ${currentPower.power.toFixed(3)}`
);
console.log(`Per-arm sample size for 90% power:   ${requiredSample.nObs}`);

// ============================================================================
// Summary
// ============================================================================
console.log("\n💡 Key Takeaways");
console.log("-".repeat(72));
console.log(
  "• Use t-based confidence intervals for sample means when the population variance is unknown."
);
console.log(
  "• Use z-intervals when instrumentation or historical monitoring gives you a known noise level."
);
console.log(
  "• Bootstrap resampling gives a robust uncertainty estimate when you want fewer distributional assumptions."
);
console.log(
  "• Gaussian KDE is useful for inspecting overlap, skew, and multi-modal behavior before rollout decisions."
);
console.log(
  "• Correcting p-values matters once you inspect multiple metrics in the same experiment."
);
console.log(
  "• Power analysis turns an observed effect into a concrete follow-up sample-size plan."
);

console.log("\n✅ Statistical Inference Playbook Complete!");
console.log("=".repeat(72));
