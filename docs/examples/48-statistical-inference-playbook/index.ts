/**
 * Example 48: Statistical Inference Playbook
 *
 * Confidence intervals, a bootstrap, a Gaussian kernel density estimate,
 * multiple-comparison correction and power analysis, applied to a small
 * A/B experiment on a checkout flow.
 */

import { mkdir } from "node:fs/promises";
import { tensor } from "deepbox/ndarray";
import { axhline, figure, groupedBar, kdeplot, legend, saveFig } from "deepbox/plot";
import {
  benjaminiHochberg,
  benjaminiYekutieli,
  bonferroni,
  bootstrap,
  cohenD,
  gaussianKde,
  hochberg,
  meanConfidenceInterval,
  meanConfidenceIntervalZ,
  meanDiffConfidenceInterval,
  proportionConfidenceInterval,
  tTestPower,
  ttestInd,
} from "deepbox/stats";

const OUTPUT_DIR = "docs/examples/48-statistical-inference-playbook/output";

console.log("=".repeat(72));
console.log("Example 48: Statistical Inference Playbook");
console.log("=".repeat(72));

await mkdir(OUTPUT_DIR, { recursive: true });

// A small A/B dataset for a checkout experiment. Each metric has one value per session or day.
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
// Part 1: Mean confidence intervals and the uplift interval
// ============================================================================
console.log("\nPart 1: Confidence Intervals");
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
// Part 2: Conversion-rate intervals for proportions
// ============================================================================
console.log("\nPart 2: Proportion Intervals");
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

// The default is the Wald interval. The Wilson score interval behaves better near 0% and 100%.
const treatmentWilson = proportionConfidenceInterval(
  treatmentConversions.successes,
  treatmentConversions.total,
  0.95,
  "wilson"
);
console.log(
  `Treatment, Wilson interval: ${(treatmentWilson.mean * 100).toFixed(2)}% | 95% CI [${(treatmentWilson.lower * 100).toFixed(2)}%, ${(treatmentWilson.upper * 100).toFixed(2)}%]`
);

// ============================================================================
// Part 3: Bootstrap the mean uplift (resample, recompute, read off percentiles)
// ============================================================================
console.log("\nPart 3: Bootstrap Uncertainty");
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
// Part 4: Gaussian KDE (a smooth density estimate)
// ============================================================================
console.log("\nPart 4: Gaussian KDE");
console.log("-".repeat(72));

const controlKde = gaussianKde(controlRevenuePerSession, { bwMethod: "silverman" });
const treatmentKde = gaussianKde(treatmentRevenuePerSession, { bwMethod: "silverman" });
const probePoints = [90, 95, 100];
const controlDensity = Array.from(controlKde.evaluate(probePoints), (value) => value.toFixed(5));
const treatmentDensity = Array.from(treatmentKde.evaluate(probePoints), (value) =>
  value.toFixed(5)
);

console.log(
  `KDE bandwidths: control ${controlKde.bandwidth.toFixed(3)}, treatment ${treatmentKde.bandwidth.toFixed(3)}`
);
console.log(`Control KDE at [90, 95, 100]:   ${controlDensity.join(", ")}`);
console.log(`Treatment KDE at [90, 95, 100]: ${treatmentDensity.join(", ")}`);

const densityFigure = figure({ width: 840, height: 520 });
kdeplot(tensor(controlRevenuePerSession), {
  color: "#1d4ed8",
  label: "control revenue/session",
  bwMethod: "silverman",
});
kdeplot(tensor(treatmentRevenuePerSession), {
  color: "#059669",
  label: "treatment revenue/session",
  bwMethod: "silverman",
});
legend();
await saveFig(`${OUTPUT_DIR}/revenue-density.svg`, { figure: densityFigure });
console.log(`Saved revenue density plot: ${OUTPUT_DIR}/revenue-density.svg`);

// ============================================================================
// Part 5: Correct p-values for multiple comparisons
// ============================================================================
console.log("\nPart 5: Multiple Comparisons");
console.log("-".repeat(72));

const metricTests = [
  {
    metric: "revenue_per_session",
    pvalue: ttestInd(tensor(controlRevenuePerSession), tensor(treatmentRevenuePerSession)).pvalue,
  },
  {
    metric: "latency_ms",
    pvalue: ttestInd(tensor(controlLatencyMs), tensor(treatmentLatencyMs)).pvalue,
  },
  {
    metric: "csat",
    pvalue: ttestInd(tensor(controlCsat), tensor(treatmentCsat)).pvalue,
  },
  {
    metric: "resolution_rate",
    pvalue: ttestInd(tensor(controlResolutionRate), tensor(treatmentResolutionRate)).pvalue,
  },
];

const rawPvalues = metricTests.map((test) => test.pvalue);
// Bonferroni and Hochberg control the chance of any false positive.
// Benjamini-Hochberg and Benjamini-Yekutieli control the false discovery rate.
// Benjamini-Yekutieli is the variant that stays valid when the tests are dependent.
const bhCorrection = benjaminiHochberg(rawPvalues, 0.05);
const byCorrection = benjaminiYekutieli(rawPvalues, 0.05);
const bonfCorrection = bonferroni(rawPvalues, 0.05);
const hochbergCorrection = hochberg(rawPvalues, 0.05);

const sci = (value: number | undefined): string => (value ?? Number.NaN).toExponential(2);
for (const [index, test] of metricTests.entries()) {
  console.log(
    `${test.metric.padEnd(20)} raw=${sci(test.pvalue)} | BH=${sci(bhCorrection.corrected[index])} | BY=${sci(byCorrection.corrected[index])} | Bonferroni=${sci(bonfCorrection.corrected[index])} | Hochberg=${sci(hochbergCorrection.corrected[index])}`
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
// Part 6: Power analysis
// ============================================================================
console.log("\nPart 6: Power Analysis");
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
console.log("\nKey Takeaways");
console.log("-".repeat(72));
console.log(
  "• meanConfidenceInterval uses the t distribution, for an unknown population variance."
);
console.log(
  "• meanConfidenceIntervalZ uses the normal distribution, for a known standard deviation."
);
console.log("• bootstrap resamples the data, so it assumes less about the distribution.");
console.log("• gaussianKde shows overlap, skew and several peaks that a mean hides.");
console.log("• Correct p-values when one experiment tests several metrics.");
console.log(
  "• tTestPower turns an observed effect size into the sample size for a follow-up test."
);

console.log("\nStatistical Inference Playbook Complete!");
console.log("=".repeat(72));
