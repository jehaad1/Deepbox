/**
 * Experimentation Platform
 *
 * A production-style experimentation workflow for Deepbox v1.0.0 that combines
 * DataFrame reporting, confidence intervals, bootstrap uplift analysis,
 * multiple-comparison correction, KDE diagnostics, and sample-size planning.
 */

import { mkdir, writeFile } from "node:fs/promises";
import { DataFrame } from "deepbox/dataframe";
import { tensor } from "deepbox/ndarray";
import { axhline, figure, groupedBar, kdeplot, legend, saveFig } from "deepbox/plot";
import { Generator } from "deepbox/random";
import {
  benjaminiHochberg,
  bootstrap,
  cohenD,
  meanConfidenceInterval,
  meanConfidenceIntervalZ,
  meanDiffConfidenceInterval,
  proportionConfidenceInterval,
  tTestPower,
  ttest_ind,
} from "deepbox/stats";

const OUTPUT_DIR = "docs/projects/09-experimentation-platform/output";
const TOTAL_SESSIONS = 3600;
const RANDOM_SEED = 20260327;

type Variant = "control" | "streamlined-checkout" | "smart-bundle";
type Device = "mobile" | "desktop";
type Segment = "self-serve" | "mid-market" | "enterprise";
type Region = "gcc" | "europe" | "north-america";

type SessionRecord = {
  readonly variant: Variant;
  readonly device: Device;
  readonly segment: Segment;
  readonly region: Region;
  readonly converted: number;
  readonly retained7d: number;
  readonly revenuePerSession: number;
  readonly orderValue: number;
  readonly latencyMs: number;
};

const rng = new Generator(RANDOM_SEED);

function choose<T>(values: readonly T[]): T {
  return values[rng.randint(0, values.length)] ?? values[0]!;
}

function clampProbability(value: number): number {
  return Math.max(0.001, Math.min(0.98, value));
}

function generateSessions(count: number): SessionRecord[] {
  const variants: readonly Variant[] = ["control", "streamlined-checkout", "smart-bundle"];
  const devices: readonly Device[] = ["mobile", "desktop"];
  const segments: readonly Segment[] = ["self-serve", "mid-market", "enterprise"];
  const regions: readonly Region[] = ["gcc", "europe", "north-america"];

  const sessions: SessionRecord[] = [];

  for (let i = 0; i < count; i++) {
    const variant = variants[i % variants.length] ?? "control";
    const device = choose(devices);
    const segment = choose(segments);
    const region = choose(regions);

    const baseConversion =
      segment === "enterprise" ? 0.12 : segment === "mid-market" ? 0.085 : 0.06;
    const devicePenalty = device === "mobile" ? -0.012 : 0;
    const regionAdjustment = region === "gcc" ? 0.004 : region === "north-america" ? 0.002 : -0.001;
    const variantAdjustment =
      variant === "streamlined-checkout"
        ? device === "mobile"
          ? 0.02
          : 0.013
        : variant === "smart-bundle"
          ? 0.008
          : 0;

    const conversionProbability = clampProbability(
      baseConversion + devicePenalty + regionAdjustment + variantAdjustment
    );
    const converted = rng.random() < conversionProbability ? 1 : 0;

    const latencyBase = device === "mobile" ? 1360 : 1020;
    const segmentLatency = segment === "enterprise" ? 40 : segment === "self-serve" ? -25 : 0;
    const variantLatencyShift =
      variant === "streamlined-checkout" ? -120 : variant === "smart-bundle" ? 45 : 0;
    const latencyMs = Math.max(
      680,
      rng.normal(latencyBase + segmentLatency + variantLatencyShift, 55)
    );

    const orderValueBase = segment === "enterprise" ? 340 : segment === "mid-market" ? 220 : 95;
    const bundleLift =
      variant === "smart-bundle" ? 42 : variant === "streamlined-checkout" ? 16 : 0;
    const orderValue = converted
      ? Math.max(45, rng.normal(orderValueBase + bundleLift, orderValueBase * 0.18))
      : 0;

    const retentionBase = segment === "enterprise" ? 0.74 : segment === "mid-market" ? 0.62 : 0.48;
    const retentionShift =
      variant === "streamlined-checkout" ? 0.02 : variant === "smart-bundle" ? 0.035 : 0;
    const retained7d =
      converted && rng.random() < clampProbability(retentionBase + retentionShift) ? 1 : 0;

    sessions.push({
      variant,
      device,
      segment,
      region,
      converted,
      retained7d,
      revenuePerSession: Number(orderValue.toFixed(2)),
      orderValue: Number(orderValue.toFixed(2)),
      latencyMs: Number(latencyMs.toFixed(2)),
    });
  }

  return sessions;
}

function sessionsForVariant(sessions: readonly SessionRecord[], variant: Variant): SessionRecord[] {
  return sessions.filter((session) => session.variant === variant);
}

function numericValues(
  sessions: readonly SessionRecord[],
  selector: (session: SessionRecord) => number
): number[] {
  return sessions.map(selector);
}

function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function sum(values: readonly number[]): number {
  return values.reduce((accumulator, value) => accumulator + value, 0);
}

function buildVariantSummary(
  variant: Variant,
  sessions: readonly SessionRecord[]
): {
  readonly variant: Variant;
  readonly sessions: number;
  readonly conversionRate: number;
  readonly conversionCi: ReturnType<typeof proportionConfidenceInterval>;
  readonly revenueMean: number;
  readonly revenueCi: ReturnType<typeof meanConfidenceInterval>;
  readonly retentionRate: number;
  readonly retentionCi: ReturnType<typeof proportionConfidenceInterval>;
  readonly latencyCi: ReturnType<typeof meanConfidenceIntervalZ>;
} {
  const conversions = numericValues(sessions, (session) => session.converted);
  const retention = numericValues(sessions, (session) => session.retained7d);
  const revenue = numericValues(sessions, (session) => session.revenuePerSession);
  const latency = numericValues(sessions, (session) => session.latencyMs);

  return {
    variant,
    sessions: sessions.length,
    conversionRate: mean(conversions),
    conversionCi: proportionConfidenceInterval(sum(conversions), sessions.length, 0.95),
    revenueMean: mean(revenue),
    revenueCi: meanConfidenceInterval(revenue, 0.95),
    retentionRate: mean(retention),
    retentionCi: proportionConfidenceInterval(sum(retention), sessions.length, 0.95),
    latencyCi: meanConfidenceIntervalZ(latency, 55, 0.95),
  };
}

type PairwiseInference = {
  readonly variant: Exclude<Variant, "control">;
  readonly revenuePvalue: number;
  readonly revenueCorrected: number;
  readonly revenueDiffCi: ReturnType<typeof meanDiffConfidenceInterval>;
  readonly latencyPvalue: number;
  readonly latencyCorrected: number;
};

console.log("═".repeat(72));
console.log("  EXPERIMENTATION PLATFORM");
console.log("  Deepbox v1.0.0 production example");
console.log("═".repeat(72));

await mkdir(OUTPUT_DIR, { recursive: true });

const sessions = generateSessions(TOTAL_SESSIONS);
const experimentFrame = new DataFrame({
  variant: sessions.map((session) => session.variant),
  device: sessions.map((session) => session.device),
  segment: sessions.map((session) => session.segment),
  region: sessions.map((session) => session.region),
  converted: sessions.map((session) => session.converted),
  retained7d: sessions.map((session) => session.retained7d),
  revenuePerSession: sessions.map((session) => session.revenuePerSession),
  orderValue: sessions.map((session) => session.orderValue),
  latencyMs: sessions.map((session) => session.latencyMs),
});

// ============================================================================
// Step 1: Operational summary
// ============================================================================
console.log("\n📊 STEP 1: Experiment Operations Summary");
console.log("─".repeat(72));
console.log(`Sessions generated: ${sessions.length}`);
console.log("Variant-level numeric means:");
console.log(experimentFrame.groupBy("variant").mean().toString());
console.log("\nVariant × device latency means:");
console.log(experimentFrame.groupBy(["variant", "device"]).mean().toString());

// ============================================================================
// Step 2: Variant scorecards
// ============================================================================
console.log("\n🧾 STEP 2: Variant Scorecards");
console.log("─".repeat(72));

const controlSessions = sessionsForVariant(sessions, "control");
const streamlinedSessions = sessionsForVariant(sessions, "streamlined-checkout");
const bundleSessions = sessionsForVariant(sessions, "smart-bundle");

const variantSummaries = [
  buildVariantSummary("control", controlSessions),
  buildVariantSummary("streamlined-checkout", streamlinedSessions),
  buildVariantSummary("smart-bundle", bundleSessions),
];

for (const summary of variantSummaries) {
  console.log(
    `${summary.variant.padEnd(21)} conv=${(summary.conversionRate * 100).toFixed(2)}% revenue/session=${summary.revenueMean.toFixed(2)} retention=${(summary.retentionRate * 100).toFixed(2)}% latency=${summary.latencyCi.mean.toFixed(1)}ms`
  );
}

// ============================================================================
// Step 3: Pairwise inference and correction
// ============================================================================
console.log("\n🧪 STEP 3: Pairwise Inference");
console.log("─".repeat(72));

const pairwiseCandidates = [
  {
    variant: "streamlined-checkout" as const,
    revenue: numericValues(streamlinedSessions, (session) => session.revenuePerSession),
    latency: numericValues(streamlinedSessions, (session) => session.latencyMs),
  },
  {
    variant: "smart-bundle" as const,
    revenue: numericValues(bundleSessions, (session) => session.revenuePerSession),
    latency: numericValues(bundleSessions, (session) => session.latencyMs),
  },
];
const controlRevenue = numericValues(controlSessions, (session) => session.revenuePerSession);
const controlLatency = numericValues(controlSessions, (session) => session.latencyMs);

const rawRevenuePvalues = pairwiseCandidates.map(
  (candidate) => ttest_ind(tensor(controlRevenue), tensor(candidate.revenue)).pvalue
);
const rawLatencyPvalues = pairwiseCandidates.map(
  (candidate) => ttest_ind(tensor(controlLatency), tensor(candidate.latency)).pvalue
);

const revenueCorrection = benjaminiHochberg(rawRevenuePvalues, 0.05);
const latencyCorrection = benjaminiHochberg(rawLatencyPvalues, 0.05);

const pairwiseInference: PairwiseInference[] = pairwiseCandidates.map((candidate, index) => ({
  variant: candidate.variant,
  revenuePvalue: rawRevenuePvalues[index] ?? 1,
  revenueCorrected: revenueCorrection.corrected[index] ?? 1,
  revenueDiffCi: meanDiffConfidenceInterval(candidate.revenue, controlRevenue, 0.95),
  latencyPvalue: rawLatencyPvalues[index] ?? 1,
  latencyCorrected: latencyCorrection.corrected[index] ?? 1,
}));

for (const result of pairwiseInference) {
  console.log(
    `${result.variant.padEnd(21)} revenue p=${result.revenuePvalue.toFixed(6)} -> ${result.revenueCorrected.toFixed(6)} | latency p=${result.latencyPvalue.toFixed(6)} -> ${result.latencyCorrected.toFixed(6)}`
  );
}

const winner =
  variantSummaries.slice().sort((left, right) => right.revenueMean - left.revenueMean)[0]
    ?.variant ?? "control";
console.log(`Selected winner by revenue/session: ${winner}`);

// ============================================================================
// Step 4: Bootstrap uplift and power planning
// ============================================================================
console.log("\n🎯 STEP 4: Decision Support");
console.log("─".repeat(72));

const winnerSessions = sessionsForVariant(sessions, winner);
const winnerRevenue = numericValues(winnerSessions, (session) => session.revenuePerSession);

const cellKeys = Array.from(
  new Set(sessions.map((session) => `${session.segment}:${session.device}`))
);
const cellUplifts = cellKeys.map((key) => {
  const [segment, device] = key.split(":") as [Segment, Device];
  const controlCell = sessions.filter(
    (session) =>
      session.variant === "control" && session.segment === segment && session.device === device
  );
  const winnerCell = sessions.filter(
    (session) =>
      session.variant === winner && session.segment === segment && session.device === device
  );
  return (
    mean(numericValues(winnerCell, (session) => session.revenuePerSession)) -
    mean(numericValues(controlCell, (session) => session.revenuePerSession))
  );
});

const upliftBootstrap = bootstrap(cellUplifts, (sample) => mean(sample), {
  nResamples: 5000,
  seed: RANDOM_SEED,
  confidenceLevel: 0.95,
});
const observedEffectSize = Math.abs(cohenD(controlRevenue, winnerRevenue));
const currentPower = tTestPower({
  effectSize: observedEffectSize,
  nObs: controlRevenue.length,
  alpha: 0.05,
});
const requiredSample = tTestPower({
  effectSize: observedEffectSize,
  alpha: 0.05,
  power: 0.9,
});

console.log(
  `Bootstrap revenue uplift vs control: ${upliftBootstrap.estimate.toFixed(2)} | 95% CI [${upliftBootstrap.ci[0].toFixed(2)}, ${upliftBootstrap.ci[1].toFixed(2)}]`
);
console.log(`Observed |Cohen's d|: ${observedEffectSize.toFixed(3)}`);
console.log(`Current power:         ${currentPower.power.toFixed(3)}`);
console.log(`Needed per arm for 90% power: ${requiredSample.nObs}`);

// ============================================================================
// Step 5: Persist scorecards and plots
// ============================================================================
console.log("\n💾 STEP 5: Reports and Artifacts");
console.log("─".repeat(72));

const summaryPayload = variantSummaries.map((summary) => ({
  variant: summary.variant,
  sessions: summary.sessions,
  conversionRate: summary.conversionRate,
  conversionCi: summary.conversionCi,
  revenueMean: summary.revenueMean,
  revenueCi: summary.revenueCi,
  retentionRate: summary.retentionRate,
  retentionCi: summary.retentionCi,
  latencyCi: summary.latencyCi,
}));
await writeFile(
  `${OUTPUT_DIR}/variant-scorecard.json`,
  JSON.stringify(summaryPayload, null, 2),
  "utf-8"
);

const upliftBootstrapForReport = {
  estimate: upliftBootstrap.estimate,
  ci: upliftBootstrap.ci,
  nResamples: upliftBootstrap.samples.length,
};

await writeFile(
  `${OUTPUT_DIR}/decision-report.json`,
  JSON.stringify(
    {
      winner,
      pairwiseInference,
      upliftBootstrap: upliftBootstrapForReport,
      currentPower,
      requiredSample,
    },
    null,
    2
  ),
  "utf-8"
);

const rateFigure = figure({ width: 860, height: 520 });
groupedBar(
  tensor([1, 2, 3]),
  [
    tensor(variantSummaries.map((summary) => summary.conversionRate)),
    tensor(variantSummaries.map((summary) => summary.retentionRate)),
  ],
  {
    colors: ["#2563eb", "#059669"],
    labels: ["conversion rate", "7d retention rate"],
  }
);
legend();
await saveFig(`${OUTPUT_DIR}/variant-rates.svg`, { figure: rateFigure });

const densityFigure = figure({ width: 860, height: 520 });
kdeplot(
  tensor(
    numericValues(controlSessions, (session) => session.orderValue).filter((value) => value > 0)
  ),
  { color: "#1d4ed8", label: "control order values", bw_method: "silverman" }
);
kdeplot(
  tensor(
    numericValues(winnerSessions, (session) => session.orderValue).filter((value) => value > 0)
  ),
  { color: "#f97316", label: `${winner} order values`, bw_method: "silverman" }
);
legend();
await saveFig(`${OUTPUT_DIR}/winner-order-value-density.svg`, {
  figure: densityFigure,
});

const significanceFigure = figure({ width: 860, height: 520 });
groupedBar(
  tensor([1, 2]),
  [tensor(rawRevenuePvalues), tensor(Array.from(revenueCorrection.corrected))],
  {
    colors: ["#7c3aed", "#f43f5e"],
    labels: ["raw revenue p-values", "BH corrected"],
  }
);
axhline(0.05, { color: "#991b1b", linewidth: 2, label: "alpha = 0.05" });
legend();
await saveFig(`${OUTPUT_DIR}/revenue-significance.svg`, {
  figure: significanceFigure,
});

console.log(`Saved variant scorecard:      ${OUTPUT_DIR}/variant-scorecard.json`);
console.log(`Saved decision report:        ${OUTPUT_DIR}/decision-report.json`);
console.log(`Saved grouped rate chart:     ${OUTPUT_DIR}/variant-rates.svg`);
console.log(`Saved order-value density:    ${OUTPUT_DIR}/winner-order-value-density.svg`);
console.log(`Saved significance chart:     ${OUTPUT_DIR}/revenue-significance.svg`);

console.log("\n✅ Experimentation Platform Complete!");
