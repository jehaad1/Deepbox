/**
 * Example 35: Advanced Clustering
 *
 * New in v1.0.0: Agglomerative, GaussianMixture, SpectralClustering,
 * OPTICS, MiniBatchKMeans, MeanShift, Birch, and AffinityPropagation.
 */

import { makeBlobs, makeMoons } from "deepbox/datasets";
import { adjustedRandScore, silhouetteScore } from "deepbox/metrics";
import {
  AffinityPropagation,
  AgglomerativeClustering,
  Birch,
  GaussianMixture,
  KMeans,
  MeanShift,
  MiniBatchKMeans,
  OPTICS,
  SpectralClustering,
} from "deepbox/ml";

console.log("=".repeat(60));
console.log("Example 35: Advanced Clustering");
console.log("=".repeat(60));

// ============================================================================
// Generate synthetic datasets
// ============================================================================

// Well-separated blobs for standard clustering
const [XBlobs, yBlobs] = makeBlobs({
  nSamples: 200,
  centers: 4,
  clusterStd: 0.8,
  randomState: 42,
});

// Non-convex moons for advanced methods
const [XMoons, yMoons] = makeMoons({
  nSamples: 200,
  noise: 0.1,
  randomState: 42,
});

console.log(`Blobs dataset: ${XBlobs.shape[0]} samples, ${XBlobs.shape[1]} features, 4 clusters`);
console.log(`Moons dataset: ${XMoons.shape[0]} samples, ${XMoons.shape[1]} features, 2 clusters`);

// ============================================================================
// Part 1: MiniBatchKMeans
// ============================================================================
console.log("\n⚡ Part 1: MiniBatchKMeans");
console.log("-".repeat(60));

// MiniBatchKMeans is a faster variant of KMeans that uses mini-batches
const mbkmeans = new MiniBatchKMeans({
  nClusters: 4,
  batchSize: 50,
  maxIter: 100,
});
mbkmeans.fit(XBlobs);

const mbkLabels = mbkmeans.predict(XBlobs);
const mbkSil = silhouetteScore(XBlobs, mbkLabels);
const mbkAri = adjustedRandScore(yBlobs, mbkLabels);

console.log("MiniBatchKMeans (k=4, batchSize=50):");
console.log(`  Silhouette Score:     ${mbkSil.toFixed(4)}`);
console.log(`  Adjusted Rand Index:  ${mbkAri.toFixed(4)}`);
console.log(`  Cluster centers shape: ${mbkmeans.clusterCenters.shape}`);

// Compare with standard KMeans
const kmeans = new KMeans({ nClusters: 4, randomState: 42 });
kmeans.fit(XBlobs);
const kmLabels = kmeans.predict(XBlobs);
const kmSil = silhouetteScore(XBlobs, kmLabels);
console.log(`  KMeans Silhouette (for comparison): ${kmSil.toFixed(4)}`);

// ============================================================================
// Part 2: Agglomerative Clustering
// ============================================================================
console.log("\n🌲 Part 2: Agglomerative Clustering");
console.log("-".repeat(60));

// Agglomerative clustering builds a hierarchy by merging closest clusters
const aggWard = new AgglomerativeClustering({
  nClusters: 4,
  linkage: "ward",
});
aggWard.fit(XBlobs);

const aggLabels = aggWard.labels;
const aggSil = silhouetteScore(XBlobs, aggLabels);
const aggAri = adjustedRandScore(yBlobs, aggLabels);

console.log("Agglomerative (Ward linkage, k=4):");
console.log(`  Silhouette Score:    ${aggSil.toFixed(4)}`);
console.log(`  Adjusted Rand Index: ${aggAri.toFixed(4)}`);

// Try different linkages
for (const linkage of ["complete", "average", "single"] as const) {
  const agg = new AgglomerativeClustering({ nClusters: 4, linkage });
  agg.fit(XBlobs);
  const sil = silhouetteScore(XBlobs, agg.labels);
  console.log(`  ${linkage.padEnd(10)} linkage — Silhouette: ${sil.toFixed(4)}`);
}

// ============================================================================
// Part 3: Gaussian Mixture Model
// ============================================================================
console.log("\n📊 Part 3: Gaussian Mixture Model");
console.log("-".repeat(60));

// GMM models data as a mixture of Gaussian distributions
const gmm = new GaussianMixture({
  nComponents: 4,
  maxIter: 100,
});
gmm.fit(XBlobs);

const gmmLabels = gmm.predict(XBlobs);
const gmmSil = silhouetteScore(XBlobs, gmmLabels);
const gmmAri = adjustedRandScore(yBlobs, gmmLabels);

console.log("Gaussian Mixture (4 components):");
console.log(`  Silhouette Score:    ${gmmSil.toFixed(4)}`);
console.log(`  Adjusted Rand Index: ${gmmAri.toFixed(4)}`);

// ============================================================================
// Part 4: Spectral Clustering
// ============================================================================
console.log("\n🌀 Part 4: Spectral Clustering");
console.log("-".repeat(60));

// Spectral Clustering uses eigenvalues of similarity matrix — good for non-convex shapes
const spectral = new SpectralClustering({
  nClusters: 2,
  affinity: "rbf",
  gamma: 10,
});
spectral.fit(XMoons);

const specLabels = spectral.labels;
const specSil = silhouetteScore(XMoons, specLabels);
const specAri = adjustedRandScore(yMoons, specLabels);

console.log("Spectral Clustering on Moons (k=2, RBF kernel):");
console.log(`  Silhouette Score:    ${specSil.toFixed(4)}`);
console.log(`  Adjusted Rand Index: ${specAri.toFixed(4)}`);

// ============================================================================
// Part 5: OPTICS
// ============================================================================
console.log("\n🔬 Part 5: OPTICS");
console.log("-".repeat(60));

// OPTICS: density-based clustering that doesn't require specifying eps
const optics = new OPTICS({
  minSamples: 5,
});
optics.fit(XBlobs);

const optLabels = optics.labels;
const optAri = adjustedRandScore(yBlobs, optLabels);

console.log("OPTICS (minSamples=5):");
console.log(`  Adjusted Rand Index: ${optAri.toFixed(4)}`);
console.log(`  Reachability values: ${optics.reachability.shape}`);

// ============================================================================
// Part 6: MeanShift
// ============================================================================
console.log("\n🎯 Part 6: MeanShift");
console.log("-".repeat(60));

// MeanShift finds clusters by seeking high-density regions
const meanshift = new MeanShift({
  bandwidth: "auto",
});
meanshift.fit(XBlobs);

const msLabels = meanshift.labels;
const msSil = silhouetteScore(XBlobs, msLabels);

console.log("MeanShift (auto bandwidth):");
console.log(`  Silhouette Score:    ${msSil.toFixed(4)}`);
console.log(`  Cluster centers shape: ${meanshift.clusterCenters.shape}`);

// ============================================================================
// Part 7: Birch
// ============================================================================
console.log("\n🌿 Part 7: Birch");
console.log("-".repeat(60));

// Birch: scalable clustering using CF-tree data structure
const birch = new Birch({
  nClusters: 4,
  threshold: 0.5,
  branchingFactor: 50,
});
birch.fit(XBlobs);

const birchLabels = birch.labels;
const birchSil = silhouetteScore(XBlobs, birchLabels);
const birchAri = adjustedRandScore(yBlobs, birchLabels);

console.log("Birch (k=4, threshold=0.5):");
console.log(`  Silhouette Score:    ${birchSil.toFixed(4)}`);
console.log(`  Adjusted Rand Index: ${birchAri.toFixed(4)}`);

// ============================================================================
// Part 8: Affinity Propagation
// ============================================================================
console.log("\n💬 Part 8: Affinity Propagation");
console.log("-".repeat(60));

// Affinity Propagation: message-passing algorithm, auto-determines number of clusters
const ap = new AffinityPropagation({
  damping: 0.9,
  maxIter: 200,
});
ap.fit(XBlobs);

const apLabels = ap.labels;
const apSil = silhouetteScore(XBlobs, apLabels);

console.log("Affinity Propagation (damping=0.9):");
console.log(`  Silhouette Score:    ${apSil.toFixed(4)}`);
console.log(`  Cluster centers shape: ${ap.clusterCenters.shape}`);

// ============================================================================
// Summary Comparison
// ============================================================================
console.log("\n📋 Clustering Comparison on Blobs Dataset");
console.log("-".repeat(60));

console.log("┌─────────────────────────┬────────────┬──────────┐");
console.log("│ Algorithm               │ Silhouette │ ARI      │");
console.log("├─────────────────────────┼────────────┼──────────┤");

const comparisons = [
  { name: "KMeans", sil: kmSil, ari: adjustedRandScore(yBlobs, kmLabels) },
  { name: "MiniBatchKMeans", sil: mbkSil, ari: mbkAri },
  { name: "Agglomerative (Ward)", sil: aggSil, ari: aggAri },
  { name: "Gaussian Mixture", sil: gmmSil, ari: gmmAri },
  { name: "MeanShift", sil: msSil, ari: adjustedRandScore(yBlobs, msLabels) },
  { name: "Birch", sil: birchSil, ari: birchAri },
  { name: "Affinity Propagation", sil: apSil, ari: adjustedRandScore(yBlobs, apLabels) },
];

for (const c of comparisons) {
  const name = c.name.padEnd(23);
  const sil = c.sil.toFixed(4).padStart(10);
  const ari = c.ari.toFixed(4).padStart(8);
  console.log(`│ ${name} │ ${sil} │ ${ari} │`);
}
console.log("└─────────────────────────┴────────────┴──────────┘");

// ============================================================================
// Summary
// ============================================================================
console.log("\n💡 Key Takeaways");
console.log("-".repeat(60));
console.log("• MiniBatchKMeans: fast KMeans for large datasets using mini-batches");
console.log("• Agglomerative: hierarchical clustering with different linkage criteria");
console.log("• Gaussian Mixture: soft clustering with probabilistic assignments");
console.log("• Spectral: graph-based, excels at non-convex cluster shapes");
console.log("• OPTICS: density-based, no need to specify epsilon parameter");
console.log("• MeanShift: auto-discovers number of clusters via density modes");
console.log("• Birch: memory-efficient, scalable to very large datasets");
console.log("• Affinity Propagation: auto-determines cluster count via message passing");

console.log("\n✅ Advanced Clustering Example Complete!");
console.log("=".repeat(60));
