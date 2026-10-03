/**
 * Example 35: Advanced Clustering
 *
 * MiniBatchKMeans, AgglomerativeClustering, GaussianMixture, SpectralClustering,
 * OPTICS, MeanShift, Birch and AffinityPropagation, scored with the silhouette
 * score and the adjusted Rand index (ARI) on synthetic blobs and moons.
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
console.log("\nPart 1: MiniBatchKMeans");
console.log("-".repeat(60));

// MiniBatchKMeans updates the centers from small random batches instead of the full dataset
const mbkmeans = new MiniBatchKMeans({
  nClusters: 4,
  batchSize: 50,
  maxIter: 100,
  randomState: 42,
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
console.log("\nPart 2: Agglomerative Clustering");
console.log("-".repeat(60));

// Agglomerative clustering starts with one cluster per sample and repeatedly merges the closest pair
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
  console.log(`  ${linkage.padEnd(10)} linkage, Silhouette: ${sil.toFixed(4)}`);
}

// ============================================================================
// Part 3: Gaussian Mixture Model
// ============================================================================
console.log("\nPart 3: Gaussian Mixture Model");
console.log("-".repeat(60));

// A Gaussian mixture models the data as a weighted sum of Gaussian distributions
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
console.log("\nPart 4: Spectral Clustering");
console.log("-".repeat(60));

// Spectral clustering uses the eigenvectors of a similarity matrix, which suits non-convex shapes
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
console.log("\nPart 5: OPTICS");
console.log("-".repeat(60));

// OPTICS is density-based and does not need a fixed neighborhood radius (eps)
const optics = new OPTICS({
  minSamples: 5,
});
optics.fit(XBlobs);

const optLabels = optics.labels;
const optAri = adjustedRandScore(yBlobs, optLabels);

console.log("OPTICS (minSamples=5):");
console.log(`  Adjusted Rand Index: ${optAri.toFixed(4)}`);
console.log(`  Reachability values: ${optics.reachability.length}`);

// ============================================================================
// Part 6: MeanShift
// ============================================================================
console.log("\nPart 6: MeanShift");
console.log("-".repeat(60));

// MeanShift moves points uphill to the modes of the density estimate. Each mode is one cluster.
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
console.log("\nPart 7: Birch");
console.log("-".repeat(60));

// Birch summarizes the data in a clustering-feature tree, so it needs one pass over the samples
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
console.log("\nPart 8: Affinity Propagation");
console.log("-".repeat(60));

// Affinity propagation passes messages between samples and chooses the number of clusters itself
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
console.log("\nClustering Comparison on Blobs Dataset");
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
console.log("\nKey Takeaways");
console.log("-".repeat(60));
console.log("• MiniBatchKMeans: KMeans on mini-batches, cheaper for large datasets");
console.log("• Agglomerative: hierarchical clustering with a choice of linkage");
console.log("• Gaussian Mixture: probabilistic assignments instead of hard ones");
console.log("• Spectral: graph-based, handles non-convex shapes such as moons");
console.log("• OPTICS: density-based, no eps to choose up front");
console.log("• MeanShift: finds the number of clusters from the density modes");
console.log("• Birch: one pass over the data, low memory");
console.log("• Affinity Propagation: finds the number of clusters by message passing");

console.log("\nAdvanced Clustering Example Complete!");
console.log("=".repeat(60));
