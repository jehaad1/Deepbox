/**
 * Movie Recommendation Engine
 *
 * Collaborative filtering recommendation system using clustering and similarity.
 *
 * Deepbox Modules Used:
 * - deepbox/ndarray: Tensor and sparse matrix operations
 * - deepbox/ml: KMeans, PCA, KNN
 * - deepbox/stats: Correlation for similarity
 * - deepbox/metrics: Clustering metrics
 * - deepbox/dataframe: Data manipulation
 */

import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { isNumericTypedArray, isTypedArray } from "deepbox/core";
import { DataFrame } from "deepbox/dataframe";
import { silhouetteScore } from "deepbox/metrics";
import { KMeans, PCA } from "deepbox/ml";
import { tensor } from "deepbox/ndarray";
import { Figure } from "deepbox/plot";

// ============================================================================
// Configuration
// ============================================================================

const OUTPUT_DIR = "docs/projects/05-recommendation-engine/output";
const NUM_USERS = 200;
const NUM_MOVIES = 50;
const SPARSITY = 0.7; // 70% of ratings are missing
const NUM_RECOMMENDATIONS = 5;

const expectNumericTypedArray = (
  value: unknown
): Float32Array | Float64Array | Int32Array | Uint8Array => {
  if (!isTypedArray(value) || !isNumericTypedArray(value)) {
    throw new Error("Expected numeric typed array");
  }
  return value;
};

// ============================================================================
// Data Generation
// ============================================================================

/**
 * Generate synthetic movie rating data
 */
function generateRatingData(
  numUsers: number,
  numMovies: number,
  sparsity: number,
  seed = 42
): {
  ratings: number[][];
  userIds: number[];
  movieIds: number[];
  movieGenres: string[];
  movieNames: string[];
} {
  let randomSeed = seed;
  const seededRandom = () => {
    randomSeed = (randomSeed * 1103515245 + 12345) & 0x7fffffff;
    return randomSeed / 0x7fffffff;
  };

  const genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Horror", "Romance", "Thriller"];
  const movieGenres = Array.from(
    { length: numMovies },
    () => genres[Math.floor(seededRandom() * genres.length)]
  );

  const movieNames = Array.from({ length: numMovies }, (_, i) => `Movie ${i + 1}`);

  // Create user preference profiles (latent factors)
  const userPrefs: number[][] = [];
  for (let u = 0; u < numUsers; u++) {
    // Each user has preferences for each genre
    const prefs = genres.map(() => seededRandom() * 2 - 0.5);
    userPrefs.push(prefs);
  }

  // Generate ratings matrix
  const ratings: number[][] = [];
  for (let u = 0; u < numUsers; u++) {
    const userRatings: number[] = [];
    for (let m = 0; m < numMovies; m++) {
      if (seededRandom() < sparsity) {
        userRatings.push(0); // Missing rating
      } else {
        // Rating based on user preference for movie's genre
        const genreIdx = genres.indexOf(movieGenres[m]);
        const basePref = userPrefs[u][genreIdx];
        const rating = Math.round(
          Math.max(1, Math.min(5, 3 + basePref * 2 + (seededRandom() - 0.5)))
        );
        userRatings.push(rating);
      }
    }
    ratings.push(userRatings);
  }

  return {
    ratings,
    userIds: Array.from({ length: numUsers }, (_, i) => i),
    movieIds: Array.from({ length: numMovies }, (_, i) => i),
    movieGenres,
    movieNames,
  };
}

/**
 * Calculate cosine similarity between two vectors
 */
function cosineSimilarity(a: number[], b: number[]): number {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    if (a[i] !== 0 && b[i] !== 0) {
      dotProduct += a[i] * b[i];
    }
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Calculate adjusted cosine similarity (for item-based CF)
 */
function adjustedCosineSimilarity(ratings: number[][], item1: number, item2: number): number {
  const numUsers = ratings.length;
  let sum = 0;
  let norm1 = 0;
  let norm2 = 0;
  let count = 0;

  // Calculate user means
  const userMeans = ratings.map((userRatings) => {
    const nonZero = userRatings.filter((r) => r > 0);
    return nonZero.length > 0 ? nonZero.reduce((a, b) => a + b, 0) / nonZero.length : 0;
  });

  for (let u = 0; u < numUsers; u++) {
    const r1 = ratings[u][item1];
    const r2 = ratings[u][item2];

    if (r1 > 0 && r2 > 0) {
      const adj1 = r1 - userMeans[u];
      const adj2 = r2 - userMeans[u];
      sum += adj1 * adj2;
      norm1 += adj1 * adj1;
      norm2 += adj2 * adj2;
      count++;
    }
  }

  if (count < 2 || norm1 === 0 || norm2 === 0) return 0;
  return sum / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

/**
 * Get top-k similar users
 */
function getTopKSimilarUsers(
  ratings: number[][],
  targetUser: number,
  k: number
): { userId: number; similarity: number }[] {
  const similarities: { userId: number; similarity: number }[] = [];

  for (let u = 0; u < ratings.length; u++) {
    if (u !== targetUser) {
      const sim = cosineSimilarity(ratings[targetUser], ratings[u]);
      if (sim > 0) {
        similarities.push({ userId: u, similarity: sim });
      }
    }
  }

  return similarities.sort((a, b) => b.similarity - a.similarity).slice(0, k);
}

/**
 * Predict rating using user-based collaborative filtering
 */
function predictRating(
  ratings: number[][],
  targetMovie: number,
  similarUsers: { userId: number; similarity: number }[]
): number {
  let weightedSum = 0;
  let simSum = 0;

  for (const { userId, similarity } of similarUsers) {
    const rating = ratings[userId][targetMovie];
    if (rating > 0) {
      weightedSum += similarity * rating;
      simSum += similarity;
    }
  }

  if (simSum === 0) return 0;
  return weightedSum / simSum;
}

// ============================================================================
// Main Execution
// ============================================================================

console.log("═".repeat(70));
console.log("  MOVIE RECOMMENDATION ENGINE");
console.log("  Built with Deepbox - TypeScript Data Science & ML Library");
console.log("═".repeat(70));

// Create output directory
if (!existsSync(OUTPUT_DIR)) {
  mkdirSync(OUTPUT_DIR, { recursive: true });
}

// ============================================================================
// Step 1: Generate Data
// ============================================================================

console.log("\n📊 STEP 1: Generating Rating Data");
console.log("─".repeat(70));

const { ratings, movieGenres, movieNames } = generateRatingData(NUM_USERS, NUM_MOVIES, SPARSITY);

// Calculate statistics
let totalRatings = 0;
let ratingSum = 0;
const ratingCounts = [0, 0, 0, 0, 0]; // counts for ratings 1-5

for (const userRatings of ratings) {
  for (const rating of userRatings) {
    if (rating > 0) {
      totalRatings++;
      ratingSum += rating;
      ratingCounts[rating - 1]++;
    }
  }
}

console.log(`\n✓ Generated rating matrix`);
console.log(`  Users: ${NUM_USERS}`);
console.log(`  Movies: ${NUM_MOVIES}`);
console.log(`  Total Ratings: ${totalRatings}`);
console.log(`  Sparsity: ${((1 - totalRatings / (NUM_USERS * NUM_MOVIES)) * 100).toFixed(1)}%`);
console.log(`  Average Rating: ${(ratingSum / totalRatings).toFixed(2)}`);

console.log(`\nRating Distribution:`);
for (let i = 0; i < 5; i++) {
  const pct = ((ratingCounts[i] / totalRatings) * 100).toFixed(1);
  const bar = "█".repeat(Math.round((ratingCounts[i] / totalRatings) * 30));
  console.log(`  ${i + 1} stars: ${bar} ${pct}%`);
}

// Genre distribution
const genreCounts: { [key: string]: number } = {};
for (const genre of movieGenres) {
  genreCounts[genre] = (genreCounts[genre] || 0) + 1;
}
console.log(`\nMovie Genres:`);
for (const [genre, count] of Object.entries(genreCounts).sort((a, b) => b[1] - a[1])) {
  console.log(`  ${genre}: ${count} movies`);
}

// ============================================================================
// Step 2: User Clustering
// ============================================================================

console.log("\n👥 STEP 2: User Clustering (K-Means)");
console.log("─".repeat(70));

// Create user feature matrix (replace 0s with mean for clustering)
const userFeatures: number[][] = ratings.map((userRatings) => {
  const nonZero = userRatings.filter((r) => r > 0);
  const userMean = nonZero.length > 0 ? nonZero.reduce((a, b) => a + b, 0) / nonZero.length : 3;
  return userRatings.map((r) => (r === 0 ? userMean : r));
});

const userFeaturesTensor = tensor(userFeatures);

// Try different k values
console.log("\nFinding optimal number of clusters...");
const kValues = [3, 4, 5, 6];
const clusterResults: { k: number; inertia: number; silhouette: number }[] = [];

for (const k of kValues) {
  const kmeans = new KMeans({ nClusters: k, randomState: 42 });
  kmeans.fit(userFeaturesTensor);
  const labels = kmeans.predict(userFeaturesTensor);

  let silhouette = 0;
  try {
    silhouette = silhouetteScore(userFeaturesTensor, labels);
  } catch (_e) {
    silhouette = 0;
  }

  clusterResults.push({
    k,
    inertia: kmeans.inertia,
    silhouette: Number(silhouette),
  });

  console.log(
    `  k=${k}: Inertia=${kmeans.inertia.toFixed(2)}, Silhouette=${Number(silhouette).toFixed(3)}`
  );
}

// Select best k based on silhouette score
const bestK = clusterResults.reduce((best, r) => (r.silhouette > best.silhouette ? r : best)).k;
console.log(`\n✓ Best k=${bestK} selected`);

// Final clustering
const kmeans = new KMeans({ nClusters: bestK, randomState: 42 });
kmeans.fit(userFeaturesTensor);
const userLabels = kmeans.predict(userFeaturesTensor);
const labelData = expectNumericTypedArray(userLabels.data);

// Cluster statistics
console.log(`\nCluster Distribution:`);
for (let c = 0; c < bestK; c++) {
  const clusterUsers = Array.from(labelData).filter((l) => l === c).length;
  console.log(
    `  Cluster ${c}: ${clusterUsers} users (${((clusterUsers / NUM_USERS) * 100).toFixed(1)}%)`
  );
}

// ============================================================================
// Step 3: Dimensionality Reduction (PCA)
// ============================================================================

console.log("\n📉 STEP 3: Dimensionality Reduction (PCA)");
console.log("─".repeat(70));

const pca = new PCA({ nComponents: 2 });
pca.fit(userFeaturesTensor);
const userProjected = pca.transform(userFeaturesTensor);

console.log(`\n✓ Reduced ${NUM_MOVIES} features to 2 components`);
const explainedVar = expectNumericTypedArray(pca.explainedVarianceRatio.data);
console.log(`  Component 1 variance: ${(explainedVar[0] * 100).toFixed(1)}%`);
console.log(`  Component 2 variance: ${(explainedVar[1] * 100).toFixed(1)}%`);
console.log(`  Total explained: ${((explainedVar[0] + explainedVar[1]) * 100).toFixed(1)}%`);

// ============================================================================
// Step 4: User-Based Collaborative Filtering
// ============================================================================

console.log("\n🎯 STEP 4: User-Based Collaborative Filtering");
console.log("─".repeat(70));

// Select a target user for demonstration
const targetUser = 0;
console.log(`\nGenerating recommendations for User ${targetUser}...`);

// Find similar users
const similarUsers = getTopKSimilarUsers(ratings, targetUser, 20);
console.log(`\nTop 5 Similar Users:`);
for (const { userId, similarity } of similarUsers.slice(0, 5)) {
  console.log(`  User ${userId}: similarity = ${similarity.toFixed(3)}`);
}

// Find movies the user hasn't rated
const unratedMovies: number[] = [];
for (let m = 0; m < NUM_MOVIES; m++) {
  if (ratings[targetUser][m] === 0) {
    unratedMovies.push(m);
  }
}

// Predict ratings for unrated movies
const predictions: { movieId: number; predictedRating: number }[] = [];
for (const movieId of unratedMovies) {
  const predicted = predictRating(ratings, movieId, similarUsers);
  if (predicted > 0) {
    predictions.push({ movieId, predictedRating: predicted });
  }
}

// Sort by predicted rating
predictions.sort((a, b) => b.predictedRating - a.predictedRating);

console.log(`\nTop ${NUM_RECOMMENDATIONS} Recommendations for User ${targetUser}:`);
const recDF = new DataFrame({
  Rank: predictions.slice(0, NUM_RECOMMENDATIONS).map((_, i) => i + 1),
  Movie: predictions.slice(0, NUM_RECOMMENDATIONS).map((p) => movieNames[p.movieId]),
  Genre: predictions.slice(0, NUM_RECOMMENDATIONS).map((p) => movieGenres[p.movieId]),
  "Predicted Rating": predictions
    .slice(0, NUM_RECOMMENDATIONS)
    .map((p) => p.predictedRating.toFixed(2)),
});
console.log(recDF.toString());

// ============================================================================
// Step 5: Item-Based Similarity
// ============================================================================

console.log("\n🎬 STEP 5: Item-Based Similarity Analysis");
console.log("─".repeat(70));

// Calculate item similarity matrix (sample)
console.log("\nCalculating movie similarities...");
const sampleMovies = [0, 10, 20, 30, 40];
const itemSimilarities: number[][] = [];

for (const i of sampleMovies) {
  const row: number[] = [];
  for (const j of sampleMovies) {
    if (i === j) {
      row.push(1.0);
    } else {
      row.push(adjustedCosineSimilarity(ratings, i, j));
    }
  }
  itemSimilarities.push(row);
}

console.log(`\nItem Similarity Matrix (sample of ${sampleMovies.length} movies):`);
console.log(`        ${sampleMovies.map((m) => `M${m}`.padStart(6)).join("")}`);
for (let i = 0; i < sampleMovies.length; i++) {
  const row = `${`M${sampleMovies[i]}`.padStart(6)} `;
  console.log(row + itemSimilarities[i].map((s) => s.toFixed(2).padStart(6)).join(""));
}

// Find similar movies for a sample movie
const sampleMovie = 0;
const movieSims: { movieId: number; similarity: number }[] = [];
for (let m = 0; m < NUM_MOVIES; m++) {
  if (m !== sampleMovie) {
    const sim = adjustedCosineSimilarity(ratings, sampleMovie, m);
    if (!Number.isNaN(sim) && sim > 0) {
      movieSims.push({ movieId: m, similarity: sim });
    }
  }
}
movieSims.sort((a, b) => b.similarity - a.similarity);

console.log(`\nMovies similar to "${movieNames[sampleMovie]}" (${movieGenres[sampleMovie]}):`);
for (const { movieId, similarity } of movieSims.slice(0, 5)) {
  console.log(`  ${movieNames[movieId]} (${movieGenres[movieId]}): ${similarity.toFixed(3)}`);
}

// ============================================================================
// Step 6: Evaluation
// ============================================================================

console.log("\n📈 STEP 6: Recommendation Evaluation");
console.log("─".repeat(70));

// Simple evaluation: predict known ratings and measure error
let totalError = 0;
let totalPredictions = 0;

for (let u = 0; u < Math.min(50, NUM_USERS); u++) {
  const simUsers = getTopKSimilarUsers(ratings, u, 10);

  for (let m = 0; m < NUM_MOVIES; m++) {
    if (ratings[u][m] > 0) {
      // Temporarily hide rating
      const actualRating = ratings[u][m];
      ratings[u][m] = 0;

      const predicted = predictRating(ratings, m, simUsers);

      // Restore rating
      ratings[u][m] = actualRating;

      if (predicted > 0) {
        totalError += Math.abs(predicted - actualRating);
        totalPredictions++;
      }
    }
  }
}

const mae = totalError / totalPredictions;
console.log(`\nCollaborative Filtering Evaluation (leave-one-out):`);
console.log(`  Predictions made: ${totalPredictions}`);
console.log(`  Mean Absolute Error: ${mae.toFixed(3)}`);
console.log(`  (Lower is better, random guess MAE ≈ 1.5)`);

// ============================================================================
// Step 7: Visualizations
// ============================================================================

console.log("\n📊 STEP 7: Generating Visualizations");
console.log("─".repeat(70));

// User clusters visualization
try {
  const fig = new Figure({ width: 800, height: 600 });
  const ax = fig.addAxes();

  const projData = expectNumericTypedArray(userProjected.data);
  const xCoords: number[] = [];
  const yCoords: number[] = [];
  for (let i = 0; i < NUM_USERS; i++) {
    xCoords.push(projData[i * 2]);
    yCoords.push(projData[i * 2 + 1]);
  }

  ax.scatter(tensor(xCoords), tensor(yCoords), { color: "#2196F3", size: 5 });
  ax.setTitle("User Clusters (PCA Projection)");
  ax.setXLabel("PC1");
  ax.setYLabel("PC2");

  const svg = fig.renderSVG();
  writeFileSync(`${OUTPUT_DIR}/user-clusters.svg`, svg.svg);
  console.log(`  ✓ Saved: ${OUTPUT_DIR}/user-clusters.svg`);
} catch (e) {
  console.log(`  ⚠ Could not generate user clusters plot: ${e}`);
}

// Rating distribution
try {
  const fig = new Figure({ width: 800, height: 400 });
  const ax = fig.addAxes();

  ax.bar(tensor([1, 2, 3, 4, 5]), tensor(ratingCounts), { color: "#FF9800" });
  ax.setTitle("Rating Distribution");
  ax.setXLabel("Rating");
  ax.setYLabel("Count");

  const svg = fig.renderSVG();
  writeFileSync(`${OUTPUT_DIR}/rating-distribution.svg`, svg.svg);
  console.log(`  ✓ Saved: ${OUTPUT_DIR}/rating-distribution.svg`);
} catch (e) {
  console.log(`  ⚠ Could not generate rating distribution: ${e}`);
}

// ============================================================================
// Summary
// ============================================================================

console.log(`\n${"═".repeat(70)}`);
console.log("  RECOMMENDATION ENGINE COMPLETE - SUMMARY");
console.log("═".repeat(70));

console.log("\n📌 Key Findings:\n");
console.log("  1. Data Overview:");
console.log(`     • ${NUM_USERS} users, ${NUM_MOVIES} movies`);
console.log(`     • ${totalRatings} ratings (${((1 - SPARSITY) * 100).toFixed(0)}% density)`);
console.log(`     • Average rating: ${(ratingSum / totalRatings).toFixed(2)}`);

console.log("\n  2. User Clustering:");
console.log(`     • Optimal clusters: ${bestK}`);
console.log(`     • Users show distinct preference patterns`);

console.log("\n  3. Recommendation Quality:");
console.log(`     • MAE: ${mae.toFixed(3)} (baseline ~1.5)`);
console.log(`     • Collaborative filtering captures user preferences`);

console.log("\n📁 Output Files:");
console.log(`   • ${OUTPUT_DIR}/user-clusters.svg`);
console.log(`   • ${OUTPUT_DIR}/rating-distribution.svg`);

console.log(`\n${"═".repeat(70)}`);
console.log("  ✅ Recommendation Engine Complete!");
console.log("═".repeat(70));
