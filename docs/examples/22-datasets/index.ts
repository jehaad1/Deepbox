/**
 * Example 22: Built-in Datasets
 *
 * Explore Deepbox's built-in datasets for quick experimentation.
 * Perfect for learning and testing ML algorithms.
 */

import {
  loadBreastCancer,
  loadConcentricRings,
  loadCropYield,
  loadCustomerSegments,
  loadDiabetes,
  loadDigits,
  loadEnergyEfficiency,
  loadFitnessScores,
  loadFlowersExtended,
  loadFruitQuality,
  loadGaussianIslands,
  loadHousingMini,
  loadIris,
  loadLeafShapes,
  loadLinnerud,
  loadMoonsMulti,
  loadPerfectlySeparable,
  loadPlantGrowth,
  loadSeedMorphology,
  loadSensorStates,
  loadSpiralArms,
  loadStudentPerformance,
  loadTrafficConditions,
  loadWeatherOutcomes,
  makeBlobs,
  makeCircles,
  makeClassification,
  makeGaussianQuantiles,
  makeMoons,
  makeRegression,
} from "deepbox/datasets";

console.log("=== Built-in Datasets ===\n");

// ─── Classic Reference Datasets ─────────────────────────────────────────────

console.log("--- Classic Reference Datasets ---\n");

console.log("1. Iris Dataset:");
console.log("-".repeat(50));
const iris = loadIris();
console.log(`Samples: ${iris.data.shape[0]}`);
console.log(`Features: ${iris.data.shape[1]}`);
console.log(`Classes: ${iris.targetNames?.join(", ") || "N/A"}`);
console.log(`Features: ${iris.featureNames?.join(", ") || "N/A"}\n`);

console.log("2. Digits Dataset:");
console.log("-".repeat(50));
const digits = loadDigits();
console.log(`Samples: ${digits.data.shape[0]}`);
console.log(`Features: ${digits.data.shape[1]} (8x8 images flattened)`);
console.log(`Classes: 10 (digits 0-9)\n`);

console.log("3. Breast Cancer Dataset:");
console.log("-".repeat(50));
const cancer = loadBreastCancer();
console.log(`Samples: ${cancer.data.shape[0]}`);
console.log(`Features: ${cancer.data.shape[1]}`);
console.log(`Classes: ${cancer.targetNames?.join(", ") || "N/A"}\n`);

console.log("4. Diabetes Dataset (Regression):");
console.log("-".repeat(50));
const diabetes = loadDiabetes();
console.log(`Samples: ${diabetes.data.shape[0]}`);
console.log(`Features: ${diabetes.data.shape[1]}`);
console.log(`Task: Regression (predict disease progression)\n`);

console.log("5. Linnerud Dataset (Multi-Output):");
console.log("-".repeat(50));
const linnerud = loadLinnerud();
console.log(`Samples: ${linnerud.data.shape[0]}`);
console.log(`Features: ${linnerud.data.shape[1]}`);
console.log(`Targets: ${linnerud.target.shape[1]} (multi-output regression)\n`);

// ─── Tabular Classification ─────────────────────────────────────────────────

console.log("--- Tabular Classification Datasets ---\n");

console.log("6. Flowers Extended (4-class Iris variant):");
console.log("-".repeat(50));
const flowers = loadFlowersExtended();
console.log(`Samples: ${flowers.data.shape[0]}, Features: ${flowers.data.shape[1]}`);
console.log(`Classes: ${flowers.targetNames?.join(", ") || "N/A"}\n`);

console.log("7. Leaf Shapes (5-class morphology):");
console.log("-".repeat(50));
const leaves = loadLeafShapes();
console.log(`Samples: ${leaves.data.shape[0]}, Features: ${leaves.data.shape[1]}`);
console.log(`Classes: ${leaves.targetNames?.join(", ") || "N/A"}\n`);

console.log("8. Fruit Quality:");
console.log("-".repeat(50));
const fruit = loadFruitQuality();
console.log(`Samples: ${fruit.data.shape[0]}, Features: ${fruit.data.shape[1]}`);
console.log(`Classes: ${fruit.targetNames?.join(", ") || "N/A"}\n`);

console.log("9. Seed Morphology:");
console.log("-".repeat(50));
const seeds = loadSeedMorphology();
console.log(`Samples: ${seeds.data.shape[0]}, Features: ${seeds.data.shape[1]}`);
console.log(`Classes: ${seeds.targetNames?.join(", ") || "N/A"}\n`);

// ─── Non-Linear Classification ──────────────────────────────────────────────

console.log("--- Non-Linear Classification Datasets ---\n");

console.log("10. Moons-Multi (3 interleaving crescents):");
console.log("-".repeat(50));
const moons = loadMoonsMulti();
console.log(`Samples: ${moons.data.shape[0]}, Features: ${moons.data.shape[1]}\n`);

console.log("11. Concentric Rings:");
console.log("-".repeat(50));
const rings = loadConcentricRings();
console.log(`Samples: ${rings.data.shape[0]}, Features: ${rings.data.shape[1]}\n`);

console.log("12. Spiral Arms:");
console.log("-".repeat(50));
const spirals = loadSpiralArms();
console.log(`Samples: ${spirals.data.shape[0]}, Features: ${spirals.data.shape[1]}\n`);

console.log("13. Gaussian Islands (3D clusters):");
console.log("-".repeat(50));
const islands = loadGaussianIslands();
console.log(`Samples: ${islands.data.shape[0]}, Features: ${islands.data.shape[1]}\n`);

// ─── Regression Datasets ────────────────────────────────────────────────────

console.log("--- Regression Datasets ---\n");

console.log("14. Plant Growth:");
console.log("-".repeat(50));
const plant = loadPlantGrowth();
console.log(`Samples: ${plant.data.shape[0]}, Features: ${plant.data.shape[1]}`);
console.log(`Features: ${plant.featureNames.join(", ")}\n`);

console.log("15. Housing-Mini:");
console.log("-".repeat(50));
const housing = loadHousingMini();
console.log(`Samples: ${housing.data.shape[0]}, Features: ${housing.data.shape[1]}`);
console.log(`Features: ${housing.featureNames.join(", ")}\n`);

console.log("16. Energy Efficiency:");
console.log("-".repeat(50));
const energy = loadEnergyEfficiency();
console.log(`Samples: ${energy.data.shape[0]}, Features: ${energy.data.shape[1]}\n`);

console.log("17. Crop Yield:");
console.log("-".repeat(50));
const crop = loadCropYield();
console.log(`Samples: ${crop.data.shape[0]}, Features: ${crop.data.shape[1]}\n`);

// ─── Clustering Datasets ────────────────────────────────────────────────────

console.log("--- Clustering Datasets ---\n");

console.log("18. Customer Segments:");
console.log("-".repeat(50));
const customers = loadCustomerSegments();
console.log(`Samples: ${customers.data.shape[0]}, Features: ${customers.data.shape[1]}`);
console.log(`Clusters: ${customers.targetNames?.join(", ") || "N/A"}\n`);

console.log("19. Sensor States:");
console.log("-".repeat(50));
const sensors = loadSensorStates();
console.log(`Samples: ${sensors.data.shape[0]}, Features: ${sensors.data.shape[1]}`);
console.log(`Modes: ${sensors.targetNames?.join(", ") || "N/A"}\n`);

// ─── Integer-Heavy Datasets ─────────────────────────────────────────────────

console.log("--- Integer-Heavy Datasets ---\n");

console.log("20. Student Performance:");
console.log("-".repeat(50));
const students = loadStudentPerformance();
console.log(`Samples: ${students.data.shape[0]}, Features: ${students.data.shape[1]}\n`);

console.log("21. Traffic Conditions:");
console.log("-".repeat(50));
const traffic = loadTrafficConditions();
console.log(`Samples: ${traffic.data.shape[0]}, Features: ${traffic.data.shape[1]}\n`);

// ─── Multi-Output Datasets ──────────────────────────────────────────────────

console.log("--- Multi-Output Datasets ---\n");

console.log("22. Fitness Scores (3 targets):");
console.log("-".repeat(50));
const fitness = loadFitnessScores();
console.log(`Samples: ${fitness.data.shape[0]}, Features: ${fitness.data.shape[1]}`);
console.log(`Targets: ${fitness.target.shape[1]} (${fitness.targetNames?.join(", ") || "N/A"})\n`);

console.log("23. Weather Outcomes (2 targets):");
console.log("-".repeat(50));
const weather = loadWeatherOutcomes();
console.log(`Samples: ${weather.data.shape[0]}, Features: ${weather.data.shape[1]}`);
console.log(`Targets: ${weather.target.shape[1]} (${weather.targetNames?.join(", ") || "N/A"})\n`);

// ─── Benchmark / Sanity-Check ───────────────────────────────────────────────

console.log("--- Benchmark / Sanity-Check ---\n");

console.log("24. Perfectly Separable:");
console.log("-".repeat(50));
const perfect = loadPerfectlySeparable();
console.log(`Samples: ${perfect.data.shape[0]}, Features: ${perfect.data.shape[1]}`);
console.log(`Classes: ${perfect.targetNames?.join(", ") || "N/A"}\n`);

// ─── Synthetic Dataset Generators ───────────────────────────────────────────

console.log("=== Synthetic Dataset Generators ===\n");

console.log("25. Make Classification:");
console.log("-".repeat(50));
const classData = makeClassification({
  nSamples: 100,
  nFeatures: 4,
  nClasses: 2,
  randomState: 42,
});
const X_class = classData[0];
console.log(`Generated ${X_class.shape[0]} samples with ${X_class.shape[1]} features\n`);

console.log("26. Make Regression:");
console.log("-".repeat(50));
const regData = makeRegression({
  nSamples: 100,
  nFeatures: 3,
  noise: 0.1,
  randomState: 42,
});
const X_reg = regData[0];
console.log(`Generated ${X_reg.shape[0]} samples with ${X_reg.shape[1]} features\n`);

console.log("27. Make Blobs:");
console.log("-".repeat(50));
const blobsData = makeBlobs({
  nSamples: 150,
  nFeatures: 2,
  centers: 3,
  randomState: 42,
});
const X_blobs = blobsData[0];
console.log(`Generated ${X_blobs.shape[0]} samples in ${3} clusters\n`);

console.log("28. Make Moons:");
console.log("-".repeat(50));
const moonsData = makeMoons({
  nSamples: 100,
  noise: 0.1,
  randomState: 42,
});
const X_moons = moonsData[0];
console.log(`Generated ${X_moons.shape[0]} samples (2 interleaving half circles)\n`);

console.log("29. Make Circles:");
console.log("-".repeat(50));
const circlesData = makeCircles({
  nSamples: 100,
  noise: 0.05,
  randomState: 42,
});
const X_circles = circlesData[0];
console.log(`Generated ${X_circles.shape[0]} samples (concentric circles)\n`);

console.log("30. Make Gaussian Quantiles:");
console.log("-".repeat(50));
const gaussData = makeGaussianQuantiles({
  nSamples: 100,
  nFeatures: 2,
  nClasses: 3,
  randomState: 42,
});
const X_gauss = gaussData[0];
console.log(`Generated ${X_gauss.shape[0]} samples in ${3} quantile-based classes\n`);

console.log("✓ All 24 built-in datasets + 6 synthetic generators explored!");
