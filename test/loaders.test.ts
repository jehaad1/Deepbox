import { describe, expect, it } from "vitest";
import type { Dataset } from "../src/datasets";
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
} from "../src/datasets";

// ─── Helpers ────────────────────────────────────────────────────────────────

/**
 * Extract a flat Float64Array of all numeric values from a tensor's backing
 * buffer, respecting offset and size.
 */
function extractValues(t: Dataset["data"]): number[] {
  const values: number[] = [];
  for (let i = 0; i < t.size; i++) {
    values.push(Number(t.data[t.offset + i]));
  }
  return values;
}

/**
 * Check that every value in the array is a finite number (no NaN / Infinity).
 */
function allFinite(values: number[]): boolean {
  return values.every((v) => Number.isFinite(v));
}

/**
 * Check that not all values are the same (degenerate data).
 */
function hasVariance(values: number[]): boolean {
  if (values.length === 0) return false;
  const first = values[0];
  return values.some((v) => v !== first);
}

/**
 * Compute Pearson correlation coefficient between two arrays.
 */
function pearsonCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumX2 = 0;
  let sumY2 = 0;
  for (let i = 0; i < n; i++) {
    sumX += x[i];
    sumY += y[i];
    sumXY += x[i] * y[i];
    sumX2 += x[i] * x[i];
    sumY2 += y[i] * y[i];
  }
  const num = n * sumXY - sumX * sumY;
  const den = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
  if (den === 0) return 0;
  return num / den;
}

/**
 * Extract column `col` from a 2D tensor into a number array.
 */
function extractColumn(t: Dataset["data"], col: number): number[] {
  const nCols = t.shape[1];
  const nRows = t.shape[0];
  const values: number[] = [];
  for (let i = 0; i < nRows; i++) {
    values.push(Number(t.data[t.offset + i * nCols + col]));
  }
  return values;
}

// ─── Shape & Metadata Tests ─────────────────────────────────────────────────

describe("Dataset Loaders", () => {
  // ─── Original datasets ──────────────────────────────────────────────

  it("should load Iris dataset", () => {
    const { data, target, featureNames, targetNames, description } = loadIris();
    expect(data.shape).toEqual([150, 4]);
    expect(target.shape).toEqual([150]);
    expect(featureNames.length).toBe(4);
    expect(targetNames?.length).toBe(3);
    expect(description.length).toBeGreaterThan(0);
  });

  it("should load Digits dataset", () => {
    const { data, target, featureNames, targetNames } = loadDigits();
    expect(data.shape).toEqual([1797, 64]);
    expect(target.shape).toEqual([1797]);
    expect(featureNames.length).toBe(64);
    expect(targetNames?.length).toBe(10);
  });

  it("should load Breast Cancer dataset", () => {
    const { data, target, featureNames, targetNames } = loadBreastCancer();
    expect(data.shape).toEqual([569, 30]);
    expect(target.shape).toEqual([569]);
    expect(featureNames.length).toBe(30);
    expect(targetNames?.length).toBe(2);
  });

  it("should load Diabetes dataset", () => {
    const { data, target, featureNames } = loadDiabetes();
    expect(data.shape).toEqual([442, 10]);
    expect(target.shape).toEqual([442]);
    expect(featureNames.length).toBe(10);
  });

  it("should load Linnerud dataset", () => {
    const { data, target, featureNames, targetNames } = loadLinnerud();
    expect(data.shape).toEqual([20, 3]);
    expect(target.shape).toEqual([20, 3]);
    expect(featureNames.length).toBe(3);
    expect(targetNames?.length).toBe(3);
  });

  // ─── Classic Tabular Classification ─────────────────────────────────

  it("should load Flowers Extended dataset", () => {
    const { data, target, featureNames, targetNames, description } = loadFlowersExtended();
    expect(data.shape).toEqual([180, 6]);
    expect(target.shape).toEqual([180]);
    expect(featureNames.length).toBe(6);
    expect(targetNames?.length).toBe(4);
    expect(description.length).toBeGreaterThan(0);
  });

  it("should load Leaf Shapes dataset", () => {
    const { data, target, featureNames, targetNames } = loadLeafShapes();
    expect(data.shape).toEqual([150, 8]);
    expect(target.shape).toEqual([150]);
    expect(featureNames.length).toBe(8);
    expect(targetNames?.length).toBe(5);
  });

  it("should load Fruit Quality dataset", () => {
    const { data, target, featureNames, targetNames } = loadFruitQuality();
    expect(data.shape).toEqual([150, 5]);
    expect(target.shape).toEqual([150]);
    expect(featureNames.length).toBe(5);
    expect(targetNames?.length).toBe(3);
  });

  it("should load Seed Morphology dataset", () => {
    const { data, target, featureNames, targetNames } = loadSeedMorphology();
    expect(data.shape).toEqual([150, 4]);
    expect(target.shape).toEqual([150]);
    expect(featureNames.length).toBe(4);
    expect(targetNames?.length).toBe(3);
  });

  // ─── Educational Non-Linear Classification ──────────────────────────

  it("should load Moons-Multi dataset", () => {
    const { data, target, featureNames, targetNames } = loadMoonsMulti();
    expect(data.shape).toEqual([150, 2]);
    expect(target.shape).toEqual([150]);
    expect(featureNames.length).toBe(2);
    expect(targetNames?.length).toBe(3);
  });

  it("should load Concentric Rings dataset", () => {
    const { data, target, featureNames, targetNames } = loadConcentricRings();
    expect(data.shape).toEqual([150, 2]);
    expect(target.shape).toEqual([150]);
    expect(featureNames.length).toBe(2);
    expect(targetNames?.length).toBe(3);
  });

  it("should load Spiral Arms dataset", () => {
    const { data, target, featureNames, targetNames } = loadSpiralArms();
    expect(data.shape).toEqual([150, 2]);
    expect(target.shape).toEqual([150]);
    expect(featureNames.length).toBe(2);
    expect(targetNames?.length).toBe(3);
  });

  it("should load Gaussian Islands dataset", () => {
    const { data, target, featureNames, targetNames } = loadGaussianIslands();
    expect(data.shape).toEqual([200, 3]);
    expect(target.shape).toEqual([200]);
    expect(featureNames.length).toBe(3);
    expect(targetNames?.length).toBe(4);
  });

  // ─── Small Regression Datasets ──────────────────────────────────────

  it("should load Plant Growth dataset", () => {
    const { data, target, featureNames } = loadPlantGrowth();
    expect(data.shape).toEqual([200, 3]);
    expect(target.shape).toEqual([200]);
    expect(featureNames.length).toBe(3);
  });

  it("should load Housing-Mini dataset", () => {
    const { data, target, featureNames } = loadHousingMini();
    expect(data.shape).toEqual([200, 4]);
    expect(target.shape).toEqual([200]);
    expect(featureNames.length).toBe(4);
  });

  it("should load Energy Efficiency dataset", () => {
    const { data, target, featureNames } = loadEnergyEfficiency();
    expect(data.shape).toEqual([200, 3]);
    expect(target.shape).toEqual([200]);
    expect(featureNames.length).toBe(3);
  });

  it("should load Crop Yield dataset", () => {
    const { data, target, featureNames } = loadCropYield();
    expect(data.shape).toEqual([200, 3]);
    expect(target.shape).toEqual([200]);
    expect(featureNames.length).toBe(3);
  });

  // ─── Clustering-First Datasets ──────────────────────────────────────

  it("should load Customer Segments dataset", () => {
    const { data, target, featureNames, targetNames } = loadCustomerSegments();
    expect(data.shape).toEqual([200, 3]);
    expect(target.shape).toEqual([200]);
    expect(featureNames.length).toBe(3);
    expect(targetNames?.length).toBe(4);
  });

  it("should load Sensor States dataset", () => {
    const { data, target, featureNames, targetNames } = loadSensorStates();
    expect(data.shape).toEqual([180, 6]);
    expect(target.shape).toEqual([180]);
    expect(featureNames.length).toBe(6);
    expect(targetNames?.length).toBe(3);
  });

  // ─── Discrete / Integer-Heavy Datasets ──────────────────────────────

  it("should load Student Performance dataset", () => {
    const { data, target, featureNames, targetNames } = loadStudentPerformance();
    expect(data.shape).toEqual([150, 3]);
    expect(target.shape).toEqual([150]);
    expect(featureNames.length).toBe(3);
    expect(targetNames?.length).toBe(3);
  });

  it("should load Traffic Conditions dataset", () => {
    const { data, target, featureNames, targetNames } = loadTrafficConditions();
    expect(data.shape).toEqual([150, 3]);
    expect(target.shape).toEqual([150]);
    expect(featureNames.length).toBe(3);
    expect(targetNames?.length).toBe(3);
  });

  // ─── Multi-Output & Structured Targets ──────────────────────────────

  it("should load Fitness Scores dataset", () => {
    const { data, target, featureNames, targetNames } = loadFitnessScores();
    expect(data.shape).toEqual([100, 3]);
    expect(target.shape).toEqual([100, 3]);
    expect(featureNames.length).toBe(3);
    expect(targetNames?.length).toBe(3);
  });

  it("should load Weather Outcomes dataset", () => {
    const { data, target, featureNames, targetNames } = loadWeatherOutcomes();
    expect(data.shape).toEqual([150, 3]);
    expect(target.shape).toEqual([150, 2]);
    expect(featureNames.length).toBe(3);
    expect(targetNames?.length).toBe(2);
  });

  // ─── Benchmark / Sanity-Check ───────────────────────────────────────

  it("should load Perfectly Separable dataset", () => {
    const { data, target, featureNames, targetNames } = loadPerfectlySeparable();
    expect(data.shape).toEqual([100, 4]);
    expect(target.shape).toEqual([100]);
    expect(featureNames.length).toBe(4);
    expect(targetNames?.length).toBe(2);
  });

  // ─── Dtype tests ────────────────────────────────────────────────────

  it("should use int32 dtype for classification targets", () => {
    expect(loadIris().target.dtype).toBe("int32");
    expect(loadDigits().target.dtype).toBe("int32");
    expect(loadBreastCancer().target.dtype).toBe("int32");
    expect(loadFlowersExtended().target.dtype).toBe("int32");
    expect(loadLeafShapes().target.dtype).toBe("int32");
    expect(loadFruitQuality().target.dtype).toBe("int32");
    expect(loadSeedMorphology().target.dtype).toBe("int32");
    expect(loadMoonsMulti().target.dtype).toBe("int32");
    expect(loadConcentricRings().target.dtype).toBe("int32");
    expect(loadSpiralArms().target.dtype).toBe("int32");
    expect(loadGaussianIslands().target.dtype).toBe("int32");
    expect(loadCustomerSegments().target.dtype).toBe("int32");
    expect(loadSensorStates().target.dtype).toBe("int32");
    expect(loadStudentPerformance().target.dtype).toBe("int32");
    expect(loadTrafficConditions().target.dtype).toBe("int32");
    expect(loadPerfectlySeparable().target.dtype).toBe("int32");
  });

  it("should use default float dtype for regression targets", () => {
    expect(loadDiabetes().target.dtype).toBe("float32");
    expect(loadLinnerud().target.dtype).toBe("float32");
    expect(loadPlantGrowth().target.dtype).toBe("float32");
    expect(loadHousingMini().target.dtype).toBe("float32");
    expect(loadEnergyEfficiency().target.dtype).toBe("float32");
    expect(loadCropYield().target.dtype).toBe("float32");
    expect(loadFitnessScores().target.dtype).toBe("float32");
    expect(loadWeatherOutcomes().target.dtype).toBe("float32");
  });

  // ─── Multi-output target shape tests ────────────────────────────────

  it("should return 2D target for Linnerud multi-output regression", () => {
    const { target } = loadLinnerud();
    expect(target.ndim).toBe(2);
    expect(target.shape).toEqual([20, 3]);
  });

  it("should return 2D target for Fitness Scores multi-output regression", () => {
    const { target } = loadFitnessScores();
    expect(target.ndim).toBe(2);
    expect(target.shape).toEqual([100, 3]);
  });

  it("should return 2D target for Weather Outcomes multi-output regression", () => {
    const { target } = loadWeatherOutcomes();
    expect(target.ndim).toBe(2);
    expect(target.shape).toEqual([150, 2]);
  });

  // ─── Metadata consistency tests ─────────────────────────────────────

  it("should have featureNames matching data columns for every dataset", () => {
    const loaders = [
      loadIris,
      loadDigits,
      loadBreastCancer,
      loadDiabetes,
      loadLinnerud,
      loadFlowersExtended,
      loadLeafShapes,
      loadFruitQuality,
      loadSeedMorphology,
      loadMoonsMulti,
      loadConcentricRings,
      loadSpiralArms,
      loadGaussianIslands,
      loadPlantGrowth,
      loadHousingMini,
      loadEnergyEfficiency,
      loadCropYield,
      loadCustomerSegments,
      loadSensorStates,
      loadStudentPerformance,
      loadTrafficConditions,
      loadFitnessScores,
      loadWeatherOutcomes,
      loadPerfectlySeparable,
    ];
    for (const loader of loaders) {
      const ds = loader();
      expect(ds.featureNames.length).toBe(ds.data.shape[1]);
      expect(ds.featureNames.every((name) => name.length > 0)).toBe(true);
      expect(ds.description.length).toBeGreaterThan(0);
    }
  });

  it("should have targetNames matching number of classes for classification datasets", () => {
    const classificationSets: Array<{ loader: () => Dataset; numClasses: number }> = [
      { loader: loadIris, numClasses: 3 },
      { loader: loadDigits, numClasses: 10 },
      { loader: loadBreastCancer, numClasses: 2 },
      { loader: loadFlowersExtended, numClasses: 4 },
      { loader: loadLeafShapes, numClasses: 5 },
      { loader: loadFruitQuality, numClasses: 3 },
      { loader: loadSeedMorphology, numClasses: 3 },
      { loader: loadMoonsMulti, numClasses: 3 },
      { loader: loadConcentricRings, numClasses: 3 },
      { loader: loadSpiralArms, numClasses: 3 },
      { loader: loadGaussianIslands, numClasses: 4 },
      { loader: loadCustomerSegments, numClasses: 4 },
      { loader: loadSensorStates, numClasses: 3 },
      { loader: loadStudentPerformance, numClasses: 3 },
      { loader: loadTrafficConditions, numClasses: 3 },
      { loader: loadPerfectlySeparable, numClasses: 2 },
    ];
    for (const { loader, numClasses } of classificationSets) {
      const ds = loader();
      expect(ds.targetNames?.length).toBe(numClasses);
    }
  });

  it("should have targetNames matching target columns for multi-output datasets", () => {
    const linnerud = loadLinnerud();
    expect(linnerud.targetNames?.length).toBe(linnerud.target.shape[1]);

    const fitness = loadFitnessScores();
    expect(fitness.targetNames?.length).toBe(fitness.target.shape[1]);

    const weather = loadWeatherOutcomes();
    expect(weather.targetNames?.length).toBe(weather.target.shape[1]);
  });

  // ─── Caching tests ─────────────────────────────────────────────────

  it("returns cached datasets on subsequent calls", () => {
    const loaders = [
      loadIris,
      loadDigits,
      loadBreastCancer,
      loadDiabetes,
      loadLinnerud,
      loadFlowersExtended,
      loadLeafShapes,
      loadSpiralArms,
      loadPerfectlySeparable,
    ];
    for (const loader of loaders) {
      const ds1 = loader();
      const ds2 = loader();
      expect(ds1.data.shape).toEqual(ds2.data.shape);
      expect(ds1.target.data[0]).toBe(ds2.target.data[0]);
    }
  });

  // ═══════════════════════════════════════════════════════════════════════
  // Data Integrity Tests
  // ═══════════════════════════════════════════════════════════════════════

  describe("Data Integrity", () => {
    it("should contain no NaN or Infinity in any dataset", () => {
      const loaders = [
        loadIris,
        loadDigits,
        loadBreastCancer,
        loadDiabetes,
        loadLinnerud,
        loadFlowersExtended,
        loadLeafShapes,
        loadFruitQuality,
        loadSeedMorphology,
        loadMoonsMulti,
        loadConcentricRings,
        loadSpiralArms,
        loadGaussianIslands,
        loadPlantGrowth,
        loadHousingMini,
        loadEnergyEfficiency,
        loadCropYield,
        loadCustomerSegments,
        loadSensorStates,
        loadStudentPerformance,
        loadTrafficConditions,
        loadFitnessScores,
        loadWeatherOutcomes,
        loadPerfectlySeparable,
      ];
      for (const loader of loaders) {
        const ds = loader();
        expect(allFinite(extractValues(ds.data))).toBe(true);
        expect(allFinite(extractValues(ds.target))).toBe(true);
      }
    });

    it("should have non-degenerate data (variance > 0 in every feature)", () => {
      const loaders = [
        loadIris,
        loadFlowersExtended,
        loadLeafShapes,
        loadFruitQuality,
        loadSeedMorphology,
        loadPlantGrowth,
        loadHousingMini,
        loadEnergyEfficiency,
        loadCropYield,
        loadCustomerSegments,
        loadSensorStates,
        loadStudentPerformance,
        loadTrafficConditions,
        loadPerfectlySeparable,
      ];
      for (const loader of loaders) {
        const ds = loader();
        for (let col = 0; col < ds.data.shape[1]; col++) {
          expect(hasVariance(extractColumn(ds.data, col))).toBe(true);
        }
      }
    });

    it("should produce deterministic results across calls", () => {
      const loaders = [
        loadIris,
        loadMoonsMulti,
        loadConcentricRings,
        loadSpiralArms,
        loadGaussianIslands,
        loadPlantGrowth,
        loadCustomerSegments,
        loadPerfectlySeparable,
      ];
      for (const loader of loaders) {
        const ds1 = loader();
        const ds2 = loader();
        const vals1 = extractValues(ds1.data);
        const vals2 = extractValues(ds2.data);
        expect(vals1).toEqual(vals2);
      }
    });
  });

  // ═══════════════════════════════════════════════════════════════════════
  // Class Balance & Label Validity Tests
  // ═══════════════════════════════════════════════════════════════════════

  describe("Class Balance & Label Validity", () => {
    it("should have valid class labels in range [0, nClasses) for classification datasets", () => {
      const classificationSets: Array<{ loader: () => Dataset; nClasses: number }> = [
        { loader: loadIris, nClasses: 3 },
        { loader: loadDigits, nClasses: 10 },
        { loader: loadBreastCancer, nClasses: 2 },
        { loader: loadFlowersExtended, nClasses: 4 },
        { loader: loadLeafShapes, nClasses: 5 },
        { loader: loadFruitQuality, nClasses: 3 },
        { loader: loadSeedMorphology, nClasses: 3 },
        { loader: loadMoonsMulti, nClasses: 3 },
        { loader: loadConcentricRings, nClasses: 3 },
        { loader: loadSpiralArms, nClasses: 3 },
        { loader: loadGaussianIslands, nClasses: 4 },
        { loader: loadCustomerSegments, nClasses: 4 },
        { loader: loadSensorStates, nClasses: 3 },
        { loader: loadStudentPerformance, nClasses: 3 },
        { loader: loadTrafficConditions, nClasses: 3 },
        { loader: loadPerfectlySeparable, nClasses: 2 },
      ];
      for (const { loader, nClasses } of classificationSets) {
        const ds = loader();
        const targetVals = extractValues(ds.target);
        const uniqueClasses = new Set(targetVals);

        // Every label is an integer in [0, nClasses)
        for (const val of targetVals) {
          expect(val).toBe(Math.floor(val));
          expect(val).toBeGreaterThanOrEqual(0);
          expect(val).toBeLessThan(nClasses);
        }

        // All classes are represented
        expect(uniqueClasses.size).toBe(nClasses);
      }
    });

    it("should have balanced classes for core classification datasets", () => {
      // These datasets are designed with equal samples per class
      const balanced: Array<{
        loader: () => Dataset;
        samplesPerClass: number;
        nClasses: number;
      }> = [
        { loader: loadIris, samplesPerClass: 50, nClasses: 3 },
        { loader: loadFlowersExtended, samplesPerClass: 45, nClasses: 4 },
        { loader: loadLeafShapes, samplesPerClass: 30, nClasses: 5 },
        { loader: loadFruitQuality, samplesPerClass: 50, nClasses: 3 },
        { loader: loadSeedMorphology, samplesPerClass: 50, nClasses: 3 },
        { loader: loadMoonsMulti, samplesPerClass: 50, nClasses: 3 },
        { loader: loadConcentricRings, samplesPerClass: 50, nClasses: 3 },
        { loader: loadSpiralArms, samplesPerClass: 50, nClasses: 3 },
        { loader: loadGaussianIslands, samplesPerClass: 50, nClasses: 4 },
        { loader: loadCustomerSegments, samplesPerClass: 50, nClasses: 4 },
        { loader: loadStudentPerformance, samplesPerClass: 50, nClasses: 3 },
        { loader: loadTrafficConditions, samplesPerClass: 50, nClasses: 3 },
        { loader: loadPerfectlySeparable, samplesPerClass: 50, nClasses: 2 },
      ];
      for (const { loader, samplesPerClass, nClasses } of balanced) {
        const ds = loader();
        const targetVals = extractValues(ds.target);
        for (let c = 0; c < nClasses; c++) {
          const count = targetVals.filter((v) => v === c).length;
          expect(count).toBe(samplesPerClass);
        }
      }
    });
  });

  // ═══════════════════════════════════════════════════════════════════════
  // Regression Feature-Target Relationship Tests
  // ═══════════════════════════════════════════════════════════════════════

  describe("Regression Feature-Target Relationships", () => {
    it("Plant Growth: target should correlate positively with sunlight", () => {
      const ds = loadPlantGrowth();
      const sunlight = extractColumn(ds.data, 0);
      const target = extractValues(ds.target);
      const r = pearsonCorrelation(sunlight, target);
      expect(r).toBeGreaterThan(0.3);
    });

    it("Plant Growth: target should correlate positively with soil quality", () => {
      const ds = loadPlantGrowth();
      const soilQuality = extractColumn(ds.data, 2);
      const target = extractValues(ds.target);
      const r = pearsonCorrelation(soilQuality, target);
      expect(r).toBeGreaterThan(0.3);
    });

    it("Housing-Mini: target should correlate positively with size", () => {
      const ds = loadHousingMini();
      const size = extractColumn(ds.data, 0);
      const target = extractValues(ds.target);
      const r = pearsonCorrelation(size, target);
      expect(r).toBeGreaterThan(0.5);
    });

    it("Housing-Mini: target should correlate positively with rooms", () => {
      const ds = loadHousingMini();
      const rooms = extractColumn(ds.data, 1);
      const target = extractValues(ds.target);
      const r = pearsonCorrelation(rooms, target);
      expect(r).toBeGreaterThan(0.1);
    });

    it("Energy Efficiency: target should correlate negatively with insulation", () => {
      const ds = loadEnergyEfficiency();
      const insulation = extractColumn(ds.data, 0);
      const target = extractValues(ds.target);
      const r = pearsonCorrelation(insulation, target);
      expect(r).toBeLessThan(-0.3);
    });

    it("Energy Efficiency: target should correlate positively with window area", () => {
      const ds = loadEnergyEfficiency();
      const windowArea = extractColumn(ds.data, 1);
      const target = extractValues(ds.target);
      const r = pearsonCorrelation(windowArea, target);
      expect(r).toBeGreaterThan(0.3);
    });

    it("Crop Yield: target should correlate positively with rainfall", () => {
      const ds = loadCropYield();
      const rainfall = extractColumn(ds.data, 0);
      const target = extractValues(ds.target);
      const r = pearsonCorrelation(rainfall, target);
      expect(r).toBeGreaterThan(0.3);
    });

    it("Crop Yield: target should correlate positively with fertilizer", () => {
      const ds = loadCropYield();
      const fertilizer = extractColumn(ds.data, 1);
      const target = extractValues(ds.target);
      const r = pearsonCorrelation(fertilizer, target);
      expect(r).toBeGreaterThan(0.3);
    });

    it("regression targets should have reasonable range (not collapsed)", () => {
      const regressionSets = [
        loadPlantGrowth,
        loadHousingMini,
        loadEnergyEfficiency,
        loadCropYield,
      ];
      for (const loader of regressionSets) {
        const ds = loader();
        const targetVals = extractValues(ds.target);
        const min = Math.min(...targetVals);
        const max = Math.max(...targetVals);
        expect(max - min).toBeGreaterThan(1);
      }
    });
  });

  // ═══════════════════════════════════════════════════════════════════════
  // Non-Linear Structure Tests
  // ═══════════════════════════════════════════════════════════════════════

  describe("Non-Linear Dataset Structure", () => {
    it("Concentric Rings: inner ring should have smaller radius than outer ring", () => {
      const ds = loadConcentricRings();
      const x = extractColumn(ds.data, 0);
      const y = extractColumn(ds.data, 1);
      const labels = extractValues(ds.target);

      // Compute mean radii per class
      const radii = [0, 0, 0];
      const counts = [0, 0, 0];
      for (let i = 0; i < labels.length; i++) {
        const c = labels[i];
        radii[c] += Math.sqrt(x[i] * x[i] + y[i] * y[i]);
        counts[c]++;
      }
      const meanRadii = radii.map((r, i) => r / counts[i]);

      // Inner < middle < outer
      expect(meanRadii[0]).toBeLessThan(meanRadii[1]);
      expect(meanRadii[1]).toBeLessThan(meanRadii[2]);
    });

    it("Gaussian Islands: clusters should be well-separated in 3D", () => {
      const ds = loadGaussianIslands();
      const labels = extractValues(ds.target);

      // Compute centroid per class
      const centroids: number[][] = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ];
      const counts = [0, 0, 0, 0];
      for (let i = 0; i < labels.length; i++) {
        const c = labels[i];
        for (let d = 0; d < 3; d++) {
          centroids[c][d] += Number(ds.data.data[ds.data.offset + i * 3 + d]);
        }
        counts[c]++;
      }
      for (let c = 0; c < 4; c++) {
        for (let d = 0; d < 3; d++) {
          centroids[c][d] /= counts[c];
        }
      }

      // Every pair of centroids should be separated by at least 4.0
      for (let a = 0; a < 4; a++) {
        for (let b = a + 1; b < 4; b++) {
          const dist = Math.sqrt(
            centroids[a].reduce((sum, v, d) => sum + (v - centroids[b][d]) ** 2, 0)
          );
          expect(dist).toBeGreaterThan(4.0);
        }
      }
    });

    it("Perfectly Separable: classes should be linearly separable with margin", () => {
      const ds = loadPerfectlySeparable();
      const labels = extractValues(ds.target);

      // Compute mean of each feature per class
      const nFeatures = ds.data.shape[1];
      const mean0 = new Array(nFeatures).fill(0);
      const mean1 = new Array(nFeatures).fill(0);
      let count0 = 0;
      let count1 = 0;

      for (let i = 0; i < labels.length; i++) {
        for (let f = 0; f < nFeatures; f++) {
          const val = Number(ds.data.data[ds.data.offset + i * nFeatures + f]);
          if (labels[i] === 0) {
            mean0[f] += val;
          } else {
            mean1[f] += val;
          }
        }
        if (labels[i] === 0) count0++;
        else count1++;
      }

      for (let f = 0; f < nFeatures; f++) {
        mean0[f] /= count0;
        mean1[f] /= count1;
      }

      // Centroids should differ in every dimension
      for (let f = 0; f < nFeatures; f++) {
        expect(Math.abs(mean1[f] - mean0[f])).toBeGreaterThan(1.0);
      }
    });

    it("Spiral Arms: points should spiral outward from origin", () => {
      const ds = loadSpiralArms();
      const x = extractColumn(ds.data, 0);
      const y = extractColumn(ds.data, 1);
      const labels = extractValues(ds.target);

      // For arm 0, later samples (higher index within arm) should be further from origin
      const arm0Radii: number[] = [];
      for (let i = 0; i < labels.length; i++) {
        if (labels[i] === 0) {
          arm0Radii.push(Math.sqrt(x[i] * x[i] + y[i] * y[i]));
        }
      }

      // Check that mean of second half radii > mean of first half
      const half = Math.floor(arm0Radii.length / 2);
      const firstHalfMean = arm0Radii.slice(0, half).reduce((a, b) => a + b, 0) / half;
      const secondHalfMean =
        arm0Radii.slice(half).reduce((a, b) => a + b, 0) / (arm0Radii.length - half);
      expect(secondHalfMean).toBeGreaterThan(firstHalfMean);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════
  // Multi-Output Target Validation
  // ═══════════════════════════════════════════════════════════════════════

  describe("Multi-Output Targets", () => {
    it("Fitness Scores: all targets should be positive", () => {
      const ds = loadFitnessScores();
      const targetVals = extractValues(ds.target);
      for (const v of targetVals) {
        expect(v).toBeGreaterThan(0);
      }
    });

    it("Fitness Scores: each target should correlate with exercise duration", () => {
      const ds = loadFitnessScores();
      const duration = extractColumn(ds.data, 0);
      const nTargets = ds.target.shape[1];
      for (let t = 0; t < nTargets; t++) {
        const targetCol: number[] = [];
        for (let i = 0; i < ds.target.shape[0]; i++) {
          targetCol.push(Number(ds.target.data[ds.target.offset + i * nTargets + t]));
        }
        const r = pearsonCorrelation(duration, targetCol);
        expect(r).toBeGreaterThan(0.3);
      }
    });

    it("Weather Outcomes: rain probability should be in [0, 1]", () => {
      const ds = loadWeatherOutcomes();
      const nTargets = ds.target.shape[1];
      for (let i = 0; i < ds.target.shape[0]; i++) {
        const rainProb = Number(ds.target.data[ds.target.offset + i * nTargets]);
        expect(rainProb).toBeGreaterThanOrEqual(0);
        expect(rainProb).toBeLessThanOrEqual(1);
      }
    });

    it("Weather Outcomes: wind speed should be non-negative", () => {
      const ds = loadWeatherOutcomes();
      const nTargets = ds.target.shape[1];
      for (let i = 0; i < ds.target.shape[0]; i++) {
        const windSpeed = Number(ds.target.data[ds.target.offset + i * nTargets + 1]);
        expect(windSpeed).toBeGreaterThanOrEqual(0);
      }
    });

    it("Weather Outcomes: rain probability should correlate positively with humidity", () => {
      const ds = loadWeatherOutcomes();
      const humidity = extractColumn(ds.data, 0);
      const nTargets = ds.target.shape[1];
      const rainProbs: number[] = [];
      for (let i = 0; i < ds.target.shape[0]; i++) {
        rainProbs.push(Number(ds.target.data[ds.target.offset + i * nTargets]));
      }
      const r = pearsonCorrelation(humidity, rainProbs);
      expect(r).toBeGreaterThan(0.3);
    });
  });

  // ═══════════════════════════════════════════════════════════════════════
  // Integer-Heavy Dataset Tests
  // ═══════════════════════════════════════════════════════════════════════

  describe("Integer-Heavy Features", () => {
    it("Student Performance: all features should be non-negative integers", () => {
      const ds = loadStudentPerformance();
      for (let col = 0; col < ds.data.shape[1]; col++) {
        const vals = extractColumn(ds.data, col);
        for (const v of vals) {
          expect(v).toBe(Math.floor(v));
          expect(v).toBeGreaterThanOrEqual(0);
        }
      }
    });

    it("Traffic Conditions: all features should be non-negative integers", () => {
      const ds = loadTrafficConditions();
      for (let col = 0; col < ds.data.shape[1]; col++) {
        const vals = extractColumn(ds.data, col);
        for (const v of vals) {
          expect(v).toBe(Math.floor(v));
          expect(v).toBeGreaterThanOrEqual(0);
        }
      }
    });

    it("Digits: pixel values should be integers in [0, 15]", () => {
      const ds = loadDigits();
      const allVals = extractValues(ds.data);
      for (const v of allVals) {
        expect(v).toBe(Math.floor(v));
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(15);
      }
    });
  });

  // ═══════════════════════════════════════════════════════════════════════
  // Value Range Tests
  // ═══════════════════════════════════════════════════════════════════════

  describe("Value Ranges", () => {
    it("Iris: sepal/petal measurements should be in physically reasonable ranges", () => {
      const ds = loadIris();
      // Sepal length: ~4.3-7.9 cm in real data
      const sepalLength = extractColumn(ds.data, 0);
      expect(Math.min(...sepalLength)).toBeGreaterThan(3);
      expect(Math.max(...sepalLength)).toBeLessThan(10);
    });

    it("Breast Cancer: features should be positive", () => {
      const ds = loadBreastCancer();
      const allVals = extractValues(ds.data);
      for (const v of allVals) {
        expect(v).toBeGreaterThan(0);
      }
    });

    it("Housing-Mini: size should be positive, rooms >= 1", () => {
      const ds = loadHousingMini();
      const size = extractColumn(ds.data, 0);
      const rooms = extractColumn(ds.data, 1);
      for (const v of size) {
        expect(v).toBeGreaterThan(0);
      }
      for (const v of rooms) {
        expect(v).toBeGreaterThanOrEqual(1);
      }
    });

    it("Energy Efficiency: orientation should be in [0, 360)", () => {
      const ds = loadEnergyEfficiency();
      const orientation = extractColumn(ds.data, 2);
      for (const v of orientation) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(360);
      }
    });

    it("Crop Yield: temperature should be in reasonable range", () => {
      const ds = loadCropYield();
      const temp = extractColumn(ds.data, 2);
      for (const v of temp) {
        expect(v).toBeGreaterThanOrEqual(10);
        expect(v).toBeLessThanOrEqual(40);
      }
    });

    it("Fruit Quality: weight, sugar, pH should be positive", () => {
      const ds = loadFruitQuality();
      const weight = extractColumn(ds.data, 0);
      const sugar = extractColumn(ds.data, 1);
      const ph = extractColumn(ds.data, 2);
      for (const v of weight) expect(v).toBeGreaterThan(0);
      for (const v of sugar) expect(v).toBeGreaterThan(0);
      for (const v of ph) expect(v).toBeGreaterThan(0);
    });

    it("Sensor States: voltage should be in reasonable range", () => {
      const ds = loadSensorStates();
      const voltage = extractColumn(ds.data, 3);
      for (const v of voltage) {
        expect(v).toBeGreaterThan(190);
        expect(v).toBeLessThan(260);
      }
    });

    it("Customer Segments: age should be positive, spending score in [0, 100]", () => {
      const ds = loadCustomerSegments();
      const age = extractColumn(ds.data, 0);
      const spending = extractColumn(ds.data, 2);
      for (const v of age) expect(v).toBeGreaterThan(10);
      for (const v of spending) {
        // Gaussian noise can push slightly beyond nominal [0,100]
        expect(v).toBeGreaterThan(-10);
        expect(v).toBeLessThan(110);
      }
    });
  });

  // ═══════════════════════════════════════════════════════════════════════
  // Clustering Quality Tests
  // ═══════════════════════════════════════════════════════════════════════

  describe("Clustering Dataset Quality", () => {
    it("Customer Segments: clusters should have distinct income profiles", () => {
      const ds = loadCustomerSegments();
      const income = extractColumn(ds.data, 1);
      const labels = extractValues(ds.target);

      const meanIncome = [0, 0, 0, 0];
      const counts = [0, 0, 0, 0];
      for (let i = 0; i < labels.length; i++) {
        meanIncome[labels[i]] += income[i];
        counts[labels[i]]++;
      }
      for (let c = 0; c < 4; c++) {
        meanIncome[c] /= counts[c];
      }

      // At least two clusters should have significantly different mean incomes
      const incomes = meanIncome.sort((a, b) => a - b);
      expect(incomes[3] - incomes[0]).toBeGreaterThan(20);
    });

    it("Sensor States: normal mode should have less variance than fault mode", () => {
      const ds = loadSensorStates();
      const labels = extractValues(ds.target);

      // Check vibration variance (feature 4) per mode
      const vibrations: number[][] = [[], [], []];
      for (let i = 0; i < labels.length; i++) {
        vibrations[labels[i]].push(Number(ds.data.data[ds.data.offset + i * 6 + 4]));
      }

      const variance = (arr: number[]): number => {
        const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
        return arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length;
      };

      // Fault mode (class 2) should have higher vibration variance than normal (class 0)
      expect(variance(vibrations[2])).toBeGreaterThan(variance(vibrations[0]));
    });
  });
});
