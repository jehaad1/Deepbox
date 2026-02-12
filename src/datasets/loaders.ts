// ---- loaders.ts ----
import { type Tensor, tensor } from "../ndarray";
import { createRng, normal01 } from "./utils";

export type Dataset = {
  data: Tensor;
  target: Tensor;
  featureNames: string[];
  targetNames?: string[];
  description: string;
};

const SYNTHETIC_NOTE =
  "Synthetic (deterministic) dataset inspired by a common ML benchmark. " +
  "Values are generated, not the original reference dataset.";

// ─────────────────────────────────────────────────────────────────────────────
// 1. Iris
// ─────────────────────────────────────────────────────────────────────────────

let __iris: { data: number[][]; target: number[] } | undefined;

function getIrisData() {
  if (__iris !== undefined) return __iris;
  const rng = createRng(1337);

  const data: number[][] = [];
  const target: number[] = [];

  for (let i = 0; i < 50; i++) {
    data.push([5.0 + rng() * 0.8, 3.4 + rng() * 0.4, 1.4 + rng() * 0.3, 0.2 + rng() * 0.2]);
    target.push(0);
  }
  for (let i = 0; i < 50; i++) {
    data.push([5.9 + rng() * 0.8, 2.8 + rng() * 0.4, 4.2 + rng() * 0.5, 1.3 + rng() * 0.3]);
    target.push(1);
  }
  for (let i = 0; i < 50; i++) {
    data.push([6.5 + rng() * 0.8, 3.0 + rng() * 0.4, 5.5 + rng() * 0.5, 2.0 + rng() * 0.4]);
    target.push(2);
  }

  __iris = { data, target };
  return __iris;
}

/**
 * Load the synthetic Iris dataset.
 *
 * 150 samples, 4 features, 3 classes (setosa, versicolor, virginica).
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[150, 4]` and `target` shape `[150]` (int32).
 */
export function loadIris(): Dataset {
  const { data, target } = getIrisData();
  return {
    data: tensor(data),
    target: tensor(target, { dtype: "int32" }),
    featureNames: [
      "sepal length (cm)",
      "sepal width (cm)",
      "petal length (cm)",
      "petal width (cm)",
    ],
    targetNames: ["setosa", "versicolor", "virginica"],
    description: `${SYNTHETIC_NOTE} 150 samples, 4 features, 3 classes.`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Digits
// ─────────────────────────────────────────────────────────────────────────────

let __digits: { data: number[][]; target: number[] } | undefined;

function getDigitsData() {
  if (__digits !== undefined) return __digits;
  const rng = createRng(7);

  const data: number[][] = [];
  const target: number[] = [];

  for (let digit = 0; digit < 10; digit++) {
    const samplesPerDigit = digit === 9 ? 177 : 180;
    for (let i = 0; i < samplesPerDigit; i++) {
      const sample: number[] = [];
      for (let j = 0; j < 64; j++) {
        sample.push(Math.floor(rng() * 16));
      }
      data.push(sample);
      target.push(digit);
    }
  }

  __digits = { data, target };
  return __digits;
}

/**
 * Load the synthetic Digits dataset.
 *
 * 1797 samples, 64 features (8×8 pixel values 0–15), 10 classes (digits 0–9).
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[1797, 64]` and `target` shape `[1797]` (int32).
 */
export function loadDigits(): Dataset {
  const { data, target } = getDigitsData();
  return {
    data: tensor(data),
    target: tensor(target, { dtype: "int32" }),
    featureNames: Array.from({ length: 64 }, (_, i) => `pixel_${i}`),
    targetNames: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    description: `${SYNTHETIC_NOTE} 1797 samples, 64 features, 10 classes.`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Breast Cancer
// ─────────────────────────────────────────────────────────────────────────────

let __breastCancer: { data: number[][]; target: number[] } | undefined;

function getBreastCancerData() {
  if (__breastCancer !== undefined) return __breastCancer;
  const rng = createRng(99);

  const data: number[][] = [];
  const target: number[] = [];

  for (let i = 0; i < 212; i++) {
    const sample: number[] = [];
    for (let j = 0; j < 30; j++) {
      sample.push(15 + rng() * 10);
    }
    data.push(sample);
    target.push(0);
  }
  for (let i = 0; i < 357; i++) {
    const sample: number[] = [];
    for (let j = 0; j < 30; j++) {
      sample.push(10 + rng() * 8);
    }
    data.push(sample);
    target.push(1);
  }

  __breastCancer = { data, target };
  return __breastCancer;
}

/**
 * Load the synthetic Breast Cancer dataset.
 *
 * 569 samples, 30 features, 2 classes (malignant, benign).
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[569, 30]` and `target` shape `[569]` (int32).
 */
export function loadBreastCancer(): Dataset {
  const { data, target } = getBreastCancerData();
  return {
    data: tensor(data),
    target: tensor(target, { dtype: "int32" }),
    featureNames: [
      "mean radius",
      "mean texture",
      "mean perimeter",
      "mean area",
      "mean smoothness",
      "mean compactness",
      "mean concavity",
      "mean concave points",
      "mean symmetry",
      "mean fractal dimension",
      "radius error",
      "texture error",
      "perimeter error",
      "area error",
      "smoothness error",
      "compactness error",
      "concavity error",
      "concave points error",
      "symmetry error",
      "fractal dimension error",
      "worst radius",
      "worst texture",
      "worst perimeter",
      "worst area",
      "worst smoothness",
      "worst compactness",
      "worst concavity",
      "worst concave points",
      "worst symmetry",
      "worst fractal dimension",
    ],
    targetNames: ["malignant", "benign"],
    description: `${SYNTHETIC_NOTE} 569 samples, 30 features, 2 classes.`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Diabetes
// ─────────────────────────────────────────────────────────────────────────────

let __diabetes: { data: number[][]; target: number[] } | undefined;

function getDiabetesData() {
  if (__diabetes !== undefined) return __diabetes;
  const rng = createRng(123);

  const data: number[][] = [];
  const target: number[] = [];

  for (let i = 0; i < 442; i++) {
    const sample: number[] = [];
    for (let j = 0; j < 10; j++) {
      sample.push(-0.1 + rng() * 0.2);
    }
    data.push(sample);
    target.push(50 + rng() * 300);
  }

  __diabetes = { data, target };
  return __diabetes;
}

/**
 * Load the synthetic Diabetes regression dataset.
 *
 * 442 samples, 10 features, continuous target.
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[442, 10]` and `target` shape `[442]`.
 */
export function loadDiabetes(): Dataset {
  const { data, target } = getDiabetesData();
  return {
    data: tensor(data),
    target: tensor(target),
    featureNames: ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"],
    description: `${SYNTHETIC_NOTE} 442 samples, 10 features (regression).`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Linnerud
// ─────────────────────────────────────────────────────────────────────────────

let __linnerud: { data: number[][]; target: number[][] } | undefined;

function getLinnerudData() {
  if (__linnerud !== undefined) return __linnerud;
  const rng = createRng(555);

  const data: number[][] = [];
  const target: number[][] = [];

  for (let i = 0; i < 20; i++) {
    data.push([
      5 + Math.floor(rng() * 15),
      100 + Math.floor(rng() * 100),
      50 + Math.floor(rng() * 200),
    ]);
    target.push([170 + rng() * 30, 60 + rng() * 20, 50 + rng() * 20]);
  }

  __linnerud = { data, target };
  return __linnerud;
}

/**
 * Load the synthetic Linnerud multi-output regression dataset.
 *
 * 20 samples, 3 exercise features, 3 physiological targets.
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[20, 3]` and `target` shape `[20, 3]`.
 */
export function loadLinnerud(): Dataset {
  const { data, target } = getLinnerudData();
  return {
    data: tensor(data),
    target: tensor(target),
    featureNames: ["Chins", "Situps", "Jumps"],
    targetNames: ["Weight", "Waist", "Pulse"],
    description: `${SYNTHETIC_NOTE} 20 samples, 3 exercise features, 3 physiological targets.`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Flowers Extended
// ─────────────────────────────────────────────────────────────────────────────

let __flowersExtended: { data: number[][]; target: number[] } | undefined;

function getFlowersExtendedData() {
  if (__flowersExtended !== undefined) return __flowersExtended;
  const rng = createRng(3001);
  const data: number[][] = [];
  const target: number[] = [];

  // Class 0: setosa
  for (let i = 0; i < 45; i++) {
    data.push([
      4.5 + rng() * 1.0,
      3.0 + rng() * 0.8,
      1.0 + rng() * 0.8,
      0.1 + rng() * 0.4,
      2.0 + rng() * 1.5,
      1.5 + rng() * 1.0,
    ]);
    target.push(0);
  }
  // Class 1: versicolor
  for (let i = 0; i < 45; i++) {
    data.push([
      5.5 + rng() * 1.5,
      2.5 + rng() * 0.8,
      3.5 + rng() * 1.5,
      1.0 + rng() * 0.8,
      4.0 + rng() * 2.0,
      2.5 + rng() * 1.0,
    ]);
    target.push(1);
  }
  // Class 2: virginica
  for (let i = 0; i < 45; i++) {
    data.push([
      6.0 + rng() * 1.9,
      2.8 + rng() * 0.8,
      4.5 + rng() * 2.0,
      1.5 + rng() * 1.0,
      5.0 + rng() * 2.0,
      3.0 + rng() * 1.5,
    ]);
    target.push(2);
  }
  // Class 3: chrysantha
  for (let i = 0; i < 45; i++) {
    data.push([
      5.0 + rng() * 1.5,
      2.2 + rng() * 0.8,
      2.5 + rng() * 1.5,
      0.8 + rng() * 0.6,
      6.0 + rng() * 2.5,
      2.0 + rng() * 1.0,
    ]);
    target.push(3);
  }

  __flowersExtended = { data, target };
  return __flowersExtended;
}

/**
 * Load the synthetic Flowers Extended classification dataset.
 *
 * 180 samples, 6 features, 4 species.
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[180, 6]` and `target` shape `[180]` (int32).
 */
export function loadFlowersExtended(): Dataset {
  const { data, target } = getFlowersExtendedData();
  return {
    data: tensor(data),
    target: tensor(target, { dtype: "int32" }),
    featureNames: [
      "sepal length (cm)",
      "sepal width (cm)",
      "petal length (cm)",
      "petal width (cm)",
      "color intensity",
      "stem thickness (mm)",
    ],
    targetNames: ["setosa", "versicolor", "virginica", "chrysantha"],
    description: `${SYNTHETIC_NOTE} 180 samples, 6 features, 4 species.`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Leaf Shapes
// ─────────────────────────────────────────────────────────────────────────────

let __leafShapes: { data: number[][]; target: number[] } | undefined;

function getLeafShapesData() {
  if (__leafShapes !== undefined) return __leafShapes;
  const rng = createRng(3002);
  const data: number[][] = [];
  const target: number[] = [];

  // Class 0: maple
  for (let i = 0; i < 30; i++) {
    data.push([
      15 + rng() * 15,
      25 + rng() * 20,
      0.8 + rng() * 0.4,
      0.3 + rng() * 0.3,
      0.3 + rng() * 0.2,
      0.5 + rng() * 0.3,
      0.7 + rng() * 0.3,
      0.6 + rng() * 0.2,
    ]);
    target.push(0);
  }
  // Class 1: oak
  for (let i = 0; i < 30; i++) {
    data.push([
      20 + rng() * 20,
      30 + rng() * 20,
      1.0 + rng() * 0.5,
      0.2 + rng() * 0.3,
      0.4 + rng() * 0.2,
      0.3 + rng() * 0.3,
      0.8 + rng() * 0.4,
      0.7 + rng() * 0.2,
    ]);
    target.push(1);
  }
  // Class 2: birch
  for (let i = 0; i < 30; i++) {
    data.push([
      8 + rng() * 10,
      15 + rng() * 15,
      1.2 + rng() * 0.6,
      0.1 + rng() * 0.2,
      0.5 + rng() * 0.2,
      0.1 + rng() * 0.2,
      1.0 + rng() * 0.5,
      0.8 + rng() * 0.15,
    ]);
    target.push(2);
  }
  // Class 3: willow
  for (let i = 0; i < 30; i++) {
    data.push([
      5 + rng() * 7,
      20 + rng() * 15,
      3.0 + rng() * 2.0,
      0.05 + rng() * 0.15,
      0.2 + rng() * 0.2,
      0.05 + rng() * 0.1,
      2.0 + rng() * 1.5,
      0.85 + rng() * 0.13,
    ]);
    target.push(3);
  }
  // Class 4: ginkgo
  for (let i = 0; i < 30; i++) {
    data.push([
      10 + rng() * 12,
      18 + rng() * 14,
      0.9 + rng() * 0.4,
      0.15 + rng() * 0.2,
      0.5 + rng() * 0.2,
      0.15 + rng() * 0.2,
      0.8 + rng() * 0.3,
      0.75 + rng() * 0.15,
    ]);
    target.push(4);
  }

  __leafShapes = { data, target };
  return __leafShapes;
}

/**
 * Load the synthetic Leaf Shapes classification dataset.
 *
 * 150 samples, 8 geometric features, 5 plant species.
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[150, 8]` and `target` shape `[150]` (int32).
 */
export function loadLeafShapes(): Dataset {
  const { data, target } = getLeafShapesData();
  return {
    data: tensor(data),
    target: tensor(target, { dtype: "int32" }),
    featureNames: [
      "area (cm²)",
      "perimeter (cm)",
      "aspect ratio",
      "curvature",
      "compactness",
      "lobedness",
      "elongation",
      "solidity",
    ],
    targetNames: ["maple", "oak", "birch", "willow", "ginkgo"],
    description: `${SYNTHETIC_NOTE} 150 samples, 8 geometric features, 5 plant species.`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Fruit Quality
// ─────────────────────────────────────────────────────────────────────────────

let __fruitQuality: { data: number[][]; target: number[] } | undefined;

function getFruitQualityData() {
  if (__fruitQuality !== undefined) return __fruitQuality;
  const rng = createRng(3003);
  const data: number[][] = [];
  const target: number[] = [];

  // Class 0: apple
  for (let i = 0; i < 50; i++) {
    data.push([150 + rng() * 70, 11 + rng() * 3, 3.2 + rng() * 0.6, 6 + rng() * 3, 5 + rng() * 3]);
    target.push(0);
  }
  // Class 1: orange
  for (let i = 0; i < 50; i++) {
    data.push([130 + rng() * 70, 9 + rng() * 3, 3.0 + rng() * 0.5, 3 + rng() * 3, 6 + rng() * 3]);
    target.push(1);
  }
  // Class 2: banana
  for (let i = 0; i < 50; i++) {
    data.push([100 + rng() * 50, 14 + rng() * 6, 4.5 + rng() * 1.0, 1 + rng() * 3, 4 + rng() * 3]);
    target.push(2);
  }

  __fruitQuality = { data, target };
  return __fruitQuality;
}

/**
 * Load the synthetic Fruit Quality classification dataset.
 *
 * 150 samples, 5 features, 3 fruit classes.
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[150, 5]` and `target` shape `[150]` (int32).
 */
export function loadFruitQuality(): Dataset {
  const { data, target } = getFruitQualityData();
  return {
    data: tensor(data),
    target: tensor(target, { dtype: "int32" }),
    featureNames: [
      "weight (g)",
      "sugar content (Brix)",
      "acidity (pH)",
      "firmness (N)",
      "color score",
    ],
    targetNames: ["apple", "orange", "banana"],
    description: `${SYNTHETIC_NOTE} 150 samples, 5 features, 3 fruit classes.`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Seed Morphology
// ─────────────────────────────────────────────────────────────────────────────

let __seedMorphology: { data: number[][]; target: number[] } | undefined;

function getSeedMorphologyData() {
  if (__seedMorphology !== undefined) return __seedMorphology;
  const rng = createRng(3005);
  const data: number[][] = [];
  const target: number[] = [];

  // Class 0: wheat
  for (let i = 0; i < 50; i++) {
    data.push([5.5 + rng() * 2.0, 2.8 + rng() * 1.0, 0.75 + rng() * 0.17, 1.3 + rng() * 0.2]);
    target.push(0);
  }
  // Class 1: rice
  for (let i = 0; i < 50; i++) {
    data.push([6.0 + rng() * 3.0, 1.8 + rng() * 0.7, 0.3 + rng() * 0.2, 1.1 + rng() * 0.2]);
    target.push(1);
  }
  // Class 2: sunflower
  for (let i = 0; i < 50; i++) {
    data.push([8.0 + rng() * 5.0, 4.0 + rng() * 3.0, 0.55 + rng() * 0.2, 0.9 + rng() * 0.2]);
    target.push(2);
  }

  __seedMorphology = { data, target };
  return __seedMorphology;
}

/**
 * Load the synthetic Seed Morphology classification dataset.
 *
 * 150 samples, 4 features, 3 seed types.
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[150, 4]` and `target` shape `[150]` (int32).
 */
export function loadSeedMorphology(): Dataset {
  const { data, target } = getSeedMorphologyData();
  return {
    data: tensor(data),
    target: tensor(target, { dtype: "int32" }),
    featureNames: ["length (mm)", "width (mm)", "roundness", "density (g/cm³)"],
    targetNames: ["wheat", "rice", "sunflower"],
    description: `${SYNTHETIC_NOTE} 150 samples, 4 features, 3 seed types.`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Moons-Multi
// ─────────────────────────────────────────────────────────────────────────────

let __moonsMulti: { data: number[][]; target: number[] } | undefined;

function getMoonsMultiData() {
  if (__moonsMulti !== undefined) return __moonsMulti;
  const rng = createRng(3006);
  const data: number[][] = [];
  const target: number[] = [];

  for (let c = 0; c < 3; c++) {
    const rotation = (c * 2 * Math.PI) / 3;
    const cosR = Math.cos(rotation);
    const sinR = Math.sin(rotation);
    for (let j = 0; j < 50; j++) {
      const t = Math.PI * (j / 50);
      const baseX = Math.cos(t);
      const baseY = Math.sin(t);
      const x = baseX * cosR - baseY * sinR;
      const y = baseX * sinR + baseY * cosR;
      data.push([x + normal01(rng) * 0.08, y + normal01(rng) * 0.08]);
      target.push(c);
    }
  }

  __moonsMulti = { data, target };
  return __moonsMulti;
}

/**
 * Load the synthetic Moons-Multi classification dataset.
 *
 * 150 samples, 2D, 3 interleaving rotated moon classes.
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[150, 2]` and `target` shape `[150]` (int32).
 */
export function loadMoonsMulti(): Dataset {
  const { data, target } = getMoonsMultiData();
  return {
    data: tensor(data),
    target: tensor(target, { dtype: "int32" }),
    featureNames: ["x", "y"],
    targetNames: ["moon_0", "moon_1", "moon_2"],
    description: `${SYNTHETIC_NOTE} 150 samples, 2D, 3 interleaving moon classes.`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. Concentric Rings
// ─────────────────────────────────────────────────────────────────────────────

let __concentricRings: { data: number[][]; target: number[] } | undefined;

function getConcentricRingsData() {
  if (__concentricRings !== undefined) return __concentricRings;
  const rng = createRng(3007);
  const data: number[][] = [];
  const target: number[] = [];

  for (let c = 0; c < 3; c++) {
    const radius = 1.0 + c * 1.5;
    for (let j = 0; j < 50; j++) {
      const angle = 2 * Math.PI * (j / 50);
      data.push([
        radius * Math.cos(angle) + normal01(rng) * 0.15,
        radius * Math.sin(angle) + normal01(rng) * 0.15,
      ]);
      target.push(c);
    }
  }

  __concentricRings = { data, target };
  return __concentricRings;
}

/**
 * Load the synthetic Concentric Rings classification dataset.
 *
 * 150 samples, 2D, 3 concentric circle classes.
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[150, 2]` and `target` shape `[150]` (int32).
 */
export function loadConcentricRings(): Dataset {
  const { data, target } = getConcentricRingsData();
  return {
    data: tensor(data),
    target: tensor(target, { dtype: "int32" }),
    featureNames: ["x", "y"],
    targetNames: ["inner", "middle", "outer"],
    description: `${SYNTHETIC_NOTE} 150 samples, 2D, 3 concentric circle classes.`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. Spiral Arms
// ─────────────────────────────────────────────────────────────────────────────

let __spiralArms: { data: number[][]; target: number[] } | undefined;

function getSpiralArmsData() {
  if (__spiralArms !== undefined) return __spiralArms;
  const rng = createRng(3008);
  const data: number[][] = [];
  const target: number[] = [];

  for (let c = 0; c < 3; c++) {
    const offset = (c * 2 * Math.PI) / 3;
    for (let j = 0; j < 50; j++) {
      const t = (j / 50) * 3 * Math.PI;
      const r = 0.5 + t * 0.15;
      const angle = t + offset;
      data.push([
        r * Math.cos(angle) + normal01(rng) * 0.1,
        r * Math.sin(angle) + normal01(rng) * 0.1,
      ]);
      target.push(c);
    }
  }

  __spiralArms = { data, target };
  return __spiralArms;
}

/**
 * Load the synthetic Spiral Arms classification dataset.
 *
 * 150 samples, 2D, 3 spiral classes.
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[150, 2]` and `target` shape `[150]` (int32).
 */
export function loadSpiralArms(): Dataset {
  const { data, target } = getSpiralArmsData();
  return {
    data: tensor(data),
    target: tensor(target, { dtype: "int32" }),
    featureNames: ["x", "y"],
    targetNames: ["arm_0", "arm_1", "arm_2"],
    description: `${SYNTHETIC_NOTE} 150 samples, 2D, 3 spiral classes.`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 13. Gaussian Islands
// ─────────────────────────────────────────────────────────────────────────────

let __gaussianIslands: { data: number[][]; target: number[] } | undefined;

function getGaussianIslandsData() {
  if (__gaussianIslands !== undefined) return __gaussianIslands;
  const rng = createRng(3009);
  const data: number[][] = [];
  const target: number[] = [];

  const centers: [number, number, number][] = [
    [3, 3, 3],
    [-3, 3, -3],
    [-3, -3, 3],
    [3, -3, -3],
  ];

  let c = 0;
  for (const center of centers) {
    for (let i = 0; i < 50; i++) {
      data.push([
        center[0] + normal01(rng) * 0.8,
        center[1] + normal01(rng) * 0.8,
        center[2] + normal01(rng) * 0.8,
      ]);
      target.push(c);
    }
    c++;
  }

  __gaussianIslands = { data, target };
  return __gaussianIslands;
}

/**
 * Load the synthetic Gaussian Islands classification dataset.
 *
 * 200 samples, 3D, 4 separated Gaussian clusters.
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[200, 3]` and `target` shape `[200]` (int32).
 */
export function loadGaussianIslands(): Dataset {
  const { data, target } = getGaussianIslandsData();
  return {
    data: tensor(data),
    target: tensor(target, { dtype: "int32" }),
    featureNames: ["x", "y", "z"],
    targetNames: ["island_0", "island_1", "island_2", "island_3"],
    description: `${SYNTHETIC_NOTE} 200 samples, 3D, 4 separated Gaussian clusters.`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 14. Plant Growth
// ─────────────────────────────────────────────────────────────────────────────

let __plantGrowth: { data: number[][]; target: number[] } | undefined;

function getPlantGrowthData() {
  if (__plantGrowth !== undefined) return __plantGrowth;
  const rng = createRng(3010);
  const data: number[][] = [];
  const target: number[] = [];

  for (let i = 0; i < 200; i++) {
    const sunlight = 4 + rng() * 8;
    const water = 100 + rng() * 400;
    const soilQuality = rng() * 10;
    data.push([sunlight, water, soilQuality]);

    const height = 5 + 2.5 * sunlight + 0.02 * water + 3 * soilQuality + (rng() - 0.5) * 10;
    target.push(height);
  }

  __plantGrowth = { data, target };
  return __plantGrowth;
}

/**
 * Load the synthetic Plant Growth regression dataset.
 *
 * 200 samples, 3 features (sunlight, water, soil quality), target: height (cm).
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[200, 3]` and `target` shape `[200]`.
 */
export function loadPlantGrowth(): Dataset {
  const { data, target } = getPlantGrowthData();
  return {
    data: tensor(data),
    target: tensor(target),
    featureNames: ["sunlight (hours/day)", "water (mL/day)", "soil quality (0-10)"],
    description: `${SYNTHETIC_NOTE} 200 samples, 3 features, target: height (cm) after 30 days.`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 15. Housing-Mini
// ─────────────────────────────────────────────────────────────────────────────

let __housingMini: { data: number[][]; target: number[] } | undefined;

function getHousingMiniData() {
  if (__housingMini !== undefined) return __housingMini;
  const rng = createRng(3011);
  const data: number[][] = [];
  const target: number[] = [];

  for (let i = 0; i < 200; i++) {
    const size = 30 + rng() * 170;
    const rooms = 1 + Math.floor(rng() * 6);
    const age = rng() * 50;
    const distance = 0.5 + rng() * 29.5;
    data.push([size, rooms, age, distance]);

    const price = 80 + 2.0 * size + 15 * rooms - 0.5 * age - 1.5 * distance + (rng() - 0.5) * 30;
    target.push(price);
  }

  __housingMini = { data, target };
  return __housingMini;
}

/**
 * Load the synthetic Housing-Mini regression dataset.
 *
 * 200 samples, 4 features (size, rooms, age, distance), target: price (thousands).
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[200, 4]` and `target` shape `[200]`.
 */
export function loadHousingMini(): Dataset {
  const { data, target } = getHousingMiniData();
  return {
    data: tensor(data),
    target: tensor(target),
    featureNames: ["size (sqm)", "rooms", "age (years)", "distance to center (km)"],
    description: `${SYNTHETIC_NOTE} 200 samples, 4 features, target: price (thousands).`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 16. Energy Efficiency
// ─────────────────────────────────────────────────────────────────────────────

let __energyEfficiency: { data: number[][]; target: number[] } | undefined;

function getEnergyEfficiencyData() {
  if (__energyEfficiency !== undefined) return __energyEfficiency;
  const rng = createRng(3012);
  const data: number[][] = [];
  const target: number[] = [];

  for (let i = 0; i < 200; i++) {
    const insulation = 1 + rng() * 7;
    const windowArea = 5 + rng() * 25;
    const orientation = rng() * 360;
    data.push([insulation, windowArea, orientation]);

    const orientationRad = orientation * (Math.PI / 180);
    const energy =
      250 - 20 * insulation + 4 * windowArea + 15 * Math.cos(orientationRad) + (rng() - 0.5) * 20;
    target.push(energy);
  }

  __energyEfficiency = { data, target };
  return __energyEfficiency;
}

/**
 * Load the synthetic Energy Efficiency regression dataset.
 *
 * 200 samples, 3 features (insulation, window area, orientation), target: energy usage (kWh).
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[200, 3]` and `target` shape `[200]`.
 */
export function loadEnergyEfficiency(): Dataset {
  const { data, target } = getEnergyEfficiencyData();
  return {
    data: tensor(data),
    target: tensor(target),
    featureNames: ["insulation (R-value)", "window area (sqm)", "orientation (degrees)"],
    description: `${SYNTHETIC_NOTE} 200 samples, 3 features, target: energy usage (kWh).`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 17. Crop Yield
// ─────────────────────────────────────────────────────────────────────────────

let __cropYield: { data: number[][]; target: number[] } | undefined;

function getCropYieldData() {
  if (__cropYield !== undefined) return __cropYield;
  const rng = createRng(3013);
  const data: number[][] = [];
  const target: number[] = [];

  for (let i = 0; i < 200; i++) {
    const rainfall = 200 + rng() * 600;
    const fertilizer = 50 + rng() * 250;
    const temperature = 15 + rng() * 20;
    data.push([rainfall, fertilizer, temperature]);

    const tempEffect = -0.015 * (temperature - 25) * (temperature - 25);
    const yieldVal = 2 + 0.005 * rainfall + 0.008 * fertilizer + tempEffect + (rng() - 0.5) * 1.0;
    target.push(yieldVal);
  }

  __cropYield = { data, target };
  return __cropYield;
}

/**
 * Load the synthetic Crop Yield regression dataset.
 *
 * 200 samples, 3 features (rainfall, fertilizer, temperature), target: yield (tons/ha).
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[200, 3]` and `target` shape `[200]`.
 */
export function loadCropYield(): Dataset {
  const { data, target } = getCropYieldData();
  return {
    data: tensor(data),
    target: tensor(target),
    featureNames: ["rainfall (mm)", "fertilizer (kg/ha)", "temperature (°C)"],
    description: `${SYNTHETIC_NOTE} 200 samples, 3 features, target: yield (tons/ha).`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 18. Customer Segments
// ─────────────────────────────────────────────────────────────────────────────

let __customerSegments: { data: number[][]; target: number[] } | undefined;

function getCustomerSegmentsData() {
  if (__customerSegments !== undefined) return __customerSegments;
  const rng = createRng(3014);
  const data: number[][] = [];
  const target: number[] = [];

  // Cluster 0
  for (let i = 0; i < 50; i++) {
    data.push([25 + normal01(rng) * 3, 30 + normal01(rng) * 5, 70 + normal01(rng) * 8]);
    target.push(0);
  }
  // Cluster 1
  for (let i = 0; i < 50; i++) {
    data.push([30 + normal01(rng) * 4, 80 + normal01(rng) * 8, 85 + normal01(rng) * 6]);
    target.push(1);
  }
  // Cluster 2
  for (let i = 0; i < 50; i++) {
    data.push([55 + normal01(rng) * 5, 50 + normal01(rng) * 8, 40 + normal01(rng) * 8]);
    target.push(2);
  }
  // Cluster 3
  for (let i = 0; i < 50; i++) {
    data.push([60 + normal01(rng) * 4, 90 + normal01(rng) * 7, 30 + normal01(rng) * 7]);
    target.push(3);
  }

  __customerSegments = { data, target };
  return __customerSegments;
}

/**
 * Load the synthetic Customer Segments clustering dataset.
 *
 * 200 samples, 3 features (age, income, spending score), 4 natural clusters.
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[200, 3]` and `target` shape `[200]` (int32).
 */
export function loadCustomerSegments(): Dataset {
  const { data, target } = getCustomerSegmentsData();
  return {
    data: tensor(data),
    target: tensor(target, { dtype: "int32" }),
    featureNames: ["age", "income (thousands)", "spending score (0-100)"],
    targetNames: ["young_budget", "young_premium", "mature_moderate", "mature_saver"],
    description: `${SYNTHETIC_NOTE} 200 samples, 3 features, 4 natural clusters.`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 19. Sensor States
// ─────────────────────────────────────────────────────────────────────────────

let __sensorStates: { data: number[][]; target: number[] } | undefined;

function getSensorStatesData() {
  if (__sensorStates !== undefined) return __sensorStates;
  const rng = createRng(3015);
  const data: number[][] = [];
  const target: number[] = [];

  // Mode 0: normal
  for (let i = 0; i < 60; i++) {
    data.push([
      50 + rng() * 10,
      1013 + rng() * 5,
      60 + rng() * 10,
      220 + rng() * 5,
      3 + rng() * 0.5,
      0.5 + rng() * 0.3,
    ]);
    target.push(0);
  }
  // Mode 1: heating
  for (let i = 0; i < 60; i++) {
    data.push([
      80 + rng() * 15,
      1010 + rng() * 8,
      40 + rng() * 15,
      218 + rng() * 8,
      5 + rng() * 1,
      0.8 + rng() * 0.4,
    ]);
    target.push(1);
  }
  // Mode 2: fault
  for (let i = 0; i < 60; i++) {
    data.push([
      30 + rng() * 40,
      1000 + rng() * 20,
      20 + rng() * 60,
      200 + rng() * 30,
      1 + rng() * 8,
      0.2 + rng() * 1.5,
    ]);
    target.push(2);
  }

  __sensorStates = { data, target };
  return __sensorStates;
}

/**
 * Load the synthetic Sensor States classification dataset.
 *
 * 180 samples, 6 sensor readings, 3 hidden operating modes.
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[180, 6]` and `target` shape `[180]` (int32).
 */
export function loadSensorStates(): Dataset {
  const { data, target } = getSensorStatesData();
  return {
    data: tensor(data),
    target: tensor(target, { dtype: "int32" }),
    featureNames: [
      "temperature (°C)",
      "pressure (hPa)",
      "humidity (%)",
      "voltage (V)",
      "vibration (mm/s)",
      "current (A)",
    ],
    targetNames: ["normal", "heating", "fault"],
    description: `${SYNTHETIC_NOTE} 180 samples, 6 sensor readings, 3 hidden operating modes.`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 20. Student Performance
// ─────────────────────────────────────────────────────────────────────────────

let __studentPerformance: { data: number[][]; target: number[] } | undefined;

function getStudentPerformanceData() {
  if (__studentPerformance !== undefined) return __studentPerformance;
  const rng = createRng(3016);
  const data: number[][] = [];
  const target: number[] = [];

  // Class 0: fail
  for (let i = 0; i < 50; i++) {
    data.push([Math.floor(rng() * 6), 8 + Math.floor(rng() * 8), 20 + Math.floor(rng() * 30)]);
    target.push(0);
  }
  // Class 1: pass
  for (let i = 0; i < 50; i++) {
    data.push([4 + Math.floor(rng() * 8), 3 + Math.floor(rng() * 6), 45 + Math.floor(rng() * 30)]);
    target.push(1);
  }
  // Class 2: excellent
  for (let i = 0; i < 50; i++) {
    data.push([10 + Math.floor(rng() * 11), Math.floor(rng() * 4), 70 + Math.floor(rng() * 31)]);
    target.push(2);
  }

  __studentPerformance = { data, target };
  return __studentPerformance;
}

/**
 * Load the synthetic Student Performance classification dataset.
 *
 * 150 samples, 3 integer features, 3 outcome classes.
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[150, 3]` and `target` shape `[150]` (int32).
 */
export function loadStudentPerformance(): Dataset {
  const { data, target } = getStudentPerformanceData();
  return {
    data: tensor(data),
    target: tensor(target, { dtype: "int32" }),
    featureNames: ["study hours (per week)", "absences", "quiz score (0-100)"],
    targetNames: ["fail", "pass", "excellent"],
    description: `${SYNTHETIC_NOTE} 150 samples, 3 integer features, 3 outcome classes.`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 21. Traffic Conditions
// ─────────────────────────────────────────────────────────────────────────────

let __trafficConditions: { data: number[][]; target: number[] } | undefined;

function getTrafficConditionsData() {
  if (__trafficConditions !== undefined) return __trafficConditions;
  const rng = createRng(3017);
  const data: number[][] = [];
  const target: number[] = [];

  // Class 0: light
  for (let i = 0; i < 50; i++) {
    data.push([Math.floor(rng() * 10), 50 + Math.floor(rng() * 40), 5 + Math.floor(rng() * 20)]);
    target.push(0);
  }
  // Class 1: moderate
  for (let i = 0; i < 50; i++) {
    data.push([
      8 + Math.floor(rng() * 8),
      25 + Math.floor(rng() * 30),
      20 + Math.floor(rng() * 30),
    ]);
    target.push(1);
  }
  // Class 2: heavy
  for (let i = 0; i < 50; i++) {
    data.push([
      15 + Math.floor(rng() * 8),
      5 + Math.floor(rng() * 25),
      45 + Math.floor(rng() * 45),
    ]);
    target.push(2);
  }

  __trafficConditions = { data, target };
  return __trafficConditions;
}

/**
 * Load the synthetic Traffic Conditions classification dataset.
 *
 * 150 samples, 3 features, 3 traffic level classes.
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[150, 3]` and `target` shape `[150]` (int32).
 */
export function loadTrafficConditions(): Dataset {
  const { data, target } = getTrafficConditionsData();
  return {
    data: tensor(data),
    target: tensor(target, { dtype: "int32" }),
    featureNames: ["time of day (hour)", "speed (km/h)", "density (vehicles/km)"],
    targetNames: ["light", "moderate", "heavy"],
    description: `${SYNTHETIC_NOTE} 150 samples, 3 features, 3 traffic level classes.`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 22. Fitness Scores
// ─────────────────────────────────────────────────────────────────────────────

let __fitnessScores: { data: number[][]; target: number[][] } | undefined;

function getFitnessScoresData() {
  if (__fitnessScores !== undefined) return __fitnessScores;
  const rng = createRng(3018);
  const data: number[][] = [];
  const target: number[][] = [];

  for (let i = 0; i < 100; i++) {
    const duration = 20 + rng() * 70;
    const intensity = 1 + rng() * 9;
    const frequency = 1 + rng() * 6;
    data.push([duration, intensity, frequency]);
    target.push([
      10 + 0.3 * duration + 5 * intensity + 3 * frequency + (rng() - 0.5) * 8,
      15 + 0.5 * duration + 2 * intensity + 5 * frequency + (rng() - 0.5) * 8,
      30 + 0.4 * duration + 1 * intensity + 2 * frequency + (rng() - 0.5) * 8,
    ]);
  }

  __fitnessScores = { data, target };
  return __fitnessScores;
}

/**
 * Load the synthetic Fitness Scores multi-output regression dataset.
 *
 * 100 samples, 3 exercise features, 3 fitness targets (strength, endurance, flexibility).
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[100, 3]` and `target` shape `[100, 3]`.
 */
export function loadFitnessScores(): Dataset {
  const { data, target } = getFitnessScoresData();
  return {
    data: tensor(data),
    target: tensor(target),
    featureNames: ["exercise duration (min)", "intensity (1-10)", "frequency (times/week)"],
    targetNames: ["strength", "endurance", "flexibility"],
    description: `${SYNTHETIC_NOTE} 100 samples, 3 exercise features, 3 fitness targets (multi-output).`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 23. Weather Outcomes
// ─────────────────────────────────────────────────────────────────────────────

let __weatherOutcomes: { data: number[][]; target: number[][] } | undefined;

function getWeatherOutcomesData() {
  if (__weatherOutcomes !== undefined) return __weatherOutcomes;
  const rng = createRng(3019);
  const data: number[][] = [];
  const target: number[][] = [];

  for (let i = 0; i < 150; i++) {
    const humidity = 20 + rng() * 80;
    const pressure = 990 + rng() * 50;
    const temperature = -5 + rng() * 40;
    data.push([humidity, pressure, temperature]);

    const rawRain =
      0.01 * humidity - 0.003 * (pressure - 1000) + 0.005 * temperature - 0.3 + (rng() - 0.5) * 0.1;
    const rainProb = Math.max(0, Math.min(1, rawRain));
    const windSpeed = 8 + 0.5 * Math.abs(pressure - 1013) + 0.3 * temperature + (rng() - 0.5) * 6;

    target.push([rainProb, Math.max(0, windSpeed)]);
  }

  __weatherOutcomes = { data, target };
  return __weatherOutcomes;
}

/**
 * Load the synthetic Weather Outcomes multi-output regression dataset.
 *
 * 150 samples, 3 features, 2 targets (rain probability, wind speed).
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[150, 3]` and `target` shape `[150, 2]`.
 */
export function loadWeatherOutcomes(): Dataset {
  const { data, target } = getWeatherOutcomesData();
  return {
    data: tensor(data),
    target: tensor(target),
    featureNames: ["humidity (%)", "pressure (hPa)", "temperature (°C)"],
    targetNames: ["rain probability", "wind speed (km/h)"],
    description: `${SYNTHETIC_NOTE} 150 samples, 3 features, 2 targets (multi-output regression).`,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// 24. Perfectly Separable
// ─────────────────────────────────────────────────────────────────────────────

let __perfectlySeparable: { data: number[][]; target: number[] } | undefined;

function getPerfectlySeparableData() {
  if (__perfectlySeparable !== undefined) return __perfectlySeparable;
  const rng = createRng(3020);
  const data: number[][] = [];
  const target: number[] = [];

  // Class 0
  for (let i = 0; i < 50; i++) {
    data.push([1 + rng() * 0.8, 0.5 + rng() * 0.6, -1 + rng() * 0.7, 2 + rng() * 0.5]);
    target.push(0);
  }
  // Class 1
  for (let i = 0; i < 50; i++) {
    data.push([4 + rng() * 0.8, 3.5 + rng() * 0.6, 2 + rng() * 0.7, 5 + rng() * 0.5]);
    target.push(1);
  }

  __perfectlySeparable = { data, target };
  return __perfectlySeparable;
}

/**
 * Load the synthetic Perfectly Separable classification dataset.
 *
 * 100 samples, 4 features, 2 linearly separable classes.
 * Deterministic — always returns the same data.
 *
 * @returns A {@link Dataset} with `data` shape `[100, 4]` and `target` shape `[100]` (int32).
 */
export function loadPerfectlySeparable(): Dataset {
  const { data, target } = getPerfectlySeparableData();
  return {
    data: tensor(data),
    target: tensor(target, { dtype: "int32" }),
    featureNames: ["feature_0", "feature_1", "feature_2", "feature_3"],
    targetNames: ["class_0", "class_1"],
    description: `${SYNTHETIC_NOTE} 100 samples, 4 features, 2 linearly separable classes.`,
  };
}
