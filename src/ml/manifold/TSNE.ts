import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { validateUnsupervisedFitInputs } from "../_validation";

type SparseRow = { indices: number[]; values: number[] };
type SparseMatrix = SparseRow[];
type SampleRow = { indices: number[]; qValues: number[] };

/**
 * t-Distributed Stochastic Neighbor Embedding (t-SNE).
 *
 * A nonlinear dimensionality reduction technique for embedding high-dimensional
 * data into a low-dimensional space (typically 2D or 3D) for visualization.
 *
 * **Algorithm**: Exact t-SNE with an optional sampling-based approximation
 * - Computes pairwise affinities in high-dimensional space using Gaussian kernel (exact)
 * - Computes pairwise affinities in low-dimensional space using Student-t distribution
 * - Minimizes KL divergence between the two distributions
 *
 * **Scalability Note**:
 * Exact t-SNE is O(n^2) in time and memory. For large datasets, use
 * `method: "approximate"` (sampled neighbors + negative sampling) or reduce samples.
 *
 * @example
 * ```ts
 * import { TSNE } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]);
 *
 * const tsne = new TSNE({ nComponents: 2, perplexity: 5 });
 * const embedding = tsne.fitTransform(X);
 * ```
 *
 * @see {@link https://deepbox.dev/docs/ml-manifold | Deepbox Manifold Learning}
 * @see van der Maaten, L.J.P.; Hinton, G.E. (2008). "Visualizing High-Dimensional Data Using t-SNE"
 */
export class TSNE {
  /** Number of dimensions in the embedding */
  private readonly nComponents: number;

  /** Perplexity parameter (related to number of nearest neighbors) */
  private readonly perplexity: number;

  /** Learning rate for gradient descent */
  private readonly learningRate: number;

  /** Number of iterations */
  private readonly nIter: number;

  /** Early exaggeration factor */
  private readonly earlyExaggeration: number;

  /** Number of iterations with early exaggeration */
  private readonly earlyExaggerationIter: number;

  /** Random seed for reproducibility */
  private readonly randomState: number | undefined;

  /** Minimum gradient norm for convergence */
  private readonly minGradNorm: number;

  /** Method for computing affinities and gradients */
  private readonly method: "exact" | "approximate";

  /** Maximum samples allowed for exact mode */
  private readonly maxExactSamples: number;

  /** Neighbor count for approximate mode */
  private readonly approximateNeighbors: number;

  /** Negative samples per point for approximate mode */
  private readonly negativeSamples: number;

  /** The embedded points after fitting */
  private embedding: number[][] = [];

  /** Whether the model has been fitted */
  private fitted = false;

  constructor(
    options: {
      readonly nComponents?: number;
      readonly perplexity?: number;
      readonly learningRate?: number;
      readonly nIter?: number;
      readonly earlyExaggeration?: number;
      readonly earlyExaggerationIter?: number;
      readonly randomState?: number;
      readonly minGradNorm?: number;
      /** "exact" uses full pairwise interactions; "approximate" uses sampling for large datasets. */
      readonly method?: "exact" | "approximate";
      /** Maximum samples allowed in exact mode before requiring approximate. */
      readonly maxExactSamples?: number;
      /** Number of neighbors to sample per point in approximate mode. */
      readonly approximateNeighbors?: number;
      /** Number of negative samples per point in approximate mode. */
      readonly negativeSamples?: number;
    } = {}
  ) {
    this.nComponents = options.nComponents ?? 2;
    this.perplexity = options.perplexity ?? 30;
    this.learningRate = options.learningRate ?? 200;
    this.nIter = options.nIter ?? 1000;
    this.earlyExaggeration = options.earlyExaggeration ?? 12;
    const earlyExaggerationIter = options.earlyExaggerationIter ?? 250;
    this.earlyExaggerationIter = Math.min(earlyExaggerationIter, this.nIter);
    this.randomState = options.randomState;
    this.minGradNorm = options.minGradNorm ?? 1e-7;
    this.method = options.method ?? "exact";
    this.maxExactSamples = options.maxExactSamples ?? 2000;
    this.approximateNeighbors =
      options.approximateNeighbors ?? Math.max(5, Math.floor(this.perplexity * 3));
    this.negativeSamples = options.negativeSamples ?? Math.max(10, Math.floor(this.perplexity * 2));

    if (!Number.isInteger(this.nComponents) || this.nComponents <= 0) {
      throw new InvalidParameterError(
        "nComponents must be positive",
        "nComponents",
        this.nComponents
      );
    }
    if (!Number.isFinite(this.perplexity) || this.perplexity <= 0) {
      throw new InvalidParameterError("perplexity must be positive", "perplexity", this.perplexity);
    }
    if (!Number.isFinite(this.learningRate) || this.learningRate <= 0) {
      throw new InvalidParameterError(
        "learningRate must be positive",
        "learningRate",
        this.learningRate
      );
    }
    if (!Number.isInteger(this.nIter) || this.nIter <= 0) {
      throw new InvalidParameterError("nIter must be a positive integer", "nIter", this.nIter);
    }
    if (!Number.isFinite(this.earlyExaggeration) || this.earlyExaggeration <= 0) {
      throw new InvalidParameterError(
        "earlyExaggeration must be positive",
        "earlyExaggeration",
        this.earlyExaggeration
      );
    }
    if (!Number.isInteger(earlyExaggerationIter) || earlyExaggerationIter < 0) {
      throw new InvalidParameterError(
        "earlyExaggerationIter must be an integer >= 0",
        "earlyExaggerationIter",
        earlyExaggerationIter
      );
    }
    if (!Number.isFinite(this.minGradNorm) || this.minGradNorm <= 0) {
      throw new InvalidParameterError(
        "minGradNorm must be positive",
        "minGradNorm",
        this.minGradNorm
      );
    }
    if (this.method !== "exact" && this.method !== "approximate") {
      throw new InvalidParameterError(
        "method must be 'exact' or 'approximate'",
        "method",
        this.method
      );
    }
    if (!Number.isInteger(this.maxExactSamples) || this.maxExactSamples <= 0) {
      throw new InvalidParameterError(
        "maxExactSamples must be a positive integer",
        "maxExactSamples",
        this.maxExactSamples
      );
    }
    if (!Number.isInteger(this.approximateNeighbors) || this.approximateNeighbors <= 0) {
      throw new InvalidParameterError(
        "approximateNeighbors must be a positive integer",
        "approximateNeighbors",
        this.approximateNeighbors
      );
    }
    if (!Number.isInteger(this.negativeSamples) || this.negativeSamples <= 0) {
      throw new InvalidParameterError(
        "negativeSamples must be a positive integer",
        "negativeSamples",
        this.negativeSamples
      );
    }
    if (options.randomState !== undefined && !Number.isFinite(options.randomState)) {
      throw new InvalidParameterError(
        "randomState must be a finite number",
        "randomState",
        options.randomState
      );
    }
  }

  /**
   * Compute pairwise squared Euclidean distances.
   */
  private computeDistances(X: number[][]): number[][] {
    const n = X.length;
    const distances: number[][] = [];

    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < n; j++) {
        if (i === j) {
          row.push(0);
        } else {
          let dist = 0;
          const xi = X[i];
          const xj = X[j];
          if (xi && xj) {
            for (let k = 0; k < xi.length; k++) {
              const diff = (xi[k] ?? 0) - (xj[k] ?? 0);
              dist += diff * diff;
            }
          }
          row.push(dist);
        }
      }
      distances.push(row);
    }

    return distances;
  }

  /**
   * Compute squared Euclidean distance between two vectors.
   */
  private computeSquaredDistance(a: number[], b: number[]): number {
    let dist = 0;
    for (let k = 0; k < a.length; k++) {
      const diff = (a[k] ?? 0) - (b[k] ?? 0);
      dist += diff * diff;
    }
    return dist;
  }

  /**
   * Sample unique indices excluding a single index.
   */
  private sampleIndices(n: number, exclude: number, k: number, rng: () => number): number[] {
    const result = new Set<number>();
    const maxAttempts = k * 10 + 100;
    let attempts = 0;

    while (result.size < k && attempts < maxAttempts) {
      const raw = Math.floor(rng() * (n - 1));
      const idx = raw >= exclude ? raw + 1 : raw;
      result.add(idx);
      attempts += 1;
    }

    if (result.size < k) {
      for (let i = 0; i < n && result.size < k; i++) {
        if (i !== exclude) {
          result.add(i);
        }
      }
    }

    return Array.from(result);
  }

  /**
   * Sample unique indices excluding a set of indices.
   */
  private sampleIndicesWithExclusions(
    n: number,
    exclusions: ReadonlySet<number>,
    k: number,
    rng: () => number
  ): number[] {
    const result = new Set<number>();
    const maxAttempts = k * 12 + 200;
    let attempts = 0;

    while (result.size < k && attempts < maxAttempts) {
      const idx = Math.floor(rng() * n);
      if (!exclusions.has(idx)) {
        result.add(idx);
      }
      attempts += 1;
    }

    if (result.size < k) {
      for (let i = 0; i < n && result.size < k; i++) {
        if (!exclusions.has(i)) {
          result.add(i);
        }
      }
    }

    return Array.from(result);
  }

  /**
   * Compute conditional probabilities P(j|i) using binary search for sigma.
   */
  private computeProbabilities(distances: number[][]): number[][] {
    const n = distances.length;
    const targetEntropy = Math.log(this.perplexity);
    const P: number[][] = [];

    for (let i = 0; i < n; i++) {
      const row: number[] = new Array(n).fill(0);

      // Binary search for sigma
      let sigmaMin = 1e-10;
      let sigmaMax = 1e10;
      let sigma = 1.0;

      for (let iter = 0; iter < 50; iter++) {
        // Compute probabilities with current sigma
        let sumExp = 0;
        for (let j = 0; j < n; j++) {
          if (i !== j) {
            const distRow = distances[i];
            const dist = distRow ? (distRow[j] ?? 0) : 0;
            sumExp += Math.exp(-dist / (2 * sigma * sigma));
          }
        }

        // Compute entropy
        let entropy = 0;
        for (let j = 0; j < n; j++) {
          if (i !== j) {
            const distRow = distances[i];
            const dist = distRow ? (distRow[j] ?? 0) : 0;
            const pij = Math.exp(-dist / (2 * sigma * sigma)) / (sumExp + 1e-10);
            if (pij > 1e-10) {
              entropy -= pij * Math.log(pij);
            }
          }
        }

        // Binary search adjustment
        if (Math.abs(entropy - targetEntropy) < 1e-5) {
          break;
        }

        if (entropy > targetEntropy) {
          sigmaMax = sigma;
          sigma = (sigma + sigmaMin) / 2;
        } else {
          sigmaMin = sigma;
          sigma = (sigma + sigmaMax) / 2;
        }
      }

      // Compute final probabilities
      let sumExp = 0;
      for (let j = 0; j < n; j++) {
        if (i !== j) {
          const distRow = distances[i];
          const dist = distRow ? (distRow[j] ?? 0) : 0;
          sumExp += Math.exp(-dist / (2 * sigma * sigma));
        }
      }

      for (let j = 0; j < n; j++) {
        if (i !== j) {
          const distRow = distances[i];
          const dist = distRow ? (distRow[j] ?? 0) : 0;
          row[j] = Math.exp(-dist / (2 * sigma * sigma)) / (sumExp + 1e-10);
        }
      }

      P.push(row);
    }

    // Symmetrize: P = (P + P') / (2n)
    const Psym: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < n; j++) {
        const pij = P[i]?.[j] ?? 0;
        const pji = P[j]?.[i] ?? 0;
        row.push((pij + pji) / (2 * n));
      }
      Psym.push(row);
    }

    return Psym;
  }

  /**
   * Compute sparse joint probabilities using sampled neighbors (approximate).
   */
  private computeProbabilitiesSparse(
    X: number[][],
    neighbors: number,
    rng: () => number
  ): SparseMatrix {
    const n = X.length;
    const targetEntropy = Math.log(this.perplexity);
    const rows: SparseMatrix = [];

    for (let i = 0; i < n; i++) {
      const neighborIndices = this.sampleIndices(n, i, neighbors, rng);
      const distances = neighborIndices.map((j) =>
        this.computeSquaredDistance(X[i] ?? [], X[j] ?? [])
      );

      let sigmaMin = 1e-10;
      let sigmaMax = 1e10;
      let sigma = 1.0;

      for (let iter = 0; iter < 50; iter++) {
        let sumExp = 0;
        for (const dist of distances) {
          sumExp += Math.exp(-dist / (2 * sigma * sigma));
        }

        let entropy = 0;
        for (const dist of distances) {
          const pij = Math.exp(-dist / (2 * sigma * sigma)) / (sumExp + 1e-10);
          if (pij > 1e-10) {
            entropy -= pij * Math.log(pij);
          }
        }

        if (Math.abs(entropy - targetEntropy) < 1e-5) {
          break;
        }

        if (entropy > targetEntropy) {
          sigmaMax = sigma;
          sigma = (sigma + sigmaMin) / 2;
        } else {
          sigmaMin = sigma;
          sigma = (sigma + sigmaMax) / 2;
        }
      }

      let sumExp = 0;
      for (const dist of distances) {
        sumExp += Math.exp(-dist / (2 * sigma * sigma));
      }

      const values: number[] = distances.map(
        (dist) => Math.exp(-dist / (2 * sigma * sigma)) / (sumExp + 1e-10)
      );

      rows.push({ indices: neighborIndices, values });
    }

    return this.symmetrizeSparse(rows, n);
  }

  /**
   * Symmetrize sparse probabilities: P = (P + P^T) / (2n).
   */
  private symmetrizeSparse(rows: SparseMatrix, n: number): SparseMatrix {
    const maps: Array<Map<number, number>> = rows.map((row) => {
      const map = new Map<number, number>();
      for (let k = 0; k < row.indices.length; k++) {
        const j = row.indices[k];
        if (j === undefined) continue;
        const val = row.values[k] ?? 0;
        map.set(j, val);
      }
      return map;
    });

    const sym: SparseMatrix = [];
    for (let i = 0; i < n; i++) {
      const row = rows[i];
      if (!row) {
        sym.push({ indices: [], values: [] });
        continue;
      }
      const indices = row.indices.slice();
      const values: number[] = [];
      for (let k = 0; k < indices.length; k++) {
        const j = indices[k];
        if (j === undefined) continue;
        const pij = maps[i]?.get(j) ?? 0;
        const pji = maps[j]?.get(i) ?? 0;
        values.push((pij + pji) / (2 * n));
      }
      sym.push({ indices, values });
    }

    return sym;
  }

  /**
   * Initialize embedding with small random values.
   */
  private initializeEmbedding(n: number): number[][] {
    const embedding: number[][] = [];

    // Simple LCG random number generator for reproducibility
    let seed = this.randomState ?? Date.now();
    const random = (): number => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };

    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < this.nComponents; j++) {
        // Initialize with small random values (normal-ish distribution)
        const u1 = random() || 0.0001;
        const u2 = random();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        row.push(z * 0.0001);
      }
      embedding.push(row);
    }

    return embedding;
  }

  /**
   * Compute Q distribution (Student-t with 1 degree of freedom).
   */
  private computeQ(Y: number[][]): { Q: number[][]; sumQ: number } {
    const n = Y.length;
    const Q: number[][] = [];
    let sumQ = 0;

    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < n; j++) {
        if (i === j) {
          row.push(0);
        } else {
          let dist = 0;
          const yi = Y[i];
          const yj = Y[j];
          if (yi && yj) {
            for (let k = 0; k < this.nComponents; k++) {
              const diff = (yi[k] ?? 0) - (yj[k] ?? 0);
              dist += diff * diff;
            }
          }
          // Student-t distribution with 1 degree of freedom
          const qij = 1 / (1 + dist);
          row.push(qij);
          sumQ += qij;
        }
      }
      Q.push(row);
    }

    return { Q, sumQ };
  }

  /**
   * Compute approximate Q using sampled pairs.
   */
  private computeQApprox(
    Y: number[][],
    neighbors: SparseMatrix,
    negativeSamples: number,
    rng: () => number
  ): { rows: SampleRow[]; sumQ: number } {
    const n = Y.length;
    const rows: SampleRow[] = [];
    let sumQ = 0;

    for (let i = 0; i < n; i++) {
      const neighborIndices = neighbors[i]?.indices ?? [];
      const exclusion = new Set<number>(neighborIndices);
      exclusion.add(i);

      const negatives = this.sampleIndicesWithExclusions(n, exclusion, negativeSamples, rng);
      const indices = neighborIndices.concat(negatives);
      const qValues: number[] = [];

      const yi = Y[i] ?? [];
      for (let k = 0; k < indices.length; k++) {
        const j = indices[k] ?? 0;
        const yj = Y[j] ?? [];
        const dist = this.computeSquaredDistance(yi, yj);
        const qij = 1 / (1 + dist);
        qValues.push(qij);
        sumQ += qij;
      }

      rows.push({ indices, qValues });
    }

    return { rows, sumQ };
  }

  /**
   * Compute gradients of KL divergence.
   */
  private computeGradients(P: number[][], Q: number[][], sumQ: number, Y: number[][]): number[][] {
    const n = Y.length;
    const gradients: number[][] = [];

    for (let i = 0; i < n; i++) {
      const grad: number[] = new Array(this.nComponents).fill(0);

      for (let j = 0; j < n; j++) {
        if (i !== j) {
          const pij = P[i]?.[j] ?? 0;
          const qij = (Q[i]?.[j] ?? 0) / (sumQ + 1e-10);

          const yi = Y[i];
          const yj = Y[j];
          if (yi && yj) {
            // Compute (1 + ||y_i - y_j||^2)^-1
            let dist = 0;
            for (let k = 0; k < this.nComponents; k++) {
              const diff = (yi[k] ?? 0) - (yj[k] ?? 0);
              dist += diff * diff;
            }
            const mult = (pij - qij) * (1 / (1 + dist));

            for (let k = 0; k < this.nComponents; k++) {
              grad[k] = (grad[k] ?? 0) + 4 * mult * ((yi[k] ?? 0) - (yj[k] ?? 0));
            }
          }
        }
      }

      gradients.push(grad);
    }

    return gradients;
  }

  /**
   * Compute approximate gradients of KL divergence using sampled pairs.
   */
  private computeGradientsApprox(
    PMaps: Array<Map<number, number>>,
    QRows: SampleRow[],
    sumQ: number,
    Y: number[][],
    exaggeration: number
  ): number[][] {
    const n = Y.length;
    const gradients: number[][] = [];
    const denom = sumQ + 1e-10;

    for (let i = 0; i < n; i++) {
      const grad: number[] = new Array(this.nComponents).fill(0);
      const yi = Y[i] ?? [];
      const pMap = PMaps[i];
      const qRow = QRows[i];

      if (qRow) {
        for (let idx = 0; idx < qRow.indices.length; idx++) {
          const j = qRow.indices[idx] ?? 0;
          if (i === j) continue;
          const yj = Y[j] ?? [];

          const pijBase = pMap?.get(j) ?? 0;
          const pij = pijBase * exaggeration;
          const qUnnorm = qRow.qValues[idx] ?? 0;
          const qij = qUnnorm / denom;

          const mult = (pij - qij) * qUnnorm;
          for (let k = 0; k < this.nComponents; k++) {
            grad[k] = (grad[k] ?? 0) + 4 * mult * ((yi[k] ?? 0) - (yj[k] ?? 0));
          }
        }
      }

      gradients.push(grad);
    }

    return gradients;
  }

  /**
   * Fit the t-SNE model and return the embedding.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @returns Low-dimensional embedding of shape (n_samples, n_components)
   */
  fitTransform(X: Tensor): Tensor {
    validateUnsupervisedFitInputs(X);

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;

    if (nSamples < 4) {
      throw new InvalidParameterError("t-SNE requires at least 4 samples", "nSamples", nSamples);
    }
    if (this.perplexity >= nSamples) {
      throw new InvalidParameterError(
        `perplexity must be less than n_samples; received perplexity=${this.perplexity}, n_samples=${nSamples}`,
        "perplexity",
        this.perplexity
      );
    }
    if (this.method === "exact" && nSamples > this.maxExactSamples) {
      throw new InvalidParameterError(
        `Exact t-SNE is O(n^2) and limited to n_samples <= ${this.maxExactSamples}; received n_samples=${nSamples}. Use method="approximate" or increase maxExactSamples.`,
        "nSamples",
        nSamples
      );
    }

    // Extract data
    const XData: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      XData.push(row);
    }

    const baseSeed = this.randomState ?? Date.now();
    const rng = (() => {
      let state = baseSeed >>> 0;
      return (): number => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 2 ** 32;
      };
    })();

    const useApproximate = this.method === "approximate";
    const neighborCount = Math.min(this.approximateNeighbors, nSamples - 1);
    const availableNegatives = Math.max(0, nSamples - 1 - neighborCount);
    const negativeCount = Math.min(this.negativeSamples, availableNegatives);
    if (useApproximate && neighborCount < 2) {
      throw new InvalidParameterError(
        "approximateNeighbors must be at least 2 for approximate mode",
        "approximateNeighbors",
        neighborCount
      );
    }
    if (useApproximate && this.perplexity >= neighborCount) {
      throw new InvalidParameterError(
        `perplexity must be less than approximateNeighbors; received perplexity=${this.perplexity}, approximateNeighbors=${neighborCount}`,
        "perplexity",
        this.perplexity
      );
    }

    // Compute joint probabilities P
    const PExact = useApproximate ? null : this.computeProbabilities(this.computeDistances(XData));
    const PSparse = useApproximate
      ? this.computeProbabilitiesSparse(XData, neighborCount, rng)
      : null;

    // Initialize embedding
    const Y = this.initializeEmbedding(nSamples);

    // Momentum terms
    const velocities: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      velocities.push(new Array(this.nComponents).fill(0));
    }

    const momentum = 0.5;
    const finalMomentum = 0.8;

    const PMaps =
      useApproximate && PSparse
        ? PSparse.map((row) => {
            const map = new Map<number, number>();
            for (let k = 0; k < row.indices.length; k++) {
              const j = row.indices[k];
              if (j === undefined) continue;
              const val = row.values[k] ?? 0;
              map.set(j, val);
            }
            return map;
          })
        : [];

    // Gradient descent
    for (let iter = 0; iter < this.nIter; iter++) {
      // Early exaggeration
      const exaggeration = iter < this.earlyExaggerationIter ? this.earlyExaggeration : 1;

      let gradients: number[][];
      if (useApproximate && PSparse) {
        const { rows: QRows, sumQ } = this.computeQApprox(Y, PSparse, negativeCount, rng);
        gradients = this.computeGradientsApprox(PMaps, QRows, sumQ, Y, exaggeration);
      } else {
        // Apply exaggeration to P
        const Pexag: number[][] = [];
        for (let i = 0; i < nSamples; i++) {
          const row: number[] = [];
          for (let j = 0; j < nSamples; j++) {
            row.push((PExact?.[i]?.[j] ?? 0) * exaggeration);
          }
          Pexag.push(row);
        }

        // Compute Q
        const { Q, sumQ } = this.computeQ(Y);

        // Compute gradients
        gradients = this.computeGradients(Pexag, Q, sumQ, Y);
      }

      // Check convergence
      let gradNorm = 0;
      for (let i = 0; i < nSamples; i++) {
        const grad = gradients[i];
        if (grad) {
          for (let k = 0; k < this.nComponents; k++) {
            gradNorm += (grad[k] ?? 0) ** 2;
          }
        }
      }
      gradNorm = Math.sqrt(gradNorm);

      if (gradNorm < this.minGradNorm) {
        break;
      }

      // Update momentum
      const currentMomentum = iter < this.earlyExaggerationIter ? momentum : finalMomentum;

      // Update embedding with momentum
      for (let i = 0; i < nSamples; i++) {
        const yi = Y[i];
        const grad = gradients[i];
        const vel = velocities[i];

        if (yi && grad && vel) {
          for (let k = 0; k < this.nComponents; k++) {
            vel[k] = currentMomentum * (vel[k] ?? 0) - this.learningRate * (grad[k] ?? 0);
            yi[k] = (yi[k] ?? 0) + (vel[k] ?? 0);
          }
        }
      }

      // Center the embedding
      const center: number[] = new Array(this.nComponents).fill(0);
      for (let i = 0; i < nSamples; i++) {
        const yi = Y[i];
        if (yi) {
          for (let k = 0; k < this.nComponents; k++) {
            center[k] = (center[k] ?? 0) + (yi[k] ?? 0);
          }
        }
      }
      for (let k = 0; k < this.nComponents; k++) {
        center[k] = (center[k] ?? 0) / nSamples;
      }
      for (let i = 0; i < nSamples; i++) {
        const yi = Y[i];
        if (yi) {
          for (let k = 0; k < this.nComponents; k++) {
            yi[k] = (yi[k] ?? 0) - (center[k] ?? 0);
          }
        }
      }
    }

    this.embedding = Y;
    this.fitted = true;

    return tensor(Y);
  }

  /**
   * Fit the model (same as fitTransform for t-SNE).
   */
  fit(X: Tensor): this {
    this.fitTransform(X);
    return this;
  }

  /**
   * Return the fitted embedding. For t-SNE, transform is equivalent to
   * returning the already-computed embedding (t-SNE is non-parametric).
   *
   * @param _X - Ignored, present for API compatibility
   * @returns Low-dimensional embedding of shape (n_samples, n_components)
   */
  transform(_X?: Tensor): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("TSNE must be fitted before transform");
    }
    return tensor(this.embedding);
  }

  /**
   * Get the embedding.
   */
  get embeddingResult(): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("TSNE must be fitted before accessing embedding");
    }
    return tensor(this.embedding);
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      nComponents: this.nComponents,
      perplexity: this.perplexity,
      learningRate: this.learningRate,
      nIter: this.nIter,
      earlyExaggeration: this.earlyExaggeration,
      earlyExaggerationIter: this.earlyExaggerationIter,
      randomState: this.randomState,
      minGradNorm: this.minGradNorm,
      method: this.method,
      maxExactSamples: this.maxExactSamples,
      approximateNeighbors: this.approximateNeighbors,
      negativeSamples: this.negativeSamples,
    };
  }
}
