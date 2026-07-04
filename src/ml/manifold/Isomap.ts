/**
 * Isomap — Isometric Mapping for nonlinear dimensionality reduction.
 *
 * Computes geodesic distances via a k-nearest-neighbor graph and
 * then applies classical MDS (multidimensional scaling) to embed
 * the data in a lower-dimensional space.
 *
 * @module ml/manifold/Isomap
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { DataValidationError, InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random } from "../../random/random";
import { validateUnsupervisedFitInputs } from "../_validation";

export class Isomap {
  private readonly nComponents: number;
  private readonly nNeighbors: number;
  private embedding_?: Tensor;
  private fitted = false;

  constructor(
    options: {
      readonly nComponents?: number;
      readonly nNeighbors?: number;
    } = {}
  ) {
    this.nComponents = options.nComponents ?? 2;
    this.nNeighbors = options.nNeighbors ?? 5;

    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new InvalidParameterError(
        "nComponents must be an integer >= 1",
        "nComponents",
        this.nComponents
      );
    }
    if (!Number.isInteger(this.nNeighbors) || this.nNeighbors < 1) {
      throw new InvalidParameterError(
        "nNeighbors must be an integer >= 1",
        "nNeighbors",
        this.nNeighbors
      );
    }
  }

  fit(X: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const n = X.shape[0] ?? 0;
    const nF = X.shape[1] ?? 0;

    if (this.nNeighbors >= n) {
      throw new InvalidParameterError(
        `nNeighbors must be < n_samples (${n})`,
        "nNeighbors",
        this.nNeighbors
      );
    }

    // Flatten X to a flat array for distance computation
    const flat = new Float64Array(n * nF);
    for (let i = 0; i < n * nF; i++) {
      flat[i] = Number(X.data[X.offset + i]);
    }

    // Step 1: Compute pairwise Euclidean distances
    const dist = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let sq = 0;
        for (let f = 0; f < nF; f++) {
          const diff = (flat[i * nF + f] ?? 0) - (flat[j * nF + f] ?? 0);
          sq += diff * diff;
        }
        const d = Math.sqrt(sq);
        dist[i * n + j] = d;
        dist[j * n + i] = d;
      }
    }

    // Step 2: Build k-NN adjacency graph
    const graph = new Float64Array(n * n).fill(Infinity);
    for (let i = 0; i < n; i++) {
      graph[i * n + i] = 0;
      // Find k nearest neighbors
      const neighbors: Array<{ idx: number; d: number }> = [];
      for (let j = 0; j < n; j++) {
        if (i !== j) neighbors.push({ idx: j, d: dist[i * n + j] ?? Infinity });
      }
      neighbors.sort((a, b) => a.d - b.d);
      for (let k = 0; k < this.nNeighbors; k++) {
        const nb = neighbors[k];
        if (nb) {
          graph[i * n + nb.idx] = nb.d;
          graph[nb.idx * n + i] = nb.d;
        }
      }
    }

    // Step 3: Shortest paths (Floyd-Warshall)
    for (let k = 0; k < n; k++) {
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const via = (graph[i * n + k] ?? Infinity) + (graph[k * n + j] ?? Infinity);
          if (via < (graph[i * n + j] ?? Infinity)) {
            graph[i * n + j] = via;
          }
        }
      }
    }

    // A disconnected k-NN graph leaves cross-component distances at Infinity,
    // which classical MDS would turn into NaN (Inf − Inf during double
    // centering), silently returning an all-NaN embedding. Detect it and fail
    // with an actionable message instead.
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (!Number.isFinite(graph[i * n + j] ?? Infinity)) {
          throw new DataValidationError(
            "Isomap: the neighborhood graph is disconnected. Increase nNeighbors so all " +
              "points are reachable, or fit on a single connected component."
          );
        }
      }
    }

    // Step 4: Classical MDS on geodesic distance matrix
    this.embedding_ = classicalMDS(graph, n, this.nComponents);
    this.fitted = true;
    return this;
  }

  fitTransform(X: Tensor): Tensor {
    this.fit(X);
    return this.embedding_!;
  }

  get embedding(): Tensor {
    if (!this.fitted || !this.embedding_) {
      throw new NotFittedError("Isomap must be fitted before accessing embedding");
    }
    return this.embedding_;
  }

  getParams(): Record<string, unknown> {
    return { nComponents: this.nComponents, nNeighbors: this.nNeighbors };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}

/**
 * Classical (metric) Multidimensional Scaling.
 *
 * Given a distance matrix, embeds points in `nComponents` dimensions
 * using the eigendecomposition of the double-centred squared-distance
 * matrix.
 */
export function classicalMDS(distMatrix: Float64Array, n: number, nComponents: number): Tensor {
  // Double-centre the squared-distance matrix: B = -0.5 * H D^2 H
  const D2 = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) {
    const d = distMatrix[i] ?? 0;
    D2[i] = d * d;
  }

  // Row means, column means, grand mean
  const rowMean = new Float64Array(n);
  const colMean = new Float64Array(n);
  let grandMean = 0;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) s += D2[i * n + j] ?? 0;
    rowMean[i] = s / n;
    grandMean += s;
  }
  grandMean /= n * n;
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += D2[i * n + j] ?? 0;
    colMean[j] = s / n;
  }

  const B = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      B[i * n + j] =
        -0.5 * ((D2[i * n + j] ?? 0) - (rowMean[i] ?? 0) - (colMean[j] ?? 0) + grandMean);
    }
  }

  // Power iteration for top nComponents eigenvectors of B
  const eigvecs = new Float64Array(n * nComponents);
  const eigvals = new Float64Array(nComponents);

  // Deflation-based power iteration
  const Bwork = new Float64Array(B);

  for (let comp = 0; comp < nComponents; comp++) {
    // Random init
    const v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = __random() - 0.5;

    // Normalise
    let norm = 0;
    for (let i = 0; i < n; i++) norm += (v[i] ?? 0) * (v[i] ?? 0);
    norm = Math.sqrt(norm);
    for (let i = 0; i < n; i++) v[i] = (v[i] ?? 0) / norm;

    // Iterate
    for (let iter = 0; iter < 300; iter++) {
      const w = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) {
          s += (Bwork[i * n + j] ?? 0) * (v[j] ?? 0);
        }
        w[i] = s;
      }
      let lambda = 0;
      for (let i = 0; i < n; i++) lambda += (w[i] ?? 0) * (v[i] ?? 0);

      norm = 0;
      for (let i = 0; i < n; i++) norm += (w[i] ?? 0) * (w[i] ?? 0);
      norm = Math.sqrt(norm);
      if (norm < 1e-15) break;

      let diff = 0;
      for (let i = 0; i < n; i++) {
        const newV = (w[i] ?? 0) / norm;
        diff += ((v[i] ?? 0) - newV) ** 2;
        v[i] = newV;
      }
      eigvals[comp] = lambda;
      if (Math.sqrt(diff) < 1e-10) break;
    }

    // Store eigenvector
    for (let i = 0; i < n; i++) eigvecs[i * nComponents + comp] = v[i] ?? 0;

    // Deflate: Bwork -= lambda * v * v^T
    const lambda = eigvals[comp] ?? 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        Bwork[i * n + j] = (Bwork[i * n + j] ?? 0) - lambda * (v[i] ?? 0) * (v[j] ?? 0);
      }
    }
  }

  // Embedding = eigvecs * diag(sqrt(max(eigvals, 0)))
  const result: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let d = 0; d < nComponents; d++) {
      const ev = eigvals[d] ?? 0;
      row.push((eigvecs[i * nComponents + d] ?? 0) * Math.sqrt(Math.max(ev, 0)));
    }
    result.push(row);
  }

  return tensor(result);
}
