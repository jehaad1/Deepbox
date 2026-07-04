/**
 * @see {@link https://deepbox.dev/docs/ml-linear | Deepbox documentation}
 */

import { InvalidParameterError, NotFittedError } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random } from "../../random/random";
import { validatePredictInputs, validateUnsupervisedFitInputs } from "../_validation";
import type { Clusterer } from "../base";

/**
 * Gaussian Mixture Model (GMM).
 *
 * Fits a mixture of Gaussian distributions using the Expectation-Maximization
 * (EM) algorithm. Each component is a multivariate Gaussian with diagonal covariance.
 *
 * @example
 * ```ts
 * import { GaussianMixture } from 'deepbox/ml';
 * import { tensor } from 'deepbox/ndarray';
 *
 * const X = tensor([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]);
 * const gmm = new GaussianMixture({ nComponents: 2 });
 * gmm.fit(X);
 * const labels = gmm.predict(X);
 * ```
 */
export class GaussianMixture implements Clusterer {
  private nComponents: number;
  private maxIter: number;
  private tol: number;
  private nInit: number;
  private regCovar: number;
  private randomState: number | undefined;

  private weights_?: number[]; // mixing weights [nComponents]
  private means_?: number[][]; // means [nComponents][nFeatures]
  private covariances_?: number[][]; // diagonal covariances [nComponents][nFeatures]
  private labels_?: Tensor;
  private nFeaturesIn_?: number;
  private fitted = false;

  constructor(
    options: {
      readonly nComponents?: number;
      readonly maxIter?: number;
      readonly tol?: number;
      readonly nInit?: number;
      readonly regCovar?: number;
      readonly randomState?: number;
    } = {}
  ) {
    this.nComponents = options.nComponents ?? 1;
    this.maxIter = options.maxIter ?? 100;
    this.tol = options.tol ?? 1e-3;
    this.nInit = options.nInit ?? 1;
    this.regCovar = options.regCovar ?? 1e-6;
    this.randomState = options.randomState;

    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new InvalidParameterError(
        "nComponents must be an integer >= 1",
        "nComponents",
        this.nComponents
      );
    }
  }

  private createRNG(seed?: number): () => number {
    if (seed !== undefined) {
      let s = seed;
      return () => {
        s = (s * 9301 + 49297) % 233280;
        return s / 233280;
      };
    }
    return __random;
  }

  /**
   * Compute log probability of x under a diagonal Gaussian.
   */
  private logGaussian(x: number[], mean: number[], variance: number[], nFeatures: number): number {
    let logP = -0.5 * nFeatures * Math.log(2 * Math.PI);
    for (let j = 0; j < nFeatures; j++) {
      const v = variance[j] ?? 1e-6;
      const diff = (x[j] ?? 0) - (mean[j] ?? 0);
      logP -= 0.5 * Math.log(v) + (0.5 * (diff * diff)) / v;
    }
    return logP;
  }

  private runEM(
    data: number[][],
    nSamples: number,
    nFeatures: number,
    rng: () => number
  ): {
    weights: number[];
    means: number[][];
    covariances: number[][];
    logLikelihood: number;
  } {
    const K = this.nComponents;

    // Initialize means by picking random data points
    const indices = new Set<number>();
    while (indices.size < Math.min(K, nSamples)) {
      indices.add(Math.floor(rng() * nSamples));
    }
    const means: number[][] = [...indices].map((i) => [...data[i]!]);
    // Pad if needed
    while (means.length < K) {
      means.push([...data[Math.floor(rng() * nSamples)]!]);
    }

    // Initialize covariances to data variance
    const globalMean = new Array<number>(nFeatures).fill(0);
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        globalMean[j] = (globalMean[j] ?? 0) + (data[i]![j] ?? 0);
      }
    }
    for (let j = 0; j < nFeatures; j++) {
      globalMean[j] = (globalMean[j] ?? 0) / nSamples;
    }
    const globalVar = new Array<number>(nFeatures).fill(0);
    for (let i = 0; i < nSamples; i++) {
      for (let j = 0; j < nFeatures; j++) {
        const d = (data[i]![j] ?? 0) - (globalMean[j] ?? 0);
        globalVar[j] = (globalVar[j] ?? 0) + d * d;
      }
    }
    for (let j = 0; j < nFeatures; j++) {
      globalVar[j] = Math.max((globalVar[j] ?? 0) / nSamples, this.regCovar);
    }

    const covariances: number[][] = [];
    for (let k = 0; k < K; k++) {
      covariances.push([...globalVar]);
    }

    // Initialize weights uniformly
    const weights = new Array<number>(K).fill(1 / K);

    // Responsibilities matrix [nSamples][K]
    const resp: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      resp.push(new Array<number>(K).fill(0));
    }

    let prevLL = -Infinity;

    for (let iter = 0; iter < this.maxIter; iter++) {
      // E-step: compute responsibilities
      for (let i = 0; i < nSamples; i++) {
        const logProbs: number[] = [];
        for (let k = 0; k < K; k++) {
          logProbs.push(
            Math.log(Math.max(weights[k] ?? 1e-300, 1e-300)) +
              this.logGaussian(data[i]!, means[k]!, covariances[k]!, nFeatures)
          );
        }
        // Log-sum-exp for normalization
        const maxLP = Math.max(...logProbs);
        const exps = logProbs.map((lp) => Math.exp(lp - maxLP));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        const row = resp[i]!;
        for (let k = 0; k < K; k++) {
          row[k] = (exps[k] ?? 0) / sumExps;
        }
      }

      // M-step: update weights, means, covariances
      for (let k = 0; k < K; k++) {
        let nk = 0;
        for (let i = 0; i < nSamples; i++) {
          nk += resp[i]![k] ?? 0;
        }
        nk = Math.max(nk, 1e-10);

        weights[k] = nk / nSamples;

        // Update mean
        const newMean = new Array<number>(nFeatures).fill(0);
        for (let i = 0; i < nSamples; i++) {
          const r = resp[i]![k] ?? 0;
          for (let j = 0; j < nFeatures; j++) {
            newMean[j] = (newMean[j] ?? 0) + r * (data[i]![j] ?? 0);
          }
        }
        for (let j = 0; j < nFeatures; j++) {
          newMean[j] = (newMean[j] ?? 0) / nk;
        }
        means[k] = newMean;

        // Update covariance (diagonal)
        const newCov = new Array<number>(nFeatures).fill(0);
        for (let i = 0; i < nSamples; i++) {
          const r = resp[i]![k] ?? 0;
          for (let j = 0; j < nFeatures; j++) {
            const d = (data[i]![j] ?? 0) - (newMean[j] ?? 0);
            newCov[j] = (newCov[j] ?? 0) + r * d * d;
          }
        }
        for (let j = 0; j < nFeatures; j++) {
          newCov[j] = Math.max((newCov[j] ?? 0) / nk, this.regCovar);
        }
        covariances[k] = newCov;
      }

      // Compute log-likelihood
      let ll = 0;
      for (let i = 0; i < nSamples; i++) {
        const logProbs: number[] = [];
        for (let k = 0; k < K; k++) {
          logProbs.push(
            Math.log(Math.max(weights[k] ?? 1e-300, 1e-300)) +
              this.logGaussian(data[i]!, means[k]!, covariances[k]!, nFeatures)
          );
        }
        const maxLP = Math.max(...logProbs);
        const lse =
          maxLP + Math.log(logProbs.map((lp) => Math.exp(lp - maxLP)).reduce((a, b) => a + b, 0));
        ll += lse;
      }

      if (Math.abs(ll - prevLL) < this.tol) break;
      prevLL = ll;
    }

    return { weights, means, covariances, logLikelihood: prevLL };
  }

  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    this.nFeaturesIn_ = nFeatures;

    if (nSamples < this.nComponents) {
      throw new InvalidParameterError(
        `n_samples=${nSamples} should be >= n_components=${this.nComponents}`,
        "nComponents",
        this.nComponents
      );
    }

    const data: number[][] = [];
    for (let i = 0; i < nSamples; i++) {
      const row: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        row.push(Number(X.data[X.offset + i * nFeatures + j]));
      }
      data.push(row);
    }

    let bestResult: ReturnType<typeof this.runEM> | undefined;
    let bestLL = -Infinity;

    for (let run = 0; run < this.nInit; run++) {
      const rng = this.createRNG(
        this.randomState !== undefined ? this.randomState + run * 7919 : undefined
      );
      const result = this.runEM(data, nSamples, nFeatures, rng);
      if (result.logLikelihood > bestLL) {
        bestLL = result.logLikelihood;
        bestResult = result;
      }
    }

    this.weights_ = bestResult!.weights;
    this.means_ = bestResult!.means;
    this.covariances_ = bestResult!.covariances;

    // Assign labels (argmax of responsibilities)
    const labels: number[] = [];
    for (let i = 0; i < nSamples; i++) {
      let bestK = 0;
      let bestP = -Infinity;
      for (let k = 0; k < this.nComponents; k++) {
        const lp =
          Math.log(Math.max(this.weights_[k] ?? 1e-300, 1e-300)) +
          this.logGaussian(data[i]!, this.means_[k]!, this.covariances_[k]!, nFeatures);
        if (lp > bestP) {
          bestP = lp;
          bestK = k;
        }
      }
      labels.push(bestK);
    }

    this.labels_ = tensor(labels, { dtype: "int32" });
    this.fitted = true;
    return this;
  }

  predict(X: Tensor): Tensor {
    if (!this.fitted || !this.weights_ || !this.means_ || !this.covariances_) {
      throw new NotFittedError("GaussianMixture must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "GaussianMixture");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const labels: number[] = [];

    for (let i = 0; i < nSamples; i++) {
      const xi: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        xi.push(Number(X.data[X.offset + i * nFeatures + j]));
      }

      let bestK = 0;
      let bestP = -Infinity;
      for (let k = 0; k < this.nComponents; k++) {
        const lp =
          Math.log(Math.max(this.weights_[k] ?? 1e-300, 1e-300)) +
          this.logGaussian(xi, this.means_[k]!, this.covariances_[k]!, nFeatures);
        if (lp > bestP) {
          bestP = lp;
          bestK = k;
        }
      }
      labels.push(bestK);
    }

    return tensor(labels, { dtype: "int32" });
  }

  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.labels_!;
  }

  /**
   * Predict posterior probabilities for each component.
   */
  predictProba(X: Tensor): Tensor {
    if (!this.fitted || !this.weights_ || !this.means_ || !this.covariances_) {
      throw new NotFittedError("GaussianMixture must be fitted before prediction");
    }
    validatePredictInputs(X, this.nFeaturesIn_ ?? 0, "GaussianMixture");

    const nSamples = X.shape[0] ?? 0;
    const nFeatures = X.shape[1] ?? 0;
    const K = this.nComponents;
    const proba: number[][] = [];

    for (let i = 0; i < nSamples; i++) {
      const xi: number[] = [];
      for (let j = 0; j < nFeatures; j++) {
        xi.push(Number(X.data[X.offset + i * nFeatures + j]));
      }

      const logProbs: number[] = [];
      for (let k = 0; k < K; k++) {
        logProbs.push(
          Math.log(Math.max(this.weights_[k] ?? 1e-300, 1e-300)) +
            this.logGaussian(xi, this.means_[k]!, this.covariances_[k]!, nFeatures)
        );
      }
      const maxLP = Math.max(...logProbs);
      const exps = logProbs.map((lp) => Math.exp(lp - maxLP));
      const sumExps = exps.reduce((a, b) => a + b, 0);
      proba.push(exps.map((e) => e / sumExps));
    }

    return tensor(proba);
  }

  get labels(): Tensor {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("GaussianMixture must be fitted to access labels");
    }
    return this.labels_;
  }

  get clusterCenters(): Tensor {
    if (!this.fitted || !this.means_) {
      throw new NotFittedError("GaussianMixture must be fitted to access cluster centers");
    }
    return tensor(this.means_);
  }

  getParams(): Record<string, unknown> {
    return {
      nComponents: this.nComponents,
      maxIter: this.maxIter,
      tol: this.tol,
      nInit: this.nInit,
      regCovar: this.regCovar,
      randomState: this.randomState,
    };
  }

  setParams(params: Record<string, unknown>): this {
    for (const [key, value] of Object.entries(params)) {
      switch (key) {
        case "nComponents":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError(
              "nComponents must be an integer >= 1",
              "nComponents",
              value
            );
          }
          this.nComponents = value;
          break;
        case "maxIter":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("maxIter must be an integer >= 1", "maxIter", value);
          }
          this.maxIter = value;
          break;
        case "tol":
          if (typeof value !== "number" || value < 0) {
            throw new InvalidParameterError("tol must be >= 0", "tol", value);
          }
          this.tol = value;
          break;
        case "nInit":
          if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
            throw new InvalidParameterError("nInit must be an integer >= 1", "nInit", value);
          }
          this.nInit = value;
          break;
        case "regCovar":
          if (typeof value !== "number" || value < 0) {
            throw new InvalidParameterError("regCovar must be >= 0", "regCovar", value);
          }
          this.regCovar = value;
          break;
        case "randomState":
          if (value !== undefined && (typeof value !== "number" || !Number.isFinite(value))) {
            throw new InvalidParameterError(
              "randomState must be a finite number",
              "randomState",
              value
            );
          }
          this.randomState = value;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
