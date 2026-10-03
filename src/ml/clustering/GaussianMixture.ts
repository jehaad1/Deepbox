/**
 * @see {@link https://deepbox.dev/docs/ml-clustering | Deepbox documentation}
 */

import { ConvergenceError, InvalidParameterError, NotFittedError, warn } from "../../core";
import { type Tensor, tensor } from "../../ndarray";
import { __random, __randomBelow, __SeededRandom, __seedToUint64 } from "../../random/random";
import {
  toFloat64View,
  validatePredictInputs,
  validateUnsupervisedFitInputs,
} from "../_validation";
import type { Clusterer, EstimatorTags } from "../base";
import { KMeans } from "./KMeans";

type CovarianceType = "full" | "tied" | "diag" | "spherical";
type InitParams = "kmeans" | "random" | "randomFromData";

const COVARIANCE_TYPES: readonly CovarianceType[] = ["full", "tied", "diag", "spherical"];
const INIT_PARAMS: readonly InitParams[] = ["kmeans", "random", "randomFromData"];
const LOG_2PI = Math.log(2 * Math.PI);
/** Added to every component's soft count so that no component weight is exactly zero. */
const COUNT_FLOOR = 10 * 2.220446049250313e-16;

/** Fitted mixture parameters, stored in flat row-major buffers. */
type Model = {
  readonly nComponents: number;
  readonly covarianceType: CovarianceType;
  readonly weights: Float64Array; // [K]
  readonly means: Float64Array; // [K * d]
  // diag / spherical: variances [K * d] (spherical repeats one value per component)
  // full: covariance matrices [K * d * d]; tied: covariance matrix [d * d]
  readonly cov: Float64Array;
  // full: lower Cholesky factors [K * d * d]; tied: [d * d]; diag / spherical: empty
  readonly chol: Float64Array;
  // log |Sigma_k| per component [K]
  readonly logDet: Float64Array;
};

type EmRun = {
  readonly model: Model;
  readonly lowerBound: number;
  readonly nIter: number;
  readonly converged: boolean;
};

/**
 * In-place lower Cholesky factorization of the d-by-d symmetric matrix at `offset` of `a`.
 *
 * @returns False if the matrix is not positive definite
 */
function choleskyInPlace(a: Float64Array, offset: number, d: number): boolean {
  for (let i = 0; i < d; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = a[offset + i * d + j] as number;
      for (let k = 0; k < j; k++) {
        sum -= (a[offset + i * d + k] as number) * (a[offset + j * d + k] as number);
      }
      if (i === j) {
        if (!(sum > 0)) return false;
        a[offset + i * d + i] = Math.sqrt(sum);
      } else {
        a[offset + i * d + j] = sum / (a[offset + j * d + j] as number);
      }
    }
    for (let j = i + 1; j < d; j++) a[offset + i * d + j] = 0;
  }
  return true;
}

/**
 * Gaussian Mixture Model (GMM).
 *
 * Fits a mixture of Gaussian distributions using the Expectation-Maximization
 * (EM) algorithm. The covariance structure is chosen with `covarianceType`:
 * "diag" (the default, one variance per feature and component), "spherical"
 * (one variance per component), "full" (a full matrix per component) or
 * "tied" (one full matrix shared by all components).
 *
 * Unlike scikit-learn, whose default is "full", the default here stays "diag" so
 * that results of earlier Deepbox versions are unchanged.
 *
 * Iteration stops when the change of the mean log-likelihood per sample falls
 * below `tol`. The run with the best log-likelihood out of `nInit` starts is kept.
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
 *
 * @see {@link https://deepbox.dev/docs/ml-clustering | Deepbox Clustering}
 */
export class GaussianMixture implements Clusterer {
  private nComponents: number;
  private maxIter: number;
  private tol: number;
  private nInit: number;
  private regCovar: number;
  private randomState: number | undefined;
  private covarianceType: CovarianceType;
  private initParams: InitParams;

  private model_?: Model;
  private labels_?: Tensor;
  private nFeaturesIn_ = 0;
  private nIter_ = 0;
  private lowerBound_ = -Infinity;
  private converged_ = false;
  private fitted = false;

  /**
   * Create a Gaussian mixture model.
   *
   * @param options - Configuration options
   * @param options.nComponents - Number of mixture components, integer >= 1 (default: 1)
   * @param options.maxIter - Maximum EM iterations per start, integer >= 1 (default: 100)
   * @param options.tol - Convergence threshold on the change of the mean log-likelihood per sample, >= 0 (default: 1e-3)
   * @param options.nInit - Number of initializations; the best run is kept, integer >= 1 (default: 1)
   * @param options.regCovar - Non-negative value added to the diagonal of every covariance (default: 1e-6)
   * @param options.randomState - Seed for reproducible initialization
   * @param options.covarianceType - "full", "tied", "diag" or "spherical" (default: "diag")
   * @param options.initParams - Initialization of the responsibilities: "kmeans" (default), "random" or "randomFromData"
   * @throws {InvalidParameterError} If any option is out of range
   */
  constructor(
    options: {
      readonly nComponents?: number;
      readonly maxIter?: number;
      readonly tol?: number;
      readonly nInit?: number;
      readonly regCovar?: number;
      readonly randomState?: number;
      readonly covarianceType?: CovarianceType;
      readonly initParams?: InitParams;
    } = {}
  ) {
    this.nComponents = options.nComponents ?? 1;
    this.maxIter = options.maxIter ?? 100;
    this.tol = options.tol ?? 1e-3;
    this.nInit = options.nInit ?? 1;
    this.regCovar = options.regCovar ?? 1e-6;
    this.randomState = options.randomState;
    this.covarianceType = options.covarianceType ?? "diag";
    this.initParams = options.initParams ?? "kmeans";

    if (!Number.isInteger(this.nComponents) || this.nComponents < 1) {
      throw new InvalidParameterError(
        "nComponents must be an integer >= 1",
        "nComponents",
        this.nComponents
      );
    }
    if (!Number.isInteger(this.maxIter) || this.maxIter < 1) {
      throw new InvalidParameterError("maxIter must be an integer >= 1", "maxIter", this.maxIter);
    }
    if (!Number.isFinite(this.tol) || this.tol < 0) {
      throw new InvalidParameterError("tol must be a finite number >= 0", "tol", this.tol);
    }
    if (!Number.isInteger(this.nInit) || this.nInit < 1) {
      throw new InvalidParameterError("nInit must be an integer >= 1", "nInit", this.nInit);
    }
    if (!Number.isFinite(this.regCovar) || this.regCovar < 0) {
      throw new InvalidParameterError(
        "regCovar must be a finite number >= 0",
        "regCovar",
        this.regCovar
      );
    }
    if (this.randomState !== undefined && !Number.isFinite(this.randomState)) {
      throw new InvalidParameterError(
        "randomState must be a finite number",
        "randomState",
        this.randomState
      );
    }
    if (!COVARIANCE_TYPES.includes(this.covarianceType)) {
      throw new InvalidParameterError(
        `covarianceType must be one of ${COVARIANCE_TYPES.map((t) => `"${t}"`).join(", ")}`,
        "covarianceType",
        this.covarianceType
      );
    }
    if (!INIT_PARAMS.includes(this.initParams)) {
      throw new InvalidParameterError(
        `initParams must be one of ${INIT_PARAMS.map((t) => `"${t}"`).join(", ")}`,
        "initParams",
        this.initParams
      );
    }
  }

  /**
   * Estimator tags. A `scoreSamples` method would make tag inference report an
   * outlier detector, so the tags are declared explicitly.
   *
   * @internal
   */
  _getTags(): Partial<EstimatorTags> {
    return { estimatorType: "clusterer", requiresY: false, hasPredictProba: true };
  }

  /** Estimate soft counts, weights, means and covariances from responsibilities (M-step). */
  private mStep(data: Float64Array, n: number, d: number, resp: Float64Array): Model {
    const K = this.nComponents;
    const nk = new Float64Array(K).fill(COUNT_FLOOR);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < K; k++) nk[k] = (nk[k] as number) + (resp[i * K + k] as number);
    }

    const means = new Float64Array(K * d);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < K; k++) {
        const r = resp[i * K + k] as number;
        if (r === 0) continue;
        for (let j = 0; j < d; j++) {
          means[k * d + j] = (means[k * d + j] as number) + r * (data[i * d + j] as number);
        }
      }
    }
    for (let k = 0; k < K; k++) {
      for (let j = 0; j < d; j++)
        means[k * d + j] = (means[k * d + j] as number) / (nk[k] as number);
    }

    let total = 0;
    for (let k = 0; k < K; k++) total += nk[k] as number;
    const weights = new Float64Array(K);
    for (let k = 0; k < K; k++) weights[k] = (nk[k] as number) / total;

    const type = this.covarianceType;
    const reg = this.regCovar;
    const logDet = new Float64Array(K);

    if (type === "diag" || type === "spherical") {
      const cov = new Float64Array(K * d);
      for (let i = 0; i < n; i++) {
        for (let k = 0; k < K; k++) {
          const r = resp[i * K + k] as number;
          if (r === 0) continue;
          for (let j = 0; j < d; j++) {
            const diff = (data[i * d + j] as number) - (means[k * d + j] as number);
            cov[k * d + j] = (cov[k * d + j] as number) + r * diff * diff;
          }
        }
      }
      for (let k = 0; k < K; k++) {
        const base = k * d;
        if (type === "spherical") {
          let mean = 0;
          for (let j = 0; j < d; j++) mean += cov[base + j] as number;
          const v = mean / d / (nk[k] as number) + reg;
          for (let j = 0; j < d; j++) cov[base + j] = v;
        } else {
          for (let j = 0; j < d; j++) {
            cov[base + j] = (cov[base + j] as number) / (nk[k] as number) + reg;
          }
        }
        let ld = 0;
        for (let j = 0; j < d; j++) {
          const v = cov[base + j] as number;
          if (!(v > 0) || !Number.isFinite(v)) this.failIllDefined();
          ld += Math.log(v);
        }
        logDet[k] = ld;
      }
      return {
        nComponents: K,
        covarianceType: type,
        weights,
        means,
        cov,
        chol: new Float64Array(0),
        logDet,
      };
    }

    // full / tied: accumulate scatter matrices around each component mean.
    const nMat = type === "full" ? K : 1;
    const cov = new Float64Array(nMat * d * d);
    const diff = new Float64Array(d);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < K; k++) {
        const r = resp[i * K + k] as number;
        if (r === 0) continue;
        for (let j = 0; j < d; j++) {
          diff[j] = (data[i * d + j] as number) - (means[k * d + j] as number);
        }
        const base = (type === "full" ? k : 0) * d * d;
        for (let a = 0; a < d; a++) {
          const ra = r * (diff[a] as number);
          for (let b = 0; b <= a; b++) {
            cov[base + a * d + b] = (cov[base + a * d + b] as number) + ra * (diff[b] as number);
          }
        }
      }
    }
    const chol = new Float64Array(nMat * d * d);
    for (let m = 0; m < nMat; m++) {
      const base = m * d * d;
      const denom = type === "full" ? (nk[m] as number) : total;
      for (let a = 0; a < d; a++) {
        for (let b = 0; b <= a; b++) {
          const v = (cov[base + a * d + b] as number) / denom + (a === b ? reg : 0);
          cov[base + a * d + b] = v;
          cov[base + b * d + a] = v;
        }
      }
      for (let i = 0; i < d * d; i++) chol[base + i] = cov[base + i] as number;
      if (!choleskyInPlace(chol, base, d)) this.failIllDefined();
      let ld = 0;
      for (let a = 0; a < d; a++) ld += 2 * Math.log(chol[base + a * d + a] as number);
      if (type === "full") logDet[m] = ld;
      else logDet.fill(ld);
    }
    return { nComponents: K, covarianceType: type, weights, means, cov, chol, logDet };
  }

  private failIllDefined(): never {
    throw new ConvergenceError(
      "Fitting the mixture model failed because some components have an ill-defined empirical covariance " +
        "(for instance caused by singleton or collapsed samples). " +
        "Try to decrease the number of components, or increase regCovar."
    );
  }

  /**
   * Fill `out` (n x K) with log(weight_k) + log N(x_i | mean_k, covariance_k).
   */
  private weightedLogProb(model: Model, data: Float64Array, n: number, d: number): Float64Array {
    const K = model.nComponents;
    const out = new Float64Array(n * K);
    const type = model.covarianceType;
    const constant = d * LOG_2PI;

    for (let k = 0; k < K; k++) {
      const logW = Math.log(model.weights[k] as number);
      const mBase = k * d;
      if (type === "diag" || type === "spherical") {
        const ld = model.logDet[k] as number;
        for (let i = 0; i < n; i++) {
          let maha = 0;
          for (let j = 0; j < d; j++) {
            const diff = (data[i * d + j] as number) - (model.means[mBase + j] as number);
            maha += (diff * diff) / (model.cov[mBase + j] as number);
          }
          out[i * K + k] = logW - 0.5 * (constant + ld + maha);
        }
      } else {
        const cBase = (type === "full" ? k : 0) * d * d;
        const ld = model.logDet[k] as number;
        const y = new Float64Array(d);
        for (let i = 0; i < n; i++) {
          let maha = 0;
          for (let a = 0; a < d; a++) {
            let s = (data[i * d + a] as number) - (model.means[mBase + a] as number);
            for (let b = 0; b < a; b++)
              s -= (model.chol[cBase + a * d + b] as number) * (y[b] as number);
            s /= model.chol[cBase + a * d + a] as number;
            y[a] = s;
            maha += s * s;
          }
          out[i * K + k] = logW - 0.5 * (constant + ld + maha);
        }
      }
    }
    return out;
  }

  /**
   * Normalize rows of `wlp` in place into log responsibilities.
   *
   * @returns Per-sample log-likelihood log p(x_i)
   */
  private static normalizeRows(wlp: Float64Array, n: number, K: number): Float64Array {
    const lpn = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const base = i * K;
      let max = -Infinity;
      for (let k = 0; k < K; k++)
        if ((wlp[base + k] as number) > max) max = wlp[base + k] as number;
      let sum = 0;
      for (let k = 0; k < K; k++) sum += Math.exp((wlp[base + k] as number) - max);
      const lse = max + Math.log(sum);
      lpn[i] = lse;
      for (let k = 0; k < K; k++) wlp[base + k] = (wlp[base + k] as number) - lse;
    }
    return lpn;
  }

  /** Initial responsibilities for one start. */
  private initialResponsibilities(
    X: Tensor,
    n: number,
    rng: () => number,
    seeded: boolean
  ): Float64Array {
    const K = this.nComponents;
    const resp = new Float64Array(n * K);
    if (this.initParams === "kmeans") {
      const seed = seeded ? Math.floor(rng() * 4294967296) : undefined;
      const km = new KMeans({
        nClusters: K,
        nInit: 1,
        ...(seed === undefined ? {} : { randomState: seed }),
      }).fit(X);
      const labels = km.labels.data as Int32Array;
      const offset = km.labels.offset;
      for (let i = 0; i < n; i++) resp[i * K + (labels[offset + i] as number)] = 1;
    } else if (this.initParams === "random") {
      for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let k = 0; k < K; k++) {
          const v = rng();
          resp[i * K + k] = v;
          sum += v;
        }
        for (let k = 0; k < K; k++) resp[i * K + k] = (resp[i * K + k] as number) / sum;
      }
    } else {
      // randomFromData: K distinct samples seed the components; all other rows start with zero weight.
      const chosen = new Set<number>();
      for (let j = n - K; j < n; j++) {
        const t = Math.min(__randomBelow(rng, j + 1), j);
        chosen.add(chosen.has(t) ? j : t);
      }
      let k = 0;
      for (const idx of chosen) resp[idx * K + k++] = 1;
    }
    return resp;
  }

  /** One EM run from the given initial responsibilities. */
  private runEM(data: Float64Array, n: number, d: number, initResp: Float64Array): EmRun {
    const K = this.nComponents;
    let model = this.mStep(data, n, d, initResp);
    let lowerBound = -Infinity;
    let converged = false;
    let nIter = 0;
    const resp = new Float64Array(n * K);

    for (let iter = 1; iter <= this.maxIter; iter++) {
      nIter = iter;
      const prev = lowerBound;

      const logResp = this.weightedLogProb(model, data, n, d);
      const lpn = GaussianMixture.normalizeRows(logResp, n, K);
      let sum = 0;
      for (let i = 0; i < n; i++) sum += lpn[i] as number;
      lowerBound = sum / n;
      for (let i = 0; i < n * K; i++) resp[i] = Math.exp(logResp[i] as number);
      model = this.mStep(data, n, d, resp);

      if (Math.abs(lowerBound - prev) < this.tol) {
        converged = true;
        break;
      }
    }
    return { model, lowerBound, nIter, converged };
  }

  /**
   * Estimate the model parameters with the EM algorithm.
   *
   * Calling `fit` again replaces the previous model.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (exists for API compatibility)
   * @returns this - The fitted estimator
   * @throws {ShapeError} If X is not 2D
   * @throws {DataValidationError} If X is empty or contains NaN/Inf
   * @throws {InvalidParameterError} If there are fewer samples than components
   * @throws {ConvergenceError} If a component's covariance is not positive definite; decrease `nComponents` or increase `regCovar`
   */
  fit(X: Tensor, _y?: Tensor): this {
    validateUnsupervisedFitInputs(X);
    const n = X.shape[0] ?? 0;
    const d = X.shape[1] ?? 0;

    if (n < this.nComponents) {
      throw new InvalidParameterError(
        `n_samples=${n} should be >= n_components=${this.nComponents}`,
        "nComponents",
        this.nComponents
      );
    }

    const data = toFloat64View(X);
    const seeded = this.randomState !== undefined;
    let rng: () => number = __random;
    if (this.randomState !== undefined) {
      const gen = new __SeededRandom(__seedToUint64(this.randomState));
      rng = () => gen.next();
    }

    let best: EmRun | undefined;
    for (let run = 0; run < this.nInit; run++) {
      const init = this.initialResponsibilities(X, n, rng, seeded);
      const result = this.runEM(data, n, d, init);
      if (best === undefined || result.lowerBound > best.lowerBound) best = result;
    }
    const final = best as EmRun;

    if (!final.converged) {
      warn(
        "Best performing initialization did not converge. Try different init parameters, " +
          "or increase maxIter or tol, or check for degenerate data.",
        "ConvergenceWarning",
        "GaussianMixture"
      );
    }

    // Final E-step so that the stored labels match predict() on the training data.
    const logResp = this.weightedLogProb(final.model, data, n, d);
    const labels = GaussianMixture.argmaxRows(logResp, n, final.model.nComponents);

    this.model_ = final.model;
    this.nFeaturesIn_ = d;
    this.nIter_ = final.nIter;
    this.lowerBound_ = final.lowerBound;
    this.converged_ = final.converged;
    this.labels_ = tensor(labels);
    this.fitted = true;
    return this;
  }

  private static argmaxRows(values: Float64Array, n: number, K: number): Int32Array {
    const labels = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      let bestK = 0;
      let bestV = -Infinity;
      for (let k = 0; k < K; k++) {
        const v = values[i * K + k] as number;
        if (v > bestV) {
          bestV = v;
          bestK = k;
        }
      }
      labels[i] = bestK;
    }
    return labels;
  }

  private requireModel(what: string): Model {
    if (!this.fitted || !this.model_) {
      throw new NotFittedError(`GaussianMixture must be fitted before ${what}`);
    }
    return this.model_;
  }

  /**
   * Predict the most probable component for each sample.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Component labels of shape (n_samples,)
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X is not 2D or has a different number of features than the training data
   */
  predict(X: Tensor): Tensor {
    const model = this.requireModel("prediction");
    validatePredictInputs(X, this.nFeaturesIn_, "GaussianMixture");
    const n = X.shape[0] ?? 0;
    const wlp = this.weightedLogProb(model, toFloat64View(X), n, this.nFeaturesIn_);
    return tensor(GaussianMixture.argmaxRows(wlp, n, model.nComponents));
  }

  /**
   * Fit the model and return the component labels of the training samples.
   *
   * @param X - Training data of shape (n_samples, n_features)
   * @param _y - Ignored (exists for API compatibility)
   * @returns Component labels of shape (n_samples,)
   */
  fitPredict(X: Tensor, _y?: Tensor): Tensor {
    this.fit(X);
    return this.labels_ as Tensor;
  }

  /**
   * Posterior probability of each component for each sample.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Float64 probabilities of shape (n_samples, n_components); each row sums to 1
   * @throws {NotFittedError} If the model has not been fitted
   * @throws {ShapeError} If X is not 2D or has a different number of features than the training data
   */
  predictProba(X: Tensor): Tensor {
    const model = this.requireModel("prediction");
    validatePredictInputs(X, this.nFeaturesIn_, "GaussianMixture");
    const n = X.shape[0] ?? 0;
    const logResp = this.weightedLogProb(model, toFloat64View(X), n, this.nFeaturesIn_);
    GaussianMixture.normalizeRows(logResp, n, model.nComponents);
    for (let i = 0; i < logResp.length; i++) logResp[i] = Math.exp(logResp[i] as number);
    return tensor(logResp).reshape([n, model.nComponents]);
  }

  /**
   * Log-likelihood of each sample under the model.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Float64 tensor of shape (n_samples,) with log p(x)
   * @throws {NotFittedError} If the model has not been fitted
   */
  scoreSamples(X: Tensor): Tensor {
    const model = this.requireModel("scoring");
    validatePredictInputs(X, this.nFeaturesIn_, "GaussianMixture");
    const n = X.shape[0] ?? 0;
    const wlp = this.weightedLogProb(model, toFloat64View(X), n, this.nFeaturesIn_);
    return tensor(GaussianMixture.normalizeRows(wlp, n, model.nComponents));
  }

  /**
   * Mean log-likelihood of the samples under the model.
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns Average of log p(x) over the samples
   * @throws {NotFittedError} If the model has not been fitted
   */
  score(X: Tensor): number {
    const perSample = this.scoreSamples(X);
    const values = perSample.data as Float64Array;
    let sum = 0;
    for (let i = 0; i < perSample.size; i++) sum += values[perSample.offset + i] as number;
    return sum / perSample.size;
  }

  /** Number of free parameters of the fitted model. */
  private nParameters(model: Model): number {
    const K = model.nComponents;
    const d = this.nFeaturesIn_;
    let covParams: number;
    switch (model.covarianceType) {
      case "full":
        covParams = (K * d * (d + 1)) / 2;
        break;
      case "tied":
        covParams = (d * (d + 1)) / 2;
        break;
      case "diag":
        covParams = K * d;
        break;
      default:
        covParams = K;
    }
    return covParams + d * K + K - 1;
  }

  /**
   * Bayesian information criterion of the model on X (lower is better).
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns -2 * logL + nParameters * ln(n_samples)
   * @throws {NotFittedError} If the model has not been fitted
   */
  bic(X: Tensor): number {
    const n = X.shape[0] ?? 0;
    return -2 * this.score(X) * n + this.nParameters(this.requireModel("scoring")) * Math.log(n);
  }

  /**
   * Akaike information criterion of the model on X (lower is better).
   *
   * @param X - Samples of shape (n_samples, n_features)
   * @returns -2 * logL + 2 * nParameters
   * @throws {NotFittedError} If the model has not been fitted
   */
  aic(X: Tensor): number {
    const n = X.shape[0] ?? 0;
    return -2 * this.score(X) * n + 2 * this.nParameters(this.requireModel("scoring"));
  }

  /**
   * Component labels of the training samples.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get labels(): Tensor {
    if (!this.fitted || !this.labels_) {
      throw new NotFittedError("GaussianMixture must be fitted to access labels");
    }
    return this.labels_;
  }

  /**
   * Component means.
   *
   * @returns Float64 tensor of shape (n_components, n_features)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get clusterCenters(): Tensor {
    return this.means;
  }

  /**
   * Component means (same as `clusterCenters`).
   *
   * @returns Float64 tensor of shape (n_components, n_features)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get means(): Tensor {
    if (!this.fitted || !this.model_) {
      throw new NotFittedError("GaussianMixture must be fitted to access cluster centers");
    }
    return tensor(this.model_.means.slice()).reshape([this.model_.nComponents, this.nFeaturesIn_]);
  }

  /**
   * Mixing weights of the components; they sum to 1.
   *
   * @returns Float64 tensor of shape (n_components,)
   * @throws {NotFittedError} If the model has not been fitted
   */
  get weights(): Tensor {
    if (!this.fitted || !this.model_) {
      throw new NotFittedError("GaussianMixture must be fitted to access weights");
    }
    return tensor(this.model_.weights.slice());
  }

  /**
   * Covariances of the components. The shape depends on `covarianceType`:
   * "full" (n_components, n_features, n_features), "tied" (n_features, n_features),
   * "diag" (n_components, n_features), "spherical" (n_components,).
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get covariances(): Tensor {
    if (!this.fitted || !this.model_) {
      throw new NotFittedError("GaussianMixture must be fitted to access covariances");
    }
    const K = this.model_.nComponents;
    const d = this.nFeaturesIn_;
    const cov = this.model_.cov;
    switch (this.model_.covarianceType) {
      case "full":
        return tensor(cov.slice()).reshape([K, d, d]);
      case "tied":
        return tensor(cov.slice()).reshape([d, d]);
      case "diag":
        return tensor(cov.slice()).reshape([K, d]);
      default: {
        const v = new Float64Array(K);
        for (let k = 0; k < K; k++) v[k] = cov[k * d] as number;
        return tensor(v);
      }
    }
  }

  /**
   * Whether EM reached the `tol` threshold in the best run.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get converged(): boolean {
    this.requireModel("reading converged");
    return this.converged_;
  }

  /**
   * Number of EM iterations of the best run.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get nIter(): number {
    this.requireModel("reading nIter");
    return this.nIter_;
  }

  /**
   * Mean log-likelihood per sample reached by the best run, evaluated before its last M-step.
   *
   * @throws {NotFittedError} If the model has not been fitted
   */
  get lowerBound(): number {
    this.requireModel("reading lowerBound");
    return this.lowerBound_;
  }

  /**
   * Get hyperparameters for this estimator.
   *
   * @returns Object containing all hyperparameters
   */
  getParams(): Record<string, unknown> {
    return {
      nComponents: this.nComponents,
      maxIter: this.maxIter,
      tol: this.tol,
      nInit: this.nInit,
      regCovar: this.regCovar,
      randomState: this.randomState,
      covarianceType: this.covarianceType,
      initParams: this.initParams,
    };
  }

  /**
   * Set the parameters of this estimator.
   *
   * @param params - Parameters to set (nComponents, maxIter, tol, nInit, regCovar, randomState, covarianceType, initParams)
   * @returns this
   * @throws {InvalidParameterError} If any parameter value is invalid
   */
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
          if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
            throw new InvalidParameterError("tol must be a finite number >= 0", "tol", value);
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
          if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
            throw new InvalidParameterError(
              "regCovar must be a finite number >= 0",
              "regCovar",
              value
            );
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
        case "covarianceType":
          if (!COVARIANCE_TYPES.includes(value as CovarianceType)) {
            throw new InvalidParameterError(
              `covarianceType must be one of ${COVARIANCE_TYPES.map((t) => `"${t}"`).join(", ")}`,
              "covarianceType",
              value
            );
          }
          this.covarianceType = value as CovarianceType;
          break;
        case "initParams":
          if (!INIT_PARAMS.includes(value as InitParams)) {
            throw new InvalidParameterError(
              `initParams must be one of ${INIT_PARAMS.map((t) => `"${t}"`).join(", ")}`,
              "initParams",
              value
            );
          }
          this.initParams = value as InitParams;
          break;
        default:
          throw new InvalidParameterError(`Unknown parameter: ${key}`, key, value);
      }
    }
    return this;
  }
}
