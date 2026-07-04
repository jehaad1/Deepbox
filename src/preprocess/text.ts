/**
 * Text feature extraction utilities.
 *
 * Provides CountVectorizer and TfidfVectorizer for converting text
 * documents into numerical feature matrices, compatible with ML pipelines.
 *
 * @module preprocess/text
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox documentation}
 */

import { getDevice, getDtype, InvalidParameterError, NotFittedError } from "../core";
import { type Tensor, Tensor as TensorClass, tensor } from "../ndarray";

type TokenPattern = RegExp;

type CountVectorizerOptions = {
  /** Regex pattern for tokenization. Default: /\b\w\w+\b/g (words of 2+ chars) */
  readonly tokenPattern?: TokenPattern;
  /** Maximum number of features (vocabulary size). If set, keeps top-N by frequency. */
  readonly maxFeatures?: number;
  /** Minimum document frequency. Terms appearing in fewer docs are ignored.
   *  If float in [0,1), treated as proportion. If int >= 1, treated as count. */
  readonly minDf?: number;
  /** Maximum document frequency. Terms appearing in more docs are ignored.
   *  If float in [0,1], treated as proportion. If int > 1, treated as count. */
  readonly maxDf?: number;
  /** Whether to convert text to lowercase before tokenizing. Default: true */
  readonly lowercase?: boolean;
  /** If true, use binary occurrence instead of counts. Default: false */
  readonly binary?: boolean;
  /** Custom stop words to remove */
  readonly stopWords?: readonly string[];
  /** N-gram range [min, max]. Default: [1, 1] (unigrams only) */
  readonly ngramRange?: readonly [number, number];
};

type TfidfVectorizerOptions = CountVectorizerOptions & {
  /** TF-IDF norm: "l1", "l2", or undefined (no normalization). Default: "l2" */
  readonly norm?: "l1" | "l2" | undefined;
  /** Whether to apply sublinear TF scaling (1 + log(tf)). Default: false */
  readonly sublinearTf?: boolean;
  /** Smooth IDF by adding 1 to document frequencies. Default: true */
  readonly smoothIdf?: boolean;
};

function defaultTokenPattern(): RegExp {
  return /\b\w\w+\b/g;
}

function tokenize(
  doc: string,
  pattern: TokenPattern,
  lowercase: boolean,
  stopWords: ReadonlySet<string>,
  ngramRange: readonly [number, number]
): string[] {
  const text = lowercase ? doc.toLowerCase() : doc;
  // Reset regex lastIndex for global patterns
  pattern.lastIndex = 0;
  const rawTokens: string[] = [];
  let match: RegExpExecArray | null = pattern.exec(text);
  while (match !== null) {
    const token = match[0];
    if (!stopWords.has(token)) {
      rawTokens.push(token);
    }
    match = pattern.exec(text);
  }

  const [minN, maxN] = ngramRange;
  if (minN === 1 && maxN === 1) {
    return rawTokens;
  }

  const ngrams: string[] = [];
  for (let n = minN; n <= maxN; n++) {
    for (let i = 0; i <= rawTokens.length - n; i++) {
      ngrams.push(rawTokens.slice(i, i + n).join(" "));
    }
  }
  return ngrams;
}

function resolveDf(value: number, nDocs: number): number {
  if (value >= 0 && value < 1) {
    return Math.floor(value * nDocs);
  }
  return value;
}

/**
 * Convert a collection of text documents to a matrix of token counts.
 *
 * Implements the bag-of-words model: each document becomes a vector of
 * word counts. The vocabulary is learned from the training data.
 *
 * @example
 * ```ts
 * import { CountVectorizer } from 'deepbox/preprocess';
 *
 * const cv = new CountVectorizer();
 * const X = cv.fitTransformText(['hello world', 'hello deepbox world']);
 * // X is a 2D tensor of shape [2, vocabulary_size]
 * ```
 */
export class CountVectorizer {
  private readonly tokenPattern: TokenPattern;
  private readonly maxFeatures: number | undefined;
  private readonly minDf: number;
  private readonly maxDf: number;
  private readonly lowercase: boolean;
  private readonly binary: boolean;
  private readonly stopWords: ReadonlySet<string>;
  private readonly ngramRange: readonly [number, number];

  private vocabulary_: Map<string, number> = new Map();
  private featureNames_: string[] = [];
  private fitted = false;

  constructor(options: CountVectorizerOptions = {}) {
    this.tokenPattern = options.tokenPattern ?? defaultTokenPattern();
    this.maxFeatures = options.maxFeatures;
    this.minDf = options.minDf ?? 1;
    this.maxDf = options.maxDf ?? 1.0;
    this.lowercase = options.lowercase ?? true;
    this.binary = options.binary ?? false;
    this.stopWords = new Set(options.stopWords ?? []);
    this.ngramRange = options.ngramRange ?? [1, 1];

    if (
      this.maxFeatures !== undefined &&
      (!Number.isInteger(this.maxFeatures) || this.maxFeatures < 1)
    ) {
      throw new InvalidParameterError(
        `maxFeatures must be a positive integer; received ${this.maxFeatures}`,
        "maxFeatures",
        this.maxFeatures
      );
    }
    const [minN, maxN] = this.ngramRange;
    if (!Number.isInteger(minN) || !Number.isInteger(maxN) || minN < 1 || maxN < minN) {
      throw new InvalidParameterError(
        `ngramRange must be [min, max] with 1 <= min <= max; received [${minN}, ${maxN}]`,
        "ngramRange",
        this.ngramRange
      );
    }
  }

  /**
   * Learn vocabulary from documents.
   *
   * @param documents - Array of text documents
   * @returns this
   */
  fitText(documents: readonly string[]): this {
    const nDocs = documents.length;
    const termDocFreq = new Map<string, number>();
    const termTotalFreq = new Map<string, number>();

    for (const doc of documents) {
      const tokens = tokenize(
        doc,
        this.tokenPattern,
        this.lowercase,
        this.stopWords,
        this.ngramRange
      );
      const seen = new Set<string>();
      for (const token of tokens) {
        termTotalFreq.set(token, (termTotalFreq.get(token) ?? 0) + 1);
        if (!seen.has(token)) {
          termDocFreq.set(token, (termDocFreq.get(token) ?? 0) + 1);
          seen.add(token);
        }
      }
    }

    // Apply min_df / max_df filtering
    const minDfAbs = resolveDf(this.minDf, nDocs);
    const maxDfAbs =
      this.maxDf <= 1.0 && this.maxDf > 0 ? Math.ceil(this.maxDf * nDocs) : this.maxDf;

    const candidates: Array<[string, number]> = [];
    for (const [term, docFreq] of termDocFreq) {
      if (docFreq >= minDfAbs && docFreq <= maxDfAbs) {
        candidates.push([term, termTotalFreq.get(term) ?? 0]);
      }
    }

    // Sort by frequency descending, then alphabetically for ties
    candidates.sort((a, b) => {
      if (b[1] !== a[1]) return b[1] - a[1];
      return a[0].localeCompare(b[0]);
    });

    // Apply maxFeatures
    const selected =
      this.maxFeatures !== undefined ? candidates.slice(0, this.maxFeatures) : candidates;

    // Sort alphabetically for final vocabulary order (sklearn convention)
    selected.sort((a, b) => a[0].localeCompare(b[0]));

    this.vocabulary_ = new Map();
    this.featureNames_ = [];
    for (let i = 0; i < selected.length; i++) {
      const term = selected[i]![0];
      this.vocabulary_.set(term, i);
      this.featureNames_.push(term);
    }

    this.fitted = true;
    return this;
  }

  /**
   * Transform documents to a document-term matrix.
   *
   * @param documents - Array of text documents
   * @returns 2D Tensor of shape [n_documents, n_features]
   */
  transformText(documents: readonly string[]): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("CountVectorizer must be fitted before transform");
    }

    const nDocs = documents.length;
    const nFeatures = this.vocabulary_.size;
    const data = new Float64Array(nDocs * nFeatures);

    for (let i = 0; i < nDocs; i++) {
      const tokens = tokenize(
        documents[i]!,
        this.tokenPattern,
        this.lowercase,
        this.stopWords,
        this.ngramRange
      );
      for (const token of tokens) {
        const idx = this.vocabulary_.get(token);
        if (idx !== undefined) {
          if (this.binary) {
            data[i * nFeatures + idx] = 1;
          } else {
            const pos = i * nFeatures + idx;
            data[pos] = (data[pos] ?? 0) + 1;
          }
        }
      }
    }

    return tensor(Array.from(data)).reshape([nDocs, nFeatures]);
  }

  /**
   * Learn vocabulary and return document-term matrix.
   *
   * @param documents - Array of text documents
   * @returns 2D Tensor of shape [n_documents, n_features]
   */
  fitTransformText(documents: readonly string[]): Tensor {
    this.fitText(documents);
    return this.transformText(documents);
  }

  /** The learned vocabulary mapping term → index */
  get vocabulary(): ReadonlyMap<string, number> {
    if (!this.fitted) throw new NotFittedError("CountVectorizer must be fitted first");
    return this.vocabulary_;
  }

  /** Feature names (terms) in vocabulary order */
  getFeatureNames(): string[] {
    if (!this.fitted) throw new NotFittedError("CountVectorizer must be fitted first");
    return [...this.featureNames_];
  }

  getParams(): Record<string, unknown> {
    return {
      maxFeatures: this.maxFeatures,
      minDf: this.minDf,
      maxDf: this.maxDf,
      lowercase: this.lowercase,
      binary: this.binary,
      ngramRange: this.ngramRange,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}

/**
 * Convert a collection of text documents to a TF-IDF feature matrix.
 *
 * Combines CountVectorizer with TF-IDF weighting. TF-IDF (Term Frequency -
 * Inverse Document Frequency) down-weights terms that appear in many documents
 * and up-weights rare, discriminative terms.
 *
 * IDF formula: `log((1 + n) / (1 + df)) + 1` (with smoothIdf=true)
 *
 * @example
 * ```ts
 * import { TfidfVectorizer } from 'deepbox/preprocess';
 *
 * const tfidf = new TfidfVectorizer();
 * const X = tfidf.fitTransformText(['hello world', 'hello deepbox']);
 * // X is a 2D tensor of shape [2, vocabulary_size] with TF-IDF weights
 * ```
 */
export class TfidfVectorizer {
  private readonly countVectorizer: CountVectorizer;
  private readonly norm: "l1" | "l2" | undefined;
  private readonly sublinearTf: boolean;
  private readonly smoothIdf: boolean;

  private idf_: Float64Array = new Float64Array(0);
  private fitted = false;

  constructor(options: TfidfVectorizerOptions = {}) {
    // Build count vectorizer options, omitting undefined keys to satisfy exactOptionalPropertyTypes
    const raw: Record<string, unknown> = {};
    if (options.tokenPattern !== undefined) raw["tokenPattern"] = options.tokenPattern;
    if (options.maxFeatures !== undefined) raw["maxFeatures"] = options.maxFeatures;
    if (options.minDf !== undefined) raw["minDf"] = options.minDf;
    if (options.maxDf !== undefined) raw["maxDf"] = options.maxDf;
    if (options.lowercase !== undefined) raw["lowercase"] = options.lowercase;
    if (options.binary !== undefined) raw["binary"] = options.binary;
    if (options.stopWords !== undefined) raw["stopWords"] = options.stopWords;
    if (options.ngramRange !== undefined) raw["ngramRange"] = options.ngramRange;
    this.countVectorizer = new CountVectorizer(raw as CountVectorizerOptions);
    this.norm = "norm" in options ? options.norm : "l2";
    this.sublinearTf = options.sublinearTf ?? false;
    this.smoothIdf = options.smoothIdf ?? true;
  }

  /**
   * Learn vocabulary and IDF weights from documents.
   *
   * @param documents - Array of text documents
   * @returns this
   */
  fitText(documents: readonly string[]): this {
    // Fit the count vectorizer
    this.countVectorizer.fitText(documents);

    // Compute document frequencies for IDF
    const countMatrix = this.countVectorizer.transformText(documents);
    const nDocs = documents.length;
    const nFeatures = this.countVectorizer.vocabulary.size;

    const df = new Float64Array(nFeatures);
    for (let i = 0; i < nDocs; i++) {
      for (let j = 0; j < nFeatures; j++) {
        if (Number(countMatrix.data[countMatrix.offset + i * nFeatures + j]) > 0) {
          df[j] = (df[j] ?? 0) + 1;
        }
      }
    }

    // Compute IDF
    this.idf_ = new Float64Array(nFeatures);
    const smooth = this.smoothIdf ? 1 : 0;
    for (let j = 0; j < nFeatures; j++) {
      this.idf_[j] = Math.log((smooth + nDocs) / (smooth + (df[j] ?? 0))) + 1;
    }

    this.fitted = true;
    return this;
  }

  /**
   * Transform documents to TF-IDF weighted matrix.
   *
   * @param documents - Array of text documents
   * @returns 2D Tensor of shape [n_documents, n_features]
   */
  transformText(documents: readonly string[]): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("TfidfVectorizer must be fitted before transform");
    }

    const countMatrix = this.countVectorizer.transformText(documents);
    const nDocs = documents.length;
    const nFeatures = this.countVectorizer.vocabulary.size;
    const data = new Float64Array(nDocs * nFeatures);

    for (let i = 0; i < nDocs; i++) {
      for (let j = 0; j < nFeatures; j++) {
        let tf = Number(countMatrix.data[countMatrix.offset + i * nFeatures + j]);
        if (this.sublinearTf && tf > 0) {
          tf = 1 + Math.log(tf);
        }
        data[i * nFeatures + j] = tf * (this.idf_[j] ?? 0);
      }
    }

    // Apply normalization
    if (this.norm) {
      for (let i = 0; i < nDocs; i++) {
        const rowStart = i * nFeatures;
        let normVal = 0;
        if (this.norm === "l2") {
          for (let j = 0; j < nFeatures; j++) {
            normVal += (data[rowStart + j] ?? 0) ** 2;
          }
          normVal = Math.sqrt(normVal);
        } else {
          for (let j = 0; j < nFeatures; j++) {
            normVal += Math.abs(data[rowStart + j] ?? 0);
          }
        }
        if (normVal > 0) {
          for (let j = 0; j < nFeatures; j++) {
            data[rowStart + j] = (data[rowStart + j] ?? 0) / normVal;
          }
        }
      }
    }

    return tensor(Array.from(data)).reshape([nDocs, nFeatures]);
  }

  /**
   * Learn vocabulary/IDF and return TF-IDF matrix.
   *
   * @param documents - Array of text documents
   * @returns 2D Tensor of shape [n_documents, n_features]
   */
  fitTransformText(documents: readonly string[]): Tensor {
    this.fitText(documents);
    return this.transformText(documents);
  }

  /** The learned vocabulary mapping term → index */
  get vocabulary(): ReadonlyMap<string, number> {
    return this.countVectorizer.vocabulary;
  }

  /** The learned IDF vector */
  get idf(): Float64Array {
    if (!this.fitted) throw new NotFittedError("TfidfVectorizer must be fitted first");
    return this.idf_;
  }

  /** Feature names (terms) in vocabulary order */
  getFeatureNames(): string[] {
    return this.countVectorizer.getFeatureNames();
  }

  getParams(): Record<string, unknown> {
    return {
      ...this.countVectorizer.getParams(),
      norm: this.norm,
      sublinearTf: this.sublinearTf,
      smoothIdf: this.smoothIdf,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}

type HashingVectorizerOptions = {
  /** Number of features (hash buckets). Default: 2^20 = 1048576. */
  readonly nFeatures?: number;
  /** Regex pattern for tokenization. Default: /\b\w\w+\b/g */
  readonly tokenPattern?: TokenPattern;
  /** Whether to convert text to lowercase before tokenizing. Default: true */
  readonly lowercase?: boolean;
  /** Use binary occurrence instead of counts. Default: false */
  readonly binary?: boolean;
  /** Custom stop words to remove */
  readonly stopWords?: readonly string[];
  /** N-gram range [min, max]. Default: [1, 1] */
  readonly ngramRange?: readonly [number, number];
  /** Normalization: "l1", "l2", or undefined (none). Default: "l2" */
  readonly norm?: "l1" | "l2" | undefined;
  /** If true, use alternate sign to reduce hash collision bias. Default: true */
  readonly alternateSign?: boolean;
};

/**
 * FNV-1a 32-bit hash function.
 *
 * Deterministic, fast, and provides reasonable distribution for feature hashing.
 */
function fnv1a(str: string): number {
  let hash = 0x811c9dc5; // FNV offset basis
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193); // FNV prime
  }
  return hash >>> 0; // ensure unsigned
}

/**
 * Convert text documents to a fixed-size feature matrix using the hashing trick.
 *
 * Unlike CountVectorizer and TfidfVectorizer, HashingVectorizer does not need
 * to build a vocabulary, making it suitable for large-scale or streaming data.
 * Features are token hashes mapped to a fixed number of buckets.
 *
 * Trade-offs:
 * - No need to fit (stateless transform)
 * - Fixed memory usage regardless of vocabulary size
 * - Cannot retrieve feature names (hash collisions are possible)
 * - Signed hashing (alternateSign) reduces collision bias
 *
 * @example
 * ```ts
 * import { HashingVectorizer } from 'deepbox/preprocess';
 *
 * const hv = new HashingVectorizer({ nFeatures: 1024 });
 * const X = hv.transformText(['hello world', 'hello deepbox world']);
 * // X is a 2D tensor of shape [2, 1024]
 * ```
 */
export class HashingVectorizer {
  private readonly nFeatures: number;
  private readonly tokenPattern: TokenPattern;
  private readonly lowercase: boolean;
  private readonly binary: boolean;
  private readonly stopWords: ReadonlySet<string>;
  private readonly ngramRange: readonly [number, number];
  private readonly norm: "l1" | "l2" | undefined;
  private readonly alternateSign: boolean;

  constructor(options: HashingVectorizerOptions = {}) {
    this.nFeatures = options.nFeatures ?? 1 << 20;
    this.tokenPattern = options.tokenPattern ?? defaultTokenPattern();
    this.lowercase = options.lowercase ?? true;
    this.binary = options.binary ?? false;
    this.stopWords = new Set(options.stopWords ?? []);
    this.ngramRange = options.ngramRange ?? [1, 1];
    this.norm = "norm" in options ? options.norm : "l2";
    this.alternateSign = options.alternateSign ?? true;

    if (!Number.isInteger(this.nFeatures) || this.nFeatures < 1) {
      throw new InvalidParameterError(
        `nFeatures must be a positive integer; received ${this.nFeatures}`,
        "nFeatures",
        this.nFeatures
      );
    }
    const [minN, maxN] = this.ngramRange;
    if (!Number.isInteger(minN) || !Number.isInteger(maxN) || minN < 1 || maxN < minN) {
      throw new InvalidParameterError(
        `ngramRange must be [min, max] with 1 <= min <= max; received [${minN}, ${maxN}]`,
        "ngramRange",
        this.ngramRange
      );
    }
  }

  /**
   * Transform documents to a fixed-size hash feature matrix.
   *
   * No fitting is required — this is a stateless transform.
   *
   * @param documents - Array of text documents
   * @returns 2D Tensor of shape [n_documents, nFeatures]
   */
  transformText(documents: readonly string[]): Tensor {
    const nDocs = documents.length;
    const nF = this.nFeatures;
    const data = new Float64Array(nDocs * nF);
    const binary = this.binary;
    const alternateSign = this.alternateSign;
    const norm = this.norm;
    const hiMod = 2147483648 % nF;

    // Generation-stamped touch tracking: `touchStamp[idx] === i` marks column
    // `idx` as written by doc `i`, so normalization sweeps only the (few)
    // non-zero columns per document instead of all nF — the old code paid two
    // O(nDocs·nF) passes over a matrix that is ~99% zeros.
    const touchStamp = norm ? new Int32Array(nF).fill(-1) : null;
    const touched: number[] = [];

    for (let i = 0; i < nDocs; i++) {
      const doc = documents[i];
      if (doc === undefined) continue;
      const tokens = tokenize(
        doc,
        this.tokenPattern,
        this.lowercase,
        this.stopWords,
        this.ngramRange
      );

      const rowStart = i * nF;
      if (touchStamp) touched.length = 0;
      for (const token of tokens) {
        const hash = fnv1a(token);
        // Modulo in the SMI domain: `%` on uint32 values >= 2^31 takes
        // V8's slow float64 path.
        const idx =
          hash < 2147483648 ? (hash | 0) % nF : ((((hash - 2147483648) | 0) % nF) + hiMod) % nF;
        const pos = rowStart + idx;

        if (binary) {
          data[pos] = 1;
        } else if (alternateSign) {
          const sign = (hash & 0x80000000) !== 0 ? -1 : 1;
          data[pos] = (data[pos] as number) + sign;
        } else {
          data[pos] = (data[pos] as number) + 1;
        }
        if (touchStamp && touchStamp[idx] !== i) {
          touchStamp[idx] = i;
          touched.push(idx);
        }
      }

      if (norm) {
        let normVal = 0;
        if (norm === "l2") {
          for (let t = 0; t < touched.length; t++) {
            const v = data[rowStart + (touched[t] as number)] as number;
            normVal += v * v;
          }
          normVal = Math.sqrt(normVal);
        } else {
          for (let t = 0; t < touched.length; t++) {
            normVal += Math.abs(data[rowStart + (touched[t] as number)] as number);
          }
        }
        if (normVal > 0) {
          const inv = 1 / normVal;
          for (let t = 0; t < touched.length; t++) {
            const p = rowStart + (touched[t] as number);
            data[p] = (data[p] as number) * inv;
          }
        }
      }
    }

    // Wrap the buffer directly (round-tripping 1M values through
    // Array.from + tensor() revalidation dominated the runtime).
    const dtype = getDtype();
    if (dtype === "float64") {
      return TensorClass.fromTypedArray({
        data,
        shape: [nDocs, nF],
        dtype,
        device: getDevice(),
      });
    }
    if (dtype === "float32") {
      return TensorClass.fromTypedArray({
        data: new Float32Array(data),
        shape: [nDocs, nF],
        dtype,
        device: getDevice(),
      });
    }
    return tensor(Array.from(data)).reshape([nDocs, nF]);
  }

  /**
   * Alias for transformText (no fitting needed).
   *
   * @param documents - Array of text documents
   * @returns 2D Tensor of shape [n_documents, nFeatures]
   */
  fitTransformText(documents: readonly string[]): Tensor {
    return this.transformText(documents);
  }

  getParams(): Record<string, unknown> {
    return {
      nFeatures: this.nFeatures,
      lowercase: this.lowercase,
      binary: this.binary,
      ngramRange: this.ngramRange,
      norm: this.norm,
      alternateSign: this.alternateSign,
    };
  }

  setParams(_params: Record<string, unknown>): this {
    return this;
  }
}
