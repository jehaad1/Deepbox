/**
 * Text feature extraction utilities.
 *
 * Provides CountVectorizer, TfidfVectorizer and HashingVectorizer for converting
 * text documents into numerical feature matrices, compatible with ML pipelines.
 *
 * @module preprocess/text
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox documentation}
 */

import {
  DataValidationError,
  getDevice,
  getDtype,
  InvalidParameterError,
  MemoryError,
  NotFittedError,
} from "../core";
import { resolveLosslessDType } from "../core/utils/dtype_utils";
import { type Tensor, Tensor as TensorClass, tensor } from "../ndarray";

type TokenPattern = RegExp;

/** Options shared by every vectorizer. */
type TextOptions = {
  /**
   * Regex that matches one token. Every match becomes a token (the whole match, not a capture
   * group). A `g` flag is added when missing, and zero-length matches are ignored.
   * Default: `/[\p{L}\p{N}_]{2,}/gu`, i.e. runs of two or more Unicode letters, digits or
   * underscores (the Unicode equivalent of scikit-learn's `\b\w\w+\b`).
   */
  readonly tokenPattern?: TokenPattern;
  /** Whether to convert text to lowercase before tokenizing. Default: true */
  readonly lowercase?: boolean;
  /** If true, use binary occurrence instead of counts. Default: false */
  readonly binary?: boolean;
  /**
   * Tokens to drop before n-grams are built: a custom list, or `"english"` for scikit-learn's
   * built-in English list. Matching is case-sensitive and happens after lowercasing.
   */
  readonly stopWords?: readonly string[] | "english";
  /** N-gram range `[min, max]` (1 <= min <= max). Default: `[1, 1]` (unigrams only). */
  readonly ngramRange?: readonly [number, number];
};

/** Options accepted by {@link CountVectorizer}. */
export type CountVectorizerOptions = TextOptions & {
  /** Maximum number of features (vocabulary size). If set, keeps the most frequent terms. */
  readonly maxFeatures?: number;
  /**
   * Minimum document frequency. A value below 1 is a proportion of the documents, a value of 1
   * or more is an absolute number of documents. Terms in fewer documents are dropped.
   * Default: 1.
   */
  readonly minDf?: number;
  /**
   * Maximum document frequency. A value in (0, 1] is a proportion of the documents, an integer
   * above 1 is an absolute number of documents. Terms in more documents are dropped.
   * Default: 1 (keep everything).
   */
  readonly maxDf?: number;
};

/** Options accepted by {@link TfidfVectorizer}. */
export type TfidfVectorizerOptions = CountVectorizerOptions & {
  /** Row normalization: "l1", "l2", or undefined for none. Default: "l2" */
  readonly norm?: "l1" | "l2" | undefined;
  /** Whether to apply sublinear TF scaling (1 + log(tf)). Default: false */
  readonly sublinearTf?: boolean;
  /** Smooth IDF by adding 1 to document frequencies. Default: true */
  readonly smoothIdf?: boolean;
  /** Enable inverse-document-frequency weighting. With false, only TF is used. Default: true */
  readonly useIdf?: boolean;
};

/** Options accepted by {@link HashingVectorizer}. */
export type HashingVectorizerOptions = TextOptions & {
  /** Number of features (hash buckets). Default: 2^20 = 1048576. */
  readonly nFeatures?: number;
  /** Row normalization: "l1", "l2", or undefined for none. Default: "l2" */
  readonly norm?: "l1" | "l2" | undefined;
  /** If true, use alternate sign to reduce hash collision bias. Default: true */
  readonly alternateSign?: boolean;
};

/** scikit-learn's `ENGLISH_STOP_WORDS` (318 words). */
const ENGLISH_STOP_WORDS: readonly string[] = (
  "a about above across after afterwards again against all almost alone along already also " +
  "although always am among amongst amoungst amount an and another any anyhow anyone anything " +
  "anyway anywhere are around as at back be became because become becomes becoming been before " +
  "beforehand behind being below beside besides between beyond bill both bottom but by call can " +
  "cannot cant co con could couldnt cry de describe detail do done down due during each eg " +
  "eight either eleven else elsewhere empty enough etc even ever every everyone everything " +
  "everywhere except few fifteen fifty fill find fire first five for former formerly forty " +
  "found four from front full further get give go had has hasnt have he hence her here " +
  "hereafter hereby herein hereupon hers herself him himself his how however hundred i ie if in " +
  "inc indeed interest into is it its itself keep last latter latterly least less ltd made " +
  "many may me meanwhile might mill mine more moreover most mostly move much must my myself " +
  "name namely neither never nevertheless next nine no nobody none noone nor not nothing now " +
  "nowhere of off often on once one only onto or other others otherwise our ours ourselves out " +
  "over own part per perhaps please put rather re same see seem seemed seeming seems serious " +
  "several she should show side since sincere six sixty so some somehow someone something " +
  "sometime sometimes somewhere still such system take ten than that the their them themselves " +
  "then thence there thereafter thereby therefore therein thereupon these they thick thin third " +
  "this those though three through throughout thru thus to together too top toward towards " +
  "twelve twenty two un under until up upon us very via was we well were what whatever when " +
  "whence whenever where whereafter whereas whereby wherein whereupon wherever whether which " +
  "while whither who whoever whole whom whose why will with within without would yet you your " +
  "yours yourself yourselves"
).split(" ");

function defaultTokenPattern(): RegExp {
  return /[\p{L}\p{N}_]{2,}/gu;
}

/** Compare strings by UTF-16 code units (locale independent, like Python's `sorted`). */
function compareStrings(a: string, b: string): number {
  if (a < b) return -1;
  return a > b ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Option validation
// ---------------------------------------------------------------------------

type ResolvedTextOptions = {
  readonly tokenPattern: RegExp;
  readonly lowercase: boolean;
  readonly binary: boolean;
  readonly stopWords: ReadonlySet<string>;
  readonly stopWordsSpec: readonly string[] | "english" | undefined;
  readonly ngramRange: readonly [number, number];
};

type ResolvedCountOptions = ResolvedTextOptions & {
  readonly maxFeatures: number | undefined;
  readonly minDf: number;
  readonly maxDf: number;
};

type OptionBag = Readonly<Record<string, unknown>>;

const TEXT_PARAM_KEYS = ["tokenPattern", "lowercase", "binary", "stopWords", "ngramRange"] as const;
const COUNT_PARAM_KEYS: readonly string[] = [...TEXT_PARAM_KEYS, "maxFeatures", "minDf", "maxDf"];
const TFIDF_OWN_KEYS: readonly string[] = ["norm", "sublinearTf", "smoothIdf", "useIdf"];
const HASHING_PARAM_KEYS: readonly string[] = [
  ...TEXT_PARAM_KEYS,
  "nFeatures",
  "norm",
  "alternateSign",
];

function resolveBoolean(value: unknown, name: string, fallback: boolean): boolean {
  if (value === undefined) return fallback;
  if (typeof value !== "boolean") {
    throw new InvalidParameterError(
      `${name} must be a boolean; received ${String(value)}`,
      name,
      value
    );
  }
  return value;
}

function resolveNorm(value: unknown): "l1" | "l2" | undefined {
  if (value === undefined || value === null) return undefined;
  if (value !== "l1" && value !== "l2") {
    throw new InvalidParameterError(
      `norm must be "l1", "l2" or undefined; received ${String(value)}`,
      "norm",
      value
    );
  }
  return value;
}

function resolveTextOptions(raw: OptionBag): ResolvedTextOptions {
  let tokenPattern: RegExp;
  const rawPattern = raw["tokenPattern"];
  if (rawPattern === undefined) {
    tokenPattern = defaultTokenPattern();
  } else if (rawPattern instanceof RegExp) {
    // Work on a private global copy: the caller's regex is never mutated, and matchAll
    // requires the g flag.
    tokenPattern = new RegExp(
      rawPattern.source,
      rawPattern.flags.includes("g") ? rawPattern.flags : `${rawPattern.flags}g`
    );
  } else {
    throw new InvalidParameterError("tokenPattern must be a RegExp", "tokenPattern", rawPattern);
  }

  const rawStop = raw["stopWords"];
  let stopWords: ReadonlySet<string>;
  let stopWordsSpec: readonly string[] | "english" | undefined;
  if (rawStop === undefined) {
    stopWords = new Set();
    stopWordsSpec = undefined;
  } else if (rawStop === "english") {
    stopWords = new Set(ENGLISH_STOP_WORDS);
    stopWordsSpec = "english";
  } else if (Array.isArray(rawStop) && rawStop.every((w) => typeof w === "string")) {
    stopWordsSpec = [...(rawStop as string[])];
    stopWords = new Set(stopWordsSpec);
  } else {
    throw new InvalidParameterError(
      'stopWords must be "english" or an array of strings',
      "stopWords",
      rawStop
    );
  }

  const rawRange = raw["ngramRange"];
  let ngramRange: readonly [number, number] = [1, 1];
  if (rawRange !== undefined) {
    const [minN, maxN] = Array.isArray(rawRange) ? rawRange : [];
    if (
      !Array.isArray(rawRange) ||
      rawRange.length !== 2 ||
      !Number.isInteger(minN) ||
      !Number.isInteger(maxN) ||
      (minN as number) < 1 ||
      (maxN as number) < (minN as number)
    ) {
      throw new InvalidParameterError(
        `ngramRange must be [min, max] with 1 <= min <= max; received ${JSON.stringify(rawRange)}`,
        "ngramRange",
        rawRange
      );
    }
    ngramRange = [minN as number, maxN as number];
  }

  return {
    tokenPattern,
    lowercase: resolveBoolean(raw["lowercase"], "lowercase", true),
    binary: resolveBoolean(raw["binary"], "binary", false),
    stopWords,
    stopWordsSpec,
    ngramRange,
  };
}

function resolveCountOptions(raw: OptionBag): ResolvedCountOptions {
  const text = resolveTextOptions(raw);

  const maxFeatures = raw["maxFeatures"];
  if (
    maxFeatures !== undefined &&
    (typeof maxFeatures !== "number" || !Number.isInteger(maxFeatures) || maxFeatures < 1)
  ) {
    throw new InvalidParameterError(
      `maxFeatures must be a positive integer; received ${String(maxFeatures)}`,
      "maxFeatures",
      maxFeatures
    );
  }

  const minDf = raw["minDf"] ?? 1;
  if (
    typeof minDf !== "number" ||
    !Number.isFinite(minDf) ||
    minDf < 0 ||
    (minDf >= 1 && !Number.isInteger(minDf))
  ) {
    throw new InvalidParameterError(
      `minDf must be a proportion in [0, 1) or an integer count >= 1; received ${String(minDf)}`,
      "minDf",
      minDf
    );
  }
  const maxDf = raw["maxDf"] ?? 1;
  if (
    typeof maxDf !== "number" ||
    !Number.isFinite(maxDf) ||
    maxDf <= 0 ||
    (maxDf > 1 && !Number.isInteger(maxDf))
  ) {
    throw new InvalidParameterError(
      `maxDf must be a proportion in (0, 1] or an integer count > 1; received ${String(maxDf)}`,
      "maxDf",
      maxDf
    );
  }

  return { ...text, maxFeatures: maxFeatures as number | undefined, minDf, maxDf };
}

function assertKnownKeys(params: OptionBag, allowed: readonly string[], owner: string): void {
  for (const key of Object.keys(params)) {
    if (!allowed.includes(key)) {
      throw new InvalidParameterError(`${owner} has no parameter named "${key}"`, key, params[key]);
    }
  }
}

function validateDocuments(documents: readonly string[]): void {
  if (!Array.isArray(documents)) {
    throw new InvalidParameterError(
      "documents must be an array of strings",
      "documents",
      typeof documents
    );
  }
  for (let i = 0; i < documents.length; i++) {
    if (typeof documents[i] !== "string") {
      throw new InvalidParameterError(
        `documents must be an array of strings; element ${i} is ${typeof documents[i]}`,
        "documents",
        documents[i]
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Tokenization and output helpers
// ---------------------------------------------------------------------------

function tokenize(doc: string, opts: ResolvedTextOptions): string[] {
  const text = opts.lowercase ? doc.toLowerCase() : doc;
  const rawTokens: string[] = [];
  for (const match of text.matchAll(opts.tokenPattern)) {
    const token = match[0];
    if (token.length > 0 && !opts.stopWords.has(token)) {
      rawTokens.push(token);
    }
  }

  const [minN, maxN] = opts.ngramRange;
  if (minN === 1 && maxN === 1) {
    return rawTokens;
  }

  const ngrams: string[] = [];
  for (let n = minN; n <= maxN; n++) {
    for (let i = 0; i <= rawTokens.length - n; i++) {
      ngrams.push(n === 1 ? (rawTokens[i] as string) : rawTokens.slice(i, i + n).join(" "));
    }
  }
  return ngrams;
}

function allocateDense(nRows: number, nCols: number, what: string): Float64Array {
  const size = nRows * nCols;
  try {
    return new Float64Array(size);
  } catch (error) {
    if (error instanceof RangeError) {
      throw new MemoryError(
        `${what} cannot allocate a dense ${nRows} x ${nCols} matrix (${size} values); ` +
          "use fewer documents per call or fewer features (maxFeatures / nFeatures)",
        { requestedBytes: size * 8, cause: error }
      );
    }
    throw error;
  }
}

/**
 * Wrap a row-major matrix as a tensor of the configured default dtype. A non-float default is
 * used only when every value fits it exactly (for example plain counts under `int32`); TF-IDF
 * weights and hashed features fall back to `float32` instead of being truncated.
 */
function toFloatTensor(data: Float64Array, nRows: number, nCols: number): Tensor {
  const dtype = resolveLosslessDType(getDtype(), data);
  if (dtype === "float64") {
    return TensorClass.fromTypedArray({
      data,
      shape: [nRows, nCols],
      dtype,
      device: getDevice(),
    });
  }
  if (dtype === "float32") {
    return TensorClass.fromTypedArray({
      data: new Float32Array(data),
      shape: [nRows, nCols],
      dtype,
      device: getDevice(),
    });
  }
  return tensor(Array.from(data), { dtype, device: getDevice() }).reshape([nRows, nCols]);
}

/** Normalize each row of `data` in place to unit L1 or L2 norm; all-zero rows stay zero. */
function normalizeRows(data: Float64Array, nRows: number, nCols: number, norm: "l1" | "l2"): void {
  for (let i = 0; i < nRows; i++) {
    const rowStart = i * nCols;
    let normVal = 0;
    if (norm === "l2") {
      for (let j = 0; j < nCols; j++) {
        const v = data[rowStart + j] as number;
        normVal += v * v;
      }
      normVal = Math.sqrt(normVal);
    } else {
      for (let j = 0; j < nCols; j++) {
        normVal += Math.abs(data[rowStart + j] as number);
      }
    }
    if (normVal > 0) {
      for (let j = 0; j < nCols; j++) {
        data[rowStart + j] = (data[rowStart + j] as number) / normVal;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// CountVectorizer
// ---------------------------------------------------------------------------

/**
 * Convert a collection of text documents to a matrix of token counts.
 *
 * Implements the bag-of-words model: each document becomes a vector of
 * word counts. The vocabulary is learned from the training data and sorted
 * alphabetically by UTF-16 code unit, which is scikit-learn's order except for
 * rare characters outside the Basic Multilingual Plane. Features whose
 * document frequency is outside `[minDf, maxDf]` are dropped, then at most
 * `maxFeatures` of the most frequent terms are kept. Fitting throws a
 * `DataValidationError` when no term survives.
 *
 * The result is a dense matrix of the default float dtype; for very large
 * vocabularies use `maxFeatures` or {@link HashingVectorizer}.
 *
 * @example
 * ```ts
 * import { CountVectorizer } from 'deepbox/preprocess';
 *
 * const cv = new CountVectorizer();
 * const X = cv.fitTransformText(['hello world', 'hello deepbox world']);
 * // X is a 2D tensor of shape [2, vocabulary_size]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox documentation}
 */
export class CountVectorizer {
  private opts: ResolvedCountOptions;

  private vocabulary_: Map<string, number> = new Map();
  private featureNames_: string[] = [];
  private docFreq_: Float64Array = new Float64Array(0);
  private fitted = false;

  constructor(options: CountVectorizerOptions = {}) {
    this.opts = resolveCountOptions(options);
  }

  /**
   * Learn vocabulary from documents. Calling it again discards the previous vocabulary.
   *
   * @param documents - Array of text documents
   * @returns this
   * @throws {InvalidParameterError} If `documents` is not an array of strings, or `maxDf`
   *   allows fewer documents than `minDf`
   * @throws {DataValidationError} If the resulting vocabulary is empty
   */
  fitText(documents: readonly string[]): this {
    validateDocuments(documents);
    const nDocs = documents.length;
    if (nDocs === 0) {
      throw new DataValidationError("Empty vocabulary: documents is an empty array");
    }
    const { minDf, maxDf, maxFeatures } = this.opts;
    const stats = new Map<string, { df: number; tf: number; last: number }>();

    for (let i = 0; i < nDocs; i++) {
      for (const token of tokenize(documents[i] as string, this.opts)) {
        let entry = stats.get(token);
        if (entry === undefined) {
          entry = { df: 0, tf: 0, last: -1 };
          stats.set(token, entry);
        }
        entry.tf += 1;
        if (entry.last !== i) {
          entry.last = i;
          entry.df += 1;
        }
      }
    }

    // Document-frequency limits. Proportions are compared without rounding, as scikit-learn
    // does: minDf = 0.5 over 5 documents keeps terms found in at least 3 documents.
    const minCount = minDf < 1 ? minDf * nDocs : minDf;
    const maxCount = maxDf <= 1 ? maxDf * nDocs : maxDf;
    if (maxCount < minCount) {
      throw new InvalidParameterError(
        "maxDf corresponds to fewer documents than minDf",
        "maxDf",
        maxDf
      );
    }

    // Each candidate is [term, ranking weight, document frequency]. The weight is the total
    // count, or the document frequency when `binary` is set (the counts are then 0/1, as in
    // scikit-learn).
    let candidates: Array<[string, number, number]> = [];
    for (const [term, entry] of stats) {
      if (entry.df >= minCount && entry.df <= maxCount) {
        candidates.push([term, this.opts.binary ? entry.df : entry.tf, entry.df]);
      }
    }

    if (maxFeatures !== undefined && candidates.length > maxFeatures) {
      // Most frequent terms first; alphabetical order breaks ties.
      candidates.sort((a, b) => (b[1] !== a[1] ? b[1] - a[1] : compareStrings(a[0], b[0])));
      candidates = candidates.slice(0, maxFeatures);
    }

    if (candidates.length === 0) {
      throw new DataValidationError(
        "Empty vocabulary: no term survived tokenization, stop word removal and the minDf / maxDf " +
          "limits. Check the documents and these options."
      );
    }

    // Final vocabulary order is alphabetical (scikit-learn convention).
    candidates.sort((a, b) => compareStrings(a[0], b[0]));

    const vocabulary = new Map<string, number>();
    const featureNames: string[] = [];
    const docFreq = new Float64Array(candidates.length);
    for (let i = 0; i < candidates.length; i++) {
      const [term, , df] = candidates[i] as [string, number, number];
      vocabulary.set(term, i);
      featureNames.push(term);
      docFreq[i] = df;
    }

    this.vocabulary_ = vocabulary;
    this.featureNames_ = featureNames;
    this.docFreq_ = docFreq;
    this.fitted = true;
    return this;
  }

  /**
   * Count the learned terms in each document. Terms outside the vocabulary are ignored.
   *
   * @param documents - Array of text documents
   * @returns Row-major `[nDocs, nFeatures]` counts (0/1 when `binary` is set)
   * @internal
   */
  transformCounts(documents: readonly string[]): Float64Array {
    if (!this.fitted) {
      throw new NotFittedError("CountVectorizer must be fitted before transform");
    }
    validateDocuments(documents);

    const nDocs = documents.length;
    const nFeatures = this.vocabulary_.size;
    const data = allocateDense(nDocs, nFeatures, "CountVectorizer.transformText");
    const binary = this.opts.binary;

    for (let i = 0; i < nDocs; i++) {
      for (const token of tokenize(documents[i] as string, this.opts)) {
        const idx = this.vocabulary_.get(token);
        if (idx !== undefined) {
          const pos = i * nFeatures + idx;
          data[pos] = binary ? 1 : (data[pos] as number) + 1;
        }
      }
    }
    return data;
  }

  /**
   * Number of fitted documents that contain each term, in vocabulary order.
   *
   * @internal
   */
  documentFrequencies(): Float64Array {
    if (!this.fitted) {
      throw new NotFittedError("CountVectorizer must be fitted first");
    }
    return this.docFreq_;
  }

  /**
   * Transform documents to a document-term matrix.
   *
   * @param documents - Array of text documents
   * @returns 2D Tensor of shape [n_documents, n_features]
   * @throws {NotFittedError} If the vectorizer has not been fitted
   */
  transformText(documents: readonly string[]): Tensor {
    const data = this.transformCounts(documents);
    return toFloatTensor(data, documents.length, this.vocabulary_.size);
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

  /** The learned vocabulary mapping each term to its index */
  get vocabulary(): ReadonlyMap<string, number> {
    if (!this.fitted) throw new NotFittedError("CountVectorizer must be fitted first");
    return this.vocabulary_;
  }

  /** Feature names (terms) in vocabulary order */
  getFeatureNames(): string[] {
    if (!this.fitted) throw new NotFittedError("CountVectorizer must be fitted first");
    return [...this.featureNames_];
  }

  /** Current parameters. Pass them to `setParams` or the constructor to rebuild a vectorizer. */
  getParams(): Record<string, unknown> {
    return {
      tokenPattern: new RegExp(this.opts.tokenPattern.source, this.opts.tokenPattern.flags),
      maxFeatures: this.opts.maxFeatures,
      minDf: this.opts.minDf,
      maxDf: this.opts.maxDf,
      lowercase: this.opts.lowercase,
      binary: this.opts.binary,
      stopWords:
        this.opts.stopWordsSpec === undefined || this.opts.stopWordsSpec === "english"
          ? this.opts.stopWordsSpec
          : [...this.opts.stopWordsSpec],
      ngramRange: [...this.opts.ngramRange],
    };
  }

  /**
   * Change parameters. Setting any parameter discards the learned vocabulary, so call
   * `fitText` again before transforming.
   *
   * @param params - Subset of the constructor options
   * @throws {InvalidParameterError} If a name is unknown or a value is invalid; nothing changes
   */
  setParams(params: Record<string, unknown>): this {
    assertKnownKeys(params, COUNT_PARAM_KEYS, "CountVectorizer");
    if (Object.keys(params).length === 0) return this;
    this.opts = resolveCountOptions({ ...this.getParams(), ...params });
    this.vocabulary_ = new Map();
    this.featureNames_ = [];
    this.docFreq_ = new Float64Array(0);
    this.fitted = false;
    return this;
  }
}

// ---------------------------------------------------------------------------
// TfidfVectorizer
// ---------------------------------------------------------------------------

/**
 * Convert a collection of text documents to a TF-IDF feature matrix.
 *
 * Combines CountVectorizer with TF-IDF weighting. TF-IDF (Term Frequency -
 * Inverse Document Frequency) down-weights terms that appear in many documents
 * and up-weights rare, discriminative terms.
 *
 * IDF formula: `log((1 + n) / (1 + df)) + 1` with `smoothIdf` (default), or
 * `log(n / df) + 1` without it, where `n` is the number of fitted documents and `df`
 * the number of documents containing the term. Rows are normalized to unit L2 norm by
 * default. The results match scikit-learn's `TfidfVectorizer` for the same options.
 *
 * @example
 * ```ts
 * import { TfidfVectorizer } from 'deepbox/preprocess';
 *
 * const tfidf = new TfidfVectorizer();
 * const X = tfidf.fitTransformText(['hello world', 'hello deepbox']);
 * // X is a 2D tensor of shape [2, vocabulary_size] with TF-IDF weights
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox documentation}
 */
export class TfidfVectorizer {
  private readonly countVectorizer: CountVectorizer;
  private norm: "l1" | "l2" | undefined;
  private sublinearTf: boolean;
  private smoothIdf: boolean;
  private useIdf: boolean;

  private idf_: Float64Array = new Float64Array(0);
  private fitted = false;

  constructor(options: TfidfVectorizerOptions = {}) {
    this.countVectorizer = new CountVectorizer(options);
    this.norm = "norm" in options ? resolveNorm(options.norm) : "l2";
    this.sublinearTf = resolveBoolean(options.sublinearTf, "sublinearTf", false);
    this.smoothIdf = resolveBoolean(options.smoothIdf, "smoothIdf", true);
    this.useIdf = resolveBoolean(options.useIdf, "useIdf", true);
  }

  /** Learn the vocabulary and the IDF weights. */
  private fitIdf(documents: readonly string[]): void {
    this.fitted = false;
    this.countVectorizer.fitText(documents);
    const df = this.countVectorizer.documentFrequencies();
    const nDocs = documents.length;
    const nFeatures = df.length;

    const idf = new Float64Array(nFeatures);
    if (this.useIdf) {
      const smooth = this.smoothIdf ? 1 : 0;
      for (let j = 0; j < nFeatures; j++) {
        idf[j] = Math.log((smooth + nDocs) / (smooth + (df[j] as number))) + 1;
      }
    } else {
      idf.fill(1);
    }

    this.idf_ = idf;
    this.fitted = true;
  }

  /** Apply TF scaling, IDF weights and normalization to a count matrix in place. */
  private weigh(data: Float64Array, nDocs: number, nFeatures: number): void {
    const idf = this.idf_;
    const sublinear = this.sublinearTf;
    for (let i = 0; i < nDocs; i++) {
      const rowStart = i * nFeatures;
      for (let j = 0; j < nFeatures; j++) {
        const count = data[rowStart + j] as number;
        if (count === 0) continue;
        data[rowStart + j] = (sublinear ? 1 + Math.log(count) : count) * (idf[j] as number);
      }
    }
    if (this.norm !== undefined) {
      normalizeRows(data, nDocs, nFeatures, this.norm);
    }
  }

  /**
   * Learn vocabulary and IDF weights from documents.
   *
   * @param documents - Array of text documents
   * @returns this
   * @throws {DataValidationError} If the resulting vocabulary is empty
   */
  fitText(documents: readonly string[]): this {
    this.fitIdf(documents);
    return this;
  }

  /**
   * Transform documents to TF-IDF weighted matrix.
   *
   * @param documents - Array of text documents
   * @returns 2D Tensor of shape [n_documents, n_features]
   * @throws {NotFittedError} If the vectorizer has not been fitted
   */
  transformText(documents: readonly string[]): Tensor {
    if (!this.fitted) {
      throw new NotFittedError("TfidfVectorizer must be fitted before transform");
    }
    const data = this.countVectorizer.transformCounts(documents);
    const nFeatures = this.idf_.length;
    this.weigh(data, documents.length, nFeatures);
    return toFloatTensor(data, documents.length, nFeatures);
  }

  /**
   * Learn vocabulary/IDF and return TF-IDF matrix.
   *
   * @param documents - Array of text documents
   * @returns 2D Tensor of shape [n_documents, n_features]
   */
  fitTransformText(documents: readonly string[]): Tensor {
    this.fitIdf(documents);
    const data = this.countVectorizer.transformCounts(documents);
    const nFeatures = this.idf_.length;
    this.weigh(data, documents.length, nFeatures);
    return toFloatTensor(data, documents.length, nFeatures);
  }

  /** The learned vocabulary mapping each term to its index */
  get vocabulary(): ReadonlyMap<string, number> {
    if (!this.fitted) throw new NotFittedError("TfidfVectorizer must be fitted first");
    return this.countVectorizer.vocabulary;
  }

  /** The learned IDF vector (all ones when `useIdf` is false) */
  get idf(): Float64Array {
    if (!this.fitted) throw new NotFittedError("TfidfVectorizer must be fitted first");
    return this.idf_;
  }

  /** Feature names (terms) in vocabulary order */
  getFeatureNames(): string[] {
    if (!this.fitted) throw new NotFittedError("TfidfVectorizer must be fitted first");
    return this.countVectorizer.getFeatureNames();
  }

  /** Current parameters. Pass them to `setParams` or the constructor to rebuild a vectorizer. */
  getParams(): Record<string, unknown> {
    return {
      ...this.countVectorizer.getParams(),
      norm: this.norm,
      sublinearTf: this.sublinearTf,
      smoothIdf: this.smoothIdf,
      useIdf: this.useIdf,
    };
  }

  /**
   * Change parameters. Changing anything except `norm` and `sublinearTf` discards the learned
   * vocabulary and IDF, so call `fitText` again before transforming.
   *
   * @param params - Subset of the constructor options
   * @throws {InvalidParameterError} If a name is unknown or a value is invalid; nothing changes
   */
  setParams(params: Record<string, unknown>): this {
    assertKnownKeys(params, [...COUNT_PARAM_KEYS, ...TFIDF_OWN_KEYS], "TfidfVectorizer");
    // Validate every value before applying any of them.
    const norm = "norm" in params ? resolveNorm(params["norm"]) : this.norm;
    const sublinearTf = resolveBoolean(params["sublinearTf"], "sublinearTf", this.sublinearTf);
    const smoothIdf = resolveBoolean(params["smoothIdf"], "smoothIdf", this.smoothIdf);
    const useIdf = resolveBoolean(params["useIdf"], "useIdf", this.useIdf);

    const countParams: Record<string, unknown> = {};
    let needsRefit = smoothIdf !== this.smoothIdf || useIdf !== this.useIdf;
    for (const key of Object.keys(params)) {
      if (!TFIDF_OWN_KEYS.includes(key)) {
        countParams[key] = params[key];
        needsRefit = true;
      }
    }
    if (Object.keys(countParams).length > 0) {
      this.countVectorizer.setParams(countParams);
    }

    this.norm = norm;
    this.sublinearTf = sublinearTf;
    this.smoothIdf = smoothIdf;
    this.useIdf = useIdf;
    if (needsRefit) {
      this.fitted = false;
      this.idf_ = new Float64Array(0);
    }
    return this;
  }
}

// ---------------------------------------------------------------------------
// HashingVectorizer
// ---------------------------------------------------------------------------

/**
 * FNV-1a 32-bit hash function.
 *
 * Deterministic, fast, and provides reasonable distribution for feature hashing.
 * It hashes the UTF-16 code units of the token, so bucket assignments differ from
 * scikit-learn's MurmurHash3 over UTF-8.
 */
function fnv1a(str: string): number {
  let hash = 0x811c9dc5; // FNV offset basis
  for (let i = 0; i < str.length; i++) {
    hash ^= str.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193); // FNV prime
  }
  return hash >>> 0; // ensure unsigned
}

type ResolvedHashingOptions = ResolvedTextOptions & {
  readonly nFeatures: number;
  readonly norm: "l1" | "l2" | undefined;
  readonly alternateSign: boolean;
};

function resolveHashingOptions(raw: OptionBag): ResolvedHashingOptions {
  const nFeatures = raw["nFeatures"] ?? 1 << 20;
  if (typeof nFeatures !== "number" || !Number.isInteger(nFeatures) || nFeatures < 1) {
    throw new InvalidParameterError(
      `nFeatures must be a positive integer; received ${String(nFeatures)}`,
      "nFeatures",
      nFeatures
    );
  }
  return {
    ...resolveTextOptions(raw),
    nFeatures,
    norm: "norm" in raw ? resolveNorm(raw["norm"]) : "l2",
    alternateSign: resolveBoolean(raw["alternateSign"], "alternateSign", true),
  };
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
 * The output is a dense `[nDocs, nFeatures]` matrix, so the default of 2^20
 * features costs 8 MB per document; choose a smaller `nFeatures` for many documents.
 *
 * @example
 * ```ts
 * import { HashingVectorizer } from 'deepbox/preprocess';
 *
 * const hv = new HashingVectorizer({ nFeatures: 1024 });
 * const X = hv.transformText(['hello world', 'hello deepbox world']);
 * // X is a 2D tensor of shape [2, 1024]
 * ```
 *
 * @see {@link https://deepbox.dev/docs/preprocess-scalers | Deepbox documentation}
 */
export class HashingVectorizer {
  private opts: ResolvedHashingOptions;

  constructor(options: HashingVectorizerOptions = {}) {
    this.opts = resolveHashingOptions(options);
  }

  /**
   * No-op, provided so the vectorizer can be used where `fitText` is expected.
   * Only validates the documents.
   *
   * @param documents - Array of text documents
   * @returns this
   */
  fitText(documents: readonly string[]): this {
    validateDocuments(documents);
    return this;
  }

  /**
   * Transform documents to a fixed-size hash feature matrix.
   *
   * No fitting is required: this is a stateless transform.
   *
   * @param documents - Array of text documents
   * @returns 2D Tensor of shape [n_documents, nFeatures]
   */
  transformText(documents: readonly string[]): Tensor {
    validateDocuments(documents);
    const nDocs = documents.length;
    const nF = this.opts.nFeatures;
    const data = allocateDense(nDocs, nF, "HashingVectorizer.transformText");
    const { binary, alternateSign, norm } = this.opts;
    const hiMod = 2147483648 % nF;

    // Generation-stamped touch tracking: `touchStamp[idx] === i` marks column
    // `idx` as written by doc `i`, so normalization sweeps only the (few)
    // non-zero columns per document instead of all nF (the old code paid two
    // O(nDocs·nF) passes over a matrix that is ~99% zeros).
    const touchStamp = norm ? new Int32Array(nF).fill(-1) : null;
    const touched: number[] = [];

    for (let i = 0; i < nDocs; i++) {
      const tokens = tokenize(documents[i] as string, this.opts);

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
    return toFloatTensor(data, nDocs, nF);
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

  /** Current parameters. Pass them to `setParams` or the constructor to rebuild a vectorizer. */
  getParams(): Record<string, unknown> {
    return {
      nFeatures: this.opts.nFeatures,
      tokenPattern: new RegExp(this.opts.tokenPattern.source, this.opts.tokenPattern.flags),
      lowercase: this.opts.lowercase,
      binary: this.opts.binary,
      stopWords:
        this.opts.stopWordsSpec === undefined || this.opts.stopWordsSpec === "english"
          ? this.opts.stopWordsSpec
          : [...this.opts.stopWordsSpec],
      ngramRange: [...this.opts.ngramRange],
      norm: this.opts.norm,
      alternateSign: this.opts.alternateSign,
    };
  }

  /**
   * Change parameters. The vectorizer is stateless, so the change applies to the next
   * `transformText` call.
   *
   * @param params - Subset of the constructor options
   * @throws {InvalidParameterError} If a name is unknown or a value is invalid; nothing changes
   */
  setParams(params: Record<string, unknown>): this {
    assertKnownKeys(params, HASHING_PARAM_KEYS, "HashingVectorizer");
    this.opts = resolveHashingOptions({ ...this.getParams(), ...params });
    return this;
  }
}
