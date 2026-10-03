/**
 * Text dataset loaders (20 Newsgroups, IMDB) via fetch.
 *
 * By default the official archives are downloaded and unpacked (the 20 Newsgroups "bydate"
 * archive that scikit-learn uses, and the Stanford `aclImdb` archive). Passing `baseUrl` reads a
 * JSON mirror with the layout described on {@link fetch20Newsgroups} and {@link fetchIMDB}
 * instead. Documents are returned as a string array with integer labels.
 *
 * @module datasets/text
 * @see {@link https://deepbox.dev/docs/datasets-builtin | Deepbox documentation}
 */

import { DataValidationError, DeepboxError, InvalidParameterError } from "../core/errors";
import { type Tensor, tensor } from "../ndarray";
import { decompressGzip, extractTarFiles } from "./_archive";
import { assertBoolean, assertPositiveInt } from "./utils";

/**
 * Result of a text dataset fetch.
 */
export type TextDataset = {
  /** Array of text documents */
  readonly texts: string[];
  /** Integer (`int32`) label tensor of shape (nSamples,) */
  readonly target: Tensor;
  /** Number of classes */
  readonly nClasses: number;
  /** Class label names */
  readonly classNames: string[];
  /** Dataset description */
  readonly description: string;
  /** Whether this is synthetic/fallback data (true when remote fetch failed) */
  readonly isSynthetic: boolean;
};

/**
 * Options shared by {@link fetch20Newsgroups} and {@link fetchIMDB}.
 */
export type TextFetchOptions = {
  /** Which subset: 'train', 'test', or 'all' (default: 'all') */
  readonly subset?: "train" | "test" | "all";
  /** Maximum number of samples to load (a positive integer). Keeps the first samples of the file. */
  readonly maxSamples?: number;
  /**
   * Base URL of a JSON mirror; the file name is appended to it. When it is set the JSON mirror
   * is read instead of the official archive. Cannot be combined with `archiveUrl`.
   */
  readonly baseUrl?: string;
  /**
   * URL of a gzip-compressed tar archive with the official layout (for example a local mirror
   * of the default archive). Default: the official archive. Cannot be combined with `baseUrl`.
   */
  readonly archiveUrl?: string;
  /** When true, return synthetic placeholder data on fetch failure. Default: false */
  readonly allowSyntheticFallback?: boolean;
  /**
   * Fetch timeout in milliseconds (`0` disables the timeout). Default: 30 000 for a JSON
   * mirror and 600 000 for an archive download, which is much larger.
   */
  readonly timeout?: number;
};

const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_ARCHIVE_TIMEOUT_MS = 600_000;
/** The "bydate" archive of 20 Newsgroups that scikit-learn downloads. */
const NEWSGROUPS_ARCHIVE_URL = "https://ndownloader.figshare.com/files/5975967";
/** The Stanford large movie review archive. */
const IMDB_ARCHIVE_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
/** Largest delay (about 24.8 days) that timers accept without overflowing. */
const MAX_TIMEOUT_MS = 2_147_483_647;
const SUBSETS: readonly string[] = ["train", "test", "all"];

const NEWSGROUP_CATEGORIES: readonly string[] = [
  "alt.atheism",
  "comp.graphics",
  "comp.os.ms-windows.misc",
  "comp.sys.ibm.pc.hardware",
  "comp.sys.mac.hardware",
  "comp.windows.x",
  "misc.forsale",
  "rec.autos",
  "rec.motorcycles",
  "rec.sport.baseball",
  "rec.sport.hockey",
  "sci.crypt",
  "sci.electronics",
  "sci.med",
  "sci.space",
  "soc.religion.christian",
  "talk.politics.guns",
  "talk.politics.mideast",
  "talk.politics.misc",
  "talk.religion.misc",
];

const IMDB_CLASSES: readonly string[] = ["negative", "positive"];

function validateTextOptions(options: TextFetchOptions): void {
  if (options.subset !== undefined && !SUBSETS.includes(options.subset)) {
    throw new InvalidParameterError(
      `subset must be 'train', 'test' or 'all'; received ${String(options.subset)}`,
      "subset",
      options.subset
    );
  }
  if (options.maxSamples !== undefined) assertPositiveInt("maxSamples", options.maxSamples);
  if (options.baseUrl !== undefined && (typeof options.baseUrl !== "string" || !options.baseUrl)) {
    throw new InvalidParameterError(
      "baseUrl must be a non-empty string",
      "baseUrl",
      options.baseUrl
    );
  }
  if (
    options.archiveUrl !== undefined &&
    (typeof options.archiveUrl !== "string" || !options.archiveUrl)
  ) {
    throw new InvalidParameterError(
      "archiveUrl must be a non-empty string",
      "archiveUrl",
      options.archiveUrl
    );
  }
  if (options.baseUrl !== undefined && options.archiveUrl !== undefined) {
    throw new InvalidParameterError(
      "Pass either baseUrl (JSON mirror) or archiveUrl (tar.gz archive), not both",
      "archiveUrl",
      options.archiveUrl
    );
  }
  if (options.allowSyntheticFallback !== undefined) {
    assertBoolean("allowSyntheticFallback", options.allowSyntheticFallback);
  }
  if (
    options.timeout !== undefined &&
    !(Number.isFinite(options.timeout) && options.timeout >= 0)
  ) {
    throw new InvalidParameterError(
      "timeout must be a non-negative number of milliseconds",
      "timeout",
      options.timeout
    );
  }
}

function joinUrl(base: string, file: string): string {
  return base.endsWith("/") ? `${base}${file}` : `${base}/${file}`;
}

type TextJson = {
  readonly data: string[];
  readonly target: number[];
  readonly target_names?: string[];
};

/**
 * Fetch and validate a text-dataset JSON file. Throws with a human-readable
 * reason on any HTTP, parsing or layout problem.
 */
async function fetchTextJson(
  url: string,
  timeoutMs: number,
  nClasses: number | undefined
): Promise<TextJson> {
  const resp = await fetch(url, timeoutSignal(timeoutMs));
  if (!resp.ok) {
    throw new DeepboxError(`HTTP ${resp.status} ${resp.statusText}`.trim());
  }
  const json = (await resp.json()) as Partial<TextJson> | null;
  if (json === null || typeof json !== "object") {
    throw new DataValidationError("response is not a JSON object");
  }
  const { data, target, target_names } = json;
  if (!Array.isArray(data) || !Array.isArray(target)) {
    throw new DataValidationError('response must contain "data" and "target" arrays');
  }
  if (data.length !== target.length) {
    throw new DataValidationError(
      `"data" has ${data.length} entries but "target" has ${target.length}`
    );
  }
  if (
    target_names !== undefined &&
    (!Array.isArray(target_names) || target_names.some((name) => typeof name !== "string"))
  ) {
    throw new DataValidationError('"target_names" must be an array of strings');
  }
  const classes = nClasses ?? target_names?.length;
  if (classes === undefined || classes < 1) {
    throw new DataValidationError('response must contain a non-empty "target_names" array');
  }
  for (let i = 0; i < data.length; i++) {
    if (typeof data[i] !== "string") {
      throw new DataValidationError(`"data"[${i}] is not a string`);
    }
    const label = target[i];
    if (!Number.isInteger(label) || (label as number) < 0 || (label as number) >= classes) {
      throw new DataValidationError(
        `"target"[${i}] must be an integer in [0, ${classes - 1}]; received ${String(label)}`
      );
    }
  }
  return json as TextJson;
}

function timeoutSignal(timeoutMs: number): { signal?: AbortSignal } {
  const init: { signal?: AbortSignal } = {};
  if (timeoutMs > 0 && typeof AbortSignal.timeout === "function") {
    // AbortSignal.timeout only accepts integers up to 2^32 - 1 in some runtimes.
    init.signal = AbortSignal.timeout(Math.min(Math.ceil(timeoutMs), MAX_TIMEOUT_MS));
  }
  return init;
}

/** Download a `.tar.gz` archive and return the unpacked tar entries. */
async function fetchTarEntries(
  url: string,
  timeoutMs: number
): Promise<{ name: string; data: Uint8Array }[]> {
  const resp = await fetch(url, timeoutSignal(timeoutMs));
  if (!resp.ok) {
    throw new DeepboxError(`HTTP ${resp.status} ${resp.statusText}`.trim());
  }
  const tar = await decompressGzip(new Uint8Array(await resp.arrayBuffer()));
  return extractTarFiles(tar);
}

/** Text of `bytes` read as ISO-8859-1 (what scikit-learn uses for 20 Newsgroups). */
function decodeLatin1(bytes: Uint8Array): string {
  const chunks: string[] = [];
  for (let i = 0; i < bytes.length; i += 8192) {
    chunks.push(String.fromCharCode(...bytes.subarray(i, i + 8192)));
  }
  return chunks.join("");
}

/** Whether a tar entry is a macOS AppleDouble metadata file (`._name`) rather than a document. */
function isMetadataEntry(path: string): boolean {
  return path.split("/").some((part) => part.startsWith("._"));
}

function compareNames(a: { name: string }, b: { name: string }): number {
  return a.name < b.name ? -1 : a.name > b.name ? 1 : 0;
}

type ParsedText = {
  readonly data: string[];
  readonly target: number[];
  readonly classNames: string[];
};

/**
 * Documents of the 20 Newsgroups "bydate" archive: the train split, then the test split, each
 * ordered by newsgroup (alphabetically) and then by file name.
 */
function parseNewsgroupsArchive(
  entries: readonly { name: string; data: Uint8Array }[],
  subset: "train" | "test" | "all"
): ParsedText {
  const pattern = /(?:^|\/)20news-bydate-(train|test)\/([^/]+)\/([^/]+)$/;
  const found: Record<"train" | "test", Map<string, { name: string; data: Uint8Array }[]>> = {
    train: new Map(),
    test: new Map(),
  };
  const categories = new Set<string>();
  for (const entry of entries) {
    const match = pattern.exec(entry.name);
    // Archives written on macOS may carry AppleDouble "._name" metadata entries; skip them.
    if (!match || isMetadataEntry(entry.name)) continue;
    const split = match[1] as "train" | "test";
    const category = match[2] as string;
    categories.add(category);
    const list = found[split].get(category) ?? [];
    list.push({ name: match[3] as string, data: entry.data });
    found[split].set(category, list);
  }
  if (categories.size === 0) {
    throw new DataValidationError(
      "archive does not contain 20news-bydate-train or 20news-bydate-test newsgroup folders"
    );
  }
  const classNames = [...categories].sort();
  const data: string[] = [];
  const target: number[] = [];
  const splits: Array<"train" | "test"> = subset === "all" ? ["train", "test"] : [subset];
  for (const split of splits) {
    for (let c = 0; c < classNames.length; c++) {
      const docs = found[split].get(classNames[c] as string) ?? [];
      docs.sort(compareNames);
      for (const doc of docs) {
        data.push(decodeLatin1(doc.data));
        target.push(c);
      }
    }
  }
  if (data.length === 0) {
    throw new DataValidationError(`archive has no documents in the ${subset} subset`);
  }
  return { data, target, classNames };
}

/**
 * Reviews of the `aclImdb` archive: the train split, then the test split, each with the
 * negative reviews (label 0) before the positive ones (label 1), ordered by file name.
 */
function parseImdbArchive(
  entries: readonly { name: string; data: Uint8Array }[],
  subset: "train" | "test" | "all"
): ParsedText {
  const pattern = /(?:^|\/)aclImdb\/(train|test)\/(neg|pos)\/[^/]+\.txt$/;
  const found: Record<
    "train" | "test",
    Record<"neg" | "pos", { name: string; data: Uint8Array }[]>
  > = { train: { neg: [], pos: [] }, test: { neg: [], pos: [] } };
  for (const entry of entries) {
    const match = pattern.exec(entry.name);
    if (!match || isMetadataEntry(entry.name)) continue;
    found[match[1] as "train" | "test"][match[2] as "neg" | "pos"].push(entry);
  }
  const decoder = new TextDecoder("utf-8");
  const data: string[] = [];
  const target: number[] = [];
  const splits: Array<"train" | "test"> = subset === "all" ? ["train", "test"] : [subset];
  for (const split of splits) {
    for (const [label, polarity] of [
      [0, "neg"],
      [1, "pos"],
    ] as const) {
      const docs = found[split][polarity];
      docs.sort(compareNames);
      for (const doc of docs) {
        data.push(decoder.decode(doc.data));
        target.push(label);
      }
    }
  }
  if (data.length === 0) {
    throw new DataValidationError(
      `archive has no reviews in the ${subset} subset (expected aclImdb/<train|test>/<neg|pos>/*.txt)`
    );
  }
  return { data, target, classNames: [...IMDB_CLASSES] };
}

/**
 * Keeps `maxSamples` documents, taking the same number from every class (the first ones of each
 * class, one class after the other) so a small sample is not made of a single class. The
 * original order of the kept documents is preserved.
 */
function classBalancedFirst(parsed: ParsedText, maxSamples: number | undefined): ParsedText {
  const n = parsed.data.length;
  if (maxSamples === undefined || maxSamples >= n) return parsed;
  const k = parsed.classNames.length;
  const byClass: number[][] = Array.from({ length: k }, () => []);
  for (let i = 0; i < n; i++) (byClass[parsed.target[i] as number] as number[]).push(i);
  const take = new Array<number>(k).fill(0);
  let remaining = maxSamples;
  while (remaining > 0) {
    let progressed = false;
    for (let c = 0; c < k && remaining > 0; c++) {
      if ((take[c] as number) < (byClass[c] as number[]).length) {
        take[c] = (take[c] as number) + 1;
        remaining--;
        progressed = true;
      }
    }
    if (!progressed) break;
  }
  const keep: number[] = [];
  for (let c = 0; c < k; c++) keep.push(...(byClass[c] as number[]).slice(0, take[c] as number));
  keep.sort((a, b) => a - b);
  return {
    data: keep.map((i) => parsed.data[i] as string),
    target: keep.map((i) => parsed.target[i] as number),
    classNames: parsed.classNames,
  };
}

function takeFirst(json: TextJson, maxSamples: number | undefined): TextJson {
  const n = maxSamples === undefined ? json.data.length : Math.min(maxSamples, json.data.length);
  return { ...json, data: json.data.slice(0, n), target: json.target.slice(0, n) };
}

/**
 * Fetch the 20 Newsgroups text classification dataset.
 *
 * By default the official "bydate" archive (the one scikit-learn downloads, about 14 MB) is
 * fetched from `https://ndownloader.figshare.com/files/5975967` and unpacked: 11 314 training
 * and 7 532 test posts in 20 newsgroups. Labels follow the alphabetical order of the newsgroup
 * names; posts are ordered by newsgroup and then by file name, the training split before the
 * test split for `subset: 'all'`, and decoded as ISO-8859-1. Headers, footers and quotes are
 * kept (scikit-learn's `remove=()`) and nothing is shuffled. With `maxSamples` the same number
 * of posts is taken from every newsgroup.
 *
 * Pass `archiveUrl` to read the same archive from another location. Pass `baseUrl` to read a
 * JSON mirror instead: `<baseUrl>/20newsgroups_<subset>.json`, an object with `data` (array of
 * strings), `target` (array of integer labels) and `target_names` (array of class names), of
 * which the first `maxSamples` documents are kept.
 *
 * @param options - Configuration options
 * @returns TextDataset with newsgroup texts and `int32` labels
 * @throws {InvalidParameterError} If an option is invalid
 * @throws {DeepboxError} If the download fails or the response has the wrong layout, and
 *   `allowSyntheticFallback` is not set
 *
 * @example
 * ```ts
 * import { fetch20Newsgroups } from 'deepbox/datasets';
 *
 * const news = await fetch20Newsgroups({ subset: 'train' });
 * console.log(news.texts.length);   // 11314
 * console.log(news.classNames[0]);  // 'alt.atheism'
 * ```
 */
export async function fetch20Newsgroups(options: TextFetchOptions = {}): Promise<TextDataset> {
  validateTextOptions(options);
  const subset = options.subset ?? "all";
  const useJson = options.baseUrl !== undefined;
  const url = useJson
    ? joinUrl(options.baseUrl as string, `20newsgroups_${subset}.json`)
    : (options.archiveUrl ?? NEWSGROUPS_ARCHIVE_URL);

  let reason: string;
  try {
    let parsed: ParsedText;
    if (useJson) {
      const json = takeFirst(
        await fetchTextJson(url, options.timeout ?? DEFAULT_TIMEOUT_MS, undefined),
        options.maxSamples
      );
      parsed = {
        data: json.data,
        target: json.target,
        classNames: [...(json.target_names as string[])],
      };
    } else {
      const entries = await fetchTarEntries(url, options.timeout ?? DEFAULT_ARCHIVE_TIMEOUT_MS);
      parsed = classBalancedFirst(parseNewsgroupsArchive(entries, subset), options.maxSamples);
    }
    return {
      texts: parsed.data,
      target: tensor(parsed.target, { dtype: "int32" }),
      nClasses: parsed.classNames.length,
      classNames: parsed.classNames,
      description: `20 Newsgroups dataset (${subset}). ${parsed.data.length} documents across ${parsed.classNames.length} categories.`,
      isSynthetic: false,
    };
  } catch (err) {
    reason = err instanceof Error ? err.message : String(err);
  }

  if (!options.allowSyntheticFallback) {
    throw new DeepboxError(
      `Failed to fetch 20 Newsgroups dataset from ${url}: ${reason}. ` +
        "Pass { archiveUrl } or { baseUrl } pointing at a mirror, or set " +
        "{ allowSyntheticFallback: true } to use synthetic placeholder data instead."
    );
  }

  // Fallback: generate a minimal representative dataset.
  // This keeps the API usable without network access.
  const texts: string[] = [];
  const labels: number[] = [];
  const samplesPerClass = Math.ceil((options.maxSamples ?? 200) / NEWSGROUP_CATEGORIES.length);

  for (let c = 0; c < NEWSGROUP_CATEGORIES.length; c++) {
    const cat = NEWSGROUP_CATEGORIES[c] as string;
    for (let i = 0; i < samplesPerClass; i++) {
      texts.push(
        `Subject: Sample ${cat} post #${i}\n\nThis is a sample document from the ${cat} newsgroup category.`
      );
      labels.push(c);
    }
  }

  const n =
    options.maxSamples === undefined ? texts.length : Math.min(options.maxSamples, texts.length);

  return {
    texts: texts.slice(0, n),
    target: tensor(labels.slice(0, n), { dtype: "int32" }),
    nClasses: NEWSGROUP_CATEGORIES.length,
    classNames: [...NEWSGROUP_CATEGORIES],
    description:
      `20 Newsgroups dataset (fallback/synthetic). ${n} documents across 20 categories. ` +
      "Note: Could not fetch from remote mirror; using generated placeholder data.",
    isSynthetic: true,
  };
}

/**
 * Fetch the IMDB movie review sentiment dataset.
 *
 * By default the official archive (about 80 MB) is fetched from
 * `https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz` and unpacked: 25 000
 * training and 25 000 test reviews, half of them negative (label 0) and half positive
 * (label 1). The unlabeled `unsup` reviews are not read. Reviews are ordered train before test
 * for `subset: 'all'`, negative before positive, then by file name, and decoded as UTF-8
 * (the HTML line breaks of the original files are kept). With `maxSamples` the same number of
 * reviews is taken from each class.
 *
 * Pass `archiveUrl` to read the same archive from another location. Pass `baseUrl` to read a
 * JSON mirror instead: `<baseUrl>/imdb_<subset>.json`, an object with `data` (array of review
 * strings) and `target` (array of labels, 0 = negative, 1 = positive), of which the first
 * `maxSamples` documents are kept.
 *
 * @param options - Configuration options
 * @returns TextDataset with review texts and `int32` labels
 * @throws {InvalidParameterError} If an option is invalid
 * @throws {DeepboxError} If the download fails or the response has the wrong layout, and
 *   `allowSyntheticFallback` is not set
 *
 * @example
 * ```ts
 * import { fetchIMDB } from 'deepbox/datasets';
 *
 * const imdb = await fetchIMDB({ subset: 'train', maxSamples: 1000 });
 * console.log(imdb.texts.length);   // 1000
 * console.log(imdb.classNames);     // ['negative', 'positive']
 * ```
 */
export async function fetchIMDB(options: TextFetchOptions = {}): Promise<TextDataset> {
  validateTextOptions(options);
  const subset = options.subset ?? "all";
  const useJson = options.baseUrl !== undefined;
  const url = useJson
    ? joinUrl(options.baseUrl as string, `imdb_${subset}.json`)
    : (options.archiveUrl ?? IMDB_ARCHIVE_URL);

  let reason: string;
  try {
    let parsed: ParsedText;
    if (useJson) {
      const json = takeFirst(
        await fetchTextJson(url, options.timeout ?? DEFAULT_TIMEOUT_MS, IMDB_CLASSES.length),
        options.maxSamples
      );
      parsed = { data: json.data, target: json.target, classNames: [...IMDB_CLASSES] };
    } else {
      const entries = await fetchTarEntries(url, options.timeout ?? DEFAULT_ARCHIVE_TIMEOUT_MS);
      parsed = classBalancedFirst(parseImdbArchive(entries, subset), options.maxSamples);
    }
    return {
      texts: parsed.data,
      target: tensor(parsed.target, { dtype: "int32" }),
      nClasses: 2,
      classNames: parsed.classNames,
      description: `IMDB sentiment dataset (${subset}). ${parsed.data.length} reviews, binary sentiment.`,
      isSynthetic: false,
    };
  } catch (err) {
    reason = err instanceof Error ? err.message : String(err);
  }

  if (!options.allowSyntheticFallback) {
    throw new DeepboxError(
      `Failed to fetch IMDB dataset from ${url}: ${reason}. ` +
        "Pass { archiveUrl } or { baseUrl } pointing at a mirror, or set " +
        "{ allowSyntheticFallback: true } to use synthetic placeholder data instead."
    );
  }

  // Fallback: generate a minimal representative dataset.
  const texts: string[] = [];
  const labels: number[] = [];
  const samplesPerClass = Math.ceil((options.maxSamples ?? 100) / 2);

  for (let i = 0; i < samplesPerClass; i++) {
    texts.push(
      `This movie was terrible. The acting was poor and the plot made no sense. I would not recommend it. Review #${i}`
    );
    labels.push(0);
  }
  for (let i = 0; i < samplesPerClass; i++) {
    texts.push(
      `This movie was fantastic! Great acting, wonderful story, and beautiful cinematography. Highly recommended. Review #${i}`
    );
    labels.push(1);
  }

  const n =
    options.maxSamples === undefined ? texts.length : Math.min(options.maxSamples, texts.length);

  return {
    texts: texts.slice(0, n),
    target: tensor(labels.slice(0, n), { dtype: "int32" }),
    nClasses: 2,
    classNames: [...IMDB_CLASSES],
    description:
      `IMDB sentiment dataset (fallback/synthetic). ${n} reviews, binary sentiment. ` +
      "Note: Could not fetch from remote mirror; using generated placeholder data.",
    isSynthetic: true,
  };
}
