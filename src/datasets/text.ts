/**
 * Text dataset loaders (20 Newsgroups, IMDB) via fetch.
 *
 * Downloads and parses standard text classification datasets from
 * public mirrors. Returns data as string arrays with integer labels.
 *
 * @module datasets/text
 * @see {@link https://deepbox.dev/docs/datasets-builtin | Deepbox documentation}
 */

import { DeepboxError } from "../core/errors";
import { type Tensor, tensor } from "../ndarray";

/**
 * Result of a text dataset fetch.
 */
export type TextDataset = {
  /** Array of text documents */
  readonly texts: string[];
  /** Integer label tensor of shape (nSamples,) */
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
 * Fetch the 20 Newsgroups text classification dataset.
 *
 * Downloads the dataset from a public mirror. Returns up to ~20,000
 * newsgroup posts across 20 categories.
 *
 * @param options - Configuration options
 * @returns TextDataset with newsgroup texts and labels
 *
 * @example
 * ```ts
 * import { fetch20Newsgroups } from 'deepbox/datasets';
 *
 * const news = await fetch20Newsgroups({ subset: 'train' });
 * console.log(news.texts.length);   // ~11314
 * console.log(news.classNames);     // ['alt.atheism', ...]
 * ```
 */
export async function fetch20Newsgroups(
  options: {
    /** Which subset: 'train', 'test', or 'all' (default: 'all') */
    readonly subset?: "train" | "test" | "all";
    /** Maximum number of samples to load */
    readonly maxSamples?: number;
    /** Base URL override for custom mirror */
    readonly baseUrl?: string;
    /** When true, return synthetic placeholder data on fetch failure. Default: false */
    readonly allowSyntheticFallback?: boolean;
  } = {}
): Promise<TextDataset> {
  const subset = options.subset ?? "all";
  const base =
    options.baseUrl ??
    "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/";

  // 20newsgroups is typically distributed as a tarball.
  // For simplicity and reliability, we use a JSON-formatted mirror.
  // If the user provides a custom URL, we try that.
  // Otherwise, we build a synthetic dataset from the sklearn bundled data.

  const NEWSGROUP_CATEGORIES = [
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

  // Try fetching from a JSON endpoint
  const url = `${base}20newsgroups_${subset}.json`;

  try {
    const resp = await fetch(url);
    if (resp.ok) {
      const json = (await resp.json()) as {
        data: string[];
        target: number[];
        target_names: string[];
      };
      const n = options.maxSamples
        ? Math.min(options.maxSamples, json.data.length)
        : json.data.length;
      return {
        texts: json.data.slice(0, n),
        target: tensor(json.target.slice(0, n)),
        nClasses: json.target_names.length,
        classNames: json.target_names,
        description: `20 Newsgroups dataset (${subset}). ${n} documents across ${json.target_names.length} categories.`,
        isSynthetic: false,
      };
    }
  } catch {
    // Fall through to synthetic generation
  }

  if (!options.allowSyntheticFallback) {
    throw new DeepboxError(
      "Failed to fetch 20 Newsgroups dataset from remote mirror. " +
        "Set { allowSyntheticFallback: true } to use synthetic placeholder data instead."
    );
  }

  // Fallback: generate a minimal representative dataset
  // This ensures the API works even without network access
  const texts: string[] = [];
  const labels: number[] = [];
  const samplesPerClass = Math.ceil((options.maxSamples ?? 200) / NEWSGROUP_CATEGORIES.length);

  for (let c = 0; c < NEWSGROUP_CATEGORIES.length; c++) {
    const cat = NEWSGROUP_CATEGORIES[c]!;
    for (let i = 0; i < samplesPerClass; i++) {
      texts.push(
        `Subject: Sample ${cat} post #${i}\n\nThis is a sample document from the ${cat} newsgroup category.`
      );
      labels.push(c);
    }
  }

  const n = options.maxSamples ? Math.min(options.maxSamples, texts.length) : texts.length;

  return {
    texts: texts.slice(0, n),
    target: tensor(labels.slice(0, n)),
    nClasses: NEWSGROUP_CATEGORIES.length,
    classNames: NEWSGROUP_CATEGORIES,
    description:
      `20 Newsgroups dataset (fallback/synthetic). ${n} documents across 20 categories. ` +
      "Note: Could not fetch from remote mirror; using generated placeholder data.",
    isSynthetic: true,
  };
}

/**
 * Fetch the IMDB movie review sentiment dataset.
 *
 * Downloads the dataset from a public mirror. Returns up to 50,000
 * movie reviews with binary sentiment labels (0=negative, 1=positive).
 *
 * @param options - Configuration options
 * @returns TextDataset with review texts and labels
 *
 * @example
 * ```ts
 * import { fetchIMDB } from 'deepbox/datasets';
 *
 * const imdb = await fetchIMDB({ subset: 'train' });
 * console.log(imdb.texts.length);   // ~25000
 * console.log(imdb.classNames);     // ['negative', 'positive']
 * ```
 */
export async function fetchIMDB(
  options: {
    /** Which subset: 'train', 'test', or 'all' (default: 'all') */
    readonly subset?: "train" | "test" | "all";
    /** Maximum number of samples to load */
    readonly maxSamples?: number;
    /** Base URL override for custom mirror */
    readonly baseUrl?: string;
    /** When true, return synthetic placeholder data on fetch failure. Default: false */
    readonly allowSyntheticFallback?: boolean;
  } = {}
): Promise<TextDataset> {
  const subset = options.subset ?? "all";
  const base = options.baseUrl ?? "https://ai.stanford.edu/~amaas/data/sentiment/";

  const IMDB_CLASSES = ["negative", "positive"];

  // Try fetching from a JSON endpoint
  const url = `${base}imdb_${subset}.json`;

  try {
    const resp = await fetch(url);
    if (resp.ok) {
      const json = (await resp.json()) as {
        data: string[];
        target: number[];
      };
      const n = options.maxSamples
        ? Math.min(options.maxSamples, json.data.length)
        : json.data.length;
      return {
        texts: json.data.slice(0, n),
        target: tensor(json.target.slice(0, n)),
        nClasses: 2,
        classNames: IMDB_CLASSES,
        description: `IMDB sentiment dataset (${subset}). ${n} reviews, binary sentiment.`,
        isSynthetic: false,
      };
    }
  } catch {
    // Fall through to synthetic generation
  }

  if (!options.allowSyntheticFallback) {
    throw new DeepboxError(
      "Failed to fetch IMDB dataset from remote mirror. " +
        "Set { allowSyntheticFallback: true } to use synthetic placeholder data instead."
    );
  }

  // Fallback: generate a minimal representative dataset
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

  const n = options.maxSamples ? Math.min(options.maxSamples, texts.length) : texts.length;

  return {
    texts: texts.slice(0, n),
    target: tensor(labels.slice(0, n)),
    nClasses: 2,
    classNames: IMDB_CLASSES,
    description:
      `IMDB sentiment dataset (fallback/synthetic). ${n} reviews, binary sentiment. ` +
      "Note: Could not fetch from remote mirror; using generated placeholder data.",
    isSynthetic: true,
  };
}
