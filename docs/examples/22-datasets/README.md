# Built-in Datasets

> **View online:** https://deepbox.dev/examples/22-datasets

Load the 24 built-in datasets and the 6 synthetic generators, and print their shapes and metadata. The datasets ship with the package, so nothing is downloaded.

## Deepbox Modules Used

| Module             | Features Used                                                                                                                       |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `deepbox/datasets` | `loadIris`, `loadDigits`, `loadBreastCancer`, `loadDiabetes`, `loadLinnerud`, `loadHousingMini`, `makeClassification`, `makeRegression`, `makeBlobs`, `makeMoons`, `makeCircles`, `makeGaussianQuantiles`, and 18 more loaders |

## What It Shows

- Each loader returns an object with `data` (features), `target` (labels or values) and, for most datasets, `featureNames` and `targetNames`.
- The datasets are grouped by use: classic reference sets, tabular classification, non-linear classification, regression, clustering, integer-heavy tables, multi-output targets and a perfectly separable sanity check.
- The generators (`makeClassification`, `makeRegression`, `makeBlobs`, `makeMoons`, `makeCircles`, `makeGaussianQuantiles`) take an options object and return an `[X, y]` pair. `randomState` makes them repeatable.
- `fetch20Newsgroups` and `fetchIMDB` load the official text archives. They need network access, so this example does not call them.

## Usage

```bash
npm run example:22
```

## Output

Console output only, about 160 lines.

## Files

```
22-datasets/
├── index.ts     # Main entry point
└── README.md    # This file
```
