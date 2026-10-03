# Dataset Transforms & Samplers

> **View online:** https://deepbox.dev/examples/46-dataset-transforms-samplers

Dataset helpers around a training loop, applied to the iris dataset: `randomSplit`, `Subset`, `filterDataset`, `mapDataset`, `WeightedRandomSampler` and `SubsetRandomSampler`. The sampler part builds a skewed two-class dataset, weights each row by the inverse of its class size, and draws batches with a `DataLoader`.

## Deepbox Modules Used

| Module             | Features Used                                                                                                    |
| ------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `deepbox/datasets` | loadIris, Subset, randomSplit, mapDataset, filterDataset, DataLoader, WeightedRandomSampler, SubsetRandomSampler |

## Usage

```bash
npm run example:46
```

## Output

- Console output only: split sizes, subset indices, a mapped sample, and the class counts in each batch drawn by the weighted sampler.
- Every split and sampler takes a seed, so the output is the same on each run.

## Files

```text
46-dataset-transforms-samplers/
├── index.ts
└── README.md
```
