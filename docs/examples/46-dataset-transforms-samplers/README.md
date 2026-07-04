# Dataset Transforms & Samplers

> **View online:** https://deepbox.dev/examples/46-dataset-transforms-samplers

A data-pipeline example for the v1.0.0 dataset helpers around training loops: `Subset`, `randomSplit`, `mapDataset`, `filterDataset`, `WeightedRandomSampler`, and `SubsetRandomSampler`.

## Deepbox Modules Used

| Module             | Features Used                                                                 |
| ------------------ | ----------------------------------------------------------------------------- |
| `deepbox/datasets` | loadIris, Subset, randomSplit, mapDataset, filterDataset, DataLoader, samplers |

## Usage

```bash
npm run example:46
```

## Output

- Console walkthrough of deterministic splitting, filtering, mapping, and sampler-driven batching

## Architecture

```text
46-dataset-transforms-samplers/
├── index.ts
└── README.md
```
