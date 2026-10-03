# DataLoader: Batching & Shuffling

> **View online:** https://deepbox.dev/examples/31-dataloader

`DataLoader` splits a dataset into batches, with optional shuffling, and is the usual way to feed a training loop. The example covers batching, a seeded shuffle, `reshuffleEachIteration`, `dropLast` and loading without labels.

## Deepbox Modules Used

| Module             | Features Used |
| ------------------ | ------------- |
| `deepbox/datasets` | DataLoader    |
| `deepbox/ndarray`  | tensor        |

## Usage

```bash
npm run example:31
```

## Output

- Console output only: batch shapes, the sample order of seeded shuffles, the number of batches with `dropLast`, and batches without labels.
- A seed alone repeats the same order in every epoch. Add `reshuffleEachIteration: true` for a new, still reproducible, order each epoch.
