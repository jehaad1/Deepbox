# Running Deepbox Examples

> **Browse online:** https://deepbox.dev/examples

This directory contains 33 self-contained examples demonstrating Deepbox capabilities.

## Prerequisites

- Node.js >= 24.13.0
- npm installed
- Deepbox library built (run `npm run build` from parent directory)

## Running Examples

All examples should be run from the **root Deepbox directory** (not from the examples directory).

### Individual Examples

From the root Deepbox directory, run:

```bash
# Getting Started
npm run example:00   # Quick Start
npm run example:01   # Tensor Basics
npm run example:02   # Tensor Operations

# DataFrames
npm run example:03   # Data Analysis & Visualization
npm run example:04   # DataFrame Basics
npm run example:05   # DataFrame GroupBy

# Classical Machine Learning
npm run example:06   # ML Pipeline
npm run example:07   # Linear Regression
npm run example:08   # Logistic Regression
npm run example:09   # Ridge & Lasso
npm run example:10   # Advanced ML Models
npm run example:11   # Tree & Ensemble Models
npm run example:12   # Complete Pipeline

# Neural Networks & Autograd
npm run example:13   # Neural Network Training
npm run example:14   # Autograd
npm run example:15   # Activation Functions

# Optimization
npm run example:16   # LR Schedulers

# Preprocessing
npm run example:17   # Encoders
npm run example:18   # Scalers

# Statistics & Linear Algebra
npm run example:19   # Statistics
npm run example:20   # Linear Algebra

# Utilities
npm run example:21   # Random Sampling
npm run example:22   # Datasets
npm run example:23   # Cross-Validation
npm run example:24   # Metrics

# Visualization & Special Topics
npm run example:25   # Plotting
npm run example:26   # Sparse Matrices

# Deep Learning Layers
npm run example:27   # CNN Layers (Conv1d, Conv2d, Pooling)
npm run example:28   # RNN, LSTM, GRU
npm run example:29   # Attention & Transformer
npm run example:30   # Normalization & Dropout
npm run example:31   # DataLoader
npm run example:32   # Module System
```

### Run All Examples

To run all examples sequentially:

```bash
npm run examples:all
```

## Example Descriptions

### 00 – Quick Start

- **Duration**: ~1 second
- **Features**: Tensors, DataFrames, linear regression
- **Output**: Console only

### 03 – Data Analysis

- **Duration**: ~2 seconds
- **Features**: DataFrames, statistics, groupby, visualization
- **Output**: 4 SVG visualizations in `03-data-analysis/output/`

### 06 – ML Pipeline

- **Duration**: ~3 seconds
- **Features**: Classification, regression, model comparison, cross-validation
- **Output**: 1 SVG visualization in `06-ml-pipeline/output/`

### 12 – Complete Pipeline

- **Duration**: ~2 seconds
- **Features**: End-to-end ML workflow with visualization
- **Output**: 1 SVG visualization in `12-complete-pipeline/output/`

### 15 – Activation Functions

- **Duration**: ~1 second
- **Features**: 9 activation functions with comparison plot
- **Output**: 1 SVG visualization in `15-activation-functions/output/`

### 25 – Plotting

- **Duration**: ~1 second
- **Features**: 5 plot types (line, scatter, bar, histogram, heatmap)
- **Output**: 5 SVG visualizations in `25-plotting/output/`

## Output Files

Examples that generate visualizations write SVG files to their respective `output/` directories. All visualizations are publication-ready.

## Troubleshooting

### "Cannot find module 'deepbox/...'"

Make sure you:

1. Are running from the **root Deepbox directory** (not from `docs/examples/`)
2. Have built the library: `npm run build`
3. Are using the npm scripts (not running tsx directly)

### Examples run but show errors

Ensure the library is properly built:

```bash
npm run build
npm run example:00  # Test with quick start
```

## Technical Details

- All examples use TypeScript with tsx for execution
- Path resolution configured via `docs/examples/tsconfig.json`
- Examples import from source files during development
- Production-ready code with proper error handling

## Development

To add a new example:

1. Create a new directory: `docs/examples/NN-example-name/`
2. Add `index.ts` as the entry point
3. Add `README.md` with description and modules used
4. Create `output/` directory with `.gitkeep`
5. Add npm script to root `package.json`:
   ```json
   "example:NN": "npx tsx --tsconfig docs/examples/tsconfig.json docs/examples/NN-example-name/index.ts"
   ```
6. Update this README and `docs/examples/README.md`

## License

MIT – See LICENSE file in parent directory
