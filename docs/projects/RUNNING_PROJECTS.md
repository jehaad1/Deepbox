# Running Deepbox Example Projects

This directory contains 6 enterprise-grade projects demonstrating Deepbox capabilities.

## Prerequisites

- Node.js >= 24.13.0
- npm installed
- Deepbox library built (run `npm run build` from parent directory)

## Running Projects

All projects should be run from the **parent directory** (not from the projects directory).

### Individual Projects

From the root Deepbox directory, run:

```bash
# Project 01: Financial Risk Analysis
npm run project:01

# Project 02: Neural Network Image Classifier
npm run project:02

# Project 03: Customer Churn Prediction
npm run project:03

# Project 04: Stock Price Forecasting
npm run project:04

# Project 05: Movie Recommendation Engine
npm run project:05

# Project 06: Sentiment Analysis System
npm run project:06
```

### Run All Projects

To run all projects sequentially:

```bash
npm run projects:all
```

## Project Descriptions

### 01 - Financial Risk Analysis

- **Duration**: ~15 seconds
- **Features**: Portfolio optimization, VaR/CVaR calculation, Monte Carlo simulation, stress testing
- **Output**: 3 SVG visualizations in `01-financial-risk-analysis/output/`

### 02 - Neural Network Image Classifier

- **Duration**: ~30 seconds
- **Features**: MLP training, digit classification, confusion matrix analysis
- **Output**: 2 SVG visualizations in `02-neural-image-classifier/output/`
- **Note**: Low accuracy expected due to random weight initialization (no gradient-based optimization)

### 03 - Customer Churn Prediction

- **Duration**: ~5 seconds
- **Features**: Multiple ML classifiers, cross-validation, confusion matrix
- **Output**: 2 SVG visualizations in `03-customer-churn-prediction/output/`

### 04 - Stock Price Forecasting

- **Duration**: ~5 seconds
- **Features**: Time series analysis, technical indicators, regression models
- **Output**: 2 SVG visualizations in `04-stock-price-forecasting/output/`

### 05 - Movie Recommendation Engine

- **Duration**: ~10 seconds
- **Features**: Collaborative filtering, K-means clustering, PCA dimensionality reduction
- **Output**: 2 SVG visualizations in `05-recommendation-engine/output/`

### 06 - Sentiment Analysis System

- **Duration**: ~3 seconds
- **Features**: TF-IDF vectorization, Logistic Regression, Naive Bayes
- **Output**: 1 SVG visualization in `06-sentiment-analysis/output/`

## Output Files

Each project generates:

- Console output with detailed analysis
- SVG visualization files in respective `output/` directories
- All visualizations are publication-ready

## Troubleshooting

### "Cannot find module 'deepbox/...'"

Make sure you:

1. Are running from the **parent directory** (not from `docs/projects/`)
2. Have built the library: `npm run build`
3. Are using the npm scripts (not running tsx directly)

### Projects run but show errors

Ensure the library is properly built:

```bash
cd /path/to/Deepbox
npm run build
npm run project:01  # Test with project 01
```

## Technical Details

- All projects use TypeScript with tsx for execution
- Path resolution configured via `docs/projects/tsconfig.json`
- Projects import from source files during development
- Production-ready code with proper error handling
- Enterprise-level quality with comprehensive documentation

## Development

To add a new project:

1. Create a new directory: `docs/projects/0X-project-name/`
2. Add `index.ts` as the entry point
3. Add npm script to parent `package.json`:
   ```json
   "project:0X": "npx tsx --tsconfig docs/projects/tsconfig.json docs/projects/0X-project-name/index.ts"
   ```
4. Update this README with project description

## License

MIT - See LICENSE file in parent directory
