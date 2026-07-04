# Model Selection & Pipeline

> **View online:** https://deepbox.dev/examples/37-model-selection-pipeline

Automated model selection and ML pipelines new in v1.0.0: GridSearchCV, RandomizedSearchCV, Pipeline, ColumnTransformer, and cross-validation.

## Deepbox Modules Used

| Module              | Features Used                                                           |
| ------------------- | ----------------------------------------------------------------------- |
| `deepbox/ml`        | Pipeline, GridSearchCV, RandomizedSearchCV, ColumnTransformer, cross_validate |
| `deepbox/preprocess`| StandardScaler, MinMaxScaler, trainTestSplit                            |
| `deepbox/datasets`  | makeClassification                                                      |

## Usage

```bash
npm run example:37
```

## Architecture

```
37-model-selection-pipeline/
├── index.ts     # Main entry point
└── README.md    # This file
```
