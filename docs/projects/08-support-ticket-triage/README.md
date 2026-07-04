# Support Ticket Triage

> **View online:** https://deepbox.dev/projects/08-support-ticket-triage

A production-style text operations pipeline for support routing, using the v1.0.0 text preprocessing stack, model selection helpers, and reporting outputs.

## Features

- **Synthetic ticket stream** with priority, channel, team, and free-text issue descriptions
- **Text vectorization** with `CountVectorizer`, `TfidfVectorizer`, and `HashingVectorizer`
- **Model selection** with `GridSearchCV`
- **Classifier comparison** across logistic regression and Naive Bayes baselines
- **Operational reporting** via DataFrame summaries and exported JSON artifacts
- **Visualization** with a labeled confusion matrix

## Deepbox Modules Used

| Module               | Features Used                                                                 |
| -------------------- | ----------------------------------------------------------------------------- |
| `deepbox/preprocess` | CountVectorizer, TfidfVectorizer, HashingVectorizer                           |
| `deepbox/ml`         | LogisticRegression, MultinomialNB, GridSearchCV, cross_validate               |
| `deepbox/metrics`    | accuracy, f1Score, confusionMatrix                                            |
| `deepbox/dataframe`  | DataFrame, string accessor, JSON export                                       |
| `deepbox/plot`       | plotConfusionMatrix, saveFig                                                  |
| `deepbox/ndarray`    | tensor                                                                        |

## Usage

```bash
npm run project:08
```

## Output

- `output/best-model-confusion-matrix.svg`
- `output/model-comparison.json`
- `output/tfidf-vocabulary-preview.json`

## Architecture

```text
08-support-ticket-triage/
├── index.ts
├── README.md
└── output/
```
