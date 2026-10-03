# Support Ticket Triage

> **View online:** https://deepbox.dev/projects/08-support-ticket-triage

Routes 640 synthetic support tickets to one of four queues (billing, account access, outage, bug report). Three text vectorizers are compared with logistic regression and Naive Bayes, and logistic regression is tuned with `GridSearchCV`.

## Features

- Synthetic tickets with priority, channel, team and a free-text description
- Text vectorization with `CountVectorizer`, `TfidfVectorizer` and `HashingVectorizer`, all with unigrams and bigrams
- Model selection with `GridSearchCV` over `C` and `maxIter`
- Cross-validation of the tuned model with `crossValidate`
- Weighted F1 and accuracy for each pipeline, using a time-ordered split (the last 20% of tickets is the test set)
- Confusion matrix plot with queue names as labels
- JSON summaries of the model comparison and the first terms of the TF-IDF vocabulary

## What to expect

Every ticket is built from one of four short phrase lists, so the queues are easy to separate. All three pipelines score 1.0 on the test set and in cross-validation, and the "best pipeline" line is a tie decided by list order. Real tickets are noisier. Use this project for the API calls, not for the scores.

## Deepbox Modules Used

| Module               | Features Used                                                          |
| -------------------- | ---------------------------------------------------------------------- |
| `deepbox/preprocess` | `CountVectorizer`, `TfidfVectorizer`, `HashingVectorizer`              |
| `deepbox/ml`         | `LogisticRegression`, `MultinomialNB`, `GridSearchCV`, `crossValidate` |
| `deepbox/metrics`    | `accuracy`, `f1Score`, `confusionMatrix`                               |
| `deepbox/dataframe`  | `DataFrame`, `groupBy`, string accessor (`str.contains`), JSON export  |
| `deepbox/plot`       | `figure`, `plotConfusionMatrix`, `saveFig`                             |
| `deepbox/ndarray`    | `tensor`                                                               |

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
