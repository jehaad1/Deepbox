# Sentiment Analysis

> **View online:** https://deepbox.dev/projects/06-sentiment-analysis

Classifies 500 synthetic reviews as positive or negative. Reviews are turned into TF-IDF vectors, and two classifiers are compared.

## Features

- Synthetic reviews built from fixed lists of positive, negative and neutral words, with 15% of the sentiment words flipped to the opposite class
- Vocabulary, bag-of-words counts and TF-IDF weights written by hand in `index.ts`. Project 08 shows the `TfidfVectorizer` from `deepbox/preprocess` instead
- `LogisticRegression` and `GaussianNB` on standardized features
- Accuracy, precision, recall, F1 and a confusion matrix
- Most positive and most negative words by smoothed count ratio

## What to expect

The data is regular, so both models score above 90%. That says little about real reviews, which have negation, sarcasm and a much larger vocabulary.

## Deepbox Modules Used

| Module               | Features Used                                                   |
| -------------------- | --------------------------------------------------------------- |
| `deepbox/ml`         | `LogisticRegression`, `GaussianNB`                              |
| `deepbox/preprocess` | `StandardScaler`, `trainTestSplit`                              |
| `deepbox/metrics`    | `accuracy`, `precision`, `recall`, `f1Score`, `confusionMatrix` |
| `deepbox/dataframe`  | `DataFrame` for the comparison table                            |
| `deepbox/ndarray`    | `tensor`, `toArray`                                             |
| `deepbox/plot`       | `Figure`, model comparison bar chart                            |

## Usage

```bash
npm run project:06
```

## Output

- Model comparison, confusion matrix and sample predictions on the console
- `output/model-comparison.svg`
