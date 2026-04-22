# Flight Delay Prediction using Logistic Regression
**Jack McDonough, Xinwen Fang — BU MET CS 777**

A large-scale flight delay prediction pipeline built on Apache Spark and deployed on Google Cloud Dataproc. Two logistic regression implementations are provided: a hand-rolled full batch gradient descent model and an MLlib-based model using Spark's built-in L-BFGS optimizer.

---

## Files
| File | Description |
|---|---|
| `grad_desc_logreg.py` | Custom logistic regression via full batch gradient descent |
| `mllib_logreg.py` | Logistic regression using Apache Spark MLlib (L-BFGS optimizer) |

---

## Dataset

The dataset is sourced from the U.S. Department of Transportation Bureau of Transportation Statistics, covering all domestic flights from 1987 to 2025.

**Download from Kaggle:**
[https://www.kaggle.com/datasets/bojasthegreat/usdot-1987-2025-flights-dataset](https://www.kaggle.com/datasets/bojasthegreat/usdot-1987-2025-flights-dataset)

Once downloaded, upload the CSV to your GCS bucket:
```bash
gsutil cp flights_final.csv gs://YOUR_BUCKET/flights_final.csv
```

---

## Requirements

- Google Cloud account with Dataproc enabled
- A Dataproc cluster (2+ workers recommended)

### Upload scripts to GCS
```bash
gsutil cp grad_desc_logreg.py gs://YOUR_BUCKET/grad_desc_logreg.py
gsutil cp mllib_logreg.py gs://YOUR_BUCKET/mllib_logreg.py
```

---

## Running grad_desc_logreg.py

### Baseline (no weighting)
```bash
gcloud dataproc jobs submit pyspark gs://YOUR_BUCKET/grad_desc_logreg.py \
  --cluster=YOUR_CLUSTER \
  --region=YOUR_REGION \
  -- \
  --input=gs://YOUR_BUCKET/flights_final.csv \
  --output=gs://YOUR_BUCKET/output \
  --mode=baseline \
  --start-year=1988 \
  --max-iter=100 \
  --learning-rate=0.1 \
  --train-ratio=0.8 \
  --seed=42
```

### Weighted (recommended)
```bash
gcloud dataproc jobs submit pyspark gs://YOUR_BUCKET/grad_desc_logreg.py \
  --cluster=YOUR_CLUSTER \
  --region=YOUR_REGION \
  -- \
  --input=gs://YOUR_BUCKET/flights_final.csv \
  --output=gs://YOUR_BUCKET/output \
  --mode=weighted \
  --start-year=1988 \
  --max-iter=100 \
  --learning-rate=0.05 \
  --train-ratio=0.8 \
  --seed=42
```

### Arguments
| Argument | Default | Description |
|---|---|---|
| `--input` | required | GCS path to input CSV |
| `--output` | required | GCS path for output |
| `--mode` | `weighted` | `baseline` or `weighted` |
| `--start-year` | `2022` | First year of data to include |
| `--max-iter` | `100` | Number of gradient descent iterations |
| `--learning-rate` | `0.1` | Gradient descent step size |
| `--train-ratio` | `0.8` | Temporal train/test split ratio |
| `--seed` | `42` | Random seed |
| `--save-predictions` | `false` | Save test predictions to parquet |

---

## Running mllib_logreg.py

### Baseline (no weighting)
```bash
gcloud dataproc jobs submit pyspark gs://YOUR_BUCKET/mllib_logreg.py \
  --cluster=YOUR_CLUSTER \
  --region=YOUR_REGION \
  -- \
  --input=gs://YOUR_BUCKET/flights_final.csv \
  --output=gs://YOUR_BUCKET/output \
  --mode=baseline \
  --start-year=1988 \
  --max-iter=20 \
  --train-ratio=0.8 \
  --seed=42
```

### Weighted (recommended)
```bash
gcloud dataproc jobs submit pyspark gs://YOUR_BUCKET/mllib_logreg.py \
  --cluster=YOUR_CLUSTER \
  --region=YOUR_REGION \
  -- \
  --input=gs://YOUR_BUCKET/flights_final.csv \
  --output=gs://YOUR_BUCKET/output \
  --mode=weighted \
  --start-year=1988 \
  --max-iter=20 \
  --train-ratio=0.8 \
  --seed=42
```

### Arguments
| Argument | Default | Description |
|---|---|---|
| `--input` | required | GCS path to input CSV |
| `--output` | required | GCS path for output |
| `--mode` | `weighted` | `baseline` or `weighted` |
| `--start-year` | `2022` | First year of data to include |
| `--max-iter` | `20` | Max L-BFGS iterations |
| `--train-ratio` | `0.8` | Random train/test split ratio |
| `--seed` | `42` | Random seed for split |
| `--save-predictions` | `false` | Save test predictions to parquet |

---

## Output

Both scripts write the following to the output path:

| Output | Description |
|---|---|
| `{mode}_metrics_json/` | Full metrics: AUC, accuracy, F1, weights, confusion matrix |
| `{mode}_confusion_csv/` | Confusion matrix as CSV |
| `{mode}_pipeline_model/` | Saved MLlib pipeline model (mllib_logreg.py only) |
| `{mode}_test_predictions_parquet/` | Per-row label, prediction, probability (requires `--save-predictions`) |

---

## Key Differences Between the Two Scripts

| | `grad_desc_logreg.py` | `mllib_logreg.py` |
|---|---|---|
| Optimizer | Full batch gradient descent | L-BFGS (MLlib default) |
| Feature encoding | Frequency encoding | One-hot encoding |
| Train/test split | Temporal (by year) | Random split |
| Iterations needed | 100+ | 20 |
| AUC achieved | 0.367 | 0.664 |

---

## Notes

- COVID years (March 2020 – June 2022) are excluded automatically in both scripts
- A delay is defined as a departure delay of 15 minutes or more (label=1), otherwise label=0
- Class imbalance (~80% on-time flights) is handled via inverse frequency weighting in weighted mode
- `grad_desc_logreg.py` uses a temporal split — training on earlier years and testing on later years — which better simulates real-world deployment
- `mllib_logreg.py` uses a random split with a fixed seed for reproducibility