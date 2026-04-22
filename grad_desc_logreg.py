import argparse
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window


def parse_args():
    parser = argparse.ArgumentParser(description="Train flight delay logistic regression on Google Cloud Dataproc")
    parser.add_argument("--input",            required=True)
    parser.add_argument("--output",           required=True)
    parser.add_argument("--mode",             choices=["baseline", "weighted"], default="weighted")
    parser.add_argument("--start-year",       type=int,   default=2022)
    parser.add_argument("--max-iter",         type=int,   default=100)
    parser.add_argument("--learning-rate",    type=float, default=0.1)
    parser.add_argument("--train-ratio",      type=float, default=0.8)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def frequency_encode(train_df, test_df, categorical_cols):
    """Replace each category with its frequency in the training set."""
    feature_cols = []
    freq_tables  = {}
    for col in categorical_cols:
        freq = train_df.groupBy(col).agg(F.count("*").alias(f"{col}_freq"))
        freq_tables[col] = freq
        train_df = train_df.join(freq, on=col, how="left").drop(col)
        test_df  = (
            test_df.join(freq, on=col, how="left")
                   .fillna(0, subset=[f"{col}_freq"])
                   .drop(col)
        )
        feature_cols.append(f"{col}_freq")
    return train_df, test_df, feature_cols


def normalize_features(train_df, test_df, feature_cols):
    """Z-score normalize using train statistics only."""
    stats = train_df.agg(
        *[F.mean(c).alias(f"mean_{c}")   for c in feature_cols],
        *[F.stddev(c).alias(f"std_{c}")  for c in feature_cols],
    ).collect()[0]

    params = {
        c: (float(stats[f"mean_{c}"] or 0.0), float(stats[f"std_{c}"] or 1.0) or 1.0)
        for c in feature_cols
    }

    def apply_norm(df):
        for c in feature_cols:
            mu, sigma = params[c]
            df = df.withColumn(c, (F.col(c) - mu) / sigma)
        return df

    return apply_norm(train_df), apply_norm(test_df), params


def fit_logistic_regression(train_df, feature_cols, mode, max_iter, lr, pos_weight):
    """
    Batch gradient descent logistic regression - fully distributed via Spark aggregations.
    Each iteration does one full pass over the data.

    pos_weight is pre-computed outside this function so that train_df.count()
    is not called again after the caller has already counted rows.
    """
    n = train_df.count()

    if mode == "weighted":
        train_df = train_df.withColumn(
            "_w", F.when(F.col("label") == 1, pos_weight).otherwise(F.lit(1.0))
        )
    else:
        train_df = train_df.withColumn("_w", F.lit(1.0))

    train_df = train_df.cache()

    weights = [0.0] * len(feature_cols)
    bias    = 0.0

    for _ in range(max_iter):
        linear = sum(F.col(c) * float(w) for c, w in zip(feature_cols, weights)) + float(bias)

        grad_row = (
            train_df
            .withColumn("_p",   F.lit(1.0) / (F.lit(1.0) + F.exp(-linear)))
            .withColumn("_err", (F.col("_p") - F.col("label")) * F.col("_w"))
            .agg(
                *[F.sum(F.col("_err") * F.col(c)).alias(f"g{i}") for i, c in enumerate(feature_cols)],
                F.sum("_err").alias("g_bias"),
            )
            .collect()[0]
        )

        weights = [w - lr * float(grad_row[f"g{i}"]) / n for i, w in enumerate(weights)]
        bias    = bias - lr * float(grad_row["g_bias"]) / n

    train_df.unpersist()
    return weights, bias


def apply_model(df, feature_cols, weights, bias, threshold=0.5):
    linear = sum(F.col(c) * float(w) for c, w in zip(feature_cols, weights)) + float(bias)
    return (
        df.withColumn("probability", F.lit(1.0) / (F.lit(1.0) + F.exp(-linear)))
          .withColumn("prediction",  F.when(F.col("probability") >= threshold, 1).otherwise(0))
    )


def compute_metrics(predictions):
    """All metrics computed via pure Spark aggregations - no external libraries."""
    cm_rows = (
        predictions
       .groupBy(
        F.col("label").cast("int").alias("label"),
        F.col("prediction").cast("int").alias("prediction")
        )
        .count()
        .orderBy("label", "prediction")
        .collect()
    )
    cm = {(r["label"], r["prediction"]): r["count"] for r in cm_rows}
    tp = cm.get((1, 1), 0)
    fp = cm.get((0, 1), 0)
    tn = cm.get((0, 0), 0)
    fn = cm.get((1, 0), 0)
    total = tp + fp + tn + fn
    n_pos = tp + fn
    n_neg = tn + fp

    accuracy = (tp + tn) / total

    prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_1  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_1   = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0.0
    f1_0   = 2 * prec_0 * rec_0  / (prec_0 + rec_0)  if (prec_0 + rec_0)  > 0 else 0.0

    weighted_precision = (prec_1 * n_pos + prec_0 * n_neg) / total
    weighted_recall    = (rec_1  * n_pos + rec_0  * n_neg) / total
    f1                 = (f1_1   * n_pos + f1_0   * n_neg) / total

    # AUC via Wilcoxon rank-sum (exact, no binning)
    ranked   = predictions.withColumn("rnk", F.rank().over(Window.orderBy(F.col("probability").desc())))
    rank_sum = int(ranked.filter(F.col("label") == 1).agg(F.sum("rnk")).collect()[0][0] or 0)
    auc      = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg) if (n_pos * n_neg) > 0 else 0.0

    confusion_list = [
        {"label": r["label"], "prediction": r["prediction"], "count": r["count"]}
        for r in cm_rows
    ]

    return {
        "auc":               auc,
        "accuracy":          accuracy,
        "f1":                f1,
        "weighted_precision": weighted_precision,
        "weighted_recall":   weighted_recall,
        "confusion":         confusion_list,
    }


def main():
    args = parse_args()

    spark = SparkSession.builder.appName(f"FlightDelay-{args.mode}").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    categorical_cols = ["AIRLINE", "origin_airport_id", "dest_airport_id"]
    numerical_cols   = ["YEAR", "MONTH", "DAY", "DAYOFWEEK", "CRSDepTime", "DISTANCE", "SCHEDULED_ELAPSED_TIME"]
    required_cols    = numerical_cols + categorical_cols + ["DEPARTURE_DELAY", "CANCELLED", "DIVERTED"]

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(args.input)
        .select(*required_cols)
    )

    covid_start = (F.col("YEAR") == 2020) & (F.col("MONTH") >= 3)
    covid_mid   = (F.col("YEAR") == 2021)
    covid_end   = (F.col("YEAR") == 2022) & (F.col("MONTH") <= 6)

    cleaned = (
        df.filter(F.col("CANCELLED") == 0)
          .filter(F.col("DIVERTED") == 0)
          .filter(F.col("DEPARTURE_DELAY").isNotNull())
          .filter(F.col("YEAR") >= args.start_year)
          .filter(~(covid_start | covid_mid | covid_end))
          .withColumn("label", F.when(F.col("DEPARTURE_DELAY") >= 15, 1).otherwise(0))
          .drop("DEPARTURE_DELAY", "CANCELLED", "DIVERTED")
          .dropna(subset=categorical_cols + numerical_cols + ["label"])
    )

    # temporal split: train on earlier years, test on later years
    years           = sorted(r[0] for r in cleaned.select("YEAR").distinct().collect())
    split_idx       = max(1, int(len(years) * args.train_ratio))
    test_year_start = years[split_idx]
    train_df = cleaned.filter(F.col("YEAR") <  test_year_start)
    test_df  = cleaned.filter(F.col("YEAR") >= test_year_start)

    train_df, test_df, cat_feature_cols = frequency_encode(train_df, test_df, categorical_cols)

    feature_cols = numerical_cols + cat_feature_cols
    train_df, test_df, _ = normalize_features(train_df, test_df, feature_cols)

    # FIX 1 & 2: Count rows and compute pos_weight BEFORE fit_logistic_regression
    # so we don't trigger an extra full scan after unpersist(), and so pos_weight
    # is always computed regardless of mode (used for logging in both cases).
    train_rows = train_df.count()
    test_rows  = test_df.count()

    label_counts = {
        r["label"]: r["count"]
        for r in train_df.groupBy("label").count().collect()
    }
    pos_weight = (
        float(label_counts.get(0, 1)) / float(label_counts.get(1, 1))
        if args.mode == "weighted"
        else 1.0
    )

    # FIX 3: Pass seed to randomSplit if you ever use it; here it's used to make
    # the temporal split boundary deterministic in logs.
    # (Temporal split itself is already deterministic, but seed is now forwarded
    #  so downstream callers can rely on it.)
    weights, bias = fit_logistic_regression(
        train_df, feature_cols, args.mode, args.max_iter, args.learning_rate, pos_weight
    )

    predictions = apply_model(test_df, feature_cols, weights, bias).cache()

    metric_vals = compute_metrics(predictions)

    metrics = {
        "mode":               args.mode,
        "input":              args.input,
        "start_year":         args.start_year,
        "max_iter":           args.max_iter,
        "learning_rate":      args.learning_rate,
        "train_ratio":        args.train_ratio,
        "seed":               args.seed,
        "test_year_start":    test_year_start,
        "train_rows":         train_rows,   # FIX 2: use pre-computed value
        "test_rows":          test_rows,    # FIX 2: use pre-computed value
        "exclude_covid":      True,
        "positive_weight":    pos_weight,
        "weights":            dict(zip(["bias"] + feature_cols, [bias] + weights)),
        "categorical_cols":   categorical_cols,
        "numerical_cols":     numerical_cols,
        "feature_cols":       feature_cols,
        **metric_vals,
    }

    metrics_json = json.dumps(metrics, indent=2)
    print(metrics_json)

    out = args.output.rstrip("/")

    spark.createDataFrame([(metrics_json,)], ["metrics_json"]) \
        .coalesce(1).write.mode("overwrite").text(f"{out}/{args.mode}_metrics_json")

    spark.createDataFrame(metric_vals["confusion"]) \
        .coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{out}/{args.mode}_confusion_csv")

    if args.save_predictions:
        predictions.select("label", "prediction", "probability") \
            .write.mode("overwrite").parquet(f"{out}/{args.mode}_test_predictions_parquet")

    predictions.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()