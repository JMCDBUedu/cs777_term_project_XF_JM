import argparse
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Train flight delay logistic regression on Google Cloud Dataproc")
    parser.add_argument("--input", required=True, help="Input CSV path in gs://... or local path")
    parser.add_argument("--output", required=True, help="Output directory in gs://... or local path")
    parser.add_argument("--mode", choices=["baseline", "weighted"], default="weighted", help="Training mode")
    parser.add_argument("--start-year", type=int, default=2022, help="Keep rows where YEAR >= start-year")
    parser.add_argument("--max-iter", type=int, default=20, help="Logistic regression max iterations")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save-predictions", action="store_true", help="Save scored test predictions")
    return parser.parse_args()


def build_pipeline(categorical_cols, numerical_cols, mode, max_iter):
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep")
        for c in categorical_cols
    ]

    encoders = [
        OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_vec")
        for c in categorical_cols
    ]

    feature_cols = [f"{c}_vec" for c in categorical_cols] + numerical_cols

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="keep"
    )

    if mode == "weighted":
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            weightCol="classWeightCol",
            maxIter=max_iter
        )
    else:
        lr = LogisticRegression(
            featuresCol="features",
            labelCol="label",
            maxIter=max_iter
        )

    return Pipeline(stages=indexers + encoders + [assembler, lr])


def main():
                                                                       
    args = parse_args()

    spark = SparkSession.builder.appName(f"FlightDelay-{args.mode}").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    categorical_cols = ["AIRLINE", "origin_airport_id", "dest_airport_id"]
    numerical_cols = ["YEAR", "MONTH", "DAY", "DAYOFWEEK", "CRSDepTime", "DISTANCE", "SCHEDULED_ELAPSED_TIME"]

    required_cols = numerical_cols + categorical_cols + ["DEPARTURE_DELAY", "CANCELLED", "DIVERTED"]

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
    in_covid    = covid_start | covid_mid | covid_end

    cleaned = (
        df.filter(F.col("CANCELLED") == 0)
          .filter(F.col("DIVERTED") == 0)
          .filter(F.col("DEPARTURE_DELAY").isNotNull())
          .filter(F.col("YEAR") >= args.start_year)
          .filter(~in_covid)
          .withColumn("label", F.when(F.col("DEPARTURE_DELAY") >= 15, F.lit(1)).otherwise(F.lit(0)))
          .drop("DEPARTURE_DELAY", "CANCELLED", "DIVERTED")
    )

    cleaned = cleaned.dropna(subset=categorical_cols + numerical_cols + ["label"])

    if args.mode == "weighted":
        counts = {row["label"]: row["count"] for row in cleaned.groupBy("label").count().collect()}
        negative_count = counts.get(0, 0)
        positive_count = counts.get(1, 0)
        if negative_count == 0 or positive_count == 0:
            raise ValueError(f"Invalid class counts: {counts}")
        positive_weight = float(negative_count) / float(positive_count)
        trainable = cleaned.withColumn(
            "classWeightCol",
            F.when(F.col("label") == 1, F.lit(positive_weight)).otherwise(F.lit(1.0))
        )
    else:
        positive_weight = None
        trainable = cleaned

    train_df, test_df = trainable.randomSplit([args.train_ratio, 1.0 - args.train_ratio], seed=args.seed)

    pipeline = build_pipeline(categorical_cols, numerical_cols, args.mode, args.max_iter)
    model = pipeline.fit(train_df)
    predictions = model.transform(test_df).cache()

    auc = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    ).evaluate(predictions)

    accuracy = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    ).evaluate(predictions)

    f1 = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="f1"
    ).evaluate(predictions)

    weighted_precision = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedPrecision"
    ).evaluate(predictions)

    weighted_recall = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="weightedRecall"
    ).evaluate(predictions)

    confusion_df = predictions.groupBy("label", "prediction").count().orderBy("label", "prediction")
    confusion_rows = [
        {"label": int(r["label"]), "prediction": float(r["prediction"]), "count": int(r["count"])}
        for r in confusion_df.collect()
    ]

    metrics = {
        "mode": args.mode,
        "input": args.input,
        "start_year": args.start_year,
        "max_iter": args.max_iter,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "train_rows": train_df.count(),
        "test_rows": test_df.count(),
        "auc": auc,
        "accuracy": accuracy,
        "f1": f1,
        "exclude_covid": True,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "positive_weight": positive_weight,
        "confusion": confusion_rows,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols
    }

    metrics_json = json.dumps(metrics, indent=2)
    print(metrics_json)

    metrics_output = f"{args.output.rstrip('/')}/{args.mode}_metrics_json"
    spark.createDataFrame([(metrics_json,)], ["metrics_json"]).coalesce(1).write.mode("overwrite").text(metrics_output)

    confusion_output = f"{args.output.rstrip('/')}/{args.mode}_confusion_csv"
    confusion_df.coalesce(1).write.mode("overwrite").option("header", "true").csv(confusion_output)

    model_output = f"{args.output.rstrip('/')}/{args.mode}_pipeline_model"
    model.write().overwrite().save(model_output)

    if args.save_predictions:
        pred_output = f"{args.output.rstrip('/')}/{args.mode}_test_predictions_parquet"
        predictions.select("label", "prediction", "probability").write.mode("overwrite").parquet(pred_output)

    predictions.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
