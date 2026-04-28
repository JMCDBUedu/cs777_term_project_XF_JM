# XGBoost, Jack here! I had not worked with xgboost on a distributed
# compute previously, i have used it for non distributed compute applications
#  and as such had a very difficult time setting this model up. 
# I could not scale it to work with our large dataset, and had to use a small subset, 
#a small number of estimators, and had to scale our year back to 1995 start. ran out of 
#time to include it in our initial report out. included here per request. 

# %pip install xgboost #may need to run if xgboost isnt on cluster

import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
)
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, StringType, DoubleType
)

spark = SparkSession.builder.appName("flight_delay_xgboost").getOrCreate()
spark.sparkContext.setLogLevel("WARN")



GCS_PATH        = "gs://jmcdonough_metcs_bucket/flights_final.csv"
SAMPLE_PATH     = "gs://jmcdonough_metcs_bucket/flight_delay_sample/"
PLOT_PATH       = "/tmp/"
DELAY_THRESHOLD = 15
SAMPLE_FRACTION = 0.05
RANDOM_SEED     = 42
MIN_YEAR        = 1995
TEST_SIZE       = 0.20

XGB_PARAMS = dict(
    n_estimators          = 50, #max amount of estimations used 
    learning_rate         = 0.05, #step size per round 
    max_depth             = 6, #per research, maximum tree depth to prevent model complexity 
    subsample             = 0.8, #fraction of rows sampled per tree
    colsample_bytree      = 0.8, #frac of subset
    scale_pos_weight      = 4.9, #class imbalance correction weight from model 2
    eval_metric           = "auc", #using AUC as our evaluation
    early_stopping_rounds = 30, #stops our run if we hit an auc cieling
    random_state          = 42, 
    n_jobs                = -1,#uses all cores abailable 
    tree_method           = "hist", #histogram training method, which is the lightest model available per research

)

# =============================================================================
# PHASE 1: LOAD & VALIDATE
# =============================================================================

schema = StructType([
    StructField("year",                        IntegerType(), True),
    StructField("month",                       IntegerType(), True),
    StructField("day",                         IntegerType(), True),
    StructField("formatted_date",              StringType(),  True),
    StructField("dayofweek",                   IntegerType(), True),
    StructField("airline",                     StringType(),  True),
    StructField("flight_number",               StringType(),  True),
    StructField("tail_number",                 StringType(),  True),
    StructField("origin_airport_id",           IntegerType(), True),
    StructField("iata_code_reporting_airline", StringType(),  True),
    StructField("dest_airport_id",             IntegerType(), True),
    StructField("departure_time_hhmm",         StringType(),  True),
    StructField("departure_delay",             DoubleType(),  True),
    StructField("departure_delay_mins",        DoubleType(),  True),
    StructField("taxi_out",                    DoubleType(),  True),
    StructField("wheels_off_hhmm",             StringType(),  True),
    StructField("scheduled_elapsed_time",      DoubleType(),  True),
    StructField("elapsed_time",                DoubleType(),  True),
    StructField("air_time",                    DoubleType(),  True),
    StructField("distance",                    DoubleType(),  True),
    StructField("wheels_on_hhmm",              StringType(),  True),
    StructField("taxi_in",                     DoubleType(),  True),
    StructField("scheduled_arrival_hhmm",      StringType(),  True),
    StructField("arrival_time_hhmm",           StringType(),  True),
    StructField("arrival_delay",               DoubleType(),  True),
    StructField("arrival_delay_mins",          DoubleType(),  True),
    StructField("cancellation_code",           StringType(),  True),
    StructField("cancelled",                   IntegerType(), True),
    StructField("diverted",                    IntegerType(), True),
    StructField("airline_delay",               DoubleType(),  True),
    StructField("weather_delay",               DoubleType(),  True),
    StructField("air_system_delay",            DoubleType(),  True),
    StructField("security_delay",              DoubleType(),  True),
    StructField("late_aircraft_delay",         DoubleType(),  True),
    StructField("crs_dep_time_hhmm",           StringType(),  True),
    StructField("crs_dep_time",                StringType(),  True),
    StructField("crsdeptime",                  IntegerType(), True),
]) #this builds and types our csv


df = (
    spark.read
    .option("header", "true")
    .option("nullValue", "")
    .option("nanValue", "NA")
    .schema(schema)
    .csv(GCS_PATH)
) #normalizes any null and na values  for later assessment or drop. 

df.cache() #saves our dataframe in spark cache 
total = df.count()
print(f"    Total rows: {total:,}") #make sure we have an expected amount of data available. 
df = df.withColumn(
    "is_delayed",
    F.when(F.col("departure_delay_mins") > DELAY_THRESHOLD, 1).otherwise(0)
) #this is our binary classifier


#performing some transformations and feature engineering
print(f"\n>>> Filtering to year >= {MIN_YEAR}...")
df = df.filter(F.col("year") >= MIN_YEAR)
total_filtered = df.count()
print(f"    Rows after year filter: {total_filtered:,}  (dropped {total - total_filtered:,})")


df = df.withColumn("dep_hour", (F.col("crsdeptime") / 100).cast("int"))
df = df.withColumn( #buckets to track observed congestion at peak travel times, before 6 there is little delay, at 6-12 we start seeing traffic pick up, between 12 and 5 we have the most flights, then we trend back down 
    "dep_period",
    F.when(F.col("dep_hour") < 6,  0)
     .when(F.col("dep_hour") < 12, 1)
     .when(F.col("dep_hour") < 17, 2)
     .when(F.col("dep_hour") < 21, 3)
     .otherwise(4)
)
#tracks weekends, saturday and sunday, in the event they behave differently 
df = df.withColumn("is_weekend", F.when(F.col("dayofweek").isin(6, 7), 1).otherwise(0))
df = df.withColumn(
    "season",
    F.when(F.col("month").isin(12, 1, 2), 0)
     .when(F.col("month").isin(3, 4, 5),  1)
     .when(F.col("month").isin(6, 7, 8),  2)
     .otherwise(3)
)
df = df.withColumn( #tracks routes, based on where a flight starts and stops, pulled from our initial EDA
    "route",
    F.concat_ws("-",
        F.col("origin_airport_id").cast("string"),
        F.col("dest_airport_id").cast("string")
    )
)

print("    Computing delay rates...") #tracks likely places where a delay was caused, 
airline_delay_rate = df.groupBy("airline").agg(F.mean("is_delayed").alias("airline_delay_rate"))
origin_delay_rate  = df.groupBy("origin_airport_id").agg(F.mean("is_delayed").alias("origin_delay_rate"))
dest_delay_rate    = df.groupBy("dest_airport_id").agg(F.mean("is_delayed").alias("dest_delay_rate"))
route_delay_rate   = df.groupBy("route").agg(F.mean("is_delayed").alias("route_delay_rate"))
hour_delay_rate    = df.groupBy("dep_hour").agg(F.mean("is_delayed").alias("hour_delay_rate"))
month_delay_rate   = df.groupBy("month").agg(F.mean("is_delayed").alias("month_delay_rate"))


df = (
    df
    .join(airline_delay_rate, on="airline",           how="left")
    .join(origin_delay_rate,  on="origin_airport_id", how="left")
    .join(dest_delay_rate,    on="dest_airport_id",   how="left")
    .join(route_delay_rate,   on="route",             how="left")
    .join(hour_delay_rate,    on="dep_hour",          how="left")
    .join(month_delay_rate,   on="month",             how="left")
) #we join the new rate data back into our model to assit in our prediction

route_medians = (
    df.groupBy("route")
    .agg(
        F.percentile_approx("taxi_out", 0.5).alias("taxi_out_median"),
        F.percentile_approx("air_time",  0.5).alias("air_time_median"),
    )
)
df = df.join(route_medians, on="route", how="left")
df = df.withColumn("taxi_out_clean", F.coalesce(F.col("taxi_out"), F.col("taxi_out_median")))
df = df.withColumn("air_time_clean",  F.coalesce(F.col("air_time"),  F.col("air_time_median")))

FEATURE_COLS = [
    "year", "month", "day", "dayofweek",
    "dep_hour", "dep_period", "is_weekend", "season",
    "distance", "scheduled_elapsed_time",
    "taxi_out_clean", "air_time_clean",
    "origin_airport_id", "dest_airport_id",
    "airline_delay_rate", "origin_delay_rate", "dest_delay_rate",
    "route_delay_rate", "hour_delay_rate", "month_delay_rate",
    "is_delayed",
] #features used, including our engineered values

df_features = df.select(FEATURE_COLS)

print(f"\n>>> Stratified sampling at {SAMPLE_FRACTION*100:.0f}%...")
sample = df_features.sampleBy(
    col="is_delayed",
    fractions={0: SAMPLE_FRACTION, 1: SAMPLE_FRACTION},
    seed=RANDOM_SEED
) #samples our dataset so we arent using the full set 

sample_count = sample.count()
print(f"    Sample rows: {sample_count:,}")
print("\n    Sample class balance:")
for row in sample.groupBy("is_delayed").count().orderBy("is_delayed").collect():
    pct = 100 * row["count"] / sample_count
    label = "Delayed    " if row["is_delayed"] == 1 else "Not delayed"
    print(f"        {label}: {row['count']:>10,}  ({pct:.1f}%)")

# write our the file with engineered features and subsetting so we can work with it wthout needing to keep it in our cache
print(f"\n>>> Writing sample to {SAMPLE_PATH} ...")
sample.write.mode("overwrite").option("header", "true").csv(SAMPLE_PATH)
print("    Done. Ready for Phase 3.")

#read back the file from our previous section, originally performed this in a notebook, this would be cell 3
df_pd = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(SAMPLE_PATH)
    .toPandas()
)
print(f"    Rows: {len(df_pd):,}  |  Columns: {df_pd.shape[1]}")


X = df_pd.drop(columns=["is_delayed"]) #our independant variables are any values that arent "is delayed"
y = df_pd["is_delayed"] #Y is the opposite
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
)
print(f"    Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")


#trains xgboost, with our eval set and tracks its auc, stopping early if it hits a cieling
print("\n>>> Training XGBoost...")
model = XGBClassifier(**XGB_PARAMS)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)
print(f"\n    Best iteration: {model.best_iteration}")
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred       = model.predict(X_test)
auc          = roc_auc_score(y_test, y_pred_proba)
ap           = average_precision_score(y_test, y_pred_proba)
report       = classification_report(y_test, y_pred, output_dict=True)


print("EVALUATION RESULTS")
print(f"\n  ROC-AUC:       {auc:.4f}  (LR baseline: {LR_AUC:.3f}  |  delta: +{auc - LR_AUC:.4f})")
print(f"  Avg Precision: {ap:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Not Delayed', 'Delayed'])}")



#this tracks our outputs, writing them to images for us to review later. 

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
    display_labels=["Not Delayed", "Delayed"]).plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"XGBoost Confusion Matrix\nROC-AUC: {auc:.4f}", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOT_PATH}confusion_matrix.png", dpi=150); plt.close()


importance_df = pd.DataFrame({
    "feature": feature_names, "importance": model.feature_importances_
}).sort_values("importance", ascending=True)
fig, ax = plt.subplots(figsize=(8, 7))
bars = ax.barh(importance_df["feature"], importance_df["importance"],
               color="steelblue", edgecolor="white")
ax.set_xlabel("Feature Importance (gain)", fontsize=11)
ax.set_title("XGBoost Feature Importances\nFlight Departure Delay Prediction", fontsize=13)
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
plt.tight_layout()
plt.savefig(f"{PLOT_PATH}feature_importance.png", dpi=150); plt.close()

# --- Precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recall, precision, color="steelblue", lw=2, label=f"XGBoost (AP={ap:.4f})")
ax.axhline(y=y_test.mean(), color="gray", linestyle="--",
           label=f"Random baseline ({y_test.mean():.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve\nFlight Departure Delay Prediction", fontsize=13)
ax.legend(); plt.tight_layout()
plt.savefig(f"{PLOT_PATH}precision_recall_curve.png", dpi=150); plt.close()

# --- Training curve ---
results = model.evals_result()
epochs  = len(results["validation_0"]["auc"])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(epochs), results["validation_0"]["auc"],
        color="steelblue", lw=1.5, label="Validation AUC")
ax.axvline(x=model.best_iteration, color="red", linestyle="--",
           label=f"Best round ({model.best_iteration})")
ax.set_xlabel("Boosting Round"); ax.set_ylabel("AUC")
ax.set_title("XGBoost Training Curve", fontsize=13)
ax.legend(); plt.tight_layout()
plt.savefig(f"{PLOT_PATH}training_curve.png", dpi=150); plt.close()

# --- Upload plots to GCS ---
for fname in ["confusion_matrix.png", "feature_importance.png",
              "precision_recall_curve.png", "training_curve.png"]:
    subprocess.run(["gsutil", "cp", f"/tmp/{fname}",
                    f"gs://jmcdonough_metcs_bucket/plots/{fname}"])
    print(f"Uploaded: gs://jmcdonough_metcs_bucket/plots/{fname}")

print("\n>>> Done.")