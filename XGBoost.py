#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================================
# PHASE 1: Ingest & Validate (pre-cleaned CSV)
# Flight Delay XGBoost Analysis
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, StringType, DoubleType
)

spark = SparkSession.builder.appName("flight_delay_xgb_phase1").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# =============================================================================
# CONFIGURATION
# =============================================================================

GCS_PATH          = "gs://jmcdonough_metcs_bucket/flights_final.csv"
CLEAN_OUTPUT_PATH = "gs://jmcdonough_metcs_bucket/flight_delay_clean/"
DELAY_THRESHOLD   = 15

# =============================================================================
# SCHEMA
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
])

# =============================================================================
# LOAD
# =============================================================================

print(">>> Loading CSV from GCS...")

df = (
    spark.read
    .option("header", "true")
    .option("nullValue", "")
    .option("nanValue", "NA")
    .schema(schema)
    .csv(GCS_PATH)
)

df.cache()
total = df.count()
print(f"    Total rows: {total:,}")

# =============================================================================
# ADD BINARY TARGET LABEL
# =============================================================================

df = df.withColumn(
    "is_delayed",
    F.when(F.col("departure_delay_mins") > DELAY_THRESHOLD, 1).otherwise(0)
)

# =============================================================================
# VALIDATION REPORT
# =============================================================================

print("\n" + "=" * 60)
print("VALIDATION REPORT")
print("=" * 60)

# Class balance
print("\n[Class Balance]")
label_counts = (
    df.groupBy("is_delayed")
    .count()
    .orderBy("is_delayed")
    .collect()
)
for row in label_counts:
    pct = 100 * row["count"] / total
    label = "Delayed    " if row["is_delayed"] == 1 else "Not delayed"
    print(f"    {label}: {row['count']:>12,}  ({pct:.1f}%)")

# Null rates
print("\n[Null Rates — Key Columns]")
key_cols = [
    "departure_delay_mins", "crsdeptime", "dayofweek", "month",
    "airline", "origin_airport_id", "dest_airport_id",
    "distance", "taxi_out", "air_time",
    "airline_delay", "weather_delay", "air_system_delay",
    "security_delay", "late_aircraft_delay",
]
delay_cause_cols = {
    "airline_delay", "weather_delay", "air_system_delay",
    "security_delay", "late_aircraft_delay"
}
null_exprs = [
    F.round(100 * F.sum(F.col(c).isNull().cast("int")) / total, 2).alias(c)
    for c in key_cols
]
null_rates = df.select(null_exprs).collect()[0].asDict()
for col_name, rate in null_rates.items():
    flag = "  *** HIGH ***" if (rate > 10 and col_name not in delay_cause_cols) else ""
    print(f"    {col_name:<35}: {rate:>6.2f}%{flag}")

# Year range
year_range = df.select(F.min("year"), F.max("year")).collect()[0]
print(f"\n[Year Range]  {year_range[0]} – {year_range[1]}")

# Cardinality
airline_count = df.select("airline").distinct().count()
origin_count  = df.select("origin_airport_id").distinct().count()
print(f"[Unique Airlines]         {airline_count}")
print(f"[Unique Origin Airports]  {origin_count}")

print("\n[Note] Null delay cause columns (airline_delay etc.) are expected —")
print("       BTS only populates these for delayed flights.")
print("=" * 60)

# =============================================================================
# WRITE PARQUET
# =============================================================================

print(f"\n>>> Writing to {CLEAN_OUTPUT_PATH} ...")
df.write.mode("overwrite").parquet(CLEAN_OUTPUT_PATH)
print("    Done. Ready for Phase 2.")


# In[7]:


# =============================================================================
# PHASE 2: Feature Engineering & Sampling
# Flight Delay XGBoost Analysis
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.appName("flight_delay_xgb_phase2").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# =============================================================================
# CONFIGURATION
# =============================================================================

CLEAN_PATH        = "gs://jmcdonough_metcs_bucket/flight_delay_clean/"
FEATURE_PATH      = "gs://jmcdonough_metcs_bucket/flight_delay_features/"
SAMPLE_PATH       = "gs://jmcdonough_metcs_bucket/flight_delay_sample/"

SAMPLE_FRACTION   = 0.05    # ~11.5M rows from 229M — adjust if driver OOMs
RANDOM_SEED       = 42
MIN_YEAR          = 1995    # pre-1995 BTS data has sparse fields

# =============================================================================
# 1. LOAD PARQUET
# =============================================================================

print(">>> Loading clean Parquet...")
df = spark.read.parquet(CLEAN_PATH)

total_raw = df.count()
print(f"    Rows loaded: {total_raw:,}")

# =============================================================================
# 2. FILTER TO 1995+
# =============================================================================

print(f"\n>>> Filtering to year >= {MIN_YEAR}...")
df = df.filter(F.col("year") >= MIN_YEAR)
total_filtered = df.count()
print(f"    Rows after year filter: {total_filtered:,}  (dropped {total_raw - total_filtered:,})")

# --- Investigate taxi_out nulls after year filter ---
print("\n>>> Null rates after year filter:")
for col_name in ("taxi_out", "air_time"):
    null_count = df.filter(F.col(col_name).isNull()).count()
    pct = 100 * null_count / total_filtered
    print(f"    {col_name:<12}: {pct:.2f}%  ({null_count:,} rows)")

# =============================================================================
# 3. FEATURE ENGINEERING
# All aggregates computed on the FULL filtered dataset before sampling.
# =============================================================================

print("\n>>> Engineering features...") 

# -----------------------------------------------------------------------------
# 3a. Temporal features
# -----------------------------------------------------------------------------

df = df.withColumn("dep_hour", (F.col("crsdeptime") / 100).cast("int"))

# Time-of-day bucket: early_morning/morning/afternoon/evening/night
df = df.withColumn(
    "dep_period",
    F.when(F.col("dep_hour") < 6,  0)   # early morning
     .when(F.col("dep_hour") < 12, 1)   # morning
     .when(F.col("dep_hour") < 17, 2)   # afternoon
     .when(F.col("dep_hour") < 21, 3)   # evening
     .otherwise(4)                       # night
)

# Is weekend
df = df.withColumn(
    "is_weekend",
    F.when(F.col("dayofweek").isin(6, 7), 1).otherwise(0)
)

# Season
df = df.withColumn(
    "season",
    F.when(F.col("month").isin(12, 1, 2), 0)   # winter
     .when(F.col("month").isin(3, 4, 5),   1)   # spring
     .when(F.col("month").isin(6, 7, 8),   2)   # summer
     .otherwise(3)                               # fall
)

# -----------------------------------------------------------------------------
# 3b. Route key
# -----------------------------------------------------------------------------

df = df.withColumn(
    "route",
    F.concat_ws("-",
        F.col("origin_airport_id").cast("string"),
        F.col("dest_airport_id").cast("string")
    )
)

# -----------------------------------------------------------------------------
# 3c. Historical delay rates (computed on full dataset)
# These capture base delay propensity per entity — a strong signal for XGBoost.
# Using mean of is_delayed rather than raw counts for portability.
# -----------------------------------------------------------------------------

print("    Computing airline delay rate...")
airline_delay_rate = (
    df.groupBy("airline")
    .agg(F.mean("is_delayed").alias("airline_delay_rate"))
)

print("    Computing origin airport delay rate...")
origin_delay_rate = (
    df.groupBy("origin_airport_id")
    .agg(F.mean("is_delayed").alias("origin_delay_rate"))
)

print("    Computing destination airport delay rate...")
dest_delay_rate = (
    df.groupBy("dest_airport_id")
    .agg(F.mean("is_delayed").alias("dest_delay_rate"))
)

print("    Computing route delay rate...")
route_delay_rate = (
    df.groupBy("route")
    .agg(F.mean("is_delayed").alias("route_delay_rate"))
)

print("    Computing dep_hour delay rate...")
hour_delay_rate = (
    df.groupBy("dep_hour")
    .agg(F.mean("is_delayed").alias("hour_delay_rate"))
)

print("    Computing month delay rate...")
month_delay_rate = (
    df.groupBy("month")
    .agg(F.mean("is_delayed").alias("month_delay_rate"))
)

# Join all rates back onto main dataframe
print("    Joining rates back onto dataset...")
df = (
    df
    .join(airline_delay_rate, on="airline",             how="left")
    .join(origin_delay_rate,  on="origin_airport_id",   how="left")
    .join(dest_delay_rate,    on="dest_airport_id",      how="left")
    .join(route_delay_rate,   on="route",                how="left")
    .join(hour_delay_rate,    on="dep_hour",             how="left")
    .join(month_delay_rate,   on="month",                how="left")
)

# -----------------------------------------------------------------------------
# 3d. Impute taxi_out and air_time nulls with route median
# These are structural nulls (older records) — median imputation per route
# is more accurate than global median.
# -----------------------------------------------------------------------------

print("    Imputing taxi_out / air_time nulls with route median...")

route_medians = (
    df.groupBy("route")
    .agg(
        F.percentile_approx("taxi_out", 0.5).alias("taxi_out_median"),
        F.percentile_approx("air_time",  0.5).alias("air_time_median"),
    )
)

df = df.join(route_medians, on="route", how="left")

df = df.withColumn(
    "taxi_out_clean",
    F.coalesce(F.col("taxi_out"), F.col("taxi_out_median"))
)
df = df.withColumn(
    "air_time_clean",
    F.coalesce(F.col("air_time"), F.col("air_time_median"))
)

# =============================================================================
# 4. SELECT FINAL FEATURE SET
# Explicitly exclude leakage columns (delay cause breakdown, actual delay mins).
# Keep only columns XGBoost will train on + the label.
# =============================================================================

FEATURE_COLS = [
    # Temporal
    "year", "month", "day", "dayofweek",
    "dep_hour", "dep_period", "is_weekend", "season",
    # Flight metadata
    "distance", "scheduled_elapsed_time",
    # Cleaned nulls
    "taxi_out_clean", "air_time_clean",
    # Categorical (as integer IDs — XGBoost handles these fine)
    "origin_airport_id", "dest_airport_id",
    # Historical delay rates
    "airline_delay_rate", "origin_delay_rate", "dest_delay_rate",
    "route_delay_rate", "hour_delay_rate", "month_delay_rate",
    # Label
    "is_delayed",
]

# Columns deliberately excluded and why:
# departure_delay_mins  — IS the label source, direct leakage
# arrival_delay_mins    — leakage (outcome of the flight)
# airline_delay etc.    — leakage (only populated when delayed)
# taxi_out / air_time   — replaced by _clean versions
# hhmm string cols      — redundant with dep_hour / crsdeptime
# tail_number           — too high cardinality, not predictive at scale
# flight_number         — too high cardinality
# cancellation_code     — pre-cleaned dataset, not relevant
# airline (string)      — replaced by airline_delay_rate

df_features = df.select(FEATURE_COLS)

# =============================================================================
# 5. VALIDATION CHECK ON FEATURE TABLE
# =============================================================================

print("\n>>> Feature table null check...")
total_feat = df_features.count()
null_exprs = [
    F.round(100 * F.sum(F.col(c).isNull().cast("int")) / total_feat, 2).alias(c)
    for c in FEATURE_COLS
]
null_rates = df_features.select(null_exprs).collect()[0].asDict()
any_high = False
for col_name, rate in null_rates.items():
    if rate > 0:
        flag = "  *** " if rate > 5 else ""
        print(f"    {col_name:<30}: {rate:.2f}%{flag}")
        if rate > 5:
            any_high = True
if not any_high:
    print("    All columns within acceptable null range.")

print(f"\n    Feature table rows: {total_feat:,}")
print(f"    Feature columns:    {len(FEATURE_COLS) - 1}  (+ label)")

# =============================================================================
# 6. WRITE FULL FEATURE TABLE
# =============================================================================

print(f"\n>>> Writing feature table to {FEATURE_PATH} ...")
df_features.write.mode("overwrite").parquet(FEATURE_PATH)
print("    Done.")

# =============================================================================
# 7. STRATIFIED SAMPLE
# Sample equal fractions from each class to preserve 84/16 balance.
# ~11.5M rows at 5% — reduce SAMPLE_FRACTION if driver OOMs in Phase 3.
# =============================================================================

print(f"\n>>> Stratified sampling at {SAMPLE_FRACTION*100:.0f}%...")

sample = df_features.sampleBy(
    col="is_delayed",
    fractions={0: SAMPLE_FRACTION, 1: SAMPLE_FRACTION},
    seed=RANDOM_SEED
)

sample_count = sample.count()
print(f"    Sample rows: {sample_count:,}")

# Class balance check on sample
print("\n    Sample class balance:")
sample_counts = sample.groupBy("is_delayed").count().orderBy("is_delayed").collect()
for row in sample_counts:
    pct = 100 * row["count"] / sample_count
    label = "Delayed    " if row["is_delayed"] == 1 else "Not delayed"
    print(f"        {label}: {row['count']:>10,}  ({pct:.1f}%)")

# =============================================================================
# 8. WRITE SAMPLE
# =============================================================================

print(f"\n>>> Writing sample to {SAMPLE_PATH} ...")
sample.write.mode("overwrite").parquet(SAMPLE_PATH)
print("    Done. Ready for Phase 3 (XGBoost training).")


# In[1]:


# =============================================================================
# PHASE 3: XGBoost Training & Evaluation
# Flight Delay XGBoost Analysis
# =============================================================================
get_ipython().run_line_magic('pip', 'install xgboost')
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Dataproc
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score
)

spark = SparkSession.builder.appName("flight_delay_xgb_phase3").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLE_PATH   = "gs://jmcdonough_metcs_bucket/flight_delay_sample/"
PLOT_PATH     = "/home/jupyter/"   # saves plots here; download from JupyterLab
RANDOM_SEED   = 42
TEST_SIZE     = 0.20

# Logistic regression baseline from prior work (for comparison table)
LR_AUC     = 0.664
LR_RECALL  = 0.62

# XGBoost params
# scale_pos_weight = not_delayed / delayed = 83 / 17 ≈ 4.9
XGB_PARAMS = dict(
    n_estimators      = 5000,
    learning_rate     = 0.05,
    max_depth         = 6,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = 4.9,
    eval_metric       = "auc",
    early_stopping_rounds = 30,
    random_state      = RANDOM_SEED,
    n_jobs            = -1,
    tree_method       = "hist",   # fastest CPU method
)

# =============================================================================
# 1. LOAD SAMPLE → PANDAS
# =============================================================================

print(">>> Loading sample from GCS...")
sample_spark = spark.read.parquet(SAMPLE_PATH)
df = sample_spark.toPandas()
print(f"    Rows: {len(df):,}  |  Columns: {df.shape[1]}")

# =============================================================================
# 2. PREPARE FEATURES & LABEL
# =============================================================================

LABEL = "is_delayed"
DROP_COLS = [LABEL]

X = df.drop(columns=DROP_COLS)
y = df[LABEL]

feature_names = X.columns.tolist()
print(f"    Features: {feature_names}")

# =============================================================================
# 3. TRAIN / TEST SPLIT
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = TEST_SIZE,
    stratify     = y,
    random_state = RANDOM_SEED,
)

print(f"\n    Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
print(f"    Train delayed rate: {y_train.mean():.3f}")
print(f"    Test  delayed rate: {y_test.mean():.3f}")

# =============================================================================
# 4. TRAIN XGBOOST
# =============================================================================

print("\n>>> Training XGBoost...")

model = XGBClassifier(**XGB_PARAMS)

model.fit(
    X_train, y_train,
    eval_set        = [(X_test, y_test)],
    verbose         = 50,   # print every 50 rounds
)

best_round = model.best_iteration
print(f"\n    Best iteration: {best_round}")

# =============================================================================
# 5. EVALUATE
# =============================================================================

print("\n>>> Evaluating...")

y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred       = model.predict(X_test)

auc = roc_auc_score(y_test, y_pred_proba)
ap  = average_precision_score(y_test, y_pred_proba)

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print(f"\n  ROC-AUC:           {auc:.4f}  (LR baseline: {LR_AUC:.3f}  |  delta: +{auc - LR_AUC:.4f})")
print(f"  Avg Precision:     {ap:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Not Delayed', 'Delayed'])}")

# Model comparison table
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)
print(f"  {'Model':<25} {'AUC':>8} {'Recall (delayed)':>18}")
print(f"  {'-'*53}")
report = classification_report(y_test, y_pred, output_dict=True)
xgb_recall = report['1']['recall']
print(f"  {'Logistic Regression':<25} {LR_AUC:>8.3f} {LR_RECALL:>18.3f}")
print(f"  {'XGBoost':<25} {auc:>8.4f} {xgb_recall:>18.4f}")
print("=" * 60)

# =============================================================================
# 6. CONFUSION MATRIX PLOT
# =============================================================================

print("\n>>> Saving confusion matrix plot...")

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Delayed", "Delayed"]
)
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"XGBoost Confusion Matrix\nROC-AUC: {auc:.4f}", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOT_PATH}confusion_matrix.png", dpi=150)
plt.close()
print(f"    Saved: {PLOT_PATH}confusion_matrix.png")

# =============================================================================
# 7. FEATURE IMPORTANCE PLOT
# =============================================================================

print(">>> Saving feature importance plot...")

importance_df = pd.DataFrame({
    "feature":    feature_names,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=True)

fig, ax = plt.subplots(figsize=(8, 7))
bars = ax.barh(
    importance_df["feature"],
    importance_df["importance"],
    color="steelblue", edgecolor="white"
)
ax.set_xlabel("Feature Importance (gain)", fontsize=11)
ax.set_title("XGBoost Feature Importances\nFlight Departure Delay Prediction", fontsize=13)
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
plt.tight_layout()
plt.savefig(f"{PLOT_PATH}feature_importance.png", dpi=150)
plt.close()
print(f"    Saved: {PLOT_PATH}feature_importance.png")

# =============================================================================
# 8. PRECISION-RECALL CURVE
# More informative than ROC for imbalanced classes.
# =============================================================================

print(">>> Saving precision-recall curve...")

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(recall, precision, color="steelblue", lw=2,
        label=f"XGBoost (AP={ap:.4f})")
ax.axhline(y=y_test.mean(), color="gray", linestyle="--",
           label=f"Random baseline ({y_test.mean():.3f})")
ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("Precision", fontsize=11)
ax.set_title("Precision-Recall Curve\nFlight Departure Delay Prediction", fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_PATH}precision_recall_curve.png", dpi=150)
plt.close()
print(f"    Saved: {PLOT_PATH}precision_recall_curve.png")

# =============================================================================
# 9. TRAINING LOSS CURVE
# =============================================================================

print(">>> Saving training loss curve...")

results = model.evals_result()
epochs  = len(results["validation_0"]["auc"])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(epochs), results["validation_0"]["auc"],
        color="steelblue", lw=1.5, label="Validation AUC")
ax.axvline(x=best_round, color="red", linestyle="--",
           label=f"Best round ({best_round})")
ax.set_xlabel("Boosting Round", fontsize=11)
ax.set_ylabel("AUC", fontsize=11)
ax.set_title("XGBoost Training Curve", fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_PATH}training_curve.png", dpi=150)
plt.close()
print(f"    Saved: {PLOT_PATH}training_curve.png")

print("\n>>> Phase 3 complete.")
print(f"    Plots saved to {PLOT_PATH}")
print(f"    Download via JupyterLab file browser.")


# In[2]:


import os
PLOT_PATH = "/root/"

# --- Confusion matrix ---
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
    display_labels=["Not Delayed", "Delayed"]).plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title(f"XGBoost Confusion Matrix\nROC-AUC: {auc:.4f}", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOT_PATH}confusion_matrix.png", dpi=150); plt.close()

# --- Feature importance ---
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

# --- Precision-recall curve ---
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
epochs = len(results["validation_0"]["auc"])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(epochs), results["validation_0"]["auc"],
        color="steelblue", lw=1.5, label="Validation AUC")
ax.axvline(x=model.best_iteration, color="red", linestyle="--",
           label=f"Best round ({model.best_iteration})")
ax.set_xlabel("Boosting Round"); ax.set_ylabel("AUC")
ax.set_title("XGBoost Training Curve", fontsize=13)
ax.legend(); plt.tight_layout()
plt.savefig(f"{PLOT_PATH}training_curve.png", dpi=150); plt.close()

print("All plots saved to /root/")
print([f for f in os.listdir("/root/") if f.endswith(".png")])


# In[3]:


import os
PLOT_PATH = "/tmp/"  # write locally first, then upload to GCS

# ... run all the plt.savefig() calls as-is, then upload:

import subprocess
for fname in ["confusion_matrix.png", "feature_importance.png", 
              "precision_recall_curve.png", "training_curve.png"]:
    subprocess.run([
        "gsutil", "cp", f"/tmp/{fname}",
        f"gs://jmcdonough_metcs_bucket/plots_5000xgboost/{fname}"
    ])
    print(f"Uploaded: gs://jmcdonough_metcs_bucket/plots/{fname}")


# In[ ]:




