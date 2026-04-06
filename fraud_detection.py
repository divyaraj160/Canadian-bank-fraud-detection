"""
Canadian Bank Transaction Fraud Detection
==========================================
Author  : Divyaraj Jadeja
GitHub  : github.com/divyaraj160
Stack   : Python · SQL · Scikit-learn · Pandas · Matplotlib · Seaborn

Description:
    End-to-end fraud detection pipeline on 12,000 synthetic Canadian bank
    transactions. Includes SQL-style exploratory analysis, feature engineering,
    Random Forest classification, and performance evaluation.

Results:
    ROC-AUC    : 0.9984
    PR-AUC     : 0.9849
    F1 (Fraud) : 0.9484
    Recall     : 93.5%
    CV ROC-AUC : 0.9992 ± 0.0005
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════
df = pd.read_csv("transactions.csv")
print(f"Loaded {len(df):,} transactions")
print(f"Fraud rate: {df['is_fraud'].mean()*100:.1f}%  ({df['is_fraud'].sum():,} fraud cases)")
print(f"Columns: {list(df.columns)}\n")


# ═══════════════════════════════════════════════════
# 2. SQL-STYLE EXPLORATORY ANALYSIS
# ═══════════════════════════════════════════════════
print("=" * 55)
print("SQL ANALYSIS LAYER")
print("=" * 55)

# Q1: Fraud rate by merchant category
print("\n-- Fraud rate by merchant category --")
fraud_by_cat = (
    df.groupby("merchant_category")
    .agg(total=("is_fraud", "count"), fraud=("is_fraud", "sum"))
    .assign(fraud_rate=lambda x: (x.fraud / x.total * 100).round(2))
    .sort_values("fraud_rate", ascending=False)
)
print(fraud_by_cat.to_string())

# Q2: Fraud rate by country (foreign transaction risk)
print("\n-- Fraud rate by transaction country --")
fraud_by_country = (
    df.groupby("country")
    .agg(total=("is_fraud", "count"), fraud=("is_fraud", "sum"))
    .assign(fraud_rate=lambda x: (x.fraud / x.total * 100).round(2))
    .sort_values("fraud_rate", ascending=False)
)
print(fraud_by_country.to_string())

# Q3: High-velocity transactions (8+ txns in 24 hours)
high_vel = df[df["txn_velocity_24h"] >= 8]
print(f"\n-- High velocity transactions (8+ per 24h) --")
print(f"Count:      {len(high_vel):,}")
print(f"Fraud rate: {high_vel['is_fraud'].mean()*100:.1f}%")

# Q4: Late night + large amount combination
late_large = df[
    (df["hour_of_day"].isin([0, 1, 2, 3, 22, 23])) &
    (df["transaction_amount"] > 500)
]
print(f"\n-- Late night (10pm–4am) + Amount > $500 --")
print(f"Count:      {len(late_large):,}")
print(f"Fraud rate: {late_large['is_fraud'].mean()*100:.1f}%")

# Q5: Foreign card-not-present transactions
foreign_cnp = df[(df["country"] != "Canada") & (df["card_present"] == 0)]
print(f"\n-- Foreign + Card Not Present --")
print(f"Count:      {len(foreign_cnp):,}")
print(f"Fraud rate: {foreign_cnp['is_fraud'].mean()*100:.1f}%")

# Q6: Avg fraud transaction amount vs legit
print(f"\n-- Average transaction amount --")
print(f"Legitimate: ${df[df.is_fraud==0]['transaction_amount'].mean():,.2f}")
print(f"Fraud:      ${df[df.is_fraud==1]['transaction_amount'].mean():,.2f}")


# ═══════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════
print("\n" + "=" * 55)
print("FEATURE ENGINEERING")
print("=" * 55)

# Encode categoricals
for col in ["country", "merchant_category", "bank", "province"]:
    df[col + "_enc"] = LabelEncoder().fit_transform(df[col])

# Derived risk features
df["is_late_night"] = df["hour_of_day"].isin([0, 1, 2, 3, 22, 23]).astype(int)
df["is_foreign"]    = (df["country"] != "Canada").astype(int)
df["high_amount"]   = (df["transaction_amount"] > 500).astype(int)
df["high_velocity"] = (df["txn_velocity_24h"] >= 5).astype(int)
df["risk_score"]    = (
    df["is_late_night"] + df["is_foreign"] + df["high_amount"] +
    df["high_velocity"] + df["new_merchant"]
)

FEATURES = [
    "transaction_amount", "hour_of_day", "days_since_last_txn",
    "txn_velocity_24h", "distance_from_home_km", "new_merchant",
    "card_present", "country_enc", "merchant_category_enc",
    "bank_enc", "province_enc", "is_late_night", "is_foreign",
    "high_amount", "high_velocity", "risk_score",
]

print(f"Total features: {len(FEATURES)}")
print(f"Engineered features: is_late_night, is_foreign, high_amount, high_velocity, risk_score")


# ═══════════════════════════════════════════════════
# 4. TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════
X = df[FEATURES]
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Test fraud cases: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")


# ═══════════════════════════════════════════════════
# 5. MODEL TRAINING — RANDOM FOREST
# ═══════════════════════════════════════════════════
print("\n" + "=" * 55)
print("MODEL: Random Forest Classifier")
print("=" * 55)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",   # handles class imbalance
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
print("Model trained.")

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]


# ═══════════════════════════════════════════════════
# 6. EVALUATION
# ═══════════════════════════════════════════════════
print("\n" + "=" * 55)
print("EVALUATION RESULTS")
print("=" * 55)

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc  = average_precision_score(y_test, y_prob)
cv_sc   = cross_val_score(rf, X, y, cv=StratifiedKFold(5), scoring="roc_auc")
report  = classification_report(y_test, y_pred,
                                 target_names=["Legitimate", "Fraud"])

print(f"\nROC-AUC Score    : {roc_auc:.4f}")
print(f"PR-AUC Score     : {pr_auc:.4f}")
print(f"CV ROC-AUC       : {cv_sc.mean():.4f} ± {cv_sc.std():.4f}")
print(f"\nClassification Report:\n{report}")

# Feature importance
feat_imp = (
    pd.DataFrame({"feature": FEATURES, "importance": rf.feature_importances_})
    .sort_values("importance", ascending=False)
)
print("Top 8 Features:")
print(feat_imp.head(8).to_string(index=False))


# ═══════════════════════════════════════════════════
# 7. VISUALIZE (save to PNG)
# ═══════════════════════════════════════════════════
fraud_by_hr = df.groupby("hour_of_day")["is_fraud"].mean() * 100
cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor("#0e110f")
for ax in axes.flat:
    ax.set_facecolor("#141814")
    ax.tick_params(colors="#8a9489", labelsize=9)
    ax.xaxis.label.set_color("#8a9489")
    ax.yaxis.label.set_color("#8a9489")
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2f2a")

C = {"fraud": "#E24B4A", "legit": "#1D9E75", "blue": "#4d9de0", "orange": "#f0a830"}

# Plot 1 — Amount distribution
ax = axes[0, 0]
ax.hist(df[df.is_fraud==0]["transaction_amount"].clip(0,1500), bins=50, alpha=0.7,
        color=C["legit"], label="Legitimate", density=True)
ax.hist(df[df.is_fraud==1]["transaction_amount"].clip(0,1500), bins=50, alpha=0.7,
        color=C["fraud"], label="Fraud", density=True)
ax.set_title("Transaction Amount Distribution", color="#e8ebe8", fontsize=11,
             fontweight="bold", pad=10)
ax.set_xlabel("Amount (CAD $)"); ax.set_ylabel("Density")
ax.legend(facecolor="#1b1f1b", labelcolor="#e8ebe8", fontsize=9)

# Plot 2 — Fraud by hour
ax = axes[0, 1]
bc = [C["fraud"] if h in [0,1,2,3,22,23] else C["blue"] for h in fraud_by_hr.index]
ax.bar(fraud_by_hr.index, fraud_by_hr.values, color=bc, width=0.8, alpha=0.9)
ax.axhline(fraud_by_hr.mean(), color=C["orange"], linestyle="--", linewidth=1.5,
           label=f"Avg {fraud_by_hr.mean():.1f}%")
ax.set_title("Fraud Rate by Hour of Day", color="#e8ebe8", fontsize=11,
             fontweight="bold", pad=10)
ax.set_xlabel("Hour"); ax.set_ylabel("Fraud Rate (%)")
ax.legend(facecolor="#1b1f1b", labelcolor="#e8ebe8", fontsize=9)

# Plot 3 — Feature importance
ax = axes[0, 2]
t10 = feat_imp.head(10)
bc3 = [C["fraud"] if i < 3 else C["blue"] for i in range(len(t10))]
ax.barh(t10["feature"][::-1], t10["importance"][::-1], color=bc3[::-1], alpha=0.9)
ax.set_title("Feature Importance (Top 10)", color="#e8ebe8", fontsize=11,
             fontweight="bold", pad=10)
ax.set_xlabel("Importance Score")

# Plot 4 — Confusion matrix
ax = axes[1, 0]
sns.heatmap(cm, annot=True, fmt="d", ax=ax,
            cmap=sns.light_palette("#1D9E75", as_cmap=True),
            xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"],
            linewidths=0.5, linecolor="#0e110f", cbar=False,
            annot_kws={"size": 14, "weight": "bold"})
ax.set_title("Confusion Matrix", color="#e8ebe8", fontsize=11, fontweight="bold", pad=10)
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")

# Plot 5 — Fraud by merchant
ax = axes[1, 1]
mc = fraud_by_cat.head(8)
bc5 = [C["fraud"] if r>10 else C["orange"] if r>5 else C["blue"] for r in mc["fraud_rate"]]
ax.barh(mc.index[::-1], mc["fraud_rate"][::-1], color=bc5[::-1], alpha=0.9)
ax.set_title("Fraud Rate by Merchant Category", color="#e8ebe8", fontsize=11,
             fontweight="bold", pad=10)
ax.set_xlabel("Fraud Rate (%)")

# Plot 6 — Scorecard
ax = axes[1, 2]; ax.axis("off")
rep_d = classification_report(y_test, y_pred, output_dict=True)
lines = [
    ("MODEL SCORECARD", None, "#22c987"),
    ("ROC-AUC Score",          f"{roc_auc:.4f}",                   "#e8ebe8"),
    ("PR-AUC Score",           f"{pr_auc:.4f}",                    "#e8ebe8"),
    ("CV ROC-AUC (5-fold)",    f"{cv_sc.mean():.4f} ± {cv_sc.std():.4f}", "#e8ebe8"),
    ("FRAUD CLASS", None, "#4d9de0"),
    ("Precision",   f"{rep_d['1']['precision']:.1%}", "#e8ebe8"),
    ("Recall",      f"{rep_d['1']['recall']:.1%}",    "#e8ebe8"),
    ("F1-Score",    f"{rep_d['1']['f1-score']:.1%}",  "#e8ebe8"),
    ("DATASET", None, "#f0a830"),
    ("Total transactions", f"{len(df):,}",                          "#e8ebe8"),
    ("Fraud cases",        f"{df.is_fraud.sum():,} ({df.is_fraud.mean()*100:.1f}%)", "#e8ebe8"),
    ("Algorithm",          "Random Forest · 200 trees",             "#8a9489"),
]
yp = 0.97
for label, val, col in lines:
    if val is None:
        ax.text(0.05, yp, label, transform=ax.transAxes, fontsize=10,
                color=col, fontweight="bold")
        yp -= 0.07
    else:
        ax.text(0.05, yp, label, transform=ax.transAxes, fontsize=9, color="#8a9489")
        ax.text(0.97, yp, val, transform=ax.transAxes, fontsize=9, color=col, ha="right")
        yp -= 0.065
ax.set_title("Score Summary", color="#e8ebe8", fontsize=11, fontweight="bold", pad=10)

plt.suptitle("Canadian Bank Fraud Detection — ML Analysis Dashboard",
             color="#e8ebe8", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout(pad=2)
plt.savefig("analysis_dashboard.png", dpi=150, bbox_inches="tight", facecolor="#0e110f")
print("\nDashboard saved to analysis_dashboard.png")
print("\nDone. All results saved.")
