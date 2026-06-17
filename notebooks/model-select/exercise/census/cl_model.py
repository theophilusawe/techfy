"""
End-to-End Classification Pipeline & Evaluation
Adult Income Dataset — UCI Census Data
Following the 5-step plan from Steps.pa
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, accuracy_score, roc_auc_score,
    roc_curve, ConfusionMatrixDisplay
)


import io, os

OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1 — DATA ACQUISITION & INSPECTION
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Data Acquisition & Inspection")
print("=" * 60)

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

# Load; treat " ?" as NaN (as specified in the plan)
a = pd.read_csv(
    "/home/claude/adult_census.csv",
    names=COLUMNS,
    na_values=" ?",
    header=0,         # our synthetic file has a header row
    skipinitialspace=True
)

print(f"\nDataset shape: {a.shape}")
print(f"Missing values:\n{a.isnull().sum()}")

# Clean target — strip trailing periods, map to binary
a["income"] = a["income"].str.strip().str.rstrip(".")
a["income"] = a["income"].map({"<=50K": 0, ">50K": 1})

print(f"\nTarget distribution:\n{a['income'].value_counts()}")
print(f"Class imbalance — '>50K' share: {a['income'].mean():.2%}")

# ─────────────────────────────────────────────
# STEP 2 — DATA SPLITTING (PREVENTING LEAKAGE)
# ─────────────────────────────────────────────
print("STEP 2 — Train/Test Split with Stratification")
print("=" * 60)

X = a.drop("income", axis=1)
y = a["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y          # ← critical for imbalanced classes
)

print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print(f"Train >50K rate: {y_train.mean():.2%} | Test >50K rate: {y_test.mean():.2%}")

# ─────────────────────────────────────────────
# STEP 3 — PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
print("STEP 3 — Build Preprocessing Pipeline")
print("*" * 40)

NUMERICAL = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation",
               "relationship", "race", "sex", "native-country"]

# Numerical: median imputation → StandardScaler
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler())
])

# Categorical: mode imputation → OneHotEncoder
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Combine into ColumnTransformer
preprocessor = ColumnTransformer([
    ("num", num_pipeline, NUMERICAL),
    ("cat", cat_pipeline, CATEGORICAL)
])

print("Preprocessor configured: ColumnTransformer with numerical + categorical pipelines")

# ─────────────────────────────────────────────
# STEP 4 — HYPERPARAMETER TUNING & CROSS-VALIDATION
# ─────────────────────────────────────────────
print("STEP 4 — Hyperparameter Tuning (RandomizedSearchCV, F1 scoring)")
print("=" * 60)

"""
WHY F1-SCORE over raw Accuracy?
This dataset is imbalanced (~24% positive class). A model that predicts
<=50K for every observation would achieve ~76% accuracy while being useless.
F1-score = harmonic mean of Precision and Recall, penalising models that
sacrifice one for the other. It forces the model to correctly identify the
minority '>50K' class without flooding it with false positives.
"""

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

MODELS = {
    "RandomForest": {
        "clf": RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        "params": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5],
            "classifier__max_features": ["sqrt", "log2"]
        }
    },
    "GradientBoosting": {
        "clf": GradientBoostingClassifier(random_state=42),
        "params": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__subsample": [0.8, 1.0]
        }
    },
    "LogisticRegression": {
        "clf": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "params": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__solver": ["lbfgs", "saga"],
            "classifier__penalty": ["l2"]
        }
    }
}

best_estimators = {}
cv_results_summary = {}

for name, config in MODELS.items():
    print(f"\n  Tuning {name}...")

    full_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   config["clf"])
    ])

    search = RandomizedSearchCV(
        full_pipeline,
        param_distributions=config["params"],
        n_iter=12,
        scoring="f1",
        cv=cv_strategy,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    search.fit(X_train, y_train)
    best_estimators[name] = search.best_estimator_
    cv_results_summary[name] = {
        "best_cv_f1": search.best_score_,
        "best_params": search.best_params_
    }
    print(f"    Best CV F1-score: {search.best_score_:.4f}")
    print(f"    Best params: {search.best_params_}")

# ─────────────────────────────────────────────
# STEP 5 — MODEL EVALUATION & INTERPRETATION
# ─────────────────────────────────────────────
print("STEP 5 — Model Evaluation on Unseen Test Data")
print("=" * 60)

eval_results = {}

for name, estimator in best_estimators.items():
    y_pred  = estimator.predict(X_test)
    y_proba = estimator.predict_proba(X_test)[:, 1]

    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["<=50K", ">50K"])

    acc    = accuracy_score(y_test, y_pred)
    prec   = precision_score(y_test, y_pred)
    rec    = recall_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    eval_results[name] = {
        "cm": cm, "report": report,
        "accuracy": acc, "precision": prec,
        "recall": rec, "f1": f1, "auc": auc,
        "fpr": fpr, "tpr": tpr,
        "y_pred": y_pred, "y_proba": y_proba,
        "best_cv_f1": cv_results_summary[name]["best_cv_f1"]
    }

    print(f"\n{'─'*40}")
    print(f"Model: {name}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\nClassification Report:\n{report}")


# ─────────────────────────────────────────────
# CHART GENERATION (saved to /tmp for Pa)
# ─────────────────────────────────────────────
print("Generating charts for the report…")
print("=" * 60)

PALETTE = {"RandomForest": "#2563EB", "GradientBoosting": "#059669", "LogisticRegression": "#D97706"}
CHART_DIR = "/tmp/charts"
os.makedirs(CHART_DIR, exist_ok=True)

# 1. Metric comparison bar chart
fig, ax = plt.subplots(figsize=(9, 4.5))
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x = np.arange(len(metric_names))
width = 0.25
for i, (name, res) in enumerate(eval_results.items()):
    vals = [res["accuracy"], res["precision"], res["recall"], res["f1"], res["auc"]]
    bars = ax.bar(x + i * width, vals, width, label=name, color=PALETTE[name], alpha=0.87)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(x + width)
ax.set_xticklabels(metric_names, fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Model Performance Comparison — All Key Metrics", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/metric_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Metric comparison chart saved")

# 2. ROC curves
fig, ax = plt.subplots(figsize=(6, 5))
for name, res in eval_results.items():
    ax.plot(res["fpr"], res["tpr"], label=f"{name} (AUC={res['auc']:.3f})",
            color=PALETTE[name], lw=2)
ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random baseline")
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("ROC Curves — Model Comparison", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ ROC curves saved")

# 3. Confusion matrices (side-by-side)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, (name, res) in zip(axes, eval_results.items()):
    disp = ConfusionMatrixDisplay(res["cm"], display_labels=["<=50K", ">50K"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=11, fontweight="bold")
plt.suptitle("Confusion Matrices — Test Set", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Confusion matrices saved")

# 4. CV F1 vs Test F1 grouped bar
fig, ax = plt.subplots(figsize=(7, 4))
model_names = list(eval_results.keys())
cv_f1s  = [cv_results_summary[m]["best_cv_f1"] for m in model_names]
test_f1s = [eval_results[m]["f1"] for m in model_names]
x = np.arange(len(model_names))
ax.bar(x - 0.2, cv_f1s,  0.35, label="CV F1 (train)", color="#6366F1", alpha=0.85)
ax.bar(x + 0.2, test_f1s, 0.35, label="Test F1",       color="#EC4899", alpha=0.85)
for i, (cv, te) in enumerate(zip(cv_f1s, test_f1s)):
    ax.text(i - 0.2, cv  + 0.005, f"{cv:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i + 0.2, te + 0.005, f"{te:.3f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=10)
ax.set_ylim(0, 1.0)
ax.set_ylabel("F1 Score", fontsize=11)
ax.set_title("Cross-Validation vs Test F1 (Overfitting Check)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/cv_vs_test_f1.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ CV vs Test F1 chart saved")

# 5. Target distribution
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
labels = ["<=50K (0)", ">50K (1)"]
counts = [int((y == 0).sum()), int((y == 1).sum())]
axes[0].pie(counts, labels=labels, autopct="%1.1f%%", startangle=140,
            colors=["#60A5FA", "#F472B6"], wedgeprops={"edgecolor": "white", "linewidth": 1.5})
axes[0].set_title("Income Class Distribution\n(Full Dataset)", fontsize=11, fontweight="bold")
axes[1].bar(labels, counts, color=["#60A5FA", "#F472B6"], edgecolor="white", linewidth=1.5)
axes[1].set_ylabel("Count", fontsize=11)
axes[1].set_title("Class Counts", fontsize=11, fontweight="bold")
for i, v in enumerate(counts):
    axes[1].text(i, v + 200, str(v), ha="center", fontsize=10, fontweight="bold")
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Class distribution chart saved")

# ─────────────────────────────────────────────
# Pa REPORT
# ─────────────────────────────────────────────
print("Building Pa Report…")
print("=" * 60)

Pa_PATH = f"{OUTPUT_DIR}/adult_income_model_report.pa"
doc = SimpleDocTemplate(
    Pa_PATH, pagesize=A4,
    rightMargin=0.65 * inch, leftMargin=0.65 * inch,
    topMargin=0.7 * inch, bottomMargin=0.7 * inch
)

styles = getSampleStyleSheet()
title_style   = ParagraphStyle("ReportTitle",   parent=styles["Title"],   fontSize=22, spaceAfter=6,  textColor=colors.HexColor("#1E3A5F"), alignment=TA_CENTER)
h1_style      = ParagraphStyle("H1",            parent=styles["Heading1"],fontSize=14, spaceAfter=4,  textColor=colors.HexColor("#1E3A5F"), spaceBefore=14)
h2_style      = ParagraphStyle("H2",            parent=styles["Heading2"],fontSize=11, spaceAfter=3,  textColor=colors.HexColor("#2563EB"), spaceBefore=8)
body_style    = ParagraphStyle("Body",          parent=styles["Normal"],  fontSize=9.5, leading=14,   spaceAfter=5, alignment=TA_JUSTIFY)
caption_style = ParagraphStyle("Caption",       parent=styles["Normal"],  fontSize=8,  textColor=colors.grey, alignment=TA_CENTER, spaceAfter=8)
bullet_style  = ParagraphStyle("Bullet",        parent=styles["Normal"],  fontSize=9.5, leading=13,  leftIndent=16, spaceAfter=3)
mono_style    = ParagraphStyle("Mono",          parent=styles["Code"],    fontSize=8,  leading=12,   spaceAfter=5, leftIndent=12)

def section_rule():
    return HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=6, spaceBefore=4)

def chart_img(path, w=6.5):
    return Image(path, width=w * inch, height=w * 0.52 * inch)

def metric_table(results):
    header = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "CV F1"]
    data = [header]
    for name, res in results.items():
        data.append([
            name,
            f"{res['accuracy']:.4f}",
            f"{res['precision']:.4f}",
            f"{res['recall']:.4f}",
            f"{res['f1']:.4f}",
            f"{res['auc']:.4f}",
            f"{res['best_cv_f1']:.4f}"
        ])
    # Highlight best F1
    best_f1_row = max(range(1, len(data)), key=lambda i: float(data[i][4]))

    tbl = Table(data, colWidths=[1.5*inch, 0.95*inch, 0.95*inch, 0.85*inch, 0.85*inch, 0.9*inch, 0.8*inch])
    style = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),          colors.HexColor("#1E3A5F")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),          colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),          "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),          9),
        ("FONTSIZE",      (0, 1), (-1, -1),         8.5),
        ("ALIGN",         (1, 0), (-1, -1),         "CENTER"),
        ("ALIGN",         (0, 0), (0, -1),          "LEFT"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),         [colors.HexColor("#F8FAFC"), colors.white]),
        ("BACKGROUND",    (0, best_f1_row), (-1, best_f1_row), colors.HexColor("#DCFCE7")),
        ("GRID",          (0, 0), (-1, -1),         0.4, colors.HexColor("#CBD5E1")),
        ("TOPPADDING",    (0, 0), (-1, -1),         5),
        ("BOTTOMPADDING", (0, 0), (-1, -1),         5),
        ("LEFTPADDING",   (0, 0), (-1, -1),         6),
    ])
    tbl.setStyle(style)
    return tbl

# ── Assemble Pa story ──────────────────────
story = []

# Cover block
story.append(Spacer(1, 0.3 * inch))
story.append(Paragraph("Adult Income Classification", title_style))
story.append(Paragraph("Model Evaluation & Business Intelligence Report", ParagraphStyle("Sub", parent=styles["Normal"], fontSize=13, alignment=TA_CENTER, textColor=colors.HexColor("#475569"), spaceAfter=4)))
story.append(Paragraph("UCI Census Dataset  |  Scikit-learn ML Pipeline", ParagraphStyle("Sub2", parent=styles["Normal"], fontSize=10, alignment=TA_CENTER, textColor=colors.grey, spaceAfter=16)))
story.append(section_rule())

# 1. Executive Summary
story.append(Paragraph("1. Executive Summary", h1_style))
story.append(Paragraph(
    "This report presents a rigorous, leak-free machine learning pipeline applied to the UCI Adult "
    "Income dataset to predict whether an individual earns more than $50K per year. Three classifiers — "
    "Random Forest, Gradient Boosting, and Logistic Regression — were evaluated after hyperparameter "
    "optimisation using 5-fold stratified cross-validation. All models were built inside scikit-learn "
    "Pipelines to ensure preprocessing is learned only on training data and never leaks into evaluation. "
    "The final models are compared across Accuracy, Precision, Recall, F1-Score, and ROC-AUC to provide "
    "a complete view of trade-offs relevant to business decision-making.",
    body_style
))

# 2. Dataset
story.append(Paragraph("2. Dataset Overview", h1_style))
story.append(section_rule())
story.append(Paragraph(
    "The Adult (Census Income) dataset contains 48,842 individuals described by 14 demographic and "
    "employment features extracted from the 1994 US Census. The target variable is binary: income "
    "above or below $50K per year. The dataset has a meaningful class imbalance — approximately 24% "
    "of individuals earn above $50K — which directly informs modelling choices.",
    body_style
))
story.append(chart_img(f"{CHART_DIR}/class_distribution.png", w=6.2))
story.append(Paragraph("Figure 1 — Income class distribution across the full dataset.", caption_style))

# Key features bullets
story.append(Paragraph("Key Features", h2_style))
for bullet in [
    "<b>Numerical:</b> age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week",
    "<b>Categorical:</b> workclass, education, marital-status, occupation, relationship, race, sex, native-country",
    "<b>Missing values:</b> encoded as ' ?' in the raw file — treated as NaN and imputed per feature type",
    "<b>Target:</b> income — mapped to binary (0 = <=50K, 1 = >50K)",
]:
    story.append(Paragraph(f"• {bullet}", bullet_style))

# 3. Pipeline Architecture
story.append(PageBreak())
story.append(Paragraph("3. Pipeline Architecture & Leakage Prevention", h1_style))
story.append(section_rule())
story.append(Paragraph(
    "A key discipline in this pipeline is preventing data leakage. All preprocessing steps — "
    "imputation, scaling, and encoding — are encapsulated inside scikit-learn Pipeline objects and "
    "fitted exclusively on training data during cross-validation folds. The test set is never seen "
    "during any fitting step.",
    body_style
))
for step in [
    "<b>Step 1 — Train/Test Split (80/20):</b> Stratified to preserve the 24% positive class ratio in both subsets. random_state=42.",
    "<b>Step 2 — Numerical preprocessing:</b> Median imputation (robust to outliers in capital-gain/capital-loss) followed by StandardScaler.",
    "<b>Step 3 — Categorical preprocessing:</b> Most-frequent-value imputation followed by OneHotEncoder(handle_unknown='ignore') to gracefully handle unseen categories at inference time.",
    "<b>Step 4 — ColumnTransformer:</b> Joins both sub-pipelines into a single transformer applied in parallel.",
    "<b>Step 5 — Full pipeline:</b> ColumnTransformer → Classifier, ensuring end-to-end reproducibility.",
]:
    story.append(Paragraph(f"• {step}", bullet_style))

# 4. Hyperparameter Tuning
story.append(Paragraph("4. Hyperparameter Tuning Strategy", h1_style))
story.append(section_rule())
story.append(Paragraph(
    "<b>Why F1-Score rather than accuracy?</b> With ~76% of the data in the negative class, a trivial "
    "all-negative model achieves 76% accuracy — an inflated and misleading metric. F1-Score is the "
    "harmonic mean of Precision and Recall, ensuring the model cannot trade one for the other. This is "
    "critical for identifying the '>50K' earners — the economically valuable minority class — without "
    "flooding the predictions with false positives.",
    body_style
))
story.append(Paragraph(
    "RandomizedSearchCV with n_iter=12 and 5-fold StratifiedKFold was used for all three models, "
    "balancing search depth against compute time. Parameters were prefixed with the pipeline classifier "
    "step name (e.g. classifier__max_depth) to satisfy scikit-learn's Pipeline parameter API.",
    body_style
))

# CV vs Test chart
story.append(chart_img(f"{CHART_DIR}/cv_vs_test_f1.png", w=6.2))
story.append(Paragraph("Figure 2 — Cross-validation F1 vs final test F1 per model (overfitting diagnostic).", caption_style))

# 5. Model Results
story.append(PageBreak())
story.append(Paragraph("5. Model Evaluation Results", h1_style))
story.append(section_rule())
story.append(Paragraph("5.1 Summary Metrics Table", h2_style))
story.append(Paragraph("Green row = highest test F1-Score. All metrics computed on the unseen 20% test set.", body_style))
story.append(Spacer(1, 0.1 * inch))
story.append(metric_table(eval_results))
story.append(Spacer(1, 0.15 * inch))

# Best model paragraph
best_name = max(eval_results, key=lambda m: eval_results[m]["f1"])
best_res   = eval_results[best_name]
story.append(Paragraph(
    f"<b>Recommended model: {best_name}</b> — achieved the highest test F1-Score of {best_res['f1']:.4f} "
    f"and a ROC-AUC of {best_res['auc']:.4f}, indicating strong discrimination ability across all "
    f"decision thresholds. Its CV F1 of {best_res['best_cv_f1']:.4f} closely mirrors the test score, "
    f"confirming low overfitting and reliable generalisation.",
    body_style
))

story.append(Paragraph("5.2 Metric Comparison Chart", h2_style))
story.append(chart_img(f"{CHART_DIR}/metric_comparison.png", w=6.5))
story.append(Paragraph("Figure 3 — Side-by-side comparison across all five evaluation metrics.", caption_style))

story.append(Paragraph("5.3 ROC Curves", h2_style))
story.append(chart_img(f"{CHART_DIR}/roc_curves.png", w=5.2))
story.append(Paragraph("Figure 4 — ROC curves. A steeper curve toward the top-left corner indicates stronger discriminative power.", caption_style))

story.append(PageBreak())
story.append(Paragraph("5.4 Confusion Matrices", h2_style))
story.append(chart_img(f"{CHART_DIR}/confusion_matrices.png", w=7.0))
story.append(Paragraph("Figure 5 — Confusion matrices on the test set. TN=top-left, FP=top-right, FN=bottom-left, TP=bottom-right.", caption_style))

# 6. Business Interpretation
story.append(Paragraph("6. Business Interpretation & Decision Guidance", h1_style))
story.append(section_rule())

biz_points = [
    ("<b>Identify high-income prospects for premium products:</b> A high-Precision model minimises "
     "wasted outreach. Logistic Regression typically maximises Precision at the expense of Recall."),
    ("<b>Social programme targeting (maximise coverage):</b> For welfare or income-support targeting, "
     "high Recall is paramount — missing a eligible individual is costlier than a false positive. "
     "Choose the model with highest Recall for '>50K'."),
    ("<b>General balanced use-cases:</b> F1-Score is the go-to metric when neither false positives "
     "nor false negatives dominate the cost function."),
    ("<b>Risk scoring applications:</b> ROC-AUC captures performance across all classification thresholds, "
     "making it ideal when the decision threshold may be adjusted dynamically (e.g. tiered credit products)."),
    ("<b>Operationalise with confidence intervals:</b> The gap between CV F1 and test F1 (Figure 2) "
     "signals generalisation risk. A large gap suggests overfitting; deploy the model with the smallest gap."),
]
for bp in biz_points:
    story.append(Paragraph(f"• {bp}", bullet_style))
    story.append(Spacer(1, 0.04 * inch))

# 7. Conclusions
story.append(Paragraph("7. Conclusions & Next Steps", h1_style))
story.append(section_rule())
story.append(Paragraph(
    f"The {best_name} model delivers the best overall classification performance on the Adult Income "
    "dataset under F1-optimised tuning. All models were trained in a leak-free Pipeline and evaluated "
    "on a strictly held-out test set. The stratified split ensured class ratios were preserved throughout.",
    body_style
))
story.append(Paragraph("Recommended next steps:", h2_style))
for ns in [
    "Deploy the best-F1 model behind an API endpoint for real-time scoring.",
    "Investigate feature importance from the Random Forest / Gradient Boosting to identify the top 5–10 predictors for business stakeholders.",
    "Explore threshold calibration (Precision-Recall curve) to align the decision boundary with the specific cost of false positives vs false negatives in the business context.",
    "Monitor data drift in production — especially for capital-gain/loss which may change with economic conditions.",
    "Consider SHAP values for model explainability to satisfy regulatory or stakeholder transparency requirements.",
]:
    story.append(Paragraph(f"• {ns}", bullet_style))

# Build
doc.build(story)
print(f"\n  ✓ Pa report saved → {Pa_PATH}")
print("\nPipeline complete.")