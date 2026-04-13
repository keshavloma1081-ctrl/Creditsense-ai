"""
CreditSense AI — Credit Risk Scoring Pipeline
==============================================
End-to-end pipeline: EDA → Preprocessing → Feature Engineering →
Model Training (XGBoost + LightGBM) → Evaluation → Model Export
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import lightgbm as lgb
import joblib

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE = Path('C:/Users/kk536/creditsense')
DATA_PATH   = BASE / 'data/credit_train.csv'
MODEL_PATH  = BASE / 'models'
REPORT_PATH = BASE / 'reports'

MODEL_PATH.mkdir(exist_ok=True)
REPORT_PATH.mkdir(exist_ok=True)

FEATURE_COLS = [
    'RevolvingUtil', 'age', 'Late30_59', 'DebtRatio',
    'MonthlyIncome', 'OpenCreditLines', 'Late90',
    'RealEstateLoans', 'Late60_89', 'Dependents'
]
TARGET = 'SeriousDlqin2yrs'

# ─────────────────────────────────────────
# 1. GENERATE DATA
# ─────────────────────────────────────────
print("=" * 60)
print("  CreditSense AI — ML Pipeline")
print("=" * 60)

np.random.seed(42)
n = 15000
default_prob = 0.067
y = np.random.binomial(1, default_prob, n)

age_final = np.where(y==1, np.random.normal(42,12,n), np.random.normal(52,14,n)).clip(21,85).astype(int)
monthly_income = np.where(y==1, np.random.lognormal(8.5,0.6,n), np.random.lognormal(9.2,0.5,n)).astype(float)
monthly_income[np.random.random(n)<0.20] = np.nan
dependents = np.random.poisson(1.0,n).clip(0,8).astype(float)
dependents[np.random.random(n)<0.025] = np.nan

df = pd.DataFrame({
    'SeriousDlqin2yrs': y,
    'RevolvingUtil': np.where(y==1, np.random.beta(5,2,n), np.random.beta(2,5,n)).clip(0,1),
    'age': age_final,
    'Late30_59': np.where(y==1, np.random.poisson(2,n), np.random.poisson(0.2,n)).clip(0,10).astype(int),
    'DebtRatio': np.where(y==1, np.random.beta(4,2,n)*1.5, np.random.beta(2,4,n)*0.8).clip(0,3),
    'MonthlyIncome': monthly_income,
    'OpenCreditLines': np.random.poisson(8,n).clip(0,30).astype(int),
    'Late90': np.where(y==1, np.random.poisson(1.5,n), np.random.poisson(0.05,n)).clip(0,10).astype(int),
    'RealEstateLoans': np.random.poisson(1.1,n).clip(0,10).astype(int),
    'Late60_89': np.where(y==1, np.random.poisson(1,n), np.random.poisson(0.03,n)).clip(0,8).astype(int),
    'Dependents': dependents,
})

df.to_csv(DATA_PATH, index=False)
print(f"\n[1] DATA GENERATED: {df.shape[0]:,} rows x {df.shape[1]} cols")
print(f"    Default rate : {df[TARGET].mean()*100:.2f}%")
print(f"    Missing vals :\n{df.isnull().sum()[df.isnull().sum()>0]}")

# ─────────────────────────────────────────
# 2. EDA PLOTS
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('CreditSense AI — Exploratory Data Analysis', fontsize=16, fontweight='bold')

axes[0,0].bar(['No Default', 'Default'],
               df[TARGET].value_counts().sort_index(),
               color=['#2ecc71', '#e74c3c'], edgecolor='white')
axes[0,0].set_title('Class Distribution')
axes[0,0].set_ylabel('Count')

axes[0,1].hist(df.loc[df[TARGET]==0, 'age'], bins=30, alpha=0.6, color='#2ecc71', label='No Default')
axes[0,1].hist(df.loc[df[TARGET]==1, 'age'], bins=30, alpha=0.6, color='#e74c3c', label='Default')
axes[0,1].set_title('Age Distribution')
axes[0,1].legend()

axes[0,2].hist(df.loc[df[TARGET]==0, 'RevolvingUtil'], bins=30, alpha=0.6, color='#2ecc71', label='No Default')
axes[0,2].hist(df.loc[df[TARGET]==1, 'RevolvingUtil'], bins=30, alpha=0.6, color='#e74c3c', label='Default')
axes[0,2].set_title('Revolving Utilization')
axes[0,2].legend()

axes[1,0].hist(df.loc[df[TARGET]==0, 'DebtRatio'].clip(0,2), bins=30, alpha=0.6, color='#2ecc71', label='No Default')
axes[1,0].hist(df.loc[df[TARGET]==1, 'DebtRatio'].clip(0,2), bins=30, alpha=0.6, color='#e74c3c', label='Default')
axes[1,0].set_title('Debt Ratio')
axes[1,0].legend()

inc_no = df.loc[(df[TARGET]==0) & df['MonthlyIncome'].notna(), 'MonthlyIncome']
inc_yes = df.loc[(df[TARGET]==1) & df['MonthlyIncome'].notna(), 'MonthlyIncome']
axes[1,1].hist(np.log1p(inc_no), bins=30, alpha=0.6, color='#2ecc71', label='No Default')
axes[1,1].hist(np.log1p(inc_yes), bins=30, alpha=0.6, color='#e74c3c', label='Default')
axes[1,1].set_title('Monthly Income (log)')
axes[1,1].legend()

corr = df[FEATURE_COLS + [TARGET]].corr()
sns.heatmap(corr[[TARGET]].drop(TARGET).sort_values(TARGET),
            annot=True, fmt='.2f', cmap='RdYlGn_r', ax=axes[1,2])
axes[1,2].set_title('Feature Correlation with Default')

plt.tight_layout()
plt.savefig(REPORT_PATH / 'eda.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n[2] EDA plot saved → reports/eda.png")

# ─────────────────────────────────────────
# 3. PREPROCESSING & FEATURE ENGINEERING
# ─────────────────────────────────────────
print("\n[3] FEATURE ENGINEERING...")

X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()

imputer = SimpleImputer(strategy='median')
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=FEATURE_COLS)

X_eng = X_imp.copy()
X_eng['TotalLatePayments']  = X_imp['Late30_59'] + X_imp['Late60_89'] + X_imp['Late90']
X_eng['IncomePerDependent'] = X_imp['MonthlyIncome'] / (X_imp['Dependents'] + 1)
X_eng['DebtToIncome']       = X_imp['DebtRatio'] * X_imp['MonthlyIncome']
X_eng['UtilizationXDebt']   = X_imp['RevolvingUtil'] * X_imp['DebtRatio']
X_eng['AgeGroup']           = pd.cut(X_imp['age'], bins=[0,30,45,60,100], labels=[0,1,2,3]).astype(int)
X_eng['IsHighUtilization']  = (X_imp['RevolvingUtil'] > 0.7).astype(int)
X_eng['HasLatePayments']    = (X_eng['TotalLatePayments'] > 0).astype(int)
X_eng['LogIncome']          = np.log1p(X_imp['MonthlyIncome'])

print(f"    Features after engineering: {X_eng.shape[1]} (was {len(FEATURE_COLS)})")

ENGINEERED_COLS = list(X_eng.columns)

X_train, X_test, y_train, y_test = train_test_split(
    X_eng, y, test_size=0.20, random_state=42, stratify=y
)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"    After SMOTE: {dict(zip(*np.unique(y_train_res, return_counts=True)))}")

# ─────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────
print("\n[4] TRAINING MODELS...")

results = {}

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_train_res)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
results['Logistic Regression'] = {'model': lr, 'proba': lr_proba, 'auc': roc_auc_score(y_test, lr_proba)}
print(f"    Logistic Regression AUC: {results['Logistic Regression']['auc']:.4f}")

xgb_model = xgb.XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='auc', random_state=42, verbosity=0
)
xgb_model.fit(X_train_res, y_train_res, eval_set=[(X_test, y_test)], verbose=False)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
results['XGBoost'] = {'model': xgb_model, 'proba': xgb_proba, 'auc': roc_auc_score(y_test, xgb_proba)}
print(f"    XGBoost AUC            : {results['XGBoost']['auc']:.4f}")

lgb_model = lgb.LGBMClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1
)
lgb_model.fit(X_train_res, y_train_res)
lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
results['LightGBM'] = {'model': lgb_model, 'proba': lgb_proba, 'auc': roc_auc_score(y_test, lgb_proba)}
print(f"    LightGBM AUC           : {results['LightGBM']['auc']:.4f}")

best_name = max(results, key=lambda k: results[k]['auc'])
best_model = results[best_name]['model']
best_proba = results[best_name]['proba']
print(f"\n    Best model: {best_name} (AUC={results[best_name]['auc']:.4f})")

# ─────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────
print("\n[5] EVALUATION...")

from sklearn.metrics import f1_score
thresholds = np.arange(0.1, 0.9, 0.01)
f1s = [f1_score(y_test, (best_proba >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[np.argmax(f1s)]
y_pred = (best_proba >= best_threshold).astype(int)

print(f"    Optimal threshold: {best_threshold:.2f}")
print(f"\n    Classification Report:\n{classification_report(y_test, y_pred, target_names=['No Default','Default'])}")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(f'CreditSense AI — Model Evaluation ({best_name})', fontsize=15, fontweight='bold')

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['proba'])
    axes[0,0].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", linewidth=2)
axes[0,0].plot([0,1],[0,1],'k--')
axes[0,0].set_title('ROC Curves')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

precision, recall, _ = precision_recall_curve(y_test, best_proba)
ap = average_precision_score(y_test, best_proba)
axes[0,1].plot(recall, precision, color='#e74c3c', linewidth=2)
axes[0,1].set_title(f'Precision-Recall (AP={ap:.3f})')
axes[0,1].grid(True, alpha=0.3)

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Default','Default'],
            yticklabels=['No Default','Default'], ax=axes[0,2])
axes[0,2].set_title('Confusion Matrix')

if hasattr(best_model, 'feature_importances_'):
    imp = pd.Series(best_model.feature_importances_, index=ENGINEERED_COLS)
    imp.sort_values().tail(12).plot(kind='barh', ax=axes[1,0], color='#3498db')
    axes[1,0].set_title('Feature Importances')
    axes[1,0].grid(True, alpha=0.3)

axes[1,1].hist(best_proba[y_test==0], bins=50, alpha=0.6, color='#2ecc71', label='No Default', density=True)
axes[1,1].hist(best_proba[y_test==1], bins=50, alpha=0.6, color='#e74c3c', label='Default', density=True)
axes[1,1].axvline(best_threshold, color='black', linestyle='--', label=f'Threshold={best_threshold:.2f}')
axes[1,1].set_title('Score Distribution')
axes[1,1].legend()

axes[1,2].plot(thresholds, f1s, color='#9b59b6', linewidth=2)
axes[1,2].axvline(best_threshold, color='red', linestyle='--')
axes[1,2].set_title('F1 vs Threshold')
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(REPORT_PATH / 'evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("    Evaluation plot saved → reports/evaluation.png")

# ─────────────────────────────────────────
# 6. SAVE MODEL
# ─────────────────────────────────────────
print("\n[6] SAVING MODEL...")

artifacts = {
    'model': best_model,
    'imputer': imputer,
    'scaler': scaler if best_name == 'Logistic Regression' else None,
    'feature_cols': FEATURE_COLS,
    'engineered_cols': ENGINEERED_COLS,
    'threshold': float(best_threshold),
    'model_name': best_name,
    'auc_score': float(results[best_name]['auc']),
}

joblib.dump(artifacts, MODEL_PATH / 'creditsense_model.pkl')

print("\n" + "="*60)
print("  PIPELINE COMPLETE")
print("="*60)
for name, res in results.items():
    marker = " ✅ BEST" if name == best_name else ""
    print(f"  {name:25s} AUC: {res['auc']:.4f}{marker}")
print(f"\n  Threshold : {best_threshold:.2f}")
print(f"  Features  : {len(ENGINEERED_COLS)}")
print(f"  Model saved → models/creditsense_model.pkl")
print("="*60)