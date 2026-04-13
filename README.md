# CreditSense AI 🏦

> Real-time credit risk scoring API for thin-file borrowers in emerging markets

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## 🎯 Problem

~40% of loan applicants in India are "thin-file" — no CIBIL score, no credit history. Traditional banks reject them automatically. NBFCs and BNPL companies need a smarter, data-driven way to assess risk.

## 💡 Solution

CreditSense AI is a machine learning-powered credit scoring engine that predicts the probability of default using behavioral and financial features — served as a real-time REST API.

---

## 🏗️ Architecture
Applicant Data (JSON)
↓
FastAPI /score
↓
Feature Engineering (18 features)
↓
Logistic Regression / XGBoost / LightGBM
↓
Default Probability + Risk Score (300–850)
↓
APPROVE / DECLINE Decision
---

## 📊 Model Performance

| Model | AUC Score |
|---|---|
| Logistic Regression | 1.0000 ✅ BEST |
| XGBoost | 1.0000 |
| LightGBM | 0.9999 |

- **Optimal threshold:** 0.38
- **Features:** 18 (10 raw + 8 engineered)
- **Class imbalance:** handled with SMOTE
- **Missing values:** handled with median imputation

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/keshavloma1081-ctrl/Creditsense-ai.git
cd Creditsense-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install fastapi uvicorn joblib xgboost lightgbm imbalanced-learn scikit-learn pandas numpy matplotlib seaborn
```

### 4. Train the model
```bash
python pipeline.py
```

### 5. Start the API
```bash
uvicorn api.scoring_api:app --reload --port 8000
```

### 6. Open Swagger UI
http://localhost:8000/docs
---

## 📡 API Usage

### Score an applicant

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "revolving_utilization": 0.45,
    "age": 35,
    "late_30_59_days": 0,
    "debt_ratio": 0.35,
    "monthly_income": 65000,
    "open_credit_lines": 8,
    "late_90_days": 0,
    "real_estate_loans": 0,
    "late_60_89_days": 0,
    "dependents": 1
  }'
```

### Response

```json
{
  "default_probability": 0.0,
  "risk_score": 849,
  "risk_band": "Excellent",
  "decision": "APPROVE",
  "model_version": "Logistic Regression v1.0",
  "threshold_used": 0.38
}
```

---

## 🧪 Risk Profiles

| Profile | Default Prob | Risk Score | Decision |
|---|---|---|---|
| Low Risk | 0.0% | 849 | ✅ APPROVE |
| Medium Risk | 49.4% | 578 | ❌ DECLINE |
| High Risk | 100% | 300 | ❌ DECLINE |

---

## ⚙️ Feature Engineering

| Feature | Description |
|---|---|
| `TotalLatePayments` | Weighted sum of 30/60/90 day lates |
| `IncomePerDependent` | Income normalized by household size |
| `DebtToIncome` | Debt ratio × monthly income |
| `UtilizationXDebt` | Interaction: utilization × debt ratio |
| `AgeGroup` | Bucketed age (0–3) |
| `IsHighUtilization` | Binary flag: utilization > 70% |
| `HasLatePayments` | Binary flag: any late payment |
| `LogIncome` | Log-transformed income |

---

## 🛠️ Tech Stack

- **ML:** XGBoost, LightGBM, Scikit-learn
- **Imbalance:** SMOTE (imbalanced-learn)
- **API:** FastAPI + Uvicorn
- **Serialization:** Joblib
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

---

## 📁 Project Structure
creditsense/
├── pipeline.py          # ML training pipeline
├── api/
│   ├── init.py
│   └── scoring_api.py   # FastAPI scoring service
├── data/
│   └── credit_train.csv
├── models/
│   └── creditsense_model.pkl
└── reports/
├── eda.png
└── evaluation.png


---

## 🎯 Target Users

- NBFCs and microfinance companies
- BNPL startups
- Payment gateways
- Any lender serving thin-file borrowers

---

## 👤 Author

**Keshav** — Data Scientist & ML Engineer
- 4 years experience in ML/AI
- Stack: Python, XGBoost, LightGBM, FastAPI, SQL, Spark

---

## 📄 License

MIT License