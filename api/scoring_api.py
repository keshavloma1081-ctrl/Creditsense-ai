from fastapi import FastAPI, HTTPException, Security, Request, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from typing import Optional, List
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import joblib
import sqlite3
import datetime
import shap
import os
from pathlib import Path

# ── Load environment variables ───────────────────────────────
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY", "creditsense-secret-2024")
API_KEYS   = os.getenv("API_KEYS", "").split(",")

# ── Load model artifacts ─────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / 'models' / 'creditsense_model.pkl'
artifacts  = joblib.load(MODEL_PATH)

model        = artifacts['model']
imputer      = artifacts['imputer']
scaler       = artifacts['scaler']
FEATURE_COLS = artifacts['feature_cols']
THRESHOLD    = artifacts['threshold']
MODEL_NAME   = artifacts['model_name']
AUC_SCORE    = artifacts['auc_score']
ENG_COLS     = artifacts['engineered_cols']

# ── Database setup ───────────────────────────────────────────
DB_PATH = Path(__file__).parent.parent / 'data' / 'predictions.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT,
            api_key       TEXT,
            age           INTEGER,
            income        REAL,
            debt_ratio    REAL,
            util          REAL,
            late_total    INTEGER,
            default_prob  REAL,
            risk_score    INTEGER,
            risk_band     TEXT,
            decision      TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def log_prediction(api_key, req, prob, score, band, decision):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        INSERT INTO predictions
        (timestamp, api_key, age, income, debt_ratio, util, late_total,
         default_prob, risk_score, risk_band, decision)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.datetime.utcnow().isoformat(),
        api_key,
        req.age,
        req.monthly_income,
        req.debt_ratio,
        req.revolving_utilization,
        req.late_30_59_days + req.late_60_89_days + req.late_90_days,
        prob, score, band, decision
    ))
    conn.commit()
    conn.close()

# ── SHAP Explainer ───────────────────────────────────────────
FEATURE_LABELS = {
    'RevolvingUtil'      : 'Revolving credit utilization',
    'age'                : 'Age',
    'Late30_59'          : 'Times 30-59 days late',
    'DebtRatio'          : 'Debt ratio',
    'MonthlyIncome'      : 'Monthly income',
    'OpenCreditLines'    : 'Open credit lines',
    'Late90'             : 'Times 90+ days late',
    'RealEstateLoans'    : 'Real estate loans',
    'Late60_89'          : 'Times 60-89 days late',
    'Dependents'         : 'Number of dependents',
    'TotalLatePayments'  : 'Total late payments',
    'IncomePerDependent' : 'Income per dependent',
    'DebtToIncome'       : 'Debt to income ratio',
    'UtilizationXDebt'   : 'Utilization x debt interaction',
    'AgeGroup'           : 'Age group',
    'IsHighUtilization'  : 'High utilization flag',
    'HasLatePayments'    : 'Has late payments flag',
    'LogIncome'          : 'Log income',
}

# Build a background dataset for SHAP
_bg = pd.DataFrame(
    np.zeros((1, len(ENG_COLS))),
    columns=ENG_COLS
)
if scaler:
    _bg_scaled = scaler.transform(_bg)
    explainer = shap.LinearExplainer(
        model,
        _bg_scaled,
        feature_perturbation="interventional"
    )
else:
    explainer = shap.TreeExplainer(model)

# ── Rate limiter ─────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── API Key auth ─────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key not in API_KEYS:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key. Contact team@creditsense.ai"
        )
    return api_key

# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="CreditSense AI",
    description="Real-time credit risk scoring API for thin-file borrowers",
    version="3.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Schemas ──────────────────────────────────────────────────
class ScoringRequest(BaseModel):
    revolving_utilization: float = Field(..., ge=0, le=1)
    age: int = Field(..., ge=18, le=100)
    late_30_59_days: int = Field(0, ge=0)
    debt_ratio: float = Field(..., ge=0)
    monthly_income: Optional[float] = Field(None, ge=0)
    open_credit_lines: int = Field(0, ge=0)
    late_90_days: int = Field(0, ge=0)
    real_estate_loans: int = Field(0, ge=0)
    late_60_89_days: int = Field(0, ge=0)
    dependents: Optional[int] = Field(None, ge=0)

class ScoringResponse(BaseModel):
    default_probability: float
    risk_score: int
    risk_band: str
    decision: str
    model_version: str
    threshold_used: float
    explanation: Optional[dict] = None

class BatchRequest(BaseModel):
    applicants: List[ScoringRequest]

class BatchResponse(BaseModel):
    total: int
    approved: int
    declined: int
    results: List[ScoringResponse]

# ── Feature engineering ──────────────────────────────────────
def build_features(req: ScoringRequest) -> pd.DataFrame:
    raw = pd.DataFrame([{
        'RevolvingUtil'   : req.revolving_utilization,
        'age'             : req.age,
        'Late30_59'       : req.late_30_59_days,
        'DebtRatio'       : req.debt_ratio,
        'MonthlyIncome'   : req.monthly_income,
        'OpenCreditLines' : req.open_credit_lines,
        'Late90'          : req.late_90_days,
        'RealEstateLoans' : req.real_estate_loans,
        'Late60_89'       : req.late_60_89_days,
        'Dependents'      : req.dependents,
    }])
    imp_arr = imputer.transform(raw[FEATURE_COLS])
    X = pd.DataFrame(imp_arr, columns=FEATURE_COLS)
    X['TotalLatePayments']  = X['Late30_59'] + X['Late60_89'] + X['Late90']
    X['IncomePerDependent'] = X['MonthlyIncome'] / (X['Dependents'] + 1)
    X['DebtToIncome']       = X['DebtRatio'] * X['MonthlyIncome']
    X['UtilizationXDebt']   = X['RevolvingUtil'] * X['DebtRatio']
    X['AgeGroup']           = pd.cut(X['age'], bins=[0,30,45,60,100], labels=[0,1,2,3]).astype(int)
    X['IsHighUtilization']  = (X['RevolvingUtil'] > 0.7).astype(int)
    X['HasLatePayments']    = (X['TotalLatePayments'] > 0).astype(int)
    X['LogIncome']          = np.log1p(X['MonthlyIncome'])
    return X

# ── Core scoring function ────────────────────────────────────
def score_one(req: ScoringRequest, explain: bool = False):
    X = build_features(req)
    X_input = scaler.transform(X) if scaler else X.values
    prob = float(model.predict_proba(X_input)[0][1])
    risk_score = int(850 - prob * 550)
    bands = [(750,'Excellent'),(700,'Good'),(650,'Fair'),(600,'Poor')]
    band = next((b for s,b in bands if risk_score >= s), 'Very Poor')
    decision = "APPROVE" if prob < THRESHOLD else "DECLINE"

    explanation = None
    if explain:
        shap_vals = explainer.shap_values(X_input)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1][0]
        else:
            shap_vals = shap_vals[0]

        shap_dict = dict(zip(ENG_COLS, shap_vals))
        sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

        top_reasons = []
        for feat, val in sorted_shap[:3]:
            label = FEATURE_LABELS.get(feat, feat)
            direction = "High" if val > 0 else "Low"
            feat_val = float(X[feat].iloc[0]) if feat in X.columns else 0
            top_reasons.append(f"{direction} {label}: {feat_val:.2f}")

        explanation = {
            "top_reasons": top_reasons,
            "shap_values": {
                FEATURE_LABELS.get(f, f): round(float(v), 4)
                for f, v in sorted_shap[:5]
            }
        }

    return prob, risk_score, band, decision, explanation

# ── Endpoints ────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "product" : "CreditSense AI",
        "version" : "3.0.0",
        "model"   : MODEL_NAME,
        "auc"     : round(AUC_SCORE, 4),
        "status"  : "healthy"
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/score", response_model=ScoringResponse)
@limiter.limit("30/minute")
def score_applicant(
    request: Request,
    req: ScoringRequest,
    explain: bool = False,
    api_key: str = Depends(verify_api_key)
):
    try:
        prob, risk_score, band, decision, explanation = score_one(req, explain)
        log_prediction(api_key, req, prob, risk_score, band, decision)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ScoringResponse(
        default_probability = round(prob, 4),
        risk_score          = risk_score,
        risk_band           = band,
        decision            = decision,
        model_version       = f"{MODEL_NAME} v3.0",
        threshold_used      = THRESHOLD,
        explanation         = explanation
    )

@app.post("/batch-score", response_model=BatchResponse)
@limiter.limit("5/minute")
def batch_score(
    request: Request,
    batch: BatchRequest,
    api_key: str = Depends(verify_api_key)
):
    if len(batch.applicants) > 1000:
        raise HTTPException(status_code=400, detail="Max 1000 applicants per batch")

    results = []
    for req in batch.applicants:
        try:
            prob, risk_score, band, decision, explanation = score_one(req)
            log_prediction(api_key, req, prob, risk_score, band, decision)
            results.append(ScoringResponse(
                default_probability = round(prob, 4),
                risk_score          = risk_score,
                risk_band           = band,
                decision            = decision,
                model_version       = f"{MODEL_NAME} v3.0",
                threshold_used      = THRESHOLD,
                explanation         = explanation
            ))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    approved = sum(1 for r in results if r.decision == "APPROVE")
    return BatchResponse(
        total    = len(results),
        approved = approved,
        declined = len(results) - approved,
        results  = results
    )

@app.get("/stats")
def stats(api_key: str = Depends(verify_api_key)):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute('''
        SELECT
            COUNT(*)           as total,
            SUM(CASE WHEN decision="APPROVE" THEN 1 ELSE 0 END) as approved,
            SUM(CASE WHEN decision="DECLINE" THEN 1 ELSE 0 END) as declined,
            ROUND(AVG(default_prob), 4) as avg_default_prob,
            ROUND(AVG(risk_score), 1)   as avg_risk_score
        FROM predictions
        WHERE api_key = ?
    ''', (api_key,))
    row = cursor.fetchone()
    conn.close()
    return {
        "total_scored"     : row[0],
        "approved"         : row[1],
        "declined"         : row[2],
        "avg_default_prob" : row[3],
        "avg_risk_score"   : row[4]
    }