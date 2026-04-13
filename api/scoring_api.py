from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ── Load artifacts ──────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / 'models' / 'creditsense_model.pkl'
artifacts  = joblib.load(MODEL_PATH)

model          = artifacts['model']
imputer        = artifacts['imputer']
scaler         = artifacts['scaler']
FEATURE_COLS   = artifacts['feature_cols']
THRESHOLD      = artifacts['threshold']
MODEL_NAME     = artifacts['model_name']
AUC_SCORE      = artifacts['auc_score']

# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="CreditSense AI",
    description="Real-time credit risk scoring API for thin-file borrowers",
    version="1.0.0"
)

# ── Request / Response schemas ───────────────────────────────
class ScoringRequest(BaseModel):
    revolving_utilization: float = Field(..., ge=0, le=1,
        description="Revolving credit utilization ratio (0-1)")
    age: int = Field(..., ge=18, le=100,
        description="Applicant age")
    late_30_59_days: int = Field(0, ge=0,
        description="Times 30-59 days past due")
    debt_ratio: float = Field(..., ge=0,
        description="Monthly debt payments / monthly income")
    monthly_income: Optional[float] = Field(None, ge=0,
        description="Monthly income (optional)")
    open_credit_lines: int = Field(0, ge=0,
        description="Number of open credit lines")
    late_90_days: int = Field(0, ge=0,
        description="Times 90+ days past due")
    real_estate_loans: int = Field(0, ge=0,
        description="Number of real estate loans")
    late_60_89_days: int = Field(0, ge=0,
        description="Times 60-89 days past due")
    dependents: Optional[int] = Field(None, ge=0,
        description="Number of dependents (optional)")

class ScoringResponse(BaseModel):
    default_probability: float
    risk_score: int
    risk_band: str
    decision: str
    model_version: str
    threshold_used: float

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
    X['AgeGroup']           = pd.cut(X['age'],
                                     bins=[0,30,45,60,100],
                                     labels=[0,1,2,3]).astype(int)
    X['IsHighUtilization']  = (X['RevolvingUtil'] > 0.7).astype(int)
    X['HasLatePayments']    = (X['TotalLatePayments'] > 0).astype(int)
    X['LogIncome']          = np.log1p(X['MonthlyIncome'])

    return X

def prob_to_risk_score(prob: float) -> int:
    return int(850 - (prob * 550))

def get_risk_band(score: int) -> str:
    if score >= 750: return "Excellent"
    if score >= 700: return "Good"
    if score >= 650: return "Fair"
    if score >= 600: return "Poor"
    return "Very Poor"

# ── Endpoints ────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "product"   : "CreditSense AI",
        "version"   : "1.0.0",
        "model"     : MODEL_NAME,
        "auc_score" : round(AUC_SCORE, 4),
        "status"    : "healthy"
    }

@app.post("/score", response_model=ScoringResponse)
def score_applicant(req: ScoringRequest):
    try:
        X = build_features(req)
        if scaler:
            X_input = scaler.transform(X)
        else:
            X_input = X.values
        prob = float(model.predict_proba(X_input)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")

    risk_score = prob_to_risk_score(prob)
    risk_band  = get_risk_band(risk_score)
    decision   = "APPROVE" if prob < THRESHOLD else "DECLINE"

    return ScoringResponse(
        default_probability = round(prob, 4),
        risk_score          = risk_score,
        risk_band           = risk_band,
        decision            = decision,
        model_version       = f"{MODEL_NAME} v1.0",
        threshold_used      = THRESHOLD
    )

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}