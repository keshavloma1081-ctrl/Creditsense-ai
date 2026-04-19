"""
CreditSense AI — Loan Officer Dashboard
========================================
Streamlit dashboard for real-time credit scoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="CreditSense AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0e1a; }
    .metric-card {
        background: #111827;
        border: 1px solid #1f2d42;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .approve-badge {
        background: rgba(0,212,170,0.15);
        border: 2px solid #00d4aa;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #00d4aa;
    }
    .decline-badge {
        background: rgba(255,77,109,0.15);
        border: 2px solid #ff4d6d;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff4d6d;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #00d4aa, #4f8eff);
        color: black;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 12px;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    MODEL_PATH = Path(__file__).parent / 'models' / 'creditsense_model.pkl'
    return joblib.load(MODEL_PATH)

artifacts    = load_model()
model        = artifacts['model']
imputer      = artifacts['imputer']
scaler       = artifacts['scaler']
FEATURE_COLS = artifacts['feature_cols']
THRESHOLD    = artifacts['threshold']
MODEL_NAME   = artifacts['model_name']
AUC_SCORE    = artifacts['auc_score']
ENG_COLS     = artifacts['engineered_cols']

# ── Feature engineering ──────────────────────────────────────
def build_features(data: dict) -> pd.DataFrame:
    raw = pd.DataFrame([data])
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

def score_applicant(data: dict):
    X = build_features(data)
    X_input = scaler.transform(X) if scaler else X.values
    prob = float(model.predict_proba(X_input)[0][1])
    risk_score = int(850 - prob * 550)
    bands = [(750,'Excellent'),(700,'Good'),(650,'Fair'),(600,'Poor')]
    band = next((b for s,b in bands if risk_score >= s), 'Very Poor')
    decision = "APPROVE" if prob < THRESHOLD else "DECLINE"
    return prob, risk_score, band, decision, X

# ── Score gauge chart ────────────────────────────────────────
def draw_gauge(score, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [300, 850], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [300, 600], 'color': '#2d0a0f'},
                {'range': [600, 650], 'color': '#2d1a0a'},
                {'range': [650, 700], 'color': '#1a1a2d'},
                {'range': [700, 750], 'color': '#0a2d1a'},
                {'range': [750, 850], 'color': '#0a2d25'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'}
    )
    return fig

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 CreditSense AI")
    st.markdown(f"**Model:** {MODEL_NAME}")
    st.markdown(f"**AUC:** {AUC_SCORE:.4f}")
    st.markdown(f"**Data:** 150,000 real records")
    st.divider()

    st.markdown("### 📋 Quick Scenarios")
    scenario = st.selectbox("Load a scenario:", 
        ["Custom", "Low Risk", "Medium Risk", "High Risk"])

    scenarios = {
        "Low Risk":    dict(util=0.12, age=52, late30=0, late60=0, late90=0, debt=0.28, income=120000, lines=9, reloans=1, dep=2),
        "Medium Risk": dict(util=0.58, age=34, late30=2, late60=0, late90=0, debt=0.55, income=45000,  lines=5, reloans=0, dep=3),
        "High Risk":   dict(util=0.88, age=26, late30=4, late60=2, late90=1, debt=1.40, income=22000,  lines=3, reloans=0, dep=4),
        "Custom":      dict(util=0.45, age=35, late30=0, late60=0, late90=0, debt=0.35, income=65000,  lines=8, reloans=0, dep=1),
    }
    s = scenarios[scenario]

# ── Main layout ──────────────────────────────────────────────
st.markdown("# 🏦 CreditSense AI — Loan Officer Dashboard")
st.markdown("*Real-time credit risk scoring for thin-file borrowers*")
st.divider()

col1, col2 = st.columns([1, 1])

# ── LEFT: Input form ─────────────────────────────────────────
with col1:
    st.markdown("### 📝 Applicant Details")

    with st.expander("💳 Credit Behavior", expanded=True):
        util   = st.slider("Revolving Utilization", 0.0, 1.0, s['util'], 0.01,
                           help="Ratio of revolving credit used")
        debt   = st.slider("Debt Ratio", 0.0, 3.0, s['debt'], 0.01,
                           help="Monthly debt / monthly income")
        late30 = st.number_input("Times 30-59 Days Late", 0, 20, s['late30'])
        late60 = st.number_input("Times 60-89 Days Late", 0, 20, s['late60'])
        late90 = st.number_input("Times 90+ Days Late",   0, 20, s['late90'])

    with st.expander("👤 Personal & Financial", expanded=True):
        age    = st.slider("Age", 18, 85, s['age'])
        income = st.number_input("Monthly Income (₹)", 0, 500000, s['income'], 1000)
        lines  = st.number_input("Open Credit Lines", 0, 40, s['lines'])
        reloans= st.number_input("Real Estate Loans", 0, 10, s['reloans'])
        dep    = st.number_input("Dependents", 0, 10, s['dep'])

    score_btn = st.button("⚡ Run Credit Score", type="primary")

# ── RIGHT: Results ───────────────────────────────────────────
with col2:
    st.markdown("### 📊 Score Result")

    if score_btn:
        data = {
            'RevolvingUtil'   : util,
            'age'             : age,
            'Late30_59'       : late30,
            'DebtRatio'       : debt,
            'MonthlyIncome'   : float(income),
            'OpenCreditLines' : lines,
            'Late90'          : late90,
            'RealEstateLoans' : reloans,
            'Late60_89'       : late60,
            'Dependents'      : float(dep),
        }

        with st.spinner("Scoring applicant..."):
            prob, risk_score, band, decision, X = score_applicant(data)

        # Score gauge
        color = '#00d4aa' if decision == 'APPROVE' else '#ff4d6d'
        st.plotly_chart(draw_gauge(risk_score, color), use_container_width=True)

        # Decision badge
        if decision == 'APPROVE':
            st.markdown(f'<div class="approve-badge">✅ APPROVED</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="decline-badge">❌ DECLINED</div>', 
                       unsafe_allow_html=True)

        st.markdown("")

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk Score", risk_score)
        m2.metric("Default Prob", f"{prob*100:.1f}%")
        m3.metric("Risk Band", band)

        # SHAP explanation
        st.markdown("### 🔍 Why This Decision?")
        try:
            import shap
            _bg = pd.DataFrame(np.zeros((1, len(ENG_COLS))), columns=ENG_COLS)
            if scaler:
                _bg_scaled = scaler.transform(_bg)
                exp = shap.LinearExplainer(model, _bg_scaled,
                      feature_perturbation="interventional")
                X_input = scaler.transform(X)
            else:
                exp = shap.TreeExplainer(model)
                X_input = X.values

            shap_vals = exp.shap_values(X_input)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1][0]
            else:
                shap_vals = shap_vals[0]

            LABELS = {
                'RevolvingUtil':'Revolving Utilization',
                'age':'Age','Late30_59':'30-59 Days Late',
                'DebtRatio':'Debt Ratio','MonthlyIncome':'Monthly Income',
                'OpenCreditLines':'Open Credit Lines','Late90':'90+ Days Late',
                'RealEstateLoans':'Real Estate Loans','Late60_89':'60-89 Days Late',
                'Dependents':'Dependents','TotalLatePayments':'Total Late Payments',
                'IncomePerDependent':'Income Per Dependent',
                'DebtToIncome':'Debt To Income','UtilizationXDebt':'Util x Debt',
                'AgeGroup':'Age Group','IsHighUtilization':'High Utilization',
                'HasLatePayments':'Has Late Payments','LogIncome':'Log Income',
            }

            shap_df = pd.DataFrame({
                'Feature': [LABELS.get(f, f) for f in ENG_COLS],
                'SHAP': shap_vals,
                'Abs': np.abs(shap_vals)
            }).sort_values('Abs', ascending=False).head(8)

            shap_df['Color'] = shap_df['SHAP'].apply(
                lambda x: '#ff4d6d' if x > 0 else '#00d4aa')

            fig = px.bar(
                shap_df, x='SHAP', y='Feature',
                orientation='h',
                color='Color',
                color_discrete_map='identity',
                title='Feature Impact on Default Risk'
            )
            fig.update_layout(
                showlegend=False,
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color':'white'},
                xaxis=dict(gridcolor='#1f2d42'),
                yaxis=dict(gridcolor='#1f2d42')
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.info("SHAP explanation not available for this model type.")

    else:
        st.info("👈 Fill in applicant details and click **Run Credit Score**")

# ── Bottom: History ──────────────────────────────────────────
st.divider()
st.markdown("### 📈 Scoring History")

DB_PATH = Path(__file__).parent / 'data' / 'predictions.db'
if DB_PATH.exists():
    conn = sqlite3.connect(DB_PATH)
    hist = pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT 20", conn)
    conn.close()

    if len(hist) > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Scored", len(hist))
        c2.metric("Approved", len(hist[hist['decision']=='APPROVE']))
        c3.metric("Declined",  len(hist[hist['decision']=='DECLINE']))
        c4.metric("Avg Risk Score", f"{hist['risk_score'].mean():.0f}")

        st.dataframe(
            hist[['timestamp','age','income','debt_ratio',
                  'util','late_total','default_prob',
                  'risk_score','risk_band','decision']].style.map(
                lambda x: 'color: #00d4aa' if x == 'APPROVE' 
                         else ('color: #ff4d6d' if x == 'DECLINE' else ''),
                subset=['decision']
            ),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No predictions yet. Score some applicants above!")
else:
    st.info("Score some applicants to see history here.")

st.divider()
st.markdown(
    f"*CreditSense AI v3.0 | Model: {MODEL_NAME} | "
    f"AUC: {AUC_SCORE:.4f} | Trained on 150,000 real records*"
)