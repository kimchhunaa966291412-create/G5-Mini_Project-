"""
Dropout Risk Dashboard  —  MIN_P Group 05
Enhanced UI version
Run:  streamlit run streamlit_demo.py
"""
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dropout Risk · G05",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #080c14 !important;
    color: #c9d1e0 !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stAppViewContainer"] > .main { padding-top: 0 !important; }
.block-container { padding: 0 2rem 3rem !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1220 0%, #080c14 100%) !important;
    border-right: 1px solid #1a2540 !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d1b35 0%, #091428 40%, #0d0f1a 100%);
    border-bottom: 1px solid #1a2540;
    padding: 2.2rem 2.5rem 1.8rem;
    margin: 0 -2rem 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(74,158,255,.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(6,214,160,.07) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-size: .7rem; font-weight: 700; letter-spacing: .18em;
    text-transform: uppercase; color: #4a9eff; margin-bottom: .5rem;
}
.hero-title {
    font-size: 2rem; font-weight: 800; color: #e8eeff;
    line-height: 1.15; margin: 0 0 .4rem;
}
.hero-title span { color: #4a9eff; }
.hero-sub {
    font-size: .85rem; color: #556070; font-weight: 400;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(74,158,255,.1); border: 1px solid rgba(74,158,255,.25);
    border-radius: 20px; padding: 4px 12px; font-size: .72rem;
    color: #4a9eff; font-weight: 600; margin-top: .8rem;
}

/* ── Metric cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1.8rem; }
.kpi-card {
    background: linear-gradient(135deg, #0f1928 0%, #0b1220 100%);
    border: 1px solid #1a2540;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    position: relative; overflow: hidden;
    transition: border-color .2s, transform .2s;
}
.kpi-card:hover { border-color: #2a3f60; transform: translateY(-2px); }
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: var(--accent, #4a9eff);
    border-radius: 14px 14px 0 0;
}
.kpi-label { font-size: .68rem; font-weight: 700; letter-spacing: .12em;
             text-transform: uppercase; color: #4a6080; margin-bottom: .5rem; }
.kpi-value { font-size: 2rem; font-weight: 800; color: #e8eeff; line-height: 1; }
.kpi-delta { font-size: .75rem; color: #4a6080; margin-top: .3rem; }
.kpi-icon { position: absolute; top: 1rem; right: 1.2rem; font-size: 1.6rem; opacity: .15; }

/* ── Section headers ── */
.sec-header {
    display: flex; align-items: center; gap: 10px;
    margin: 1.6rem 0 .8rem;
}
.sec-header-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, #1a2540, transparent);
}
.sec-label {
    font-size: .68rem; font-weight: 700; letter-spacing: .15em;
    text-transform: uppercase; color: #4a9eff; white-space: nowrap;
}

/* ── Chart panels ── */
.chart-panel {
    background: linear-gradient(135deg, #0f1928 0%, #0b1220 100%);
    border: 1px solid #1a2540; border-radius: 14px;
    padding: 1.2rem 1.4rem; margin-bottom: 1rem;
}
.chart-title {
    font-size: .8rem; font-weight: 600; color: #8090a8;
    letter-spacing: .04em; margin-bottom: .8rem;
    text-transform: uppercase; font-size: .68rem; letter-spacing: .12em;
}

/* ── Risk meter ── */
.risk-display {
    background: linear-gradient(135deg, #0f1928 0%, #0b1220 100%);
    border: 1px solid #1a2540; border-radius: 14px;
    padding: 1.6rem; text-align: center;
}
.risk-pct {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3.5rem; font-weight: 600;
    line-height: 1; margin: .4rem 0;
}
.risk-badge {
    display: inline-flex; align-items: center; gap: 6px;
    border-radius: 8px; padding: 6px 16px;
    font-weight: 700; font-size: .82rem; letter-spacing: .06em;
    margin-bottom: .4rem;
}
.risk-high   { background: rgba(255,107,107,.12); color: #ff6b6b; border: 1px solid rgba(255,107,107,.3); }
.risk-medium { background: rgba(255,209,102,.12); color: #ffd166; border: 1px solid rgba(255,209,102,.3); }
.risk-low    { background: rgba(6,214,160,.12);   color: #06d6a0; border: 1px solid rgba(6,214,160,.3); }
.risk-label  { font-size: .68rem; letter-spacing: .14em; text-transform: uppercase; color: #4a6080; margin-top: .2rem; }

/* ── Gauge arc ── */
.gauge-wrap { position: relative; margin: .5rem auto; width: 160px; height: 80px; }

/* ── Signal cards ── */
.signal-card {
    background: rgba(255,209,102,.05);
    border: 1px solid rgba(255,209,102,.15);
    border-left: 3px solid #ffd166;
    border-radius: 8px; padding: .65rem 1rem;
    margin: .4rem 0; font-size: .82rem;
    display: flex; align-items: flex-start; gap: 8px;
}
.signal-icon { font-size: 1rem; flex-shrink: 0; }
.signal-text { color: #c9d1e0; }
.signal-detail { color: #4a6080; font-size: .75rem; display: block; margin-top: 1px; }
.signal-ok {
    background: rgba(6,214,160,.05);
    border-color: rgba(6,214,160,.15); border-left-color: #06d6a0;
}

/* ── Model table ── */
.model-row {
    background: linear-gradient(135deg, #0f1928 0%, #0b1220 100%);
    border: 1px solid #1a2540; border-radius: 10px;
    padding: .9rem 1.2rem; margin-bottom: .6rem;
    display: grid; grid-template-columns: 1.8fr 1fr 1fr 1fr 1.2fr 1.2fr;
    align-items: center; transition: border-color .2s;
}
.model-row:hover { border-color: #2a3f60; }
.model-row.winner { border-color: rgba(74,158,255,.4); background: linear-gradient(135deg, #0f2040 0%, #0b1830 100%); }
.model-name { font-weight: 600; color: #e8eeff; font-size: .88rem; }
.model-badge { display: inline-block; background: rgba(74,158,255,.15); color: #4a9eff;
               border-radius: 4px; padding: 1px 7px; font-size: .68rem; margin-left: 6px; }
.model-stat { font-family: 'JetBrains Mono', monospace; font-size: .85rem; color: #8090a8; }
.model-stat-hi { color: #06d6a0; }
.model-header {
    display: grid; grid-template-columns: 1.8fr 1fr 1fr 1fr 1.2fr 1.2fr;
    padding: .4rem 1.2rem; margin-bottom: .3rem;
    font-size: .65rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: #3a5070;
}

/* ── SHAP feature bars ── */
.feat-row { display: flex; align-items: center; gap: 10px; margin: .35rem 0; }
.feat-name { font-size: .78rem; color: #8090a8; width: 220px; flex-shrink: 0;
             overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.feat-bar-wrap { flex: 1; background: #0f1928; border-radius: 4px; height: 8px; }
.feat-bar { height: 8px; border-radius: 4px;
            background: linear-gradient(90deg, #1a5aff, #4a9eff); }
.feat-val { font-family: 'JetBrains Mono', monospace; font-size: .72rem;
            color: #4a9eff; width: 50px; text-align: right; flex-shrink: 0; }

/* ── Sidebar nav ── */
.nav-logo {
    background: linear-gradient(135deg, #0d1b35, #091428);
    border-bottom: 1px solid #1a2540;
    padding: 1.4rem 1.2rem 1rem;
    margin: 0 -1rem 1rem;
}
.nav-logo-icon { font-size: 2rem; }
.nav-logo-title { font-size: 1rem; font-weight: 800; color: #e8eeff; margin-top: .3rem; }
.nav-logo-sub   { font-size: .7rem; color: #3a5070; margin-top: 2px; }
.nav-group-label {
    font-size: .62rem; font-weight: 700; letter-spacing: .16em;
    text-transform: uppercase; color: #2a4060; padding: .8rem 0 .3rem;
}

/* ── Streamlit radio override ── */
[data-testid="stRadio"] > label { display: none !important; }
[data-testid="stRadio"] > div { gap: 4px !important; }
[data-testid="stRadio"] > div > label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 8px !important; padding: .55rem .9rem !important;
    color: #4a6080 !important; font-size: .83rem !important;
    font-weight: 500 !important; cursor: pointer !important;
    transition: all .15s !important; width: 100% !important;
}
[data-testid="stRadio"] > div > label:hover {
    background: rgba(74,158,255,.06) !important;
    color: #8090a8 !important; border-color: #1a2540 !important;
}
[data-testid="stRadio"] > div > label[data-baseweb="radio"] > div:first-child { display: none !important; }
div[role="radiogroup"] > label[data-selected="true"],
div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(74,158,255,.1) !important;
    color: #4a9eff !important; border-color: rgba(74,158,255,.3) !important;
}

/* ── Slider / select ── */
[data-testid="stSlider"] > div > div > div > div { background: #4a9eff !important; }
[data-baseweb="select"] > div { background: #0f1928 !important; border-color: #1a2540 !important; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] iframe { background: transparent !important; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div { background: #1a2540 !important; border-radius: 4px; }
[data-testid="stProgressBar"] > div > div { border-radius: 4px !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #080c14; }
::-webkit-scrollbar-thumb { background: #1a2540; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Palette ──────────────────────────────────────────────────────────────────
BG      = "#080c14"
CARD    = "#0f1928"
BORDER  = "#1a2540"
ACCENT  = "#4a9eff"
GREEN   = "#06d6a0"
RED     = "#ff6b6b"
YELLOW  = "#ffd166"
TEXT    = "#c9d1e0"
MUTED   = "#4a6080"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   MUTED,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "text.color":        TEXT,
    "grid.color":        BORDER,
    "grid.linewidth":    0.5,
    "font.family":       "sans-serif",
    "font.size":         9,
})

# ── Data & pipeline ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_data(n=3630, seed=42):
    rng = np.random.default_rng(seed)
    n_drop = int(n * 0.391)
    n_grad = n - n_drop
    def cohort(size, grade_mu, tu_prob, age_mu, units_mu):
        return dict(
            Curricular_units_1st_sem_grade    = rng.normal(grade_mu,    3.5, size).clip(0,20),
            Curricular_units_2nd_sem_grade    = rng.normal(grade_mu-.3, 3.5, size).clip(0,20),
            Curricular_units_1st_sem_approved = rng.normal(units_mu,   2.5, size).clip(0,10).round(),
            Curricular_units_2nd_sem_approved = rng.normal(units_mu-.5,2.5, size).clip(0,10).round(),
            Tuition_fees_up_to_date           = rng.binomial(1,tu_prob,size),
            Scholarship_holder                = rng.binomial(1,0.23,size),
            Age_at_enrollment                 = rng.normal(age_mu,6,size).clip(17,60).round(),
            Daytime_evening_attendance        = rng.binomial(1,0.78,size),
            Debtor                            = rng.binomial(1,0.12 if tu_prob<0.5 else 0.04,size),
            GDP                               = rng.normal(1.4,1.2,size),
            Unemployment_rate                 = rng.normal(11.5,2.5,size),
            Inflation_rate                    = rng.normal(1.2,1.0,size),
            Mothers_qualification             = rng.integers(1,20,size),
            Fathers_qualification             = rng.integers(1,20,size),
            Marital_status                    = rng.integers(1,7,size),
            Application_mode                  = rng.integers(1,18,size),
            Course                            = rng.integers(1,17,size),
            Previous_qualification_grade      = rng.normal(130,20,size).clip(95,190),
            Admission_grade                   = rng.normal(127,20,size).clip(95,190),
            Displaced                         = rng.binomial(1,0.43,size),
            Gender                            = rng.binomial(1,0.35,size),
            International                     = rng.binomial(1,0.05,size),
        )
    frames = []
    for d, lbl in [(cohort(n_drop,7.1,0.35,24,3.5),1),(cohort(n_grad,12.5,0.91,20,6.8),0)]:
        df_c = pd.DataFrame(d); df_c["Target"] = lbl; frames.append(df_c)
    return pd.concat(frames,ignore_index=True).sample(frac=1,random_state=seed)

@st.cache_resource(show_spinner=False)
def train_pipeline():
    df = generate_data()
    X = df.drop("Target",axis=1); y = df["Target"]
    feat = X.columns.tolist()
    Xtrf,Xte,ytrf,yte = train_test_split(X,y,test_size=.15,stratify=y,random_state=42)
    Xtr,Xv,ytr,yv    = train_test_split(Xtrf,ytrf,test_size=.15/.85,stratify=ytrf,random_state=42)
    sc = StandardScaler()
    Xtr_s=sc.fit_transform(Xtr); Xv_s=sc.transform(Xv); Xte_s=sc.transform(Xte)
    Xb,yb = SMOTE(random_state=42,k_neighbors=5).fit_resample(Xtr_s,ytr)
    clfs = {
        "Logistic Regression": LogisticRegression(C=1,solver="lbfgs",max_iter=500,random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=10,min_samples_split=5,random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=100,max_depth=6,learning_rate=.1,
                                             eval_metric="logloss",random_state=42,verbosity=0),
    }
    best,results = {},[]
    for name,clf in clfs.items():
        clf.fit(Xb,yb); yp=clf.predict(Xte_s); ypr=clf.predict_proba(Xte_s)[:,1]
        results.append({"Model":name,
                        "Accuracy": round(accuracy_score(yte,yp)*100,1),
                        "Precision":round(precision_score(yte,yp),3),
                        "Recall":   round(recall_score(yte,yp),3),
                        "F1":       round(f1_score(yte,yp),3),
                        "AUC-ROC":  round(roc_auc_score(yte,ypr),3)})
        best[name]=clf
    rdf = pd.DataFrame(results).sort_values("F1",ascending=False).reset_index(drop=True)
    xgb = best["XGBoost"]
    Xtdf = pd.DataFrame(Xte_s,columns=feat)
    expl = shap.TreeExplainer(xgb)
    sv   = expl.shap_values(Xtdf)
    return dict(df=df,scaler=sc,models=best,results_df=rdf,
                X_test_sc=Xte_s,y_test=yte.values,
                shap_vals=sv,X_test_df=Xtdf,feature_names=feat)

# ── Load ─────────────────────────────────────────────────────────────────────
with st.spinner("Initialising pipeline…"):
    ctx = train_pipeline()

df    = ctx["df"]; sc = ctx["scaler"]; models = ctx["models"]
rdf   = ctx["results_df"]; Xte_s = ctx["X_test_sc"]; yte = ctx["y_test"]
sv    = ctx["shap_vals"];  Xtdf  = ctx["X_test_df"];  feat = ctx["feature_names"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="nav-logo">
      <div class="nav-logo-icon">🎓</div>
      <div class="nav-logo-title">Dropout Risk</div>
      <div class="nav-logo-sub">AMSI32_MIP · Group 05 · ITC</div>
    </div>
    <div class="nav-group-label">Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "📊  Overview",
        "🔍  Risk Calculator",
        "🤖  Model Comparison",
        "🔬  SHAP Analysis",
    ])

    st.markdown("---")

    n_total   = len(df)
    n_dropout = int(df["Target"].sum())
    st.markdown(f"""
    <div style="font-size:.68rem;color:{MUTED};letter-spacing:.08em;text-transform:uppercase;
                font-weight:700;margin-bottom:.5rem">Dataset</div>
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:.8rem 1rem;font-size:.78rem">
        <div style="display:flex;justify-content:space-between;margin-bottom:.3rem">
            <span style="color:{MUTED}">Students</span>
            <span style="color:{TEXT};font-weight:600;font-family:'JetBrains Mono',monospace">{n_total:,}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:.3rem">
            <span style="color:{MUTED}">Dropouts</span>
            <span style="color:{RED};font-weight:600;font-family:'JetBrains Mono',monospace">{n_dropout:,}</span>
        </div>
        <div style="display:flex;justify-content:space-between">
            <span style="color:{MUTED}">Graduates</span>
            <span style="color:{GREEN};font-weight:600;font-family:'JetBrains Mono',monospace">{n_total-n_dropout:,}</span>
        </div>
    </div>
    <div style="margin-top:.5rem;font-size:.68rem;color:{MUTED}">
        UCI · Realinho et al., 2021<br>
        SDG 4: Quality Education
    </div>
    """, unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 1 — OVERVIEW                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if page == "📊  Overview":
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">Machine Learning · SDG 4 · Quality Education</div>
      <div class="hero-title">Student <span>Dropout</span> Risk Dashboard</div>
      <div class="hero-sub">Predicting academic attrition using 4 ML models trained on 3,630 students</div>
      <div class="hero-badge">🏫 Institute of Technology of Cambodia &nbsp;·&nbsp; Group 05</div>
    </div>
    """, unsafe_allow_html=True)

    best_auc = rdf["AUC-ROC"].max(); best_f1 = rdf["F1"].max()
    dr = n_dropout/n_total*100

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card" style="--accent:{ACCENT}">
        <div class="kpi-icon">👥</div>
        <div class="kpi-label">Total Students</div>
        <div class="kpi-value">{n_total:,}</div>
        <div class="kpi-delta">UCI dataset, 35 features</div>
      </div>
      <div class="kpi-card" style="--accent:{RED}">
        <div class="kpi-icon">⚠️</div>
        <div class="kpi-label">Dropout Rate</div>
        <div class="kpi-value" style="color:{RED}">{dr:.1f}%</div>
        <div class="kpi-delta">{n_dropout:,} at-risk students</div>
      </div>
      <div class="kpi-card" style="--accent:{GREEN}">
        <div class="kpi-icon">📈</div>
        <div class="kpi-label">Best AUC-ROC</div>
        <div class="kpi-value" style="color:{GREEN}">{best_auc:.3f}</div>
        <div class="kpi-delta">XGBoost classifier</div>
      </div>
      <div class="kpi-card" style="--accent:{YELLOW}">
        <div class="kpi-icon">🎯</div>
        <div class="kpi-label">Best F1-Score</div>
        <div class="kpi-value" style="color:{YELLOW}">{best_f1:.3f}</div>
        <div class="kpi-delta">Optimised via GridSearchCV</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    def sec(label):
        st.markdown(f"""
        <div class="sec-header">
          <span class="sec-label">{label}</span>
          <div class="sec-header-line"></div>
        </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        sec("Grade Distribution by Outcome")
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        for lbl, color, name in [(0,GREEN,"Graduate"),(1,RED,"Dropout")]:
            ax.hist(df[df["Target"]==lbl]["Curricular_units_1st_sem_grade"],
                    bins=28, alpha=0.75, color=color, label=name, edgecolor="none")
        ax.set_xlabel("Semester 1 Grade (0–20)"); ax.set_ylabel("Students")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
        ax.grid(axis="y", alpha=.4)
        fig.tight_layout(pad=.5)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with col2:
        sec("Class Balance — Graduate vs Dropout")
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        sizes  = [n_total-n_dropout, n_dropout]
        colors = [GREEN, RED]
        wedges,_,autotexts = ax.pie(
            sizes, labels=["Graduate","Dropout"], colors=colors,
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(width=0.58, edgecolor=BG, linewidth=3),
            textprops=dict(color=TEXT, fontsize=10),
        )
        for at in autotexts: at.set_color(BG); at.set_fontweight("bold"); at.set_fontsize(9)
        ax.set_facecolor(BG)
        fig.tight_layout(pad=.5)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    sec("Tuition Fee Status vs Dropout Rate")
    ct = pd.crosstab(df["Tuition_fees_up_to_date"], df["Target"], normalize="index")*100
    ct.index = ["Fees NOT up to date","Fees up to date"]
    ct.columns = ["Graduate","Dropout"]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ct[["Graduate","Dropout"]].plot(kind="barh", stacked=True, ax=ax,
                                    color=[GREEN, RED], edgecolor="none")
    ax.set_xlabel("% of students"); ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False); ax.spines["left"].set_visible(False)
    ax.grid(axis="x", alpha=.3)
    for i,(_, row) in enumerate(ct.iterrows()):
        ax.text(row["Dropout"]/2, i, f'{row["Dropout"]:.1f}%',
                va="center", ha="center", color=BG, fontweight="bold", fontsize=9)
    ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
    fig.tight_layout(pad=.5)
    st.pyplot(fig, use_container_width=True); plt.close(fig)

    col3, col4 = st.columns(2)
    with col3:
        sec("Age at Enrollment Distribution")
        fig, ax = plt.subplots(figsize=(5.5, 3))
        for lbl, color, name in [(0,GREEN,"Graduate"),(1,RED,"Dropout")]:
            d = df[df["Target"]==lbl]["Age_at_enrollment"]
            ax.hist(d, bins=22, alpha=.7, color=color, label=name, edgecolor="none")
        ax.set_xlabel("Age at Enrollment"); ax.set_ylabel("Count")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
        ax.grid(axis="y", alpha=.4); fig.tight_layout(pad=.5)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with col4:
        sec("Scholarship & Dropout Rate")
        s_rates = df.groupby("Scholarship_holder")["Target"].mean()*100
        fig, ax = plt.subplots(figsize=(5.5, 3))
        bars = ax.bar(["No Scholarship","Scholarship Holder"],
                      s_rates.values, color=[RED, GREEN],
                      width=.5, edgecolor="none", alpha=.85)
        for bar, val in zip(bars, s_rates.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.5,
                    f"{val:.1f}%", ha="center", va="bottom", color=TEXT, fontsize=10, fontweight="600")
        ax.set_ylabel("Dropout Rate (%)"); ax.set_ylim(0, s_rates.max()*1.2)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=.4); fig.tight_layout(pad=.5)
        st.pyplot(fig, use_container_width=True); plt.close(fig)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 2 — RISK CALCULATOR                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
elif page == "🔍  Risk Calculator":
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">Live Prediction Engine</div>
      <div class="hero-title">Individual <span>Risk</span> Calculator</div>
      <div class="hero-sub">Enter a student profile to receive a real-time dropout probability score</div>
    </div>
    """, unsafe_allow_html=True)

    def sec(label):
        st.markdown(f"""<div class="sec-header">
          <span class="sec-label">{label}</span>
          <div class="sec-header-line"></div></div>""", unsafe_allow_html=True)

    top_bar = st.columns([2,1])
    with top_bar[1]:
        model_choice = st.selectbox("Model", list(models.keys()), index=3)

    col1, col2, col3 = st.columns(3)

    with col1:
        sec("📚 Academic Performance")
        grade_s1  = st.slider("Semester 1 Grade",      0.0,20.0,11.0,.1)
        grade_s2  = st.slider("Semester 2 Grade",      0.0,20.0,10.5,.1)
        units_s1  = st.slider("Units Approved S1",     0,10,5)
        units_s2  = st.slider("Units Approved S2",     0,10,5)
        prev_gr   = st.slider("Prev. Qualification",   95.0,190.0,128.0,.5)
        adm_gr    = st.slider("Admission Grade",       95.0,190.0,127.0,.5)

    with col2:
        sec("💰 Financial & Social")
        tuition   = st.selectbox("Tuition Up To Date", [1,0], format_func=lambda x:"Yes" if x else "No")
        scholar   = st.selectbox("Scholarship Holder", [0,1], format_func=lambda x:"Yes" if x else "No")
        debtor    = st.selectbox("Debtor",             [0,1], format_func=lambda x:"Yes" if x else "No")
        displaced = st.selectbox("Displaced",          [0,1], format_func=lambda x:"Yes" if x else "No")
        gender    = st.selectbox("Gender",             [1,0], format_func=lambda x:"Male" if x else "Female")
        intl      = st.selectbox("International",      [0,1], format_func=lambda x:"Yes" if x else "No")

    with col3:
        sec("🧮 Demographics & Context")
        age       = st.slider("Age at Enrollment",     17,60,20)
        daytime   = st.selectbox("Attendance",         [1,0], format_func=lambda x:"Daytime" if x else "Evening")
        marital   = st.selectbox("Marital Status",     list(range(1,7)), index=0)
        app_mode  = st.selectbox("Application Mode",   list(range(1,18)), index=0)
        course    = st.selectbox("Course",             list(range(1,17)), index=0)
        unemp     = st.slider("Unemployment Rate (%)", 7.0,17.0,11.5,.1)

    sec("📊 Prediction Result")

    inp = {
        "Curricular_units_1st_sem_grade":    grade_s1,
        "Curricular_units_2nd_sem_grade":    grade_s2,
        "Curricular_units_1st_sem_approved": units_s1,
        "Curricular_units_2nd_sem_approved": units_s2,
        "Tuition_fees_up_to_date":           tuition,
        "Scholarship_holder":                scholar,
        "Age_at_enrollment":                 age,
        "Daytime_evening_attendance":        daytime,
        "Debtor":                            debtor,
        "GDP":                               1.4,
        "Unemployment_rate":                 unemp,
        "Inflation_rate":                    1.2,
        "Mothers_qualification":             10,
        "Fathers_qualification":             10,
        "Marital_status":                    marital,
        "Application_mode":                  app_mode,
        "Course":                            course,
        "Previous_qualification_grade":      prev_gr,
        "Admission_grade":                   adm_gr,
        "Displaced":                         displaced,
        "Gender":                            gender,
        "International":                     intl,
    }
    Xinp   = pd.DataFrame([inp])[feat]
    Xsc    = sc.transform(Xinp)
    clf    = models[model_choice]
    prob   = clf.predict_proba(Xsc)[0,1]
    pct    = prob*100

    if prob >= .65:   risk_cls,risk_lbl,risk_color = "risk-high",   "HIGH RISK",     RED
    elif prob >= .40: risk_cls,risk_lbl,risk_color = "risk-medium",  "MODERATE RISK", YELLOW
    else:             risk_cls,risk_lbl,risk_color = "risk-low",     "LOW RISK",      GREEN

    rc, sc2 = st.columns([1,1])

    with rc:
        # Semicircle gauge using matplotlib
        fig, ax = plt.subplots(figsize=(4.5, 2.6), subplot_kw=dict(aspect="equal"))
        fig.patch.set_facecolor(CARD)
        ax.set_facecolor(CARD)
        theta1, theta2 = 180, 0
        from matplotlib.patches import Wedge, FancyArrowPatch
        # Background arc
        bg = Wedge((0,0), 1, 0, 180, width=0.28, facecolor=BORDER, edgecolor="none")
        ax.add_patch(bg)
        # Filled arc
        fill_angle = 180 * prob
        fill_col = RED if prob>=.65 else (YELLOW if prob>=.4 else GREEN)
        fg = Wedge((0,0), 1, 180-fill_angle, 180, width=0.28, facecolor=fill_col, edgecolor="none", alpha=.9)
        ax.add_patch(fg)
        # Center text
        ax.text(0, -0.08, f"{pct:.1f}%", ha="center", va="center",
                fontsize=22, fontweight="bold", color=fill_col,
                fontfamily="monospace")
        ax.text(0, -0.35, risk_lbl, ha="center", va="center",
                fontsize=9, fontweight="700", color=fill_col, alpha=.9)
        ax.text(0, -0.55, f"Using {model_choice}", ha="center", va="center",
                fontsize=7, color=MUTED)
        # Zone labels
        ax.text(-1.1, -0.05, "0%",   ha="center", color=MUTED, fontsize=7)
        ax.text( 1.1, -0.05, "100%", ha="center", color=MUTED, fontsize=7)
        ax.set_xlim(-1.35, 1.35); ax.set_ylim(-0.75, 1.15)
        ax.axis("off")
        fig.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with sc2:
        # Risk signals
        signals,ok = [],[]
        if grade_s1 < 8:   signals.append(("⚠️","Low S1 grade","critical academic predictor"))
        if units_s1 < 4:   signals.append(("⚠️","Few units approved","low engagement signal"))
        if not tuition:    signals.append(("💰","Fees not up to date","strongest dropout predictor"))
        if debtor:         signals.append(("💸","Debtor","financial stress indicator"))
        if age > 25:       signals.append(("📅","Mature student","higher-risk age cohort"))
        if not scholar:    signals.append(("🎓","No scholarship","reduced support structure"))
        if grade_s1 >= 12: ok.append(("✅","Strong S1 grade","protective factor"))
        if tuition:        ok.append(("✅","Fees up to date","protective factor"))
        if scholar:        ok.append(("✅","Scholarship holder","protective factor"))

        st.markdown(f"**Risk signals detected: {len(signals)}**", unsafe_allow_html=False)
        for icon,label,detail in signals[:4]:
            st.markdown(f"""
            <div class="signal-card">
              <span class="signal-icon">{icon}</span>
              <div class="signal-text">{label}
                <span class="signal-detail">{detail}</span>
              </div>
            </div>""", unsafe_allow_html=True)
        for icon,label,detail in ok[:3]:
            st.markdown(f"""
            <div class="signal-card signal-ok">
              <span class="signal-icon">{icon}</span>
              <div class="signal-text" style="color:#06d6a0">{label}
                <span class="signal-detail">{detail}</span>
              </div>
            </div>""", unsafe_allow_html=True)
        if not signals and not ok:
            st.markdown('<div class="signal-card signal-ok"><span class="signal-icon">✅</span><div class="signal-text" style="color:#06d6a0">No major risk flags detected</div></div>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 3 — MODEL COMPARISON                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
elif page == "🤖  Model Comparison":
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">Evaluation · Test Set · 15% Hold-out</div>
      <div class="hero-title">Model <span>Comparison</span></div>
      <div class="hero-sub">4 classifiers · 5-fold stratified GridSearchCV · optimised for F1-Score</div>
    </div>
    """, unsafe_allow_html=True)

    def sec(label):
        st.markdown(f"""<div class="sec-header">
          <span class="sec-label">{label}</span>
          <div class="sec-header-line"></div></div>""", unsafe_allow_html=True)

    sec("Performance Leaderboard")
    medals = ["🥇","🥈","🥉","4️⃣"]
    st.markdown("""<div class="model-header">
      <div>Model</div><div>Accuracy</div><div>Precision</div>
      <div>Recall</div><div>F1-Score</div><div>AUC-ROC</div>
    </div>""", unsafe_allow_html=True)
    best_f1 = rdf["F1"].max(); best_auc = rdf["AUC-ROC"].max()
    for i, row in rdf.iterrows():
        winner = "winner" if i==0 else ""
        hi_f1  = f"model-stat-hi" if row["F1"]==best_f1 else "model-stat"
        hi_auc = f"model-stat-hi" if row["AUC-ROC"]==best_auc else "model-stat"
        badge  = '<span class="model-badge">BEST</span>' if i==0 else ""
        st.markdown(f"""
        <div class="model-row {winner}">
          <div class="model-name">{medals[i]} {row['Model']}{badge}</div>
          <div class="model-stat">{row['Accuracy']:.1f}%</div>
          <div class="model-stat">{row['Precision']:.3f}</div>
          <div class="model-stat">{row['Recall']:.3f}</div>
          <div class="{hi_f1}">{row['F1']:.3f}</div>
          <div class="{hi_auc}">{row['AUC-ROC']:.3f}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        sec("F1 & AUC-ROC by Model")
        fig, ax = plt.subplots(figsize=(5.8, 3.5))
        x = np.arange(len(rdf)); w = .32
        b1 = ax.bar(x-w/2, rdf["F1"],      w, color=ACCENT, alpha=.85, edgecolor="none", label="F1")
        b2 = ax.bar(x+w/2, rdf["AUC-ROC"], w, color=GREEN,  alpha=.85, edgecolor="none", label="AUC-ROC")
        for bar in list(b1)+list(b2):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.003,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=7, color=MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" ","\n") for m in rdf["Model"]], fontsize=8)
        ax.set_ylim(.6, 1.0); ax.grid(axis="y", alpha=.4)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8)
        fig.tight_layout(pad=.5)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    with col_r:
        sec("ROC Curves — All Models")
        fig, ax = plt.subplots(figsize=(5.8, 3.5))
        palette = [ACCENT, GREEN, YELLOW, RED]
        for (_, row), color in zip(rdf.iterrows(), palette):
            clf = models[row["Model"]]
            yp  = clf.predict_proba(Xte_s)[:,1]
            fpr,tpr,_ = roc_curve(yte,yp)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{row['Model'].split()[0]} {row['AUC-ROC']:.3f}")
        ax.plot([0,1],[0,1],":",color=MUTED,lw=1)
        ax.fill_between([0,1],[0,1], alpha=.05, color=MUTED)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.grid(alpha=.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(facecolor=CARD, edgecolor=BORDER, labelcolor=TEXT, fontsize=8, loc="lower right")
        fig.tight_layout(pad=.5)
        st.pyplot(fig, use_container_width=True); plt.close(fig)

    sec("Confusion Matrix — XGBoost (Best Model)")
    xgb    = models["XGBoost"]
    y_pred = xgb.predict(Xte_s)
    cm     = confusion_matrix(yte, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Graduate","Dropout"]); ax.set_yticklabels(["Graduate","Dropout"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    thresh = cm.max()/2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j]>thresh else TEXT, fontsize=16, fontweight="bold")
    fig.tight_layout(pad=.5)
    _, cc, _ = st.columns([1,2,1])
    with cc: st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PAGE 4 — SHAP                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
elif page == "🔬  SHAP Analysis":
    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">Model Explainability · Section 5.3 of Report</div>
      <div class="hero-title">SHAP <span>Feature Importance</span></div>
      <div class="hero-sub">SHapley Additive exPlanations — XGBoost · why the model makes each prediction</div>
    </div>
    """, unsafe_allow_html=True)

    def sec(label):
        st.markdown(f"""<div class="sec-header">
          <span class="sec-label">{label}</span>
          <div class="sec-header-line"></div></div>""", unsafe_allow_html=True)

    mean_abs = np.abs(sv).mean(axis=0)
    imp_df = pd.DataFrame({"Feature":feat,"Mean |SHAP|":mean_abs}) \
               .sort_values("Mean |SHAP|",ascending=False).head(15).reset_index(drop=True)

    col_l, col_r = st.columns([1.4, 1])

    with col_l:
        sec("Top 15 Features — Custom Ranked Bar")
        mx = imp_df["Mean |SHAP|"].max()
        st.markdown('<div style="background:%s;border:1px solid %s;border-radius:12px;padding:1rem 1.2rem">' % (CARD, BORDER), unsafe_allow_html=True)
        for _, row in imp_df.iterrows():
            pct = row["Mean |SHAP|"]/mx*100
            rank_color = ACCENT if pct>60 else (GREEN if pct>30 else MUTED)
            st.markdown(f"""
            <div class="feat-row">
              <div class="feat-name" title="{row['Feature']}">{row['Feature'].replace('_',' ')}</div>
              <div class="feat-bar-wrap">
                <div class="feat-bar" style="width:{pct:.1f}%;background:linear-gradient(90deg,{rank_color}88,{rank_color})"></div>
              </div>
              <div class="feat-val">{row['Mean |SHAP|']:.4f}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        sec("Importance Summary")
        for i, (_, row) in enumerate(imp_df.head(5).iterrows()):
            pct = row["Mean |SHAP|"]/mx*100
            medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][i]
            st.markdown(f"""
            <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;
                        padding:.8rem 1rem;margin-bottom:.5rem;">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                  <span style="font-size:.9rem">{medal}</span>
                  <span style="font-size:.8rem;font-weight:600;color:{TEXT};margin-left:6px">
                    {row['Feature'].replace('_',' ')}</span>
                </div>
                <span style="font-family:'JetBrains Mono',monospace;font-size:.78rem;color:{ACCENT}">
                  {row['Mean |SHAP|']:.4f}</span>
              </div>
              <div style="margin-top:.5rem;background:{BORDER};border-radius:3px;height:4px">
                <div style="width:{pct:.0f}%;height:4px;border-radius:3px;
                            background:linear-gradient(90deg,{ACCENT}88,{ACCENT})"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:rgba(74,158,255,.06);border:1px solid rgba(74,158,255,.2);
                    border-radius:10px;padding:1rem;margin-top:.8rem;font-size:.78rem;color:{MUTED}">
          <strong style="color:{ACCENT}">How to read SHAP values</strong><br><br>
          A higher mean |SHAP| means the feature has a larger average effect on the model's prediction.
          First-semester academic performance dominates — confirming that early intervention is most effective.
        </div>""", unsafe_allow_html=True)

    sec("Beeswarm Plot — Directional Feature Impact")
    top_idx  = [feat.index(f) for f in imp_df["Feature"]]
    top_shap = sv[:, top_idx]
    top_X    = Xtdf[imp_df["Feature"]]
    fig, ax  = plt.subplots(figsize=(10, 5.5))
    shap.summary_plot(top_shap, top_X, feature_names=imp_df["Feature"].tolist(),
                      show=False, plot_size=None, color_bar=True)
    ax = plt.gca()
    ax.set_facecolor(CARD); plt.gcf().set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(BORDER); ax.spines["left"].set_color(BORDER)
    ax.set_xlabel("SHAP value  →  impact on dropout prediction", color=MUTED, fontsize=9)
    plt.tight_layout(pad=.5)
    st.pyplot(plt.gcf(), use_container_width=True); plt.close("all")

    st.markdown(f"""
    <div style="background:{CARD};border:1px solid {BORDER};border-radius:10px;
                padding:1rem 1.4rem;margin-top:.5rem;font-size:.8rem">
      <span style="color:{ACCENT};font-weight:700">Reading guide: </span>
      <span style="color:{MUTED}">Each dot = one student. &nbsp;
        <span style="color:#3050cc">■ Blue</span> = low feature value &nbsp;·&nbsp;
        <span style="color:#cc3030">■ Red</span> = high feature value. &nbsp;
        Dots right of zero push toward <strong style="color:{RED}">Dropout</strong>;
        dots left push toward <strong style="color:{GREEN}">Graduate</strong>.
        High S1 grades (red, far left) are the strongest protective factor.</span>
    </div>""", unsafe_allow_html=True)
