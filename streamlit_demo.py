"""
Student Dropout Early Warning System
======================================
Interactive Streamlit dashboard — AMSI32_MIP Mini Project (Group 05)
Institute of Technology of Cambodia | Dept. of Applied Mathematics & Statistics
SDG 4: Quality Education (Targets 4.1, 4.3)

Run with:
    streamlit run streamlit_demo.py

Requirements:
    streamlit, pandas, numpy, scikit-learn, xgboost, shap, imbalanced-learn,
    matplotlib, (optional) ucimlrepo

Dataset:
    UCI 'Predict Students' Dropout and Academic Success' (Realinho et al., 2021)
    On first run the app tries, in order:
      1. A local file named data.csv / dataset.csv / student_dropout.csv
         (semicolon-, comma-, or tab-separated) in the working directory
      2. The `ucimlrepo` package (pip install ucimlrepo)
      3. A direct download from the UCI Machine Learning Repository
"""

import io
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

SEED = 42

st.set_page_config(
    page_title="Student Dropout Early Warning System",
    page_icon="🎓",
    layout="wide",
)


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def find_col(df, *keywords):
    """Return the first column whose name contains ALL keywords (case-insensitive)."""
    kws = [k.lower() for k in keywords]
    for col in df.columns:
        if all(k in col.lower() for k in kws):
            return col
    return None


def unwrap_shap(shap_vals):
    """Normalise SHAP output across shap/XGBoost versions to a 2D ndarray
    of SHAP values for the positive (Dropout) class."""
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    arr = np.asarray(shap_vals)
    if arr.ndim == 3:  # (n_samples, n_features, n_classes)
        arr = arr[:, :, -1]
    return arr


# ──────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading UCI Student Dropout dataset...")
def load_data():
    local_names = ["data.csv", "dataset.csv", "student_dropout.csv"]

    df_raw = None
    for name in local_names:
        p = Path(name)
        if p.exists():
            for sep in [";", ",", "\t"]:
                try:
                    tmp = pd.read_csv(p, sep=sep)
                    if tmp.shape[1] > 5:
                        df_raw = tmp
                        break
                except Exception:
                    continue
        if df_raw is not None:
            break

    if df_raw is None:
        try:
            from ucimlrepo import fetch_ucirepo
            repo = fetch_ucirepo(id=697)
            df_raw = pd.concat([repo.data.features, repo.data.targets], axis=1)
        except Exception:
            df_raw = None

    if df_raw is None:
        url = (
            "https://archive.ics.uci.edu/static/public/697/"
            "predict+students+dropout+and+academic+success.zip"
        )
        with urllib.request.urlopen(url, timeout=60) as resp:
            zf = zipfile.ZipFile(io.BytesIO(resp.read()))
            csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
            with zf.open(csv_name) as f:
                df_raw = pd.read_csv(f, sep=";")

    df_raw.columns = df_raw.columns.str.strip()

    # Locate / normalise the target column
    target_col = None
    for col in df_raw.columns:
        if col.strip().lower() == "target":
            target_col = col
            break
    if target_col is None:
        target_col = df_raw.columns[-1]
    if target_col != "Target":
        df_raw = df_raw.rename(columns={target_col: "Target"})

    df_raw["Target"] = df_raw["Target"].astype(str).str.strip()
    label_map = {}
    for v in df_raw["Target"].unique():
        vl = v.lower().replace("-", "").replace(" ", "")
        if vl == "dropout":
            label_map[v] = "Dropout"
        elif vl == "graduate":
            label_map[v] = "Graduate"
        elif vl == "enrolled":
            label_map[v] = "Enrolled"
    if label_map:
        df_raw["Target"] = df_raw["Target"].map(label_map).fillna(df_raw["Target"])

    # Binary subset: Dropout (1) vs Graduate (0) — as per report Sec. 4.2
    df = df_raw[df_raw["Target"].isin(["Dropout", "Graduate"])].copy()
    df["Target"] = df["Target"].map({"Dropout": 1, "Graduate": 0})

    return df


# ──────────────────────────────────────────────────────────────────────────
# Model training (cached for the app session)
# ──────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training XGBoost early-warning model...")
def train_model(df):
    feature_cols = [c for c in df.columns if c != "Target"]
    X, y = df[feature_cols], df["Target"]

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED
    )

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_raw)
    X_te_sc = scaler.transform(X_te_raw)

    smote = SMOTE(k_neighbors=5, random_state=SEED)
    X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_sc, y_tr)

    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_tr_sm, y_tr_sm)

    y_pred = model.predict(X_te_sc)
    y_prob = model.predict_proba(X_te_sc)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred),
        "Recall": recall_score(y_te, y_pred),
        "F1-Score": f1_score(y_te, y_pred),
        "AUC-ROC": roc_auc_score(y_te, y_prob),
    }

    explainer = shap.TreeExplainer(model)
    shap_values_test = unwrap_shap(explainer.shap_values(X_te_sc))

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "X_test_scaled": X_te_sc,
        "y_test": y_te.values,
        "explainer": explainer,
        "shap_values_test": shap_values_test,
        "medians": X[feature_cols].median(),
    }


# ──────────────────────────────────────────────────────────────────────────
# Load data + model
# ──────────────────────────────────────────────────────────────────────────

DATA_OK = True
DATA_ERROR = ""
try:
    df = load_data()
    artifacts = train_model(df)
except Exception as exc:  # pragma: no cover
    DATA_OK = False
    DATA_ERROR = str(exc)


# ──────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎓 Group 05")
    st.markdown(
        "**AMSI32_MIP — Mini Project**  \n"
        "Institute of Technology of Cambodia  \n"
        "Dept. of Applied Mathematics & Statistics"
    )
    st.markdown("---")
    st.markdown("**SDG Alignment**")
    st.markdown("SDG 4 — Quality Education (Targets 4.1 & 4.3)")
    st.markdown("---")

    if DATA_OK:
        st.success(f"Model ready · {len(df):,} students")
        m = artifacts["metrics"]
        st.caption(
            f"Live test AUC-ROC: **{m['AUC-ROC']:.3f}**  \n"
            f"Live test F1-Score: **{m['F1-Score']:.3f}**"
        )
    else:
        st.error("Dataset could not be loaded.")
        st.caption(DATA_ERROR)
        st.caption(
            "Place `data.csv` (the UCI dropout dataset, "
            "semicolon-separated) in the working directory and rerun."
        )


# ──────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────

st.title("🎓 Student Dropout Early Warning Dashboard")
st.markdown(
    "Predicting **dropout risk** from first-semester academic and financial "
    "indicators, with model explanations via **SHAP** — supporting **SDG 4: "
    "Quality Education**."
)

if not DATA_OK:
    st.warning(
        "Cannot continue without the dataset. See the sidebar for setup "
        "instructions."
    )
    st.stop()

# Resolve column names used in the interactive form
COLS = {
    "s1_approved": find_col(df, "1st sem", "approved"),
    "s1_grade":    find_col(df, "1st sem", "grade"),
    "s2_approved": find_col(df, "2nd sem", "approved"),
    "tuition":     find_col(df, "tuition"),
    "scholarship": find_col(df, "scholarship"),
    "age":         find_col(df, "age"),
    "debtor":      find_col(df, "debtor"),
    "course":      find_col(df, "course"),
    "attendance":  find_col(df, "attendance"),
    "gender":      find_col(df, "gender"),
}

missing = [k for k, v in COLS.items() if v is None]
if missing:
    st.error(f"Could not locate expected columns in the dataset for: {missing}")
    st.stop()

tab_predict, tab_perf, tab_shap, tab_about = st.tabs(
    ["🔮 Risk Prediction", "📊 Model Performance", "🔍 Feature Importance", "💡 Recommendations"]
)


# ──────────────────────────────────────────────────────────────────────────
# Tab 1 — Risk Prediction
# ──────────────────────────────────────────────────────────────────────────

with tab_predict:
    st.subheader("Enter Student Profile")
    st.caption(
        "Only the features most predictive of dropout risk (Table 6 of the "
        "report, ranked by SHAP importance) are shown here. All other "
        "features are held at the dataset median for a 'typical' student."
    )

    course_options = sorted(df[COLS["course"]].dropna().unique().tolist())

    with st.form("student_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Academic — Semester 1 & 2**")
            s1_units = st.slider("Curricular units approved (Semester 1)", 0, 26, 5)
            s1_grade = st.slider("Average grade — Semester 1 (0–20)", 0.0, 20.0, 11.0, 0.1)
            s2_units = st.slider("Curricular units approved (Semester 2)", 0, 26, 5)
            st.markdown("**Demographics**")
            age = st.slider("Age at enrollment", 17, 70, 19)
            course = st.selectbox("Course (code)", course_options)

        with col2:
            st.markdown("**Financial status**")
            tuition = st.radio("Tuition fees up to date?", ["Yes", "No"], horizontal=True)
            scholarship = st.radio("Scholarship holder?", ["No", "Yes"], horizontal=True)
            debtor = st.radio("Debtor (outstanding debt)?", ["No", "Yes"], horizontal=True)
            st.markdown("**Enrollment mode**")
            attendance = st.radio("Attendance mode", ["Daytime", "Evening"], horizontal=True)
            gender = st.radio("Gender", ["Female", "Male"], horizontal=True)

        submitted = st.form_submit_button("Predict Dropout Risk", type="primary")

    if submitted:
        x = artifacts["medians"].copy()
        x[COLS["s1_approved"]] = s1_units
        x[COLS["s1_grade"]] = s1_grade
        x[COLS["s2_approved"]] = s2_units
        x[COLS["age"]] = age
        x[COLS["course"]] = course
        x[COLS["tuition"]] = 1 if tuition == "Yes" else 0
        x[COLS["scholarship"]] = 1 if scholarship == "Yes" else 0
        x[COLS["debtor"]] = 1 if debtor == "Yes" else 0
        x[COLS["attendance"]] = 1 if attendance == "Daytime" else 0
        x[COLS["gender"]] = 1 if gender == "Male" else 0

        x_df = pd.DataFrame([x])[artifacts["feature_cols"]]
        x_scaled = artifacts["scaler"].transform(x_df)

        prob_dropout = float(artifacts["model"].predict_proba(x_scaled)[0, 1])

        st.markdown("---")
        res_col, shap_col = st.columns([1, 1.3])

        with res_col:
            if prob_dropout >= 0.5:
                st.error(f"### ⚠️ High Dropout Risk — {prob_dropout:.1%}")
            elif prob_dropout >= 0.3:
                st.warning(f"### 🟡 Moderate Dropout Risk — {prob_dropout:.1%}")
            else:
                st.success(f"### ✅ Low Dropout Risk — {prob_dropout:.1%}")

            st.progress(min(int(prob_dropout * 100), 100))

            st.markdown("**Suggested actions (Table 7, report):**")
            actions = []
            if s1_units < 5 or s1_grade < 11:
                actions.append(
                    "🔵 **Academic Early Alert** — refer to an academic tutor "
                    "for a one-on-one study plan review (Week 6–8)."
                )
            if tuition == "No" or debtor == "Yes":
                actions.append(
                    "🟢 **Financial Hardship Flag** — proactive outreach by "
                    "the student financial support office; scholarship referral."
                )
            if age >= 25 and attendance == "Evening":
                actions.append(
                    "🟣 **Mature Student Support** — flexible scheduling, "
                    "online resources, and peer mentoring."
                )
            if not actions:
                actions.append("✅ No specific risk flags triggered for this profile.")
            for a in actions:
                st.markdown(f"- {a}")

        with shap_col:
            shap_row = unwrap_shap(artifacts["explainer"].shap_values(x_scaled))[0]
            contrib = pd.Series(shap_row, index=artifacts["feature_cols"])
            top_contrib = contrib.abs().sort_values(ascending=False).head(8).index
            plot_df = contrib.loc[top_contrib].sort_values()

            fig, ax = plt.subplots(figsize=(6, 4.5))
            colors = ["#2ECC71" if v < 0 else "#E74C3C" for v in plot_df.values]
            ax.barh(plot_df.index, plot_df.values, color=colors)
            ax.axvline(0, color="grey", linewidth=0.8)
            ax.set_title("Top SHAP Contributions — This Student", fontsize=11, fontweight="bold")
            ax.set_xlabel("← pushes toward Graduate   |   pushes toward Dropout →")
            plt.tight_layout()
            st.pyplot(fig)


# ──────────────────────────────────────────────────────────────────────────
# Tab 2 — Model Performance
# ──────────────────────────────────────────────────────────────────────────

with tab_perf:
    st.subheader("Model Performance Comparison (Table 5, Report)")
    report_table = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost (Best)"],
        "Accuracy": ["76.3%", "78.9%", "82.6%", "84.6%"],
        "Precision": [0.721, 0.754, 0.811, 0.839],
        "Recall": [0.738, 0.761, 0.814, 0.856],
        "F1-Score": [0.729, 0.757, 0.812, 0.847],
        "AUC-ROC": [0.841, 0.803, 0.901, 0.921],
    })
    st.dataframe(report_table, hide_index=True, use_container_width=True)
    st.caption("Reference values from the project report (Section 5.2), test set, dropout class.")

    st.markdown("---")
    st.subheader("Live XGBoost Performance (this session)")
    m = artifacts["metrics"]
    cols = st.columns(5)
    for c, (label, val) in zip(cols, m.items()):
        c.metric(label, f"{val:.1%}" if label == "Accuracy" else f"{val:.3f}")

    st.caption(
        "Computed live on a fresh 85/15 stratified split + SMOTE + fixed "
        "hyperparameters (n_estimators=300, learning_rate=0.1, max_depth=6, "
        "subsample=0.8, colsample_bytree=0.8). Values may differ slightly "
        "from the report's GridSearchCV-tuned, three-way split results."
    )

    cm = confusion_matrix(
        artifacts["y_test"], artifacts["model"].predict(artifacts["X_test_scaled"])
    )
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center",
                color="white" if val > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Graduate", "Dropout"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Graduate", "Dropout"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — XGBoost (live test set)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)


# ──────────────────────────────────────────────────────────────────────────
# Tab 3 — Feature Importance (SHAP)
# ──────────────────────────────────────────────────────────────────────────

with tab_shap:
    st.subheader("Global Feature Importance (Live SHAP)")
    st.caption("Mean |SHAP value| computed on the live held-out test set.")

    mean_abs = np.abs(artifacts["shap_values_test"]).mean(axis=0)
    imp_df = (
        pd.DataFrame({"Feature": artifacts["feature_cols"], "Mean |SHAP|": mean_abs})
        .sort_values("Mean |SHAP|", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp_df["Feature"][::-1], imp_df["Mean |SHAP|"][::-1], color="#3498DB")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 10 Feature Importances — XGBoost (SHAP)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Reported Top 10 Features (Table 6, Report)")
    table6 = pd.DataFrame({
        "Rank": list(range(1, 11)),
        "Feature": [
            "Curricular units (S1) approved", "Tuition fees up to date",
            "Curricular units (S1) grade", "Scholarship holder",
            "Age at enrollment", "Curricular units (S2) approved",
            "Debtor", "Course (field of study)",
            "Daytime / Evening attendance", "Gender",
        ],
        "Mean |SHAP|": [0.412, 0.318, 0.287, 0.201, 0.178, 0.163, 0.142, 0.119, 0.098, 0.071],
        "Interpretation": [
            "Higher → Graduate. Failing S1 is the strongest warning sign.",
            "Yes → Graduate. Outstanding fees indicate high risk.",
            "Higher → Graduate. Grade independently predicts outcome.",
            "Yes → Graduate. Scholarship receipt is protective.",
            "Older → Dropout. Mature students face higher risk.",
            "Higher → Graduate. Second semester performance matters.",
            "Yes → Dropout. Outstanding debt is an early warning.",
            "Varies by course. Some courses show different patterns.",
            "Evening → Dropout. Evening students have higher risk.",
            "Male → higher risk. Modest effect after controlling others.",
        ],
    })
    st.dataframe(table6, hide_index=True, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────
# Tab 4 — Recommendations / About
# ──────────────────────────────────────────────────────────────────────────

with tab_about:
    st.subheader("Practical Recommendations for Intervention (Table 7, Report)")
    table7 = pd.DataFrame({
        "Recommendation": ["Academic Early Alert", "Financial Hardship Flag", "Mature Student Support"],
        "Target Group": [
            "Students with <5 S1 units approved OR S1 grade <11/20",
            "Students with outstanding tuition or debtor status at enrollment",
            "Students aged 25+ in evening programmes",
        ],
        "Trigger Signal": ["End of Week 6–8", "Enrollment record", "Enrollment record"],
        "Proposed Action": [
            "Automatic referral to academic tutor for a one-on-one study plan review",
            "Proactive outreach by student financial support office; scholarship referral",
            "Dedicated flexible scheduling, online resources, and peer mentoring",
        ],
    })
    st.dataframe(table7, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("SDG 4 Contribution")
    st.markdown(
        "This early-warning tool supports **SDG 4 — Quality Education**:\n\n"
        "- **Target 4.1** — ensure completion of quality education\n"
        "- **Target 4.3** — equal access to affordable tertiary education\n\n"
        "by enabling institutions to identify and support at-risk students "
        "within the first **4–5 months** of enrollment, using academic "
        "(Semester 1) and financial (enrollment-time) signals."
    )

    st.markdown("---")
    st.subheader("Limitations")
    st.markdown(
        "- Trained on data from a single Portuguese polytechnic (2008–2019); "
        "validation on local ITC student data is needed before deployment.\n"
        "- 'Enrolled' (still-studying) students were excluded from training.\n"
        "- No behavioural / LMS click-stream data is used.\n"
        "- A formal fairness audit (by gender) is recommended before deployment.\n"
        "- Post-pandemic dynamics may require periodic retraining."
    )

    st.markdown("---")
    st.caption(
        "Group 05 · Chheoun Kimchhun (e20230312) · DUY Nemol (e20230313) · "
        "HORN Samborokisin (e20230314)  \n"
        "AMSI32_MIP — Mini Project · Institute of Technology of Cambodia · "
        "Academic Year 2025–2026"
    )
