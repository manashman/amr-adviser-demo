# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
shap.initjs()

st.set_page_config(page_title="AMR Empiric Adviser (Demo)", layout="centered")
st.title("AMR Empiric Antibiotic Adviser — Demo (ciprofloxacin)")

MODEL_FILE = "amr_model_ciprofloxacin.pkl"
DEFAULT_CSV = "GLASS_style_AMR_3yr_medium.csv"

@st.cache_data
def load_csv_from_repo(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df

def preprocess_and_train(df, target_ab='ciprofloxacin'):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['organism'] = df['organism'].astype(str).str.strip().str.lower()
    df['specimen'] = df['specimen'].astype(str).str.strip().str.lower()
    df['antibiotic'] = df['antibiotic'].astype(str).str.strip().str.lower()
    df['interpretation'] = df['interpretation'].astype(str).str.strip().str.upper()
    df['resistant'] = df['interpretation'].map({'R':1,'S':0,'I':0}).fillna(0).astype(int)

    df_target = df[df['antibiotic']==target_ab].copy()
    if df_target.empty:
        return None, None

    df_target['org_spec'] = df_target['organism'] + '___' + df_target['specimen']
    df_target = df_target.sort_values(['org_spec','date']).reset_index(drop=True)
    df_target['recent_res_rate_prior'] = (df_target.groupby('org_spec')['resistant']
                                         .apply(lambda s: s.shift(1).rolling(window=3, min_periods=1).mean())
                                         .reset_index(level=0, drop=True))
    df_target['recent_res_rate_prior'] = df_target['recent_res_rate_prior'].fillna(df_target.groupby('org_spec')['resistant'].transform('mean'))
    df_target['month'] = df_target['date'].dt.to_period('M')
    months_sorted = sorted(df_target['month'].unique())
    month_map = {m:i for i,m in enumerate(months_sorted)}
    df_target['month_idx'] = df_target['month'].map(month_map).astype(int)
    df_target['month_of_year'] = df_target['date'].dt.month
    df_target['org_ab'] = df_target['organism'] + '_' + df_target['antibiotic']
    df_target['spec_ab'] = df_target['specimen'] + '_' + df_target['antibiotic']

    # age mapping if present
    if 'age_group' in df_target.columns:
        mapping = {'0-18':0,'19-40':1,'41-60':2,'>60':3}
        df_target['age_group_clean'] = df_target['age_group'].astype(str).str.strip().map(mapping).fillna(1).astype(int)
        df_target.drop(columns=['age_group'], inplace=True, errors=True)

    for col in [c for c in df_target.columns if str(c).lower().startswith('interpretation')]:
        df_target.drop(columns=[col], inplace=True, errors=True)

    obj_cols = [c for c in df_target.select_dtypes(include=['object']).columns if c!='date']
    if obj_cols:
        df_enc = pd.get_dummies(df_target, columns=obj_cols, drop_first=True)
    else:
        df_enc = df_target.copy()

    exclude = ['antibiotic','interpretation','resistant','date','org_spec','month']
    features = [c for c in df_enc.columns if c not in exclude]
    X = df_enc[features].fillna(0)
    y = df_enc['resistant'].astype(int)

    # temporal split (train first 80% months)
    split_idx = int(0.8*(df_enc['month_idx'].max()+1))
    train_idx = df_enc['month_idx'] < split_idx
    X_train, X_test = X.loc[train_idx], X.loc[~train_idx]
    y_train, y_test = y.loc[train_idx], y.loc[~train_idx]

    pos = y_train.sum(); neg = len(y_train)-pos
    scale_pos_weight = neg/pos if pos>0 else 1.0

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              scale_pos_weight=scale_pos_weight, random_state=42,
                              n_estimators=150, max_depth=5, learning_rate=0.1)
    model.fit(X_train, y_train)

    return model, (df_enc, features, X_test, y_test)

st.markdown("**Step 1:** Upload a CSV (optional). If you don't upload, the repo CSV (if present) will be used.")
uploaded = st.file_uploader("Upload antibiogram CSV (long format with columns: date, organism, specimen, antibiotic, interpretation)", type=['csv'])

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("File uploaded. Using uploaded CSV.")
else:
    df = load_csv_from_repo(DEFAULT_CSV)
    if df is None:
        st.warning("No CSV found in repo and no file uploaded. Please upload a CSV or add the CSV to the repo.")
        st.stop()
    else:
        st.info(f"Using CSV from repo: {DEFAULT_CSV}")

# Try load model file if exists
if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        st.success("Loaded saved model from repo.")
        # need df_enc/features to build UI; train quickly to get schema (cheap)
        model_tmp, meta = preprocess_and_train(df)
        if meta is None:
            st.error("CSV didn't contain target antibiotic; re-upload or change CSV.")
            st.stop()
        df_enc, features, X_test, y_test = meta
    except Exception:
        st.warning("Saved model could not be loaded; training from CSV.")
        model, meta = preprocess_and_train(df)
        if model is None:
            st.error("Could not train model from CSV.")
            st.stop()
        df_enc, features, X_test, y_test = meta
        joblib.dump(model, MODEL_FILE)
else:
    model, meta = preprocess_and_train(df)
    if model is None:
        st.error("Could not train model from CSV (target antibiotic missing).")
        st.stop()
    df_enc, features, X_test, y_test = meta
    joblib.dump(model, MODEL_FILE)
    st.success("Trained model and saved to repo.")

st.sidebar.header("Input for prediction")
organism = st.sidebar.selectbox("Organism", sorted(set([o for o in df['organism'].astype(str).str.lower().unique()])))
specimen = st.sidebar.selectbox("Specimen", sorted(set([s for s in df['specimen'].astype(str).str.lower().unique()])))
inpatient = st.sidebar.selectbox("Inpatient?", ["No","Yes"])
age_group = st.sidebar.selectbox("Age group", sorted(set(df['age_group'].astype(str).unique())) if 'age_group' in df.columns else ["19-40","41-60",">60","0-18"])
month_idx = st.sidebar.slider("Month index (for trend)", min_value=int(df_enc['month_idx'].min()), max_value=int(df_enc['month_idx'].max()), value=int(df_enc['month_idx'].max()))

def build_input_row(organism, specimen, inpatient, age_group, month_idx, features):
    row = pd.Series(0, index=features)
    row['inpatient'] = 1 if inpatient=="Yes" else 0
    row['month_idx'] = int(month_idx)
    row['month_of_year'] = 1
    # estimate recent_res_rate_prior from dataset for same organism (simple heuristic)
    if 'interpretation' in df.columns:
        try:
            row['recent_res_rate_prior'] = df[df['organism'].astype(str).str.lower()==organism]['interpretation'].map({'R':1,'S':0,'I':0}).mean()
            if np.isnan(row['recent_res_rate_prior']):
                row['recent_res_rate_prior'] = 0.5
        except Exception:
            row['recent_res_rate_prior'] = 0.5
    else:
        row['recent_res_rate_prior'] = 0.5
    if 'age_group_clean' in features:
        mapping = {'0-18':0,'19-40':1,'41-60':2,'>60':3}
        row['age_group_clean'] = mapping.get(age_group,1)
    org_col = f"organism_{organism}"
    spec_col = f"specimen_{specimen}"
    if org_col in row.index:
        row[org_col] = 1
    if spec_col in row.index:
        row[spec_col] = 1
    org_ab = f"org_ab_{organism}_ciprofloxacin"
    spec_ab = f"spec_ab_{specimen}_ciprofloxacin"
    if org_ab in row.index:
        row[org_ab] = 1
    if spec_ab in row.index:
        row[spec_ab] = 1
    return row.values.reshape(1, -1), row

if st.button("Predict"):
    input_vec, input_row = build_input_row(organism, specimen, inpatient, age_group, month_idx, features)
    prob = model.predict_proba(input_vec)[0,1]
    st.metric("Probability of Resistance (ciprofloxacin)", f"{prob:.2f}")
    explainer = shap.TreeExplainer(model)
    try:
        sv = explainer.shap_values(pd.DataFrame(input_vec, columns=features))
    except Exception:
        sv = explainer.shap_values(pd.DataFrame(input_vec, columns=features).astype(float))
    shap_scores = pd.Series(sv[0], index=features).abs().sort_values(ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(6,3))
    shap_scores.plot.barh(ax=ax)
    ax.set_xlabel("Absolute SHAP value")
    ax.invert_yaxis()
    st.pyplot(fig)
    st.write("Top contributing features:")
    st.table(shap_scores.reset_index().rename(columns={'index':'feature',0:'abs_shap'}))
