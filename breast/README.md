# Breast Cancer Diagnosis Predictor

Streamlit app that predicts Benign vs Malignant from tumor measurements,
using a trained `DecisionTreeClassifier`.

## Files
- `app.py` — the Streamlit app (modern, responsive UI with sidebar form + gauge chart)
- `requirements.txt` — pinned dependencies
- `.streamlit/config.toml` — theme colors
- `breast_cancer_model.pkl` — copy your model file here (loaded successfully, no corruption issues)

## About the features
The model expects 16 features in this exact order:
`id`, 8 raw tumor measurements (`radius_mean`, `texture_mean`, `perimeter_mean`,
`area_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`,
`concave points_mean`), and 7 engineered features (`shape_irregularity`,
`border_complexity`, `tumor_aggressiveness`, `radius_texture_interaction`,
`radius_concavity_interaction`, `concavity_density`, `malignancy_risk_score`).

Since the formulas for the engineered features weren't available, the app
currently takes them as **direct manual inputs** in the sidebar. If you later
get the exact formulas used during training (e.g.
`radius_texture_interaction = radius_mean * texture_mean`), share them and
the app can auto-calculate those fields from the raw measurements instead —
much faster to use.

## Deploy on GitHub + Streamlit Community Cloud

1. **Create a GitHub repo**
   ```bash
   cd breast_cancer_app
   git init
   git add .
   git commit -m "Breast cancer diagnosis predictor app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```

2. **Go to** [share.streamlit.io](https://share.streamlit.io) and sign in
   with your GitHub account.

3. Click **"New app"** → pick your repo, branch `main`, and set
   **Main file path** to `app.py`.

4. Click **Deploy**. Streamlit Cloud installs `requirements.txt`
   automatically and gives you a live URL like
   `https://<your-app-name>.streamlit.app`.
