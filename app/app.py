import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import torch
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler

from src.config import (MODELS_DIR, DATASETS, COLUMN_NAMES, USEFUL_SENSORS,
                        MAX_RUL, CLASSIFICATION_W, PROCESSED_DIR, SEQUENCE_LENGTH)
from src.preprocessing import drop_useless_columns, normalize_by_operating_condition
from src.models.deep_learning import DEVICE, FlexibleModel

st.set_page_config(page_title="NASA C-MAPSS RUL", layout="wide", page_icon="✈️")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: -10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classical_models():
    models = {}
    for f in MODELS_DIR.glob("*.pkl"):
        with open(f, "rb") as file:
            models[f.stem] = pickle.load(file)
    return models


@st.cache_resource
def load_optuna_params():
    path = PROCESSED_DIR / "optuna_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_train_data(ds_id):
    return pd.read_parquet(PROCESSED_DIR / f"train_{ds_id}.parquet")


def get_ml_feature_cols(train_data):
    exclude = {"unit_id", "cycle", "rul", "label"}
    return [c for c in train_data.columns if c not in exclude
            and "roll_mean" not in c and "roll_std" not in c and "trend" not in c]


def build_and_load_model(model_type, ds_id, n_features, params):
    hidden = params["hidden_size"]
    n_layers = params["num_layers"]
    dropout = params["dropout"]
    d_model = params.get("d_model", 64)
    nhead = params.get("nhead", 4)

    model = FlexibleModel(n_features, model_type, hidden, n_layers, dropout, d_model, nhead).to(DEVICE)

    pt_path = MODELS_DIR / f"{model_type}_optuna_{ds_id}.pt"
    if pt_path.exists():
        state = torch.load(pt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state)
    model.eval()
    return model


def predict_dl_single(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(DEVICE)
        return model(X_t).cpu().numpy()


def render_validation(last_cycle, container):
    with container:
        st.subheader("✅ Validación con RUL real")
        rul_file = st.file_uploader("Subí el archivo RUL real (ej: RUL_FD001.txt)", type=["csv", "txt"], key="rul_upload")

        if rul_file is not None:
            rul_real = pd.read_csv(rul_file, sep=r"\s+", header=None, names=["rul_real"])

            if len(rul_real) == len(last_cycle):
                last_cycle = last_cycle.copy()
                last_cycle["rul_real"] = rul_real["rul_real"].values
                last_cycle["error"] = last_cycle["predicted_rul"] - last_cycle["rul_real"]
                last_cycle["abs_error"] = last_cycle["error"].abs()

                rmse = np.sqrt(np.mean(last_cycle["error"] ** 2))
                mae = last_cycle["abs_error"].mean()
                ss_res = np.sum(last_cycle["error"] ** 2)
                ss_tot = np.sum((last_cycle["rul_real"] - last_cycle["rul_real"].mean()) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                d = last_cycle["predicted_rul"].values - last_cycle["rul_real"].values
                nasa = np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1))

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("RMSE", f"{rmse:.2f}")
                col2.metric("MAE", f"{mae:.2f}")
                col3.metric("R²", f"{r2:.4f}")
                col4.metric("NASA Score", f"{nasa:.1f}")

                fig_scatter = px.scatter(
                    last_cycle, x="rul_real", y="predicted_rul",
                    color="status",
                    color_discrete_map={"CRÍTICO": "#ff4757", "NORMAL": "#2ed573"},
                    title="Predicho vs Real",
                    template="plotly_white",
                    labels={"rul_real": "RUL Real", "predicted_rul": "RUL Predicho"},
                    hover_data=["unit_id", "error"],
                )
                max_val = max(last_cycle["rul_real"].max(), last_cycle["predicted_rul"].max())
                fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                                     line=dict(dash="dash", color="gray"))
                fig_scatter.update_layout(height=500)
                st.plotly_chart(fig_scatter, use_container_width=True)

                col_left, col_right = st.columns(2)
                with col_left:
                    fig_err = px.histogram(
                        last_cycle, x="error", nbins=30,
                        title="Distribución del error (predicho - real)",
                        template="plotly_white",
                        color_discrete_sequence=["#636EFA"],
                    )
                    fig_err.add_vline(x=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_err, use_container_width=True)

                with col_right:
                    fig_bar_err = px.bar(
                        last_cycle.sort_values("abs_error", ascending=False).head(10),
                        x="unit_id", y="abs_error",
                        title="Top 10 motores con mayor error",
                        template="plotly_white",
                        color="abs_error", color_continuous_scale="OrRd",
                        labels={"unit_id": "Motor ID", "abs_error": "Error absoluto"},
                    )
                    st.plotly_chart(fig_bar_err, use_container_width=True)

                acertados_10 = (last_cycle["abs_error"] <= 10).sum()
                acertados_20 = (last_cycle["abs_error"] <= 20).sum()
                total = len(last_cycle)

                col1, col2 = st.columns(2)
                col1.metric("Aciertos (±10 ciclos)", f"{acertados_10}/{total} ({acertados_10/total*100:.1f}%)")
                col2.metric("Aciertos (±20 ciclos)", f"{acertados_20}/{total} ({acertados_20/total*100:.1f}%)")

                st.subheader("Detalle de errores")
                display_cols = ["unit_id", "predicted_rul", "rul_real", "error", "abs_error", "status"]
                if any(c.startswith("rul_") and c != "rul_real" for c in last_cycle.columns):
                    model_cols = [c for c in last_cycle.columns if c.startswith("rul_") and c != "rul_real"]
                    display_cols = ["unit_id"] + model_cols + ["predicted_rul", "rul_real", "error", "abs_error", "status"]
                st.dataframe(
                    last_cycle[display_cols].sort_values("abs_error", ascending=False).reset_index(drop=True),
                    use_container_width=True,
                )
            else:
                st.error(f"El archivo RUL tiene {len(rul_real)} valores pero hay {len(last_cycle)} motores")


classical_models = load_classical_models()
optuna_params = load_optuna_params()

st.markdown('<p class="main-header">✈️ NASA C-MAPSS — RUL Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predictive Maintenance Dashboard — Machine Learning & Deep Learning</p>', unsafe_allow_html=True)

st.sidebar.title("⚙️ Configuración")

dataset_id = st.sidebar.selectbox("Sub-dataset", DATASETS, help="FD001=simple, FD004=complejo")

prediction_mode = st.sidebar.radio(
    "Modo de predicción",
    ["ML Clásico (rápido)", "Deep Learning (preciso)", "Ensemble DL (mejor)"],
    help="Ensemble promedia los 4 modelos DL para mayor precisión"
)

if prediction_mode == "ML Clásico (rápido)":
    reg_models = {k: v for k, v in classical_models.items() if "clf" not in k}
    ml_model_name = st.sidebar.selectbox("Modelo", list(reg_models.keys()))
elif prediction_mode == "Deep Learning (preciso)":
    dl_model_choice = st.sidebar.selectbox("Modelo DL", ["lstm", "gru", "tcn", "transformer"])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {dataset_id}")
st.sidebar.markdown(f"**MAX RUL:** {MAX_RUL} ciclos")
st.sidebar.markdown(f"**Umbral falla:** {CLASSIFICATION_W} ciclos")
gpu_status = f"✅ {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "❌ CPU"
st.sidebar.markdown(f"**GPU:** {gpu_status}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Predicción", "📊 ML Clásico", "🧠 Deep Learning", "🔍 SHAP", "ℹ️ Acerca de"])

with tab1:
    st.header("Predicción de RUL")

    uploaded_file = st.file_uploader("Subí un archivo de test NASA C-MAPSS", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=r"\s+", header=None, names=COLUMN_NAMES)
            n_motors = df["unit_id"].nunique()
            st.success(f"✅ Archivo cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas | {n_motors} motores")

            if prediction_mode == "ML Clásico (rápido)":
                train_data = load_train_data("FD001")
                test_data = pd.read_parquet(PROCESSED_DIR / f"test_{dataset_id}.parquet")
                ml_cols = get_ml_feature_cols(train_data)
                available = [c for c in ml_cols if c in test_data.columns]
                X = test_data[available].values
                predictions = reg_models[ml_model_name].predict(X)
                method_used = f"ML Clásico — {ml_model_name}"
                last_cycle = test_data[["unit_id"]].copy()
                last_cycle["predicted_rul"] = np.clip(predictions, 0, MAX_RUL).round(1)

            else:
                with st.spinner("🧠 Cargando datos de entrenamiento y modelos DL..."):
                    train_data = load_train_data(dataset_id)

                    exclude = {"unit_id", "cycle", "rul", "label"}
                    cols = [c for c in train_data.columns if c not in exclude]

                    train_norm = normalize_by_operating_condition(train_data)
                    scaler = MinMaxScaler()
                    scaler.fit(train_norm[cols])

                    df_test_norm = normalize_by_operating_condition(df)

                    missing_cols = [c for c in cols if c not in df_test_norm.columns]
                    for mc in missing_cols:
                        df_test_norm[mc] = 0

                    df_test_norm[cols] = scaler.transform(df_test_norm[cols])

                    rul_max = 125.0
                    n_features = len(cols)

                    model_types = ["lstm", "gru", "tcn", "transformer"]
                    all_preds = []
                    individual_preds = {}

                    progress = st.progress(0)

                    for idx, mt in enumerate(model_types):
                        key = f"{mt}_{dataset_id}"
                        if key not in optuna_params:
                            progress.progress((idx + 1) / len(model_types))
                            continue

                        params = optuna_params[key]["best_params"]
                        model = build_and_load_model(mt, dataset_id, n_features, params)

                        motor_preds = []
                        test_units = df_test_norm["unit_id"].unique()

                        for unit_id in test_units:
                            unit_data = df_test_norm[df_test_norm["unit_id"] == unit_id]

                            if len(unit_data) >= SEQUENCE_LENGTH:
                                seq = unit_data[cols].values[-SEQUENCE_LENGTH:]
                            else:
                                seq = unit_data[cols].values
                                pad_len = SEQUENCE_LENGTH - len(seq)
                                seq = np.vstack([np.tile(seq[0], (pad_len, 1)), seq])

                            seq = seq.reshape(1, SEQUENCE_LENGTH, -1)
                            pred = predict_dl_single(model, seq)[0] * rul_max
                            motor_preds.append(pred)

                        all_preds.append(np.array(motor_preds))
                        individual_preds[mt.upper()] = np.array(motor_preds)
                        progress.progress((idx + 1) / len(model_types))

                    if prediction_mode == "Ensemble DL (mejor)":
                        predictions = np.mean(all_preds, axis=0)
                        method_used = "Ensemble DL (LSTM + GRU + TCN + Transformer)"
                    else:
                        chosen = dl_model_choice.upper()
                        if chosen in individual_preds:
                            predictions = individual_preds[chosen]
                            method_used = f"Deep Learning — {chosen}"
                        else:
                            predictions = all_preds[-1]
                            method_used = "Deep Learning — Transformer"

                    last_cycle = df.groupby("unit_id").last().reset_index()
                    last_cycle["predicted_rul"] = np.clip(predictions, 0, MAX_RUL).round(1)

                    for mt_name, mt_preds in individual_preds.items():
                        last_cycle[f"rul_{mt_name}"] = np.clip(mt_preds, 0, MAX_RUL).round(1)

                    progress.empty()

            last_cycle["status"] = np.where(last_cycle["predicted_rul"] <= CLASSIFICATION_W, "CRÍTICO", "NORMAL")

            st.markdown(f"**Método:** `{method_used}`")

            # ── Predicción vs Realidad ──
            rul_real_path = ROOT / "data" / "raw" / f"RUL_{dataset_id}.txt"
            if rul_real_path.exists():
                rul_real = pd.read_csv(rul_real_path, sep=r"\s+", header=None, names=["rul_real"])
                if len(rul_real) == len(last_cycle):
                    real_criticos = int((rul_real["rul_real"] <= CLASSIFICATION_W).sum())
                    pred_criticos = int((last_cycle["status"] == "CRÍTICO").sum())
                    real_normales = len(rul_real) - real_criticos
                    pred_normales = len(last_cycle) - pred_criticos

                    aciertos = 0
                    for i in range(len(last_cycle)):
                        real_status = "CRÍTICO" if rul_real.iloc[i]["rul_real"] <= CLASSIFICATION_W else "NORMAL"
                        if last_cycle.iloc[i]["status"] == real_status:
                            aciertos += 1

                    st.markdown("### 🆚 Predicción vs Realidad")
                    col_r, col_p, col_a = st.columns(3)
                    with col_r:
                        st.markdown("**Realidad**")
                        st.metric("Críticos reales", f"{real_criticos} de {len(rul_real)}")
                        st.metric("Normales reales", f"{real_normales} de {len(rul_real)}")
                    with col_p:
                        st.markdown("**Predicción**")
                        st.metric("Críticos predichos", f"{pred_criticos} de {len(last_cycle)}")
                        st.metric("Normales predichos", f"{pred_normales} de {len(last_cycle)}")
                    with col_a:
                        st.markdown("**Resultado**")
                        st.metric("✅ Aciertos", f"{aciertos} de {len(last_cycle)} ({aciertos / len(last_cycle) * 100:.1f}%)")

            # ── Métricas generales ──
            col1, col2, col3, col4 = st.columns(4)
            n_criticos = (last_cycle["status"] == "CRÍTICO").sum()
            col1.metric("✈️ Motores", len(last_cycle))
            col2.metric("⏱️ RUL promedio", f"{last_cycle['predicted_rul'].mean():.1f}")
            col3.metric("⚠️ Críticos", n_criticos)
            col4.metric("📊 % Críticos", f"{n_criticos / len(last_cycle) * 100:.1f}%")

            # ── Gráfico principal ──
            fig_main = px.bar(
                last_cycle.sort_values("predicted_rul"),
                x="unit_id", y="predicted_rul", color="status",
                color_discrete_map={"CRÍTICO": "#ff4757", "NORMAL": "#2ed573"},
                title="RUL predicho por motor",
                template="plotly_white",
                labels={"unit_id": "Motor ID", "predicted_rul": "RUL predicho (ciclos)"},
            )
            fig_main.add_hline(y=CLASSIFICATION_W, line_dash="dash", line_color="red",
                             annotation_text=f"Umbral = {CLASSIFICATION_W}")
            fig_main.update_layout(height=500)
            st.plotly_chart(fig_main, use_container_width=True)

            # ── Gráficos secundarios ──
            col_left, col_mid, col_right = st.columns(3)
            with col_left:
                fig_dist = px.histogram(
                    last_cycle, x="predicted_rul", color="status",
                    color_discrete_map={"CRÍTICO": "#ff4757", "NORMAL": "#2ed573"},
                    title="Distribución de RUL", template="plotly_white", nbins=20,
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            with col_mid:
                fig_pie = px.pie(
                    last_cycle, names="status", color="status",
                    color_discrete_map={"CRÍTICO": "#ff4757", "NORMAL": "#2ed573"},
                    title="Estado de la flota", template="plotly_white",
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_right:
                fig_box = px.box(
                    last_cycle, y="predicted_rul", color="status",
                    color_discrete_map={"CRÍTICO": "#ff4757", "NORMAL": "#2ed573"},
                    title="Distribución por estado", template="plotly_white",
                )
                st.plotly_chart(fig_box, use_container_width=True)

            # ── Comparación DL ──
            if prediction_mode != "ML Clásico (rápido)" and len(individual_preds) > 0:
                st.subheader("🔬 Comparación entre modelos DL")

                model_cols = [c for c in last_cycle.columns if c.startswith("rul_")]
                if model_cols:
                    compare_df = last_cycle[["unit_id"] + model_cols + ["predicted_rul"]].copy()
                    compare_df = compare_df.rename(columns={"predicted_rul": "ENSEMBLE"})

                    melt_df = compare_df.melt(id_vars="unit_id", var_name="Modelo", value_name="RUL")
                    melt_df["Modelo"] = melt_df["Modelo"].str.replace("rul_", "")

                    fig_compare = px.line(
                        melt_df.sort_values(["unit_id", "Modelo"]),
                        x="unit_id", y="RUL", color="Modelo",
                        title="Predicción por modelo — Cada motor",
                        template="plotly_white",
                        color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"],
                    )
                    fig_compare.add_hline(y=CLASSIFICATION_W, line_dash="dash", line_color="red")
                    fig_compare.update_layout(height=450)
                    st.plotly_chart(fig_compare, use_container_width=True)

                    st.subheader("📊 Estadísticas por modelo")
                    stats = {}
                    for col in model_cols:
                        col_name = col.replace("rul_", "")
                        vals = last_cycle[col]
                        stats[col_name] = {
                            "Mean RUL": vals.mean(),
                            "Std RUL": vals.std(),
                            "Min RUL": vals.min(),
                            "Max RUL": vals.max(),
                            "Críticos": int((vals <= CLASSIFICATION_W).sum()),
                        }
                    vals = last_cycle["predicted_rul"]
                    stats["ENSEMBLE"] = {
                        "Mean RUL": vals.mean(),
                        "Std RUL": vals.std(),
                        "Min RUL": vals.min(),
                        "Max RUL": vals.max(),
                        "Críticos": int((vals <= CLASSIFICATION_W).sum()),
                    }
                    st.dataframe(pd.DataFrame(stats).T.round(2), use_container_width=True)

            # ── Motores críticos ──
            top_critical = last_cycle[last_cycle["status"] == "CRÍTICO"].nsmallest(5, "predicted_rul")
            if len(top_critical) > 0:
                st.subheader("⚠️ Motores más críticos")
                display_cols = ["unit_id", "predicted_rul", "status"]
                if any(c.startswith("rul_") for c in last_cycle.columns):
                    display_cols = ["unit_id"] + [c for c in last_cycle.columns if c.startswith("rul_")] + ["predicted_rul", "status"]
                st.dataframe(top_critical[display_cols].reset_index(drop=True), use_container_width=True)

            # ── Detalle completo ──
            st.subheader("📋 Detalle completo")
            display_all = ["unit_id", "predicted_rul", "status"]
            if any(c.startswith("rul_") for c in last_cycle.columns):
                display_all = ["unit_id"] + [c for c in last_cycle.columns if c.startswith("rul_")] + ["predicted_rul", "status"]
            st.dataframe(
                last_cycle[display_all].sort_values("predicted_rul").reset_index(drop=True),
                use_container_width=True,
            )

            csv = last_cycle[display_all].to_csv(index=False)
            st.download_button("📥 Descargar predicciones CSV", csv, "predicciones_rul.csv", "text/csv")

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)
    else:
        st.markdown("""
        ### ¿Cómo usar?
        1. Seleccioná el **sub-dataset** en el sidebar (FD001-FD004)
        2. Elegí el **modo de predicción** (ML Clásico, DL o Ensemble)
        3. Subí el **archivo de test** correspondiente (ej: `test_FD001.txt`)
        4. El sistema compara automáticamente con la realidad

        **Recomendación:** Usá **Ensemble DL** para la mejor precisión
        """)

with tab2:
    st.header("📊 Resultados — ML Clásico")

    classical_data = {
        "Modelo": ["Linear Regression", "SVR", "Random Forest", "XGBoost"],
        "RMSE": [21.91, 19.61, 17.92, 18.31],
        "MAE": [17.61, 13.82, 12.92, 13.39],
        "R²": [0.722, 0.777, 0.814, 0.806],
        "NASA Score": [1318.89, 1624.85, 886.38, 1058.09],
    }
    classical_df = pd.DataFrame(classical_data)

    metric = st.selectbox("Métrica", ["RMSE", "MAE", "R²", "NASA Score"])

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(classical_df, use_container_width=True)
    with col2:
        ascending = metric != "R²"
        fig = px.bar(
            classical_df.sort_values(metric, ascending=ascending),
            x="Modelo", y=metric, color="Modelo",
            title=f"{metric} por modelo — FD001",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Clasificación binaria (FD001)")
    clf_data = {
        "Modelo": ["Random Forest", "XGBoost"],
        "Accuracy": [0.910, 0.910],
        "Precision": [0.900, 0.900],
        "Recall": [0.720, 0.720],
        "F1": [0.800, 0.800],
        "AUC-ROC": [0.983, 0.980],
    }
    st.dataframe(pd.DataFrame(clf_data), use_container_width=True)

with tab3:
    st.header("🧠 Resultados — Deep Learning (Optuna + Ensemble)")

    dl_data = {
        "Dataset": ["FD001"] * 5 + ["FD002"] * 5 + ["FD003"] * 5 + ["FD004"] * 5,
        "Modelo": ["LSTM", "GRU", "TCN", "Transformer", "Ensemble"] * 4,
        "RMSE": [14.24, 14.89, 14.39, 13.16, 11.97,
                 12.80, 14.86, 13.43, 14.28, 11.95,
                 15.56, 13.71, 16.82, 13.34, 13.46,
                 17.92, 18.66, 21.48, 17.77, 17.32],
        "R²": [0.873, 0.861, 0.870, 0.892, 0.910,
               0.892, 0.855, 0.881, 0.866, 0.906,
               0.860, 0.891, 0.836, 0.897, 0.895,
               0.824, 0.809, 0.747, 0.827, 0.835],
    }
    dl_df = pd.DataFrame(dl_data)

    selected_ds = st.selectbox("Dataset", DATASETS, key="dl_ds")
    filtered = dl_df[dl_df["Dataset"] == selected_ds]

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(filtered[["Modelo", "RMSE", "R²"]].reset_index(drop=True), use_container_width=True)
    with col2:
        colors = {"LSTM": "#636EFA", "GRU": "#EF553B", "TCN": "#00CC96", "Transformer": "#AB63FA", "Ensemble": "#FFA15A"}
        fig = px.bar(
            filtered, x="Modelo", y="RMSE", color="Modelo",
            title=f"RMSE — {selected_ds}", template="plotly_white",
            color_discrete_map=colors,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Heatmap — Todos los datasets")
    hm_metric = st.radio("Métrica", ["RMSE", "R²"], horizontal=True)
    pivot = dl_df.pivot(index="Modelo", columns="Dataset", values=hm_metric)
    cscale = "RdYlGn_r" if hm_metric == "RMSE" else "RdYlGn"
    fig_heat = px.imshow(
        pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        color_continuous_scale=cscale,
        title=f"{hm_metric} — Modelos × Datasets",
        text_auto=".2f" if hm_metric == "RMSE" else ".3f",
    )
    fig_heat.update_layout(height=400)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Ensemble vs Mejor individual")
    comparison = []
    for ds in DATASETS:
        ds_data = dl_df[dl_df["Dataset"] == ds]
        individual = ds_data[ds_data["Modelo"] != "Ensemble"]
        best = individual.loc[individual["RMSE"].idxmin()]
        ens = ds_data[ds_data["Modelo"] == "Ensemble"].iloc[0]
        comparison.append({
            "Dataset": ds, "Mejor individual": best["Modelo"],
            "RMSE individual": best["RMSE"], "RMSE Ensemble": ens["RMSE"],
            "Mejora": round(best["RMSE"] - ens["RMSE"], 2),
        })
    st.dataframe(pd.DataFrame(comparison), use_container_width=True)

with tab4:
    st.header("🔍 Interpretabilidad — SHAP")

    shap_data = {
        "Sensor": ["sensor_11", "sensor_9", "sensor_4", "sensor_14", "sensor_12",
                    "sensor_7", "sensor_15"],
        "Descripción": [
            "Temperatura salida HPC", "Presión total salida HPC",
            "Temperatura salida LPC", "Relación presión fan/bypass",
            "Velocidad corrector fan", "Presión total salida fan",
            "Relación bypass",
        ],
        "SHAP XGBoost": [9.85, 7.14, 5.88, 4.31, 4.11, 3.20, 2.80],
        "SHAP Random Forest": [15.83, 8.91, 4.73, 2.10, 3.60, 2.31, 1.90],
    }
    shap_df = pd.DataFrame(shap_data)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            shap_df, x="SHAP XGBoost", y="Sensor", orientation="h",
            title="SHAP — XGBoost", template="plotly_white",
            color="SHAP XGBoost", color_continuous_scale="OrRd",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(
            shap_df, x="SHAP Random Forest", y="Sensor", orientation="h",
            title="SHAP — Random Forest", template="plotly_white",
            color="SHAP Random Forest", color_continuous_scale="OrRd",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(shap_df, use_container_width=True)

    st.subheader("Hallazgos clave")
    st.markdown("""
    - **sensor_11** (temperatura HPC) es el predictor #1 en todos los datasets y modelos
    - **sensor_4** (temperatura LPC) aparece consistentemente en el top 5 de los 4 sub-datasets
    - Los sensores de **temperatura y presión del compresor de alta presión** dominan las predicciones
    - SHAP revela que el impacto de sensor_11 **aumenta exponencialmente** cerca de la falla
    - La consistencia entre XGBoost y Random Forest valida la robustez de los hallazgos
    """)

with tab5:
    st.header("ℹ️ Acerca del proyecto")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### NASA C-MAPSS RUL Prediction

        Estudio comparativo de Machine Learning y Deep Learning para predicción
        de vida útil remanente en motores turbofan.

        **Dataset:** NASA C-MAPSS
        - 4 sub-datasets (FD001-FD004) con complejidad creciente
        - 21 sensores + 3 configuraciones operacionales
        - Run-to-failure: motores desde estado sano hasta falla

        **Pipeline completo:**
        1. EDA exhaustivo (6 notebooks con visualizaciones 3D)
        2. Preprocesamiento + normalización por condición operacional
        3. Feature engineering (rolling mean, std, trend)
        4. ML Clásico: Linear Regression, SVR, Random Forest, XGBoost
        5. Deep Learning: LSTM, GRU, TCN, Transformer (PyTorch + GPU)
        6. Optimización: Optuna (800 trials) + Ensemble averaging
        7. Interpretabilidad: SHAP values (global y local)
        8. Web app interactiva (Streamlit)

        ---

        **Universidad LEAD** — BCD-6210 Minería de Datos Avanzada

        Dr. Juan Murillo-Morera
        """)

    with col2:
        st.markdown("### Resultados")
        st.metric("Mejor RMSE", "11.95", "FD002")
        st.metric("Mejor R²", "0.910", "FD001")
        st.metric("Sensores clave", "sensor_11, 4")
        st.metric("Modelos", "8 + Ensemble")
        st.metric("Optimización", "Optuna 800 trials")
        st.metric("Framework", "PyTorch + GPU")
