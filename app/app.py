import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import pickle
import numpy as np
import pandas as pd
import torch
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

from src.config import (MODELS_DIR, DATASETS, COLUMN_NAMES, USEFUL_SENSORS,
                        MAX_RUL, CLASSIFICATION_W, PROCESSED_DIR, SEQUENCE_LENGTH)
from src.preprocessing import Preprocessor
from src.models.deep_learning import DEVICE, FlexibleModel

st.set_page_config(page_title="NASA C-MAPSS RUL", layout="wide", page_icon="✈️")

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem; font-weight: 700;
    background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.sub-header { font-size: 1.1rem; color: #666; margin-top: -10px; margin-bottom: 20px; }
.metric-card {
    background: #f8f9fa; border-radius: 10px;
    padding: 15px; border-left: 4px solid #4C72B0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_classical_models():
    models = {}
    for f in MODELS_DIR.glob("*.pkl"):
        with open(f, "rb") as fh:
            models[f.stem] = pickle.load(fh)
    return models


@st.cache_resource
def load_optuna_params():
    path = PROCESSED_DIR / "optuna_results.json"
    return json.load(open(path)) if path.exists() else {}


@st.cache_data
def load_train_data(ds_id):
    return pd.read_parquet(PROCESSED_DIR / f"train_{ds_id}.parquet")


def get_ml_feature_cols(train_data):
    exclude = {"unit_id", "cycle", "rul", "label"}
    return [c for c in train_data.columns if c not in exclude
            and "roll_mean" not in c and "roll_std" not in c and "trend" not in c]


def build_and_load_model(model_type, ds_id, n_features, params):
    model = FlexibleModel(
        n_features, model_type,
        params["hidden_size"], params["num_layers"], params["dropout"],
        params.get("d_model", 64), params.get("nhead", 4)
    ).to(DEVICE)
    pt_path = MODELS_DIR / f"{model_type}_optuna_{ds_id}.pt"
    if pt_path.exists():
        model.load_state_dict(torch.load(pt_path, map_location=DEVICE, weights_only=False))
    model.eval()
    return model


def predict_dl_single(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X).to(DEVICE)).cpu().numpy()


def load_rul_real(ds_id):
    path = ROOT / "data" / "raw" / f"RUL_{ds_id}.txt"
    if not path.exists():
        return None
    return pd.read_csv(path, sep=r"\s+", header=None, names=["rul_real"])["rul_real"].values


classical_models = load_classical_models()
optuna_params    = load_optuna_params()

st.markdown('<p class="main-header">✈️ NASA C-MAPSS — RUL Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predictive Maintenance Dashboard — ML & Deep Learning</p>',
            unsafe_allow_html=True)

st.sidebar.title("⚙️ Configuracion")
dataset_id      = st.sidebar.selectbox("Sub-dataset", DATASETS)
prediction_mode = st.sidebar.radio(
    "Modo de prediccion",
    ["ML Clasico (rapido)", "Deep Learning (preciso)", "Ensemble DL (mejor)"]
)
if prediction_mode == "ML Clasico (rapido)":
    reg_models    = {k: v for k, v in classical_models.items() if "clf" not in k}
    ml_model_name = st.sidebar.selectbox("Modelo", list(reg_models.keys()))
elif prediction_mode == "Deep Learning (preciso)":
    dl_model_choice = st.sidebar.selectbox("Modelo DL", ["lstm", "gru", "tcn", "transformer"])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** `{dataset_id}`")
st.sidebar.markdown(f"**MAX RUL:** {MAX_RUL} ciclos")
st.sidebar.markdown(f"**Umbral falla:** {CLASSIFICATION_W} ciclos")
gpu_info = f"✅ {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "💻 CPU"
st.sidebar.markdown(f"**Device:** {gpu_info}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🔮 Prediccion", "📊 ML Clasico", "🧠 Deep Learning", "🔍 SHAP", "ℹ️ Acerca de"])

with tab1:
    st.header("Prediccion de RUL")
    uploaded_file = st.file_uploader(
        "Subi un archivo de test NASA C-MAPSS (.txt o .csv)", type=["csv", "txt"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, sep=r"\s+", header=None, names=COLUMN_NAMES)
            n_motors = df["unit_id"].nunique()
            st.success(f"✅ Archivo cargado — {df.shape[0]:,} filas | {n_motors} motores")

            with st.spinner("Calculando predicciones..."):

                if prediction_mode == "ML Clasico (rapido)":
                    test_data = pd.read_parquet(PROCESSED_DIR / f"test_{dataset_id}.parquet")
                    train_data = load_train_data("FD001")
                    ml_cols    = get_ml_feature_cols(train_data)
                    avail      = [c for c in ml_cols if c in test_data.columns]
                    raw_preds  = reg_models[ml_model_name].predict(test_data[avail].values)
                    predictions = np.clip(raw_preds, 0, MAX_RUL)
                    method_used = f"ML Clasico — {ml_model_name}"
                    last_cycle  = test_data.groupby("unit_id").last().reset_index()[["unit_id"]]
                    last_cycle["predicted_rul"] = predictions.round(1)

                else:
                    train_data = load_train_data(dataset_id)
                    exclude    = {"unit_id", "cycle", "rul", "label"}
                    cols       = [c for c in train_data.columns if c not in exclude]
                    prep       = Preprocessor()
                    tr_norm    = prep.normalize_by_operating_condition(train_data)
                    scaler     = MinMaxScaler()
                    scaler.fit(tr_norm[cols])

                    df_proc = prep.normalize_by_operating_condition(df)
                    for c in cols:
                        if c not in df_proc.columns:
                            df_proc[c] = 0
                    df_proc[cols] = scaler.transform(df_proc[cols])

                    model_types      = ["lstm", "gru", "tcn", "transformer"]
                    all_preds        = []
                    individual_preds = {}
                    progress         = st.progress(0, text="Cargando modelos DL...")

                    def build_sequences_batch(df_proc, cols, unit_ids):
                        seqs = []
                        for uid in unit_ids:
                            seq = df_proc[df_proc["unit_id"] == uid][cols].values
                            if len(seq) >= SEQUENCE_LENGTH:
                                seq = seq[-SEQUENCE_LENGTH:]
                            else:
                                pad = np.tile(seq[0], (SEQUENCE_LENGTH - len(seq), 1))
                                seq = np.vstack([pad, seq])
                            seqs.append(seq)
                        return np.array(seqs, dtype=np.float32)


                    unit_ids = list(df_proc["unit_id"].unique())
                    X_batch  = build_sequences_batch(df_proc, cols, unit_ids)

                    for idx, mt in enumerate(model_types):
                        key = f"{mt}_{dataset_id}"
                        progress.progress(idx / len(model_types),
                                        text=f"Prediciendo con {mt.upper()}...")
                        if key not in optuna_params:
                            progress.progress((idx + 1) / len(model_types),
                                            text=f"{mt.upper()} — sin modelo entrenado")
                            continue

                        params = optuna_params[key]["best_params"]
                        model  = build_and_load_model(mt, dataset_id, len(cols), params)
                        model.eval()

                        with torch.no_grad():
                            X_tensor = torch.FloatTensor(X_batch).to(DEVICE)
                            preds    = model(X_tensor).cpu().numpy() * MAX_RUL
                            preds    = np.clip(preds, 0, MAX_RUL)

                        all_preds.append(preds)
                        individual_preds[mt.upper()] = preds.round(1)
                        progress.progress((idx + 1) / len(model_types),
                                        text=f"{mt.upper()} completado ✓")

                    progress.empty()

                    if prediction_mode == "Ensemble DL (mejor)" and all_preds:
                        predictions = np.mean(all_preds, axis=0)
                        method_used = "Ensemble DL (LSTM + GRU + TCN + Transformer)"
                    elif individual_preds:
                        predictions = individual_preds.get(dl_model_choice.upper(), all_preds[-1])
                        method_used = f"Deep Learning — {dl_model_choice.upper()}"
                    else:
                        st.error("No se encontraron modelos entrenados para este dataset.")
                        st.stop()

                    last_cycle = pd.DataFrame({
                        "unit_id": list(df_proc["unit_id"].unique()),
                        "predicted_rul": predictions.round(1),
                    })
                    for mt_name, mt_preds in individual_preds.items():
                        last_cycle[f"rul_{mt_name}"] = mt_preds.round(1)

            last_cycle["status"] = np.where(
                last_cycle["predicted_rul"] <= CLASSIFICATION_W, "CRITICO", "NORMAL")
            last_cycle["predicted_critical"] = last_cycle["status"] == "CRITICO"

            rul_real = load_rul_real(dataset_id)

            st.markdown("---")

            if rul_real is not None and len(rul_real) == len(last_cycle):
                last_cycle["rul_real"]       = rul_real
                last_cycle["real_critical"]  = last_cycle["rul_real"] <= CLASSIFICATION_W
                last_cycle["correct"]        = last_cycle["predicted_critical"] == last_cycle["real_critical"]
                last_cycle["error_abs"]      = (last_cycle["predicted_rul"] - last_cycle["rul_real"]).abs()

                real_crit  = int(last_cycle["real_critical"].sum())
                pred_crit  = int(last_cycle["predicted_critical"].sum())
                aciertos   = int(last_cycle["correct"].sum())
                pct_ok     = aciertos / len(last_cycle) * 100
                rmse_val   = float(np.sqrt(np.mean((last_cycle["predicted_rul"] - last_cycle["rul_real"]) ** 2)))
                mae_val    = float(last_cycle["error_abs"].mean())

                tp = int((last_cycle["real_critical"] & last_cycle["predicted_critical"]).sum())
                fp = int((~last_cycle["real_critical"] & last_cycle["predicted_critical"]).sum())
                fn = int((last_cycle["real_critical"] & ~last_cycle["predicted_critical"]).sum())
                tn = int((~last_cycle["real_critical"] & ~last_cycle["predicted_critical"]).sum())

                st.subheader("📊 Prediccion vs Realidad")

                c1, c2, c3, c4, c5, c6 = st.columns(6)
                c1.metric("✈️ Motores totales",    len(last_cycle))
                c2.metric("⚠️ Criticos reales",    real_crit,
                          delta=f"{real_crit/len(last_cycle)*100:.1f}% de la flota")
                c3.metric("🔴 Criticos predichos", pred_crit,
                          delta=f"{pred_crit - real_crit:+d} vs real",
                          delta_color="inverse")
                c4.metric("✅ Clasificaciones OK", f"{aciertos}/{len(last_cycle)}",
                          delta=f"{pct_ok:.1f}%")
                c5.metric("📉 RMSE", f"{rmse_val:.2f} ciclos")
                c6.metric("📉 MAE",  f"{mae_val:.2f} ciclos")

                col_cm, col_comp = st.columns([1, 2])

                with col_cm:
                    st.markdown("**Matriz de confusion**")
                    cm_df = pd.DataFrame(
                        [[tp, fp], [fn, tn]],
                        index=["Real: CRITICO", "Real: NORMAL"],
                        columns=["Pred: CRITICO", "Pred: NORMAL"]
                    )
                    fig_cm = go.Figure(go.Heatmap(
                        z=[[tp, fp], [fn, tn]],
                        x=["Pred: CRITICO", "Pred: NORMAL"],
                        y=["Real: CRITICO", "Real: NORMAL"],
                        text=[[f"TP={tp}", f"FP={fp}"], [f"FN={fn}", f"TN={tn}"]],
                        texttemplate="%{text}",
                        colorscale="Blues",
                        showscale=False,
                    ))
                    fig_cm.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(fig_cm, use_container_width=True)
                    if fn > 0:
                        st.warning(f"⚠️ {fn} motor(es) critico(s) NO detectado(s) — riesgo operacional")
                    if fp > 0:
                        st.info(f"ℹ️ {fp} falsa(s) alarma(s) — mantenimiento innecesario")

                with col_comp:
                    st.markdown("**RUL predicho vs real por motor**")
                    comp_df = last_cycle.copy()
                    comp_df["color"] = comp_df["correct"].map({True: "Correcto", False: "Error"})
                    fig_comp = go.Figure()
                    fig_comp.add_trace(go.Scatter(
                        x=comp_df["unit_id"], y=comp_df["rul_real"],
                        mode="markers", name="RUL Real",
                        marker=dict(color="#55A868", size=8, symbol="circle"),
                    ))
                    fig_comp.add_trace(go.Scatter(
                        x=comp_df["unit_id"], y=comp_df["predicted_rul"],
                        mode="markers", name="RUL Predicho",
                        marker=dict(
                            color=comp_df["correct"].map({True: "#4C72B0", False: "#C44E52"}),
                            size=8, symbol="diamond",
                        ),
                    ))
                    fig_comp.add_hline(y=CLASSIFICATION_W, line_dash="dash",
                                       line_color="red", annotation_text=f"W={CLASSIFICATION_W}")
                    fig_comp.update_layout(
                        height=300, template="plotly_white",
                        xaxis_title="Motor ID", yaxis_title="RUL (ciclos)",
                        legend=dict(orientation="h", y=1.1),
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)

                st.markdown("**Scatter predicho vs real**")
                fig_scatter = px.scatter(
                    last_cycle, x="rul_real", y="predicted_rul",
                    color="correct",
                    color_discrete_map={True: "#4C72B0", False: "#C44E52"},
                    labels={"rul_real": "RUL Real", "predicted_rul": "RUL Predicho",
                            "correct": "Clasificacion"},
                    hover_data=["unit_id", "rul_real", "predicted_rul", "status"],
                    template="plotly_white", height=400,
                )
                max_val = max(last_cycle["rul_real"].max(), last_cycle["predicted_rul"].max()) * 1.05
                fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                                      line=dict(color="gray", dash="dash"))
                fig_scatter.add_vline(x=CLASSIFICATION_W, line_dash="dot", line_color="red",
                                      annotation_text="W real")
                fig_scatter.add_hline(y=CLASSIFICATION_W, line_dash="dot", line_color="orange",
                                      annotation_text="W pred")
                st.plotly_chart(fig_scatter, use_container_width=True)

                with st.expander("🔴 Motores criticos NO detectados (Falsos Negativos)"):
                    fn_df = last_cycle[last_cycle["real_critical"] & ~last_cycle["predicted_critical"]]
                    if len(fn_df) > 0:
                        st.dataframe(fn_df[["unit_id", "rul_real", "predicted_rul", "status"]].reset_index(drop=True),
                                     use_container_width=True)
                    else:
                        st.success("Ningun motor critico fue perdido")

                with st.expander("🟡 Falsas alarmas (Falsos Positivos)"):
                    fp_df = last_cycle[~last_cycle["real_critical"] & last_cycle["predicted_critical"]]
                    if len(fp_df) > 0:
                        st.dataframe(fp_df[["unit_id", "rul_real", "predicted_rul", "status"]].reset_index(drop=True),
                                     use_container_width=True)
                    else:
                        st.success("Sin falsas alarmas")

            else:
                st.subheader("📊 Resultados de prediccion")
                pred_crit = int(last_cycle["predicted_critical"].sum())
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("✈️ Motores",    len(last_cycle))
                c2.metric("⏱️ RUL prom.", f"{last_cycle['predicted_rul'].mean():.1f}")
                c3.metric("⚠️ Criticos",  pred_crit)
                c4.metric("📊 % Criticos", f"{pred_crit/len(last_cycle)*100:.1f}%")
                st.info("Para ver comparacion vs realidad, asegurate de tener "
                        f"`RUL_{dataset_id}.txt` en `data/raw/`")

            st.markdown(f"**Metodo:** `{method_used}`")
            st.markdown("---")

            fig_bar = px.bar(
                last_cycle.sort_values("predicted_rul"),
                x="unit_id", y="predicted_rul", color="status",
                color_discrete_map={"CRITICO": "#ff4757", "NORMAL": "#2ed573"},
                title="RUL predicho por motor",
                template="plotly_white",
                labels={"unit_id": "Motor ID", "predicted_rul": "RUL predicho (ciclos)"},
            )
            if rul_real is not None and len(rul_real) == len(last_cycle):
                fig_bar.add_scatter(
                    x=last_cycle.sort_values("predicted_rul")["unit_id"],
                    y=last_cycle.sort_values("predicted_rul")["rul_real"],
                    mode="markers", name="RUL Real",
                    marker=dict(color="black", size=6, symbol="x"),
                )
            fig_bar.add_hline(y=CLASSIFICATION_W, line_dash="dash", line_color="red",
                              annotation_text=f"Umbral = {CLASSIFICATION_W}")
            fig_bar.update_layout(height=500)
            st.plotly_chart(fig_bar, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.plotly_chart(px.histogram(
                    last_cycle, x="predicted_rul", color="status",
                    color_discrete_map={"CRITICO": "#ff4757", "NORMAL": "#2ed573"},
                    title="Distribucion RUL", template="plotly_white", nbins=20),
                    use_container_width=True)
            with c2:
                st.plotly_chart(px.pie(
                    last_cycle, names="status", color="status",
                    color_discrete_map={"CRITICO": "#ff4757", "NORMAL": "#2ed573"},
                    title="Estado de la flota", template="plotly_white"),
                    use_container_width=True)
            with c3:
                st.plotly_chart(px.box(
                    last_cycle, y="predicted_rul", color="status",
                    color_discrete_map={"CRITICO": "#ff4757", "NORMAL": "#2ed573"},
                    title="Distribucion por estado", template="plotly_white"),
                    use_container_width=True)

            if prediction_mode != "ML Clasico (rapido)" and individual_preds:
                st.subheader("🔬 Comparacion entre modelos DL")
                model_cols = [c for c in last_cycle.columns if c.startswith("rul_")
                              and c != "rul_real"]
                if model_cols:
                    melt_df = last_cycle[["unit_id"] + model_cols + ["predicted_rul"]].rename(
                        columns={"predicted_rul": "ENSEMBLE"}).melt(
                        id_vars="unit_id", var_name="Modelo", value_name="RUL")
                    melt_df["Modelo"] = melt_df["Modelo"].str.replace("rul_", "")
                    fig_cmp = px.line(
                        melt_df.sort_values(["unit_id", "Modelo"]),
                        x="unit_id", y="RUL", color="Modelo",
                        title="Prediccion por modelo — cada motor",
                        template="plotly_white")
                    fig_cmp.add_hline(y=CLASSIFICATION_W, line_dash="dash", line_color="red")
                    fig_cmp.update_layout(height=400)
                    st.plotly_chart(fig_cmp, use_container_width=True)

            top_crit = last_cycle[last_cycle["status"] == "CRITICO"].nsmallest(5, "predicted_rul")
            if len(top_crit) > 0:
                st.subheader("⚠️ Top 5 motores mas criticos")
                display_cols = ["unit_id", "predicted_rul"]
                if "rul_real" in last_cycle.columns:
                    display_cols += ["rul_real", "error_abs"]
                display_cols += ["status"]
                st.dataframe(top_crit[display_cols].reset_index(drop=True),
                             use_container_width=True)

            st.subheader("📋 Detalle completo")
            all_cols = ["unit_id"] + \
                       [c for c in last_cycle.columns if c.startswith("rul_") and c != "rul_real"] + \
                       ["predicted_rul"]
            if "rul_real" in last_cycle.columns:
                all_cols += ["rul_real", "error_abs", "correct"]
            all_cols += ["status"]
            all_cols = [c for c in all_cols if c in last_cycle.columns]
            st.dataframe(
                last_cycle[all_cols].sort_values("predicted_rul").reset_index(drop=True),
                use_container_width=True)
            st.download_button(
                "📥 Descargar CSV",
                last_cycle[all_cols].to_csv(index=False),
                "predicciones_rul.csv", "text/csv")

        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")
            st.exception(e)

    else:
        st.markdown("""
        ### Como usar esta app
        1. Selecciona el **sub-dataset** en el sidebar (`FD001` a `FD004`)
        2. Elige el **modo de prediccion** (ML Clasico, DL individual o Ensemble)
        3. Subi el **archivo de test** (ej: `test_FD001.txt`)
        4. Si existe `RUL_FD001.txt` en `data/raw/`, la app compara automaticamente
           predicciones vs realidad con matriz de confusion y metricas completas

        ### Modos de prediccion
        | Modo | Velocidad | Precision |
        |------|-----------|-----------|
        | ML Clasico | ⚡ Instantaneo | Buena |
        | Deep Learning | 🔄 ~5s | Muy buena |
        | Ensemble DL | 🔄 ~15s | Mejor |
        """)

with tab2:
    st.header("📊 ML Clasico — Resultados")
    classical_data = {
        "Modelo":    ["Linear Regression", "SVR", "Random Forest", "XGBoost"],
        "RMSE":      [21.91, 19.61, 17.92, 18.31],
        "MAE":       [17.61, 13.82, 12.92, 13.39],
        "R²":        [0.722, 0.777, 0.814, 0.806],
        "NASA Score":[1318.9, 1624.9, 886.4, 1058.1],
    }
    classical_df = pd.DataFrame(classical_data)
    metric = st.selectbox("Metrica", ["RMSE", "MAE", "R²", "NASA Score"])
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(classical_df, use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(
            classical_df.sort_values(metric, ascending=(metric != "R²")),
            x="Modelo", y=metric, color="Modelo",
            title=f"{metric} — FD001", template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Set2),
            use_container_width=True)
    st.subheader("Clasificacion binaria — FD001")
    st.dataframe(pd.DataFrame({
        "Modelo":    ["Random Forest", "XGBoost"],
        "Accuracy":  [0.910, 0.910],
        "Precision": [0.900, 0.900],
        "Recall":    [0.720, 0.720],
        "F1":        [0.800, 0.800],
        "AUC-ROC":   [0.983, 0.980],
    }), use_container_width=True)

with tab3:
    st.header("🧠 Deep Learning — Resultados")
    dl_data = {
        "Dataset": ["FD001"] * 5 + ["FD002"] * 5 + ["FD003"] * 5 + ["FD004"] * 5,
        "Modelo":  ["LSTM", "GRU", "TCN", "Transformer", "Ensemble"] * 4,
        "RMSE":    [14.24, 14.89, 14.39, 13.16, 11.97,
                    12.80, 14.86, 13.43, 14.28, 11.95,
                    15.56, 13.71, 16.82, 13.34, 13.46,
                    17.92, 18.66, 21.48, 17.77, 17.32],
        "R²":      [0.873, 0.861, 0.870, 0.892, 0.910,
                    0.892, 0.855, 0.881, 0.866, 0.906,
                    0.860, 0.891, 0.836, 0.897, 0.895,
                    0.824, 0.809, 0.747, 0.827, 0.835],
    }
    dl_df       = pd.DataFrame(dl_data)
    selected_ds = st.selectbox("Dataset", DATASETS, key="dl_ds")
    filtered    = dl_df[dl_df["Dataset"] == selected_ds]
    c1, c2      = st.columns(2)
    with c1:
        st.dataframe(filtered[["Modelo", "RMSE", "R²"]].reset_index(drop=True),
                     use_container_width=True)
    with c2:
        colors_dl = {"LSTM": "#636EFA", "GRU": "#EF553B", "TCN": "#00CC96",
                     "Transformer": "#AB63FA", "Ensemble": "#FFA15A"}
        st.plotly_chart(px.bar(filtered, x="Modelo", y="RMSE", color="Modelo",
                               title=f"RMSE — {selected_ds}", template="plotly_white",
                               color_discrete_map=colors_dl),
                        use_container_width=True)
    st.subheader("Heatmap — Todos los datasets")
    hm_metric = st.radio("Metrica", ["RMSE", "R²"], horizontal=True)
    pivot = dl_df.pivot(index="Modelo", columns="Dataset", values=hm_metric)
    st.plotly_chart(px.imshow(
        pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        color_continuous_scale="RdYlGn_r" if hm_metric == "RMSE" else "RdYlGn",
        title=f"{hm_metric} — Modelos x Datasets", text_auto=".2f"),
        use_container_width=True)

with tab4:
    st.header("🔍 SHAP — Interpretabilidad")
    shap_data = {
        "Sensor":      ["sensor_11", "sensor_9", "sensor_4", "sensor_14", "sensor_12"],
        "Descripcion": ["Temp HPC", "Presion HPC", "Temp LPC", "Presion fan", "Vel fan"],
        "SHAP XGB":    [9.85, 7.14, 5.88, 4.31, 4.11],
        "SHAP RF":     [15.83, 8.91, 4.73, 2.10, 3.60],
    }
    shap_df = pd.DataFrame(shap_data)
    c1, c2  = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(shap_df, x="SHAP XGB", y="Sensor", orientation="h",
                               title="SHAP — XGBoost", template="plotly_white",
                               color="SHAP XGB", color_continuous_scale="OrRd"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(shap_df, x="SHAP RF", y="Sensor", orientation="h",
                               title="SHAP — Random Forest", template="plotly_white",
                               color="SHAP RF", color_continuous_scale="OrRd"),
                        use_container_width=True)
    st.dataframe(shap_df, use_container_width=True)
    st.markdown("""
    **Hallazgos principales:**
    - `sensor_11` (Temperatura HPC) es el indicador de degradacion mas importante
    - `sensor_4` (Temperatura LPC) es consistentemente el segundo mas relevante
    - Los mismos sensores son importantes tanto en XGBoost como en Random Forest,
      lo que valida que el hallazgo es robusto y no depende del algoritmo
    """)

with tab5:
    st.header("ℹ️ Acerca del proyecto")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
        ### NASA C-MAPSS RUL Prediction
        Estudio comparativo de ML vs Deep Learning para prediccion de vida util remanente
        en motores turbofan usando el dataset NASA C-MAPSS.

        **Dataset:** 4 sub-datasets (FD001-FD004) | 21 sensores | Run-to-failure

        **Pipeline completo:**
        EDA → Preprocessing → Feature Engineering →
        ML Clasico → Deep Learning (LSTM / GRU / TCN / Transformer) →
        Optuna (800 trials) → Ensemble → SHAP → Web App

        ---
        **Universidad LEAD** — BCD-6210 Advanced Data Mining — IC 2026

        Dr. Juan Murillo-Morera
        """)
    with c2:
        st.metric("Mejor RMSE",     "11.95", "FD002 Ensemble")
        st.metric("Mejor R²",       "0.910", "FD001 Ensemble")
        st.metric("Sensor clave",   "sensor_11 (HPC)")
        st.metric("Optimizacion",   "Optuna — 800 trials")
        st.metric("Framework DL",   "PyTorch")
        st.metric("Clasificacion",  "AUC = 0.983")
