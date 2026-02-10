import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from main import preparar_df, evaluar_modelo, pronosticar

st.set_page_config(
    page_title="Call Center Forecast",
    page_icon="📞",
    layout="wide"
)

# ---------- ESTILO ----------
st.markdown("""
<style>
.metric-card {
    background-color: #0f172a;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}
.metric-title { font-size: 14px; color: #9CA3AF; }
.metric-value { font-size: 28px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown(
    "<h1 style='text-align:center;'>📞 Call Center – Forecast de Llamadas</h1>"
    "<p style='text-align:center;color:gray;'>Predicción con modelos reales</p>",
    unsafe_allow_html=True
)

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Configuración")
file = st.sidebar.file_uploader("📂 Sube tu CSV", type=["csv"])
dias = st.sidebar.slider("📅 Días a pronosticar", 7, 90, 30)

# ---------- MAIN ----------
if file:
    df = pd.read_csv(file)
    df = preparar_df(df)

    mae, rmse = evaluar_modelo(df)

    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>MAE</div>
            <div class='metric-value'>{mae:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>RMSE</div>
            <div class='metric-value'>{rmse:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>Modelos reales</div>
            <div class='metric-value'>5</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ---------- PRONOSTICO ----------
    st.subheader("🔮 Pronóstico por modelos")

    @st.cache_data(show_spinner=False)
    def cached_pronostico(df, dias):
        return pronosticar(df, dias)

    with st.spinner("⏳ Entrenando modelos, por favor espera..."):
        df_forecast = cached_pronostico(df, dias)

    modelos = [
        "ARIMA / SARIMA",
        "Holt-Winters",
        "Prophet",
        "XGBoost",
        "LSTM"
    ]

    modelos_opciones = ["Todos los modelos"] + modelos
    modelo_sel = st.selectbox("📌 Selecciona el modelo a visualizar", modelos_opciones)

    left, right = st.columns([2, 1])

    with left:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["fecha"], df["volumen"], label="Histórico", linewidth=2)

        if modelo_sel == "Todos los modelos":
            for m in modelos:
                ax.plot(df_forecast["fecha"], df_forecast[m], "--", linewidth=2, label=m)
            ax.set_title("Evolución y proyección – Todos los modelos")
        else:
            ax.plot(df_forecast["fecha"], df_forecast[modelo_sel], "--", label=modelo_sel, linewidth=2)
            ax.set_title(f"Evolución y proyección – {modelo_sel}")

        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)

    with right:
        st.markdown("### 📊 Resultados")
        if modelo_sel == "Todos los modelos":
            st.dataframe(df_forecast[["fecha"] + modelos], height=300)
        else:
            st.dataframe(df_forecast[["fecha", modelo_sel]], height=300)

    st.divider()

    # ---------- DESCARGA ----------
    st.subheader("⬇️ Exportar resultados")

    if modelo_sel == "Todos los modelos":
        df_excel = df_forecast[["fecha"] + modelos].copy()
        df_excel["fecha"] = df_excel["fecha"].dt.strftime("%Y-%m-%d")
        st.download_button(
            "📥 Descargar TODOS los modelos",
            df_excel.to_csv(index=False).encode("utf-8"),
            "pronostico_todos_los_modelos.csv",
            "text/csv"
        )
    else:
        df_excel = df_forecast[["fecha", modelo_sel]].copy()
        df_excel["fecha"] = df_excel["fecha"].dt.strftime("%Y-%m-%d")
        st.download_button(
            f"📥 Descargar {modelo_sel}",
            df_excel.to_csv(index=False).encode("utf-8"),
            f"pronostico_{modelo_sel.replace(' ','_')}.csv",
            "text/csv"
        )

    # ---------- COMPARACIÓN REAL ----------
    st.divider()
    st.subheader("📊 Comparación real vs pronóstico (últimos 30 días)")

    comp_file = st.file_uploader(
        "📂 Sube archivo real con últimos 30 días",
        type=["csv","xlsx"],
        key="comp"
    )

    if comp_file:
        real_df = pd.read_excel(comp_file) if comp_file.name.endswith("xlsx") else pd.read_csv(comp_file)
        real_df.columns = real_df.columns.str.lower()
        real_df["fecha"] = pd.to_datetime(real_df["fecha"])
        real_df["volumen"] = pd.to_numeric(real_df["volumen"])

        real_30 = real_df.sort_values("fecha").tail(30)

        if modelo_sel != "Todos los modelos":
            comp = pd.merge(
                real_30,
                df_forecast[["fecha", modelo_sel]],
                on="fecha",
                how="inner"
            )

            comp["error_%"] = (
                abs(comp["volumen"] - comp[modelo_sel]) / comp["volumen"] * 100
            ).round(2)

            col1, col2 = st.columns([2,1])

            with col1:
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(comp["fecha"], comp["volumen"], label="Real", linewidth=2)
                ax.plot(comp["fecha"], comp[modelo_sel], "--", label="Pronóstico", linewidth=2)
                ax.set_title("Real vs Pronóstico (30 días)")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)

            with col2:
                st.metric("📉 Error promedio (%)", f"{comp['error_%'].mean():.2f}%")
                st.dataframe(comp, height=300)
        else:
            st.warning("Selecciona un solo modelo para comparar.")

else:
    st.info("👈 Sube un archivo CSV para iniciar el análisis.")
