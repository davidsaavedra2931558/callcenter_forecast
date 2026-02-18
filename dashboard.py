import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from main import preparar_df, evaluar_modelo, pronosticar

st.set_page_config(
    page_title="Call Center Forecast",
    page_icon="📞",
    layout="wide"
)

# ---------- ESTILO PROFESIONAL ----------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
}

.metric-card {
    background: linear-gradient(145deg, #1f2937, #111827);
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    color: white;
    box-shadow: 0 6px 18px rgba(0,0,0,0.4);
    transition: 0.3s;
}

.metric-card:hover {
    transform: translateY(-5px);
}

.metric-title {
    font-size: 14px;
    color: #9CA3AF;
}

.metric-value {
    font-size: 32px;
    font-weight: bold;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<h1 style='text-align:center;'>📞 Call Center Forecast</h1>
<p style='text-align:center; color:gray; font-size:18px;'>
Predicción inteligente con múltiples modelos de Machine Learning
</p>
""", unsafe_allow_html=True)

st.divider()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("⚙️ Configuración")
    st.markdown("---")
    file = st.file_uploader("📂 Sube tu histórico (CSV)", type=["csv"])
    dias = st.slider("📅 Días a pronosticar", 7, 90, 30)
    st.markdown("---")
    st.info("Proyecto Forecast – Portafolio Data Science")

# ---------- MAIN ----------
if file:

    df = pd.read_csv(file)
    df = preparar_df(df)

    mae, rmse = evaluar_modelo(df)

    # ---------- TARJETAS ----------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MAE</div>
            <div class="metric-value">{mae:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">RMSE</div>
            <div class="metric-value">{rmse:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Modelos Activos</div>
            <div class="metric-value">5</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ---------- PRONOSTICO ----------
    st.subheader("🔮 Proyección de Llamadas")

    @st.cache_data(show_spinner=False)
    def cached_pronostico(df, dias):
        return pronosticar(df, dias)

    with st.spinner("⏳ Entrenando modelos..."):
        df_forecast = cached_pronostico(df, dias)

    modelos = [
        "ARIMA / SARIMA",
        "Holt-Winters",
        "Prophet",
        "XGBoost",
        "LSTM"
    ]

    modelo_sel = st.selectbox(
        "📌 Selecciona modelo a visualizar",
        ["Todos los modelos"] + modelos
    )

    # ---------- GRAFICO MEJORADO ----------
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        df["fecha"],
        df["volumen"],
        label="Histórico",
        linewidth=3
    )

    if modelo_sel == "Todos los modelos":
        for m in modelos:
            ax.plot(
                df_forecast["fecha"],
                df_forecast[m],
                linestyle="--",
                linewidth=2,
                label=m
            )
    else:
        ax.plot(
            df_forecast["fecha"],
            df_forecast[modelo_sel],
            linestyle="--",
            linewidth=3,
            label=modelo_sel
        )

    ax.set_title("Evolución y Proyección")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    st.divider()

    # ---------- TABLA ----------
    st.subheader("📊 Resultados Detallados")

    if modelo_sel == "Todos los modelos":
        st.dataframe(df_forecast, use_container_width=True)
    else:
        st.dataframe(
            df_forecast[["fecha", modelo_sel]],
            use_container_width=True
        )

    st.divider()

    # ---------- DESCARGA ----------
    st.subheader("⬇️ Exportar Pronóstico")

    if modelo_sel == "Todos los modelos":
        df_export = df_forecast[["fecha"] + modelos].copy()
        nombre = "forecast_completo.csv"
    else:
        df_export = df_forecast[["fecha", modelo_sel]].copy()
        nombre = f"forecast_{modelo_sel}.csv"

    df_export["fecha"] = df_export["fecha"].dt.strftime("%Y-%m-%d")

    st.download_button(
        "📥 Descargar CSV",
        df_export.to_csv(index=False).encode("utf-8"),
        nombre,
        "text/csv"
    )

else:
    st.markdown("""
    <div style='text-align:center; padding:50px;'>
        <h3>👈 Sube un archivo CSV para comenzar</h3>
        <p style='color:gray;'>El archivo debe contener columnas: fecha y volumen</p>
    </div>
    """, unsafe_allow_html=True)
