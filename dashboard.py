import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from main import (
    preparar_df, evaluar_modelo, pronosticar, 
    comparar_modelos, analisis_por_dia, 
    intervalos_confianza, seleccionar_mejor_modelo
)
import logging
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Contact Point 360 | Forecast Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CSS CORPORATITO (BLANCO/NEGRO/AZUL) ----------
st.markdown("""
<style>
    /* Fuentes */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Fondo principal */
    .stApp {
        background-color: #f5f5f5;
    }
    
    /* Header corporativo */
    .header-corporate {
        background-color: #ffffff;
        padding: 1.5rem 2rem;
        border-bottom: 3px solid #0066b3;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    .logo-text {
        font-size: 1.8rem;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0;
    }
    
    .logo-subtext {
        font-size: 0.9rem;
        color: #666666;
        margin: 0;
    }
    
    .accent-bar {
        height: 3px;
        background: linear-gradient(90deg, #0066b3 0%, #004080 100%);
        width: 100%;
    }
    
    /* Tarjetas corporativas */
    .card-corporate {
        background: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border: 1px solid #eaeaea;
        transition: all 0.2s ease;
    }
    
    .card-corporate:hover {
        box-shadow: 0 4px 12px rgba(0,102,179,0.1);
        border-color: #0066b3;
    }
    
    .card-title {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #666666;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .card-value {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a1a;
        line-height: 1.2;
    }
    
    .card-footer {
        font-size: 0.8rem;
        color: #888888;
        margin-top: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Badges */
    .badge-corporate {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background-color: #f0f7ff;
        color: #0066b3;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        border: 1px solid #cce4ff;
    }
    
    .badge-success {
        background-color: #e6f7e6;
        color: #1a7d1a;
        border-color: #b8e0b8;
    }
    
    .badge-warning {
        background-color: #fff4e6;
        color: #b35900;
        border-color: #ffd6b8;
    }
    
    /* Botón principal */
    .btn-primary-corporate {
        background-color: #0066b3;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.95rem;
        cursor: pointer;
        width: 100%;
        transition: all 0.2s;
        border: 1px solid #0066b3;
    }
    
    .btn-primary-corporate:hover {
        background-color: #004080;
        border-color: #004080;
    }
    
    /* Sidebar corporativa - MODIFICADA */
    .sidebar-corporate {
        background: #ffffff;
        border-right: 1px solid #eaeaea;
    }
    
    /* Divisores */
    .divider-corporate {
        height: 1px;
        background: linear-gradient(90deg, transparent, #eaeaea, transparent);
        margin: 2rem 0;
    }
    
    /* Tabs corporativos */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: transparent;
        padding: 0;
        border-bottom: 1px solid #eaeaea;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        font-weight: 500;
        color: #666666;
        padding: 0.75rem 1.5rem;
        background: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #0066b3 !important;
        border-bottom: 2px solid #0066b3;
    }
    
    /* Selectores */
    .stSelectbox label {
        color: #1a1a1a !important;
        font-weight: 500;
        font-size: 0.85rem;
    }
    
    /* Tablas */
    .dataframe {
        border: 1px solid #eaeaea;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Footer */
    .footer-corporate {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #eaeaea;
        color: #888888;
        font-size: 0.8rem;
    }
    
    /* KPI indicator */
    .kpi-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        background: #f5f5f5;
        border-radius: 4px;
        font-size: 0.75rem;
        color: #666666;
    }
</style>
""", unsafe_allow_html=True)

# ---------- HEADER CORPORATIVO ----------
st.markdown("""
<div class="header-corporate">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <h1 class="logo-text">CONTACT POINT 360</h1>
            <p class="logo-subtext">Forecast Analytics · Intelligent Call Center Prediction</p>
        </div>
        <div class="badge-corporate">
            v2.0 · Producción
        </div>
    </div>
</div>
<div class="accent-bar"></div>
""", unsafe_allow_html=True)

# ---------- SIDEBAR MODIFICADA (SOLO ESTA PARTE CAMBIA) ----------
with st.sidebar:
    st.markdown("""
    <div style="padding: 0rem 0rem 1rem 0rem; border-bottom: 2px solid #0066b3;">
        <h3 style="color: #000000; margin:0; font-weight: 600;">⚙️ Panel de Control</h3>
        <p style="color: #666666; margin:0.25rem 0 0 0; font-size: 0.85rem;">Configuración del análisis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Carga de archivos
    st.markdown("#### 📂 Datos de Entrada")
    file = st.file_uploader(
        "Seleccionar archivo CSV",
        type=["csv"],
        label_visibility="collapsed",
        help="Formato: fecha (YYYY-MM-DD), volumen (número)"
    )
    
    if file:
        st.markdown(f"""
        <div style="background: #f5f5f5; padding: 0.75rem; border-radius: 4px; margin: 0.5rem 0; border-left: 3px solid #0066b3;">
            <small><strong>{file.name}</strong> · {file.size/1024:.1f}KB</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 1.5rem 0; border-color: #eaeaea;'>", unsafe_allow_html=True)
    
    # Parámetros
    st.markdown("#### 🎯 Parámetros")
    
    dias = st.number_input(
        "Días a pronosticar",
        min_value=7,
        max_value=90,
        value=30,
        help="Período de proyección en días"
    )
    
    confianza = st.selectbox(
        "Nivel de confianza",
        options=[90, 95, 99],
        index=1,
        help="Intervalo de confianza para las predicciones"
    )
    
    st.markdown("<hr style='margin: 1.5rem 0; border-color: #eaeaea;'>", unsafe_allow_html=True)
    
    # Configuración avanzada
    with st.expander("⚙️ Configuración avanzada", expanded=False):
        detectar_atipicos = st.checkbox("Detectar outliers", value=True)
        incluir_festivos = st.checkbox("Incluir festivos", value=True)
        
        st.markdown("**Modelos activos:**")
        modelos_activos = {
            "ARIMA/SARIMA": st.checkbox("ARIMA/SARIMA", value=True),
            "Holt-Winters": st.checkbox("Holt-Winters", value=True),
            "Prophet": st.checkbox("Prophet", value=True),
            "XGBoost": st.checkbox("XGBoost", value=True),
            "LSTM": st.checkbox("LSTM", value=True)
        }
    
    st.markdown("""
    <div style="background: #f5f5f5; padding: 1rem; border-radius: 4px; margin-top: 2rem;">
        <small style="color: #666666;">
            <strong>📌 Contact Point 360</strong><br>
            Análisis predictivo para optimización de operaciones en contact centers.
        </small>
    </div>
    """, unsafe_allow_html=True)

# ---------- MAIN CONTENT ----------
if file is not None:
    
    # Cargar datos
    with st.spinner("📊 Procesando datos..."):
        df = pd.read_csv(file)
        df = preparar_df(df, detectar_atipicos)
        mae, rmse = evaluar_modelo(df)
    
    # ---------- RESUMEN EJECUTIVO ----------
    st.markdown("#### 📊 Resumen Ejecutivo")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="card-corporate">
            <div class="card-title">📅 Registros históricos</div>
            <div class="card-value">{len(df):,}</div>
            <div class="card-footer">
                <span class="badge-corporate">{len(df)//365}+ años</span>
                <span>datos disponibles</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="card-corporate">
            <div class="card-title">📞 Volumen promedio</div>
            <div class="card-value">{df['volumen'].mean():.0f}</div>
            <div class="card-footer">
                <span class="badge-corporate">llamadas/día</span>
                <span>pico máx: {df['volumen'].max():.0f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="card-corporate">
            <div class="card-title">📊 MAE Referencia</div>
            <div class="card-value">{mae:.1f}</div>
            <div class="card-footer">
                <span class="badge-warning">error absoluto</span>
                <span>±{rmse:.1f} RMSE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        modelos_count = sum(modelos_activos.values())
        st.markdown(f"""
        <div class="card-corporate">
            <div class="card-title">🤖 Modelos activos</div>
            <div class="card-value">{modelos_count}/5</div>
            <div class="card-footer">
                <span class="badge-success">listos para entrenar</span>
                <span>ML ensemble</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='divider-corporate'></div>", unsafe_allow_html=True)
    
    # ---------- SECCIÓN DE PRONÓSTICO ----------
    col_left, col_right = st.columns([1.2, 0.8])
    
    with col_left:
        st.markdown("#### 🔮 Generar Pronóstico")
        
        generar_btn = st.button(
            "🚀 Ejecutar pronóstico",
            type="primary",
            use_container_width=True
        )
        
        if generar_btn:
            with st.spinner("⏳ Entrenando modelos..."):
                df_forecast = pronosticar(df, dias, modelos_activos)
                df_forecast = intervalos_confianza(df_forecast)
                st.session_state['df_forecast'] = df_forecast
                st.session_state['pronostico_generado'] = True
                st.success("✅ Pronóstico completado")
    
    with col_right:
        if st.session_state.get('pronostico_generado', False):
            comparacion = comparar_modelos(df, st.session_state['df_forecast'])
            mejor_modelo = seleccionar_mejor_modelo(df, st.session_state['df_forecast'])
            
            st.markdown(f"""
            <div style="background: #f0f7ff; border: 1px solid #cce4ff; border-radius: 8px; padding: 1.5rem;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <span style="font-size: 2rem;">🏆</span>
                    <div>
                        <small style="color: #666666;">MEJOR MODELO</small>
                        <h3 style="margin:0; color: #0066b3;">{mejor_modelo}</h3>
                        <small style="color: #888888;">MAE: {comparacion.loc[mejor_modelo, 'MAE']:.1f}</small>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    if st.session_state.get('pronostico_generado', False):
        df_forecast = st.session_state['df_forecast']
        
        # Tabs corporativos
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Visualización", 
            "📊 Comparativa", 
            "📋 Datos",
            "📥 Exportar"
        ])
        
        with tab1:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                modelos_disponibles = [col for col in df_forecast.columns 
                                      if col not in ['fecha', 'dow', 'is_weekend', 
                                                   'is_holiday', 'mes', 'dia_mes',
                                                   'promedio_modelos', 'upper_90', 
                                                   'lower_90', 'upper_95', 'lower_95']]
                
                modelo_sel = st.selectbox(
                    "Seleccionar modelo:",
                    ["Ensemble (promedio)", f"Mejor modelo ({mejor_modelo})"] + modelos_disponibles,
                    label_visibility="collapsed"
                )
            
            with col2:
                mostrar_ic = st.checkbox("Mostrar IC 95%", value=True)
            
            # Gráfico corporativo
            fig = go.Figure()
            
            # Histórico
            fig.add_trace(go.Scatter(
                x=df['fecha'],
                y=df['volumen'],
                mode='lines',
                name='Histórico',
                line=dict(color='#1a1a1a', width=2),
                hovertemplate='<b>%{x}</b><br>Volumen: %{y}<extra></extra>'
            ))
            
            # Pronóstico
            if modelo_sel == "Ensemble (promedio)" and 'promedio_modelos' in df_forecast.columns:
                if mostrar_ic:
                    fig.add_trace(go.Scatter(
                        x=df_forecast['fecha'],
                        y=df_forecast['upper_95'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='none'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=df_forecast['fecha'],
                        y=df_forecast['lower_95'],
                        mode='lines',
                        fill='tonexty',
                        fillcolor='rgba(0,102,179,0.1)',
                        line=dict(width=0),
                        name='IC 95%',
                        hoverinfo='none'
                    ))
                
                fig.add_trace(go.Scatter(
                    x=df_forecast['fecha'],
                    y=df_forecast['promedio_modelos'],
                    mode='lines+markers',
                    name='Ensemble',
                    line=dict(color='#0066b3', width=2.5),
                    marker=dict(size=4, color='#0066b3'),
                    hovertemplate='<b>%{x}</b><br>Ensemble: %{y}<extra></extra>'
                ))
            
            elif modelo_sel == f"Mejor modelo ({mejor_modelo})":
                fig.add_trace(go.Scatter(
                    x=df_forecast['fecha'],
                    y=df_forecast[mejor_modelo],
                    mode='lines+markers',
                    name=mejor_modelo,
                    line=dict(color='#004080', width=2.5, dash='dash'),
                    marker=dict(size=4, color='#004080'),
                    hovertemplate=f'<b>%{{x}}</b><br>{mejor_modelo}: %{{y}}<extra></extra>'
                ))
            
            else:
                if modelo_sel in df_forecast.columns:
                    fig.add_trace(go.Scatter(
                        x=df_forecast['fecha'],
                        y=df_forecast[modelo_sel],
                        mode='lines+markers',
                        name=modelo_sel,
                        line=dict(color='#666666', width=2, dash='dot'),
                        marker=dict(size=4),
                        hovertemplate=f'<b>%{{x}}</b><br>{modelo_sel}: %{{y}}<extra></extra>'
                    ))
            
            fig.update_layout(
                template='none',
                height=450,
                margin=dict(l=40, r=40, t=40, b=40),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=12)
                ),
                hovermode='x unified',
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            fig.update_xaxes(
                gridcolor='#eaeaea',
                linecolor='#eaeaea',
                tickfont=dict(size=11)
            )
            
            fig.update_yaxes(
                gridcolor='#eaeaea',
                linecolor='#eaeaea',
                tickfont=dict(size=11)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Métricas rápidas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="card-corporate" style="padding: 1rem;">
                    <div style="font-size: 0.8rem; color: #666666;">Período pronóstico</div>
                    <div style="font-size: 1.2rem; font-weight: 600;">
                        {df_forecast['fecha'].min().strftime('%d/%m/%Y')} - {df_forecast['fecha'].max().strftime('%d/%m/%Y')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if 'promedio_modelos' in df_forecast.columns:
                    st.markdown(f"""
                    <div class="card-corporate" style="padding: 1rem;">
                        <div style="font-size: 0.8rem; color: #666666;">Promedio proyectado</div>
                        <div style="font-size: 1.2rem; font-weight: 600;">
                            {df_forecast['promedio_modelos'].mean():.0f} llamadas/día
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="card-corporate" style="padding: 1rem;">
                    <div style="font-size: 0.8rem; color: #666666;">Volumen total estimado</div>
                    <div style="font-size: 1.2rem; font-weight: 600;">
                        {df_forecast['promedio_modelos'].sum():.0f} llamadas
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            comparacion = comparar_modelos(df, df_forecast)
            
            if not comparacion.empty:
                # Gráfico de comparación
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('MAE por Modelo', 'RMSE por Modelo'),
                    specs=[[{'type': 'bar'}, {'type': 'bar'}]]
                )
                
                fig.add_trace(
                    go.Bar(
                        x=comparacion.index, 
                        y=comparacion['MAE'],
                        name='MAE',
                        marker_color='#0066b3',
                        text=comparacion['MAE'].round(1),
                        textposition='outside'
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=comparacion.index, 
                        y=comparacion['RMSE'],
                        name='RMSE',
                        marker_color='#1a1a1a',
                        text=comparacion['RMSE'].round(1),
                        textposition='outside'
                    ),
                    row=1, col=2
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white'
                )
                
                fig.update_xaxes(tickangle=45, gridcolor='#eaeaea')
                fig.update_yaxes(gridcolor='#eaeaea')
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Tabla de rendimiento
                st.markdown("##### 📊 Tabla Comparativa")
                
                comparacion_display = comparacion.copy()
                comparacion_display['MAE'] = comparacion_display['MAE'].round(1)
                comparacion_display['RMSE'] = comparacion_display['RMSE'].round(1)
                comparacion_display['Precisión (%)'] = (100 - (comparacion_display['MAE'] / df['volumen'].mean() * 100)).round(1)
                
                st.dataframe(
                    comparacion_display.style.highlight_min(subset=['MAE'], color='#e6f3ff'),
                    use_container_width=True
                )
        
        with tab3:
            st.markdown("##### 📋 Datos del Pronóstico")
            
            col1, col2 = st.columns(2)
            with col1:
                fecha_inicio = st.date_input(
                    "Fecha inicio",
                    value=df_forecast['fecha'].min(),
                    min_value=df_forecast['fecha'].min(),
                    max_value=df_forecast['fecha'].max()
                )
            
            with col2:
                fecha_fin = st.date_input(
                    "Fecha fin",
                    value=df_forecast['fecha'].max(),
                    min_value=df_forecast['fecha'].min(),
                    max_value=df_forecast['fecha'].max()
                )
            
            # Filtrar datos
            mask = (df_forecast['fecha'].dt.date >= fecha_inicio) & \
                   (df_forecast['fecha'].dt.date <= fecha_fin)
            df_filtrado = df_forecast[mask]
            
            # Columnas a mostrar
            cols_mostrar = ['fecha']
            if 'promedio_modelos' in df_filtrado.columns:
                cols_mostrar.extend(['promedio_modelos', 'lower_95', 'upper_95'])
            
            st.dataframe(
                df_filtrado[cols_mostrar].style.format({
                    'promedio_modelos': '{:.0f}',
                    'lower_95': '{:.0f}',
                    'upper_95': '{:.0f}'
                }),
                use_container_width=True,
                height=400
            )
        
        with tab4:
            st.markdown("##### 📥 Exportar Resultados")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV
                if 'promedio_modelos' in df_forecast.columns:
                    df_export = df_forecast[['fecha', 'promedio_modelos', 'lower_95', 'upper_95']].copy()
                    nombre = "forecast_contact_point_360.csv"
                else:
                    df_export = df_forecast[['fecha'] + modelos_disponibles].copy()
                    nombre = "forecast_completo.csv"
                
                df_export['fecha'] = df_export['fecha'].dt.strftime('%Y-%m-%d')
                
                st.download_button(
                    "📥 Descargar CSV",
                    df_export.to_csv(index=False).encode('utf-8'),
                    nombre,
                    "text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Excel
                try:
                    with pd.ExcelWriter('forecast.xlsx', engine='openpyxl') as writer:
                        df_export.to_excel(writer, sheet_name='Pronóstico', index=False)
                        if not comparacion.empty:
                            comparacion.to_excel(writer, sheet_name='Rendimiento')
                    
                    with open('forecast.xlsx', 'rb') as f:
                        st.download_button(
                            "📥 Descargar Excel",
                            f,
                            nombre.replace('.csv', '.xlsx'),
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                except:
                    st.info("pip install openpyxl para exportar Excel")

else:
    # Landing page
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem; background: white; border-radius: 8px; border: 1px solid #eaeaea;">
        <img src="https://via.placeholder.com/150x150/0066b3/ffffff?text=CP360" style="width: 100px; margin-bottom: 2rem;">
        <h2 style="color: #1a1a1a; margin-bottom: 1rem;">Contact Point 360</h2>
        <p style="color: #666666; max-width: 500px; margin: 0 auto 2rem auto;">
            Plataforma de análisis predictivo para optimización de operaciones en contact centers.
            Cargue su archivo CSV para comenzar.
        </p>
        <div style="display: flex; gap: 2rem; justify-content: center; color: #888888;">
            <div>📅 Datos históricos</div>
            <div>🤖 5 modelos ML</div>
            <div>📈 Visualizaciones</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer-corporate">
    <span>CONTACT POINT 360 · Forecast Analytics</span>
    <span style="margin: 0 1rem;">|</span>
    <span>v2.0 · Desarrollado para optimización de contact centers</span>
</div>
""", unsafe_allow_html=True)