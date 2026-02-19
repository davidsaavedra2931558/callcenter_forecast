import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import holidays
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    filename=f'forecast_log_{datetime.now().strftime("%Y%m%d")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Días festivos de Colombia
co_holidays = holidays.Colombia()

# ---------------- PREPARAR DATA ----------------
def preparar_df(df, detectar_atipicos=True, umbral_atipicos=2.5):
    """
    Prepara y limpia el dataframe para el análisis
    """
    try:
        df = df.copy()
        df.columns = df.columns.str.lower().str.strip()
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["volumen"] = pd.to_numeric(df["volumen"], errors="coerce")
        df = df.dropna().sort_values("fecha").reset_index(drop=True)

        # Features temporales básicas
        df["dow"] = df["fecha"].dt.weekday
        df["is_weekend"] = df["dow"].isin([5,6]).astype(int)
        df["is_holiday"] = df["fecha"].isin(co_holidays).astype(int)
        
        # Features adicionales
        df["mes"] = df["fecha"].dt.month
        df["dia_mes"] = df["fecha"].dt.day
        df["semana_anio"] = df["fecha"].dt.isocalendar().week.astype(int)
        df["mes_verano"] = df["mes"].isin([6,7,8]).astype(int)  # Meses de verano
        df["finde_mes"] = df["fecha"].dt.is_month_end.astype(int)
        df["inicio_mes"] = (df["dia_mes"] <= 3).astype(int)
        
        # Identificar días atípicos si se solicita
        if detectar_atipicos:
            df = identificar_dias_atipicos(df, umbral_atipicos)
        
        logging.info(f"Data preparada correctamente: {len(df)} registros")
        return df
        
    except Exception as e:
        logging.error(f"Error en preparar_df: {str(e)}")
        raise e

def identificar_dias_atipicos(df, umbral=2.5):
    """
    Identifica días con volumen anormalmente alto/bajo
    """
    df = df.copy()
    media = df["volumen"].mean()
    std = df["volumen"].std()
    
    df["es_atipico"] = (np.abs(df["volumen"] - media) > umbral * std).astype(int)
    df["tipo_atipico"] = np.where(
        df["volumen"] > media + umbral * std, "pico",
        np.where(df["volumen"] < media - umbral * std, "valle", "normal")
    )
    return df

# ---------------- METRICA SIMPLE ----------------
def evaluar_modelo(df):
    """
    Evalúa un modelo lineal simple como referencia
    """
    try:
        X = np.arange(len(df)).reshape(-1, 1)
        y = df["volumen"].values
        model = LinearRegression().fit(X, y)
        pred = model.predict(X)
        mae = mean_absolute_error(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        
        logging.info(f"Evaluación base - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        return mae, rmse
        
    except Exception as e:
        logging.error(f"Error en evaluar_modelo: {str(e)}")
        return 0, 0

# ---------------- FEATURES ----------------
def crear_features_lag(df):
    """
    Crea features de lag y medias móviles para modelos ML
    """
    df = df.copy()
    df["lag_1"] = df["volumen"].shift(1)
    df["lag_2"] = df["volumen"].shift(2)
    df["lag_3"] = df["volumen"].shift(3)
    df["lag_7"] = df["volumen"].shift(7)
    df["lag_14"] = df["volumen"].shift(14)
    df["lag_21"] = df["volumen"].shift(21)
    df["lag_28"] = df["volumen"].shift(28)
    
    # Medias móviles
    df["roll_7"] = df["volumen"].rolling(7, min_periods=1).mean()
    df["roll_14"] = df["volumen"].rolling(14, min_periods=1).mean()
    df["roll_30"] = df["volumen"].rolling(30, min_periods=1).mean()
    
    # Features temporales
    df["dow"] = df["fecha"].dt.weekday
    df["is_weekend"] = df["dow"].isin([5,6]).astype(int)
    df["is_holiday"] = df["fecha"].isin(co_holidays).astype(int)
    df["mes"] = df["fecha"].dt.month
    df["dia_mes"] = df["fecha"].dt.day
    
    return df.dropna()

# ---------------- PRONOSTICO ----------------
def pronosticar(df, dias, modelos_activos=None):
    """
    Genera pronósticos usando múltiples modelos
    """
    if modelos_activos is None:
        modelos_activos = {
            "ARIMA/SARIMA": True,
            "Holt-Winters": True,
            "Prophet": True,
            "XGBoost": True,
            "LSTM": True
        }
    
    try:
        serie = df.set_index("fecha")["volumen"]
        
        fechas_futuras = pd.date_range(
            serie.index[-1] + pd.Timedelta(days=1),
            periods=dias, freq="D"
        )
        
        resultados = {"fecha": fechas_futuras}
        
        # 1. ARIMA/SARIMA
        if modelos_activos.get("ARIMA/SARIMA", True):
            try:
                arima_pred = SARIMAX(
                    serie, 
                    order=(1,1,1), 
                    seasonal_order=(1,1,1,7),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                ).fit(disp=False).forecast(dias)
                resultados["ARIMA / SARIMA"] = arima_pred.round().astype(int)
                logging.info("ARIMA/SARIMA completado")
            except Exception as e:
                logging.error(f"Error en ARIMA: {str(e)}")
                resultados["ARIMA / SARIMA"] = [serie.mean()] * dias
        
        # 2. Holt-Winters
        if modelos_activos.get("Holt-Winters", True):
            try:
                hw_model = ExponentialSmoothing(
                    serie, 
                    trend="add", 
                    seasonal="add", 
                    seasonal_periods=7,
                    initialization_method="estimated"
                ).fit()
                hw_pred = hw_model.forecast(dias)
                resultados["Holt-Winters"] = hw_pred.round().astype(int)
                logging.info("Holt-Winters completado")
            except Exception as e:
                logging.error(f"Error en Holt-Winters: {str(e)}")
                resultados["Holt-Winters"] = [serie.mean()] * dias
        
        # 3. Prophet
        if modelos_activos.get("Prophet", True):
            try:
                # Preparar festivos para Prophet
                festivos = pd.DataFrame({
                    "ds": pd.to_datetime(list(co_holidays.keys())),
                    "holiday": "CO"
                })
                
                df_p = df.rename(columns={"fecha": "ds", "volumen": "y"})
                p = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False,
                    holidays=festivos,
                    seasonality_mode='multiplicative'
                )
                p.fit(df_p)
                
                future = p.make_future_dataframe(periods=dias)
                forecast = p.predict(future)
                prophet_pred = forecast["yhat"].tail(dias).values
                resultados["Prophet"] = prophet_pred.round().astype(int)
                logging.info("Prophet completado")
            except Exception as e:
                logging.error(f"Error en Prophet: {str(e)}")
                resultados["Prophet"] = [serie.mean()] * dias
        
        # 4. XGBoost
        if modelos_activos.get("XGBoost", True):
            try:
                df_lag = crear_features_lag(df)
                feature_cols = ["lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_21", "lag_28",
                               "roll_7", "roll_14", "roll_30", "dow", "is_weekend", "is_holiday", 
                               "mes", "dia_mes"]
                
                X = df_lag[feature_cols]
                y = df_lag["volumen"]
                
                model_xgb = xgb.XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
                model_xgb.fit(X, y)
                
                # Predicción recursiva
                last_row = df_lag.iloc[-1]
                last_features = last_row[feature_cols].values.tolist()
                xgb_preds = []
                
                for i in range(dias):
                    pred = model_xgb.predict(np.array(last_features).reshape(1, -1))[0]
                    xgb_preds.append(max(0, pred))  # No negativos
                    
                    # Actualizar features para siguiente predicción
                    new_features = last_features.copy()
                    # Actualizar lags
                    new_features[0] = pred  # lag_1
                    new_features[1] = last_features[0]  # lag_2
                    new_features[2] = last_features[1]  # lag_3
                    new_features[3] = last_features[2]  # lag_7
                    new_features[4] = last_features[3]  # lag_14
                    new_features[5] = last_features[4]  # lag_21
                    new_features[6] = last_features[5]  # lag_28
                    
                    # Actualizar medias móviles
                    ultimos_7 = xgb_preds[-7:] if len(xgb_preds) >= 7 else [pred] * 7
                    ultimos_14 = xgb_preds[-14:] if len(xgb_preds) >= 14 else [pred] * 14
                    ultimos_30 = xgb_preds[-30:] if len(xgb_preds) >= 30 else [pred] * 30
                    
                    new_features[7] = np.mean(ultimos_7)  # roll_7
                    new_features[8] = np.mean(ultimos_14)  # roll_14
                    new_features[9] = np.mean(ultimos_30)  # roll_30
                    
                    # Actualizar fecha
                    nueva_fecha = fechas_futuras[i]
                    new_features[10] = nueva_fecha.weekday()  # dow
                    new_features[11] = 1 if nueva_fecha.weekday() in [5,6] else 0  # is_weekend
                    new_features[12] = 1 if nueva_fecha in co_holidays else 0  # is_holiday
                    new_features[13] = nueva_fecha.month  # mes
                    new_features[14] = nueva_fecha.day  # dia_mes
                    
                    last_features = new_features
                
                resultados["XGBoost"] = np.array(xgb_preds).round().astype(int)
                logging.info("XGBoost completado")
            except Exception as e:
                logging.error(f"Error en XGBoost: {str(e)}")
                resultados["XGBoost"] = [serie.mean()] * dias
        
        # 5. LSTM
        if modelos_activos.get("LSTM", True):
            try:
                # Preparar datos para LSTM
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(serie.values.reshape(-1, 1))
                
                # Crear secuencias
                sequence_length = 30
                X_lstm, y_lstm = [], []
                
                for i in range(sequence_length, len(scaled_data)):
                    X_lstm.append(scaled_data[i-sequence_length:i])
                    y_lstm.append(scaled_data[i])
                
                X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
                
                if len(X_lstm) > 0:
                    # Construir modelo LSTM
                    model = Sequential([
                        LSTM(64, return_sequences=True, input_shape=(sequence_length, 1)),
                        Dropout(0.2),
                        LSTM(32, return_sequences=False),
                        Dropout(0.2),
                        Dense(16, activation='relu'),
                        Dense(1)
                    ])
                    
                    model.compile(optimizer='adam', loss='mse')
                    
                    # Entrenar
                    model.fit(
                        X_lstm, y_lstm,
                        epochs=20,
                        batch_size=16,
                        validation_split=0.1,
                        verbose=0
                    )
                    
                    # Predecir
                    last_sequence = scaled_data[-sequence_length:]
                    lstm_preds = []
                    
                    for _ in range(dias):
                        next_pred = model.predict(
                            last_sequence.reshape(1, sequence_length, 1),
                            verbose=0
                        )[0, 0]
                        lstm_preds.append(next_pred)
                        last_sequence = np.roll(last_sequence, -1)
                        last_sequence[-1] = next_pred
                    
                    # Desescalar
                    lstm_pred = scaler.inverse_transform(
                        np.array(lstm_preds).reshape(-1, 1)
                    ).ravel()
                    
                    resultados["LSTM"] = lstm_pred.round().astype(int)
                    logging.info("LSTM completado")
                else:
                    resultados["LSTM"] = [serie.mean()] * dias
                    
            except Exception as e:
                logging.error(f"Error en LSTM: {str(e)}")
                resultados["LSTM"] = [serie.mean()] * dias
        
        # Crear DataFrame final
        df_out = pd.DataFrame(resultados)
        
        # Agregar features a las fechas futuras
        df_out["dow"] = df_out["fecha"].dt.weekday
        df_out["is_weekend"] = df_out["dow"].isin([5,6]).astype(int)
        df_out["is_holiday"] = df_out["fecha"].isin(co_holidays).astype(int)
        df_out["mes"] = df_out["fecha"].dt.month
        df_out["dia_mes"] = df_out["fecha"].dt.day
        
        return df_out
        
    except Exception as e:
        logging.error(f"Error general en pronosticar: {str(e)}")
        raise e

# ---------------- COMPARAR MODELOS ----------------
def comparar_modelos(df, df_forecast, dias_reales=14):
    """
    Compara el rendimiento de todos los modelos
    """
    try:
        reales = df["volumen"].tail(dias_reales).values
        modelos = ["ARIMA / SARIMA", "Holt-Winters", "Prophet", "XGBoost", "LSTM"]
        
        resultados = {}
        for modelo in modelos:
            if modelo in df_forecast.columns:
                pred = df_forecast[modelo].head(dias_reales).values
                mae = mean_absolute_error(reales, pred)
                rmse = np.sqrt(mean_squared_error(reales, pred))
                resultados[modelo] = {"MAE": mae, "RMSE": rmse}
        
        return pd.DataFrame(resultados).T
        
    except Exception as e:
        logging.error(f"Error en comparar_modelos: {str(e)}")
        return pd.DataFrame()

# ---------------- SELECCIONAR MEJOR MODELO ----------------
def seleccionar_mejor_modelo(df, df_forecast):
    """
    Selecciona el modelo con mejor rendimiento
    """
    try:
        comparacion = comparar_modelos(df, df_forecast)
        if not comparacion.empty:
            mejor_modelo = comparacion["MAE"].idxmin()
            return mejor_modelo
        return "ARIMA / SARIMA"
    except Exception as e:
        logging.error(f"Error en seleccionar_mejor_modelo: {str(e)}")
        return "ARIMA / SARIMA"

# ---------------- ANALISIS POR DIA ----------------
def analisis_por_dia(df):
    """
    Analiza el volumen promedio por día de la semana
    """
    try:
        tmp = df.copy()
        tmp["dia"] = tmp["fecha"].dt.day_name()
        return tmp.groupby("dia")["volumen"].mean().reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
    except Exception as e:
        logging.error(f"Error en analisis_por_dia: {str(e)}")
        return pd.Series()

# ---------------- INTERVALOS DE CONFIANZA ----------------
def intervalos_confianza(df_forecast):
    """
    Calcula intervalos de confianza basados en la variabilidad de los modelos
    """
    try:
        modelos = ["ARIMA / SARIMA", "Holt-Winters", "Prophet", "XGBoost", "LSTM"]
        modelos_presentes = [m for m in modelos if m in df_forecast.columns]
        
        if modelos_presentes:
            media = df_forecast[modelos_presentes].mean(axis=1)
            std = df_forecast[modelos_presentes].std(axis=1)
            
            df_forecast["promedio_modelos"] = media.round()
            df_forecast["upper_90"] = (media + 1.645 * std).round()
            df_forecast["lower_90"] = (media - 1.645 * std).round()
            df_forecast["upper_95"] = (media + 1.96 * std).round()
            df_forecast["lower_95"] = (media - 1.96 * std).round()
        
        return df_forecast
        
    except Exception as e:
        logging.error(f"Error en intervalos_confianza: {str(e)}")
        return df_forecast