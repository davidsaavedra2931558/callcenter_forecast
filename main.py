import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# -------------------------
# PREPARAR DATA
# -------------------------
def preparar_df(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["volumen"] = pd.to_numeric(df["volumen"], errors="coerce")
    df = df.dropna().sort_values("fecha").reset_index(drop=True)

    # Variables calendario
    df["mes"] = df["fecha"].dt.month
    df["es_verano"] = df["mes"].isin([6,7,8]).astype(int)

    # Festivos simples (ejemplo, puedes ajustar)
    festivos = [
        "2023-01-01","2023-07-20","2023-12-25",
        "2024-01-01","2024-07-20","2024-12-25",
        "2025-01-01","2025-07-20","2025-12-25"
    ]
    df["es_festivo"] = df["fecha"].isin(pd.to_datetime(festivos)).astype(int)

    # Outliers automáticos
    q1 = df["volumen"].quantile(0.25)
    q3 = df["volumen"].quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    df["outlier"] = ((df["volumen"] < lim_inf) | (df["volumen"] > lim_sup)).astype(int)

    return df

# -------------------------
# MÉTRICAS
# -------------------------
def evaluar_modelo(df):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["volumen"].values
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    return mean_absolute_error(y, pred), np.sqrt(mean_squared_error(y, pred))

# -------------------------
# PRONOSTICAR REAL
# -------------------------
def pronosticar(df, dias):

    serie = df.set_index("fecha")["volumen"]
    fechas_futuras = pd.date_range(
        start=serie.index[-1] + pd.Timedelta(days=1),
        periods=dias,
        freq="D"
    )

    # ---------- ARIMA ----------
    arima = SARIMAX(serie, order=(1,1,1), seasonal_order=(1,1,1,7)).fit(disp=False)
    arima_pred = arima.forecast(dias)

    # ---------- HOLT ----------
    hw = ExponentialSmoothing(
        serie, trend="add", seasonal="add", seasonal_periods=7
    ).fit()
    hw_pred = hw.forecast(dias)

    # ---------- PROPHET con festivos ----------
    df_p = df.rename(columns={"fecha":"ds","volumen":"y"})
    holidays = df[df["es_festivo"]==1][["fecha"]].rename(columns={"fecha":"ds"})
    holidays["holiday"] = "festivo"

    p = Prophet(daily_seasonality=True, holidays=holidays)
    p.add_seasonality(name="verano", period=365, fourier_order=5)
    p.fit(df_p)

    future = p.make_future_dataframe(periods=dias)
    prophet_pred = p.predict(future)["yhat"].tail(dias).values

    # ---------- XGBOOST con variables ----------
    feats = ["mes","es_verano","es_festivo","outlier"]
    X = df[feats]
    y = df["volumen"]
    model_xgb = xgb.XGBRegressor(n_estimators=300)
    model_xgb.fit(X, y)

    futuro = pd.DataFrame({
        "mes": fechas_futuras.month,
        "es_verano": fechas_futuras.month.isin([6,7,8]).astype(int),
        "es_festivo": 0,
        "outlier": 0
    })
    xgb_pred = model_xgb.predict(futuro)

    # ---------- LSTM ----------
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(y.values.reshape(-1,1))

    Xs, ys = [], []
    for i in range(30, len(scaled)):
        Xs.append(scaled[i-30:i])
        ys.append(scaled[i])
    Xs, ys = np.array(Xs), np.array(ys)

    model = Sequential([
        LSTM(50, activation="relu", input_shape=(30,1)),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xs, ys, epochs=10, verbose=0)

    last = scaled[-30:]
    preds = []
    for _ in range(dias):
        p = model.predict(last.reshape(1,30,1), verbose=0)
        preds.append(p[0,0])
        last = np.vstack([last[1:], p])

    lstm_pred = scaler.inverse_transform(np.array(preds).reshape(-1,1)).ravel()

    return pd.DataFrame({
        "fecha": fechas_futuras,
        "ARIMA / SARIMA": arima_pred.round().astype(int),
        "Holt-Winters": hw_pred.round().astype(int),
        "Prophet": prophet_pred.round().astype(int),
        "XGBoost": xgb_pred.round().astype(int),
        "LSTM": lstm_pred.round().astype(int),
    })
