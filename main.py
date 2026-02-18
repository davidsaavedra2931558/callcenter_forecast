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
import holidays

co_holidays = holidays.Colombia()

# ---------------- PREPARAR DATA ----------------
def preparar_df(df):
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["volumen"] = pd.to_numeric(df["volumen"], errors="coerce")
    df = df.dropna().sort_values("fecha").reset_index(drop=True)

    df["dow"] = df["fecha"].dt.weekday
    df["is_weekend"] = df["dow"].isin([5,6]).astype(int)
    df["is_holiday"] = df["fecha"].isin(co_holidays).astype(int)
    return df

# ---------------- METRICA SIMPLE ----------------
def evaluar_modelo(df):
    X = np.arange(len(df)).reshape(-1, 1)
    y = df["volumen"].values
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    return mean_absolute_error(y, pred), np.sqrt(mean_squared_error(y, pred))

# ---------------- FEATURES ----------------
def crear_features_lag(df):
    df = df.copy()
    df["lag_1"] = df["volumen"].shift(1)
    df["lag_7"] = df["volumen"].shift(7)
    df["lag_14"] = df["volumen"].shift(14)
    df["roll_7"] = df["volumen"].rolling(7).mean()
    df["roll_14"] = df["volumen"].rolling(14).mean()
    return df.dropna()

# ---------------- PRONOSTICO ----------------
def pronosticar(df, dias):

    serie = df.set_index("fecha")["volumen"]

    fechas_futuras = pd.date_range(
        serie.index[-1] + pd.Timedelta(days=1),
        periods=dias, freq="D"
    )

    arima_pred = SARIMAX(
        serie, order=(1,1,1), seasonal_order=(1,1,1,7)
    ).fit(disp=False).forecast(dias)

    hw_pred = ExponentialSmoothing(
        serie, trend="add", seasonal="add", seasonal_periods=7
    ).fit().forecast(dias)

    festivos = pd.DataFrame({
        "ds": pd.to_datetime(list(co_holidays.keys())),
        "holiday": "CO"
    })

    df_p = df.rename(columns={"fecha":"ds","volumen":"y"})
    p = Prophet(daily_seasonality=True, holidays=festivos)
    p.fit(df_p)
    prophet_pred = p.predict(
        p.make_future_dataframe(periods=dias)
    )["yhat"].tail(dias).values

    # ----- XGBOOST -----
    df_lag = crear_features_lag(df)
    X = df_lag[["lag_1","lag_7","lag_14","roll_7","roll_14","is_weekend","is_holiday"]]
    y = df_lag["volumen"]

    model_xgb = xgb.XGBRegressor(
        n_estimators=150, learning_rate=0.05, max_depth=4
    )
    model_xgb.fit(X, y)

    last = df_lag.iloc[-1][X.columns].values.tolist()
    xgb_preds = []

    for _ in range(dias):
        pred = model_xgb.predict(np.array(last).reshape(1,-1))[0]
        xgb_preds.append(pred)
        last = [
            pred, last[0], last[1],
            np.mean(last), np.mean(last[:3]),
            last[-2], last[-1]
        ]

    # ----- LSTM -----
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(y.values.reshape(-1,1))
    Xs, ys = [], []

    for i in range(30, len(scaled)):
        Xs.append(scaled[i-30:i])
        ys.append(scaled[i])

    Xs, ys = np.array(Xs), np.array(ys)

    model = Sequential([LSTM(32, input_shape=(30,1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xs, ys, epochs=5, verbose=0)

    last = scaled[-30:]
    preds = []

    for _ in range(dias):
        p = model.predict(last.reshape(1,30,1), verbose=0)
        preds.append(p[0,0])
        last = np.vstack([last[1:], p])

    lstm_pred = scaler.inverse_transform(
        np.array(preds).reshape(-1,1)
    ).ravel()

    df_out = pd.DataFrame({
        "fecha": fechas_futuras,
        "ARIMA / SARIMA": arima_pred.round().astype(int),
        "Holt-Winters": hw_pred.round().astype(int),
        "Prophet": prophet_pred.round().astype(int),
        "XGBoost": np.array(xgb_preds).round().astype(int),
        "LSTM": lstm_pred.round().astype(int),
    })

    df_out["dow"] = df_out["fecha"].dt.weekday
    df_out["is_weekend"] = df_out["dow"].isin([5,6]).astype(int)
    df_out["is_holiday"] = df_out["fecha"].isin(co_holidays).astype(int)

    return df_out

# ---------------- MEJOR MODELO ----------------
def seleccionar_mejor_modelo(df, df_forecast):
    errores = {}
    reales = df["volumen"].tail(14).values

    for col in ["ARIMA / SARIMA","Holt-Winters","Prophet","XGBoost","LSTM"]:
        try:
            pred = df_forecast[col].head(14).values
            errores[col] = mean_absolute_error(reales, pred)
        except:
            pass

    return min(errores, key=errores.get)

# ---------------- ANALISIS POR DIA ----------------
def analisis_por_dia(df):
    tmp = df.copy()
    tmp["dia"] = tmp["fecha"].dt.day_name()
    return tmp.groupby("dia")["volumen"].mean().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    )

# ---------------- INTERVALOS ----------------
def intervalos_confianza(df_forecast):
    media = df_forecast[
        ["ARIMA / SARIMA","Holt-Winters","Prophet","XGBoost","LSTM"]
    ].mean(axis=1)

    df_forecast["upper"] = (media * 1.1).round()
    df_forecast["lower"] = (media * 0.9).round()
    return df_forecast

