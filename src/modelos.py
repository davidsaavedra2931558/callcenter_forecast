import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

# =========================
# HOLT WINTERS
# =========================
def modelo_holt(serie, pasos):
    modelo = ExponentialSmoothing(serie, trend="add").fit()
    pred = modelo.forecast(pasos)
    return np.round(pred).astype(int)

# =========================
# ARIMA
# =========================
def modelo_arima(serie, pasos):
    modelo = ARIMA(serie, order=(2,1,2)).fit()
    pred = modelo.forecast(pasos)
    return np.round(pred).astype(int)
