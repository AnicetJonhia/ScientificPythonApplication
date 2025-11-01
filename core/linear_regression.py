
import numpy as np

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN = True
except Exception:
    SKLEARN = False


def fit_linear_regression(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x_flat = x.reshape(-1, 1)
    if SKLEARN:
        model = LinearRegression()
        model.fit(x_flat, y)
        coef = model.coef_[0]
        intercept = model.intercept_
        return coef, intercept, model.predict
    else:
        # fallback numpy polyfit
        coef, intercept = np.polyfit(x, y, 1)
        def predict_fn(xx):
            xx = np.array(xx, dtype=float)
            return coef * xx + intercept
        return coef, intercept, predict_fn