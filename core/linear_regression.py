
import numpy as np

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN = True
except Exception:
    SKLEARN = False


def fit_linear_regression(x, y, return_score=False):

    """
    desc:
        Ajuste un modèle de régression linéaire simple (y = a*x + b) 
        à partir des données fournies.  
        Utilise scikit-learn si disponible, sinon un fallback basé sur numpy.polyfit.

    params:
        x : Liste ou tableau numpy contenant les valeurs des variables indépendantes.
        y : Liste ou tableau numpy contenant les valeurs des variables dépendantes (cibles).

    return:
        Un tuple contenant :
            - coef : Coefficient directeur (pente) de la droite de régression.
            - intercept : Ordonnée à l’origine (biais) de la droite.
            - predict_fn : Fonction permettant de prédire les valeurs de y à partir de nouvelles valeurs de x.
    """

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if x.size != y.size:
        raise ValueError("Les tableaux x et y doivent avoir la même longueur.")

    x_flat = x.reshape(-1, 1)
    if SKLEARN:
        model = LinearRegression()
        model.fit(x_flat, y)
        coef = model.coef_[0]
        intercept = model.intercept_
        predict_fn = model.predict
        if return_score:
            score = model.score(x_flat, y)
            return coef, intercept, predict_fn, score
        return coef, intercept, predict_fn
    else:
        # fallback numpy polyfit
        coef, intercept = np.polyfit(x, y, 1)
        def predict_fn(xx):
            xx = np.array(xx, dtype=float)
            return coef * xx + intercept
        if return_score:
            # compute R^2 manually
            y_pred = coef * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
            return coef, intercept, predict_fn, r2
        return coef, intercept, predict_fn