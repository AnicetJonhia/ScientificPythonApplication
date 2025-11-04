
import numpy as np

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN = True
except Exception:
    SKLEARN = False


def fit_linear_regression(x, y):

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