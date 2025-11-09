import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.linear_regression import fit_linear_regression


def render():
    st.header("Régression linéaire")

    uploaded = st.file_uploader("Charger un CSV avec colonnes 'x' et 'y'", type=["csv"], key="reg_upload")

    # Initialize session state keys for inputs so we can update them when a file is uploaded
    if 'x_input' not in st.session_state:
        st.session_state['x_input'] = "1,2,3,4,5"
    if 'y_input' not in st.session_state:
        st.session_state['y_input'] = "2,4,5,4,5"

    # If user uploaded a CSV, parse and update the session state so the text areas reflect file data
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            if 'x' in df.columns and 'y' in df.columns:
                st.session_state['x_input'] = ','.join(map(str, df['x'].tolist()))
                st.session_state['y_input'] = ','.join(map(str, df['y'].tolist()))
                st.success('Données importées depuis le fichier et champs mis à jour.')
            else:
                st.warning("Le CSV doit contenir des colonnes 'x' et 'y'.")
        except Exception as e:
            st.error(f"Erreur lecture CSV: {e}")

    # Render text areas bound to session state keys so they update when we change session_state
    x_text = st.text_area("x (valeurs séparées par des virgules)", key='x_input', height=80)
    y_text = st.text_area("y (valeurs séparées par des virgules)", key='y_input', height=80)

    use_matplotlib = st.checkbox("Afficher avec Matplotlib (alternatif)", value=False)

    if st.button("Ajuster"):
        try:
            xs = np.array([float(v) for v in st.session_state['x_input'].strip().split(',') if v.strip()])
            ys = np.array([float(v) for v in st.session_state['y_input'].strip().split(',') if v.strip()])
        except Exception as e:
            st.error(f"Erreur lecture des données: {e}")
            return

        # request score too
        res = fit_linear_regression(xs, ys, return_score=True)
        if len(res) == 4:
            coef, intercept, predict, score = res
        else:
            coef, intercept, predict = res
            score = None

        # predict on training data
        try:
            y_pred = predict(xs.reshape(-1, 1))
        except Exception:
            y_pred = predict(xs)

        # metrics
        residuals = ys - y_pred
        rmse = np.sqrt(np.mean(residuals ** 2))
        st.success(f"y = {coef:.4f} x + {intercept:.4f}")
        if score is not None:
            st.write(f"R² (score) = {score:.4f}")
        else:
            # compute R^2 manually
            ss_res = np.sum((ys - y_pred) ** 2)
            ss_tot = np.sum((ys - np.mean(ys)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
            st.write(f"R² (calculé) = {r2:.4f}")
        st.write(f"RMSE = {rmse:.4f}")

        xs_line = np.linspace(xs.min(), xs.max(), 200)
        try:
            ys_line = predict(xs_line.reshape(-1, 1))
        except Exception:
            ys_line = predict(xs_line)

        if use_matplotlib:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(xs, ys, label='Données', color="#149911")
            ax.plot(xs_line, ys_line, 'r-', label='Régression')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.4)
            st.pyplot(fig)
        else:
            # Plotly scatter + line
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', name='Données', marker=dict(color='#149911')))
            fig.add_trace(go.Scatter(x=xs_line, y=ys_line, mode='lines', name='Régression', line=dict(color='red')))
            fig.update_layout(xaxis_title='x', yaxis_title='y', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
