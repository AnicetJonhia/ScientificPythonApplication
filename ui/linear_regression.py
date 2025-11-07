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

    if st.button("Ajuster"):
        try:
            xs = np.array([float(v) for v in st.session_state['x_input'].strip().split(',') if v.strip()])
            ys = np.array([float(v) for v in st.session_state['y_input'].strip().split(',') if v.strip()])
        except Exception as e:
            st.error(f"Erreur lecture des données: {e}")
            return

        coef, intercept, predict = fit_linear_regression(xs, ys)
        st.success(f"y = {coef:.4f} x + {intercept:.4f}")

        xs_line = np.linspace(xs.min(), xs.max(), 200)
        try:
            ys_line = predict(xs_line.reshape(-1, 1))
        except Exception:
            ys_line = predict(xs_line)

        # Plotly scatter + line
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers', name='Données', marker=dict(color='#149911')))
        fig.add_trace(go.Scatter(x=xs_line, y=ys_line, mode='lines', name='Régression', line=dict(color='red')))
        fig.update_layout(xaxis_title='x', yaxis_title='y', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
