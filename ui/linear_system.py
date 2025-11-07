import streamlit as st
import numpy as np
import ast
import plotly.graph_objects as go

from core.linear_system import solve_linear_system


def render():
    st.header("Systèmes linéaires")

    a_text = st.text_area("Entrer la matrice A (ex: [[2,1],[1,3]])", value="[[2,1],[1,3]]", height=120)
    b_text = st.text_input("Entrer le vecteur b (ex: [8,13])", value="[8,13]")

    if st.button("Résoudre"):
        try:
            A = ast.literal_eval(a_text)
            b = ast.literal_eval(b_text)
            A = np.array(A, dtype=float)
            b = np.array(b, dtype=float)
        except Exception as e:
            st.error(f"Erreur lecture : {e}")
            return

        try:
            x = solve_linear_system(A, b)
            st.success(f"Solution : {np.round(x,6)}")
        except Exception as e:
            st.error(f"Erreur calcul : {e}")
            return

        # Plot if 2x2 using Plotly
        if A.shape[0] == 2 and A.shape[1] == 2:
            xs = np.linspace(-10, 10, 300)
            a11, a12 = A[0]
            a21, a22 = A[1]
            b1, b2 = b

            y1 = (b1 - a11*xs)/a12 if a12 != 0 else None
            y2 = (b2 - a21*xs)/a22 if a22 != 0 else None

            fig = go.Figure()
            if y1 is not None:
                fig.add_trace(go.Scatter(x=xs, y=y1, mode='lines', name='équation 1'))
            if y2 is not None:
                fig.add_trace(go.Scatter(x=xs, y=y2, mode='lines', name='équation 2'))

            fig.add_trace(go.Scatter(x=[x[0]], y=[x[1]], mode='markers', name='solution', marker=dict(color='red', size=10)))
            fig.update_layout(xaxis=dict(range=[-10,10]), yaxis=dict(range=[-10,10]), xaxis_title='x', yaxis_title='y', template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('Graphique disponible pour 2x2 uniquement')
