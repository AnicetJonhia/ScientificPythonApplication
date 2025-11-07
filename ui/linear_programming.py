import streamlit as st
import numpy as np
import plotly.graph_objects as go

from core.linear_programming import solve_lp


def render():
    """Streamlit renderer for a 2-variable linear programming example using Plotly."""
    st.header("Programmation linéaire")

    c_text = st.text_input("Coefficients de la fonction objectif (c1,c2)", value="3,2")
    constraints_text = st.text_area(
        "Contraintes (une par ligne, format: a1,a2<=b). Exemple: 1,0<=4",
        value="1,0<=4\n0,1<=3\n1,1<=5",
        height=150,
    )

    if st.button("Résoudre"):
        try:
            c = [float(v.strip()) for v in c_text.split(",") if v.strip()]
            lines = [ln.strip() for ln in constraints_text.splitlines() if ln.strip()]
            A, b = [], []
            for ln in lines:
                if "<=" in ln:
                    lhs, rhs = ln.split("<=")
                    coeffs = [float(x) for x in lhs.split(",")]
                    A.append(coeffs)
                    b.append(float(rhs))
                else:
                    raise ValueError("Contrainte mal formatée (utiliser <=)")

            sol, opt = solve_lp(c, A, b, maximize=True)
            st.success(f"Solution: {sol} | Opt = {opt:.4f}")

            if len(c) == 2:
                x = np.linspace(0, max(10, max(b) if b else 10), 400)
                fig = go.Figure()
                for (a1, a2), bi in zip(A, b):
                    # éviter division par zéro
                    if a2 == 0:
                        y = np.full_like(x, np.nan)
                    else:
                        y = (bi - a1 * x) / a2
                    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f"{a1}x + {a2}y ≤ {bi}"))

                if isinstance(sol, dict) and "x0" in sol and "x1" in sol:
                    fig.add_trace(go.Scatter(x=[sol['x0']], y=[sol['x1']], mode='markers', name='Point optimal', marker=dict(color='red', size=10)))

                fig.update_layout(xaxis_title='x₁', yaxis_title='x₂', template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur: {e}")
