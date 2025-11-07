import streamlit as st
import numpy as np
import ast
import plotly.graph_objects as go

from core.stochastic_process import simulate_markov_chain, empirical_distribution


def render():
    st.header("Processus stochastiques — Chaînes de Markov")

    P_text = st.text_area(
        'Matrice de transition (ex: [[0.5,0.5,0],[0.2,0.4,0.4],[0,0.3,0.7]])',
        value='[[0.5,0.5,0],[0.2,0.4,0.4],[0,0.3,0.7]]',
        height=150,
    )
    init = st.number_input('Etat initial (index)', min_value=0, value=0, step=1)
    steps = st.number_input('Nombre d étapes', min_value=1, value=200, step=1)

    if st.button('Simuler'):
        try:
            P = ast.literal_eval(P_text)
            P = np.array(P, dtype=float)
        except Exception as e:
            st.error(f'Erreur lecture: {e}')
            return

        states = simulate_markov_chain(P, initial_state=int(init), steps=int(steps))
        dist = empirical_distribution(states, n_states=P.shape[0])
        st.success(f'Distribution empirique finale: {np.round(dist,4)}')

        # Plotly line chart for states
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(states))), y=states, mode='lines+markers', name='Etat'))
        fig.update_layout(xaxis_title='Temps', yaxis_title='Etat', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
