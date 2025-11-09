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
    seed = st.number_input('Graine RNG (optionnel pour reproductibilité, 0 = aléatoire)', value=0, step=1)
    show_stationary = st.checkbox('Calculer distribution stationnaire (valeurs propres/power)', value=True)

    if st.button('Simuler'):
        try:
            P = ast.literal_eval(P_text)
            P = np.array(P, dtype=float)
        except Exception as e:
            st.error(f'Erreur lecture: {e}')
            return

        rng = None
        if int(seed) != 0:
            rng = np.random.default_rng(int(seed))

        states = simulate_markov_chain(P, initial_state=int(init), steps=int(steps), rng=rng)
        dist = empirical_distribution(states, n_states=P.shape[0])
        st.success(f'Distribution empirique finale: {np.round(dist,4)}')

        if show_stationary:
            try:
                from core.stochastic_process import stationary_distribution
                pi = stationary_distribution(P)
                st.write('Distribution stationnaire (approx):')
                st.write(np.round(pi, 6))
            except Exception as e:
                st.warning(f"Impossible de calculer la distribution stationnaire: {e}")

        # Plotly line chart for states
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(states))), y=states, mode='lines+markers', name='Etat'))
        fig.update_layout(xaxis_title='Temps', yaxis_title='Etat', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
