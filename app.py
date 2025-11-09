import streamlit as st
try:
    from streamlit_option_menu import option_menu
    _HAS_OPTION_MENU = True
except Exception:
    _HAS_OPTION_MENU = False

from ui.linear_system import render as render_linear_system
from ui.linear_programming import render as render_lp
from ui.linear_regression import render as render_regression
from ui.stochastic import render as render_stochastic



APP_TITLE = "Application graphique de résolution et de modélisation mathématique"
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Sidebar styled menu using streamlit-option-menu for a button-like web menu
if _HAS_OPTION_MENU:
    with st.sidebar:
        selected = option_menu(
            menu_title="Menu",
            options=["Systèmes linéaires", "Programmation linéaire", "Régression linéaire", "Processus stochastiques"],
            icons=["gear", "graph-up", "graph-up-arrow", "dice-5"],
            menu_icon="app-indicator",
            default_index=0,
            orientation="vertical",
        )
else:
    with st.sidebar:
        st.sidebar.title("Menu")
        selected = st.radio("Navigation", ("🧮  Systèmes linéaires", "📈  Programmation linéaire", "📉  Régression linéaire", "🎲  Processus stochastiques"))
        # normalize to same labels as option_menu
        if isinstance(selected, str) and selected.startswith('🧮'):
            selected = 'Systèmes linéaires'
        elif isinstance(selected, str) and selected.startswith('📈'):
            selected = 'Programmation linéaire'
        elif isinstance(selected, str) and selected.startswith('📉'):
            selected = 'Régression linéaire'
        elif isinstance(selected, str) and selected.startswith('🎲'):
            selected = 'Processus stochastiques'

# Sidebar footer with author / affiliation
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**RANDRIANAMBININA Tokiniaina Jean Anicet Jonhia**  \nL3IDEV  \nESTI",
    unsafe_allow_html=True,
)

if selected == "Systèmes linéaires":
    render_linear_system()
elif selected == "Programmation linéaire":
    render_lp()
elif selected == "Régression linéaire":
    render_regression()
elif selected == "Processus stochastiques":
    render_stochastic()
