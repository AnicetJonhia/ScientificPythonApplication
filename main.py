import streamlit as st

from ui.linear_system import render as render_linear_system
from ui.linear_programming import render as render_lp
from ui.linear_regression import render as render_regression
from ui.stochastic import render as render_stochastic


st.set_page_config(page_title="Math Solver App", layout="wide")

# Sidebar styled menu using radio (list-like selection) with simple icons
st.sidebar.title("Math Solver App")
menu = st.sidebar.radio(
    "Navigation",
    (
        "🧮  Systèmes linéaires",
        "📈  Programmation linéaire",
        "📉  Régression linéaire",
        "🎲  Processus stochastiques",
    ),
    index=0,
)

if menu.startswith("🧮"):
    render_linear_system()
elif menu.startswith("📈"):
    render_lp()
elif menu.startswith("📉"):
    render_regression()
elif menu.startswith("🎲"):
    render_stochastic()
