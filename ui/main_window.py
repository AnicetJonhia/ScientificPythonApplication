
import customtkinter as ctk
from ui.linear_system_ui import LinearSystemFrame
from ui.linear_programming_ui import LinearProgrammingFrame
from ui.linear_regression_ui import LinearRegressionFrame
from ui.stochastic_ui import StochasticFrame


class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self.title("Math Solver App")
        self.geometry("1000x700")

        # Header
        header = ctk.CTkFrame(self, height=60)
        header.pack(side="top", fill="x")
        ctk.CTkLabel(header, text="Math Solver App", font=(None, 20)).pack(padx=20, pady=10, anchor="w")

        # Main body : TabView
        tabview = ctk.CTkTabview(self)
        tabview.pack(expand=True, fill="both", padx=12, pady=12)

        tabview.add("Systèmes linéaires")
        tabview.add("Programmation linéaire")
        tabview.add("Régression linéaire")
        tabview.add("Processus stochastiques")

        # Récupérer le conteneur d'un onglet via tabview.tab(name)
        LinearSystemFrame(tabview.tab("Systèmes linéaires")).pack(expand=True, fill="both")
        LinearProgrammingFrame(tabview.tab("Programmation linéaire")).pack(expand=True, fill="both")
        LinearRegressionFrame(tabview.tab("Régression linéaire")).pack(expand=True, fill="both")
        StochasticFrame(tabview.tab("Processus stochastiques")).pack(expand=True, fill="both")