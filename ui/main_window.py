import customtkinter as ctk
from ui.linear_system_ui import LinearSystemFrame
from ui.linear_programming_ui import LinearProgrammingFrame
from ui.linear_regression_ui import LinearRegressionFrame
from ui.stochastic_ui import StochasticFrame

class MainApp(ctk.CTk):
    """
    desc:
        Application principale pour Math Solver App.
        Fournit une interface graphique avec onglets pour accéder aux différents modules :
        - Systèmes linéaires
        - Programmation linéaire
        - Régression linéaire
        - Processus stochastiques

    params:
        Aucun

    return:
        Une fenêtre interactive CTk avec onglets pour chaque module.
    """

    def __init__(self):
        """
        desc:
            Initialise la fenêtre principale, configure le thème, crée un header et un CTkTabview
            avec un onglet pour chaque module scientifique.
        params:
            Aucun
        return:
            Aucun (crée et configure l'interface graphique)
        """
        super().__init__()

        # --- Apparence ---
        ctk.set_appearance_mode("system")   # mode clair/sombre automatique
        ctk.set_default_color_theme("blue") # thème couleur

        self.title("Math Solver App")
        self.geometry("1000x700")

        # --- Header ---
        header = ctk.CTkFrame(self, height=60)
        header.pack(side="top", fill="x")
        ctk.CTkLabel(header, text="Math Solver App", font=(None, 20)).pack(padx=20, pady=10, anchor="w")

        # --- Corps principal : TabView pour les modules ---
        tabview = ctk.CTkTabview(self)
        tabview.pack(expand=True, fill="both", padx=12, pady=12)

        tabview.add("Systèmes linéaires")
        tabview.add("Programmation linéaire")
        tabview.add("Régression linéaire")
        tabview.add("Processus stochastiques")

        # Ajouter les frames des modules dans chaque onglet
        LinearSystemFrame(tabview.tab("Systèmes linéaires")).pack(expand=True, fill="both")
        LinearProgrammingFrame(tabview.tab("Programmation linéaire")).pack(expand=True, fill="both")
        LinearRegressionFrame(tabview.tab("Régression linéaire")).pack(expand=True, fill="both")
        StochasticFrame(tabview.tab("Processus stochastiques")).pack(expand=True, fill="both")
