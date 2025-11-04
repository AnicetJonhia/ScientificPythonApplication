import customtkinter as ctk
from core.linear_programming import solve_lp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class LinearProgrammingFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        # --- Partie gauche : saisie ---
        left = ctk.CTkFrame(self, width=320)
        left.pack(side="left", fill="y", padx=12, pady=12)

        ctk.CTkLabel(left, text="Maximiser Z = c1*x + c2*y (entrez c1,c2)").pack(anchor="w")
        self.c_entry = ctk.CTkEntry(left)
        self.c_entry.insert(0, "3,2")
        self.c_entry.pack(fill="x", pady=6)

        ctk.CTkLabel(left, text="Contraintes (chaque ligne: a1,a2<=b). Ex: 1,0<=4").pack(anchor="w")
        self.constraints_text = ctk.CTkTextbox(left, height=8)
        self.constraints_text.insert("0.0", "1,0<=4\n0,1<=3\n1,1<=5")
        self.constraints_text.pack(fill="x", pady=6)

        self.solve_btn = ctk.CTkButton(left, text="Résoudre", command=self.solve)
        self.solve_btn.pack(pady=8)

        self.result_label = ctk.CTkLabel(left, text="Solution : -")
        self.result_label.pack(pady=6)

        # --- Partie droite : graphique ---
        self.fig, self.ax = plt.subplots(figsize=(4.5, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side="right", fill="both", expand=True, padx=12, pady=12)

    def solve(self):
        try:
            # Lecture des coefficients
            c = [float(v.strip()) for v in self.c_entry.get().split(',')]
            lines = [ln.strip() for ln in self.constraints_text.get("0.0", "end").splitlines() if ln.strip()]
            A, b = [], []

            for ln in lines:
                if '<=' in ln:
                    lhs, rhs = ln.split('<=')
                    coeffs = [float(x) for x in lhs.split(',')]
                    A.append(coeffs)
                    b.append(float(rhs))
                else:
                    raise ValueError('Contrainte mal formatée (utiliser <=)')

            sol, opt = solve_lp(c, A, b, maximize=True)
            self.result_label.configure(text=f"Solution: {sol} | Opt = {opt:.2f}")

            # --- Dessiner graphique si 2 variables ---
            if len(c) == 2:
                self.plot_feasible_region(A, b, sol)
        except Exception as e:
            self.result_label.configure(text=f"Erreur: {e}")

    def plot_feasible_region(self, A, b, sol):
        self.ax.clear()

        x = np.linspace(0, 10, 200)
        for (a1, a2), bi in zip(A, b):
            y = (bi - a1 * x) / a2
            self.ax.plot(x, y, label=f"{a1}x + {a2}y ≤ {bi}")

        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.set_xlabel("x₁")
        self.ax.set_ylabel("x₂")
        self.ax.grid(True, linestyle="--", alpha=0.4)

        # Afficher point optimal
        if "x0" in sol and "x1" in sol:
            self.ax.scatter(sol["x0"], sol["x1"], color="red", s=80, label="Point optimal")
            self.ax.text(sol["x0"] + 0.2, sol["x1"], "Optimum", color="red")

        self.ax.legend()
        self.canvas.draw()
