import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from core.linear_system import solve_linear_system


class LinearSystemFrame(ctk.CTkFrame):
    """
    desc:
        Interface graphique CustomTkinter pour résoudre un système linéaire AX = b.  
        L’utilisateur peut saisir la matrice A et le vecteur b, obtenir la solution,
        et visualiser le graphique si le système est 2x2.

    params:
        parent : Widget parent dans lequel ce frame est intégré.

    return:
        Aucun (composant interactif avec saisie et graphique).
    """

    def __init__(self, parent):
        """Initialise l’interface avec champs de saisie et zone graphique."""
        super().__init__(parent)

        # === Zone gauche : saisie des données ===
        left = ctk.CTkFrame(self, width=320)
        left.pack(side="left", fill="y", padx=12, pady=12)

        ctk.CTkLabel(left, text="Entrer la matrice A (ex: [[2,1],[1,3]])").pack(anchor="w", pady=(6,0))
        self.entry_a = ctk.CTkTextbox(left, height=6)
        self.entry_a.insert("0.0", "[[2,1],[1,3]]")
        self.entry_a.pack(fill="x", pady=6)

        ctk.CTkLabel(left, text="Entrer le vecteur b (ex: [8,13])").pack(anchor="w")
        self.entry_b = ctk.CTkEntry(left)
        self.entry_b.insert(0, "[8,13]")
        self.entry_b.pack(fill="x", pady=6)

        self.solve_btn = ctk.CTkButton(left, text="Résoudre", command=self.solve)
        self.solve_btn.pack(pady=8)

        self.result_label = ctk.CTkLabel(left, text="Solution : -")
        self.result_label.pack(pady=6)

        # === Zone droite : graphique ===
        right = ctk.CTkFrame(self)
        right.pack(side="left", expand=True, fill="both", padx=12, pady=12)

        self.fig, self.ax = plt.subplots(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def solve(self):
        """
        desc:
            Lit la matrice A et le vecteur b depuis les champs de saisie,
            résout le système linéaire AX=b via `solve_linear_system()`, 
            affiche la solution, et trace un graphique si le système est 2x2.

        params:
            Aucun (utilise les champs de saisie internes).

        return:
            Aucun (met à jour le label de résultat et le graphique).
        """
        # --- Lecture des données ---
        try:
            A = eval(self.entry_a.get("0.0", "end").strip())
            b = eval(self.entry_b.get())
            A = np.array(A, dtype=float)
            b = np.array(b, dtype=float)
        except Exception as e:
            self.result_label.configure(text=f"Erreur lecture : {e}")
            return

        # --- Calcul de la solution ---
        try:
            x = solve_linear_system(A, b)
            self.result_label.configure(text=f"Solution : {np.round(x,6)}")
        except Exception as e:
            self.result_label.configure(text=f"Erreur calcul : {e}")
            return

        # --- Graphique ---
        self.ax.clear()
        if A.shape[0] == 2 and A.shape[1] == 2:
            # Ax=b -> deux équations a11*x + a12*y = b1, a21*x + a22*y = b2
            xs = np.linspace(-10, 10, 300)
            a11, a12 = A[0]
            a21, a22 = A[1]
            b1, b2 = b

            y1 = (b1 - a11*xs)/a12 if a12 != 0 else None
            y2 = (b2 - a21*xs)/a22 if a22 != 0 else None

            if y1 is not None:
                self.ax.plot(xs, y1, label='équation 1')
            if y2 is not None:
                self.ax.plot(xs, y2, label='équation 2')

            self.ax.plot(x[0], x[1], 'o', color='red', label='solution')
            self.ax.set_xlim(-10,10)
            self.ax.set_ylim(-10,10)
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("y")
            self.ax.legend()
            self.ax.grid(True, linestyle='--', alpha=0.4)
        else:
            self.ax.text(0.5, 0.5, 'Graphique disponible pour 2x2 uniquement', ha='center')

        self.canvas.draw()
