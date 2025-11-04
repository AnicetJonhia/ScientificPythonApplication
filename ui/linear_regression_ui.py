import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from core.linear_regression import fit_linear_regression
from tkinter import filedialog


class LinearRegressionFrame(ctk.CTkFrame):
    """
    desc:
        Interface graphique basée sur CustomTkinter permettant de visualiser
        une régression linéaire simple.  
        L’utilisateur peut charger un fichier CSV ou saisir manuellement
        des valeurs de x et y, puis afficher la droite d’ajustement
        ainsi que la formule correspondante.

    params:
        parent : Widget parent dans lequel ce frame est intégré.

    return:
        Aucun (composant graphique interactif).
    """

    def __init__(self, parent):
        """Initialise la fenêtre de saisie des données et la zone du graphique."""
        super().__init__(parent)

        # === Zone gauche : saisie et contrôle ===
        left = ctk.CTkFrame(self, width=320)
        left.pack(side="left", fill="y", padx=12, pady=12)

        ctk.CTkButton(left, text="Charger CSV (x,y)", command=self.load_csv).pack(pady=6)
        ctk.CTkLabel(left, text="ou collez x et y séparés par des virgules").pack(anchor='w')

        ctk.CTkLabel(left, text="x:").pack(anchor='w')
        self.x_text = ctk.CTkTextbox(left, height=4)
        self.x_text.pack(fill='x', pady=4)

        ctk.CTkLabel(left, text="y:").pack(anchor='w')
        self.y_text = ctk.CTkTextbox(left, height=4)
        self.y_text.pack(fill='x', pady=4)

        ctk.CTkButton(left, text="Ajuster", command=self.fit).pack(pady=8)
        self.result_label = ctk.CTkLabel(left, text="Résultat : -")
        self.result_label.pack(pady=6)

        # === Zone droite : graphique Matplotlib ===
        right = ctk.CTkFrame(self)
        right.pack(side='left', expand=True, fill='both', padx=12, pady=12)

        # Initialiser la figure matplotlib
        self.fig, self.ax = plt.subplots(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill='both', expand=True)
        self.canvas.draw()  # Forcer le rendu initial

    def load_csv(self):
        """
        desc:
            Ouvre une boîte de dialogue pour sélectionner un fichier CSV,
            puis charge les colonnes 'x' et 'y' s’il les trouve.

        params:
            Aucun (utilise un sélecteur de fichiers standard).

        return:
            Aucun (remplit les champs de saisie avec les données importées).
        """
        path = filedialog.askopenfilename(filetypes=[('CSV', '*.csv')])
        if not path:
            return
        df = pd.read_csv(path)
        if 'x' in df.columns and 'y' in df.columns:
            self.x_text.delete('0.0', 'end')
            self.y_text.delete('0.0', 'end')
            self.x_text.insert('0.0', ','.join(map(str, df['x'].tolist())))
            self.y_text.insert('0.0', ','.join(map(str, df['y'].tolist())))

    def fit(self):
        """
        desc:
            Extrait les valeurs de x et y saisies, exécute la régression linéaire
            via `fit_linear_regression()`, puis affiche la droite ajustée et
            la formule correspondante sur le graphique.

        params:
            Aucun (utilise les champs de saisie internes).

        return:
            Aucun (met à jour le graphique et le label de résultat).
        """
        try:
            xs = np.array([float(v) for v in self.x_text.get('0.0', 'end').strip().split(',') if v.strip()])
            ys = np.array([float(v) for v in self.y_text.get('0.0', 'end').strip().split(',') if v.strip()])
        except Exception as e:
            self.result_label.configure(text=f"Erreur lecture: {e}")
            return

        # Ajustement avec sklearn (si dispo) ou numpy.polyfit
        coef, intercept, predict = fit_linear_regression(xs, ys)
        self.result_label.configure(text=f"y = {coef:.4f} x + {intercept:.4f}")

        # === Tracer le graphique ===
        self.ax.clear()
        self.ax.scatter(xs, ys, label='Données', color="#149911")
        xs_line = np.linspace(xs.min(), xs.max(), 200)
        try:
            ys_line = predict(xs_line.reshape(-1, 1))  # sklearn LinearRegression
        except Exception:
            ys_line = predict(xs_line)  # fallback numpy.polyfit

        self.ax.plot(xs_line, ys_line, 'r-', label='Régression')
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.legend()
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.canvas.draw()
