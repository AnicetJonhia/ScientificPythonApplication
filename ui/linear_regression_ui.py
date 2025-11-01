
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from core.linear_regression import fit_linear_regression
from tkinter import filedialog

class LinearRegressionFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

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

        right = ctk.CTkFrame(self)
        right.pack(side='left', expand=True, fill='both', padx=12, pady=12)
        self.fig, self.ax = plt.subplots(tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[('CSV', '*.csv')])
        if not path:
            return
        df = pd.read_csv(path)
        if 'x' in df.columns and 'y' in df.columns:
            self.x_text.delete('0.0','end')
            self.y_text.delete('0.0','end')
            self.x_text.insert('0.0', ','.join(map(str, df['x'].tolist())))
            self.y_text.insert('0.0', ','.join(map(str, df['y'].tolist())))

    def fit(self):
        try:
            xs = np.array([float(v) for v in self.x_text.get('0.0','end').strip().split(',') if v.strip()])
            ys = np.array([float(v) for v in self.y_text.get('0.0','end').strip().split(',') if v.strip()])
        except Exception as e:
            self.result_label.configure(text=f"Erreur lecture: {e}")
            return

        coef, intercept, predict = fit_linear_regression(xs, ys)
        self.result_label.configure(text=f"y = {coef:.4f} x + {intercept:.4f}")

        self.ax.clear()
        self.ax.scatter(xs, ys, label='données')
        xs_line = np.linspace(xs.min(), xs.max(), 200)
        ys_line = predict(xs_line)
        self.ax.plot(xs_line, ys_line, label='régression')
        self.ax.legend()
        self.canvas.draw()