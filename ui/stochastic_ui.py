
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from core.stochastic_process import simulate_markov_chain, empirical_distribution

class StochasticFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        left = ctk.CTkFrame(self, width=320)
        left.pack(side='left', fill='y', padx=12, pady=12)

        ctk.CTkLabel(left, text='Matrice de transition (ex: [[0.5,0.5,0],[0.2,0.4,0.4],[0,0.3,0.7]])').pack(anchor='w')
        self.P_text = ctk.CTkTextbox(left, height=6)
        self.P_text.insert('0.0', '[[0.5,0.5,0],[0.2,0.4,0.4],[0,0.3,0.7]]')
        self.P_text.pack(fill='x', pady=6)

        ctk.CTkLabel(left, text='Etat initial (index)').pack(anchor='w')
        self.init_entry = ctk.CTkEntry(left)
        self.init_entry.insert(0, '0')
        self.init_entry.pack(fill='x', pady=6)

        self.steps_entry = ctk.CTkEntry(left)
        self.steps_entry.insert(0, '200')
        ctk.CTkLabel(left, text='Nombre d étapes').pack(anchor='w')
        self.steps_entry.pack(fill='x', pady=6)

        ctk.CTkButton(left, text='Simuler', command=self.simulate).pack(pady=8)
        self.result_label = ctk.CTkLabel(left, text='-')
        self.result_label.pack(pady=6)

        right = ctk.CTkFrame(self)
        right.pack(side='left', expand=True, fill='both', padx=12, pady=12)
        self.fig, self.ax = plt.subplots(tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def simulate(self):
        try:
            P = eval(self.P_text.get('0.0','end'))
            P = np.array(P, dtype=float)
            init = int(self.init_entry.get())
            steps = int(self.steps_entry.get())
        except Exception as e:
            self.result_label.configure(text=f'Erreur lecture: {e}')
            return

        states = simulate_markov_chain(P, initial_state=init, steps=steps)
        dist = empirical_distribution(states, n_states=P.shape[0])
        self.result_label.configure(text=f'Distribution empirique finale: {np.round(dist,4)}')

        self.ax.clear()
        # tracé de la trajectoire des états
        self.ax.plot(states, alpha=0.7)
        self.ax.set_ylabel('Etat')
        self.ax.set_xlabel('Temps')
        self.canvas.draw()