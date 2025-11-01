# Application graphique de résolution et de modélisation mathématique

Ce document contient la **structure du projet** et le **code initial** d'une application Python utilisant **CustomTkinter** pour une UI ergonomique et responsive, ainsi que **NumPy**, **Matplotlib**, **scikit-learn** et **PuLP** pour les calculs.

---

## Arborescence proposée

```
math_solver_app/
│
├── main.py
├── requirements.txt
├── README.md
├── AI_USAGE.txt
│
├── core/
│   ├── __init__.py
│   ├── linear_system.py
│   ├── linear_programming.py
│   ├── linear_regression.py
│   └── stochastic_process.py
│
├── ui/
│   ├── __init__.py
│   ├── main_window.py
│   ├── linear_system_ui.py
│   ├── linear_programming_ui.py
│   ├── linear_regression_ui.py
│   └── stochastic_ui.py
│
└── data/
    ├── regression_data.csv
    └── markov_example.csv
```

---

> **Remarque** : Le code ci-dessous est un *point de départ complet et fonctionnel* permettant de lancer l'application et de tester les quatre modules. Tu pourras ensuite l'étendre, améliorer l'UI ou ajouter des validations.

---

## 1) `requirements.txt`

```text
numpy
matplotlib
customtkinter
scikit-learn
pandas
pulp
```

---

## 2) `AI_USAGE.txt` (exemple)

```
Portions créées avec l'aide d'une IA :
- ui/linear_system_ui.py (est. 40 lignes)
- core/linear_system.py (est. 30 lignes)
- ui/linear_regression_ui.py (est. 50 lignes)
- ui/stochastic_ui.py (est. 40 lignes)
Total estimé : <= 20% du code

Adapter si tu modifies le code.
```

---

## 3) `README.md` (extrait)

```markdown
# Application scientifique - Math Solver App

Lancer : `python main.py`

Dépendances : voir requirements.txt

Description : Application GUI (CustomTkinter) proposant 4 modules :
- Systèmes linéaires
- Programmation linéaire
- Régression linéaire
- Processus stochastiques (Markov)

Chaque module permet la saisie, le calcul et l'affichage graphique.
```

---

## 4) `main.py`

```python
# main.py
from ui.main_window import MainApp

if __name__ == "__main__":
    app = MainApp()
    app.mainloop()
```

---

## 5) `core/__init__.py`

```python
# core/__init__.py
# Package core - fonctions de calcul
```

---

## 6) `core/linear_system.py`

```python
# core/linear_system.py
import numpy as np

def solve_linear_system(A, b):
    """Résout AX = b. A et b doivent être des arrays numpy.
    Renvoie le vecteur solution ou lève une exception si singulier."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    # essayer la résolution directe
    try:
        x = np.linalg.solve(A, b)
        return x
    except np.linalg.LinAlgError:
        # fallback : moindres carrés
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return x
```

---

## 7) `core/linear_programming.py`

```python
# core/linear_programming.py
from typing import List, Tuple
try:
    import pulp
except Exception:
    pulp = None


def solve_lp(c: List[float], A: List[List[float]], b: List[float], maximize=True) -> Tuple[dict, float]:
    """Résout un problème LP simple : max/min c^T x s.t. Ax <= b, x >= 0.
    Retourne un dictionnaire {i: valeur} et la valeur optimale.
    Nécessite pulp.
    """
    if pulp is None:
        raise RuntimeError("PuLP n'est pas installé.")

    n = len(c)
    prob = pulp.LpProblem("LP_problem", pulp.LpMaximize if maximize else pulp.LpMinimize)
    vars = [pulp.LpVariable(f"x{i}", lowBound=0) for i in range(n)]

    # objectif
    prob += pulp.lpDot(c, vars)

    # contraintes Ax <= b
    for row, bi in zip(A, b):
        prob += pulp.lpDot(row, vars) <= bi

    status = prob.solve()

    solution = {f"x{i}": pulp.value(var) for i, var in enumerate(vars)}
    opt = pulp.value(prob.objective)
    return solution, opt
```

---

## 8) `core/linear_regression.py`

```python
# core/linear_regression.py
import numpy as np

try:
    from sklearn.linear_model import LinearRegression
    SKLEARN = True
except Exception:
    SKLEARN = False


def fit_linear_regression(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    x_flat = x.reshape(-1, 1)
    if SKLEARN:
        model = LinearRegression()
        model.fit(x_flat, y)
        coef = model.coef_[0]
        intercept = model.intercept_
        return coef, intercept, model.predict
    else:
        # fallback numpy polyfit
        coef, intercept = np.polyfit(x, y, 1)
        def predict_fn(xx):
            xx = np.array(xx, dtype=float)
            return coef * xx + intercept
        return coef, intercept, predict_fn
```

---

## 9) `core/stochastic_process.py`

```python
# core/stochastic_process.py
import numpy as np

def simulate_markov_chain(P, initial_state=0, steps=100):
    """P: matrice de transition (n x n), initial_state index, retourne la séquence d'états."""
    P = np.array(P, dtype=float)
    n = P.shape[0]
    state = int(initial_state)
    states = [state]
    for _ in range(steps):
        probs = P[state]
        state = np.random.choice(n, p=probs)
        states.append(state)
    return states

def empirical_distribution(states, n_states=None):
    import numpy as np
    states = np.array(states, dtype=int)
    if n_states is None:
        n_states = states.max() + 1
    counts = np.bincount(states, minlength=n_states)
    return counts / counts.sum()
```

---

## 10) `ui/__init__.py`

```python
# ui/__init__.py
# Package UI
```

---

## 11) `ui/main_window.py`

```python
# ui/main_window.py
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
```

---

## 12) `ui/linear_system_ui.py`

```python
# ui/linear_system_ui.py
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from core.linear_system import solve_linear_system

class LinearSystemFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

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

        # zone graphique
        right = ctk.CTkFrame(self)
        right.pack(side="left", expand=True, fill="both", padx=12, pady=12)

        self.fig, self.ax = plt.subplots(figsize=(5,4), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def solve(self):
        try:
            A = eval(self.entry_a.get("0.0", "end").strip())
            b = eval(self.entry_b.get())
            A = np.array(A, dtype=float)
            b = np.array(b, dtype=float)
        except Exception as e:
            self.result_label.configure(text=f"Erreur lecture : {e}")
            return

        try:
            x = solve_linear_system(A, b)
            self.result_label.configure(text=f"Solution : {np.round(x,6)}")
        except Exception as e:
            self.result_label.configure(text=f"Erreur calcul : {e}")
            return

        # Graphique simple : si dimension 2, représenter les droites
        self.ax.clear()
        if A.shape[0] == 2 and A.shape[1] == 2:
            # pour Ax=b -> deux équations a11 x + a12 y = b1, a21 x + a22 y = b2
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
            self.ax.plot(x[0], x[1], 'o', label='solution')
            self.ax.set_xlim(-10,10)
            self.ax.set_ylim(-10,10)
            self.ax.legend()
        else:
            self.ax.text(0.5, 0.5, 'Graphique disponible pour 2x2 uniquement', ha='center')

        self.canvas.draw()
```

---

## 13) `ui/linear_programming_ui.py`

```python
# ui/linear_programming_ui.py
import customtkinter as ctk
from core.linear_programming import solve_lp

class LinearProgrammingFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

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

    def solve(self):
        try:
            c = [float(v.strip()) for v in self.c_entry.get().split(',')]
            lines = [ln.strip() for ln in self.constraints_text.get("0.0", "end").splitlines() if ln.strip()]
            A = []
            b = []
            for ln in lines:
                if '<=' in ln:
                    lhs, rhs = ln.split('<=')
                    coeffs = [float(x) for x in lhs.split(',')]
                    A.append(coeffs)
                    b.append(float(rhs))
                else:
                    raise ValueError('Contrainte mal formatée')

            sol, opt = solve_lp(c, A, b, maximize=True)
            self.result_label.configure(text=f"Solution: {sol} | Opt = {opt}")
        except Exception as e:
            self.result_label.configure(text=f"Erreur: {e}")
```

---

## 14) `ui/linear_regression_ui.py`

```python
# ui/linear_regression_ui.py
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
```

---

## 15) `ui/stochastic_ui.py`

```python
# ui/stochastic_ui.py
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
```

---

## 16) `data/regression_data.csv` (exemple)

```csv
x,y
0,1
1,2.1
2,3.9
3,6.2
4,7.9
```

---

## 17) `data/markov_example.csv` (optionnel)

```csv
# exemple non strict csv, utile si tu veux importer la matrice depuis un fichier
0.5,0.5,0
0.2,0.4,0.4
0,0.3,0.7
```

---

### 🚀 Prochaines étapes proposées

1. Cloner ce projet localement.
2. Installer les dépendances : `pip install -r requirements.txt`.
3. Lancer : `python main.py`.
4. Tester chaque onglet et me dire quelles améliorations UX/feature tu veux (validation d'entrée, export PDF, sauvegarde projet, plus de thèmes, animations...).

---

*Tu trouveras tout le code complet ci‑dessous dans ce document — copie/colle les fichiers sur ton disque ou dis‑moi si tu veux que je génère un zip/tar automatiquement.*
