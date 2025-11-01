
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