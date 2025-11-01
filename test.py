import customtkinter as ctk

# Configuration globale
ctk.set_appearance_mode("dark")  # "light", "dark", ou "system"
ctk.set_default_color_theme("green")  # thèmes : "blue", "dark-blue", "green"

# Création de la fenêtre principale
root = ctk.CTk()
root.title("Application moderne")
root.geometry("400x250")

# Ajout de widgets modernes
label = ctk.CTkLabel(root, text="Bienvenue sur CustomTkinter 👋", font=("Arial", 16))
label.pack(pady=20)

entry = ctk.CTkEntry(root, placeholder_text="Votre nom...")
entry.pack(pady=10)

def dire_bonjour():
    nom = entry.get()
    label.configure(text=f"Bonjour, {nom or 'inconnu'} !")

button = ctk.CTkButton(root, text="Dire bonjour", command=dire_bonjour)
button.pack(pady=10)

root.mainloop()
