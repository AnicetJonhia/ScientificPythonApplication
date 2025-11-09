# Installation et lancement — Math Solver App

Ce guide explique comment installer et exécuter l'application sur Linux et Windows, créer un environnement virtuel et lancer l'application depuis le terminal avec la commande unique `python main.py`.

## Pré-requis
- Python 3.10+ recommandé.
- Git (optionnel) pour cloner le dépôt.

## Récupérer le projet
Dans votre terminal :

```bash
# cloner depuis GitHub (optionnel)
git clone https://github.com/AnicetJonhia/ScientificPythonApplication.git
cd "ScientificPythonApplication"  # ou le nom du dossier
```

Assurez-vous que votre `cwd` est le dossier du projet (celui contenant `main.py`, `app.py`, `requirements.txt`).

---

## Linux / macOS

1. Ouvrir un terminal.
2. Placer vous dans le dossier du projet :

```bash
cd /chemin/vers/ScientificPythonApplication
```

3. Créer et activer un environnement virtuel (venv) :

```bash
# créer
python3 -m venv venv
# activer
source venv/bin/activate
```

4. Installer les dépendances :

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

5. Lancer l'application (une seule commande) :

```bash
python main.py
```

Cette commande ouvre Streamlit et vous verrez l'URL locale . Ouvrez-la dans votre navigateur.

---

## Windows (PowerShell)

1. Ouvrir PowerShell.
2. Aller dans le dossier du projet :

```powershell
cd "C:\chemin\vers\ScientificPythonApplication"
```

3. Créer et activer un environnement virtuel :

```powershell
python -m venv venv
# activer
venv\Scripts\Activate.ps1
```


4. Installer les dépendances :

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

5. Lancer l'application :

```powershell
python main.py
```

---

## Remarques complémentaires

- Pour arrêter l'application, retournez dans le terminal et appuyez sur `Ctrl+C`.

- Si vous préférez lancer Streamlit directement :

```bash
streamlit run app.py
```

## Dépannage rapide
- Erreur "Could not find the 'streamlit' module": assurez-vous d'avoir activé l'environnement virtuel et installé les dépendances.
- Si un port est occupé (ex. 8501), Streamlit vous proposera un autre port libre

---
