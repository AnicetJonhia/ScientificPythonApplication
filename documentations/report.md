# Rapport technique — Math Solver App

## 1. Introduction
Courte présentation du projet et des objectifs.

## 2. Architecture du projet
- /core : fonctions de calcul (systèmes linéaires, PL, régression, stochastique)
- /ui : modules Streamlit pour chaque fonctionnalité
- /data : jeux de données d'exemple

## 3. Modules et algorithmes
### 3.1 Systèmes linéaires
- Méthodes disponibles : résolution directe (numpy.linalg.solve), elimination de Gauss (avec pivot partiel), décomposition LU (Doolittle), moindres carrés (lstsq).
- Cas test : matrice 3x3 et vecteur b (exemple dans l'application).

### 3.2 Programmation linéaire
- Modélisation via PuLP (si installé). Supporte maximisation/minimisation, contraintes <=, >=, ==, et bornes de variables.
- Exemple test : Max Z = 3x + 2y sous contraintes illustratives.

### 3.3 Régression linéaire
- Utilise scikit-learn LinearRegression si disponible, sinon numpy.polyfit en fallback.
- Mesures : R², RMSE. Support d'upload CSV (colonnes x,y) et mise à jour automatique des champs.

### 3.4 Processus stochastiques
- Simulation de chaînes de Markov, distribution empirique, estimation de la distribution stationnaire (valeur propre principale ou power iteration).

## 4. Interface utilisateur
- Application Streamlit avec menu latéral permettant d'accéder aux 4 modules.
- Chaque module : saisie des données, upload CSV, exécution, visualisation (Plotly/Matplotlib pour certains graphiques).

## 5. Jeux de tests et captures d'écran
(liste des jeux de tests fournis dans /data et captures à insérer)

## 6. Limitations et pistes d'amélioration
- Décomposition LU sans pivot génère des erreurs sur certaines matrices; on peut ajouter pivotage complet.
- Interface UX : possibilité d'ajouter éditeur de matrice plus ergonomique.

## 7. Conclusion
Résumé des fonctionnalités et recommandations.

---
