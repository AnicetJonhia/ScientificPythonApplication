
import numpy as np

def simulate_markov_chain(P, initial_state=0, steps=100, rng=None):
    """
    desc:
        Simule une chaîne de Markov discrète à partir d'une matrice de transition donnée.  
        À chaque étape, l'état suivant est choisi aléatoirement selon les probabilités de transition.

    params:
        P : Matrice carrée (liste de listes ou tableau numpy) représentant les probabilités de transition entre états.  
        initial_state : Indice de l’état initial (entier, par défaut 0).  
        steps : Nombre d’étapes de simulation (entier, par défaut 100).

    return:
        states : Liste des états visités pendant la simulation (longueur = steps + 1).
    """

    P = np.array(P, dtype=float)
    n = P.shape[0]
    state = int(initial_state)
    states = [state]
    # use provided RNG (numpy RandomState/Generator) if given for reproducibility
    if rng is None:
        rng = np.random
    for _ in range(steps):
        probs = P[state]
        state = rng.choice(n, p=probs)
        states.append(state)
    return states

def empirical_distribution(states, n_states=None):
    """
    desc:
        Calcule la distribution empirique (fréquence relative) des états visités dans une chaîne de Markov.  
        Permet d’estimer la probabilité stationnaire à partir des observations simulées.

    params:
        states : Liste ou tableau numpy contenant la séquence d’états visités.  
        n_states : Nombre total d’états possibles (optionnel, déduit automatiquement si None).

    return:
        distribution : Tableau numpy représentant la fréquence relative de chaque état (somme = 1).
    """
    import numpy as np
    states = np.array(states, dtype=int)
    if n_states is None:
        n_states = states.max() + 1
    counts = np.bincount(states, minlength=n_states)
    return counts / counts.sum()