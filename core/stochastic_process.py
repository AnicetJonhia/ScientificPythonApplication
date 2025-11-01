
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