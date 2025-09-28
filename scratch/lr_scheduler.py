import numpy as np

def cosine_annealing(lr_init, lr_final, t, T):
    """
    Cosine annealing learning rate scheduler.

    Args:
        lr_init (float): Initial learning rate ℓ0
        lr_final (float): Final learning rate ℓT
        t (int): Current iteration (0 ≤ t ≤ T)
        T (int): Total number of iterations

    Returns:
        float: Updated learning rate ℓt
    """
    return lr_final + 0.5 * (lr_init - lr_final) * (1 + np.cos(np.pi * t / T))
