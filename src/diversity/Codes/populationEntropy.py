import numpy as np

def population_entropy(population: np.ndarray, bins: int = 20, lb=None, ub=None):
    """
    Calculates normalized entropy per dimension (0–1 scale)
    using 1D histograms.
    E(t) = -Σ p_i * log(p_i) / log(bins)

    Args:
        population: (N x D) numpy array
        bins: number of bins (default ~ sqrt(N))
        lb, ub: optional lower/upper bounds (lists or arrays)
    Returns:
        ent_avg: mean entropy across dimensions (float)
        ent_dim: per-dimension entropy (np.ndarray)
    """
    N, D = population.shape
    ent_dim = np.zeros(D, dtype=float)
    bins = max(2, min(bins, int(np.sqrt(max(N, 2)))))

    for j in range(D):
        x = population[:, j]
        if lb is not None and ub is not None:
            rmin, rmax = lb[j], ub[j]
        else:
            rmin, rmax = float(np.min(x)), float(np.max(x))
            if rmax == rmin:
                ent_dim[j] = 0.0
                continue
            eps = 1e-12 * max(1.0, abs(rmax - rmin))
            rmin -= eps
            rmax += eps

        hist, _ = np.histogram(x, bins=bins, range=(rmin, rmax))
        p = hist.astype(float) / N
        p = p[p > 0.0]
        if p.size == 0:
            ent_dim[j] = 0.0
        else:
            ent = -np.sum(p * np.log(p))
            ent_dim[j] = float(ent / np.log(bins))

    ent_avg = float(np.mean(ent_dim)) if D > 0 else 0.0
    return ent_avg, ent_dim