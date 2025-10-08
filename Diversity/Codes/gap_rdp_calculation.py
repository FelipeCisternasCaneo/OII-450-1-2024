import numpy as np

def compute_gap_rdp(best: float, opt: float):
    """
    GAP = ((Best - Opt) / Best) * 100
    RDP = ((Best - Opt) / Opt) * 100
    Handles division by zero.
    """
    gap = ((best - opt) / best) * 100.0 if best not in (0.0, np.inf, -np.inf) else 0.0
    rdp = ((best - opt) / opt) * 100.0 if opt  not in (0.0, np.inf, -np.inf) else 0.0
    return float(gap), float(rdp)