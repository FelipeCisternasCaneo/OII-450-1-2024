import numpy as np

# ==== VARIABLES DE ESTADO PERSISTENTES ====
_FEs = 0
_rec = 1
_rM = None
_rM_cost = None
_initialized = False
_best_cost = np.inf
_best_M = None

def iterarCCMGO(iter, maxIter, dim, population, fitness, best, lb, ub, fo):
    """
    MGO fiel al paper, con CrissCross (HCS + VCS) integrado (CCMGO).
    - fo(x) devuelve (x_clipped, fitness) -> usamos fitness = fo(x)[1]
    - Mantiene estado global (FEs, rM, rM_cost, rec, best_M, best_cost)
    - CrissCross implementado según ecuaciones (12)-(14) del paper CCMGO.
    Referencias: CCMGO paper (Yue & Li, 2025). :contentReference[oaicite:2]{index=2}
    """
    global _FEs, _rec, _rM, _rM_cost, _initialized, _best_cost, _best_M

    # --- normalizar límites ---
    if isinstance(lb, list): lb = lb[0]
    if isinstance(ub, list): ub = ub[0]
    lb = float(lb); ub = float(ub)

    N = len(population)
    rec_num = 10
    d1 = 0.2
    w = 2.0

    # ---------- Inicialización ----------
    if not _initialized or iter == 0:
        _initialized = True
        _FEs = 0
        _rec = 1
        _best_cost = np.inf

        _rM = np.zeros((N, dim, rec_num))
        _rM_cost = np.full((N, rec_num), np.inf)

        # Inicializar población uniforme en [lb, ub]
        population = np.random.rand(N, dim) * (ub - lb) + lb

        for i in range(N):
            _, fitness[i] = fo(population[i])   # fitness = fo(...)[1]
            _FEs += 1
            if fitness[i] < _best_cost:
                _best_cost = fitness[i]
                _best_M = population[i].copy()

        # registrar primera generación
        for i in range(N):
            _rM[i, :, 0] = population[i]
            _rM_cost[i, 0] = fitness[i]

        _rec = 2
        return population

    # ---------- adapt dynamic divide_num (ecuación 15 del paper CCMGO) ----------
    # divide_num = ((FEs / MaxFEs) + 1) * dim / 4
    divide_num = int(max(1, (( _FEs / float(max(1, maxIter * N))) + 1.0) * (dim / 4.0)))
    # fallback si sale 0
    if divide_num < 1:
        divide_num = 1

    # ---------- Determination of wind direction (como antes) ----------
    calPositions = population.copy()
    div_perm = np.random.permutation(dim)
    for j in range(divide_num):
        idx = div_perm[j % dim]
        th = _best_M[idx]
        mask = calPositions[:, idx] > th
        # choose majority side
        if np.sum(mask) < calPositions.shape[0] / 2:
            mask = ~mask
        calPositions = calPositions[mask]
        if calPositions.shape[0] == 0:
            calPositions = population.copy()
            break

    # dirX and D_wind
    D = _best_M - calPositions
    D_wind = np.sum(D, axis=0) / float(calPositions.shape[0])

    # ---------- Spore dispersal / Dual propagation (igual que MGO) ----------
    beta = calPositions.shape[0] / float(N)
    gama = 1.0 / np.sqrt(max(1e-12, 1.0 - beta**2))
    E = 1.0 - (_FEs / float(max(1, maxIter * N)))   # fuerza del viento aproximada

    step = w * (np.random.rand(dim) - 0.5) * E
    step2 = 0.1 * w * (np.random.rand(dim) - 0.5) * E * (1.0 + 0.5 * (1.0 + np.tanh(beta / max(1e-12, gama))) * E)
    step3 = 0.1 * (np.random.rand() - 0.5) * E

    # act vector (actCal original): act_j = 1 if 1/(1.5 - 10*r5_j) >= 0.5
    denom = 1.0 + (0.5 - 10.0 * (np.random.rand(dim)))  # reorganización numerica estable
    # avoid div by zero
    safe = np.abs(denom) > 1e-12
    tmp = np.zeros(dim)
    tmp[safe] = 1.0 / denom[safe]
    act = (tmp >= 0.5).astype(float)

    # containers
    newM = population.copy()
    newM_cost = fitness.copy()

    # registrar primero si rec == 1
    if _rec == 1:
        for i in range(N):
            _rM[i, :, 0] = population[i]
            _rM_cost[i, 0] = fitness[i]
        _rec = 2

    # main individual updates (spore dispersal + dual propagation)
    for i in range(N):
        Mi = population[i].copy()
        if np.random.rand() > d1:
            Mn = Mi + step * D_wind
        else:
            Mn = Mi + step2 * D_wind

        # dual propagation with prob 0.8
        if np.random.rand() < 0.8:
            if np.random.rand() > 0.5:
                idx = div_perm[0]  # like matlab div_num(1)
                Mn[idx] = _best_M[idx] + step3 * D_wind[idx]
            else:
                Mn = (1.0 - act) * Mn + act * _best_M

        # boundary absorption
        Mn = np.clip(Mn, lb, ub)

        # evaluate (use fo -> tuple)
        _, fmn = fo(Mn)
        _FEs += 1

        # greedy update
        if fmn < newM_cost[i]:
            newM[i] = Mn.copy()
            newM_cost[i] = fmn

        # register for cryptobiosis
        if (_rec - 1) < rec_num:
            _rM[i, :, _rec - 1] = newM[i]
            _rM_cost[i, _rec - 1] = newM_cost[i]

        # update global best if improved
        if newM_cost[i] < _best_cost:
            _best_cost = newM_cost[i]
            _best_M = newM[i].copy()

    _rec += 1

    # ---------- Cryptobiosis (igual que paper) ----------
    if _rec > rec_num or _FEs >= maxIter * N:
        # elegir el mejor registro por individuo y reemplazar
        for i in range(N):
            # solo considerar registros válidos (0 .. _rec-2)
            last = min(_rec - 1, rec_num)
            idx_best = np.argmin(_rM_cost[i, :last])
            population[i] = _rM[i, :, idx_best].copy()
            fitness[i] = _rM_cost[i, idx_best]
            # actualizar global si corresponde
            if fitness[i] < _best_cost:
                _best_cost = fitness[i]
                _best_M = population[i].copy()

        # reset records
        _rM[:, :, :] = 0.0
        _rM_cost[:, :] = np.inf
        _rec = 1

    # ---------- Ahora aplicamos CrissCross (CC): HCS + VCS ----------
    # Según CCMGO: aplicar CC antes de terminar la iteración para mejorar intercambio
    # Implementación práctica (fiel a ecuaciones 12-14) y control del cómputo de FEs.
    # HCS: emparejar individuos aleatoriamente en pares, generar H1 (para i) y H2 (para k)
    # y hacer selección codiciosa. Esto produce ~N evaluaciones (N/2 pares * 2 evals).
    perm = np.random.permutation(N)
    # iterate pairs (if odd, last one pairs with a random partner)
    pairs = []
    for idx in range(0, N, 2):
        i = perm[idx]
        if idx + 1 < N:
            k = perm[idx + 1]
        else:
            # pair last with a random partner (not itself)
            choices = list(range(N))
            choices.remove(i)
            k = np.random.choice(choices)
        pairs.append((i, k))

    # HCS: for each pair (i,k) compute H_i and H_k per eq (12)-(13)
    for (i, k) in pairs:
        x_i = newM[i]
        x_k = newM[k]

        # r1,r2 are either scalars or vectors; paper uses scalars in [0,1], c1,c2 in [-1,1].
        r1 = np.random.rand(dim)
        r2 = np.random.rand(dim)
        c1 = np.random.uniform(-1.0, 1.0, size=dim)
        c2 = np.random.uniform(-1.0, 1.0, size=dim)

        H_i = r1 * x_i + (1.0 - r1) * x_k + c1 * (x_i - x_k)
        H_k = r2 * x_k + (1.0 - r2) * x_i + c2 * (x_k - x_i)

        # clip and evaluate: to keep FEs ~ N, evaluate H_i for i and H_k for k (one eval per offspring)
        H_i = np.clip(H_i, lb, ub)
        _, f_H_i = fo(H_i)
        _FEs += 1
        if f_H_i < newM_cost[i]:
            newM[i] = H_i.copy()
            newM_cost[i] = f_H_i
            if f_H_i < _best_cost:
                _best_cost = f_H_i
                _best_M = H_i.copy()

        H_k = np.clip(H_k, lb, ub)
        _, f_H_k = fo(H_k)
        _FEs += 1
        if f_H_k < newM_cost[k]:
            newM[k] = H_k.copy()
            newM_cost[k] = f_H_k
            if f_H_k < _best_cost:
                _best_cost = f_H_k
                _best_M = H_k.copy()

    # VCS: for each individual, pick two distinct dimensions j1,j2 and compute V (eq 14)
    # Evaluate V and adopt if better (greedy)
    for i in range(N):
        x = newM[i].copy()
        # choose two different dims
        j1, j2 = np.random.choice(dim, size=2, replace=False)
        r3 = np.random.rand()
        V = x.copy()
        V[j1] = r3 * x[j1] + (1.0 - r3) * x[j2]
        V = np.clip(V, lb, ub)
        _, fV = fo(V)
        _FEs += 1
        if fV < newM_cost[i]:
            newM[i] = V.copy()
            newM_cost[i] = fV
            if fV < _best_cost:
                _best_cost = fV
                _best_M = V.copy()

    # NOTA sobre conteo de FEs:
    # - En esta implementación contamos _FEs cada vez que llamamos fo(...).
    # - El paper contabiliza FEs en bloques (antes/después del CC); aquí el conteo es el real.
    # Si deseas reproducir exactamente la misma contabilidad de FEs del paper (p.ej.
    # incrementos por bloques en vez de por-evaluación), indícamelo y lo ajusto.

    # Finalmente, dejamos newM como la nueva población (con su fitness guardada en newM_cost)
    # y devolvemos la población actualizada.
    # Actualizamos los arrays de fitness proporcionados por el solver (por si el solver los reutiliza)
    for i in range(N):
        fitness[i] = newM_cost[i]

    return newM
