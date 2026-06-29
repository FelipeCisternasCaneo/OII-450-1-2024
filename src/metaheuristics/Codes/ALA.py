# Metaheuristics/Codes/ALA.py
import numpy as np
import math

# ---------------------------------------------------------------------
# Utilidades 
# ---------------------------------------------------------------------

def _to_vec_bound(b, dim, name):
    """Convierte un bound escalar a vector (dim,), o valida shape si ya es vector."""
    if b is None:
        return None
    if np.isscalar(b):
        return np.full(dim, float(b), dtype=float)
    b = np.asarray(b, dtype=float)
    if b.shape != (dim,):
        raise ValueError(f"{name} shape {b.shape} incompatible with dim={dim}")
    return b


def _levy_step(dim, beta=1.5, rng=np.random):
    """Vuelo de Lévy (Mantegna) con parámetro beta=1.5 """
    sigma = (math.gamma(1.0 + beta) * math.sin(math.pi * beta / 2.0) /
             (math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))) ** (1.0 / beta)
    u = rng.normal(0.0, sigma, size=dim)
    v = rng.normal(0.0, 1.0, size=dim)
    return u / (np.abs(v) ** (1.0 / beta))


def _safe_eval(fo, x):
    """
    Evalúa fo(x) y devuelve un float.
    - Soporta retornos tipo: f, (f, x), (x, f), (algo, ..., f), numpy escalares, etc.
    - Si no encuentra un escalar numérico, devuelve +inf.
    - Sanea NaN/Inf devolviendo +inf.
    """
    out = fo(x)

    def _extract_numeric(obj):
        # Numéricos nativos / numpy escalares
        if isinstance(obj, (int, float, np.integer, np.floating)):
            return float(obj)
        # Arrays: solo si son escalares de tamaño 1
        if isinstance(obj, np.ndarray):
            if obj.size == 1:
                return float(obj.reshape(-1)[0])
            return None
        # Tuplas / listas: busca algún escalar numérico adentro (preferimos el ÚLTIMO, como muchos benchmarks)
        if isinstance(obj, (tuple, list)):
            # primero intenta el segundo elemento (convención común (x, f))
            if len(obj) >= 2:
                val = _extract_numeric(obj[1])
                if val is not None:
                    return val
            # si no, intenta el primero
            if len(obj) >= 1:
                val = _extract_numeric(obj[0])
                if val is not None:
                    return val
            # y por si acaso, recorre de derecha a izquierda
            for elem in reversed(obj):
                val = _extract_numeric(elem)
                if val is not None:
                    return val
            return None
        # Último intento: castear directo 
        try:
            arr = np.array(obj, dtype=float).ravel()
            if arr.size:
                return float(arr[-1])
        except Exception:
            pass
        return None

    val = _extract_numeric(out)
    if val is None or not np.isfinite(val):
        return float('inf')
    return float(val)



def _initialization(N, dim, ub, lb):
    """
    Réplica del initialization(N,dim,ub,lb) del MATLAB:
    - Si lb/ub son escalares: uniforme en [lb, ub] por componente.
    - Si son vectores (dim,): se vectoriza el muestreo por componente.
    """
    lb_vec = _to_vec_bound(lb, dim, 'lb') if lb is not None else np.full(dim, -np.inf)
    ub_vec = _to_vec_bound(ub, dim, 'ub') if ub is not None else np.full(dim,  np.inf)
    # muestreo uniforme componente a componente
    r = np.random.rand(N, dim)
    return lb_vec + r * (ub_vec - lb_vec)


# ---------------------------------------------------------------------
# Implementación
# 
# ---------------------------------------------------------------------
def ALA(N, Max_iter, lb, ub, dim, fobj):
    """
    Artificial Lemming Algorithm :
    - N: número de agentes
    - Max_iter: iteraciones máximas
    - lb, ub: límites (escalares o vectores de tamaño dim)
    - dim: dimensión
    - fobj: función objetivo (minimización), fobj(x)->float
    Retorna: (Score, Position, Convergence_curve)
    """
    # Inicialización
    X = _initialization(N, dim, ub, lb)                  # población (N x dim)
    Position = np.zeros(dim, dtype=float)                # mejor posición
    Score = float('inf')                                  # mejor valor
    fitness = np.zeros(N, dtype=float)                   # fitness por individuo
    Convergence_curve = np.zeros(Max_iter, dtype=float)  # registro por iteración

    # Evaluación inicial y mejor global
    for i in range(N):
        fi = _safe_eval(fobj, X[i, :])
        fitness[i] = fi
        if fi < Score:
            Score = fi
            Position = X[i, :].copy()

    # Bucle principal (Iter = 1..Max_iter)
    for Iter in range(1, Max_iter + 1):
        # Parámetros por iteración (congelados)
        RB = np.random.randn(N, dim)                                # Browniano
        F = 1.0 if np.random.rand() < 0.5 else -1.0                 # bandera direccional
        theta = 2.0 * np.arctan(1.0 - (Iter / Max_iter))            # parámetro temporal

        # --- Generar TODOS los candidatos con Position CONGELADO ---
        Xnew = np.empty_like(X)
        best_prev = Position.copy()  # "Position" congelado durante la generación 

        for i in range(N):
            # E = 2*log(1/rand)*theta; usa rand en (0,1] para evitar log(0)
            u = max(np.random.rand(), 1e-300)
            E = 2.0 * np.log(1.0 / u) * theta

            Xi = X[i, :]
            j = np.random.randint(0, N)       # puede coincidir con i (como randi(N) en MATLAB)
            Xrand = X[j, :]

            if E > 1.0:
                if np.random.rand() < 0.3:
                    # Migración (30%)
                    r1 = 2.0 * np.random.rand(dim) - 1.0
                    step_mix = r1 * (best_prev - Xi) + (1.0 - r1) * (Xi - Xrand)
                    Xcand = best_prev + F * RB[i, :] * step_mix
                else:
                    # Madrigueras (70%)
                    r2 = np.random.rand() * (1.0 + np.sin(0.5 * Iter))
                    Xcand = Xi + F * r2 * (best_prev - Xrand)
            else:
                if np.random.rand() < 0.5:
                    # Forrajeo (espiral)
                    radius = np.linalg.norm(best_prev - Xi)
                    r3 = np.random.rand()
                    spiral = radius * (np.sin(2.0 * np.pi * r3) + np.cos(2.0 * np.pi * r3))
                    Xcand = best_prev + F * Xi * spiral * np.random.rand()
                else:
                    # Evasión (Lévy)
                    G = 2.0 * (1.0 if np.random.rand() >= 0.5 else -1.0) * (1.0 - (Iter / Max_iter))
                    step_levy = _levy_step(dim, beta=1.5)
                    Xcand = best_prev + F * G * step_levy * (best_prev - Xi)

            # Corrección de límites 
            lb_vec = _to_vec_bound(lb, dim, 'lb') if lb is not None else np.full(dim, -np.inf)
            ub_vec = _to_vec_bound(ub, dim, 'ub') if ub is not None else np.full(dim,  np.inf)
            Xnew[i, :] = np.clip(Xcand, lb_vec, ub_vec)

        # --- Evaluación + aceptación codiciosa + actualización del mejor ---
        for i in range(N):
            newPopfit = _safe_eval(fobj, Xnew[i, :])
            if newPopfit < fitness[i]:
                X[i, :] = Xnew[i, :]
                fitness[i] = newPopfit
            if fitness[i] < Score:
                Position = X[i, :].copy()
                Score = fitness[i]

        # Registrar curva de convergencia
        Convergence_curve[Iter - 1] = Score

    return Score, Position, Convergence_curve


# ---------------------------------------------------------------------
# Wrappers de iteración única
# ---------------------------------------------------------------------

def _iterarALA_core(maxIter, iter, dim, population, best, fo, lb, ub, old_fitness=None):
    """
    UNA iteración (mejor congelado al generar candidatos).
    Devuelve: (new_population, new_fitness)
    """
    pop = np.asarray(population, dtype=float)
    if pop.ndim != 2 or pop.shape[1] != dim:
        raise ValueError(f"population shape {pop.shape} incompatible with dim={dim}")
    N = pop.shape[0]

    lb_vec = _to_vec_bound(lb, dim, 'lb') if lb is not None else np.full(dim, -np.inf)
    ub_vec = _to_vec_bound(ub, dim, 'ub') if ub is not None else np.full(dim,  np.inf)

    if old_fitness is None:
        old_fitness = np.array([_safe_eval(fo, pop[i]) for i in range(N)], dtype=float)
    else:
        old_fitness = np.asarray(old_fitness, dtype=float).copy()
        if old_fitness.shape != (N,):
            raise ValueError(f"fitness shape {old_fitness.shape} incompatible with N={N}")

    # Parámetros por iteración
    ratio = float(iter) / float(maxIter) if maxIter > 0 else 1.0
    ratio = min(max(ratio, 0.0), 1.0)
    theta = 2.0 * np.arctan(1.0 - ratio)
    F = 1.0 if np.random.rand() < 0.5 else -1.0
    RB = np.random.randn(N, dim)

    best_prev = np.asarray(best, dtype=float).copy()

    # Generación de candidatos con best_prev congelado
    Xcand = np.empty_like(pop)
    for i in range(N):
        u = max(np.random.rand(), 1e-300)
        E = 2.0 * np.log(1.0 / u) * theta

        Xi = pop[i]
        j = np.random.randint(0, N)  # puede ser i
        Xrand = pop[j]

        if E > 1.0:
            if np.random.rand() < 0.3:
                r1 = 2.0 * np.random.rand(dim) - 1.0
                step_mix = r1 * (best_prev - Xi) + (1.0 - r1) * (Xi - Xrand)
                Xnew = best_prev + F * RB[i] * step_mix
            else:
                r2 = np.random.rand() * (1.0 + np.sin(0.5 * iter))
                Xnew = Xi + F * r2 * (best_prev - Xrand)
        else:
            if np.random.rand() < 0.5:
                radius = np.linalg.norm(best_prev - Xi)
                r3 = np.random.rand()
                spiral = radius * (np.sin(2.0 * np.pi * r3) + np.cos(2.0 * np.pi * r3))
                Xnew = best_prev + F * Xi * spiral * np.random.rand()
            else:
                G = 2.0 * (1.0 if np.random.rand() >= 0.5 else -1.0) * (1.0 - ratio)
                step_levy = _levy_step(dim, beta=1.5)
                Xnew = best_prev + F * G * step_levy * (best_prev - Xi)

        Xcand[i] = np.clip(Xnew, lb_vec, ub_vec)

    # Evaluación + aceptación + mejor global (interno a la iteración)
    new_population = pop.copy()
    new_fitness = old_fitness.copy()
    fbest = _safe_eval(fo, best_prev)
    best_vec = best_prev.copy()

    for i in range(N):
        fit_i = _safe_eval(fo, Xcand[i])
        if fit_i < new_fitness[i]:
            new_population[i] = Xcand[i]
            new_fitness[i] = fit_i
        if new_fitness[i] < fbest:
            fbest = new_fitness[i]
            best_vec = new_population[i].copy()

    return new_population, new_fitness


def iterarALA(maxIter, iter, dim, population, fitness, best, fo, lb, ub):
    """Wrapper: devuelve (new_population, new_fitness)."""
    return _iterarALA_core(
        maxIter=maxIter, iter=iter, dim=dim,
        population=population, best=best, fo=fo,
        lb=lb, ub=ub, old_fitness=fitness
    )