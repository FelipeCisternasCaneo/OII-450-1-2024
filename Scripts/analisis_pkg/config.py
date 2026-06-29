import os
from util.util import cargar_configuracion, cargar_configuracion_exp
from solver.domain_managers import ensure_registered
from solver.domain_managers.registry import get_all as get_all_domains

# ========= Config Files =========
CONFIG_FILE = "./util/json/dir.json"
EXPERIMENTS_FILE = "./util/json/experiments_config.json"
ANALYSIS_FILE = "./util/json/analysis.json"

CONFIG, EXPERIMENTS = cargar_configuracion_exp(CONFIG_FILE, EXPERIMENTS_FILE)
ANALYSIS_CONFIG = cargar_configuracion(ANALYSIS_FILE)

# Modo de terminación
MODO_TERMINACION = "iter"
if isinstance(EXPERIMENTS, dict):
    exp_seccion = EXPERIMENTS.get("experimentos", {})
    MODO_TERMINACION = exp_seccion.get("modo_terminacion", "iter").lower()

# Configuración de performance
BATCH_SIZE = ANALYSIS_CONFIG["performance"]["batch_size"]
GC_INTERVAL = ANALYSIS_CONFIG["performance"]["gc_interval"]
RAM_WARNING = ANALYSIS_CONFIG["performance"]["ram_warning"]
CACHE_MAX_SIZE = ANALYSIS_CONFIG["performance"]["cache_max_size"]

# Configuración de gráficos
GRAFICOS_POR_CORRIDA = ANALYSIS_CONFIG["graficos"]["graficos_por_corrida"]
MODO_LOGARITMICO = ANALYSIS_CONFIG["graficos"]["modo_logaritmico"]

# Directorios de salida
DIRS = CONFIG["dirs"]
DIR_FITNESS = DIRS["fitness"]
DIR_RESUMEN = DIRS["resumen"]
DIR_TRANSITORIO = DIRS["transitorio"]
DIR_GRAFICOS = DIRS["graficos"]
DIR_BEST = DIRS["best"]
DIR_BOXPLOT = DIRS["boxplot"]
DIR_VIOLIN = DIRS["violinplot"]

# Lista de metaheurísticas a procesar
MHS_LIST = EXPERIMENTS["mhs"]


# ========= Modelo de datos =========
class InstancesMhs:
    def __init__(self):
        self.div = []
        self.fitness = []
        self.time = []
        self.xpl = []
        self.xpt = []
        self.bestFitness = []
        self.bestTime = []
        # diversidad & gaps
        self.ent = []
        self.divj_mean = []
        self.divj_min = []
        self.divj_max = []
        self.gap = []
        self.rdp = []
        # series representativas (para gráficos "best")
        self.xpl_iter = None
        self.xpt_iter = None
        self.iter_vector = None


# ========= Parametrización por problema (desde Domain Registry) =========
ensure_registered()
PROBLEMS = {dtype: entry.analysis_meta for dtype, entry in get_all_domains().items()}


# ========= Mapeo de columnas (tolerante a variaciones) =========
COLUMN_MAPPINGS = {
    "best_fitness": [
        "best_fitness",
        "bestFitness",
        "fitness",
        "Fitness",
        "best fitness",
    ],
    "time": ["time", "Time", "tiempo", "Tiempo"],
    "XPL": ["XPL", "xpl", "exploration", "Exploration"],
    "XPT": ["XPT", "xpt", "exploitation", "Exploitation"],
    "iter": ["iter", "iteration", "Iteration", "iteracion", "Iteracion"],
    "ENT": ["ENT", "ent", "entropy", "Entropy"],
    "Divj_mean": ["Divj_mean", "divj_mean", "diversity_mean"],
    "Divj_min": ["Divj_min", "divj_min", "diversity_min"],
    "Divj_max": ["Divj_max", "divj_max", "diversity_max"],
    "GAP": ["GAP", "gap", "Gap"],
    "RDP": ["RDP", "rdp", "Rdp"],
}
