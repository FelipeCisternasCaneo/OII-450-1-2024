import os
import sys

# Permite ejecutar este script directamente (python Scripts/analisis.py)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
import uuid
import time
import psutil

from Util.util import cargar_configuracion_exp, writeTofile
from BD.sqlite import BD
from Util.log import escribir_resumenes
import json

# ========= CACHÉ DE BD (Versión picklable) =========
_CACHE_BLOB = {}
_CACHE_INSTANCIAS = {}

def _obtener_blob_cached(instancia_id: str, incluir_binarizacion: bool):
    """Cachea los resultados de obtenerArchivos."""
    cache_key = (instancia_id, incluir_binarizacion)
    
    if cache_key in _CACHE_BLOB:
        return _CACHE_BLOB[cache_key]
    
    bd_local = BD()
    
    if incluir_binarizacion:
        blob = bd_local.obtenerArchivos(instancia_id)
    else:
        blob = bd_local.obtenerArchivos(instancia_id, incluir_binarizacion=False)
    
    result = tuple(blob) if blob else tuple()
    
    if len(_CACHE_BLOB) < CACHE_MAX_SIZE:
        _CACHE_BLOB[cache_key] = result
    
    return result

def _obtener_instancias_cached(lista_instancias_str: str):
    """Cachea las consultas de instancias."""
    if lista_instancias_str in _CACHE_INSTANCIAS:
        return _CACHE_INSTANCIAS[lista_instancias_str]
    
    bd_local = BD()
    instancias = bd_local.obtenerInstancias(lista_instancias_str)
    result = tuple(instancias) if instancias else tuple()
    _CACHE_INSTANCIAS[lista_instancias_str] = result
    
    return result

def limpiar_cache_bd():
    """Limpia el caché de BD."""
    global _CACHE_BLOB, _CACHE_INSTANCIAS
    _CACHE_BLOB.clear()
    _CACHE_INSTANCIAS.clear()
    print("[INFO] Caché de BD limpiado")

def _mostrar_estadisticas_cache():
    """Muestra estadísticas de uso del caché."""
    print(f"\n[CACHÉ] Estadísticas:")
    print(f"  Blobs en caché: {len(_CACHE_BLOB)}/{CACHE_MAX_SIZE}")
    print(f"  Instancias en caché: {len(_CACHE_INSTANCIAS)}")

# ========= Config =========
CONFIG_FILE = './util/json/dir.json'
EXPERIMENTS_FILE = './util/json/experiments_config.json'
ANALYSIS_FILE = './util/json/analysis.json'
CONFIG, EXPERIMENTS = cargar_configuracion_exp(CONFIG_FILE, EXPERIMENTS_FILE)

# Cargar configuración de análisis
with open(ANALYSIS_FILE, 'r', encoding='utf-8') as f:
    ANALYSIS_CONFIG = json.load(f)

# Configuración de performance
BATCH_SIZE = ANALYSIS_CONFIG['performance']['batch_size']
GC_INTERVAL = ANALYSIS_CONFIG['performance']['gc_interval']
RAM_WARNING = ANALYSIS_CONFIG['performance']['ram_warning']
CACHE_MAX_SIZE = ANALYSIS_CONFIG['performance']['cache_max_size']

# Configuración de gráficos
GRAFICOS_POR_CORRIDA = ANALYSIS_CONFIG['graficos']['graficos_por_corrida']
MODO_LOGARITMICO = ANALYSIS_CONFIG['graficos']['modo_logaritmico']

DIRS = CONFIG["dirs"]
DIR_FITNESS      = DIRS["fitness"]
DIR_RESUMEN      = DIRS["resumen"]
DIR_TRANSITORIO  = DIRS["transitorio"]
DIR_GRAFICOS     = DIRS["graficos"]
DIR_BEST         = DIRS["best"]
DIR_BOXPLOT      = DIRS["boxplot"]
DIR_VIOLIN       = DIRS["violinplot"]

MHS_LIST = EXPERIMENTS["mhs"]
bd = BD()

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

# ========= Parametrización por problema =========
PROBLEMS = {
    # name: {subfolder, inst_key, uses_bin, title_prefix, obtenerArchivos_kwargs}
    "BEN":  dict(sub="BEN",  inst_key="BEN",  uses_bin=False, title_prefix="",     obtenerArchivos_kwargs={"incluir_binarizacion": False}),
    "SCP":  dict(sub="SCP",  inst_key="SCP",  uses_bin=True,  title_prefix="scp",  obtenerArchivos_kwargs={}),
    "USCP": dict(sub="USCP", inst_key="USCP", uses_bin=True,  title_prefix="uscp", obtenerArchivos_kwargs={}),
}

# ========= Mapeo de columnas (tolerante a variaciones) =========
COLUMN_MAPPINGS = {
    'best_fitness': ['best_fitness', 'bestFitness', 'fitness', 'Fitness', 'best fitness'],
    'time': ['time', 'Time', 'tiempo', 'Tiempo'],
    'XPL': ['XPL', 'xpl', 'exploration', 'Exploration'],
    'XPT': ['XPT', 'xpt', 'exploitation', 'Exploitation'],
    'iter': ['iter', 'iteration', 'Iteration', 'iteracion', 'Iteracion'],
    'ENT': ['ENT', 'ent', 'entropy', 'Entropy'],
    'Divj_mean': ['Divj_mean', 'divj_mean', 'diversity_mean'],
    'Divj_min': ['Divj_min', 'divj_min', 'diversity_min'],
    'Divj_max': ['Divj_max', 'divj_max', 'diversity_max'],
    'GAP': ['GAP', 'gap', 'Gap'],
    'RDP': ['RDP', 'rdp', 'Rdp']
}

def _normalizar_columnas(df):
    """
    Normaliza los nombres de columnas del DataFrame según COLUMN_MAPPINGS.
    Retorna el DataFrame normalizado y un diccionario con las columnas encontradas.
    """
    # Limpiar espacios en blanco
    df.columns = df.columns.str.strip()
    
    # Crear mapeo de columnas encontradas
    columnas_encontradas = {}
    renombrar = {}
    
    for col_estandar, variaciones in COLUMN_MAPPINGS.items():
        for col_actual in df.columns:
            if col_actual in variaciones:
                renombrar[col_actual] = col_estandar
                columnas_encontradas[col_estandar] = True
                break
    
    # Renombrar columnas
    if renombrar:
        df = df.rename(columns=renombrar)
    
    return df, columnas_encontradas

def _validar_csv(data, nombre_archivo):
    """
    Valida que el CSV tenga las columnas mínimas necesarias.
    Retorna (data_normalizado, es_valido, mensaje_error)
    """
    if data.empty:
        return None, False, "CSV vacío"
    
    # Normalizar columnas
    data, cols_encontradas = _normalizar_columnas(data)
    
    # Verificar columnas obligatorias
    columnas_obligatorias = ['best_fitness', 'time']
    columnas_faltantes = [col for col in columnas_obligatorias if col not in cols_encontradas]
    
    if columnas_faltantes:
        cols_disponibles = ', '.join(data.columns.tolist())
        return None, False, f"Faltan columnas: {columnas_faltantes}. Disponibles: [{cols_disponibles}]"
    
    # VALIDAR QUE LAS COLUMNAS SEAN NUMÉRICAS
    for col in ['best_fitness', 'time']:
        try:
            # Intentar convertir a numérico
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Si después de la conversión todo es NaN, el CSV está corrupto
            if data[col].isna().all():
                return None, False, f"Columna '{col}' contiene solo valores no numéricos o está corrupta"
            
            # Eliminar filas con valores NaN en columnas críticas
            if data[col].isna().any():
                filas_antes = len(data)
                data = data.dropna(subset=[col])
                filas_despues = len(data)
                if filas_despues == 0:
                    return None, False, f"Todas las filas tienen valores inválidos en '{col}'"
                print(f"      [WARN] '{nombre_archivo}': {filas_antes - filas_despues} filas con NaN en '{col}' eliminadas")
        
        except Exception as e:
            return None, False, f"Error al validar columna '{col}': {str(e)}"
    
    # Validar columnas opcionales de diversidad
    for col in ['ENT', 'Divj_mean', 'Divj_min', 'Divj_max', 'GAP', 'RDP']:
        if col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except:
                # Si falla, simplemente eliminar la columna
                data = data.drop(columns=[col])
    
    return data, True, None

# ========= Helpers =========

def _actualizar_datos(mhs_instances, mh, archivo_fitness, data):
    """Versión vectorizada con validación"""
    inst = mhs_instances[mh]
    
    # VALIDAR que las columnas sean numéricas antes de calcular
    if not pd.api.types.is_numeric_dtype(data['best_fitness']):
        print(f"      [ERROR] Columna 'best_fitness' no es numérica para {mh}")
        return
    
    if not pd.api.types.is_numeric_dtype(data['time']):
        print(f"      [ERROR] Columna 'time' no es numérica para {mh}")
        return
    
    # Operaciones vectorizadas (más rápidas)
    inst.fitness.append(data['best_fitness'].min())
    inst.time.append(data['time'].sum().round(3))
    
    # XPL y XPT son opcionales
    if 'XPL' in data.columns and pd.api.types.is_numeric_dtype(data['XPL']):
        inst.xpl.append(data['XPL'].mean().round(2))
    if 'XPT' in data.columns and pd.api.types.is_numeric_dtype(data['XPT']):
        inst.xpt.append(data['XPT'].mean().round(2))
    
    archivo_fitness.write(f'{mh}, {inst.fitness[-1]}\n')

    # Diversidad usando operaciones de pandas (más eficientes)
    for col, target in [
        ('ENT', 'ent'), ('Divj_mean', 'divj_mean'),
        ('Divj_min', 'divj_min'), ('Divj_max', 'divj_max'),
        ('GAP', 'gap'), ('RDP', 'rdp'),
    ]:
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            # Verificar que no todo sea NaN
            if not data[col].isna().all():
                getattr(inst, target).append(data[col].mean())


def _parse_result_filename(nombre_archivo: str):
    """Extrae (mh, instancia, corrida) desde el nombre del CSV.

    El formato esperado en este repo suele ser:
        - En BD.iteraciones.nombre: <mh>_<instancia>
        - En archivos sueltos:      <mh>_<instancia>_<id>.csv

    Importante: <mh> puede contener guiones bajos (por ejemplo: PSO_mealpy).
    Por eso no podemos usar split('_')[:2].
    """

    base = os.path.basename(str(nombre_archivo)).strip()
    if base.lower().endswith(".csv"):
        base = base[:-4]

    parts = base.split("_")

    # Caso 1: nombre almacenado en BD (sin corrida): <mh>_<instancia>
    if len(parts) >= 2 and not parts[-1].isdigit():
        mh = "_".join(parts[:-1])
        instancia = parts[-1]
        return mh, instancia, None

    # Caso 2: archivo suelto con corrida al final: <mh>_<instancia>_<id>
    if len(parts) >= 3 and parts[-1].isdigit():
        mh = "_".join(parts[:-2])
        instancia = parts[-2]
        corrida = parts[-1]
        return mh, instancia, corrida

    return None, None, None

def _graficar_por_corrida(iteraciones, fitness, xpl, xpt, tiempo, mh, problem_id, corrida, subfolder, binarizacion=None):
    out = os.path.join(DIR_GRAFICOS, subfolder) if binarizacion is None \
          else os.path.join(DIR_GRAFICOS, subfolder, str(binarizacion))
    os.makedirs(out, exist_ok=True)

    # Convergencia con modo logarítmico según configuración
    fpath = os.path.join(out, f'Convergence_{mh}_{subfolder}_{problem_id}_{corrida}' + (f'_{binarizacion}' if binarizacion else '') + '.pdf')
    _, ax = plt.subplots()
    
    # Determinar modo de visualización
    fitness_array = np.array(fitness)
    y_data = fitness_array
    ylabel = "Fitness"
    
    if MODO_LOGARITMICO == 'log_transform':
        # ln(fitness) como en TJO original
        # Evitar log(0) o log(negativos)
        y_data = np.log(np.maximum(fitness_array, 1e-10))
        ylabel = "Ln Objective function"
    elif MODO_LOGARITMICO == 'log_scale':
        # Escala logarítmica en el eje (valores originales)
        ax.set_yscale('log')
        ylabel = "Fitness (log scale)"
    elif MODO_LOGARITMICO == 'auto':
        # Solo usar escala log si valores muy grandes
        fitness_range = np.max(fitness_array) - np.min(fitness_array)
        fitness_max = np.max(fitness_array)
        if fitness_max > 0 and (fitness_range / fitness_max > 1e-3 and fitness_max > 1e6):
            ax.set_yscale('log')
            ylabel = "Fitness (log scale)"
    # else: 'none' - lineal normal
    
    ax.plot(iteraciones, y_data)
    ax.set_ylabel(ylabel)
    ax.set_title(f'Convergence {mh}\n{problem_id} - Run {corrida}' + (f' - ({binarizacion})' if binarizacion else ''))
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(fpath); plt.close()

    # XPL vs XPT (sin cambios)
    fpath = os.path.join(out, f'Percentage_{mh}_{subfolder}_{problem_id}_{corrida}' + (f'_{binarizacion}' if binarizacion else '') + '.pdf')
    _, ax = plt.subplots()
    ax.plot(iteraciones, xpl, color="r", label=rf"$\overline{{XPL}}$: {np.round(np.mean(xpl), 2)}%")
    ax.plot(iteraciones, xpt, color="b", label=rf"$\overline{{XPT}}$: {np.round(np.mean(xpt), 2)}%")
    ax.set_title(f'XPL% - XPT% {mh}\n{problem_id} - Run {corrida}' + (f' - ({binarizacion})' if binarizacion else ''))
    ax.set_ylabel("Percentage"); ax.set_xlabel("Iteration")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(fpath); plt.close()

    # Tiempo por iteración (sin cambios)
    fpath = os.path.join(out, f'Time_{mh}_{subfolder}_{problem_id}_{corrida}' + (f'_{binarizacion}' if binarizacion else '') + '.pdf')
    _, ax = plt.subplots()
    ax.plot(iteraciones, tiempo, label='Time per Iteration')
    ax.set_title(f'Time per Iteration {mh}\n{problem_id} - Run {corrida}' + (f' - ({binarizacion})' if binarizacion else ''))
    ax.set_ylabel("Time (s)"); ax.set_xlabel("Iteration")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(fpath); plt.close()

def _graficar_box_violin(instancia_id, subfolder, binarizacion=None, title_prefix="", num_experimentos=1):
    """
    Genera boxplot y violinplot solo si hay múltiples experimentos.
    """
    # Si solo hay 1 experimento, no generar gráficos estadísticos
    if num_experimentos <= 1:
        return
    
    suf = f'_{binarizacion}' if binarizacion else ''
    datos_path = os.path.join(DIR_FITNESS, f'{subfolder}/fitness_{subfolder}_{instancia_id}{suf}.csv')
    try:
        df = pd.read_csv(datos_path)
        df.columns = df.columns.str.strip()
        if 'FITNESS' not in df.columns or 'MH' not in df.columns:
            print(f"[ERROR] Columnas necesarias no encontradas en {datos_path}"); return
    except FileNotFoundError:
        print(f"[ERROR] Archivo no encontrado: {datos_path}"); return
    except Exception as e:
        print(f"[ERROR] Error al leer {datos_path}: {e}"); return

    # Detectar si conviene escala log (para evitar que un outlier aplaste al resto)
    fitness_vals = pd.to_numeric(df['FITNESS'], errors='coerce').dropna().to_numpy(dtype=float)
    fitness_vals = fitness_vals[np.isfinite(fitness_vals)]
    fitness_min = float(np.min(fitness_vals)) if fitness_vals.size else np.nan
    fitness_max = float(np.max(fitness_vals)) if fitness_vals.size else np.nan
    use_log_y = False
    if np.isfinite(fitness_min) and np.isfinite(fitness_max) and fitness_min > 0:
        if MODO_LOGARITMICO == 'log_scale':
            use_log_y = True
        elif MODO_LOGARITMICO == 'auto' and fitness_max > 1e6:
            use_log_y = True

    # Boxplot
    out = os.path.join(DIR_BOXPLOT, subfolder); os.makedirs(out, exist_ok=True)
    fpath = os.path.join(out, f'boxplot_fitness_{subfolder}_{instancia_id}{suf}.pdf')
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='MH', y='FITNESS', data=df, hue='MH', palette='Set2', legend=False)
    if use_log_y:
        plt.yscale('log')
        ylabel = 'Fitness (log scale)'
    else:
        ylabel = 'Fitness'
    plt.title(f'Boxplot Fitness\n{title_prefix}{instancia_id}' + (f' - {binarizacion}' if binarizacion else ''))
    plt.xlabel('Metaheurística'); plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(fpath); plt.close()

    # Violin
    out = os.path.join(DIR_VIOLIN, subfolder); os.makedirs(out, exist_ok=True)
    fpath = os.path.join(out, f'violinplot_fitness_{subfolder}_{instancia_id}{suf}.pdf')
    plt.figure(figsize=(10, 6))
    # Nota: para que el violín se vea "normal", lo dejamos en escala lineal.
    # En escala log (con outliers enormes) suele verse contraintuitivo.
    sns.violinplot(x='MH', y='FITNESS', data=df, hue='MH', palette='Set3', legend=False)
    ylabel = 'Fitness'
    plt.title(f'Violinplot Fitness\n{title_prefix}{instancia_id}' + (f' - {binarizacion}' if binarizacion else ''))
    plt.xlabel('Metaheurística'); plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(fpath); plt.close()

def _graficar_best(instancia_id, mhs_instances, subfolder, title_prefix="", binarizacion=None):
    # detectar mejores
    best_f, best_t, best_f_mh, best_t_mh = np.inf, np.inf, "", ""
    for name in MHS_LIST:
        mh = mhs_instances[name]
        min_f = min(mh.bestFitness) if len(mh.bestFitness)>0 else np.inf
        min_t = min(mh.bestTime)    if len(mh.bestTime)>0    else np.inf
        if min_f < best_f: best_f, best_f_mh = min_f, name
        if min_t < best_t: best_t, best_t_mh = min_t, name

    out = os.path.join(DIR_BEST, subfolder); os.makedirs(out, exist_ok=True)
    suf = f'_{binarizacion}' if binarizacion else ''
    title_id = f'{title_prefix}{instancia_id}' + (f' - {binarizacion}' if binarizacion else '')

    # Fitness con modo logarítmico según configuración
    fig, ax = plt.subplots(figsize=(10, 6))
    all_fitness_values = []
    ylabel = "Fitness"
    
    # Determinar modo de visualización
    if MODO_LOGARITMICO == 'log_transform':
        # ln(fitness) como en TJO original
        for name in MHS_LIST:
            mh = mhs_instances[name]
            if len(mh.bestFitness):
                fitness_vals = np.array(mh.bestFitness)
                all_fitness_values.extend(fitness_vals)
                # Aplicar ln, evitando log(0)
                y_data = np.log(np.maximum(fitness_vals, 1e-10))
                ax.plot(range(len(y_data)), y_data, label=name, linewidth=2)
        ylabel = "Ln Objective function"
    else:
        # Graficar valores originales
        for name in MHS_LIST:
            mh = mhs_instances[name]
            if len(mh.bestFitness):
                fitness_vals = np.array(mh.bestFitness)
                all_fitness_values.extend(fitness_vals)
                ax.plot(range(len(fitness_vals)), fitness_vals, label=name, linewidth=2)
        
        # Determinar escala del eje
        if len(all_fitness_values) > 0:
            all_vals = np.asarray(all_fitness_values, dtype=float)
            all_vals = all_vals[np.isfinite(all_vals)]
            if all_vals.size > 0:
                fitness_min = float(np.min(all_vals))
                fitness_max = float(np.max(all_vals))
            else:
                fitness_min, fitness_max = np.nan, np.nan

            if MODO_LOGARITMICO == 'log_scale':
                if np.isfinite(fitness_min) and fitness_min > 0:
                    ax.set_yscale('log')
                    ylabel = "Fitness (log scale)"
            elif MODO_LOGARITMICO == 'auto':
                # Auto: si la instancia tiene fitness en varias órdenes de magnitud,
                # usar escala log para evitar que todo quede aplastado cerca de 0.
                if np.isfinite(fitness_min) and fitness_min > 0 and np.isfinite(fitness_max):
                    if fitness_max > 1e6:
                        ax.set_yscale('log')
                        ylabel = "Fitness (log scale)"
    
    ax.set_ylabel(ylabel, fontsize=12)
    
    ax.set_title(f'Best Fitness per MH\n{title_id}\nBest: {best_f_mh} ({best_f:.4e})', fontsize=13)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, f'fitness_{subfolder}_{instancia_id}{suf}.pdf'))
    plt.close()

    # Time (sin cambios en escala)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in MHS_LIST:
        mh = mhs_instances[name]
        if len(mh.bestTime):
            ax.plot(range(len(mh.bestTime)), mh.bestTime, label=name, linewidth=2)
    
    ax.set_title(f'Best Time per MH\n{title_id}\nBest: {best_t_mh} ({best_t:.2f} s)', fontsize=13)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out, f'time_{subfolder}_{instancia_id}{suf}.pdf'))
    plt.close()

    # XPL vs XPT combinado (sin cambios)
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple",
              "tab:cyan", "tab:pink", "tab:brown", "tab:olive", "tab:gray"]
    cmap = {name: colors[i % len(colors)] for i, name in enumerate(MHS_LIST)}

    plt.figure(figsize=(10, 6))
    any_series, max_len = False, 0
    for name in MHS_LIST:
        mh = mhs_instances[name]
        if mh.xpl_iter is None or mh.xpt_iter is None: continue
        any_series = True
        x = np.arange(len(mh.xpl_iter))
        c = cmap[name]
        plt.plot(x, mh.xpt_iter, linestyle='-',  linewidth=2,
                 label=f'{name} XPT% (avg {np.round(np.mean(mh.xpt_iter), 2)}%)', color=c)
        plt.plot(x, mh.xpl_iter, linestyle='--', linewidth=2,
                 label=f'{name} XPL% (avg {np.round(np.mean(mh.xpl_iter), 2)}%)', color=c)
        max_len = max(max_len, len(x))
    
    if any_series:
        plt.title(f'Exploration (XPL) vs Exploitation (XPT) per MH\n{title_id}', fontsize=13)
        plt.ylabel("Percentage (%)", fontsize=12)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylim(0, 100)
        if max_len <= 1: plt.xlim(-0.5, 0.5)
        plt.legend(loc='upper right', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f'xpl_xpt_{subfolder}_{instancia_id}{suf}.pdf'))
        plt.close()
    else:
        plt.close()

def _procesar_archivos(problem_cfg, instancia_id, blob, archivo_fitness, mhs_instances, bin_actual=None, batch_size=BATCH_SIZE):
    """
    Procesa archivos con validación robusta de CSVs.
    """
    import gc
    
    # Filtrar archivos relevantes
    archivos_filtrados = []
    for item in blob:
        if problem_cfg["uses_bin"]:
            nombre_archivo, contenido, binarizacion = item
            if str(binarizacion).strip() != str(bin_actual).strip():
                continue
        else:
            nombre_archivo, contenido = item
            binarizacion = None
        
        mh, _, _ = _parse_result_filename(nombre_archivo)
        if mh and mh in MHS_LIST:
            archivos_filtrados.append((item, binarizacion))
    
    total = len(archivos_filtrados)
    if total == 0:
        archivo_fitness.close()
        return
    
    print(f"      Total archivos: {total} | Batch: {batch_size}")
    
    corrida = 1
    archivos_procesados = 0
    archivos_con_errores = 0
    
    # Procesar en lotes
    for inicio in range(0, total, batch_size):
        fin = min(inicio + batch_size, total)
        lote = archivos_filtrados[inicio:fin]
        
        # Mostrar progreso con memoria
        if total > batch_size:
            num_lote = (inicio // batch_size) + 1
            total_lotes = (total - 1) // batch_size + 1
            mem = psutil.virtual_memory()
            print(f"      Lote {num_lote}/{total_lotes} | RAM: {mem.percent:.1f}% | {mem.available / 1024**3:.1f} GB libre")
        
        # Procesar archivos del lote
        for item, binarizacion in lote:
            if problem_cfg["uses_bin"]:
                nombre_archivo, contenido, _ = item
            else:
                nombre_archivo, contenido = item
            
            mh, _, _ = _parse_result_filename(nombre_archivo)
            if not mh:
                archivos_con_errores += 1
                continue
            
            # Archivo temporal único
            unique_id = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
            temp = os.path.join(DIR_TRANSITORIO, f'{nombre_archivo}_{unique_id}.csv')
            
            writeTofile(contenido, temp)
            try:
                data = pd.read_csv(temp)
                
                # VALIDAR CSV
                data_normalizado, es_valido, error_msg = _validar_csv(data, nombre_archivo)
                
                if not es_valido:
                    print(f"[WARN] Archivo inválido '{nombre_archivo}': {error_msg}")
                    archivos_con_errores += 1
                    if os.path.exists(temp):
                        try: os.remove(temp)
                        except: pass
                    continue
                
                data = data_normalizado
                
            except Exception as e:
                print(f"[ERROR] '{nombre_archivo}': {e}")
                archivos_con_errores += 1
                if os.path.exists(temp):
                    try: os.remove(temp)
                    except: pass
                continue
            
            # Actualizar datos
            _actualizar_datos(mhs_instances, mh, archivo_fitness, data)
            
            # Mejor corrida
            new_final = float(data['best_fitness'].iloc[-1])
            prev_final = np.inf
            prev_series = getattr(mhs_instances[mh], 'bestFitness', None)
            if prev_series is not None and len(prev_series) > 0:
                try:    prev_final = float(prev_series.iloc[-1])
                except: prev_final = float(prev_series[-1])
            
            if new_final < prev_final:
                mhs_instances[mh].bestFitness = data['best_fitness']
                mhs_instances[mh].bestTime    = data['time']
                if 'XPL' in data.columns:
                    mhs_instances[mh].xpl_iter = data['XPL']
                if 'XPT' in data.columns:
                    mhs_instances[mh].xpt_iter = data['XPT']
            
            # Gráficos por corrida (opcional)
            if GRAFICOS_POR_CORRIDA and 'iter' in data.columns and 'XPL' in data.columns and 'XPT' in data.columns:
                pid = problem_cfg["title_prefix"] + str(instancia_id) if problem_cfg["title_prefix"] else str(instancia_id)
                _graficar_por_corrida(
                    data['iter'], data['best_fitness'],
                    data['XPL'], data['XPT'], data['time'],
                    mh, pid, corrida, problem_cfg["sub"], binarizacion
                )
            
            # Limpiar temporal
            if os.path.exists(temp):
                try: os.remove(temp)
                except: pass
            
            corrida += 1
            archivos_procesados += 1
        
        # Garbage collection cada GC_INTERVAL archivos
        if archivos_procesados % GC_INTERVAL == 0:
            gc.collect()
            
            # Si RAM muy alta, limpieza agresiva
            mem_check = psutil.virtual_memory()
            if mem_check.percent > RAM_WARNING:
                print(f"      [!] RAM alta ({mem_check.percent:.1f}%) - Limpieza agresiva")
                gc.collect()
    
    if archivos_con_errores > 0:
        print(f"      [WARN] {archivos_con_errores} archivos con errores fueron omitidos")
    
    archivo_fitness.close()

def _escribir_div_gap(problem_cfg, instancia_id, mhs_instances, binarizacion, fh_div, fh_gap):
    for name in MHS_LIST:
        mh = mhs_instances[name]
        if len(mh.ent) or len(mh.divj_mean) or len(mh.divj_min) or len(mh.divj_max):
            ent_avg       = np.round(np.mean(mh.ent), 6)       if len(mh.ent)       else np.nan
            divj_mean_avg = np.round(np.mean(mh.divj_mean), 6) if len(mh.divj_mean) else np.nan
            divj_min_avg  = np.round(np.mean(mh.divj_min), 6)  if len(mh.divj_min)  else np.nan
            divj_max_avg  = np.round(np.mean(mh.divj_max), 6)  if len(mh.divj_max)  else np.nan
            fh_div.write(f"{name}, {ent_avg}, {divj_mean_avg}, {divj_min_avg}, {divj_max_avg}\n")

        if len(mh.gap) or len(mh.rdp):
            gap_avg = np.round(np.mean(mh.gap), 6) if len(mh.gap) else np.nan
            rdp_avg = np.round(np.mean(mh.rdp), 6) if len(mh.rdp) else np.nan
            fh_gap.write(f"{name}, {gap_avg}, {rdp_avg}\n")

# ========= Orquestador =========
def _procesar_instancia_binarizacion(inst, binarizacion, cfg, uses_bin, title_prefix, num_experimentos):
    """Procesa una combinación instancia-binarización."""
    instancia_id = inst[1]
    pid = f"{title_prefix}{instancia_id}" if title_prefix else f"{instancia_id}"
    
    if uses_bin:
        print(f"    -- Procesando {pid} con binarización: {binarizacion}")
    else:
        print(f"    -- Procesando {pid}")
    
    incluir_bin = cfg["obtenerArchivos_kwargs"].get("incluir_binarizacion", True)
    blob = _obtener_blob_cached(instancia_id, incluir_bin)
    blob = list(blob) if blob else []
    
    if not blob:
        return None
    
    sub = cfg["sub"]
    mhs_map = {name: InstancesMhs() for name in MHS_LIST}
    
    out_res = os.path.join(DIR_RESUMEN, sub)
    out_fit = os.path.join(DIR_FITNESS, sub)
    os.makedirs(out_res, exist_ok=True)
    os.makedirs(out_fit, exist_ok=True)
    
    suf = f"_{binarizacion}" if uses_bin else ""
    
    fh_fit  = open(os.path.join(out_fit, f'fitness_{sub}_{instancia_id}{suf}.csv'), 'w')
    fh_rf   = open(os.path.join(out_res, f'resumen_fitness_{sub}_{instancia_id}{suf}.csv'), 'w')
    fh_rt   = open(os.path.join(out_res, f'resumen_times_{sub}_{instancia_id}{suf}.csv'), 'w')
    fh_rp   = open(os.path.join(out_res, f'resumen_percentage_{sub}_{instancia_id}{suf}.csv'), 'w')
    fh_div  = open(os.path.join(out_res, f'resumen_diversity_{sub}_{instancia_id}{suf}.csv'), 'w')
    fh_gap  = open(os.path.join(out_res, f'resumen_gap_{sub}_{instancia_id}{suf}.csv'), 'w')
    
    fh_fit.write("MH, FITNESS\n")
    fh_rf.write("MH, min best, avg. best, std. best\n")
    fh_rt.write("MH, min time (s), avg. time (s), std time (s)\n")
    fh_rp.write("MH, avg. XPL%, avg. XPT%\n")
    fh_div.write("MH, avg. ENT, avg. Divj_mean, avg. Divj_min, avg. Divj_max\n")
    fh_gap.write("MH, avg. GAP, avg. RDP\n")
    
    _procesar_archivos(cfg, instancia_id, blob, fh_fit, mhs_map, bin_actual=binarizacion)

    # Avisar si alguna MH no aportó datos para esta instancia
    faltantes = [name for name in MHS_LIST if len(mhs_map[name].fitness) == 0 and len(mhs_map[name].time) == 0]
    if faltantes:
        print(f"      [WARN] Sin datos ({len(faltantes)}): {', '.join(faltantes)}")
    
    escribir_resumenes(mhs_map, fh_rf, fh_rt, fh_rp, MHS_LIST)
    _graficar_best(instancia_id, mhs_map, subfolder=sub, title_prefix=title_prefix, binarizacion=binarizacion)
    _graficar_box_violin(instancia_id, subfolder=sub, binarizacion=binarizacion, 
                        title_prefix=title_prefix, num_experimentos=num_experimentos)  # ← AGREGAR PARÁMETRO
    _escribir_div_gap(cfg, instancia_id, mhs_map, binarizacion, fh_div, fh_gap)
    
    fh_fit.close(); fh_rf.close(); fh_rt.close(); fh_rp.close(); fh_div.close(); fh_gap.close()
    
    return f"{pid}{suf} completado"

def analizar_problema(problem_name: str, n_jobs=-1):
    """Analiza un problema con paralelización y caché."""
    cfg = PROBLEMS[problem_name]
    sub, inst_key, uses_bin, title_prefix = cfg["sub"], cfg["inst_key"], cfg["uses_bin"], cfg["title_prefix"]
    
    # Verificar si el problema está habilitado en la configuración
    problem_enabled_map = {
        "BEN": EXPERIMENTS.get("ben", False),
        "SCP": EXPERIMENTS.get("scp", False),
        "USCP": EXPERIMENTS.get("uscp", False),
    }
    
    if not problem_enabled_map.get(problem_name, False):
        print(f"[INFO] Análisis {sub} deshabilitado en la configuración (experiments_config.json)")
        return
    
    os.makedirs(DIR_TRANSITORIO, exist_ok=True)
    
    lista_inst = ', '.join([f'"{func}"' for func in EXPERIMENTS["instancias"][inst_key]])
    instancias = list(_obtener_instancias_cached(lista_inst))
    
    if not instancias:
        print(f"[INFO] No hay instancias para procesar en {sub}")
        return
    
    # Obtener número de experimentos desde la configuración
    num_experimentos = EXPERIMENTS["experimentos"][inst_key].get("num_experimentos", 1)
    
    print(f"[INFO] Ejecutando análisis {sub} con paralelización...")
    print(f"[INFO] Número de experimentos: {num_experimentos}")
    print("[INFO] Pre-cargando datos en caché...")
    
    incluir_bin = cfg["obtenerArchivos_kwargs"].get("incluir_binarizacion", True)
    for inst in instancias:
        _obtener_blob_cached(inst[1], incluir_bin)
    
    print(f"[INFO] Caché pre-cargado: {len(_CACHE_BLOB)} blobs")
    print("-"*50, "\nIniciando procesamiento de instancias...\n")
    
    bin_list = EXPERIMENTS["DS_actions"] if uses_bin else [None]
    tareas = [(inst, bin_val, cfg, uses_bin, title_prefix, num_experimentos)  # ← AGREGAR num_experimentos
              for inst in instancias 
              for bin_val in bin_list]
    
    resultados = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_procesar_instancia_binarizacion)(*tarea) 
        for tarea in tareas
    )
    
    completados = [r for r in resultados if r is not None]
    
    print(f"\n[INFO] Análisis {sub} completado con éxito.")
    print(f"[INFO] Procesadas {len(completados)} combinaciones instancia-binarización")
    _mostrar_estadisticas_cache()
    print("-"*50)

def main():
    """Ejecuta el análisis con paralelización y caché."""
    start_time = time.time()
    
    try:
        analizar_problema("BEN", n_jobs=-1)
        analizar_problema("SCP", n_jobs=-1)
        analizar_problema("USCP", n_jobs=-1)
    finally:
        end_time = time.time()
        print(f"\n{'='*50}")
        print(f"Tiempo total de ejecución: {end_time - start_time:.2f} segundos")
        print(f"{'='*50}")
        limpiar_cache_bd()

if __name__ == "__main__":
    main()
