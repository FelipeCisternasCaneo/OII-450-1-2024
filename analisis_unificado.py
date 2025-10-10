import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Util.util import cargar_configuracion_exp, writeTofile
from BD.sqlite import BD
from Util.log import escribir_resumenes

# ========= Config =========
CONFIG_FILE = './util/json/dir.json'
EXPERIMENTS_FILE = './util/json/experiments_config.json'
CONFIG, EXPERIMENTS = cargar_configuracion_exp(CONFIG_FILE, EXPERIMENTS_FILE)

DIRS = CONFIG["dirs"]
DIR_FITNESS      = DIRS["fitness"]
DIR_RESUMEN      = DIRS["resumen"]
DIR_TRANSITORIO  = DIRS["transitorio"]
DIR_GRAFICOS     = DIRS["graficos"]
DIR_BEST         = DIRS["best"]
DIR_BOXPLOT      = DIRS["boxplot"]
DIR_VIOLIN       = DIRS["violinplot"]

GRAFICOS_POR_CORRIDA = False   # pon False si no quieres los gráficos por corrida

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

# ========= Helpers =========

def _actualizar_datos(mhs_instances, mh, archivo_fitness, data):
    inst = mhs_instances[mh]
    inst.fitness.append(np.min(data['best_fitness']))
    inst.time.append(np.round(np.sum(data['time']), 3))
    inst.xpl.append(np.round(np.mean(data['XPL']), 2))
    inst.xpt.append(np.round(np.mean(data['XPT']), 2))
    archivo_fitness.write(f'{mh}, {np.min(data["best_fitness"])}\n')

    # diversidad / gaps (promedio por corrida)
    for col, target in [
        ('ENT', 'ent'),
        ('Divj_mean', 'divj_mean'),
        ('Divj_min', 'divj_min'),
        ('Divj_max', 'divj_max'),
        ('GAP', 'gap'),
        ('RDP', 'rdp'),
    ]:
        if col in data.columns:
            getattr(inst, target).append(float(np.nanmean(data[col])))

def _graficar_por_corrida(iteraciones, fitness, xpl, xpt, tiempo, mh, problem_id, corrida, subfolder, binarizacion=None):
    out = os.path.join(DIR_GRAFICOS, subfolder) if binarizacion is None \
          else os.path.join(DIR_GRAFICOS, subfolder, str(binarizacion))
    os.makedirs(out, exist_ok=True)

    # Convergencia
    fpath = os.path.join(out, f'Convergence_{mh}_{subfolder}_{problem_id}_{corrida}' + (f'_{binarizacion}' if binarizacion else '') + '.pdf')
    _, ax = plt.subplots()
    ax.plot(iteraciones, fitness)
    ax.set_title(f'Convergence {mh}\n{problem_id} - Run {corrida}' + (f' - ({binarizacion})' if binarizacion else ''))
    ax.set_ylabel("Fitness"); ax.set_xlabel("Iteration")
    plt.tight_layout(); plt.savefig(fpath); plt.close()

    # XPL vs XPT
    fpath = os.path.join(out, f'Percentage_{mh}_{subfolder}_{problem_id}_{corrida}' + (f'_{binarizacion}' if binarizacion else '') + '.pdf')
    _, ax = plt.subplots()
    ax.plot(iteraciones, xpl, color="r", label=rf"$\overline{{XPL}}$: {np.round(np.mean(xpl), 2)}%")
    ax.plot(iteraciones, xpt, color="b", label=rf"$\overline{{XPT}}$: {np.round(np.mean(xpt), 2)}%")
    ax.set_title(f'XPL% - XPT% {mh}\n{problem_id} - Run {corrida}' + (f' - ({binarizacion})' if binarizacion else ''))
    ax.set_ylabel("Percentage"); ax.set_xlabel("Iteration")
    ax.legend(loc='upper right')
    plt.tight_layout(); plt.savefig(fpath); plt.close()

    # Tiempo por iteración
    fpath = os.path.join(out, f'Time_{mh}_{subfolder}_{problem_id}_{corrida}' + (f'_{binarizacion}' if binarizacion else '') + '.pdf')
    _, ax = plt.subplots()
    ax.plot(iteraciones, tiempo, label='Time per Iteration')
    ax.set_title(f'Time per Iteration {mh}\n{problem_id} - Run {corrida}' + (f' - ({binarizacion})' if binarizacion else ''))
    ax.set_ylabel("Time (s)"); ax.set_xlabel("Iteration")
    ax.legend(loc='upper right')
    plt.tight_layout(); plt.savefig(fpath); plt.close()

def _graficar_box_violin(instancia_id, subfolder, binarizacion=None, title_prefix=""):
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

    # Boxplot
    out = os.path.join(DIR_BOXPLOT, subfolder); os.makedirs(out, exist_ok=True)
    fpath = os.path.join(out, f'boxplot_fitness_{subfolder}_{instancia_id}{suf}.pdf')
    sns.boxplot(x='MH', y='FITNESS', data=df, hue='MH', palette='Set2', legend=False)
    plt.title(f'Boxplot Fitness\n{title_prefix}{instancia_id}' + (f' - {binarizacion}' if binarizacion else ''))
    plt.xlabel('Metaheurística'); plt.ylabel('Fitness')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout(); plt.savefig(fpath); plt.close()

    # Violin
    out = os.path.join(DIR_VIOLIN, subfolder); os.makedirs(out, exist_ok=True)
    fpath = os.path.join(out, f'violinplot_fitness_{subfolder}_{instancia_id}{suf}.pdf')
    sns.violinplot(x='MH', y='FITNESS', data=df, hue='MH', palette='Set3', legend=False)
    plt.title(f'Violinplot Fitness\n{title_prefix}{instancia_id}' + (f' - {binarizacion}' if binarizacion else ''))
    plt.xlabel('Metaheurística'); plt.ylabel('Fitness')
    plt.xticks(rotation=45, ha='right')
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

    # Fitness
    for name in MHS_LIST:
        mh = mhs_instances[name]
        if len(mh.bestFitness): plt.plot(range(len(mh.bestFitness)), mh.bestFitness, label=name)
    plt.title(f'Best Fitness per MH \n {title_id}\nBest: {best_f_mh} ({best_f})')
    plt.ylabel("Fitness"); plt.xlabel("Iteration"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out, f'fitness_{subfolder}_{instancia_id}{suf}.pdf')); plt.close()

    # Time
    for name in MHS_LIST:
        mh = mhs_instances[name]
        if len(mh.bestTime): plt.plot(range(len(mh.bestTime)), mh.bestTime, label=name)
    plt.title(f'Best Time per MH \n {title_id}\nBest: {best_t_mh} ({best_t:.2f} s)')
    plt.ylabel("Time (s)"); plt.xlabel("Iteration"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out, f'time_{subfolder}_{instancia_id}{suf}.pdf')); plt.close()

    # XPL vs XPT combinado
    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple",
              "tab:cyan", "tab:pink", "tab:brown", "tab:olive", "tab:gray"]
    cmap = {name: colors[i % len(colors)] for i, name in enumerate(MHS_LIST)}

    plt.figure(figsize=(8,5))
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
        plt.title(f'Exploration (XPL) vs Exploitation (XPT) per MH\n{title_id}')
        plt.ylabel("Percentage (%)"); plt.xlabel("Iteration"); plt.ylim(0,100)
        if max_len <= 1: plt.xlim(-0.5, 0.5)
        plt.legend(loc='upper right', fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(out, f'xpl_xpt_{subfolder}_{instancia_id}{suf}.pdf')); plt.close()
    else:
        plt.close()

def _procesar_archivos(problem_cfg, instancia_id, blob, archivo_fitness, mhs_instances, bin_actual=None):
    corrida = 1
    for item in blob:
        if problem_cfg["uses_bin"]:
            nombre_archivo, contenido, binarizacion = item
            if str(binarizacion).strip() != str(bin_actual).strip():
                continue
        else:
            nombre_archivo, contenido = item
            binarizacion = None

        # mh del nombre
        try:
            mh, _ = nombre_archivo.split('_')[:2]
        except ValueError:
            print(f"[ADVERTENCIA] Archivo '{nombre_archivo}' con nombre inválido. Se omite.")
            continue
        if mh not in MHS_LIST:
            continue

        # guardar temporal y leer
        temp = os.path.join(DIR_TRANSITORIO, f'{nombre_archivo}.csv')
        writeTofile(contenido, temp)
        try:
            data = pd.read_csv(temp)
        except Exception as e:
            print(f"[ERROR] Fallo al leer '{temp}': {e}")
            os.remove(temp); continue

        # acumular resúmenes
        _actualizar_datos(mhs_instances, mh, archivo_fitness, data)

        # corrida representativa = la que termina con mejor best_fitness
        new_final = float(data['best_fitness'].iloc[-1])
        prev_final = np.inf
        prev_series = getattr(mhs_instances[mh], 'bestFitness', None)
        if prev_series is not None and len(prev_series) > 0:
            try:    prev_final = float(prev_series.iloc[-1])  # pandas Series
            except: prev_final = float(prev_series[-1])       # list/ndarray

        if new_final < prev_final:
            mhs_instances[mh].bestFitness = data['best_fitness']
            mhs_instances[mh].bestTime    = data['time']
            mhs_instances[mh].xpl_iter    = data['XPL']
            mhs_instances[mh].xpt_iter    = data['XPT']

        # gráficos por corrida (opcional)
        if GRAFICOS_POR_CORRIDA:
            pid = problem_cfg["title_prefix"] + str(instancia_id) if problem_cfg["title_prefix"] else str(instancia_id)
            _graficar_por_corrida(
                iteraciones=data['iter'],
                fitness=data['best_fitness'],
                xpl=data['XPL'], xpt=data['XPT'], tiempo=data['time'],
                mh=mh, problem_id=pid, corrida=corrida,
                subfolder=problem_cfg["sub"], binarizacion=binarizacion
            )

        os.remove(temp)
        corrida += 1

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

# ========= Orquestador por problema =========
def analizar_problema(problem_name: str):
    cfg = PROBLEMS[problem_name]
    sub, inst_key, uses_bin, title_prefix = cfg["sub"], cfg["inst_key"], cfg["uses_bin"], cfg["title_prefix"]

    os.makedirs(DIR_TRANSITORIO, exist_ok=True)
    lista_inst = ', '.join([f'"{func}"' for func in EXPERIMENTS["instancias"][inst_key]])
    instancias = bd.obtenerInstancias(lista_inst)

    print(f"[INFO] Ejecutando análisis {sub}...")
    print("-"*50, "\nIniciando procesamiento de instancias...\n")

    for inst in instancias:
        instancia_id = inst[1]
        pid = f"{title_prefix}{instancia_id}" if title_prefix else f"{instancia_id}"
        print(f"[INFO] Procesando instancia: {pid}")

        blob = bd.obtenerArchivos(instancia_id, **cfg["obtenerArchivos_kwargs"])

        if not blob:
            print(f"[ADVERTENCIA] La instancia {pid} no tiene experimentos asociados. Saltando...\n")
            continue

        # lista de binarizaciones (solo para SCP/USCP)
        bin_list = EXPERIMENTS["DS_actions"] if uses_bin else [None]

        for binarizacion in bin_list:
            if uses_bin:
                print(f"    -- Binarización: {binarizacion}")

            mhs_map = {name: InstancesMhs() for name in MHS_LIST}

            # carpetas
            out_res = os.path.join(DIR_RESUMEN, sub)
            out_fit = os.path.join(DIR_FITNESS, sub)
            os.makedirs(out_res, exist_ok=True)
            os.makedirs(out_fit, exist_ok=True)

            suf = f"_{binarizacion}" if uses_bin else ""
            # archivos de salida
            fh_fit  = open(os.path.join(out_fit, f'fitness_{sub}_{instancia_id}{suf}.csv'), 'w')
            fh_rf   = open(os.path.join(out_res, f'resumen_fitness_{sub}_{instancia_id}{suf}.csv'), 'w')
            fh_rt   = open(os.path.join(out_res, f'resumen_times_{sub}_{instancia_id}{suf}.csv'), 'w')
            fh_rp   = open(os.path.join(out_res, f'resumen_percentage_{sub}_{instancia_id}{suf}.csv'), 'w')
            fh_div  = open(os.path.join(out_res, f'resumen_diversity_{sub}_{instancia_id}{suf}.csv'), 'w')
            fh_gap  = open(os.path.join(out_res, f'resumen_gap_{sub}_{instancia_id}{suf}.csv'), 'w')

            # headers
            fh_fit.write("MH, FITNESS\n")
            fh_rf.write("MH, min best, avg. best, std. best\n")
            fh_rt.write("MH, min time (s), avg. time (s), std time (s)\n")
            fh_rp.write("MH, avg. XPL%, avg. XPT%\n")
            fh_div.write("MH, avg. ENT, avg. Divj_mean, avg. Divj_min, avg. Divj_max\n")
            fh_gap.write("MH, avg. GAP, avg. RDP\n")

            # procesar blob
            _procesar_archivos(cfg, instancia_id, blob, fh_fit, mhs_map, bin_actual=binarizacion)

            # resúmenes core
            escribir_resumenes(mhs_map, fh_rf, fh_rt, fh_rp, MHS_LIST)

            # gráficos comparativos
            _graficar_best(instancia_id, mhs_map, subfolder=sub, title_prefix=title_prefix, binarizacion=binarizacion)
            _graficar_box_violin(instancia_id, subfolder=sub, binarizacion=binarizacion, title_prefix=title_prefix)

            # diversidad & gaps
            _escribir_div_gap(cfg, instancia_id, mhs_map, binarizacion, fh_div, fh_gap)

            # cerrar
            fh_fit.close(); fh_rf.close(); fh_rt.close(); fh_rp.close(); fh_div.close(); fh_gap.close()

        print("")

    print(f"[INFO] Análisis {sub} completado con éxito.")
    print("-"*50)

# ========= Punto de entrada =========
def main():
    # elige qué problemas ejecutar
    analizar_problema("BEN")
    analizar_problema("SCP")
    analizar_problema("USCP")

if __name__ == "__main__":
    main()
