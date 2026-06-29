import os
import time
import uuid
import psutil
import gc
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from util.util import writeTofile
from util.log import escribir_resumenes
from solver.domain_managers.registry import get_all as get_all_domains

# Imports del paquete de análisis
from analisis_pkg.config import (
    BATCH_SIZE,
    GC_INTERVAL,
    RAM_WARNING,
    DIR_TRANSITORIO,
    DIR_RESUMEN,
    DIR_FITNESS,
    MHS_LIST,
    GRAFICOS_POR_CORRIDA,
    MODO_TERMINACION,
    PROBLEMS,
    EXPERIMENTS,
    InstancesMhs,
)
from analisis_pkg.cache import (
    _obtener_blob_cached,
    _obtener_instancias_cached,
    _seleccionar_binarizaciones_disponibles,
    limpiar_cache_bd,
    _mostrar_estadisticas_cache,
    _CACHE_BLOB,
)
from analisis_pkg.parser import (
    _parse_result_filename,
    _validar_csv,
    _actualizar_datos,
)
from analisis_pkg.plotting import (
    _graficar_por_corrida,
    _graficar_box_violin,
    _graficar_best,
)


def _procesar_archivos(
    problem_cfg,
    instancia_id,
    blob,
    archivo_fitness,
    mhs_instances,
    bin_actual=None,
    batch_size=BATCH_SIZE,
):
    """
    Procesa archivos con validación robusta de CSVs.
    """
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
            print(
                f"      Lote {num_lote}/{total_lotes} | RAM: {mem.percent:.1f}% | {mem.available / 1024**3:.1f} GB libre"
            )

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
            temp = os.path.join(DIR_TRANSITORIO, f"{nombre_archivo}_{unique_id}.csv")

            writeTofile(contenido, temp)
            try:
                data = pd.read_csv(temp)

                # VALIDAR CSV
                data_normalizado, es_valido, error_msg = _validar_csv(
                    data, nombre_archivo
                )

                if not es_valido:
                    print(f"[WARN] Archivo inválido '{nombre_archivo}': {error_msg}")
                    archivos_con_errores += 1
                    if os.path.exists(temp):
                        try:
                            os.remove(temp)
                        except:
                            pass
                    continue

                data = data_normalizado

            except Exception as e:
                print(f"[ERROR] '{nombre_archivo}': {e}")
                archivos_con_errores += 1
                if os.path.exists(temp):
                    try:
                        os.remove(temp)
                    except:
                        pass
                continue

            # Actualizar datos
            _actualizar_datos(mhs_instances, mh, archivo_fitness, data)

            # Mejor corrida
            new_final = float(data["best_fitness"].iloc[-1])
            prev_final = np.inf
            prev_series = getattr(mhs_instances[mh], "bestFitness", None)
            if prev_series is not None and len(prev_series) > 0:
                try:
                    prev_final = float(prev_series.iloc[-1])
                except:
                    prev_final = float(prev_series[-1])

            if new_final < prev_final:
                mhs_instances[mh].bestFitness = data["best_fitness"]
                mhs_instances[mh].bestTime = data["time"]
                if "XPL" in data.columns:
                    mhs_instances[mh].xpl_iter = data["XPL"]
                if "XPT" in data.columns:
                    mhs_instances[mh].xpt_iter = data["XPT"]

                if "nfe" in data.columns and MODO_TERMINACION in ["fe", "both"]:
                    mhs_instances[mh].iter_vector = pd.to_numeric(
                        data["nfe"], errors="coerce"
                    )
                    mhs_instances[mh].iter_vector.name = "nfe"
                elif "iter" in data.columns:
                    mhs_instances[mh].iter_vector = pd.to_numeric(
                        data["iter"], errors="coerce"
                    )
                    mhs_instances[mh].iter_vector.name = "iter"

            # Gráficos por corrida (opcional)
            if (
                GRAFICOS_POR_CORRIDA
                and "iter" in data.columns
                and "XPL" in data.columns
                and "XPT" in data.columns
            ):
                pid = (
                    problem_cfg["title_prefix"] + str(instancia_id)
                    if problem_cfg["title_prefix"]
                    else str(instancia_id)
                )
                col_x_raw = (
                    data["nfe"]
                    if ("nfe" in data.columns and MODO_TERMINACION in ["fe", "both"])
                    else data["iter"]
                )
                col_x = pd.to_numeric(col_x_raw, errors="coerce")
                _graficar_por_corrida(
                    col_x,
                    data["best_fitness"],
                    data["XPL"],
                    data["XPT"],
                    data["time"],
                    mh,
                    pid,
                    corrida,
                    problem_cfg["sub"],
                    binarizacion,
                )

            # Limpiar temporal
            if os.path.exists(temp):
                try:
                    os.remove(temp)
                except:
                    pass

            corrida += 1
            archivos_procesados += 1

        # Garbage collection cada GC_INTERVAL archivos
        if archivos_procesados % GC_INTERVAL == 0:
            gc.collect()

            # Si RAM muy alta, limpieza agresiva
            mem_check = psutil.virtual_memory()
            if mem_check.percent > RAM_WARNING:
                print(
                    f"      [!] RAM alta ({mem_check.percent:.1f}%) - Limpieza agresiva"
                )
                gc.collect()

    if archivos_con_errores > 0:
        print(
            f"      [WARN] {archivos_con_errores} archivos con errores fueron omitidos"
        )

    archivo_fitness.close()


def _escribir_div_gap(
    problem_cfg, instancia_id, mhs_instances, binarizacion, fh_div, fh_gap
):
    for name in MHS_LIST:
        mh = mhs_instances[name]
        if len(mh.ent) or len(mh.divj_mean) or len(mh.divj_min) or len(mh.divj_max):
            ent_avg = np.round(np.mean(mh.ent), 6) if len(mh.ent) else np.nan
            divj_mean_avg = (
                np.round(np.mean(mh.divj_mean), 6) if len(mh.divj_mean) else np.nan
            )
            divj_min_avg = (
                np.round(np.mean(mh.divj_min), 6) if len(mh.divj_min) else np.nan
            )
            divj_max_avg = (
                np.round(np.mean(mh.divj_max), 6) if len(mh.divj_max) else np.nan
            )
            fh_div.write(
                f"{name}, {ent_avg}, {divj_mean_avg}, {divj_min_avg}, {divj_max_avg}\n"
            )

        if len(mh.gap) or len(mh.rdp):
            gap_avg = np.round(np.mean(mh.gap), 6) if len(mh.gap) else np.nan
            rdp_avg = np.round(np.mean(mh.rdp), 6) if len(mh.rdp) else np.nan
            fh_gap.write(f"{name}, {gap_avg}, {rdp_avg}\n")


def _procesar_instancia_binarizacion(
    inst, binarizacion, cfg, uses_bin, title_prefix, num_experimentos
):
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
    if uses_bin and binarizacion is not None:
        out_res = os.path.join(out_res, str(binarizacion))
        out_fit = os.path.join(out_fit, str(binarizacion))
    os.makedirs(out_res, exist_ok=True)
    os.makedirs(out_fit, exist_ok=True)

    # Subcarpetas por tipo de resumen
    out_res_fitness = os.path.join(out_res, "fitness")
    out_res_times = os.path.join(out_res, "times")
    out_res_percentage = os.path.join(out_res, "percentage")
    out_res_diversity = os.path.join(out_res, "diversity")
    out_res_gap = os.path.join(out_res, "gap")
    os.makedirs(out_res_fitness, exist_ok=True)
    os.makedirs(out_res_times, exist_ok=True)
    os.makedirs(out_res_percentage, exist_ok=True)
    os.makedirs(out_res_diversity, exist_ok=True)
    os.makedirs(out_res_gap, exist_ok=True)

    suf = ""

    fh_fit = open(os.path.join(out_fit, f"fitness_{sub}_{instancia_id}{suf}.csv"), "w")
    fh_rf = open(
        os.path.join(out_res_fitness, f"resumen_fitness_{sub}_{instancia_id}{suf}.csv"),
        "w",
    )
    fh_rt = open(
        os.path.join(out_res_times, f"resumen_times_{sub}_{instancia_id}{suf}.csv"), "w"
    )
    fh_rp = open(
        os.path.join(
            out_res_percentage, f"resumen_percentage_{sub}_{instancia_id}{suf}.csv"
        ),
        "w",
    )
    fh_div = open(
        os.path.join(
            out_res_diversity, f"resumen_diversity_{sub}_{instancia_id}{suf}.csv"
        ),
        "w",
    )
    fh_gap = open(
        os.path.join(out_res_gap, f"resumen_gap_{sub}_{instancia_id}{suf}.csv"), "w"
    )

    fh_fit.write("MH, FITNESS\n")
    fh_rf.write("MH, min best, avg. best, std. best\n")
    fh_rt.write("MH, min time (s), avg. time (s), std time (s)\n")
    fh_rp.write("MH, avg. XPL%, avg. XPT%\n")
    fh_div.write("MH, avg. ENT, avg. Divj_mean, avg. Divj_min, avg. Divj_max\n")
    fh_gap.write("MH, avg. GAP, avg. RDP\n")

    _procesar_archivos(
        cfg, instancia_id, blob, fh_fit, mhs_map, bin_actual=binarizacion
    )

    # Si no se obtuvo ningún dato útil, evitar generar resúmenes/gráficos vacíos
    has_any_data = any(
        (len(mhs_map[name].fitness) > 0) or (len(mhs_map[name].time) > 0)
        for name in MHS_LIST
    )
    if not has_any_data:
        print(
            f"      [WARN] Sin datos para {pid}{suf} - se omite escritura de resúmenes/gráficos"
        )
        fh_fit.close()
        fh_rf.close()
        fh_rt.close()
        fh_rp.close()
        fh_div.close()
        fh_gap.close()
        return None

    # Avisar si alguna MH no aportó datos para esta instancia
    faltantes = [
        name
        for name in MHS_LIST
        if len(mhs_map[name].fitness) == 0 and len(mhs_map[name].time) == 0
    ]
    if faltantes:
        print(f"      [WARN] Sin datos ({len(faltantes)}): {', '.join(faltantes)}")

    escribir_resumenes(mhs_map, fh_rf, fh_rt, fh_rp, MHS_LIST)
    _graficar_best(
        instancia_id,
        mhs_map,
        subfolder=sub,
        title_prefix=title_prefix,
        binarizacion=binarizacion,
    )
    _graficar_box_violin(
        instancia_id,
        subfolder=sub,
        binarizacion=binarizacion,
        title_prefix=title_prefix,
        num_experimentos=num_experimentos,
        out_fit_dir=out_fit,
    )
    _escribir_div_gap(cfg, instancia_id, mhs_map, binarizacion, fh_div, fh_gap)

    fh_fit.close()
    fh_rf.close()
    fh_rt.close()
    fh_rp.close()
    fh_div.close()
    fh_gap.close()

    return f"{pid}{suf} completado"


def analizar_problema(problem_name: str, n_jobs=-1):
    """Analiza un problema con paralelización y caché."""
    cfg = PROBLEMS[problem_name]
    sub, inst_key, uses_bin, title_prefix = (
        cfg["sub"],
        cfg["inst_key"],
        cfg["uses_bin"],
        cfg["title_prefix"],
    )

    # Verificar si el problema está habilitado en la configuración
    all_domains = get_all_domains()
    entry = all_domains.get(problem_name)
    if entry is None:
        print(f"[WARN] Dominio '{problem_name}' no registrado — se omite análisis")
        return

    if not EXPERIMENTS.get(entry.config_key, False):
        print(
            f"[INFO] Análisis {sub} deshabilitado en la configuración (experiments_config.json)"
        )
        return

    os.makedirs(DIR_TRANSITORIO, exist_ok=True)

    lista_inst = list(EXPERIMENTS["instancias"][inst_key])
    instancias = list(_obtener_instancias_cached(tuple(lista_inst)))

    if not instancias:
        print(f"[INFO] No hay instancias para procesar en {sub}")
        return

    num_experimentos = EXPERIMENTS["experimentos"][inst_key].get("num_experimentos", 1)

    print(f"[INFO] Ejecutando análisis {sub} con paralelización...")
    print(f"[INFO] Número de experimentos: {num_experimentos}")
    print("[INFO] Pre-cargando datos en caché...")

    incluir_bin = cfg["obtenerArchivos_kwargs"].get("incluir_binarizacion", True)
    for inst in instancias:
        _obtener_blob_cached(inst[1], incluir_bin)

    print(f"[INFO] Caché pre-cargado: {len(_CACHE_BLOB)} blobs")
    print("-" * 50, "\nIniciando procesamiento de instancias...\n")

    ds_actions = EXPERIMENTS.get("DS_actions", []) if uses_bin else []
    tareas = []
    for inst in instancias:
        instancia_id = inst[1]
        if uses_bin:
            bins_inst = _seleccionar_binarizaciones_disponibles(
                instancia_id, ds_actions
            )
            if not bins_inst:
                print(f"[WARN] {sub} {instancia_id}: sin binarizaciones detectadas")
                continue
            for bin_val in bins_inst:
                tareas.append(
                    (inst, bin_val, cfg, uses_bin, title_prefix, num_experimentos)
                )
        else:
            tareas.append((inst, None, cfg, uses_bin, title_prefix, num_experimentos))

    resultados = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_procesar_instancia_binarizacion)(*tarea) for tarea in tareas
    )

    completados = [r for r in resultados if r is not None]

    print(f"\n[INFO] Análisis {sub} completado con éxito.")
    print(f"[INFO] Procesadas {len(completados)} combinaciones instancia-binarización")
    _mostrar_estadisticas_cache()
    print("-" * 50)


def main():
    """Ejecuta el análisis con paralelización y caché."""
    start_time = time.time()

    try:
        for problem_name in PROBLEMS:
            analizar_problema(problem_name, n_jobs=-1)
    finally:
        end_time = time.time()
        print(f"\n{'=' * 50}")
        print(f"Tiempo total de ejecución: {end_time - start_time:.2f} segundos")
        print(f"{'=' * 50}")
        limpiar_cache_bd()
