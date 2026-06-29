import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from analisis_pkg.config import (
    DIR_GRAFICOS,
    DIR_BOXPLOT,
    DIR_VIOLIN,
    DIR_BEST,
    DIR_FITNESS,
    MODO_LOGARITMICO,
    MODO_TERMINACION,
    MHS_LIST,
)


def _graficar_por_corrida(
    iteraciones,
    fitness,
    xpl,
    xpt,
    tiempo,
    mh,
    problem_id,
    corrida,
    subfolder,
    binarizacion=None,
):
    out = (
        os.path.join(DIR_GRAFICOS, subfolder)
        if binarizacion is None
        else os.path.join(DIR_GRAFICOS, subfolder, str(binarizacion))
    )
    os.makedirs(out, exist_ok=True)

    # Convergencia con modo logarítmico según configuración
    fpath = os.path.join(
        out, f"Convergence_{mh}_{subfolder}_{problem_id}_{corrida}.pdf"
    )
    _, ax = plt.subplots()

    # Determinar modo de visualización
    fitness_array = np.array(fitness)
    y_data = fitness_array
    ylabel = "Fitness"

    if MODO_LOGARITMICO == "log_transform":
        # ln(fitness) como en TJO original
        # Evitar log(0) o log(negativos)
        y_data = np.log(np.maximum(fitness_array, 1e-10))
        ylabel = "Ln Objective function"
    elif MODO_LOGARITMICO == "log_scale":
        # Escala logarítmica en el eje (valores originales)
        ax.set_yscale("log")
        ylabel = "Fitness (log scale)"
    elif MODO_LOGARITMICO == "auto":
        # Solo usar escala log si valores muy grandes
        fitness_range = np.max(fitness_array) - np.min(fitness_array)
        fitness_max = np.max(fitness_array)
        if fitness_max > 0 and (
            fitness_range / fitness_max > 1e-3 and fitness_max > 1e6
        ):
            ax.set_yscale("log")
            ylabel = "Fitness (log scale)"

    ax.plot(iteraciones, y_data)
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Convergence {mh}\n{problem_id} - Run {corrida}"
        + (f" - ({binarizacion})" if binarizacion else "")
    )
    ax.set_xlabel(
        "Function Evaluations (FE)"
        if MODO_TERMINACION in ["fe", "both"]
        else "Iteration"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()

    # XPL vs XPT
    fpath = os.path.join(out, f"Percentage_{mh}_{subfolder}_{problem_id}_{corrida}.pdf")
    _, ax = plt.subplots()
    ax.plot(
        iteraciones,
        xpl,
        color="r",
        label=rf"$\overline{{XPL}}$: {np.round(np.mean(xpl), 2)}%",
    )
    ax.plot(
        iteraciones,
        xpt,
        color="b",
        label=rf"$\overline{{XPT}}$: {np.round(np.mean(xpt), 2)}%",
    )
    ax.set_title(
        f"XPL% - XPT% {mh}\n{problem_id} - Run {corrida}"
        + (f" - ({binarizacion})" if binarizacion else "")
    )
    ax.set_ylabel("Percentage")
    ax.set_xlabel(
        "Function Evaluations (FE)"
        if MODO_TERMINACION in ["fe", "both"]
        else "Iteration"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()

    # Tiempo por iteración
    fpath = os.path.join(out, f"Time_{mh}_{subfolder}_{problem_id}_{corrida}.pdf")
    _, ax = plt.subplots()
    ax.plot(iteraciones, tiempo, label="Time per Iteration")
    ax.set_title(
        f"Time per Iteration {mh}\n{problem_id} - Run {corrida}"
        + (f" - ({binarizacion})" if binarizacion else "")
    )
    ax.set_ylabel("Time (s)")
    ax.set_xlabel(
        "Function Evaluations (FE)"
        if MODO_TERMINACION in ["fe", "both"]
        else "Iteration"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()


def _graficar_box_violin(
    instancia_id,
    subfolder,
    binarizacion=None,
    title_prefix="",
    num_experimentos=1,
    out_fit_dir=None,
):
    """
    Genera boxplot y violinplot solo si hay múltiples experimentos.
    """
    base_fit_dir = out_fit_dir or os.path.join(DIR_FITNESS, subfolder)
    datos_path = os.path.join(base_fit_dir, f"fitness_{subfolder}_{instancia_id}.csv")
    try:
        df = pd.read_csv(datos_path)
        df.columns = df.columns.str.strip()
        if "FITNESS" not in df.columns or "MH" not in df.columns:
            print(f"[ERROR] Columnas necesarias no encontradas en {datos_path}")
            return
    except FileNotFoundError:
        print(f"[ERROR] Archivo no encontrado: {datos_path}")
        return
    except Exception as e:
        print(f"[ERROR] Error al leer {datos_path}: {e}")
        return

    # Determinar la cantidad real de corridas/experimentos por MH
    if df.empty:
        return
    max_runs = df["MH"].value_counts().max()
    if max_runs <= 1:
        return

    # Detectar si conviene escala log
    fitness_vals = (
        pd.to_numeric(df["FITNESS"], errors="coerce").dropna().to_numpy(dtype=float)
    )
    fitness_vals = fitness_vals[np.isfinite(fitness_vals)]
    fitness_min = float(np.min(fitness_vals)) if fitness_vals.size else np.nan
    fitness_max = float(np.max(fitness_vals)) if fitness_vals.size else np.nan
    use_log_y = False
    if np.isfinite(fitness_min) and np.isfinite(fitness_max) and fitness_min > 0:
        if MODO_LOGARITMICO == "log_scale":
            use_log_y = True
        elif MODO_LOGARITMICO == "auto" and fitness_max > 1e6:
            use_log_y = True

    # Boxplot
    out = os.path.join(DIR_BOXPLOT, subfolder)
    os.makedirs(out, exist_ok=True)
    if binarizacion:
        out = os.path.join(DIR_BOXPLOT, subfolder, str(binarizacion))
        os.makedirs(out, exist_ok=True)
    fpath = os.path.join(out, f"boxplot_fitness_{subfolder}_{instancia_id}.pdf")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="MH", y="FITNESS", data=df, hue="MH", palette="Set2", legend=False)
    if use_log_y:
        plt.yscale("log")
        ylabel = "Fitness (log scale)"
    else:
        ylabel = "Fitness"
    plt.title(
        f"Boxplot Fitness\n{title_prefix}{instancia_id}"
        + (f" - {binarizacion}" if binarizacion else "")
    )
    plt.xlabel("Metaheurística")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()

    # Violin
    out = os.path.join(DIR_VIOLIN, subfolder)
    os.makedirs(out, exist_ok=True)
    if binarizacion:
        out = os.path.join(DIR_VIOLIN, subfolder, str(binarizacion))
        os.makedirs(out, exist_ok=True)
    fpath = os.path.join(out, f"violinplot_fitness_{subfolder}_{instancia_id}.pdf")
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="MH", y="FITNESS", data=df, hue="MH", palette="Set3", legend=False)
    ylabel = "Fitness"
    plt.title(
        f"Violinplot Fitness\n{title_prefix}{instancia_id}"
        + (f" - {binarizacion}" if binarizacion else "")
    )
    plt.xlabel("Metaheurística")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()


def _graficar_best(
    instancia_id, mhs_instances, subfolder, title_prefix="", binarizacion=None
):
    # Si no hay ninguna serie disponible, no generar PDFs vacíos
    has_any_fitness = any(len(mhs_instances[name].bestFitness) for name in MHS_LIST)
    has_any_time = any(len(mhs_instances[name].bestTime) for name in MHS_LIST)
    has_any_xplxpt = any(
        (
            mhs_instances[name].xpl_iter is not None
            and mhs_instances[name].xpt_iter is not None
        )
        for name in MHS_LIST
    )
    if not (has_any_fitness or has_any_time or has_any_xplxpt):
        print(
            f"      [WARN] Sin series para graficar: {subfolder} {instancia_id}"
            + (f" ({binarizacion})" if binarizacion else "")
        )
        return

    # detectar mejores
    best_f, best_t, best_f_mh, best_t_mh = np.inf, np.inf, "", ""
    for name in MHS_LIST:
        mh = mhs_instances[name]
        min_f = min(mh.bestFitness) if len(mh.bestFitness) > 0 else np.inf
        min_t = min(mh.bestTime) if len(mh.bestTime) > 0 else np.inf
        if min_f < best_f:
            best_f, best_f_mh = min_f, name
        if min_t < best_t:
            best_t, best_t_mh = min_t, name

    out = os.path.join(DIR_BEST, subfolder)
    if binarizacion:
        out = os.path.join(out, str(binarizacion))
    os.makedirs(out, exist_ok=True)
    suf = ""
    title_id = f"{title_prefix}{instancia_id}" + (
        f" - {binarizacion}" if binarizacion else ""
    )

    # Fitness con modo logarítmico según configuración
    fig, ax = plt.subplots(figsize=(10, 6))
    all_fitness_values = []
    ylabel = "Fitness"

    # Determinar modo de visualización
    if MODO_LOGARITMICO == "log_transform":
        # ln(fitness) como en TJO original
        for name in MHS_LIST:
            mh = mhs_instances[name]
            if len(mh.bestFitness):
                fitness_vals = np.array(mh.bestFitness)
                all_fitness_values.extend(fitness_vals)
                y_data = np.log(np.maximum(fitness_vals, 1e-10))
                x_data = (
                    mh.iter_vector
                    if getattr(mh, "iter_vector", None) is not None
                    and len(getattr(mh, "iter_vector", [])) == len(y_data)
                    else range(len(y_data))
                )
                ax.plot(x_data, y_data, label=name, linewidth=2)
        ylabel = "Ln Objective function"
    else:
        # Graficar valores originales
        for name in MHS_LIST:
            mh = mhs_instances[name]
            if len(mh.bestFitness):
                fitness_vals = np.array(mh.bestFitness)
                all_fitness_values.extend(fitness_vals)
                x_data = (
                    mh.iter_vector
                    if getattr(mh, "iter_vector", None) is not None
                    and len(getattr(mh, "iter_vector", [])) == len(fitness_vals)
                    else range(len(fitness_vals))
                )
                ax.plot(x_data, fitness_vals, label=name, linewidth=2)

        # Determinar escala del eje
        if len(all_fitness_values) > 0:
            all_vals = np.asarray(all_fitness_values, dtype=float)
            all_vals = all_vals[np.isfinite(all_vals)]
            if all_vals.size > 0:
                fitness_min = float(np.min(all_vals))
                fitness_max = float(np.max(all_vals))
            else:
                fitness_min, fitness_max = np.nan, np.nan

            if MODO_LOGARITMICO == "log_scale":
                if np.isfinite(fitness_min) and fitness_min > 0:
                    ax.set_yscale("log")
                    ylabel = "Fitness (log scale)"
            elif MODO_LOGARITMICO == "auto":
                if (
                    np.isfinite(fitness_min)
                    and fitness_min > 0
                    and np.isfinite(fitness_max)
                ):
                    if fitness_max > 1e6:
                        ax.set_yscale("log")
                        ylabel = "Fitness (log scale)"

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"Best Fitness per MH\n{title_id}\nBest: {best_f_mh} ({best_f:.4e})",
        fontsize=13,
    )
    eje_x_lbl = (
        "Function Evaluations (FE)"
        if MODO_TERMINACION in ["fe", "both"]
        else "Iteration"
    )
    ax.set_xlabel(eje_x_lbl, fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if has_any_fitness:
        plt.savefig(os.path.join(out, f"fitness_{subfolder}_{instancia_id}{suf}.pdf"))
    plt.close()

    # Time
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in MHS_LIST:
        mh = mhs_instances[name]
        if len(mh.bestTime):
            x_data = (
                mh.iter_vector
                if getattr(mh, "iter_vector", None) is not None
                and len(getattr(mh, "iter_vector", [])) == len(mh.bestTime)
                else range(len(mh.bestTime))
            )
            ax.plot(x_data, mh.bestTime, label=name, linewidth=2)

    ax.set_title(
        f"Best Time per MH\n{title_id}\nBest: {best_t_mh} ({best_t:.2f} s)", fontsize=13
    )
    ax.set_ylabel("Time (s)", fontsize=12)
    eje_x_lbl = (
        "Function Evaluations (FE)"
        if MODO_TERMINACION in ["fe", "both"]
        else "Iteration"
    )
    ax.set_xlabel(eje_x_lbl, fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if has_any_time:
        plt.savefig(os.path.join(out, f"time_{subfolder}_{instancia_id}{suf}.pdf"))
    plt.close()

    # XPL vs XPT combinado
    colors = [
        "tab:red",
        "tab:blue",
        "tab:green",
        "tab:orange",
        "tab:purple",
        "tab:cyan",
        "tab:pink",
        "tab:brown",
        "tab:olive",
        "tab:gray",
    ]
    cmap = {name: colors[i % len(colors)] for i, name in enumerate(MHS_LIST)}

    plt.figure(figsize=(10, 6))
    any_series, max_len = False, 0
    for name in MHS_LIST:
        mh = mhs_instances[name]
        if mh.xpl_iter is None or mh.xpt_iter is None:
            continue
        any_series = True
        x = (
            mh.iter_vector
            if getattr(mh, "iter_vector", None) is not None
            and len(getattr(mh, "iter_vector", [])) == len(mh.xpl_iter)
            else np.arange(len(mh.xpl_iter))
        )
        c = cmap[name]
        plt.plot(
            x,
            mh.xpt_iter,
            linestyle="-",
            linewidth=2,
            label=f"{name} XPT% (avg {np.round(np.mean(mh.xpt_iter), 2)}%)",
            color=c,
        )
        plt.plot(
            x,
            mh.xpl_iter,
            linestyle="--",
            linewidth=2,
            label=f"{name} XPL% (avg {np.round(np.mean(mh.xpl_iter), 2)}%)",
            color=c,
        )
        max_len = max(max_len, len(x))

    if any_series:
        plt.title(
            f"Exploration (XPL) vs Exploitation (XPT) per MH\n{title_id}", fontsize=13
        )
        plt.ylabel("Percentage (%)", fontsize=12)
        eje_x_lbl = (
            "Function Evaluations (FE)"
            if MODO_TERMINACION in ["fe", "both"]
            else "Iteration"
        )
        plt.xlabel(eje_x_lbl, fontsize=12)
        plt.ylim(0, 100)
        if max_len <= 1:
            plt.xlim(-0.5, 0.5)
        plt.legend(loc="upper right", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out, f"xpl_xpt_{subfolder}_{instancia_id}{suf}.pdf"))
        plt.close()
    else:
        plt.close()
