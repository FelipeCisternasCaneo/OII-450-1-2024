import os
import pandas as pd
import numpy as np
from scipy import stats
import glob

def generar_reporte_latex(df, file_out, problem_name):
    """
    Genera un reporte en LaTeX (Kruskal-Wallis y Wilcoxon) para un dataframe
    que tiene columnas [MH, FITNESS, INSTANCE].
    """
    if df.empty:
        return
        
    instancias = df['INSTANCE'].unique()
    
    with open(file_out, 'w') as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{booktabs}\n")
        f.write("\\usepackage{multirow}\n")
        f.write("\\usepackage{geometry}\n")
        f.write("\\geometry{margin=1in}\n")
        f.write("\\begin{document}\n\n")
        
        for inst in instancias:
            df_inst = df[df['INSTANCE'] == inst]
            mhs = df_inst['MH'].unique()
            
            # Recolectar datos por MH
            data_por_mh = {}
            for mh in mhs:
                # Filtrar NaN e inf
                vals = pd.to_numeric(df_inst[df_inst['MH'] == mh]['FITNESS'], errors='coerce').dropna().to_numpy()
                if len(vals) > 0:
                    data_por_mh[mh] = vals
            
            if len(data_por_mh) < 2:
                continue
                
            # Kruskal-Wallis global
            try:
                kw_stat, kw_p = stats.kruskal(*data_por_mh.values())
            except ValueError:
                kw_p = 1.0
                
            # Calcular promedios para rankear
            mean_fitness = {mh: np.mean(vals) for mh, vals in data_por_mh.items()}
            # Identificar la mejor (menor fitness asumiendo minimización)
            best_mh = min(mean_fitness, key=mean_fitness.get)
            best_vals = data_por_mh[best_mh]
            
            f.write(f"\\section*{{Statistical Analysis - {problem_name} - {inst}}}\n")
            f.write(f"\\textbf{{Kruskal-Wallis p-value:}} {kw_p:.4e} \\\\\n\n")
            
            f.write("\\begin{table}[h!]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{lrrcc}\n")
            f.write("\\toprule\n")
            f.write("Algorithm & Mean Fitness & Std Dev & Wilcoxon p-value & Significance \\\\\n")
            f.write("\\midrule\n")
            
            # Ordenar por fitness prom
            sorted_mhs = sorted(mean_fitness.keys(), key=lambda x: mean_fitness[x])
            
            for mh in sorted_mhs:
                vals = data_por_mh[mh]
                mean_val = np.mean(vals)
                std_val  = np.std(vals)
                
                if mh == best_mh:
                    p_val_str = "-"
                    sig_str = "Best"
                else:
                    try:
                        _, p_val = stats.wilcoxon(best_vals, vals, zero_method='zsplit')
                        p_val_str = f"{p_val:.4e}"
                        # Significativo si p < 0.05
                        sig_str = "+" if p_val >= 0.05 else ("$\\approx$" if p_val >= 0.05 else "-")
                        # La convención normal: - significa la mejor es estadísticamente mejor (diferencia sig), 
                        # + si la otra MH es mejor (no pasaría si best_mh tiene menor promedio, a menos que outliers),
                        # = si no hay diferencia significativa
                        if p_val < 0.05:
                            sig_str = "$\\approx$" if mean_val == mean_fitness[best_mh] else "-" 
                        else:
                            sig_str = "$=$"
                    except ValueError: # muestras iguales o muy pequeñas
                        p_val_str = "N/A"
                        sig_str = "N/A"
                        
                f.write(f"{mh} & {mean_val:.4e} & {std_val:.4e} & {p_val_str} & {sig_str} \\\\\n")
                
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write(f"\\caption{{Pairwise Wilcoxon tests against best algorithm ({best_mh}) for {inst}. '=' means no significant difference, '-' means {best_mh} is significantly better.}}\n")
            f.write("\\end{table}\n\n")
            
        f.write("\\end{document}\n")
        
def procesar_archivos_fitness(directorio, pattern="fitness_*_*.csv"):
    """
    Busca los CSVs de fitness generados por analisis.py y genera un reporte consolidado.
    """
    archivos = glob.glob(os.path.join(directorio, "**", pattern), recursive=True)
    if not archivos:
        print(f"No se encontraron archivos en {directorio}")
        return
        
    dfs = []
    for arch in archivos:
        try:
            df = pd.read_csv(arch)
            df.columns = df.columns.str.strip()
            
            # Extraer problem y instancia del nombre de archivo (ej: fitness_BEN_F1.csv)
            base = os.path.basename(arch).replace(".csv", "").split("_")
            problem = base[1] if len(base) > 1 else "Unknown"
            inst = base[2] if len(base) > 2 else "Unknown"
            
            if 'MH' in df.columns and 'FITNESS' in df.columns:
                df['PROBLEM'] = problem
                df['INSTANCE'] = inst
                dfs.append(df)
        except Exception as e:
            print(f"Error procesando {arch}: {e}")
            
    if dfs:
        df_total = pd.concat(dfs, ignore_index=True)
        # Agrupar por problema
        for prob in df_total['PROBLEM'].unique():
            df_prob = df_total[df_total['PROBLEM'] == prob]
            out_file = f"Reporte_Estadistico_{prob}.tex"
            generar_reporte_latex(df_prob, out_file, prob)
            print(f"Reporte LaTeX generado: {out_file}")

if __name__ == "__main__":
    # Apunta al directorio donde `analisis.py` deposita fitness por corrida
    # (Por ej: Resultados/fitness/BEN/)
    directorio_fitness = os.path.join("Resultados", "fitness")
    if os.path.exists(directorio_fitness):
        procesar_archivos_fitness(directorio_fitness)
    else:
        print(f"Directorio no encontrado: {directorio_fitness}")
