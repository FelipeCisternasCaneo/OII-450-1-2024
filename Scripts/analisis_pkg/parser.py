import os
import pandas as pd
from analisis_pkg.config import MHS_LIST, COLUMN_MAPPINGS


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
    columnas_obligatorias = ["best_fitness", "time"]
    columnas_faltantes = [
        col for col in columnas_obligatorias if col not in cols_encontradas
    ]

    if columnas_faltantes:
        cols_disponibles = ", ".join(data.columns.tolist())
        return (
            None,
            False,
            f"Faltan columnas: {columnas_faltantes}. Disponibles: [{cols_disponibles}]",
        )

    # VALIDAR QUE LAS COLUMNAS SEAN NUMÉRICAS
    for col in ["best_fitness", "time"]:
        try:
            # Intentar convertir a numérico
            data[col] = pd.to_numeric(data[col], errors="coerce")

            # Si después de la conversión todo es NaN, el CSV está corrupto
            if data[col].isna().all():
                return (
                    None,
                    False,
                    f"Columna '{col}' contiene solo valores no numéricos o está corrupta",
                )

            # Eliminar filas con valores NaN en columnas críticas
            if data[col].isna().any():
                filas_antes = len(data)
                data = data.dropna(subset=[col])
                filas_despues = len(data)
                if filas_despues == 0:
                    return (
                        None,
                        False,
                        f"Todas las filas tienen valores inválidos en '{col}'",
                    )
                print(
                    f"      [WARN] '{nombre_archivo}': {filas_antes - filas_despues} filas con NaN en '{col}' eliminadas"
                )

        except Exception as e:
            return None, False, f"Error al validar columna '{col}': {str(e)}"

    # Validar columnas opcionales de diversidad
    for col in ["ENT", "Divj_mean", "Divj_min", "Divj_max", "GAP", "RDP"]:
        if col in data.columns:
            try:
                data[col] = pd.to_numeric(data[col], errors="coerce")
            except:
                # Si falla, simplemente eliminar la columna
                data = data.drop(columns=[col])

    return data, True, None


def _actualizar_datos(mhs_instances, mh, archivo_fitness, data):
    """Versión vectorizada con validación"""
    inst = mhs_instances[mh]

    # VALIDAR que las columnas sean numéricas antes de calcular
    if not pd.api.types.is_numeric_dtype(data["best_fitness"]):
        print(f"      [ERROR] Columna 'best_fitness' no es numérica para {mh}")
        return

    if not pd.api.types.is_numeric_dtype(data["time"]):
        print(f"      [ERROR] Columna 'time' no es numérica para {mh}")
        return

    # Operaciones vectorizadas (más rápidas)
    inst.fitness.append(data["best_fitness"].min())
    inst.time.append(data["time"].sum().round(3))

    # XPL y XPT son opcionales
    if "XPL" in data.columns and pd.api.types.is_numeric_dtype(data["XPL"]):
        inst.xpl.append(data["XPL"].mean().round(2))
    if "XPT" in data.columns and pd.api.types.is_numeric_dtype(data["XPT"]):
        inst.xpt.append(data["XPT"].mean().round(2))

    archivo_fitness.write(f"{mh}, {inst.fitness[-1]}\n")

    # Diversidad usando operaciones de pandas (más eficientes)
    for col, target in [
        ("ENT", "ent"),
        ("Divj_mean", "divj_mean"),
        ("Divj_min", "divj_min"),
        ("Divj_max", "divj_max"),
        ("GAP", "gap"),
        ("RDP", "rdp"),
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

    # Caso 0 (preferido): si el nombre empieza con una MH conocida, inferir mh por prefijo.
    for mh_name in sorted(MHS_LIST, key=len, reverse=True):
        prefix = f"{mh_name}_"
        if base.startswith(prefix):
            tail = base[len(prefix) :]
            tail_parts = [p for p in tail.split("_") if p]
            instancia = tail_parts[0] if len(tail_parts) >= 1 else None
            corrida = (
                tail_parts[1]
                if len(tail_parts) >= 2 and tail_parts[1].isdigit()
                else None
            )
            return mh_name, instancia, corrida

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
