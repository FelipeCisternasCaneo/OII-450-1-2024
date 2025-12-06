import sqlite3
import json
import re
import textwrap
from collections import defaultdict
from typing import List, Any

# --- CONFIGURACIÓN ---
DB_FILE = './BD/resultados.db'
TABLA_INSTANCIAS = 'instancias'
TABLA_EXPERIMENTOS = 'experimentos'

# Dimensiones fijas para BEN
DIMENSIONES_BEN_DEFAULT = {f"F{i}": [30] for i in range(1, 14)}
DIMENSIONES_BEN_DEFAULT.update({
    "F14": [2], "F15": [4], "F16": [2], "F17": [2], "F18": [2],
    "F19": [3], "F20": [6], "F21": [4], "F22": [4], "F23": [4]
})

COMENTARIOS = {
    "ben": "Indica si se van a procesar funciones BEN.",
    "scp": "Indica si se van a procesar instancias SCP.",
    "uscp": "Indica si se van a procesar instancias USCP.",
    "kp": "Indica si se van a procesar instancias KP.",
    "mhs": "Lista de metaheurísticas que se van a procesar.",
    "DS_actions": "Lista de acciones de binarización de la población.",
    "dimensiones": "Mapea cada función BEN a las dimensiones que utiliza.",
    "experimentos": "Parámetros globales de iteraciones y población para cada problema.",
    "instancias": "Identificadores de los instancias de los problemas que se van a procesar."
}

# --- HERRAMIENTAS DE EXTRACCIÓN (Lógica de BD) ---

def natural_sort_key(s: str) -> List:
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', str(s))]

def ejecutar_consulta(cursor: sqlite3.Cursor, query: str, params: tuple = None) -> List[Any]:
    cursor.execute(query, params or ())
    return cursor.fetchall()

def obtener_binarizaciones(cursor: sqlite3.Cursor) -> List[str]:
    # 1. Nueva versión
    try:
        res = ejecutar_consulta(cursor, f"SELECT DISTINCT binarizacion FROM {TABLA_EXPERIMENTOS} WHERE binarizacion IS NOT NULL AND binarizacion != '' AND binarizacion != 'N/A'")
        if res: return sorted([r[0] for r in res])
    except: pass
    # 2. Antigua versión
    try:
        res = ejecutar_consulta(cursor, f"SELECT DISTINCT paramMH FROM {TABLA_EXPERIMENTOS} WHERE paramMH LIKE '%DS:%'")
        bins = set()
        for (p,) in res:
            m = re.search(r'DS:([A-Za-z0-9\-]+)', p)
            if m: bins.add(m.group(1))
        return sorted(list(bins))
    except:
        return []

def limpiar_instancia(nombre: str, tipo: str) -> str:
    if tipo == 'SCP': return nombre.replace('scp', '')
    elif tipo == 'USCP': return nombre.replace('uscp', '')
    elif tipo == 'KP': return nombre.replace('kp', '')
    return nombre

# --- HERRAMIENTAS DE FORMATO (Lógica Visual) ---

def fmt_bool(val: bool) -> str:
    return "true" if val else "false"

def fmt_list_inline(lista: list) -> str:
    """Convierte ['a', 'b'] en la cadena '["a", "b"]'."""
    return json.dumps(lista, ensure_ascii=False)

def fmt_dict_inline(diccionario: dict) -> str:
    """Convierte {'a': 1} en '{"a": 1}'."""
    return json.dumps(diccionario, ensure_ascii=False)

def fmt_dimensiones_ben(dims: dict) -> str:
    """
    Formatea el bloque de dimensiones para que parezca una cuadrícula.
    Agrupa por filas para que no sea una lista infinita vertical.
    """
    if not dims: return "{}"
    
    # Ordenar por el número de función F1, F2...
    keys = sorted(dims.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
    
    items = []
    for k in keys:
        val = json.dumps(dims[k])
        items.append(f'"{k}": {val}')
    
    # Agrupar elementos
    lines = []
    current_line = []
    current_len = 0
    
    for item in items:
        if current_len + len(item) > 85 and current_line:
            lines.append(", ".join(current_line))
            current_line = []
            current_len = 0
        current_line.append(item)
        current_len += len(item) + 2
    
    if current_line:
        lines.append(", ".join(current_line))
        
    contenido = ",\n            ".join(lines)
    return f"{{\n            {contenido}\n        }}"

def fmt_instancias_wrap(lista: list, indent: str = "            ") -> str:
    """
    Formatea una lista de strings para que haga 'wrap' (salto de línea)
    si es muy larga, manteniendo la indentación.
    """
    if not lista: return "[]"
    json_items = [json.dumps(x) for x in lista]
    full_str = ", ".join(json_items)
    
    if len(full_str) < 80:
        return f"[{full_str}]"
    
    # Ajuste de texto manual para listas largas
    wrapper = textwrap.TextWrapper(width=80, break_long_words=False, break_on_hyphens=False)
    wrapped_lines = wrapper.wrap(full_str)
    
    # Añadir indentación a partir de la segunda línea
    aligned_lines = [wrapped_lines[0]]
    for line in wrapped_lines[1:]:
        aligned_lines.append(indent + line)
        
    # Corrección: Join fuera del f-string para compatibilidad
    contenido = '\n'.join(aligned_lines)
    return f"[{contenido}]"


def generar_json_string(data: dict) -> str:
    """Construye el string JSON final pieza por pieza."""
    parts = []
    parts.append("{")
    
    # 1. Comentarios
    if "_comentarios" in data:
        coms = json.dumps(data["_comentarios"], ensure_ascii=False, indent=4)
        # Indentamos el bloque entero
        coms = "\n".join("    " + line for line in coms.splitlines())
        parts.append(f'    "_comentarios": {coms.strip()},')
        parts.append("") # Espacio vacío

    # 2. Booleans
    parts.append(f'    "ben": {fmt_bool(data["ben"])},')
    parts.append(f'    "scp": {fmt_bool(data["scp"])},')
    parts.append(f'    "uscp": {fmt_bool(data["uscp"])},')
    parts.append(f'    "kp": {fmt_bool(data["kp"])},')
    parts.append("")

    # 3. Listas Simples (MHS, Actions)
    parts.append(f'    "mhs": {fmt_list_inline(data["mhs"])},')
    parts.append("")
    parts.append(f'    "DS_actions": {fmt_list_inline(data["DS_actions"])},')
    parts.append("")

    # 4. Dimensiones (Formato especial cuadrícula)
    parts.append('    "dimensiones": {')
    dims_ben_str = fmt_dimensiones_ben(data["dimensiones"].get("BEN", {}))
    parts.append(f'        "BEN": {dims_ben_str}')
    parts.append('    },')
    parts.append("")

    # 5. Experimentos (Key: Value en una línea)
    parts.append('    "experimentos": {')
    exps = []
    for k, v in data["experimentos"].items():
        exps.append(f'        "{k}": {fmt_dict_inline(v)}')
    parts.append(",\n".join(exps))
    parts.append('    },')
    parts.append("")

    # 6. Instancias (Con wrap para BEN)
    parts.append('    "instancias": {')
    inst_lines = []
    for k, v in data["instancias"].items():
        val_str = fmt_instancias_wrap(v)
        inst_lines.append(f'        "{k}": {val_str}')
    parts.append(",\n".join(inst_lines))
    parts.append('    }')

    parts.append("}")
    return "\n".join(parts)


def analizar_base_de_datos(db_path: str):
    print(f"--- LEYENDO BD: {db_path} ---")
    
    data_final = {}
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # --- Recolección de Datos ---
        query = f"SELECT DISTINCT i.tipo_problema FROM {TABLA_EXPERIMENTOS} e JOIN {TABLA_INSTANCIAS} i ON e.fk_id_instancia = i.id_instancia"
        tipos = sorted([r[0] for r in ejecutar_consulta(cursor, query)])

        mhs = sorted([r[0] for r in ejecutar_consulta(cursor, f"SELECT DISTINCT MH FROM {TABLA_EXPERIMENTOS}")])
        ds_actions = obtener_binarizaciones(cursor)

        # Inicializamos los 4 problemas
        instancias = {k: [] for k in ['BEN', 'SCP', 'USCP', 'KP']}
        dims_ben = DIMENSIONES_BEN_DEFAULT.copy()

        for prob in tipos:
            query = f"SELECT DISTINCT i.nombre FROM {TABLA_EXPERIMENTOS} e JOIN {TABLA_INSTANCIAS} i ON e.fk_id_instancia = i.id_instancia WHERE i.tipo_problema = ? ORDER BY i.nombre"
            raw = [r[0] for r in ejecutar_consulta(cursor, query, (prob,))]
            
            if prob == 'BEN':
                q_exp = f"SELECT DISTINCT e.experimento FROM {TABLA_EXPERIMENTOS} e JOIN {TABLA_INSTANCIAS} i ON e.fk_id_instancia = i.id_instancia WHERE i.tipo_problema = 'BEN'"
                exp_nombres = [r[0] for r in ejecutar_consulta(cursor, q_exp)]
                func_dims = defaultdict(set)
                for n in exp_nombres:
                    p = n.split()
                    if len(p)==2: func_dims[p[0]].add(int(p[1]))
                for f, d in func_dims.items(): dims_ben[f] = sorted(list(d))
                instancias['BEN'] = sorted(func_dims.keys(), key=natural_sort_key)
            elif prob in instancias:
                limpios = [limpiar_instancia(n, prob) for n in raw]
                instancias[prob] = sorted(list(set(limpios)), key=natural_sort_key)

        exp_params = {}
        defaults = {
            'BEN': {'iteraciones': 100, 'poblacion': 30, 'num_experimentos': 1},
            'SCP': {'iteraciones': 5, 'poblacion': 30, 'num_experimentos': 1},
            'USCP': {'iteraciones': 5, 'poblacion': 10, 'num_experimentos': 1},
            'KP': {'iteraciones': 5, 'poblacion': 10, 'num_experimentos': 1}
        }
        
        for prob in ['BEN', 'SCP', 'USCP', 'KP']:
            params = defaults[prob].copy()
            if prob in tipos:
                # Intentar sacar params reales
                res = ejecutar_consulta(cursor, f"SELECT e.paramMH FROM {TABLA_EXPERIMENTOS} e JOIN {TABLA_INSTANCIAS} i ON e.fk_id_instancia = i.id_instancia WHERE i.tipo_problema = ? LIMIT 1", (prob,))
                if res:
                    matches = re.findall(r'(iter|pop):(\d+)', res[0][0])
                    for k, v in matches:
                        if k == 'iter': params['iteraciones'] = int(v)
                        if k == 'pop': params['poblacion'] = int(v)
            exp_params[prob] = params

        # --- Construcción del Diccionario de Datos ---
        data_final = {
            "_comentarios": COMENTARIOS,
            "ben": "BEN" in tipos,
            "scp": "SCP" in tipos,
            "uscp": "USCP" in tipos,
            "kp": "KP" in tipos,
            "mhs": mhs,
            "DS_actions": ds_actions,
            "dimensiones": {"BEN": dims_ben},
            "experimentos": exp_params,
            "instancias": instancias
        }

    # --- Generación del JSON Visual ---
    json_output = generar_json_string(data_final)
    
    print("\n" + "="*50)
    print("--- RESULTADO ---")
    print("="*50 + "\n")
    print(json_output)

if __name__ == '__main__':
    analizar_base_de_datos(DB_FILE)