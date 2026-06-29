from bd.sqlite import BD
from analisis_pkg.config import CACHE_MAX_SIZE

# ========= CACHÉ DE BD (Versión picklable) =========
_CACHE_BLOB = {}
_CACHE_INSTANCIAS = {}
_CACHE_BINARIZACIONES = {}


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


def _obtener_instancias_cached(nombres_tuple):
    """Cachea las consultas de instancias. Recibe una tupla de nombres (hashable)."""
    if nombres_tuple in _CACHE_INSTANCIAS:
        return _CACHE_INSTANCIAS[nombres_tuple]

    bd_local = BD()
    instancias = bd_local.obtenerInstancias(list(nombres_tuple))
    result = tuple(instancias) if instancias else tuple()
    _CACHE_INSTANCIAS[nombres_tuple] = result

    return result


def _obtener_binarizaciones_cached(instancia_id: str):
    """Cachea binarizaciones disponibles por instancia."""
    if instancia_id in _CACHE_BINARIZACIONES:
        return _CACHE_BINARIZACIONES[instancia_id]

    bd_local = BD()
    bins = bd_local.obtenerBinarizaciones(instancia_id)
    result = tuple(bins) if bins else tuple()
    _CACHE_BINARIZACIONES[instancia_id] = result
    return result


def limpiar_cache_bd():
    """Limpia el caché de BD."""
    global _CACHE_BLOB, _CACHE_INSTANCIAS, _CACHE_BINARIZACIONES
    _CACHE_BLOB.clear()
    _CACHE_INSTANCIAS.clear()
    _CACHE_BINARIZACIONES.clear()
    print("[INFO] Caché de BD limpiado")


def _mostrar_estadisticas_cache():
    """Muestra estadísticas de uso del caché."""
    print(f"\n[CACHÉ] Estadísticas:")
    print(f"  Blobs en caché: {len(_CACHE_BLOB)}/{CACHE_MAX_SIZE}")
    print(f"  Instancias en caché: {len(_CACHE_INSTANCIAS)}")
    print(f"  Binarizaciones en caché: {len(_CACHE_BINARIZACIONES)}")


def _seleccionar_binarizaciones_disponibles(instancia_id: str, ds_actions):
    """Selecciona binarizaciones a analizar para una instancia.

    - Si `ds_actions` está configurado, se incluyen:
      - coincidencias exactas con una acción base
      - variantes caóticas que empiezan con "<accion>_" (por ejemplo: "S2_Logistic")
    - Si `ds_actions` está vacío, se usan todas las binarizaciones detectadas en la BD.
    """
    disponibles = [
        str(b).strip()
        for b in _obtener_binarizaciones_cached(instancia_id)
        if str(b).strip()
    ]
    if not disponibles:
        return list(ds_actions) if ds_actions else []

    ds_actions = [str(a).strip() for a in (ds_actions or []) if str(a).strip()]
    if not ds_actions:
        return disponibles

    base_set = set(ds_actions)
    selected = []
    for b in disponibles:
        if b in base_set:
            selected.append(b)
            continue
        for a in ds_actions:
            if b.startswith(f"{a}_"):
                selected.append(b)
                break

    # Fallback: si nada matchea, mantener DS_actions para no romper flujos viejos
    return selected if selected else list(ds_actions)
