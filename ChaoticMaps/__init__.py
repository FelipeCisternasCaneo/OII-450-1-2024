"""Paquete de mapas caóticos.

Este archivo expone una API pública estable para que otros módulos puedan
importar directamente desde `ChaoticMaps`.
"""

from .chaoticMaps import (  # noqa: F401
	CHAOTIC_MAP_NAMES,
	circleMap,
	get_chaotic_map,
	logisticMap,
	piecewiseMap,
	sineMap,
	singerMap,
	sinusoidalMap,
	tentMap,
)

__all__ = [
	"CHAOTIC_MAP_NAMES",
	"get_chaotic_map",
	"logisticMap",
	"piecewiseMap",
	"sineMap",
	"singerMap",
	"sinusoidalMap",
	"tentMap",
	"circleMap",
]
