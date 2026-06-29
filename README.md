# Framework de Optimización Metaheurística (OII-450-1-2024)

Este repositorio contiene un framework unificado para la ejecución y análisis de algoritmos de optimización metaheurística (como PSO, GWO, WOA, SBOA, etc.) aplicados a diferentes dominios de problemas (Benchmark CEC2017, Set Covering Problem - SCP, y Unicost Set Covering Problem - USCP).

---

## Estructura del Repositorio

El proyecto está organizado de manera modular y limpia bajo las siguientes carpetas:

```text
oii-450-1-2024/
├── src/                               # Código fuente principal del framework
│   ├── bd/                            # Gestor de base de datos SQLite (sqlite.py)
│   ├── chaotic_maps/                  # Mapas caóticos optimizados con Numba
│   ├── discretization/                # Funciones de transferencia y binarización
│   ├── diversity/                     # Métricas y cálculos de diversidad
│   ├── metaheuristics/                # Códigos y firmas de las 35+ metaheurísticas
│   ├── problem/                       # Definiciones de problemas y evaluación (Benchmark, SCP, USCP)
│   ├── solver/                        # Universal Solver, adapter y despachador de experimentos
│   └── util/                          # Utilidades (visualización, logs, configuración)
│
├── data/                              # Datos de entrada y bases de datos
│   ├── database/                      # Contiene resultados.db y shards/
│   ├── instances/                     # Instancias de problemas SCP/USCP
│   └── dim_cec2017.txt                # Datos de dimensiones para Benchmark
│
├── scripts/                           # Scripts de análisis, poblado y mantenimiento de BD
│   ├── poblarDB.py                    # Configura los experimentos en la base de datos
│   ├── analisis.py                    # Analiza y reporta resultados
│   ├── crearBD.py                     # Inicializa el esquema de base de datos
│   └── ...
│
├── tests/                             # Pruebas automatizadas (pytest)
│   ├── test_domain_registry.py
│   ├── test_scp_domain_chaotic.py
│   └── ...
│
├── docs/                              # Documentación técnica y manuales
│   └── explicaciones_manuales/        # Manuales de usuario y archivos LaTeX/PDF
│
├── outputs/                           # Archivos generados en tiempo de ejecución (Ignorados por Git)
│   ├── logs/                          # Logs de consola y logs de Slurm (SSH)
│   ├── plots/                         # Gráficos de trayectorias e historial de búsqueda
│   └── results/                       # Archivos CSV temporales de la ejecución
│
└── legacy/                            # Historial de copias de seguridad y scratchs
    ├── backups/                       # Código heredado e histórico
    └── scratch/                       # Scripts temporales de desarrollo
```

---

## Cómo Ejecutar el Proyecto

### 1. Requisitos Previos
Se recomienda el uso de un entorno virtual con Python 3.11+. Las dependencias principales incluyen `numpy`, `scipy`, `pandas`, `numba`, `joblib`, y `matplotlib`.
Para instalar las dependencias:
```bash
pip install -r requirements.txt
```

### 2. Inicializar la Base de Datos
Para crear el esquema e inyectar la configuración de los experimentos en la base de datos local:
```bash
python scripts/crearBD.py
python scripts/poblarDB.py
```

### 3. Ejecutar los Solvers
- **Ejecución local secuencial**:
  ```bash
  python main.py
  ```
- **Ejecución local en paralelo con Joblib**:
  ```bash
  python main_joblib.py --n-jobs <numero_de_cores>
  ```
- **Ejecución local distribuida en Windows** (Levanta varias consolas a la vez):
  ```bash
  python levantarCMD.py
  ```
- **Ejecución en Clúster HPC con Slurm**:
  ```bash
  sbatch run_main.sh
  # o usando Joblib con shards para evitar concurrencia en NFS
  sbatch run_main_joblib.sh
  ```

---

## Pruebas y Validación

El proyecto cuenta con una suite completa de pruebas unitarias y de comparación A/B (legacy vs universal):
```bash
python -m pytest tests/
```
