# ===== csv_writer.py =====
import os
from typing import List, TextIO

def open_csv(path: str, header: List[str], mkdirs: bool = True) -> TextIO:
    if mkdirs:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = open(path, "w")
    fh.write(",".join(header) + "\n")
    return fh

def write_csv_row(fh: TextIO, values: List[object]) -> None:
    fh.write(",".join(str(v) for v in values) + "\n")

def close_csv(fh: TextIO) -> None:
    try:
        fh.close()
    except Exception:
        pass
