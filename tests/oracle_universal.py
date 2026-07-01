import sys
import os
import json
import numpy as np
import random as _random

# Agregar paths del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from tests.test_ab_ben import _reset_ccmgo_globals, _build_userData, run_universal_ben

ALL_MHS = [
    'GWO', 'PSO', 'WOA', 'SCA', 'FOX', 'NO', 'PSA', 'AOA', 'EOO',
    'EBWOA', 'RSA', 'LOA', 'QSO', 'WOM',
    'HBA', 'GOA', 'FLO', 'POA', 'TDO', 'SBOA', 'SHO', 'PGA',
    'SSO', 'DLO', 'DOA', 'EHO', 'DRA', 'WO', 'DHOA', 'ALA', 'CCMGO',
    'APO', 'CLO', 'TJO', 'GOAT', 'HLOA',
]

ORACLE_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "oracle_universal_results.json"))

def generate_results():
    results = {}
    for mh in ALL_MHS:
        try:
            conv, nfe = run_universal_ben(
                seed=42,
                mh_name=mh,
                function='F1',
                dim=10,
                lb=-100,
                ub=100,
                pop_size=10,
                max_iter=15
            )
            results[mh] = {
                'best_fitness': float(conv[-1]),
                'nfe': int(nfe[-1]),
                'trajectory': [float(v) for v in conv]
            }
        except Exception as e:
            results[mh] = {
                'error': f"{type(e).__name__}: {e}"
            }
    return results

def main():
    action = sys.argv[1] if len(sys.argv) > 1 else 'run'
    
    if action == 'save':
        print("Generando resultados basales (ANTES)...")
        results = generate_results()
        with open(ORACLE_FILE, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Resultados basales guardados en: {ORACLE_FILE}")
        return 0
        
    elif action == 'verify':
        print("Verificando resultados actuales contra el Oráculo (DESPUES)...")
        if not os.path.exists(ORACLE_FILE):
            print(f"Error: No se encontró el archivo del oráculo: {ORACLE_FILE}")
            print("Ejecuta primero: python tests/oracle_universal.py save")
            return 1
            
        with open(ORACLE_FILE, 'r', encoding='utf-8') as f:
            baseline = json.load(f)
            
        current = generate_results()
        
        mismatches = 0
        for mh in ALL_MHS:
            b_data = baseline.get(mh, {})
            c_data = current.get(mh, {})
            
            if 'error' in b_data or 'error' in c_data:
                if b_data.get('error') != c_data.get('error'):
                    print(f"[-] {mh}: Error mismatch. Baseline: {b_data.get('error')} | Current: {c_data.get('error')}")
                    mismatches += 1
                continue
                
            # Comparar fitness final
            b_fit = b_data['best_fitness']
            c_fit = c_data['best_fitness']
            
            # Comparar NFE final
            b_nfe = b_data['nfe']
            c_nfe = c_data['nfe']
            
            # Comparar trayectorias
            b_traj = np.array(b_data['trajectory'])
            c_traj = np.array(c_data['trajectory'])
            
            fit_match = np.allclose(b_traj, c_traj, atol=1e-12)
            nfe_match = (b_nfe == c_nfe)
            
            if fit_match and nfe_match:
                print(f"[+] {mh}: OK (Fitness y NFE idénticos)")
            else:
                print(f"[-] {mh}: DIVERGE!")
                if not fit_match:
                    print(f"    Fitness inicial: baseline={b_traj[0]:.6e} vs current={c_traj[0]:.6e}")
                    print(f"    Fitness final: baseline={b_fit:.6e} vs current={c_fit:.6e}")
                if not nfe_match:
                    print(f"    NFE: baseline={b_nfe} vs current={c_nfe}")
                mismatches += 1
                
        if mismatches == 0:
            print("\n>>> ¡ORÁCULO PASADO CON ÉXITO! Todos los resultados son 100% idénticos.")
            return 0
        else:
            print(f"\n>>> ¡FALLÓ EL ORÁCULO! {mismatches} metaheurísticas divergieron.")
            return 1
            
    else:
        print("Uso: python tests/oracle_universal.py [save|verify]")
        return 1

if __name__ == '__main__':
    sys.exit(main())
