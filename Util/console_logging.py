# ===== console_logging.py =====
from colorama import Fore, init
init(autoreset=True)

def print_initial(tag: str, best_initial: float):
    print(f"{tag} - Initial Best Fitness: {best_initial:.2e}")
    print("-" * 102)

def print_iteration(it: int, max_it: int, best: float, opt: float,
                    dt: float, xpt: float, xpl: float, div_t: float,
                    print_every_quarter: bool = True,
                    progress_mode: str = 'iter',
                    fe: int | None = None,
                    max_fe: int | None = None):
    mode = (progress_mode or 'iter').lower()

    if mode == 'fe' and max_fe is not None and fe is not None:
        if max_fe <= 0:
            should_print = True
        else:
            step = max(1, max_fe // 4) if print_every_quarter else 1
            should_print = (fe % step == 0) or (fe >= max_fe)
    else:
        if max_it <= 0:
            should_print = True
        else:
            step = max(1, max_it // 4) if print_every_quarter else 1
            should_print = (it % step == 0) or (it == max_it)

    if should_print:
        if mode == 'fe' and fe is not None and max_fe is not None:
            display_fe = min(fe, max_fe)  # clamp para que no se vea que se pasó de 5000 visualmente
            progress_label = f"FE: {display_fe:<6}/{max_fe:<6}"
        else:
            progress_label = f"Iteration: {it:<4}"
            
        msg = (
            f"{progress_label} | "
            f"Best Fitness: {best:>7.2e} | "
            f"Optimum: {opt:>9.2e} | "
            f"Time (s): {dt:>4.3f} | "
            f"XPT: {xpt:>6.2f} | "
            f"XPL: {xpl:>6.2f} | "
            f"DIV: {div_t:>5.2f}"
        )
        print(msg)

def print_final(best_final: float, t_start: float, t_end: float):
    print("-" * 102)
    print(f"{Fore.GREEN}Execution time (s): {(t_end - t_start):.2f}")
    print(f"{Fore.GREEN}Best Fitness: {best_final:.2e}")
    print("-" * 102)
