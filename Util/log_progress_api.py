# ===== log_progress_api.py =====
# Backward-compatible helpers if you want a single import point.

from console_logging import print_initial, print_iteration, print_final

def log_progress(iter, maxIter, bestFitness, optimo, timeExecuted, XPT, XPL, div_t,
                 write_to_file: bool = False, csv_handle=None, csv_values: list = None):
    """
    Console-only by default. If write_to_file=True and csv_handle/csv_values provided,
    it also writes one CSV row.
    """
    # console
    print_iteration(
        it=iter, max_it=maxIter, best=bestFitness, opt=optimo,
        dt=timeExecuted, xpt=XPT, xpl=XPL, div_t=div_t
    )

    # optional CSV write
    if write_to_file and csv_handle is not None and csv_values is not None:
        csv_handle.write(",".join(str(v) for v in csv_values) + "\n")
