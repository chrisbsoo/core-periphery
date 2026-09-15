"""
Analysis pipeline: sweeps the (a,b) grid, fits an empirical regret
exponent for UCB1 at each cell (log-log slope of regret vs T against
the exact K=2 oracle), across four distinct [L,U]-relationship cases,
and produces heatmaps.

Case 1 -- identical:         [L1,U1] == [L2,U2]
Case 2 -- uniform_shift:     same width, one arm uniformly better
Case 3 -- crossover:         different widths, floors/ceilings cross
                              (which arm is better depends on attention level)
Case 4 -- strict_dominance:  L1 > U2, no real tradeoff

Pure local execution -- numba/prange only, no Modal.
"""

import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from environment import compute_oracle_table_k2
from algorithm import run_ucb1_monte_carlo, run_balanced_explore_monte_carlo,  run_rawucb_monte_carlo
from algorithm import run_glr_ucb_monte_carlo, run_saturation_aware_monte_carlo

SCRIPT_DIR = Path(__file__).parent


CASES = {
    "identical":        dict(L1=0.20, U1=0.75, L2=0.20, U2=0.75),
    "uniform_shift":    dict(L1=0.25, U1=0.75, L2=0.20, U2=0.70),
    "crossover":        dict(L1=0.20, U1=0.70, L2=0.30, U2=0.65),
    "strict_dominance": dict(L1=0.60, U1=0.90, L2=0.10, U2=0.50),
}


def _format_eta(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.2f}h"


def fit_regret_exponent(algo_name, L1, U1, a, b, L2, U2, sigma, n_mc, T_values, base_seed=42, verbose=True):
    log_T, log_regret = [], []
    for T in T_values:
        t0 = time.time()
        _, oracle_val = compute_oracle_table_k2(L1, U1, a, b, L2, U2, T)
        if algo_name == "UCB1":
            totals = run_ucb1_monte_carlo(n_mc, T, L1, U1, a, b, L2, U2, sigma, base_seed)
        elif algo_name == "RAWUCB":
            totals = run_rawucb_monte_carlo(n_mc, T, L1, U1, a, b, L2, U2, sigma, base_seed)
        elif algo_name == "GLRUCB":
            totals = run_glr_ucb_monte_carlo(n_mc, T, L1, U1, a, b, L2, U2, sigma, base_seed)
        elif algo_name == "SWUCB":
            totals = run_saturation_aware_monte_carlo(n_mc, T, L1, U1, a, b, L2, U2, sigma, base_seed)
        else:
            totals = run_balanced_explore_monte_carlo(n_mc, T, L1, U1, a, b, L2, U2, sigma, base_seed)
        regret = oracle_val - totals.mean()
        log_T.append(np.log(T))
        log_regret.append(np.log(max(regret, 1e-6)))

        if verbose:  # FIXED: now inside the loop, prints per-T instead of once
            print(f"      T={T:6d}  regret={regret:10.2f}  ({time.time()-t0:.2f}s)")

    slope, _ = np.polyfit(log_T, log_regret, 1)
    slope = max(slope, 0.0)  # negative is a noise artifact, not a real finding

    # reliability check: is the LARGEST-T regret big enough relative to
    # Monte Carlo noise to trust the fitted slope at all?
    last_regret = np.exp(log_regret[-1])
    noise_floor = sigma * np.sqrt(T_values[-1] / n_mc)
    reliable = last_regret > 2 * noise_floor

    return slope, reliable


def run_sweep(algo_name, case_name, case_params, a_grid, b_grid, sigma, n_mc, T_values, verbose_cells=False):
    L1, U1, L2, U2 = case_params["L1"], case_params["U1"], case_params["L2"], case_params["U2"]
    exponent_grid = np.full((len(a_grid), len(b_grid)), np.nan)
    reliable_grid = np.zeros((len(a_grid), len(b_grid)), dtype=bool)

    total_cells = sum(1 for a in a_grid for b in b_grid if a > 0 and b > 0)
    done_cells = 0
    sweep_start = time.time()

    print(f"  [{case_name}] {total_cells} cells to run (grid has {len(a_grid)}x{len(b_grid)} = "
          f"{len(a_grid)*len(b_grid)}, minus degenerate a<=0 or b<=0 edges)")

    for i, a in enumerate(a_grid):
        for j, b in enumerate(b_grid):
            if a <= 0 or b <= 0:
                continue

            cell_start = time.time()
            try:
                # FIXED: unpack the (slope, reliable) tuple properly
                slope, reliable = fit_regret_exponent(
                    algo_name, L1, U1, a, b, L2, U2, sigma, n_mc, T_values, verbose=verbose_cells
                )
                exponent_grid[i, j] = slope
                reliable_grid[i, j] = reliable
                status = f"exponent={slope:.3f} ({'reliable' if reliable else 'noise-dominated'})"
            except Exception as e:
                status = f"FAILED ({e})"

            done_cells += 1
            cell_time = time.time() - cell_start
            elapsed = time.time() - sweep_start
            avg_per_cell = elapsed / done_cells
            remaining = (total_cells - done_cells) * avg_per_cell

            print(f"  [{case_name}] cell {done_cells}/{total_cells} "
                  f"(a={a:.2f}, b={b:.2f}): {status}  "
                  f"[{cell_time:.1f}s this cell | elapsed {_format_eta(elapsed)} | "
                  f"ETA {_format_eta(remaining)}]")

    print(f"  [{case_name}] done in {_format_eta(time.time() - sweep_start)}")
    return exponent_grid, reliable_grid


def plot_heatmap(algo_name, exponent_grid, reliable_grid, a_grid, b_grid, case_name, save_path=None):
    if save_path is None:
        save_path = f"{SCRIPT_DIR}/result_png/{algo_name}/regret_exponent_{case_name}.png"
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        exponent_grid,
        origin="lower",
        extent=[b_grid[0], b_grid[-1], a_grid[0], a_grid[-1]],
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=0.4, vmax=1.1,
    )

    # mark unreliable (noise-dominated) cells with an X, same convention
    # used for statistical significance earlier in this project
    n_a, n_b = len(a_grid), len(b_grid)
    for i in range(n_a):
        for j in range(n_b):
            if a_grid[i] > 0 and b_grid[j] > 0 and not reliable_grid[i, j]:
                ax.scatter(b_grid[j], a_grid[i], marker="x", color="black", s=40, linewidths=1.5)

    ax.plot(b_grid, b_grid, "k--", linewidth=1.5, label="a = b (the boundary)")
    ax.set_xlabel("b (decay exponent)")
    ax.set_ylabel("a (rise exponent)")
    ax.set_title(f"{algo_name}empirical regret exponent -- case: {case_name}\n"
                  "lower/greener = more sublinear, ~1.0 = linear regret\n"
                  "X = regret too small relative to MC noise to trust the fit")
    ax.legend(loc="upper left")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("fitted regret exponent")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  saved {save_path}")
    plt.close(fig)

def save_summary_table(algo_name, case_name, exponent_grid, reliable_grid, a_grid, b_grid):
    rows = []
    for i, a in enumerate(a_grid):
        for j, b in enumerate(b_grid):
            if a <= 0 or b <= 0:
                continue
            rows.append((a, b, exponent_grid[i, j], reliable_grid[i, j], a - b))
    arr = np.array(rows, dtype=[('a', 'f8'), ('b', 'f8'), ('exponent', 'f8'),
                                  ('reliable', 'bool'), ('gap', 'f8')])
    np.savetxt(f"{SCRIPT_DIR}/result_csv/{algo_name}/summary_{case_name}.csv", arr,
               delimiter=',', header='a,b,exponent,reliable,a_minus_b',
               fmt=['%.2f', '%.2f', '%.4f', '%d', '%.2f'], comments='')

    reliable_mask = arr['reliable']
    diag_mask = np.isclose(arr['a'], arr['b'], atol=0.06)
    print(f"  [{case_name}] summary: mean exponent (a=b, reliable) = "
          f"{arr['exponent'][diag_mask & reliable_mask].mean():.3f}, "
          f"mean exponent (a>b, reliable) = "
          f"{arr['exponent'][(arr['gap']>0.06) & reliable_mask].mean():.3f}, "
          f"mean exponent (a<b, reliable) = "
          f"{arr['exponent'][(arr['gap']<-0.06) & reliable_mask].mean():.3f}")


def run_all_cases(algo_name, a_grid, b_grid, sigma=0.15, n_mc=100,
                   T_values=(500, 1000, 2000, 4000, 8000),
                   verbose_cells=False):
    results = {}
    n_cases = len(CASES)
    overall_start = time.time()

    for case_idx, (case_name, params) in enumerate(CASES.items(), start=1):
        print(f"\n=== Case {case_idx}/{n_cases}: {case_name} "
              f"[overall elapsed {_format_eta(time.time()-overall_start)}] ===")
        exponent_grid, reliable_grid = run_sweep(algo_name, case_name, params, a_grid, b_grid, sigma, n_mc, T_values, verbose_cells)
        results[case_name] = (exponent_grid, reliable_grid)
        save_summary_table(algo_name, case_name, exponent_grid, reliable_grid, a_grid, b_grid)
        plot_heatmap(algo_name, exponent_grid, reliable_grid, a_grid, b_grid, case_name)
        np.savez(f"{SCRIPT_DIR}/result_npz/{algo_name}/regret_exponent_{case_name}.npz",
                 exponent_grid=exponent_grid, reliable_grid=reliable_grid, a_grid=a_grid, b_grid=b_grid)

    print(f"\nAll {n_cases} cases done. Total time: {_format_eta(time.time()-overall_start)}")
    return results


if __name__ == "__main__":
    a_grid = np.linspace(0.1, 1.0, 5)
    b_grid = np.linspace(0.1, 1.0, 5)
    run_all_cases("SWUCB", a_grid, b_grid)