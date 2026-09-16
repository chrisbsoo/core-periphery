"""
Rho-gradient (alpha,beta) grid: for each (alpha,beta) cell, a smooth
horizontal strip sweeps the dominance-crossover parameter rho from -1
(Crossover) to +1 (Dominance), colored by that algorithm's regret at
each point. Green = low regret, red = high regret.

Requires environment.py, greedy.py, ucb.py on the path.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import time
sys.path.insert(0, '/home/claude')  # adjust to wherever environment.py/greedy.py/ucb.py live

from environment import regret_benchmark
from ucb import run_ssw_ucb_monte_carlo, run_glr_ucb_monte_carlo, run_ucb1_monte_carlo, \
    run_rawucb_monte_carlo, run_sw_ucb_monte_carlo
from greedy import run_pure_greedy_monte_carlo, run_etc_monte_carlo, run_btc_monte_carlo


def rho_to_LU(theta_deg, r=0.15, Lm=0.35, Um=0.65):
    """theta in [-45,45] sweeps rho=sin(2*theta) from -1 to 1, at fixed
    separation magnitude r, around midpoint (Lm,Um)."""
    theta = np.radians(theta_deg)
    x, y = r * np.cos(theta), r * np.sin(theta)
    L1, L2 = Lm + x / 2, Lm - x / 2
    U1, U2 = Um + y / 2, Um - y / 2
    return L1, U1, L2, U2


def compute_cell_gradient(alpha, beta, sigma, n_rho, n_mc, algo_fn,
                           T_values=(500, 1000, 2000, 4000, 8000), seed=42):
    """For one (alpha,beta), sweep rho from -1 to 1, return the FITTED
    REGRET EXPONENT at each point -- not raw regret at a single T.
    Raw regret at one T is not comparable across (alpha,beta) cells,
    since different points have different natural scales (e.g. T^1 vs
    T^0.7 at T=2000 differ by ~9x before any algorithm-specific
    constant even enters) -- fitting the exponent is the metric
    actually used everywhere else in this project for exactly this
    reason."""
    thetas = np.linspace(-45, 45, n_rho)
    exponents = np.zeros(n_rho)
    log_T = np.log(np.array(T_values, dtype=np.float64))
    for i, th in enumerate(thetas):
        L1, U1, L2, U2 = rho_to_LU(th)
        log_regret = np.zeros(len(T_values))
        for k, T in enumerate(T_values):
            oracle = regret_benchmark(L1, U1, L2, U2, alpha, beta, T)
            totals = algo_fn(n_mc, T, L1, U1, L2, U2, alpha, beta, sigma, seed)
            regret = max(oracle - totals.mean(), 1e-6)  # floor to avoid log(0)
            log_regret[k] = np.log(regret)
        slope, _ = np.polyfit(log_T, log_regret, 1)
        exponents[i] = slope
    return exponents


def plot_rho_grid(grid_size=5, sigma=0.1, n_rho=15, n_mc=100,
                   T_values=(500, 1000, 2000, 4000, 8000),
                   algo_fn=run_ssw_ucb_monte_carlo, algo_name='SSW-UCB',
                   alpha_range=(0.2, 1.0), out_path=None):
    alphas = np.linspace(*alpha_range, grid_size)
    betas = np.linspace(*alpha_range, grid_size)

    n_cells_total = sum(1 for a in alphas for b in betas if a >= b)
    n_evals = n_cells_total * n_rho * len(T_values)
    print(f'{algo_name}: {n_cells_total} cells to compute (grid_size={grid_size}, '
          f'n_rho={n_rho}, n_mc={n_mc}, {len(T_values)} T-values '
          f'-> {n_evals} regret evaluations for exponent fitting)')

    all_exponents = np.full((grid_size, grid_size, n_rho), np.nan)
    t_start = time.time()
    cells_done = 0
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            if a >= b:  # scope: alpha >= beta only
                all_exponents[i, j] = compute_cell_gradient(a, b, sigma, n_rho, n_mc, algo_fn, T_values)
                cells_done += 1
                elapsed = time.time() - t_start
                rate = elapsed / cells_done
                remaining = rate * (n_cells_total - cells_done)
                print(f'  [{cells_done}/{n_cells_total}] '
                      f'(a={a:.2f}, b={b:.2f})  '
                      f'elapsed={elapsed:6.1f}s  eta={remaining:6.1f}s', flush=True)
    print(f'{algo_name}: done in {time.time() - t_start:.1f}s total')

    # ---- SAVE THE RAW NUMBERS, ALWAYS, BEFORE ANY PLOTTING ----
    # .npy: exact array, easy to reload for re-plotting without re-running MC
    # .csv: long format (alpha, beta, rho, exponent), easy to inspect/analyze
    base = out_path.rsplit('.', 1)[0] if out_path else f'rho_grid_{algo_name.lower().replace("-", "_")}'
    npy_path = base + '.npy'
    csv_path = base + '.csv'
    np.save(npy_path, all_exponents)

    thetas = np.linspace(-45, 45, n_rho)
    rhos = np.sin(2 * np.radians(thetas))
    with open(csv_path, 'w') as f:
        f.write('alpha,beta,rho,fitted_exponent\n')
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                if a >= b:
                    for k, rv in enumerate(rhos):
                        f.write(f'{a:.4f},{b:.4f},{rv:.4f},{all_exponents[i,j,k]:.6f}\n')
    print(f'saved numbers to {npy_path} and {csv_path}')

    # ---- PLOTTING ----
    vmin = np.nanpercentile(all_exponents, 5)
    vmax = np.nanpercentile(all_exponents, 95)

    fig = plt.figure(figsize=(grid_size * 1.9 + 1.2, grid_size * 1.9 + 1.0), constrained_layout=True)
    gs = fig.add_gridspec(grid_size, grid_size + 1,
                           width_ratios=[1] * grid_size + [0.08],
                           wspace=0.15, hspace=0.15)
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i in range(grid_size):
        for j in range(grid_size):
            ax = fig.add_subplot(gs[grid_size - 1 - i, j])  # alpha increases upward
            strip = all_exponents[i, j][np.newaxis, :]
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor('#999999')
                spine.set_linewidth(0.8)
            if np.isnan(strip).all():
                ax.set_facecolor('#e8e8e8')
                continue
            ax.imshow(strip, aspect='auto', cmap=cmap, norm=norm,
                      extent=[-1, 1, 0, 1], interpolation='bilinear')
            if j == 0:
                ax.set_ylabel(f'{alphas[i]:.2f}', fontsize=11, rotation=0,
                               ha='right', va='center', labelpad=8)
            if i == 0:
                ax.set_xlabel(f'{betas[j]:.2f}', fontsize=11, labelpad=8)

    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label('fitted regret exponent  (Regret $= \\Theta(T^{\\,\\mathrm{exponent}})$)',
                  fontsize=11, rotation=270, labelpad=20)
    cb.ax.tick_params(labelsize=10)

    subtitle = (r'each cell sweeps $\rho=-1$ (Crossover) $\rightarrow$ $\rho=+1$ (Dominance)'
                f'  |  fit over T = {min(T_values)}\u2013{max(T_values)} ({len(T_values)} pts)')
    fig.suptitle(f'{algo_name}: fitted regret exponent vs $\\rho$ per $(\\alpha,\\beta)$ cell\n'
                 f'{subtitle}',
                 fontsize=14, fontweight='bold')
    fig.supxlabel(r'$\beta \rightarrow$', fontsize=15)
    fig.supylabel(r'$\alpha \rightarrow$', fontsize=15)

    out_path = out_path or (base + '.png')
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f'saved plot to {out_path}')


ALGORITHMS = {
    'pure_greedy': run_pure_greedy_monte_carlo,
    'etc':         run_etc_monte_carlo,
    'btc':         run_btc_monte_carlo,
    'ucb1':        run_ucb1_monte_carlo,
    'rawucb':      run_rawucb_monte_carlo,
    'sw_ucb':      run_sw_ucb_monte_carlo,
    'ssw_ucb':     run_ssw_ucb_monte_carlo,
    'glr_ucb':     run_glr_ucb_monte_carlo,
}

if __name__ == '__main__':
    plot_rho_grid(grid_size=5, n_rho=15, n_mc=100,
                   algo_fn=run_ssw_ucb_monte_carlo, algo_name='SSW-UCB')