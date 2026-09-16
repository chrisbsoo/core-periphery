"""
Pooled gradient grid: spatially pools a saved (grid_size, grid_size,
n_rho) array down to fewer, bigger (alpha,beta) cells -- but unlike
summary_grid.py, each pooled cell still renders as a full rho-gradient
strip (rho averaged across the pooled block, not collapsed away).

Usage:
    python3 pooled_gradient_grid.py rho_grid_glr_ucb.npy --pool 4 --name GLR-UCB
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import warnings

warnings.filterwarnings('ignore', message='.*All-NaN.*')
warnings.filterwarnings('ignore', message='.*Mean of empty slice.*')


def pool_gradient_array(arr, alphas, betas, pool_size):
    """Pools the first two (spatial) axes of a (grid_size, grid_size,
    n_rho) array via nanmean, leaving the rho axis (last) untouched."""
    grid_size = arr.shape[0]
    new_size = grid_size // pool_size
    trimmed = new_size * pool_size
    if trimmed != grid_size:
        arr = arr[:trimmed, :trimmed, :]
        alphas = alphas[:trimmed]
        betas = betas[:trimmed]

    n_rho = arr.shape[2]
    pooled = np.full((new_size, new_size, n_rho), np.nan)
    for i in range(new_size):
        for j in range(new_size):
            block = arr[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size, :]
            if not np.all(np.isnan(block)):
                pooled[i, j, :] = np.nanmean(block, axis=(0, 1))

    pooled_alphas = np.array([alphas[i*pool_size:(i+1)*pool_size].mean() for i in range(new_size)])
    pooled_betas = np.array([betas[j*pool_size:(j+1)*pool_size].mean() for j in range(new_size)])
    return pooled, pooled_alphas, pooled_betas


def plot_pooled_gradient_grid(npy_path, pool_size=4, algo_name=None,
                               alpha_range=(0.2, 1.0), out_path=None):
    arr = np.load(npy_path)
    grid_size = arr.shape[0]
    algo_name = algo_name or npy_path.rsplit('/', 1)[-1].replace('rho_grid_', '').replace('.npy', '')

    alphas = np.linspace(*alpha_range, grid_size)
    betas = np.linspace(*alpha_range, grid_size)

    arr, alphas, betas = pool_gradient_array(arr, alphas, betas, pool_size)
    new_size = arr.shape[0]

    vmin = np.nanpercentile(arr, 5)
    vmax = np.nanpercentile(arr, 95)
    cmap = plt.cm.RdYlGn_r
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(new_size * 1.9 + 1.2, new_size * 1.9 + 1.0), constrained_layout=True)
    gs = fig.add_gridspec(new_size, new_size + 1,
                           width_ratios=[1] * new_size + [0.08],
                           wspace=0.15, hspace=0.15)

    for i in range(new_size):
        for j in range(new_size):
            ax = fig.add_subplot(gs[new_size - 1 - i, j])  # alpha increases upward
            strip = arr[i, j][np.newaxis, :]
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
    cb.set_label('fitted regret exponent', fontsize=11, rotation=270, labelpad=20)
    cb.ax.tick_params(labelsize=10)

    subtitle = f'{pool_size}\u00d7{pool_size} pooled ({grid_size}\u00d7{grid_size} \u2192 {new_size}\u00d7{new_size})'
    fig.suptitle(f'{algo_name}: exponent vs $\\rho$\n{subtitle}',
                 fontsize=13, fontweight='bold')
    fig.supxlabel(r'$\beta \rightarrow$', fontsize=14)
    fig.supylabel(r'$\alpha \rightarrow$', fontsize=14)

    out_path = out_path or f'{algo_name}_pooled{pool_size}.png'
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f'saved {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='folder containing .npy files saved by rho_grid.py')
    parser.add_argument('output_dir', help='folder to save the pooled gradient grids into')
    parser.add_argument('--pool', type=int, default=4, help='block size to pool, e.g. 4 for 16x16->4x4')
    args = parser.parse_args()

    import os, glob

    npy_files = sorted(glob.glob(os.path.join(args.input_dir, '*.npy')))
    if not npy_files:
        print(f'no .npy files found in {args.input_dir}')
        raise SystemExit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'found {len(npy_files)} .npy files, pooling {args.pool}x{args.pool}')
    for npy_path in npy_files:
        algo_name = os.path.basename(npy_path).replace('rho_grid_', '').replace('.npy', '')
        out_path = os.path.join(args.output_dir, f'{algo_name}_pooled{args.pool}.png')
        plot_pooled_gradient_grid(npy_path, pool_size=args.pool, algo_name=algo_name, out_path=out_path)

    print('done')