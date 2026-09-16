
import numpy as np
import matplotlib.pyplot as plt
import argparse
import warnings

warnings.filterwarnings('ignore', message='.*All-NaN.*')
warnings.filterwarnings('ignore', message='.*Mean of empty slice.*')


def summarize(npy_path, agg='mean'):
    """Loads a (grid_size, grid_size, n_rho) array (NaN for out-of-scope
    cells) and collapses the last axis per the chosen aggregate."""
    arr = np.load(npy_path)
    grid_size = arr.shape[0]

    if agg == 'mean':
        summary = np.nanmean(arr, axis=2)
    elif agg == 'max':
        summary = np.nanmax(arr, axis=2)
    elif agg == 'min':
        summary = np.nanmin(arr, axis=2)
    elif agg == 'std':
        summary = np.nanstd(arr, axis=2)
    elif agg == 'range':
        summary = np.nanmax(arr, axis=2) - np.nanmin(arr, axis=2)
    else:
        raise ValueError(f"unknown agg '{agg}', use mean/max/min/std/range")

    return summary, grid_size


def plot_summary_grid(npy_path, agg='mean', algo_name=None,
                       alpha_range=(0.2, 1.0), out_path=None):
    summary, grid_size = summarize(npy_path, agg)
    algo_name = algo_name or npy_path.rsplit('/', 1)[-1].replace('.npy', '')
    alphas = np.linspace(*alpha_range, grid_size)
    betas = np.linspace(*alpha_range, grid_size)

    vmin = np.nanpercentile(summary, 5)
    vmax = np.nanpercentile(summary, 95)
    cmap = plt.cm.RdYlGn_r

    fig, ax = plt.subplots(figsize=(grid_size * 0.55 + 2.2, grid_size * 0.55 + 1.5),
                            constrained_layout=True)
    # origin='lower' so alpha increases upward, matching rho_grid.py's convention
    im = ax.imshow(summary, cmap=cmap, vmin=vmin, vmax=vmax,
                    origin='lower', aspect='equal')

    ax.set_xticks(range(grid_size))
    ax.set_xticklabels([f'{b:.2f}' for b in betas], rotation=90, fontsize=8)
    ax.set_yticks(range(grid_size))
    ax.set_yticklabels([f'{a:.2f}' for a in alphas], fontsize=8)
    ax.set_xlabel(r'$\beta \rightarrow$', fontsize=13)
    ax.set_ylabel(r'$\alpha \rightarrow$', fontsize=13)

    # grey out the alpha<beta (out-of-scope) cells explicitly
    for i in range(grid_size):
        for j in range(grid_size):
            if np.isnan(summary[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='#e8e8e8'))

    cb = fig.colorbar(im, ax=ax)
    cb.set_label(f'{agg} regret exponent (across $\\rho$)', fontsize=11, rotation=270, labelpad=18)

    fig.suptitle(f'{algo_name}: {agg} exponent per $(\\alpha,\\beta)$ cell', fontsize=13, fontweight='bold')

    out_path = out_path or f'modal_outputs/summary/{algo_name}_summary_{agg}.png'
    plt.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f'saved {out_path}')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='folder containing .npy files saved by rho_grid.py')
    parser.add_argument('output_dir', help='folder to save the summary heatmaps into')
    parser.add_argument('--agg', default='mean', choices=['mean', 'max', 'min', 'std', 'range', 'all'],
                         help='use "all" to produce mean AND max for every file')
    args = parser.parse_args()

    import os, glob

    npy_files = sorted(glob.glob(os.path.join(args.input_dir, '*.npy')))
    if not npy_files:
        print(f'no .npy files found in {args.input_dir}')
        raise SystemExit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    aggs = ['mean', 'max'] if args.agg == 'all' else [args.agg]

    print(f'found {len(npy_files)} .npy files, producing {len(aggs)} heatmap(s) each')
    for npy_path in npy_files:
        algo_name = os.path.basename(npy_path).replace('rho_grid_', '').replace('.npy', '')
        for agg in aggs:
            out_path = os.path.join(args.output_dir, f'{algo_name}_{agg}.png')
            plot_summary_grid(npy_path, agg=agg, algo_name=algo_name, out_path=out_path)

    print('done')