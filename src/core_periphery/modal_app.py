"""
Modal wrapper for the rho-grid sweep. Runs each algorithm in its own
container, in parallel, then pulls the resulting PNGs back locally.

Usage:
    modal run modal_app.py
"""

import modal

app = modal.App("attention-bandits-rho-grid")

# ---------------------------------------------------------------------
# Image: numpy/numba/matplotlib, plus the project's own source files
# mounted in. Adjust the local paths below to wherever your files
# actually live on disk.
# ---------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "numba", "matplotlib", "scipy")
    .add_local_python_source("environment", "greedy", "ucb", "rho_grid")
)


@app.function(image=image, cpu=4.0, timeout=3600)
def run_one_algorithm(algo_name: str, grid_size: int, n_rho: int, n_mc: int) -> dict:
    """Runs plot_rho_grid for a single algorithm inside its own
    container, returns ALL THREE output files as bytes (png, npy,
    csv) -- Modal containers are ephemeral, so anything not explicitly
    read back and returned here is lost when the container tears
    down, no matter what plot_rho_grid itself wrote to disk."""
    from rho_grid import plot_rho_grid, ALGORITHMS

    base = f"/tmp/rho_grid_{algo_name}"
    plot_rho_grid(
        grid_size=grid_size,
        n_rho=n_rho,
        n_mc=n_mc,
        algo_fn=ALGORITHMS[algo_name],
        algo_name=algo_name,
        out_path=base + ".png",
    )
    result = {}
    for ext in ("png", "npy", "csv"):
        with open(f"{base}.{ext}", "rb") as f:
            result[ext] = f.read()
    return result


@app.local_entrypoint()
def main(grid_size: int = 5, n_rho: int = 15, n_mc: int = 100):
    from rho_grid import ALGORITHMS
    import os

    algo_names = list(ALGORITHMS.keys())
    print(f"Dispatching {len(algo_names)} algorithms to Modal, in parallel...")

    os.makedirs("modal_outputs", exist_ok=True)
    for algo_name, result in zip(
        algo_names,
        run_one_algorithm.map(
            algo_names,
            kwargs={"grid_size": grid_size, "n_rho": n_rho, "n_mc": n_mc},
        ),
    ):
        for ext, data in result.items():
            out_path = f"modal_outputs/rho_grid_{algo_name}.{ext}"
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"saved {out_path}")

    print("All algorithms complete.")