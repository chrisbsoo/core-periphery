"""
Two-armed floor/ceiling environment, exact K=2 oracle, the closed-form
alpha>=beta regret benchmark, and shared model utilities.

target_i(n, t) = L_i + (U_i - L_i) / (1 + exp(-(n**a - t**b)))

Since someone is always available (no sleeping/availability constraint),
exactly one arm is pulled every round: n1 + n2 = (rounds elapsed so far).
This lets the K=2 oracle collapse state (n1,n2,t) to (t,n2) alone --
O(T^2).

a, b are shared across both arms throughout (the theory -- No-Switching,
Recoverability, the regret benchmark -- assumes shared exponents; there
is no per-arm a,b in this project).
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def S(x):
    return 1.0 / (1.0 + np.exp(-x))


@njit(cache=True)
def func_target(n, t, L, U, a, b):
    """The TRUE mean of one arm, given its own pull count n, global round
    t, and its own (L, U), with a, b shared across arms. Never visible
    to a learner directly."""
    n_term = n ** a if n > 0 else 0.0
    t_term = t ** b if t > 0 else 0.0
    return L + (U - L) * S(n_term - t_term)


@njit(cache=True)
def precompute_AB(a, b, T):
    """A = sum_t (1-w_t), B = sum_t w_t, where w_t = S((t-1)^a - t^b).
    F_i = A*L_i + B*U_i exactly (Section: Regret Benchmark). Used both
    by the closed-form benchmark below and by GLR-UCB's own linear
    model of F_i."""
    A, B = 0.0, 0.0
    for t in range(1, T + 1):
        w = S((t - 1) ** a - t ** b)
        A += 1.0 - w
        B += w
    return A, B


@njit(cache=True)
def compute_oracle_table_k2(L1, U1, L2, U2, a, b, T):
    """Exact O(T^2) DP oracle. Valid for ANY a,b (including alpha<beta,
    where Theorem no-switching does not apply and no closed form
    exists). Kept as the fallback for that regime, and as a standalone
    correctness check against the closed-form benchmark for alpha>=beta."""
    V_next = np.zeros(T + 2, dtype=np.float64)
    action_table = np.ones((T + 1, T + 1), dtype=np.int8)
    V1_at_start = 0.0

    for t in range(T, 0, -1):
        V_cur = np.zeros(T + 2, dtype=np.float64)
        for n2 in range(0, t):
            n1_before = (t - 1) - n2
            val1 = func_target(n1_before, t, L1, U1, a, b) + V_next[n2]
            val2 = func_target(n2, t, L2, U2, a, b) + V_next[min(n2 + 1, T + 1)]
            if val1 >= val2:
                action_table[t, n2] = 1
                V_cur[n2] = val1
            else:
                action_table[t, n2] = 2
                V_cur[n2] = val2
        V_next = V_cur
        if t == 1:
            V1_at_start = V_cur[0]

    return action_table, V1_at_start


@njit(cache=True)
def closed_form_benchmark(L1, U1, L2, U2, a, b, T):
    """max(F1,F2) via Theorem no-switching, valid for alpha>=beta only.
    O(T) instead of O(T^2): this is the speedup the DP oracle no longer
    needs to be re-run for every heatmap cell in that regime."""
    A, B = precompute_AB(a, b, T)
    F1 = A * L1 + B * U1
    F2 = A * L2 + B * U2
    return F1 if F1 >= F2 else F2


@njit(cache=True)
def regret_benchmark(L1, U1, L2, U2, a, b, T):
    """Single entry point for the oracle value: uses the closed form
    when alpha>=beta (Theorem no-switching), falls back to the exact
    DP oracle otherwise (alpha<beta, where no closed form is proven).
    This is the function every downstream regret computation should
    call -- never call compute_oracle_table_k2 directly unless the DP
    action table itself is needed (e.g. simulate_oracle_realized)."""
    if a >= b:
        return closed_form_benchmark(L1, U1, L2, U2, a, b, T)
    else:
        _, V1 = compute_oracle_table_k2(L1, U1, L2, U2, a, b, T)
        return V1


@njit(cache=True, parallel=True)
def simulate_oracle_realized(n_mc, T, L1, U1, L2, U2, a, b, sigma, base_seed, action_table):
    """Sanity-check utility: forward-simulates the oracle's OWN action
    table using realized noisy rewards, to validate the DP oracle's
    predicted value against Monte Carlo. Requires the DP action table
    (compute_oracle_table_k2), so this only applies to the alpha<beta
    fallback path or as a direct correctness check."""
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        total = 0.0
        for t in range(1, T + 1):
            a_star = action_table[t, n2]
            if a_star == 1:
                mu = func_target(n1, t, L1, U1, a, b)
                total += np.random.normal(mu, sigma)
                n1 += 1
            else:
                mu = func_target(n2, t, L2, U2, a, b)
                total += np.random.normal(mu, sigma)
                n2 += 1
        totals[i] = total
    return totals


@njit(cache=True)
def rho(L1, U1, L2, U2):
    """Dominance-crossover parameter (Definition: dominance-crossover
    parameter). rho>0 Dominance, rho<0 Crossover, rho=0 boundary.
    Returns 0.0 for the fully Degenerate case (both x,y zero) as a
    safe sentinel rather than raising."""
    x = L1 - L2
    y = U1 - U2
    denom = x * x + y * y
    if denom == 0.0:
        return 0.0
    return 2.0 * x * y / denom