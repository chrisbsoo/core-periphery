"""
Two-armed floor/ceiling environment, exact K=2 oracle, and a Monte Carlo
runner for evaluating algorithms from algorithm.py.

target_i(n, t) = L_i + (U_i - L_i) / (1 + exp(-(n**a_i - t**b_i)))

Since someone is always available (no sleeping/availability constraint),
exactly one arm is pulled every round: n1 + n2 = (rounds elapsed so far).
This lets the K=2 oracle collapse state (n1,n2,t) to (t,n2) alone --
O(T^2), same complexity as the original core-periphery oracle.
"""

import numpy as np
from numba import njit, prange


@njit(cache=True)
def func_target(n, t, L, U, a, b):
    """The TRUE mean of one arm, given its own pull count n, global round
    t, and its own (L, U, a, b). Never visible to a learner directly."""
    n_term = n ** a if n > 0 else 0.0
    t_term = t ** b if t > 0 else 0.0
    return L + (U - L) / (1.0 + np.exp(-(n_term - t_term)))


@njit(cache=True)
def compute_oracle_table_k2(L1, U1, a, b, L2, U2, T):
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


@njit(cache=True, parallel=True)
def simulate_oracle_realized(n_mc, T, L1, U1, a1, b1, L2, U2, a2, b2, sigma, base_seed, action_table):
    """Sanity-check utility: forward-simulates the oracle's OWN action
    table using realized noisy rewards, to validate the DP oracle's
    predicted value against Monte Carlo (same validation style used
    throughout this project)."""
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        total = 0.0
        for t in range(1, T + 1):
            a_star = action_table[t, n2]
            if a_star == 1:
                mu = func_target(n1, t, L1, U1, a1, b1)
                total += np.random.normal(mu, sigma)
                n1 += 1
            else:
                mu = func_target(n2, t, L2, U2, a2, b2)
                total += np.random.normal(mu, sigma)
                n2 += 1
        totals[i] = total
    return totals