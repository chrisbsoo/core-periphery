"""
Greedy-family algorithms: pure greedy, ETC (block exploration), BTC
(alternating / balanced exploration). All three share the same "simple
state" shape (n1, sum1, n2, sum2) -- no history beyond running counts
and sums is needed -- so they share one Monte Carlo runner shape.

See greedy_analysis.tex (Section: ETC Analysis) for the proven regret
characterization of ETC and BTC in the Contested Region.
"""

import numpy as np
from numba import njit, prange
from environment import func_target


@njit(cache=True)
def pure_greedy_decide(n1, sum1, n2, sum2, t):
    """No explicit exploration phase at all: play each arm once to
    initialize, then always exploit the current higher empirical mean."""
    if n1 == 0:
        return 1
    if n2 == 0:
        return 2
    m1 = sum1 / n1
    m2 = sum2 / n2
    return 1 if m1 >= m2 else 2


@njit(cache=True)
def etc_decide(n1, sum1, n2, sum2, t, T, c0=2.0):
    """Block exploration: arm 1 exclusively for the first m/2 rounds,
    arm 2 exclusively for the next m/2, then commit permanently to the
    higher empirical mean. m = ceil(c0 * T^(2/3))."""
    m = int(np.ceil(c0 * T ** (2.0 / 3.0)))
    if m % 2 == 1:
        m += 1
    half = m // 2
    if t <= half:
        return 1
    if t <= m:
        return 2
    m1 = sum1 / n1
    m2 = sum2 / n2
    return 1 if m1 >= m2 else 2


@njit(cache=True)
def btc_decide(n1, sum1, n2, sum2, t, T, c0=2.0):
    """Alternating exploration: always play whichever arm has fewer
    pulls so far, for the first m rounds, then commit permanently to
    the higher empirical mean. m = ceil(c0 * T^(2/3))."""
    m = int(np.ceil(c0 * T ** (2.0 / 3.0)))
    if t <= m:
        return 1 if n1 <= n2 else 2
    m1 = sum1 / n1
    m2 = sum2 / n2
    return 1 if m1 >= m2 else 2


@njit(cache=True, parallel=True)
def run_pure_greedy_monte_carlo(n_mc, T, L1, U1, L2, U2, a, b, sigma, base_seed):
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        sum1, sum2 = 0.0, 0.0
        total = 0.0
        for t in range(1, T + 1):
            arm = pure_greedy_decide(n1, sum1, n2, sum2, t)
            if arm == 1:
                r = np.random.normal(func_target(n1, t, L1, U1, a, b), sigma)
                sum1 += r
                n1 += 1
            else:
                r = np.random.normal(func_target(n2, t, L2, U2, a, b), sigma)
                sum2 += r
                n2 += 1
            total += r
        totals[i] = total
    return totals


@njit(cache=True, parallel=True)
def run_etc_monte_carlo(n_mc, T, L1, U1, L2, U2, a, b, sigma, base_seed, c0=2.0):
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        sum1, sum2 = 0.0, 0.0
        total = 0.0
        for t in range(1, T + 1):
            arm = etc_decide(n1, sum1, n2, sum2, t, T, c0)
            if arm == 1:
                r = np.random.normal(func_target(n1, t, L1, U1, a, b), sigma)
                sum1 += r
                n1 += 1
            else:
                r = np.random.normal(func_target(n2, t, L2, U2, a, b), sigma)
                sum2 += r
                n2 += 1
            total += r
        totals[i] = total
    return totals


@njit(cache=True, parallel=True)
def run_btc_monte_carlo(n_mc, T, L1, U1, L2, U2, a, b, sigma, base_seed, c0=2.0):
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        sum1, sum2 = 0.0, 0.0
        total = 0.0
        for t in range(1, T + 1):
            arm = btc_decide(n1, sum1, n2, sum2, t, T, c0)
            if arm == 1:
                r = np.random.normal(func_target(n1, t, L1, U1, a, b), sigma)
                sum1 += r
                n1 += 1
            else:
                r = np.random.normal(func_target(n2, t, L2, U2, a, b), sigma)
                sum2 += r
                n2 += 1
            total += r
        totals[i] = total
    return totals