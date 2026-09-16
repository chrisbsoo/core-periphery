"""
UCB-family algorithms: UCB1, RAW-UCB, SW-UCB, SSW-UCB, GLR-UCB.

Grouped by the state shape each decision rule needs:
  - "simple" (n, sum):              UCB1
  - "windowed" (n, prefix history): RAW-UCB, SW-UCB, SSW-UCB
  - "GLR" (sufficient statistics):  GLR-UCB

See ucb_analysis.tex for the saturation-collapse mechanism motivating
SSW-UCB, and the linear-model motivation for GLR-UCB.
"""

import numpy as np
from numba import njit, prange
from environment import func_target, S, precompute_AB


# ---------------------------------------------------------------------
# UCB1 (simple state)
# ---------------------------------------------------------------------

@njit(cache=True)
def ucb1_decide(n1, sum1, n2, sum2, t, ucb_c=1.0):
    if n1 == 0:
        return 1
    if n2 == 0:
        return 2
    m1 = sum1 / n1
    m2 = sum2 / n2
    bonus1 = ucb_c * np.sqrt(2.0 * np.log(t + 2.0) / n1)
    bonus2 = ucb_c * np.sqrt(2.0 * np.log(t + 2.0) / n2)
    return 1 if (m1 + bonus1) >= (m2 + bonus2) else 2


@njit(cache=True, parallel=True)
def run_ucb1_monte_carlo(n_mc, T, L1, U1, L2, U2, a, b, sigma, base_seed, ucb_c=1.0):
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        sum1, sum2 = 0.0, 0.0
        total = 0.0
        for t in range(1, T + 1):
            arm = ucb1_decide(n1, sum1, n2, sum2, t, ucb_c)
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


# ---------------------------------------------------------------------
# RAW-UCB (windowed state: tests every window, adopts the most
# pessimistic index -- Seznec et al. 2020)
# ---------------------------------------------------------------------

@njit(cache=True)
def rawucb_decide(n1, prefix1, n2, prefix2, t, alpha=1.0, subgaussian=1.0):
    """EFF-RAW-UCB (Seznec et al. 2020): tests a GEOMETRIC sequence of
    window lengths (1,2,4,8,...) instead of every integer window. The
    naive version (every w=1..n) is O(n) per decision, O(T^2) per
    trial -- confirmed directly to scale exactly as T^2, making it by
    far the most expensive algorithm in this project at large T. This
    version is O(log n) per decision, O(T log T) per trial."""
    if n1 == 0:
        return 1
    if n2 == 0:
        return 2

    log_term = alpha * np.log(t + 2.0)

    best1 = 1e18
    w = 1
    while w <= n1:
        windowed_mean = (prefix1[n1] - prefix1[n1 - w]) / w
        bonus = np.sqrt(2.0 * subgaussian ** 2 * log_term / w)
        idx = windowed_mean + bonus
        if idx < best1:
            best1 = idx
        w *= 2
    if w // 2 != n1:  # always also test the full window
        windowed_mean = prefix1[n1] / n1
        bonus = np.sqrt(2.0 * subgaussian ** 2 * log_term / n1)
        idx = windowed_mean + bonus
        if idx < best1:
            best1 = idx

    best2 = 1e18
    w = 1
    while w <= n2:
        windowed_mean = (prefix2[n2] - prefix2[n2 - w]) / w
        bonus = np.sqrt(2.0 * subgaussian ** 2 * log_term / w)
        idx = windowed_mean + bonus
        if idx < best2:
            best2 = idx
        w *= 2
    if w // 2 != n2:
        windowed_mean = prefix2[n2] / n2
        bonus = np.sqrt(2.0 * subgaussian ** 2 * log_term / n2)
        idx = windowed_mean + bonus
        if idx < best2:
            best2 = idx

    return 1 if best1 >= best2 else 2


@njit(cache=True, parallel=True)
def run_rawucb_monte_carlo(n_mc, T, L1, U1, L2, U2, a, b, sigma, base_seed,
                             alpha=1.0, subgaussian=1.0):
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        prefix1 = np.zeros(T + 1)
        prefix2 = np.zeros(T + 1)
        total = 0.0
        for t in range(1, T + 1):
            arm = rawucb_decide(n1, prefix1, n2, prefix2, t, alpha, subgaussian)
            if arm == 1:
                r = np.random.normal(func_target(n1, t, L1, U1, a, b), sigma)
                n1 += 1
                prefix1[n1] = prefix1[n1 - 1] + r
            else:
                r = np.random.normal(func_target(n2, t, L2, U2, a, b), sigma)
                n2 += 1
                prefix2[n2] = prefix2[n2 - 1] + r
            total += r
        totals[i] = total
    return totals


# ---------------------------------------------------------------------
# SW-UCB (windowed state: SINGLE fixed window, unlike RAW-UCB's
# multi-window test or R-ed-UCB's window growing with n -- Garivier &
# Moulines 2011)
# ---------------------------------------------------------------------

@njit(cache=True)
def sw_ucb_decide(n1, prefix1, n2, prefix2, t, tau=50, xi=1.0, subgaussian=1.0):
    if n1 == 0:
        return 1
    if n2 == 0:
        return 2

    w1 = min(n1, tau)
    mean1 = (prefix1[n1] - prefix1[n1 - w1]) / w1
    bonus1 = np.sqrt(xi * np.log(min(t, tau) + 2.0) / w1)
    idx1 = mean1 + subgaussian * bonus1

    w2 = min(n2, tau)
    mean2 = (prefix2[n2] - prefix2[n2 - w2]) / w2
    bonus2 = np.sqrt(xi * np.log(min(t, tau) + 2.0) / w2)
    idx2 = mean2 + subgaussian * bonus2

    return 1 if idx1 >= idx2 else 2


@njit(cache=True, parallel=True)
def run_sw_ucb_monte_carlo(n_mc, T, L1, U1, L2, U2, a, b, sigma, base_seed,
                             tau=50, xi=1.0, subgaussian=1.0):
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        prefix1 = np.zeros(T + 1)
        prefix2 = np.zeros(T + 1)
        total = 0.0
        for t in range(1, T + 1):
            arm = sw_ucb_decide(n1, prefix1, n2, prefix2, t, tau, xi, subgaussian)
            if arm == 1:
                r = np.random.normal(func_target(n1, t, L1, U1, a, b), sigma)
                n1 += 1
                prefix1[n1] = prefix1[n1 - 1] + r
            else:
                r = np.random.normal(func_target(n2, t, L2, U2, a, b), sigma)
                n2 += 1
                prefix2[n2] = prefix2[n2 - 1] + r
            total += r
        totals[i] = total
    return totals


# ---------------------------------------------------------------------
# SSW-UCB (windowed state: saturation-gated -- proposed algorithm)
# ---------------------------------------------------------------------

@njit(cache=True)
def ssw_ucb_index(n, prefix, t, sigma, short_frac=0.25, z_thresh=2.0, alpha=1.0):
    full_mean = prefix[n] / n
    short_w = max(1, int(short_frac * n))
    short_mean = (prefix[n] - prefix[n - short_w]) / short_w

    se = sigma * np.sqrt(1.0 / short_w + 1.0 / n)
    z = abs(short_mean - full_mean) / se if se > 0 else 0.0

    log_term = alpha * np.log(t + 2.0)
    if z < z_thresh:
        return full_mean + np.sqrt(2.0 * sigma ** 2 * log_term / n)
    else:
        return short_mean + np.sqrt(2.0 * sigma ** 2 * log_term / short_w)


@njit(cache=True)
def ssw_ucb_decide(n1, prefix1, n2, prefix2, t, sigma, short_frac=0.25, z_thresh=2.0, alpha=1.0):
    if n1 == 0:
        return 1
    if n2 == 0:
        return 2
    idx1 = ssw_ucb_index(n1, prefix1, t, sigma, short_frac, z_thresh, alpha)
    idx2 = ssw_ucb_index(n2, prefix2, t, sigma, short_frac, z_thresh, alpha)
    return 1 if idx1 >= idx2 else 2


@njit(cache=True, parallel=True)
def run_ssw_ucb_monte_carlo(n_mc, T, L1, U1, L2, U2, a, b, sigma, base_seed,
                              short_frac=0.25, z_thresh=2.0, alpha=1.0):
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        prefix1 = np.zeros(T + 1)
        prefix2 = np.zeros(T + 1)
        total = 0.0
        for t in range(1, T + 1):
            arm = ssw_ucb_decide(n1, prefix1, n2, prefix2, t, sigma, short_frac, z_thresh, alpha)
            if arm == 1:
                r = np.random.normal(func_target(n1, t, L1, U1, a, b), sigma)
                n1 += 1
                prefix1[n1] = prefix1[n1 - 1] + r
            else:
                r = np.random.normal(func_target(n2, t, L2, U2, a, b), sigma)
                n2 += 1
                prefix2[n2] = prefix2[n2 - 1] + r
            total += r
        totals[i] = total
    return totals


# ---------------------------------------------------------------------
# GLR-UCB (own state shape: sufficient statistics + commitment --
# proposed algorithm)
# ---------------------------------------------------------------------

@njit(cache=True)
def inv2x2(M):
    """Closed-form 2x2 inverse: skips numba's LAPACK path, which has
    fixed overhead not worth paying for a matrix this small. ~1.7-2x
    faster than np.linalg.inv for this specific use, confirmed by
    direct benchmark."""
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    det = a * d - b * c
    out = np.empty((2, 2))
    out[0, 0] = d / det; out[0, 1] = -b / det
    out[1, 0] = -c / det; out[1, 1] = a / det
    return out


@njit(cache=True)
def glr_ucb_decide(n1, sum1, XtX1, Xty1, n2, sum2, XtX2, Xty2, t, committed,
                    A, B, sigma, delta=0.05, c1=0.8, c2=1.6, min_floor=3, ucb_c=1.0, ridge=1e-6):
    if committed == 1:
        return 1, 1
    if committed == 2:
        return 2, 2
    if n1 < min_floor:
        return 1, 0
    if n2 < min_floor:
        return 2, 0

    M1 = XtX1 + ridge * np.eye(2)
    M2 = XtX2 + ridge * np.eye(2)
    M1_inv = inv2x2(M1)
    M2_inv = inv2x2(M2)
    beta1 = M1_inv @ Xty1
    beta2 = M2_inv @ Xty2
    c_vec = np.array([A, B])
    F1_hat = c_vec @ beta1
    F2_hat = c_vec @ beta2
    var1 = c_vec @ M1_inv @ c_vec
    var2 = c_vec @ M2_inv @ c_vec
    V = sigma ** 2 * (var1 + var2)

    diff = F1_hat - F2_hat
    Vb = V if V > 1 else 1.0001
    boundary = np.sqrt(2 * Vb * (c1 * np.log(np.log(Vb) + np.e) + c2 * np.log(1.0 / delta)))

    if abs(diff) > boundary:
        return (1, 1) if diff > 0 else (2, 2)

    m1, m2 = sum1 / n1, sum2 / n2
    bonus1 = ucb_c * np.sqrt(2.0 * np.log(t + 2.0) / n1)
    bonus2 = ucb_c * np.sqrt(2.0 * np.log(t + 2.0) / n2)
    choice = 1 if (m1 + bonus1) >= (m2 + bonus2) else 2
    return choice, 0


@njit(cache=True, parallel=True)
def run_glr_ucb_monte_carlo(n_mc, T, L1, U1, L2, U2, a, b, sigma, base_seed,
                              delta=0.05, c1=0.8, c2=1.6, min_floor=3, ucb_c=1.0):
    A, B = precompute_AB(a, b, T)
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        sum1, sum2 = 0.0, 0.0
        XtX1 = np.zeros((2, 2)); Xty1 = np.zeros(2)
        XtX2 = np.zeros((2, 2)); Xty2 = np.zeros(2)
        committed = 0
        total = 0.0
        for t in range(1, T + 1):
            arm, committed = glr_ucb_decide(n1, sum1, XtX1, Xty1, n2, sum2, XtX2, Xty2,
                                              t, committed, A, B, sigma, delta, c1, c2, min_floor, ucb_c)
            if arm == 1:
                r = np.random.normal(func_target(n1, t, L1, U1, a, b), sigma)
                w = S(n1 ** a - t ** b)
                x0, x1 = 1.0 - w, w
                XtX1[0, 0] += x0 * x0; XtX1[0, 1] += x0 * x1
                XtX1[1, 0] += x1 * x0; XtX1[1, 1] += x1 * x1
                Xty1[0] += x0 * r; Xty1[1] += x1 * r
                sum1 += r
                n1 += 1
            else:
                r = np.random.normal(func_target(n2, t, L2, U2, a, b), sigma)
                w = S(n2 ** a - t ** b)
                x0, x1 = 1.0 - w, w
                XtX2[0, 0] += x0 * x0; XtX2[0, 1] += x0 * x1
                XtX2[1, 0] += x1 * x0; XtX2[1, 1] += x1 * x1
                Xty2[0] += x0 * r; Xty2[1] += x1 * r
                sum2 += r
                n2 += 1
            total += r
        totals[i] = total
    return totals