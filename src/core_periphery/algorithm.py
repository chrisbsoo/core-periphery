"""
UCB1 -- continuous exploration, never fully commits. Adapted to the
two-armed floor/ceiling environment (see environment.py).
"""

import numpy as np
from numba import njit, prange

@njit(cache=True)
def func_target(n, t, L, U, a, b):
    n_term = n ** a if n > 0 else 0.0
    t_term = t ** b if t > 0 else 0.0
    return L + (U - L) / (1.0 + np.exp(-(n_term - t_term)))

@njit(cache=True)
def S(x):
    return 1.0 / (1.0 + np.exp(-x))

@njit(cache=True)
def precompute_AB(a, b, T):
    A, B = 0.0, 0.0
    for t in range(1, T + 1):
        w = S((t - 1)**a - t**b)
        A += 1 - w
        B += w
    return A, B

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


@njit(cache=True)
def balanced_explore_decide(n1, sum1, n2, sum2, t, T, c=1.0):
    m = int(c * T ** (2.0 / 3.0))
    if t <= m:
        return 1 if n1 <= n2 else 2
    if n1 == 0:
        return 1
    if n2 == 0:
        return 2
    m1 = sum1 / n1
    m2 = sum2 / n2
    return 1 if m1 >= m2 else 2

@njit(cache=True)
def rawucb_decide(n1, prefix1, n2, prefix2, t, alpha=1.0, subgaussian=1.0):
    if n1 == 0:
        return 1
    if n2 == 0:
        return 2

    log_term = alpha * np.log(t + 2.0)

    best1 = 1e18
    for w in range(1, n1 + 1):
        windowed_sum = prefix1[n1] - prefix1[n1 - w]
        windowed_mean = windowed_sum / w
        bonus = np.sqrt(2.0 * subgaussian**2 * log_term / w)
        idx = windowed_mean + bonus
        if idx < best1:
            best1 = idx

    best2 = 1e18
    for w in range(1, n2 + 1):
        windowed_sum = prefix2[n2] - prefix2[n2 - w]
        windowed_mean = windowed_sum / w
        bonus = np.sqrt(2.0 * subgaussian**2 * log_term / w)
        idx = windowed_mean + bonus
        if idx < best2:
            best2 = idx

    return 1 if best1 >= best2 else 2

@njit(cache=True)
def saturation_aware_index(n, prefix, t, sigma, short_frac=0.25, z_thresh=2.0, alpha=1.0):
    full_mean = prefix[n] / n
    short_w = max(1, int(short_frac * n))
    short_mean = (prefix[n] - prefix[n - short_w]) / short_w

    se = sigma * np.sqrt(1.0/short_w + 1.0/n)
    z = abs(short_mean - full_mean) / se if se > 0 else 0.0

    log_term = alpha * np.log(t + 2.0)
    if z < z_thresh:
        idx = full_mean + np.sqrt(2.0*sigma**2*log_term/n)
    else:
        idx = short_mean + np.sqrt(2.0*sigma**2*log_term/short_w)
    return idx


@njit(cache=True)
def saturation_aware_decide(n1, prefix1, n2, prefix2, t, sigma, short_frac=0.25, z_thresh=2.0, alpha=1.0):
    if n1 == 0:
        return 1
    if n2 == 0:
        return 2
    idx1 = saturation_aware_index(n1, prefix1, t, sigma, short_frac, z_thresh, alpha)
    idx2 = saturation_aware_index(n2, prefix2, t, sigma, short_frac, z_thresh, alpha)
    return 1 if idx1 >= idx2 else 2

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
    M1_inv = np.linalg.inv(M1)
    M2_inv = np.linalg.inv(M2)
    beta1 = M1_inv @ Xty1
    beta2 = M2_inv @ Xty2
    c_vec = np.array([A, B])
    F1_hat = c_vec @ beta1
    F2_hat = c_vec @ beta2
    var1 = c_vec @ M1_inv @ c_vec
    var2 = c_vec @ M2_inv @ c_vec
    V = sigma**2 * (var1 + var2)

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
def run_ucb1_monte_carlo(n_mc, T, L1, U1, a, b, L2, U2, sigma, base_seed, ucb_c=1.0):
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        sum1, sum2 = 0.0, 0.0
        total = 0.0
        for t in range(1, T + 1):
            arm = ucb1_decide(n1, sum1, n2, sum2, t, ucb_c)
            if arm == 1:
                mu = func_target(n1, t, L1, U1, a, b)
                r = np.random.normal(mu, sigma)
                sum1 += r
                n1 += 1
            else:
                mu = func_target(n2, t, L2, U2, a, b)
                r = np.random.normal(mu, sigma)
                sum2 += r
                n2 += 1
            total += r
        totals[i] = total
    return totals

@njit(cache=True, parallel=True)
def run_balanced_explore_monte_carlo(n_mc, T, L1, U1, a, b, L2, U2, sigma, base_seed, c=1.0):
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        sum1, sum2 = 0.0, 0.0
        total = 0.0
        for t in range(1, T + 1):
            arm = balanced_explore_decide(n1, sum1, n2, sum2, t, T, c)
            if arm == 1:
                mu = func_target(n1, t, L1, U1, a, b)
                r = np.random.normal(mu, sigma)
                sum1 += r
                n1 += 1
            else:
                mu = func_target(n2, t, L2, U2, a, b)
                r = np.random.normal(mu, sigma)
                sum2 += r
                n2 += 1
            total += r
        totals[i] = total
    return totals


@njit(cache=True, parallel=True)
def run_rawucb_monte_carlo(n_mc, T, L1, U1, a, b, L2, U2, sigma, base_seed,
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
                mu = func_target(n1, t, L1, U1, a, b)
                r = np.random.normal(mu, sigma)
                n1 += 1
                prefix1[n1] = prefix1[n1 - 1] + r
            else:
                mu = func_target(n2, t, L2, U2, a, b)
                r = np.random.normal(mu, sigma)
                n2 += 1
                prefix2[n2] = prefix2[n2 - 1] + r
            total += r
        totals[i] = total
    return totals



@njit(cache=True, parallel=True)
def run_glr_ucb_monte_carlo(n_mc, T, L1, U1, a, b, L2, U2, sigma, base_seed,
                              delta=0.05, c1=0.8, c2=1.6, min_floor=3, ucb_c=1.0):
    A, B = precompute_AB(a, b, T)  # a,b must be shared between arms for this algorithm
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
                mu = func_target(n1, t, L1, U1, a, b)
                r = np.random.normal(mu, sigma)
                w = S(n1**a - t**b)
                x0, x1 = 1 - w, w
                XtX1[0, 0] += x0*x0; XtX1[0, 1] += x0*x1; XtX1[1, 0] += x1*x0; XtX1[1, 1] += x1*x1
                Xty1[0] += x0 * r; Xty1[1] += x1 * r
                sum1 += r
                n1 += 1
            else:
                mu = func_target(n2, t, L2, U2, a, b)
                r = np.random.normal(mu, sigma)
                w = S(n2**a - t**b)
                x0, x1 = 1 - w, w
                XtX2[0, 0] += x0*x0; XtX2[0, 1] += x0*x1; XtX2[1, 0] += x1*x0; XtX2[1, 1] += x1*x1
                Xty2[0] += x0 * r; Xty2[1] += x1 * r
                sum2 += r
                n2 += 1
            total += r
        totals[i] = total
    return totals

@njit(cache=True, parallel=True)
def run_saturation_aware_monte_carlo(n_mc, T, L1, U1, a, b, L2, U2, sigma, base_seed,
                                       short_frac=0.25, z_thresh=2.0, alpha=1.0):
    totals = np.zeros(n_mc)
    for i in prange(n_mc):
        np.random.seed(base_seed + i * 7919 + 1)
        n1, n2 = 0, 0
        prefix1 = np.zeros(T + 1)
        prefix2 = np.zeros(T + 1)
        total = 0.0
        for t in range(1, T + 1):
            arm = saturation_aware_decide(n1, prefix1, n2, prefix2, t, sigma, short_frac, z_thresh, alpha)
            if arm == 1:
                mu = func_target(n1, t, L1, U1, a, b)
                r = np.random.normal(mu, sigma)
                n1 += 1
                prefix1[n1] = prefix1[n1 - 1] + r
            else:
                mu = func_target(n2, t, L2, U2, a, b)
                r = np.random.normal(mu, sigma)
                n2 += 1
                prefix2[n2] = prefix2[n2 - 1] + r
            total += r
        totals[i] = total
    return totals
