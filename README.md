# Contesting Time-Attention Bandits (CTAB)

A two-armed bandit model where each arm's value both **rises under
sustained attention** and **independently decays with elapsed
time**, combining rested-rising and restless-rotting dynamics on the
same arm. Motivated by resource-allocation problems where neglect is
never free: labor platforms, energy infrastructure, cybersecurity
patch budgets, and startup product-direction decisions.

## Model

At each round `t`, the learner selects arm `i` and observes a noisy
reward with mean

```
mu_i(t, n_i) = L_i + (U_i - L_i) * S(n_i^alpha - t^beta)
```

where `S` is the logistic sigmoid, `n_i` is arm `i`'s own pull
count, `t` is the shared global round, and `L_i < U_i` are arm `i`'s
floor and ceiling. Domain: `alpha, beta in (0, 1]`, `K = 2` arms.

**Named regions** (the `alpha, beta` axis):

| Region | Condition | Key property |
|---|---|---|
| Attention-Dominant | `alpha > beta` | any deficit fully recovers |
| Time-Dominant | `alpha < beta` | out of scope; No-Switching provably fails here |
| Contested | `alpha = beta in (0,1)` | deficit shrinks but never fully closes |
| Critical Contested | `alpha = beta = 1` | deficit frozen exactly, forever |

**Floor-Ceiling cases** (the `L, U` axis, independent of `alpha, beta`):

| Case | Condition | Meaning |
|---|---|---|
| Dominance | `(L_i-L_j)(U_i-U_j) >= 0` | one arm pointwise beats the other |
| Crossover | `(L_i-L_j)(U_i-U_j) < 0` | genuine trade-off, no free lunch |

A continuous version of this split, `rho = 2xy/(x^2+y^2)` where
`x = L_i-L_j`, `y = U_i-U_j`, gives `rho in [-1, 1]`: `rho = -1` is
symmetric Crossover, `rho = +1` is symmetric Dominance, `rho = 0` is
the boundary.

## Key Results

**Proven, for `alpha >= beta`** (Attention-Dominant + Contested):

- **No-Switching Theorem**: the optimal policy is always constant
  commitment to a single arm -- never switch. This justifies
  `max(F_1, F_2)` as an exact, `O(T)` regret benchmark in place of
  the general `O(T^2)` DP oracle.
- **Tight regret characterization**:
  - Critical Contested (`a=b=1`): `Theta(T)`
  - Contested (`a=b=c<1`): `Theta(T^c)`
- **Partial regret characterization**:
  - Attention-Dominant (`a>b`): lower bound `Omega(log T)` proven;
    matching upper bound is **open**. Empirical evidence suggests
    standard algorithms (UCB1, BTC) do *not* achieve `O(log T)`
    here -- both show polynomial-looking exponents around `0.6-0.7`.

**Out of scope**: Time-Dominant (`alpha < beta`) -- No-Switching
provably fails here (explicit counterexample), and no regret
characterization is attempted.

**Open conjecture**: Dominance implies optimal constant commitment
for *any* `alpha, beta > 0`, not just `alpha >= beta`. Strongly
evidenced numerically (extensive search, no counterexample found),
not proven.

## Algorithms

**Greedy family** (`greedy.py`): Pure Greedy, ETC (block
exploration), BTC (balanced/alternating exploration). ETC and BTC
are both proven `O(T^min(1, 2/3+c))` in the Contested Region,
conditional on a stated exploration-gap assumption.

**UCB family** (`ucb.py`): UCB1 (proven `Omega(T)` at Critical
Contested), RAW-UCB, R-ed-UCB -- both literature baselines shown to
**saturation collapse** into UCB1-like behavior once an arm's value
saturates near `L` or `U`, since old and new observations become
statistically indistinguishable at that point.

**Proposed algorithms**: SSW-UCB (saturation-aware sliding window,
strong on Dominance), GLR-UCB (linear-model confidence sequence on
`F_1 - F_2`, strong away from the Crossover boundary, with a sharp
weakness confined to `rho` near `0`).

## Empirical Findings Worth Knowing

- **SW-UCB (fixed window) beats RAW-UCB (adaptive multi-window)**
  despite being the simpler algorithm -- adaptive window selection is
  precisely what gets fooled by saturation; a fixed window has
  nothing to get fooled by.
- **GLR-UCB achieves near-optimal exponents (`~0.03`) almost
  everywhere in the Attention-Dominant region**, except a sharp
  spike (`~0.87`) confined to `rho in [-0.27, 0.27]`, peaking at
  `rho ~ +-0.05` -- exactly the Dominance/Crossover boundary, where
  its confidence signal is weakest. Whether this reflects a genuine
  asymptotic rate or a finite-`T` transient is not yet established.

## Project Structure

```
src/core_periphery/
    environment.py             model (func_target), K=2 DP oracle,
                                closed-form alpha>=beta benchmark
                                (regret_benchmark), rho()
    greedy.py                  Pure Greedy, ETC, BTC + Monte Carlo runners
    ucb.py                     UCB1, RAW-UCB, SW-UCB, SSW-UCB, GLR-UCB
                                + Monte Carlo runners
    rho_grid.py                 sweeps (alpha,beta) x rho, fits regret
                                exponents via log-log regression across
                                multiple T, saves .png/.npy/.csv
    modal_app.py                runs rho_grid.py's sweep on Modal, one
                                container per algorithm in parallel
    summary_grid.py             collapses each cell's rho-strip into one
                                number (mean/max/etc), standard heatmap
                                -- batch-processes a folder of .npy files
    pooled_gradient.py           spatially pools the (alpha,beta) grid to
                                fewer, bigger cells while KEEPING the
                                full rho-gradient visible in each pooled
                                cell (unlike summary_grid.py, which
                                collapses rho away)
    regret_benchmark_verify.py   verifies the closed-form alpha>=beta
                                benchmark against the O(T^2) DP oracle
    archive/                    earlier/superseded implementations
    modal_outputs/
        csv/                    raw (alpha,beta,rho,exponent) data per algorithm
        npy/                    raw exponent arrays, one per algorithm
        png/                    rho-gradient grid plots
        pooled/                  pooled_gradient.py outputs
        summary/                 summary_grid.py outputs
docs/
bandits-env/
```

## Usage

```bash
# run the rho-gradient sweep for all algorithms in parallel on Modal
modal run src/core_periphery/modal_app.py --grid-size 16 --n-rho 30 --n-mc 100

# collapse each algorithm's rho-strip into a single summary heatmap
python3 src/core_periphery/summary_grid.py modal_outputs/npy modal_outputs/summary --agg all

# pool the (alpha,beta) grid down while keeping rho-gradients visible
python3 src/core_periphery/pooled_gradient.py modal_outputs/npy modal_outputs/pooled --pool 4
```

## Helpful Prompt for AI

Paste the block below at the start of a new AI conversation to get
instant, accurate context on this project -- no need to re-explain
the model, results, or codebase from scratch.

```
I'm working on "Time-Attention Competing Bandits" (TACB), a research
project on a two-armed bandit model where each arm's value both
rises under sustained attention and independently decays with
elapsed time. Here's the context:

MODEL
mu_i(t, n_i) = L_i + (U_i - L_i) * S(n_i^alpha - t^beta)
where S is the logistic sigmoid, n_i is arm i's own pull count,
t is the shared global round, and L_i < U_i are arm i's floor and
ceiling. Domain: alpha, beta in (0, 1]. K=2 arms.

NAMED REGIONS (alpha, beta axis)
- Attention-Dominant (alpha > beta): any deficit fully recovers
- Time-Dominant (alpha < beta): outside scope, not analyzed
- Contested (alpha = beta in (0,1)): deficit shrinks but never closes
- Critical Contested (alpha = beta = 1): deficit frozen exactly, forever

FLOOR-CEILING CASES (L, U axis, independent of alpha/beta)
- Dominance: (L_i-L_j)(U_i-U_j) >= 0 -- one arm pointwise beats the other
- Crossover: (L_i-L_j)(U_i-U_j) < 0 -- genuine trade-off, no free lunch
- rho = 2xy/(x^2+y^2) in [-1,1] is a continuous version of this split,
  x = L_i-L_j, y = U_i-U_j. rho=-1 symmetric Crossover, rho=+1
  symmetric Dominance, rho=0 is the boundary.

PROVEN RESULTS (all for alpha >= beta, i.e. Attention-Dominant + Contested)
- No-Switching Theorem: optimal policy is always constant commitment
  to one arm (never switch), justifying regret benchmark max(F1,F2)
- Regret is TIGHT (Theta, both bounds proven):
  - Critical Contested (a=b=1): Theta(T)
  - Contested (a=b=c<1): Theta(T^c)
- Regret is PARTIAL (one bound only):
  - Attention-Dominant (a>b): lower bound Omega(log T) proven;
    matching upper bound OPEN -- empirical evidence suggests it may
    NOT be log T for standard algorithms (UCB1, BTC both show
    polynomial-looking exponents ~0.6-0.7 there)
- Time-Dominant (a<b): out of scope entirely, No-Switching provably
  fails here (explicit counterexample exists)
- Conjecture (unproven, strongly evidenced numerically): Dominance
  implies optimal commitment for ANY alpha,beta>0, not just a>=b

ALGORITHMS (codebase: environment.py, greedy.py, ucb.py)
Greedy family: Pure Greedy, ETC (block exploration), BTC (balanced/
  alternating exploration) -- ETC and BTC both proven O(T^min(1,2/3+c))
  in the Contested Region, conditional on a stated exploration-gap
  assumption
UCB family: UCB1 (proven Omega(T) at Critical Contested), RAW-UCB,
  R-ed-UCB (both literature baselines -- shown to "saturation
  collapse" into UCB1-like behavior once an arm's value saturates
  near L or U, since old/new observations become indistinguishable)
Proposed: SSW-UCB (saturation-aware sliding window, strong on
  Dominance), GLR-UCB (linear-model confidence sequence on F1-F2,
  strong away from the Crossover boundary, weak exactly at rho~0)

KEY EMPIRICAL FINDINGS
- SW-UCB (fixed window) beats RAW-UCB (adaptive multi-window) despite
  being simpler -- adaptivity is what gets fooled by saturation, not
  insufficient sophistication
- GLR-UCB achieves near-optimal exponents (~0.03) almost everywhere
  in Attention-Dominant, EXCEPT a sharp spike (~0.87) confined to
  rho in [-0.27, 0.27], peaking at rho~+-0.05 -- exactly the
  Dominance/Crossover boundary where its confidence signal is weakest

CODEBASE STRUCTURE (all in src/core_periphery/)
- environment.py: model (func_target), K=2 DP oracle, closed-form
  alpha>=beta benchmark (regret_benchmark), rho()
- greedy.py: Pure Greedy, ETC, BTC + Monte Carlo runners
- ucb.py: UCB1, RAW-UCB, SW-UCB, SSW-UCB, GLR-UCB + Monte Carlo runners
- rho_grid.py: sweeps (alpha,beta) x rho, fits regret exponents via
  log-log regression across multiple T, saves .png/.npy/.csv
- modal_app.py: runs rho_grid.py's sweep on Modal, one container per
  algorithm in parallel
- summary_grid.py: collapses each cell's rho-strip into one number
  (mean/max/etc), standard heatmap -- takes a folder of .npy files
- pooled_gradient.py: spatially pools the (alpha,beta) grid to fewer,
  bigger cells while KEEPING the full rho-gradient visible in each
  pooled cell (unlike summary_grid.py, which collapses rho away)
- regret_benchmark_verify.py: verifies the closed-form alpha>=beta
  benchmark against the O(T^2) DP oracle
- modal_outputs/{csv,npy,png,pooled,summary}/: outputs by type

WHEN HELPING WITH THIS PROJECT
- Always verify numerically before asserting a theoretical claim --
  this project has caught several wrong derivations this way already
- Distinguish clearly between PROVEN (cite the theorem/proposition)
  and EMPIRICAL (cite the specific test) -- don't blur the two
- The paper structure is: Model -> Regret (basic results) -> Theory
  of the Model (lower bounds, Floor-Ceiling cases, rho) -> Applications
  -> Related Work -> Greedy Analysis -> UCB Analysis -> Proposed
  Algorithms -> Empirical Results
```

## License

See `LICENSE`.
