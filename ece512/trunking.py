"""Trunking, Markov chains and Erlang-B/C (Week 3 lecture).

Notation follows the Week 3 slides:

* ``lam`` = call arrival rate lambda (calls/s, over *all* users in the trunk),
  ``mu`` = service rate (1/mu = mean holding time, s/call),
  ``rho = lam / mu`` = offered traffic (Erlangs), ``c`` = channels in the trunk.
* Discrete-time Markov chains use row-vector distributions: pi_{n+1} = pi_n P.

Pure computation only -- plotting lives in the lecture notebook.
"""

from dataclasses import dataclass
import heapq
import math

import numpy as np
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# Spectral efficiency  eta_S = eta_B * eta_C * eta_T
# ---------------------------------------------------------------------------


def hex_cell_area(radius):
    """Area of a hexagonal cell with center-to-vertex radius R: (3*sqrt(3)/2) R^2."""
    return 1.5 * np.sqrt(3.0) * np.asarray(radius, dtype=float) ** 2


def first_tier_sir(n_cluster, beta):
    """First-tier SIR approximation Lambda ~= (D/R)^beta / 6 = (sqrt(3N))^beta / 6 (linear)."""
    return np.sqrt(3.0 * np.asarray(n_cluster, dtype=float)) ** beta / 6.0


def spectral_efficiency(total_bw_hz, channel_bw_hz, n_cluster, cell_area_m2, gos=0.02):
    """Decompose FDMA spectral efficiency eta_S = eta_B eta_C eta_T (Erlangs/m^2/Hz).

    With c = floor(W / (W_ch N)) channels per cell and A = carried-capacity
    offered traffic at grade of service ``gos`` (Erlang B):

    * eta_B = 1 / W_ch            channels per Hz            (bandwidth efficiency)
    * eta_C = 1 / (N * A_cell)    channel-set reuses per m^2 (spatial efficiency)
    * eta_T = A / c               Erlangs per channel        (trunking efficiency)

    so eta_S = A / (W_ch N c A_cell) ~= A / (W A_cell) Erlangs/m^2/Hz.
    Returns a dict with keys ``c, A, eta_B, eta_C, eta_T, eta_S``.
    """
    c = int(total_bw_hz // (channel_bw_hz * n_cluster))
    a = offered_traffic_for_gos(c, gos) if c > 0 else 0.0
    eta_b = 1.0 / channel_bw_hz
    eta_c = 1.0 / (n_cluster * cell_area_m2)
    eta_t = a / c if c > 0 else 0.0
    return dict(c=c, A=a, eta_B=eta_b, eta_C=eta_c, eta_T=eta_t, eta_S=eta_b * eta_c * eta_t)


# ---------------------------------------------------------------------------
# Erlang B / Erlang C
# ---------------------------------------------------------------------------


def erlang_b(rho, c):
    """Erlang-B blocking probability B(rho, c) = (rho^c / c!) / sum_{k=0}^{c} rho^k / k!.

    Uses the numerically stable recursion B(rho, 0) = 1,
    B(rho, k) = rho B(rho, k-1) / (k + rho B(rho, k-1)), so large c (hundreds or
    thousands of channels) never overflows. ``rho`` and ``c`` broadcast.
    """
    rho, c = np.broadcast_arrays(np.asarray(rho, dtype=float), np.asarray(c, dtype=int))
    out = np.ones(rho.shape)
    b = np.ones(rho.shape)
    for k in range(1, int(c.max(initial=0)) + 1):
        b = rho * b / (k + rho * b)
        out = np.where(c == k, b, out)
    return out[()] if out.ndim == 0 else out


def erlang_b_direct(rho, c):
    """Erlang B straight from the slide formula with factorials (float64).

    Kept only to show why the recursion is needed: rho**c / c! overflows to
    inf/inf = nan for c >~ 170.
    """
    rho = np.float64(rho)
    with np.errstate(over="ignore", invalid="ignore"):
        terms = [rho**k / np.float64(math.gamma(k + 1)) if k <= 170 else np.float64(np.inf)
                 for k in range(c + 1)]
        return terms[-1] / np.sum(terms)


def erlang_c(rho, c):
    """Erlang-C probability of delay for M/M/c/inf (rho < c):

        C(rho, c) = rho^c / (rho^c + c! (1 - rho/c) sum_{k=0}^{c-1} rho^k / k!)
                  = c B(rho, c) / (c - rho (1 - B(rho, c)))

    The second (stable) form is used; returns 1 where rho >= c (unstable queue).
    """
    rho, c = np.broadcast_arrays(np.asarray(rho, dtype=float), np.asarray(c, dtype=int))
    b = np.asarray(erlang_b(rho, c), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(rho < c, c * b / (c - rho * (1 - b)), 1.0)
    return out[()] if out.ndim == 0 else out


def offered_traffic_for_gos(c, gos):
    """Largest offered traffic A (Erlangs) such that B(A, c) <= gos (inverse Erlang B)."""
    if c <= 0:
        return 0.0
    hi = c + 10.0 * np.sqrt(c) + 10.0
    while erlang_b(hi, c) < gos:
        hi *= 2
    return brentq(lambda a: erlang_b(a, c) - gos, 0.0, hi, xtol=1e-10)


def channels_for_gos(rho, gos):
    """Smallest number of channels c with B(rho, c) <= gos."""
    b, k = 1.0, 0
    while b > gos:
        k += 1
        b = rho * b / (k + rho * b)
    return k


def erlang_c_wait_tail(t, lam, mu, c):
    """Pr(W > t) = C(rho, c) exp(-(c mu - lam) t) for M/M/c FCFS (all calls)."""
    return erlang_c(lam / mu, c) * np.exp(-(c * mu - lam) * np.asarray(t, dtype=float))


def erlang_c_mean_queue(rho, c):
    """Mean number waiting N_Q = sum_{i>=c} (i - c) pi(i) = C(rho, c) rho / (c - rho)."""
    return erlang_c(rho, c) * rho / (c - rho)


def erlang_c_mean_wait(lam, mu, c):
    """Mean wait in queue (Little: N_Q = lam W_Q) -> W_Q = C(rho, c) / (c mu - lam)."""
    return erlang_c(lam / mu, c) / (c * mu - lam)


# ---------------------------------------------------------------------------
# Markov chains
# ---------------------------------------------------------------------------


def two_state_P(a, b):
    """Two-state chain from the slides: P = [[a, 1-a], [b, 1-b]]."""
    return np.array([[a, 1 - a], [b, 1 - b]], dtype=float)


def two_state_stationary(a, b):
    """Stationary pi for P = [[a, 1-a], [b, 1-b]]: pi = [b, 1-a] / (1 - a + b)."""
    return np.array([b, 1 - a]) / (1 - a + b)


def dtmc_evolve(pi0, P, n_steps):
    """Distributions pi_0, pi_1, ..., pi_n with pi_{n+1} = pi_n P (rows of the result)."""
    P = np.asarray(P, dtype=float)
    out = np.empty((n_steps + 1, P.shape[0]))
    out[0] = pi0
    for n in range(n_steps):
        out[n + 1] = out[n] @ P
    return out


def stationary_distribution(P):
    """Solve pi = pi P with sum(pi) = 1 (least squares on the stacked system)."""
    P = np.asarray(P, dtype=float)
    k = P.shape[0]
    A = np.vstack([P.T - np.eye(k), np.ones(k)])
    rhs = np.zeros(k + 1)
    rhs[-1] = 1.0
    return np.linalg.lstsq(A, rhs, rcond=None)[0]


def simulate_dtmc(P, n_steps, x0=0, rng=None):
    """Sample path X(0..n_steps) of a discrete-time Markov chain."""
    rng = np.random.default_rng(rng)
    cdf = np.cumsum(np.asarray(P, dtype=float), axis=1)
    x = np.empty(n_steps + 1, dtype=int)
    x[0] = x0
    u = rng.random(n_steps)
    for n in range(n_steps):
        x[n + 1] = min(np.searchsorted(cdf[x[n]], u[n], side="right"), cdf.shape[0] - 1)
    return x


def birth_death_stationary(birth, death):
    """Stationary distribution of a finite birth-death chain from detailed balance.

    ``birth[i]`` = rate i -> i+1 (i = 0..K-1), ``death[i]`` = rate i+1 -> i.
    Detailed balance pi(i) birth[i] = pi(i+1) death[i] gives
    pi(i) = pi(0) prod_{k<i} birth[k] / death[k]; computed in the log domain.
    """
    birth = np.asarray(birth, dtype=float)
    death = np.asarray(death, dtype=float)
    logw = np.concatenate([[0.0], np.cumsum(np.log(birth) - np.log(death))])
    w = np.exp(logw - logw.max())
    return w / w.sum()


def generator_stationary(Q):
    """Stationary pi of a CTMC with generator Q (global balance pi Q = 0, sum pi = 1)."""
    Q = np.asarray(Q, dtype=float)
    k = Q.shape[0]
    A = np.vstack([Q.T, np.ones(k)])
    rhs = np.zeros(k + 1)
    rhs[-1] = 1.0
    return np.linalg.lstsq(A, rhs, rcond=None)[0]


def mmc_generator(lam, mu, c, n_states):
    """Generator of the M/M/c birth-death chain truncated to states 0..n_states-1
    (birth lam, death min(i, c) mu). With n_states = c + 1 this is M/M/c/c."""
    Q = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        Q[i, i + 1] = lam
        Q[i + 1, i] = min(i + 1, c) * mu
    Q -= np.diag(Q.sum(axis=1))
    return Q


def mmcc_stationary(rho, c):
    """M/M/c/c occupancy pi(i) = (rho^i / i!) / sum_k rho^k / k!  (truncated Poisson)."""
    return birth_death_stationary(np.full(c, rho), np.arange(1, c + 1))


def mmc_stationary(rho, c, n_max):
    """M/M/c/inf occupancy pi(0..n_max) (rho < c): rho^i/i! pi0 for i <= c and
    c^c/c! (rho/c)^i pi0 for i > c (slide 'Erlang-C: Detailed Balance')."""
    i = np.arange(n_max + 1)
    logw = np.where(i <= c,
                    i * np.log(rho) - np.array([math.lgamma(k + 1) for k in i]),
                    c * np.log(c) - math.lgamma(c + 1) + i * np.log(rho / c))
    pi0 = 1.0 / (sum(rho**k / math.factorial(k) for k in range(c))
                 + rho**c / math.factorial(c) / (1 - rho / c))
    return pi0 * np.exp(logw)


# ---------------------------------------------------------------------------
# Poisson processes
# ---------------------------------------------------------------------------


def poisson_process(rate, t_end, rng=None):
    """Event times of a rate-``rate`` Poisson process on (0, t_end] (exp inter-arrivals)."""
    rng = np.random.default_rng(rng)
    n_guess = int(rate * t_end + 6 * np.sqrt(rate * t_end) + 10)
    t = np.cumsum(rng.exponential(1.0 / rate, n_guess))
    while t[-1] < t_end:
        t = np.concatenate([t, t[-1] + np.cumsum(rng.exponential(1.0 / rate, n_guess))])
    return t[t <= t_end]


def counts_in_windows(times, tau, t_end):
    """Number of events A(k tau, (k+1) tau] in consecutive windows of length tau."""
    edges = np.arange(0.0, t_end + 1e-12, tau)
    return np.histogram(times, bins=edges)[0]


# ---------------------------------------------------------------------------
# Discrete-event simulation of M/M/c/c (Erlang B) and M/M/c FCFS (Erlang C)
# ---------------------------------------------------------------------------


@dataclass
class QueueSim:
    """Result of :func:`simulate_mmc`.

    arrivals, service: per-call arrival time and holding time;
    blocked: bool mask (loss system only); wait: time in queue (nan if blocked);
    state_frac[i]: fraction of time with i calls in the system over [0, t_end].
    """
    arrivals: np.ndarray
    service: np.ndarray
    blocked: np.ndarray
    wait: np.ndarray
    state_frac: np.ndarray
    t_end: float

    @property
    def blocking(self):
        return self.blocked.mean()

    @property
    def delayed(self):
        w = self.wait[~self.blocked]
        return np.mean(w > 0)


def simulate_mmc(lam, mu, c, n_calls, loss=True, rng=None):
    """Simulate ``n_calls`` Poisson(lam) arrivals with exp(mu) holding times on c channels.

    loss=True  -> M/M/c/c, blocked calls cleared (Erlang B).
    loss=False -> M/M/c/inf FCFS, blocked calls delayed (Erlang C).
    Uses a heap of channel free-times: a call starts at max(arrival, earliest free).
    """
    rng = np.random.default_rng(rng)
    arr = np.cumsum(rng.exponential(1.0 / lam, n_calls))
    svc = rng.exponential(1.0 / mu, n_calls)
    free = [0.0] * c
    heapq.heapify(free)
    blocked = np.zeros(n_calls, dtype=bool)
    wait = np.full(n_calls, np.nan)
    for k in range(n_calls):
        earliest = free[0]
        if loss and earliest > arr[k]:
            blocked[k] = True
            continue
        start = max(arr[k], earliest)
        wait[k] = start - arr[k]
        heapq.heapreplace(free, start + svc[k])
    t_end = arr[-1]
    ok = ~blocked
    dep = arr[ok] + wait[ok] + svc[ok]
    times = np.concatenate([arr[ok], dep])
    steps = np.concatenate([np.ones(ok.sum()), -np.ones(ok.sum())])
    order = np.argsort(times, kind="stable")
    times, steps = times[order], steps[order]
    keep = times <= t_end
    times, steps = np.concatenate([[0.0], times[keep], [t_end]]), steps[keep]
    level = np.concatenate([[0], np.cumsum(steps)]).astype(int)
    dur = np.diff(times)
    state_frac = np.bincount(level, weights=dur) / t_end
    return QueueSim(arr, svc, blocked, wait, state_frac, t_end)
