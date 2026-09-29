"""Narrowband MIMO channels, metrics, capacity and receivers (Week 9).

Conventions follow the Week 9 lecture:

* ``H`` is the M_R x M_T channel matrix in  y = H x + n  (receive antennas are rows).
  Every function accepts a single matrix or a batch shaped ``(..., M_R, M_T)``.
* ``H_w`` is spatially white: i.i.d. CN(0, 1) entries (``ece512.fading.iid_rayleigh``).
* ``lambda_i`` are the eigenvalues of H H^H (= squared singular values sigma_i^2 of H),
  sorted in descending order.
* ``snr_db`` is the total transmit SNR E_s/N_0 in dB. The open-loop signal model is
  y = sqrt(E_s/M_T) H x + n with unit-energy symbols x_i and n ~ CN(0, N_0 I), so each
  receive antenna sees average SNR E_s/N_0 when E[|h_ij|^2] = 1.
"""

import itertools
from math import comb

import numpy as np
from scipy.integrate import quad
from scipy.special import eval_genlaguerre, gammaln

from ece512.fading import iid_rayleigh


def _lin(snr_db):
    return 10 ** (np.asarray(snr_db, dtype=float) / 10)


# ---------------------------------------------------------------------------
# Channel metrics
# ---------------------------------------------------------------------------

def eigenvalues(H):
    """Eigenvalues lambda_i of H H^H, descending, length min(M_R, M_T) (the non-trivial ones).

    lambda_i = sigma_i^2 where H = U Sigma V^H is the SVD (lecture: "Singular Value Decomposition").
    """
    return singular_values(H) ** 2


def singular_values(H):
    """Singular values sigma_i of H (descending), shape ``(..., min(M_R, M_T))``."""
    return np.linalg.svd(np.asarray(H), compute_uv=False)


def frobenius_sq(H):
    """Squared Frobenius norm ||H||_F^2 = Tr(H H^H) = sum_ij |h_ij|^2 = sum_i lambda_i."""
    H = np.asarray(H)
    return np.sum(np.abs(H) ** 2, axis=(-2, -1))


def condition_number(H):
    """Condition number kappa = sigma_max / sigma_min (ratio of singular values of H).

    Equivalently sqrt(lambda_max / lambda_min) in terms of the eigenvalues of H H^H.
    Rank-deficient channels give a huge (numerically) or infinite kappa.
    """
    s = singular_values(H)
    with np.errstate(divide="ignore"):
        return s[..., 0] / s[..., -1]


def demmel_condition_number(H):
    """Demmel condition number kappa_D = ||H||_F / sigma_min."""
    s = singular_values(H)
    with np.errstate(divide="ignore"):
        return np.sqrt(frobenius_sq(H)) / s[..., -1]


def rank(H, tol=1e-9):
    """Numerical rank r = number of singular values above ``tol * sigma_max``."""
    s = singular_values(H)
    return np.sum(s > tol * s[..., :1], axis=-1)


# ---------------------------------------------------------------------------
# Channel models
# ---------------------------------------------------------------------------

def exponential_correlation(n, rho):
    """n x n exponential correlation matrix R = [rho^|i-j|] (uniform linear array model).

    A complex ``rho`` gives R_ij = rho^(j-i) for j >= i and its conjugate below the diagonal.
    """
    i = np.arange(n)
    d = i[None, :] - i[:, None]
    R = np.where(d >= 0, np.power(complex(rho), np.abs(d)), np.conj(np.power(complex(rho), np.abs(d))))
    return R.real if np.isrealobj(rho) else R


def sqrtm_psd(R):
    """Hermitian square root R^{1/2} of a positive semidefinite matrix (via eigh)."""
    w, V = np.linalg.eigh(np.asarray(R))
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.conj().T


def kronecker_channel(R_r, R_t, n=None, rng=None, H_w=None):
    """Kronecker model H = R_r^{1/2} H_w R_t^{1/2} (lecture "Kronecker Model (I)").

    ``R_r`` is M_R x M_R, ``R_t`` is M_T x M_T (unit diagonals keep E[|h_ij|^2] = 1).
    Returns ``n`` realisations shaped ``(n, M_R, M_T)`` (or one matrix if ``n`` is None).
    Pass ``H_w`` to reuse a white channel (common random numbers across parameter sweeps).
    """
    R_r, R_t = np.atleast_2d(R_r), np.atleast_2d(R_t)
    if H_w is None:
        shape = (R_r.shape[0], R_t.shape[0]) if n is None else (n, R_r.shape[0], R_t.shape[0])
        H_w = iid_rayleigh(shape, rng)
    return sqrtm_psd(R_r) @ H_w @ sqrtm_psd(R_t)


def ula_steering(m, theta, spacing=0.5):
    """Uniform linear array response a(theta) = [exp(j 2 pi (d/lambda) k sin theta)]_{k=0}^{m-1}."""
    k = np.arange(m)
    return np.exp(2j * np.pi * spacing * k * np.sin(theta))


def los_channel(m_r, m_t, theta_r=0.0, theta_t=0.0, spacing=0.5):
    """Rank-one far-field LOS matrix H_LOS = a_R(theta_r) a_T(theta_t)^T, unit-modulus entries.

    With broadside angles (default) every entry is 1 -- the "all ones" rank-deficient channel.
    """
    return np.outer(ula_steering(m_r, theta_r, spacing), ula_steering(m_t, theta_t, spacing))


def ricean_channel(k_factor, H_los, H_w):
    """Ricean MIMO channel H = sqrt(1/(1+K)) H_w + sqrt(K/(1+K)) H_LOS (lecture "Ricean Fading")."""
    k = float(k_factor)
    return np.sqrt(1 / (1 + k)) * np.asarray(H_w) + np.sqrt(k / (1 + k)) * np.asarray(H_los)


def keyhole_channel(m_r, m_t, n=None, rng=None):
    """Pin-hole (keyhole) channel H = h_r h_t^T with independent CN(0, I) vectors: rank one."""
    rng = np.random.default_rng(rng)
    lead = () if n is None else (n,)
    h_r = iid_rayleigh(lead + (m_r, 1), rng)
    h_t = iid_rayleigh(lead + (1, m_t), rng)
    return h_r @ h_t


def estimate_correlations(H):
    """Sample correlations from N snapshots (lecture "Model Parameter Estimation"):

    R_RX = (1/N) sum H H^H,  R_TX = (1/N) sum H^H H.  (For the Kronecker model with
    unit-diagonal R_r, R_t these equal M_T R_r and M_R R_t.)
    """
    H = np.asarray(H)
    Hh = np.conj(np.swapaxes(H, -1, -2))
    return np.mean(H @ Hh, axis=0), np.mean(Hh @ H, axis=0)


def frequency_response(H_taps, n_fft):
    """Per-subcarrier matrices H_i = sum_l H_l exp(-j 2 pi l i / N) for an L-tap MIMO channel.

    ``H_taps`` is ``(L, M_R, M_T)``; returns ``(N, M_R, M_T)``. (np.fft sign convention, which
    matches y(k) = sum_l H_l x(k - l).)
    """
    return np.fft.fft(np.asarray(H_taps), n=n_fft, axis=0)


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------

def capacity_open_loop(H, snr_db):
    """Open-loop mutual information (bit/s/Hz), equal power E_s/M_T per antenna:

        I = log2 det(I_{M_R} + E_s/(M_T N_0) H H^H) = sum_i log2(1 + E_s/(M_T N_0) lambda_i).

    ``snr_db`` may be an array; the result broadcasts as ``snr.shape + batch shape``.
    """
    H = np.asarray(H)
    lam = eigenvalues(H)
    m_t = H.shape[-1]
    rho = _lin(snr_db)
    rho = rho.reshape(rho.shape + (1,) * lam.ndim)
    return np.sum(np.log2(1 + rho / m_t * lam), axis=-1)


def waterfill(lam, snr_db, m_t):
    """Water-filling power allocation over eigenmodes (lecture "Closed Loop Capacity (IV)").

    gamma_i = (mu - M_T N_0 / (E_s lambda_i))^+ with sum_i gamma_i = M_T.
    ``lam`` is a 1-D array of eigenvalues. Returns (gamma, mu) with gamma in the order of ``lam``.
    """
    lam = np.asarray(lam, dtype=float)
    rho = float(_lin(snr_db))
    floor = np.full(lam.shape, np.inf)
    pos = lam > 0
    floor[pos] = m_t / (rho * lam[pos])
    if not pos.any():
        return np.zeros_like(lam), 0.0
    f = np.sort(floor)
    mu = np.inf
    for k in range(len(f), 0, -1):
        if not np.isfinite(f[k - 1]):
            continue
        mu = (m_t + f[:k].sum()) / k
        if mu > f[k - 1]:
            break
    gamma = np.clip(mu - floor, 0, None)
    gamma[~pos] = 0.0
    return gamma, mu


def capacity_closed_loop(H, snr_db):
    """Closed-loop (water-filling) capacity C = max sum_i log2(1 + E_s gamma_i lambda_i / (M_T N_0)).

    Works on a single matrix or a batch (loops over the batch); scalar ``snr_db``.
    """
    H = np.asarray(H)
    m_t = H.shape[-1]
    lam = eigenvalues(H)
    flat = lam.reshape(-1, lam.shape[-1])
    rho = float(_lin(snr_db))
    out = np.empty(len(flat))
    for i, l in enumerate(flat):
        g, _ = waterfill(l, snr_db, m_t)
        out[i] = np.sum(np.log2(1 + rho * g * l / m_t))
    return out.reshape(lam.shape[:-1]) if lam.ndim > 1 else out[0]


def outage_capacity(samples, p):
    """C_out,p such that Pr(I <= C_out,p) = p (``p`` as a fraction, e.g. 0.1 for 10 %)."""
    return np.quantile(np.asarray(samples), p)


def ergodic_capacity_iid(m_t, m_r, snr_db):
    """Exact ergodic open-loop capacity of the i.i.d. Rayleigh M_R x M_T channel (Telatar 1999):

        C = int_0^inf log2(1 + rho lambda / M_T) sum_{k=0}^{m-1} k!/(k+n-m)! [L_k^{n-m}(lambda)]^2
            lambda^{n-m} e^{-lambda} dlambda,   m = min(M_T, M_R), n = max(M_T, M_R),

    i.e. m times the average capacity of one unordered eigenvalue of the Wishart matrix.
    """
    m, n = min(m_t, m_r), max(m_t, m_r)
    a = n - m

    def density(x):  # sum over k of Laguerre terms times weight
        s = 0.0
        for k in range(m):
            s += np.exp(gammaln(k + 1) - gammaln(k + a + 1)) * eval_genlaguerre(k, a, x) ** 2
        return s * x**a * np.exp(-x)

    out = []
    for rho in np.atleast_1d(_lin(snr_db)):
        val, _ = quad(lambda x: np.log2(1 + rho * x / m_t) * density(x), 0, np.inf, limit=200)
        out.append(val)
    out = np.array(out)
    return out if np.ndim(snr_db) else out[0]


def capacity_siso_rayleigh(snr_db):
    """Closed form E[log2(1 + Gamma |h|^2)] = log2(e) exp(1/Gamma) E_1(1/Gamma), h ~ CN(0,1)."""
    from scipy.special import exp1

    g = _lin(snr_db)
    return np.log2(np.e) * np.exp(1 / g) * exp1(1 / g)


# ---------------------------------------------------------------------------
# Receivers for spatial multiplexing  y = sqrt(E_s/M_T) H x + n
# ---------------------------------------------------------------------------

def transmit_sm(H, x, snr_db, rng=None):
    """Spatial-multiplexing received vectors y = sqrt(E_s/M_T) H x + n, n ~ CN(0, I) (N_0 = 1).

    ``H``: ``(n, M_R, M_T)``, ``x``: ``(n, M_T)`` unit-energy symbols. Returns ``(n, M_R)``.
    """
    rng = np.random.default_rng(rng)
    H = np.asarray(H)
    a = np.sqrt(_lin(snr_db) / H.shape[-1])
    y = a * np.einsum("nij,nj->ni", H, x)
    return y + iid_rayleigh(y.shape, rng)


def zf_equalize(H, y, snr_db):
    """Zero-forcing estimate x_hat = G^dagger y with G = sqrt(E_s/M_T) H, G^dagger = (G^H G)^{-1} G^H."""
    G = np.sqrt(_lin(snr_db) / np.shape(H)[-1]) * np.asarray(H)
    return np.einsum("nij,nj->ni", np.linalg.pinv(G), y)


def mmse_equalize(H, y, snr_db):
    """Linear MMSE estimate x_hat = G^H (G G^H + R_n)^{-1} y with R_n = N_0 I = I, E[x x^H] = I."""
    G = np.sqrt(_lin(snr_db) / np.shape(H)[-1]) * np.asarray(H)
    Gh = np.conj(np.swapaxes(G, -1, -2))
    W = Gh @ np.linalg.inv(G @ Gh + np.eye(G.shape[-2]))
    return np.einsum("nij,nj->ni", W, y)


def ml_detect(H, y, snr_db, points):
    """Maximum-likelihood detection x_hat = argmin_x ||y - G x||, exhaustive over points^M_T.

    Returns symbol indices ``(n, M_T)``.
    """
    H = np.asarray(H)
    m_t = H.shape[-1]
    cand_idx = np.array(list(itertools.product(range(len(points)), repeat=m_t)))  # (K, M_T)
    cand = np.asarray(points)[cand_idx]
    G = np.sqrt(_lin(snr_db) / m_t) * H
    hyp = np.einsum("nij,kj->nki", G, cand)  # (n, K, M_R)
    d = np.sum(np.abs(y[:, None, :] - hyp) ** 2, axis=-1)
    return cand_idx[np.argmin(d, axis=1)]


# ---------------------------------------------------------------------------
# Alamouti space-time block code (2 transmit antennas)
# ---------------------------------------------------------------------------

def alamouti_transmit(H, s, snr_db, rng=None):
    """Send symbol pairs (s1, s2) with the Alamouti code over two slots (lecture "Space-Time Coding").

    Slot 1 sends [s1, s2], slot 2 sends [-s2*, s1*], each scaled by sqrt(E_s/2) so the total
    transmit energy per slot is E_s. ``H``: ``(n, M_R, 2)`` (constant over both slots),
    ``s``: ``(n, 2)``. Returns ``Y`` shaped ``(n, M_R, 2)`` (column = slot), N_0 = 1.
    """
    rng = np.random.default_rng(rng)
    a = np.sqrt(_lin(snr_db) / 2)
    X = np.stack([s, np.stack([-np.conj(s[:, 1]), np.conj(s[:, 0])], axis=-1)], axis=-1)  # (n, 2, 2)
    Y = a * (np.asarray(H) @ X)
    return Y + iid_rayleigh(Y.shape, rng)


def alamouti_combine(H, Y, snr_db):
    """Linear Alamouti combiner. Returns estimates of (s1, s2) normalised so that
    s_hat = s + noise with noise variance 2 / (E_s/N_0 ||H||_F^2) -- full diversity 2 M_R."""
    H = np.asarray(H)
    a = np.sqrt(_lin(snr_db) / 2)
    h1, h2 = H[..., 0], H[..., 1]
    y1, y2 = Y[..., 0], Y[..., 1]
    s1 = np.sum(np.conj(h1) * y1 + h2 * np.conj(y2), axis=-1)
    s2 = np.sum(np.conj(h2) * y1 - h1 * np.conj(y2), axis=-1)
    norm = a * frobenius_sq(H)
    return np.stack([s1, s2], axis=-1) / norm[:, None]


def ber_bpsk_mrc(gamma_bar_db, n_branches):
    """BER of BPSK (or per bit of Gray QPSK) with L-branch MRC in i.i.d. Rayleigh fading:

        P_b = ((1-mu)/2)^L sum_{k=0}^{L-1} C(L-1+k, k) ((1+mu)/2)^k,  mu = sqrt(g/(1+g)),

    where g is the mean per-bit SNR per branch. Also the ZF stream BER with L = M_R - M_T + 1.
    """
    g = _lin(gamma_bar_db)
    mu = np.sqrt(g / (1 + g))
    L = int(n_branches)
    return ((1 - mu) / 2) ** L * sum(comb(L - 1 + k, k) * ((1 + mu) / 2) ** k for k in range(L))
