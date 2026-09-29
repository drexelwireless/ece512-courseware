"""Path loss, co-channel interference, noise, and link budgets (Week 2 lecture).

Equations follow the Week 2 deck (``lectures/week02-interference-linkbudget``):

* Log-distance path loss with lognormal shadowing
      P_RX|dB(d) = E[P_RX|dB(d0)] - 10 beta log10(d/d0) + eps,  eps ~ N(0, sigma^2)
* Downlink SIR  Lambda = R^-beta / sum_i D_i^-beta  (worst case, first tier)
      Assumption #1:  Lambda = Q^beta / i0,  Q = D/R = sqrt(3N)
      Assumption #2:  Lambda = 1 / (2[(Q-1)^-b + Q^-b + (Q+1)^-b])
      120 deg sectors: Lambda = 1 / [(Q+0.7)^-b + Q^-b];  60 deg: Lambda = (Q+0.7)^b
* Thermal noise N = k T B F;  sensitivity S_rx = kTB + F + (SNR)_min  (dB)
* Maximum allowable path loss
      L_PATH,MAX = P_T + G_T + G_R - S_RX - M_SHAD - L_I + G_HO
* Shadow margin: edge outage  P_EDGE = Q(M_SHAD / sigma),
      area outage  P_AREA = Q(X) - exp(XY + Y^2/2) Q(X+Y),
      X = M_SHAD/sigma,  Y = 2 sigma ln(10) / (10 beta)
* Coverage: N1/N2 = 10^(-2 (L1-L2) / (10 beta))

All powers in dB/dBm unless the name says ``_lin``; SIRs from the ``sir_*``
functions are linear power ratios (convert with ``units.linear_to_db``).
"""

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from .cells import cochannel_offsets
from .hexgrid import hex_vertices

BOLTZMANN = 1.380649e-23  # J/K


def qfunc(x):
    """Gaussian tail Q(x) = Pr(Z > x), Z ~ N(0, 1)."""
    return norm.sf(x)


def qfuncinv(p):
    """Inverse Gaussian tail: Q^{-1}(p)."""
    return norm.isf(p)


# ---------------------------------------------------------------------------
# Log-distance path loss and lognormal shadowing
# ---------------------------------------------------------------------------

def log_distance_rx_dbm(d, p0_dbm, beta, d0=1.0):
    """Mean received power  mu(d) = E[P_RX|dB(d0)] - 10 beta log10(d/d0)  (dBm)."""
    return p0_dbm - 10.0 * beta * np.log10(np.asarray(d, dtype=float) / d0)


def shadowed_rx_dbm(d, p0_dbm, beta, sigma_db, d0=1.0, rng=None):
    """Simulated measurements: log-distance mean plus eps ~ N(0, sigma^2) in dB."""
    rng = np.random.default_rng(rng)
    mu = log_distance_rx_dbm(d, p0_dbm, beta, d0)
    return mu + sigma_db * rng.standard_normal(np.shape(mu))


def fit_log_distance(d, p_dbm, d0=1.0):
    """Least-squares fit of the log-distance model to measurements.

    Regresses P_RX|dB on x = 10 log10(d/d0).  Returns ``(p0_hat, beta_hat,
    sigma_hat)`` where sigma_hat is the RMS residual (n-2 degrees of freedom),
    the estimate of the shadow standard deviation.
    """
    x = 10.0 * np.log10(np.asarray(d, dtype=float) / d0)
    y = np.asarray(p_dbm, dtype=float)
    A = np.column_stack([np.ones_like(x), -x])
    (p0, beta), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ np.array([p0, beta])
    sigma = np.sqrt(np.sum(resid**2) / (len(y) - 2))
    return p0, beta, sigma


# ---------------------------------------------------------------------------
# Co-channel interference (downlink SIR)
# ---------------------------------------------------------------------------

def reuse_ratio(n_cluster):
    """Co-channel reuse ratio Q = D/R = sqrt(3N)."""
    return np.sqrt(3.0 * np.asarray(n_cluster, dtype=float))


def sir_equal_distance(n_cluster, beta, i0=6):
    """Assumption #1 (all interferers at D): Lambda = Q^beta / i0 (linear)."""
    return reuse_ratio(n_cluster) ** beta / i0


def cluster_size_for_sir(sir_db, beta, i0=6):
    """Real-valued N solving Q^beta / i0 = Lambda (Assumption #1)."""
    q = (i0 * 10.0 ** (np.asarray(sir_db) / 10.0)) ** (1.0 / beta)
    return q**2 / 3.0


def sir_assumption2(n_cluster, beta):
    """Assumption #2: two interferers each at D-R, D, D+R (linear)."""
    q = reuse_ratio(n_cluster)
    return 0.5 / ((q - 1) ** -beta + q**-beta + (q + 1) ** -beta)


def sir_sector120(n_cluster, beta):
    """120-degree sectoring worst case, i0 = 2 at D+0.7R and D (linear)."""
    q = reuse_ratio(n_cluster)
    return 1.0 / ((q + 0.7) ** -beta + q**-beta)


def sir_sector60(n_cluster, beta):
    """60-degree sectoring worst case, i0 = 1 at D+0.7R (linear)."""
    return (reuse_ratio(n_cluster) + 0.7) ** beta


def _in_sector(angle, k, n_sectors, orientation):
    """True where ``angle`` (rad) lies in sector k of n equal sectors."""
    width = 2 * np.pi / n_sectors
    return np.floor(((angle - orientation) % (2 * np.pi)) / width).astype(int) == k


def sir_map(z, i, j, radius, beta, n_sectors=1, orientation=0.0, n_tiers=1):
    """Downlink SIR (linear) at mobile positions ``z`` (complex) in the home cell at 0.

    Signal Lambda = |z|^-beta / sum_i |z - c_i|^-beta over co-channel base stations
    c_i (``n_tiers`` rings of the co-channel lattice).  With ``n_sectors`` > 1 the
    mobile is served by the ideal sector (unit gain inside, zero outside)
    containing it, and a co-channel base station interferes only if its
    same-index sector points at the mobile.
    """
    z = np.asarray(z, dtype=complex)
    bss = cochannel_bs(i, j, radius, n_tiers)
    with np.errstate(divide="ignore"):
        s = np.abs(z) ** -beta
    interference = np.zeros(z.shape)
    if n_sectors == 1:
        for c in bss:
            interference += np.abs(z - c) ** -beta
    else:
        width = 2 * np.pi / n_sectors
        k = np.floor(((np.angle(z) - orientation) % (2 * np.pi)) / width).astype(int)
        for c in bss:
            hit = _in_sector(np.angle(z - c), k, n_sectors, orientation)
            interference += hit * np.abs(z - c) ** -beta
    with np.errstate(divide="ignore"):
        return s / interference


def cochannel_bs(i, j, radius, n_tiers=1):
    """Co-channel base-station positions (excluding home cell at 0), ``n_tiers`` rings."""
    u = cochannel_offsets(i, j, radius)
    u1, u2 = u[0], u[1]
    pts = []
    for m in range(-n_tiers, n_tiers + 1):
        for n in range(-n_tiers, n_tiers + 1):
            # hex-lattice "ring" index of m u1 + n u2 (u2 = u1 rotated 60 deg)
            ring = max(abs(m), abs(n), abs(m + n))
            if 0 < ring <= n_tiers:
                pts.append(m * u1 + n * u2)
    return np.array(pts)


def worst_case_sir_geometric(i, j, radius, beta, n_sectors=1, orientation=0.0,
                             n_tiers=1, n_edge=60):
    """Minimum SIR (linear) over the boundary of the home hexagon (edges + corners)."""
    v = hex_vertices(0j, radius * (1 - 1e-9))
    t = np.linspace(0, 1, n_edge, endpoint=False)
    boundary = np.concatenate([v[k] + t * (v[k + 1] - v[k]) for k in range(6)])
    return np.min(sir_map(boundary, i, j, radius, beta, n_sectors, orientation, n_tiers))


# ---------------------------------------------------------------------------
# Noise and link budgets
# ---------------------------------------------------------------------------

def thermal_noise_dbm(bandwidth_hz, temp_k=290.0):
    """Thermal noise power 10 log10(k T B) in dBm."""
    return 10.0 * np.log10(BOLTZMANN * temp_k * np.asarray(bandwidth_hz, dtype=float)) + 30.0


def noise_psd_dbm_hz(temp_k=290.0):
    """Noise PSD N_0 = k T in dBm/Hz (-174 dBm/Hz at 290 K)."""
    return thermal_noise_dbm(1.0, temp_k)


def cascade_noise_figure_db(nf_db, gain_db):
    """Friis cascade: F = F1 + (F2-1)/G1 + (F3-1)/(G1 G2) + ...  (dB in, dB out).

    ``nf_db`` and ``gain_db`` list the stages in signal order (the last gain is unused).
    """
    f = 10.0 ** (np.asarray(nf_db, dtype=float) / 10.0)
    g = 10.0 ** (np.asarray(gain_db, dtype=float) / 10.0)
    total, g_cum = f[0], 1.0
    for k in range(1, len(f)):
        g_cum *= g[k - 1]
        total += (f[k] - 1.0) / g_cum
    return 10.0 * np.log10(total)


def sensitivity_dbm(bandwidth_hz, noise_figure_db, snr_min_db, temp_k=290.0):
    """Minimum received power  S_rx = kTB + F + SNR_min  (dBm)."""
    return thermal_noise_dbm(bandwidth_hz, temp_k) + noise_figure_db + snr_min_db


def fspl_db(distance_m, freq_hz):
    """Free-space path loss 20 log10(4 pi d / lambda) (dB).

    Equivalent to 20 log10(d_km) + 20 log10(f_GHz) + 92.45.
    """
    from .propagation import free_space_path_loss_db
    return free_space_path_loss_db(distance_m, freq_hz)


def fspl_db_practical(distance_m, freq_hz):
    """Lecture form FSPL = 20 log10(R_km) + 20 log10(f_GHz) + 92.45 (dB).

    Same as :func:`fspl_db` up to the rounded constant (exact: 92.4478).
    """
    return (20.0 * np.log10(np.asarray(distance_m) / 1e3)
            + 20.0 * np.log10(np.asarray(freq_hz) / 1e9) + 92.45)


def received_power_dbm(ptx_dbm, gains_db=(), losses_db=()):
    """Link budget sum  P_RX = P_TX + sum(gains) - sum(losses)  (dBm)."""
    return ptx_dbm + np.sum(gains_db) - np.sum(losses_db)


def max_path_loss_db(ptx_dbm, gt_db, gr_db, srx_dbm, m_shad_db=0.0, l_i_db=0.0, g_ho_db=0.0):
    """Revised maximum allowable path loss
    L_PATH,MAX = P_T + G_T + G_R - S_RX - M_SHAD - L_I + G_HO  (dB)."""
    return ptx_dbm + gt_db + gr_db - srx_dbm - m_shad_db - l_i_db + g_ho_db


def max_range(l_max_db, p0_loss_db, beta, d0=1.0):
    """Distance at which a log-distance loss L(d0) + 10 beta log10(d/d0) hits L_MAX."""
    return d0 * 10.0 ** ((l_max_db - p0_loss_db) / (10.0 * beta))


def cell_count_ratio(delta_l_db, beta):
    """N1/N2 = 10^(-2 (L1-L2) / (10 beta)): cell sites needed, system 1 vs 2."""
    return 10.0 ** (-2.0 * np.asarray(delta_l_db) / (10.0 * beta))


# ---------------------------------------------------------------------------
# Shadow margin, area outage, handoff gain
# ---------------------------------------------------------------------------

def edge_outage(m_shad_db, sigma_db):
    """P_EDGE = Q(M_SHAD / sigma)."""
    return qfunc(np.asarray(m_shad_db) / sigma_db)


def shadow_margin_db(p_edge, sigma_db):
    """M_SHAD = sigma Q^{-1}(P_EDGE)."""
    return sigma_db * qfuncinv(p_edge)


def area_outage(m_shad_db, sigma_db, beta):
    """P_AREA = Q(X) - exp(XY + Y^2/2) Q(X+Y), X = M/sigma, Y = 2 sigma ln10 / (10 beta).

    Users uniform over a circular cell of radius R; margin M_SHAD defined at r = R.
    """
    x = np.asarray(m_shad_db, dtype=float) / sigma_db
    y = 2.0 * sigma_db * np.log(10.0) / (10.0 * beta)
    # exp(.)Q(.) evaluated in log domain for numerical stability
    return qfunc(x) - np.exp(x * y + y**2 / 2.0 + norm.logsf(x + y))


def simulate_area_outage(m_shad_db, sigma_db, beta, n_users=100_000, rng=None):
    """Monte Carlo area outage: users uniform in a disk, independent shadowing."""
    rng = np.random.default_rng(rng)
    r = np.sqrt(rng.random(n_users))            # r/R, uniform over area
    mu_minus_th = m_shad_db - 10.0 * beta * np.log10(r)   # mu(r) - P_TH
    eps = sigma_db * rng.standard_normal(n_users)
    return np.mean(mu_minus_th + eps < 0)


def area_outage_margin_db(p_area, sigma_db, beta):
    """Shadow margin M_SHAD giving area outage ``p_area``."""
    return brentq(lambda m: area_outage(m, sigma_db, beta) - p_area, -50, 100)


def handoff_edge_outage(m_db, sigma_db, n_bs=2, rho=0.5):
    """Edge outage when the mobile may use the best of ``n_bs`` equidistant BSs.

    Outage only if all links are below threshold.  Shadowing on the links is
    Gaussian with pairwise correlation ``rho`` (common + independent parts):
    eps_k = sigma (sqrt(rho) a + sqrt(1-rho) b_k).
        Pr = int phi(a) [1 - Q((-M/sigma - sqrt(rho) a)/sqrt(1-rho))]^n da
    """
    x = np.asarray(m_db, dtype=float)[..., None] / sigma_db
    if rho >= 1.0:
        return qfunc(x[..., 0])
    a, w = np.polynomial.hermite_e.hermegauss(80)
    w = w / np.sqrt(2 * np.pi)
    inner = norm.cdf((-x - np.sqrt(rho) * a) / np.sqrt(1.0 - rho))
    return np.sum(w * inner**n_bs, axis=-1)


def handoff_margin_db(p_edge, sigma_db, n_bs=2, rho=0.5):
    """Margin M such that the best-of-n_bs edge outage equals ``p_edge``."""
    return brentq(lambda m: handoff_edge_outage(m, sigma_db, n_bs, rho) - p_edge, -50, 100)


def handoff_gain_db(p_edge, sigma_db, n_bs=2, rho=0.5):
    """G_HO = M_SHAD(single cell) - M(with handoff to best of n_bs)."""
    return shadow_margin_db(p_edge, sigma_db) - handoff_margin_db(p_edge, sigma_db, n_bs, rho)


def simulate_handoff_edge_outage(m_db, sigma_db, n_bs=2, rho=0.5, n_trials=100_000, rng=None):
    """Monte Carlo counterpart of :func:`handoff_edge_outage`."""
    rng = np.random.default_rng(rng)
    common = rng.standard_normal((n_trials, 1))
    indep = rng.standard_normal((n_trials, n_bs))
    eps = sigma_db * (np.sqrt(rho) * common + np.sqrt(1 - rho) * indep)
    return np.mean(np.max(m_db + eps, axis=1) < 0)
