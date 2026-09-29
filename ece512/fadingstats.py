"""Second-order fading statistics, channel dispersion and error rates in fading (Week 7 lecture).

Sections follow the Week 7 deck:

* Level crossing rate (LCR) and average fade duration (AFD) for Rayleigh and Ricean
  fading with 2-D isotropic scattering, plus estimators that measure them from a
  sampled envelope.
* Multipath dispersion parameters: mean excess delay, RMS delay spread, coherence
  bandwidth and coherence time.
* Fading channel classification (flat / frequency selective, slow / fast).
* Modulation performance in flat fading: P_e = int P_e(gamma) p(gamma) d gamma.

Conventions (as in the lecture): envelope r = |h|, Omega_p = E[r^2] = R_rms^2,
normalised level rho = R / sqrt(Omega_p), maximum Doppler frequency f_m (Hz),
Ricean K = s^2 / (2 sigma^2). Channel *generators* live in :mod:`ece512.fading`.
"""

import numpy as np
from scipy import integrate, special, stats

from .modulation import qfunc

# ---------------------------------------------------------------------------
# Level crossing rate and average fade duration (theory)
# ---------------------------------------------------------------------------


def marcum_q1(a, b):
    """First-order Marcum Q function Q_1(a, b).

    Q_1(a, b) = Pr(X > b^2) for X noncentral chi-square with 2 degrees of freedom
    and noncentrality a^2 (used in the Ricean envelope CDF).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    safe_a = np.where(a > 0, a, 1.0)  # ncx2 needs a positive noncentrality
    return np.where(a > 0, stats.ncx2.sf(b**2, 2, safe_a**2), np.exp(-(b**2) / 2))


def envelope_cdf(rho, k_factor=0.0):
    """Pr(r <= R) for a Ricean envelope, rho = R / sqrt(Omega_p) (lecture AFD slides).

    Pr(r <= R) = 1 - Q_1( sqrt(2K), sqrt(2(K+1)) rho ); K = 0 gives 1 - exp(-rho^2).
    """
    rho = np.asarray(rho, dtype=float)
    if k_factor == 0:
        return -np.expm1(-(rho**2))
    return 1.0 - marcum_q1(np.sqrt(2 * k_factor), np.sqrt(2 * (k_factor + 1)) * rho)


def lcr_rician(rho, fm_hz, k_factor=0.0):
    """Ricean level crossing rate (crossings/s, positive-going), lecture key equation:

    LCR(R) = sqrt(2 pi (K+1)) f_m rho exp(-K - (K+1) rho^2) I_0(2 rho sqrt(K(K+1)))

    with rho = R / sqrt(Omega_p). Assumes 2-D isotropic scattering and a LOS
    component at zero Doppler offset (f_s = f_c).
    """
    rho = np.asarray(rho, dtype=float)
    k = float(k_factor)
    x = 2 * rho * np.sqrt(k * (k + 1))
    # I0(x) exp(-...) evaluated as i0e(x) exp(x - ...) for numerical stability
    return np.sqrt(2 * np.pi * (k + 1)) * fm_hz * rho * special.i0e(x) * np.exp(x - k - (k + 1) * rho**2)


def lcr_rayleigh(rho, fm_hz):
    """Rayleigh level crossing rate: LCR(R) = sqrt(2 pi) f_m rho exp(-rho^2)."""
    rho = np.asarray(rho, dtype=float)
    return np.sqrt(2 * np.pi) * fm_hz * rho * np.exp(-(rho**2))


def afd_rician(rho, fm_hz, k_factor=0.0):
    """Ricean average fade duration (s): AFD(R) = Pr(r <= R) / LCR(R)."""
    return envelope_cdf(rho, k_factor) / lcr_rician(rho, fm_hz, k_factor)


def afd_rayleigh(rho, fm_hz):
    """Rayleigh average fade duration: AFD(R) = (exp(rho^2) - 1) / (rho f_m sqrt(2 pi))."""
    rho = np.asarray(rho, dtype=float)
    return np.expm1(rho**2) / (rho * fm_hz * np.sqrt(2 * np.pi))


# ---------------------------------------------------------------------------
# LCR / AFD measured from a sampled envelope
# ---------------------------------------------------------------------------


def normalized_envelope(h):
    """Envelope |h| normalised to unit RMS, i.e. rho(t) = r(t) / sqrt(mean(r^2))."""
    r = np.abs(np.asarray(h))
    return r / np.sqrt(np.mean(r**2))


def count_crossings(r, level):
    """Number of positive-going crossings of ``level`` (r[n-1] < level <= r[n])."""
    r = np.asarray(r)
    return int(np.count_nonzero((r[:-1] < level) & (r[1:] >= level)))


def fade_durations(r, level, fs_hz):
    """Durations (s) of the complete fades of ``r`` below ``level``.

    A fade starts with a negative-going crossing and ends with the next
    positive-going crossing; fades cut off by the ends of the record are dropped.
    Crossing instants are linearly interpolated between samples, so fades shorter
    than a few samples are still measured without a large quantisation bias.
    """
    r = np.asarray(r, dtype=float)
    below = r < level
    edges = np.diff(below.astype(np.int8))
    down = np.flatnonzero(edges == 1)  # r[i] >= level > r[i+1]
    up = np.flatnonzero(edges == -1)  # r[i] < level <= r[i+1]
    if below[0]:
        up = up[1:]
    n = min(len(down), len(up))
    down, up = down[:n], up[:n]

    def t_cross(i):
        return i + (level - r[i]) / (r[i + 1] - r[i])

    return (t_cross(up) - t_cross(down)) / fs_hz


def measured_lcr(r, levels, fs_hz):
    """Measured LCR (crossings/s) of the envelope ``r`` at each value in ``levels``."""
    duration = len(r) / fs_hz
    return np.array([count_crossings(r, lv) for lv in np.atleast_1d(levels)]) / duration


def measured_afd(r, levels, fs_hz):
    """Measured AFD (s): mean duration of the complete fades below each level (NaN if none)."""
    out = []
    for lv in np.atleast_1d(levels):
        d = fade_durations(r, lv, fs_hz)
        out.append(d.mean() if d.size else np.nan)
    return np.array(out)


# ---------------------------------------------------------------------------
# Multipath dispersion parameters
# ---------------------------------------------------------------------------


def delay_spread_parameters(delays_s, powers, powers_in_db=False):
    """Mean excess delay, RMS delay spread and maximum excess delay of a power delay profile.

    tau_bar = sum(C_k^2 tau_k) / sum(C_k^2),  tau2_bar = sum(C_k^2 tau_k^2) / sum(C_k^2),
    sigma_tau = sqrt(tau2_bar - tau_bar^2)   (lecture "Time Dispersion" slide),
    with delays measured relative to the first arriving path.

    Returns a dict with keys ``mean_excess_delay``, ``rms_delay_spread``,
    ``max_excess_delay`` (seconds).
    """
    tau = np.asarray(delays_s, dtype=float)
    p = np.asarray(powers, dtype=float)
    if powers_in_db:
        p = 10 ** (p / 10)
    tau = tau - tau.min()
    mean = np.sum(p * tau) / np.sum(p)
    second = np.sum(p * tau**2) / np.sum(p)
    return {
        "mean_excess_delay": mean,
        "rms_delay_spread": np.sqrt(max(second - mean**2, 0.0)),
        "max_excess_delay": tau.max(),
    }


def coherence_bandwidth(sigma_tau_s, correlation=0.5):
    """Coherence bandwidth rule of thumb B_c ~ 1 / (c sigma_tau) (Rappaport Sec. 5.4.2).

    correlation = 0.9 -> B_c ~ 1 / (50 sigma_tau); correlation = 0.5 -> B_c ~ 1 / (5 sigma_tau).
    """
    factor = {0.9: 50.0, 0.5: 5.0}.get(correlation)
    if factor is None:
        raise ValueError("correlation must be 0.5 or 0.9")
    return 1.0 / (factor * np.asarray(sigma_tau_s, dtype=float))


def coherence_time(fm_hz, definition="geometric"):
    """Coherence time T_c ~ 1 / f_m (Rappaport Sec. 5.4.3).

    ``"0.5"``: T_c = 9 / (16 pi f_m) (time correlation above 0.5);
    ``"geometric"``: geometric mean of that and 1/f_m, T_c = 0.423 / f_m (popular rule of thumb).
    """
    fm = np.asarray(fm_hz, dtype=float)
    if definition == "0.5":
        return 9.0 / (16 * np.pi * fm)
    if definition == "geometric":
        return np.sqrt(9.0 / (16 * np.pi)) / fm
    raise ValueError("definition must be 'geometric' or '0.5'")


def frequency_response(delays_s, gains, f_hz):
    """H(f) = sum_k a_k exp(-j 2 pi f tau_k) of a discrete multipath channel."""
    tau = np.asarray(delays_s, dtype=float)
    a = np.asarray(gains)
    return np.exp(-2j * np.pi * np.multiply.outer(np.asarray(f_hz, dtype=float), tau)) @ a


def frequency_correlation(delays_s, powers, delta_f_hz):
    """Normalised spaced-frequency correlation |R(Delta f)| of a WSSUS channel with the given PDP.

    R(Delta f) = sum_k P_k exp(-j 2 pi Delta f tau_k) / sum_k P_k (Fourier transform of the PDP).
    """
    p = np.asarray(powers, dtype=float)
    return np.abs(frequency_response(delays_s, p / p.sum(), delta_f_hz))


def time_correlation(delta_t_s, fm_hz):
    """Clarke time correlation of the complex gain: J_0(2 pi f_m Delta t)."""
    return special.j0(2 * np.pi * fm_hz * np.asarray(delta_t_s, dtype=float))


def classify_channel(signal_bw_hz, bc_hz, symbol_period_s, tc_s):
    """Rappaport classification: returns (``"flat"``|``"frequency selective"``, ``"slow"``|``"fast"``).

    Flat if B_s < B_c, else frequency selective; slow if T_s < T_c, else fast.
    """
    disp = "flat" if signal_bw_hz < bc_hz else "frequency selective"
    time = "slow" if symbol_period_s < tc_s else "fast"
    return disp, time


# ---------------------------------------------------------------------------
# Modulation performance in flat fading
# ---------------------------------------------------------------------------

# Nearest-neighbour Gray-coded approximations P_b(gamma_b) ~ a Q(sqrt(b gamma_b)).
_QFORM = {
    "bpsk": (1.0, 2.0),
    "qpsk": (1.0, 2.0),
}


def _q_coefficients(scheme):
    scheme = scheme.lower()
    if scheme in _QFORM:
        return _QFORM[scheme]
    if scheme.endswith("qam"):
        m = int(scheme[:-3])
        k = np.log2(m)
        return 4 / k * (1 - 1 / np.sqrt(m)), 3 * k / (m - 1)
    raise ValueError(f"unknown scheme {scheme!r}; use 'bpsk', 'qpsk' or 'Mqam' (e.g. '16qam')")


def ber_awgn(ebno_db, scheme="bpsk"):
    """Bit error rate in AWGN vs E_b/N_0 (dB): P_b = a Q(sqrt(b E_b/N_0)).

    Exact for BPSK/QPSK; nearest-neighbour approximation for square M-QAM
    (a = 4/k (1 - 1/sqrt(M)), b = 3k/(M-1), k = log2 M).
    """
    a, b = _q_coefficients(scheme)
    g = 10 ** (np.asarray(ebno_db, dtype=float) / 10)
    return a * qfunc(np.sqrt(b * g))


def ber_rayleigh(ebno_db, scheme="bpsk"):
    """Bit error rate in flat Rayleigh fading vs mean E_b/N_0 = Gamma (dB).

    Averaging a Q(sqrt(b gamma)) over p(gamma) = exp(-gamma/Gamma)/Gamma gives
    P_b = (a/2) (1 - sqrt((b Gamma/2) / (1 + b Gamma/2))); BPSK: 0.5 (1 - sqrt(Gamma/(1+Gamma))).
    """
    a, b = _q_coefficients(scheme)
    g = 10 ** (np.asarray(ebno_db, dtype=float) / 10)
    x = b * g / 2
    return a / 2 * (1 - np.sqrt(x / (1 + x)))


def snr_pdf(gamma, gamma_bar, k_factor=0.0):
    """Density p(gamma) of the instantaneous SNR gamma = alpha^2 E_b/N_0 with mean Gamma.

    Rayleigh (K = 0): exp(-gamma/Gamma)/Gamma. Ricean:
    (1+K) e^{-K} / Gamma exp(-(1+K) gamma/Gamma) I_0(2 sqrt(K (1+K) gamma / Gamma)).
    """
    g = np.asarray(gamma, dtype=float)
    k = float(k_factor)
    x = 2 * np.sqrt(k * (1 + k) * g / gamma_bar)
    return (1 + k) / gamma_bar * special.i0e(x) * np.exp(x - k - (1 + k) * g / gamma_bar)


def average_error_probability(pe_of_gamma, gamma_bar_db, k_factor=0.0):
    """Numerically evaluate P_e = int_0^inf P_e(gamma) p(gamma) d gamma (lecture key equation).

    ``pe_of_gamma`` is the conditional error probability as a function of the linear
    instantaneous SNR; ``gamma_bar_db`` may be an array of mean SNRs (dB).
    """
    out = []
    for gb in np.atleast_1d(10 ** (np.asarray(gamma_bar_db, dtype=float) / 10)):
        # p(gamma) is negligible beyond 50 Gamma; the break point at Gamma keeps quad
        # from missing the narrow peak of p(gamma) when K is large
        val, _ = integrate.quad(lambda g: pe_of_gamma(g) * snr_pdf(g, gb, k_factor), 0, 50 * gb,
                                points=[gb], limit=400)
        out.append(val)
    return np.array(out) if np.ndim(gamma_bar_db) else out[0]
