"""Small-scale fading theory helpers (Week 5 lecture: Doppler, Rayleigh/Ricean, Doppler spectrum).

Channel *generators* live in :mod:`ece512.fading` (``rayleigh_sos``, ``rician_sos``,
``clarke_gans``, ...). This module holds the matching closed-form results so that
simulations can be checked against theory.

Conventions (as in the lecture): envelope r = |h|, average envelope power
Omega_p = E[r^2] = 2 sigma^2 = r_rms^2, maximum Doppler frequency f_m = v / lambda,
Ricean K = A^2 / (2 sigma^2).
"""

import numpy as np
from scipy import optimize, special, stats

C_LIGHT = 299_792_458.0  # speed of light (m/s)


# ---------------------------------------------------------------------------
# Doppler shift
# ---------------------------------------------------------------------------
def max_doppler_hz(speed_mps, fc_hz):
    """Maximum Doppler frequency f_m = v / lambda = f_c v / c (Hz)."""
    return np.asarray(speed_mps, dtype=float) * np.asarray(fc_hz, dtype=float) / C_LIGHT


def doppler_shift_hz(fm_hz, theta_rad):
    """Doppler shift of a plane wave arriving at angle theta to the direction of motion.

    f_{D,n} = f_m cos(theta_n)   (lecture "Doppler Shift" key equation).
    """
    return np.asarray(fm_hz, dtype=float) * np.cos(theta_rad)


# ---------------------------------------------------------------------------
# Multipath phasor sums (central limit theorem demo)
# ---------------------------------------------------------------------------
def random_phasor_sum(n_paths, n_trials, rng=None):
    """Sum of ``n_paths`` equal-amplitude phasors with i.i.d. uniform phases.

    a = (1/sqrt(N)) sum_n exp(-j phi_n), so E[|a|^2] = 1 for every N. As N grows,
    h_I = Re(a) and h_Q = Im(a) become Gaussian (CLT) and |a| becomes Rayleigh.
    Returns ``n_trials`` complex samples.
    """
    rng = np.random.default_rng(rng)
    phi = rng.uniform(0, 2 * np.pi, (n_trials, n_paths))
    return np.exp(-1j * phi).sum(axis=1) / np.sqrt(n_paths)


# ---------------------------------------------------------------------------
# Envelope / power distributions
# ---------------------------------------------------------------------------
def rayleigh_pdf(r, omega_p=1.0):
    """Rayleigh envelope pdf, (2r/Omega_p) exp(-r^2/Omega_p) = (r/sigma^2) exp(-r^2/(2 sigma^2))."""
    r = np.asarray(r, dtype=float)
    return np.where(r >= 0, 2 * r / omega_p * np.exp(-r**2 / omega_p), 0.0)


def rayleigh_cdf(r, omega_p=1.0):
    """Pr(r < r_min) = 1 - exp(-r_min^2 / r_rms^2)   (lecture key equation, r_rms^2 = Omega_p)."""
    r = np.asarray(r, dtype=float)
    return np.where(r >= 0, -np.expm1(-r**2 / omega_p), 0.0)


def exponential_power_pdf(x, omega_p=1.0):
    """pdf of the envelope power r^2 under Rayleigh fading: (1/Omega_p) exp(-x/Omega_p)."""
    x = np.asarray(x, dtype=float)
    return np.where(x >= 0, np.exp(-x / omega_p) / omega_p, 0.0)


def _rice_params(k_factor, omega_p):
    """(b, sigma) for scipy.stats.rice given K and Omega_p = A^2 + 2 sigma^2."""
    sigma = np.sqrt(omega_p / (2 * (k_factor + 1)))
    return np.sqrt(2 * k_factor), sigma  # b = A / sigma = sqrt(2K)


def rician_pdf(r, k_factor, omega_p=1.0):
    """Ricean envelope pdf with K = A^2/(2 sigma^2) and Omega_p = A^2 + 2 sigma^2.

    p(r) = 2(K+1) r / Omega_p * exp(-K - (K+1) r^2/Omega_p) * I_0(2 r sqrt(K(K+1)/Omega_p)).
    K = 0 reduces to the Rayleigh pdf.
    """
    r = np.asarray(r, dtype=float)
    k = float(k_factor)
    x = 2 * r * np.sqrt(k * (k + 1) / omega_p)
    # I0(x) = i0e(x) e^x keeps the product finite for large K.
    p = 2 * (k + 1) * r / omega_p * np.exp(-k - (k + 1) * r**2 / omega_p + x) * special.i0e(x)
    return np.where(r >= 0, p, 0.0)


def rician_cdf(r, k_factor, omega_p=1.0):
    """Ricean envelope CDF Pr(r < r_min) = 1 - Q_1(sqrt(2K), r_min sqrt(2(K+1)/Omega_p))."""
    b, sigma = _rice_params(k_factor, omega_p)
    return stats.rice.cdf(np.asarray(r, dtype=float), b, scale=sigma)


# ---------------------------------------------------------------------------
# Fade margin and outage
# ---------------------------------------------------------------------------
def rayleigh_outage_probability(margin_db):
    """Outage probability for fade margin M = r_rms^2 / r_min^2 (dB) under Rayleigh fading.

    Pr(r < r_min) = 1 - exp(-1/M).
    """
    m = 10 ** (np.asarray(margin_db, dtype=float) / 10)
    return -np.expm1(-1 / m)


def rayleigh_fade_margin_db(outage):
    """Fade margin (dB) giving outage probability ``outage`` under Rayleigh fading.

    Inverting the lecture example: r_min^2/r_rms^2 = -ln(1 - P_out), so
    M_dB = -10 log10(-ln(1 - P_out)); P_out = 1 % gives 19.98 dB ~ 20 dB.
    """
    p = np.asarray(outage, dtype=float)
    return -10 * np.log10(-np.log1p(-p))


def rician_outage_probability(margin_db, k_factor):
    """Outage probability for fade margin ``margin_db`` under Ricean fading with factor K."""
    r_min = 10 ** (-np.asarray(margin_db, dtype=float) / 20)  # r_min / r_rms with Omega_p = 1
    return rician_cdf(r_min, k_factor)


def rician_fade_margin_db(outage, k_factor):
    """Fade margin (dB) giving outage probability ``outage`` under Ricean fading (numeric)."""
    f = lambda m_db: np.log(rician_outage_probability(m_db, k_factor)) - np.log(outage)
    return optimize.brentq(f, -20.0, 80.0)


def sir_outage_rayleigh(threshold_db, mean_sir_db):
    """Pr(Lambda < Lambda_min) when desired signal and one interferer both undergo Rayleigh fading.

    The ratio of two independent exponentials gives
    Pr = Lambda_min / (Lambda_min + mean SIR) = 1 - mean/(mean + Lambda_min)
    (lecture: 0 dB threshold, 10 dB mean -> 1/11 ~ 0.09).
    """
    x = 10 ** (np.asarray(threshold_db, dtype=float) / 10)
    s = 10 ** (np.asarray(mean_sir_db, dtype=float) / 10)
    return x / (x + s)


# ---------------------------------------------------------------------------
# Doppler spectrum and autocorrelation (Clarke's isotropic model)
# ---------------------------------------------------------------------------
def doppler_psd(f_hz, fm_hz, omega_p=1.0, antenna_gain=1.0):
    """Clarke/Jakes U-shaped Doppler power spectrum (baseband, f relative to f_c).

    S(f) = G Omega_p / (pi f_m sqrt(1 - (f/f_m)^2)) for |f| < f_m, else 0.
    With ``antenna_gain=1.5`` (vertical dipole) and Omega_p = 1 this is the lecture's
    key equation S(f) = 1.5 / (pi f_m sqrt(1 - ((f - f_c)/f_m)^2)). The integral over f
    is G * Omega_p.
    """
    f = np.asarray(f_hz, dtype=float)
    u = f / fm_hz
    out = np.zeros_like(u)
    inside = np.abs(u) < 1
    out[inside] = antenna_gain * omega_p / (np.pi * fm_hz * np.sqrt(1 - u[inside] ** 2))
    return out


def clarke_autocorrelation(tau_s, fm_hz, omega_p=1.0):
    """Autocorrelation of the complex gain, R(tau) = E[h(t) h*(t+tau)] = Omega_p J_0(2 pi f_m tau).

    It is the inverse Fourier transform of :func:`doppler_psd` (with G = 1).
    """
    return omega_p * special.j0(2 * np.pi * fm_hz * np.asarray(tau_s, dtype=float))


def empirical_autocorrelation(h, max_lag):
    """Normalised sample autocorrelation R[k] = mean(h[n] h*[n+k]) / mean(|h|^2), k = 0..max_lag."""
    h = np.asarray(h)
    n = len(h)
    r = np.array([np.vdot(h[: n - k], h[k:]) / (n - k) for k in range(max_lag + 1)])
    return r / r[0].real


# ---------------------------------------------------------------------------
# Flat vs frequency-selective fading
# ---------------------------------------------------------------------------
def time_frequency_response(amplitudes, delays_s, doppler_hz, phases, t_s, f_hz):
    """Time-varying channel frequency response T(t, f) of a discrete multipath channel.

    T(t, f) = sum_n C_n exp(j(2 pi f_{D,n} t - phi_n)) exp(-j 2 pi f tau_n),
    i.e. the Fourier transform over delay of h(t, tau) = sum_n C_n e^{-j phi_n(t)} delta(tau - tau_n).
    Returns an array of shape (len(t_s), len(f_hz)). If all tau_n are (nearly) equal,
    |T(t, f)| = |h_FLAT(t)| does not depend on f: flat fading.
    """
    c = np.asarray(amplitudes, dtype=complex)
    tau = np.asarray(delays_s, dtype=float)
    fd = np.asarray(doppler_hz, dtype=float)
    ph = np.asarray(phases, dtype=float)
    t = np.atleast_1d(np.asarray(t_s, dtype=float))
    f = np.atleast_1d(np.asarray(f_hz, dtype=float))
    time_part = c * np.exp(1j * (2 * np.pi * np.multiply.outer(t, fd) - ph))  # (T, N)
    freq_part = np.exp(-2j * np.pi * np.multiply.outer(tau, f))  # (N, F)
    return time_part @ freq_part


def rms_delay_spread(delays_s, powers):
    """RMS delay spread sigma_tau = sqrt(mean(tau^2) - mean(tau)^2) of a power delay profile (linear powers)."""
    p = np.asarray(powers, dtype=float)
    tau = np.asarray(delays_s, dtype=float)
    p = p / p.sum()
    mean = np.sum(p * tau)
    return np.sqrt(np.sum(p * tau**2) - mean**2)
