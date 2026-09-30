"""Small-scale fading channel generators (shared by Weeks 5, 7, 8, 9).

Conventions: time-varying complex baseband gain h(t) with E[|h|^2] = 1 unless
stated otherwise; maximum Doppler frequency f_m = v / lambda (Hz).

Keep the signatures of existing functions stable -- other modules and lecture
notebooks depend on them.
"""

import numpy as np


def rayleigh_sos(fm_hz, t, n_paths=64, rng=None):
    """Rayleigh fading via a sum of sinusoids (Clarke's 2-D isotropic model).

    ``n_paths`` plane waves arrive with random angles theta_n and phases phi_n:
        h(t) = (1/sqrt(n)) * sum_n exp(j(2 pi f_m cos(theta_n) t + phi_n))
    Returns complex array shaped like ``t`` with E[|h|^2] = 1.
    """
    rng = np.random.default_rng(rng)
    t = np.asarray(t, dtype=float)
    theta = rng.uniform(0, 2 * np.pi, n_paths)
    phi = rng.uniform(0, 2 * np.pi, n_paths)
    doppler = fm_hz * np.cos(theta)
    h = np.exp(1j * (2 * np.pi * np.multiply.outer(t, doppler) + phi)).sum(axis=-1)
    return h / np.sqrt(n_paths)


def rician_sos(fm_hz, t, k_factor, los_angle=0.0, n_paths=64, rng=None):
    """Ricean fading: LOS component plus Rayleigh scatter, E[|h|^2] = 1.

    h = sqrt(K/(K+1)) exp(j 2 pi f_m cos(los_angle) t) + sqrt(1/(K+1)) h_Rayleigh
    """
    t = np.asarray(t, dtype=float)
    los = np.exp(1j * 2 * np.pi * fm_hz * np.cos(los_angle) * t)
    scatter = rayleigh_sos(fm_hz, t, n_paths, rng)
    return np.sqrt(k_factor / (k_factor + 1)) * los + np.sqrt(1 / (k_factor + 1)) * scatter


def clarke_gans(fm_hz, n_samples, fs_hz, rng=None):
    """Rayleigh fading via the Clarke/Gans frequency-domain method (Rappaport Fig. 5.24).

    Complex Gaussian noise is shaped in frequency by sqrt(S_E(f)), the U-shaped
    Doppler spectrum S_E(f) = 1 / (pi f_m sqrt(1 - (f/f_m)^2)) for |f| < f_m,
    then transformed to time. Returns ``n_samples`` complex gains sampled at
    ``fs_hz`` with E[|h|^2] = 1.
    """
    rng = np.random.default_rng(rng)
    f = np.fft.fftfreq(n_samples, d=1 / fs_hz)
    inside = np.abs(f) < fm_hz  # strict inequality: the spectrum is infinite at |f| = f_m
    if inside.sum() < 2:
        raise ValueError("fs_hz / n_samples resolution too coarse for fm_hz")
    shape = np.zeros(n_samples)
    shape[inside] = 1.0 / np.sqrt(1.0 - (f[inside] / fm_hz) ** 2)
    g = (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)) / np.sqrt(2)
    h = np.fft.ifft(g * np.sqrt(shape))
    return h / np.sqrt(np.mean(np.abs(h) ** 2))


def iid_rayleigh(shape, rng=None):
    """Independent CN(0,1) gains (block/flat Rayleigh fading, e.g. MIMO H)."""
    rng = np.random.default_rng(rng)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def multipath_impulse_response(delays_s, powers_db, fs_hz, rng=None):
    """Random tapped-delay-line channel sampled at ``fs_hz``.

    Each path gets an independent CN(0, p_k) gain; returns the discrete impulse
    response (length covers the largest delay) normalised to unit total mean power.
    """
    rng = np.random.default_rng(rng)
    delays = np.asarray(delays_s, dtype=float)
    p = 10 ** (np.asarray(powers_db, dtype=float) / 10)
    p = p / p.sum()
    taps = np.zeros(int(np.round(delays.max() * fs_hz)) + 1, dtype=complex)
    gains = iid_rayleigh(len(p), rng) * np.sqrt(p)
    for d, g in zip(delays, gains):
        taps[int(np.round(d * fs_hz))] += g
    return taps
