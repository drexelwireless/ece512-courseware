"""Digital modulation building blocks (shared by Weeks 7, 8, 9).

Constellations have unit average symbol energy. Keep existing signatures stable.
"""

import numpy as np
from scipy.special import erfc


def qfunc(x):
    """Gaussian tail probability Q(x) = Pr(Z > x), Z ~ N(0, 1)."""
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2))


def psk_constellation(m):
    """M-PSK points (Gray-ordered labels 0..M-1), unit energy."""
    k = np.arange(m)
    gray = k ^ (k >> 1)
    pts = np.exp(1j * (2 * np.pi * k / m + (np.pi / 4 if m == 4 else 0)))
    out = np.empty(m, dtype=complex)
    out[gray] = pts
    return out


def qam_constellation(m):
    """Square M-QAM points (Gray-coded per axis), unit average energy."""
    side = int(round(np.sqrt(m)))
    if side * side != m:
        raise ValueError("square QAM requires M = 4, 16, 64, ...")
    levels = np.arange(-(side - 1), side, 2)
    g = np.arange(side) ^ (np.arange(side) >> 1)
    axis = np.empty(side)
    axis[g] = levels
    pts = (axis[:, None] + 1j * axis[None, :]).ravel()  # label = I_label * side + Q_label
    return pts / np.sqrt(np.mean(np.abs(pts) ** 2))


def constellation(name, m):
    """``constellation("psk", 4)`` / ``constellation("qam", 16)``; BPSK = psk 2."""
    return psk_constellation(m) if name.lower() == "psk" else qam_constellation(m)


def random_symbols(points, n, rng=None):
    """Draw ``n`` random symbol indices and return (indices, symbols)."""
    rng = np.random.default_rng(rng)
    idx = rng.integers(0, len(points), n)
    return idx, points[idx]


def awgn(x, snr_db, rng=None):
    """Add complex white Gaussian noise for average SNR (per symbol) ``snr_db``,
    assuming unit-energy symbols."""
    rng = np.random.default_rng(rng)
    n0 = 10 ** (-np.asarray(snr_db) / 10)
    noise = (rng.standard_normal(np.shape(x)) + 1j * rng.standard_normal(np.shape(x))) * np.sqrt(n0 / 2)
    return x + noise


def detect(y, points):
    """Minimum-distance (ML for AWGN) detection; returns symbol indices."""
    y = np.asarray(y)
    return np.argmin(np.abs(y[..., None] - points), axis=-1)


def symbol_error_rate(tx_idx, rx_idx):
    return float(np.mean(np.asarray(tx_idx) != np.asarray(rx_idx)))


def bits_per_symbol(points):
    return int(np.log2(len(points)))


def bit_error_rate(tx_idx, rx_idx, m):
    """Bit error rate between label sequences (labels are Gray-coded bit patterns)."""
    diff = np.bitwise_xor(np.asarray(tx_idx), np.asarray(rx_idx))
    nbits = int(np.log2(m))
    errors = sum(((diff >> b) & 1).sum() for b in range(nbits))
    return errors / (len(np.atleast_1d(tx_idx)) * nbits)


def ber_bpsk_awgn(ebno_db):
    """Theoretical BPSK/QPSK bit error rate in AWGN: Q(sqrt(2 Eb/N0))."""
    g = 10 ** (np.asarray(ebno_db) / 10)
    return qfunc(np.sqrt(2 * g))


def ber_bpsk_rayleigh(ebno_db):
    """Theoretical BPSK bit error rate in flat Rayleigh fading (mean Eb/N0 = Gamma):
    0.5 (1 - sqrt(Gamma / (1 + Gamma)))."""
    g = 10 ** (np.asarray(ebno_db) / 10)
    return 0.5 * (1 - np.sqrt(g / (1 + g)))
