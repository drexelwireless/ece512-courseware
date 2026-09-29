"""Baseband OFDM building blocks (Week 7 lecture, OFDM section).

Discrete-time model with N = ``n_fft`` sub-carriers and a cyclic prefix (CP) of
``n_cp`` samples. Unitary transforms are used (``norm="ortho"``), so unit-energy
sub-carrier symbols give unit-power time samples and the noise variance per
sample equals the noise variance per sub-carrier. With a discrete channel
impulse response h[n] of length L <= n_cp + 1 (T_CP >= tau_max) the CP turns the
linear convolution into a circular one and each sub-carrier sees a flat gain:

    Y_k = H_k X_k + N_k,   H_k = sum_n h[n] exp(-j 2 pi k n / N)   (lecture key equation)

Sub-carrier index k = 0..N-1 in FFT order. Pilots use a comb pattern; the
channel is estimated by least squares (LS) at the pilots, H_p = Y_p / X_p, and
interpolated across the data sub-carriers.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from . import modulation as mod


def modulate(X, n_cp):
    """OFDM modulator: IFFT each row of ``X`` (shape (n_sym, N)) and prepend the CP.

    Returns the serial time-domain signal of length n_sym (N + n_cp).
    """
    X = np.atleast_2d(X)
    x = np.fft.ifft(X, axis=1, norm="ortho")
    if n_cp:
        x = np.concatenate([x[:, -n_cp:], x], axis=1)
    return x.ravel()


def demodulate(y, n_fft, n_cp):
    """OFDM demodulator: split into symbols, drop the CP and FFT; returns shape (n_sym, N)."""
    n_sym = len(y) // (n_fft + n_cp)
    blocks = np.asarray(y)[: n_sym * (n_fft + n_cp)].reshape(n_sym, n_fft + n_cp)
    return np.fft.fft(blocks[:, n_cp:], axis=1, norm="ortho")


def multipath_channel(x, h):
    """Pass ``x`` through the discrete multipath channel ``h`` (linear convolution, same length as x).

    Consecutive OFDM symbols overlap through the channel memory, so ISI appears
    whenever len(h) - 1 > n_cp.
    """
    return np.convolve(x, h)[: len(x)]


def channel_frequency_response(h, n_fft):
    """Per-sub-carrier channel gains H_k = FFT_N{h[n]} (unnormalised DFT)."""
    if len(h) > n_fft:
        raise ValueError("impulse response longer than the FFT size")
    return np.fft.fft(h, n_fft)


def comb_pilot_indices(n_fft, spacing):
    """Comb pilot sub-carriers 0, spacing, 2*spacing, ... plus the last sub-carrier (no extrapolation)."""
    idx = np.arange(0, n_fft, spacing)
    if idx[-1] != n_fft - 1:
        idx = np.append(idx, n_fft - 1)
    return idx


def data_indices(n_fft, pilot_idx=()):
    """Sub-carriers not used by pilots."""
    return np.setdiff1d(np.arange(n_fft), np.asarray(pilot_idx, dtype=int))


def build_grid(data_symbols, n_fft, pilot_idx=(), pilot_value=1.0 + 0j):
    """Place data symbols (shape (n_sym, n_data)) and pilots on the N sub-carriers; returns X."""
    data_symbols = np.atleast_2d(data_symbols)
    X = np.full((data_symbols.shape[0], n_fft), pilot_value, dtype=complex)
    X[:, data_indices(n_fft, pilot_idx)] = data_symbols
    return X


def interpolate_channel(h_pilots, pilot_idx, n_fft, method="linear"):
    """Interpolate pilot estimates (shape (n_sym, n_pilots)) to all N sub-carriers.

    ``method``: ``"linear"`` (real/imag parts separately), ``"cubic"`` (cubic spline) or
    ``"nearest"`` (piecewise constant).
    """
    h_pilots = np.atleast_2d(h_pilots)
    k = np.arange(n_fft)
    pilot_idx = np.asarray(pilot_idx)
    if method == "linear":
        return np.array([np.interp(k, pilot_idx, row.real) + 1j * np.interp(k, pilot_idx, row.imag) for row in h_pilots])
    if method == "cubic":
        return CubicSpline(pilot_idx, h_pilots, axis=1)(k)
    if method == "nearest":
        nearest = np.abs(k[:, None] - pilot_idx[None, :]).argmin(axis=1)
        return h_pilots[:, nearest]
    raise ValueError("method must be 'linear', 'cubic' or 'nearest'")


def ls_channel_estimate(Y, pilot_idx, pilot_value=1.0 + 0j, method="linear"):
    """Least-squares channel estimate H_p = Y_p / X_p at the pilots, interpolated to all sub-carriers."""
    Y = np.atleast_2d(Y)
    h_p = Y[:, pilot_idx] / pilot_value
    return interpolate_channel(h_p, pilot_idx, Y.shape[1], method)


def equalize(Y, H):
    """One-tap zero-forcing equalizer per sub-carrier: X_hat_k = Y_k / H_k."""
    return Y / H


def simulate_link(points, n_fft, n_cp, h, snr_db, n_sym=20, pilot_spacing=None,
                  pilot_value=1.0 + 0j, method="linear", rng=None):
    """End-to-end OFDM link through a static multipath channel ``h`` with AWGN.

    ``snr_db`` is the mean SNR per sub-carrier (E_s/N_0 for unit-energy symbols and
    a unit-power channel). With ``pilot_spacing=None`` the receiver knows H exactly
    (perfect CSI) and all sub-carriers carry data; otherwise comb pilots are inserted
    and the LS estimate (interpolated with ``method``) is used for equalization.

    Returns a dict with ``ber``, ``ser``, ``X``, ``Y``, ``H`` (true, length N),
    ``H_hat`` (shape (n_sym, N)), ``X_hat`` (equalized data symbols), ``data_idx``,
    ``pilot_idx``, ``tx_idx``, ``rx_idx``.
    """
    rng = np.random.default_rng(rng)
    points = np.asarray(points)
    pilot_idx = np.array([], dtype=int) if pilot_spacing is None else comb_pilot_indices(n_fft, pilot_spacing)
    d_idx = data_indices(n_fft, pilot_idx)
    tx_idx, s = mod.random_symbols(points, n_sym * len(d_idx), rng)
    X = build_grid(s.reshape(n_sym, len(d_idx)), n_fft, pilot_idx, pilot_value)
    y = mod.awgn(multipath_channel(modulate(X, n_cp), h), snr_db, rng)
    Y = demodulate(y, n_fft, n_cp)
    H = channel_frequency_response(h, n_fft)
    if pilot_spacing is None:
        H_hat = np.broadcast_to(H, Y.shape)
    else:
        H_hat = ls_channel_estimate(Y, pilot_idx, pilot_value, method)
    X_hat = equalize(Y, H_hat)[:, d_idx]
    rx_idx = mod.detect(X_hat.ravel(), points)
    return {
        "ber": mod.bit_error_rate(tx_idx, rx_idx, len(points)),
        "ser": mod.symbol_error_rate(tx_idx, rx_idx),
        "X": X, "Y": Y, "H": H, "H_hat": H_hat, "X_hat": X_hat,
        "data_idx": d_idx, "pilot_idx": pilot_idx, "tx_idx": tx_idx, "rx_idx": rx_idx,
    }
