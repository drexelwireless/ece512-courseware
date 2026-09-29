"""Smart antennas: steering vectors, spatial covariance, DOA estimation, beamforming (Week 6).

Conventions (match the Week 6 lecture):

* Element positions ``(x_i, y_i)`` are given **in wavelengths**, so the wavenumber is
  ``k = 2*pi`` and ``k*x_i = 2*pi*x_i/lambda``.
* Angles ``theta`` are in **radians**, measured from the +x axis (for a ULA laid out on the
  x axis this is the angle from the array axis: broadside is ``theta = pi/2``).
* Steering vector (lecture "Steering Vector", generic geometry)::

      a_i(theta) = exp(-j k (x_i cos(theta) + y_i sin(theta))),   i = 1..M

* Beamformer output ``y = w^H x``; the (power) pattern of weight ``w`` is ``|w^H a(theta)|^2``.
  For downlink the same pattern applies by reciprocity when element ``i`` is fed ``w_i^*``.
* Signal model ``x(t) = G s(t) + n(t)``, ``R = E[x x^H] = G P G^H + sigma^2 I = U Lambda U^H``.
"""

import numpy as np

K_WAVENUMBER = 2.0 * np.pi  # k = 2*pi/lambda with lengths in wavelengths


# --------------------------------------------------------------------------------------
# Geometry and steering vectors
# --------------------------------------------------------------------------------------
def ula_positions(M, d=0.5):
    """ULA on the x axis: ``x_i = (i-1) d``, ``y_i = 0`` (``d`` in wavelengths)."""
    x = d * np.arange(M, dtype=float)
    return x, np.zeros(M)


def uca_positions(M, radius=0.5):
    """UCA of radius ``R`` (wavelengths): ``x_i = R cos(2 pi (i-1)/M)``, ``y_i = R sin(2 pi (i-1)/M)``."""
    phi = 2.0 * np.pi * np.arange(M) / M
    return radius * np.cos(phi), radius * np.sin(phi)


def uca_radius_for_spacing(M, d):
    """Radius (wavelengths) of a UCA whose adjacent elements are ``d`` wavelengths apart."""
    return d / (2.0 * np.sin(np.pi / M))


def steering_vector(theta, x, y=None):
    """Generic steering vector(s) ``a_i(theta) = exp(-jk(x_i cos theta + y_i sin theta))``.

    Parameters
    ----------
    theta : float or array (K,)  -- direction(s) in radians.
    x, y  : arrays (M,)          -- element positions in wavelengths (``y`` defaults to 0).

    Returns an (M,) vector for scalar ``theta`` or an (M, K) matrix whose columns are
    ``a(theta_k)`` (the "steering matrix" ``A``).
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x) if y is None else np.asarray(y, dtype=float)
    th = np.asarray(theta, dtype=float)
    phase = np.multiply.outer(x, np.cos(th)) + np.multiply.outer(y, np.sin(th))
    return np.exp(-1j * K_WAVENUMBER * phase)


def ula_steering(theta, M, d=0.5):
    """ULA steering vector ``[1, e^{-jkd cos theta}, ..., e^{-jk(M-1)d cos theta}]^T``."""
    return steering_vector(theta, *ula_positions(M, d))


def uca_steering(theta, M, radius=0.5):
    """UCA steering vector (lecture "Steering Vector: Uniform Circular Array")."""
    return steering_vector(theta, *uca_positions(M, radius))


def array_pattern(w, A):
    """Power pattern ``|w^H a(theta)|^2`` for each column of the steering matrix ``A`` (M, K)."""
    return np.abs(np.conj(w) @ A) ** 2


def normalized_pattern_db(w, A, floor_db=-50.0):
    """Pattern in dB relative to its maximum over the grid, clipped at ``floor_db``."""
    p = array_pattern(w, A)
    return np.maximum(10.0 * np.log10(p / p.max() + 1e-300), floor_db)


def ula_grating_lobes(theta0, d):
    """Directions (rad, in [0, pi]) where a ULA steered to ``theta0`` has full-gain main/grating lobes.

    The array factor of ``w = a(theta0)`` peaks wherever ``kd(cos theta - cos theta0) = 2 pi m``,
    i.e. ``cos theta = cos theta0 + m/d`` for integer ``m`` (``m = 0`` is the main lobe).
    """
    m_max = int(np.ceil(2.0 * d)) + 1
    c = np.cos(theta0) + np.arange(-m_max, m_max + 1) / d
    c = c[np.abs(c) <= 1.0 + 1e-12]
    return np.sort(np.arccos(np.clip(c, -1.0, 1.0)))


def ula_max_spacing(theta_max_from_broadside):
    """Largest ULA spacing (wavelengths) with no grating lobe when scanning up to the given angle.

    ``d < 1 / (1 + |sin(theta_max_from_broadside)|)``; equals ``1/2`` for full scan (90 degrees).
    """
    return 1.0 / (1.0 + np.abs(np.sin(theta_max_from_broadside)))


# --------------------------------------------------------------------------------------
# Signal model and spatial covariance
# --------------------------------------------------------------------------------------
def spatial_signature(alphas, A):
    """Spatial signature ``g = sum_i alpha_i a(theta_i)`` (``A`` has columns ``a(theta_i)``)."""
    return A @ np.asarray(alphas)


def covariance(G, powers, sigma2):
    """Theoretical spatial covariance ``R = G P G^H + sigma^2 I`` with ``P = diag(powers)``."""
    G = np.atleast_2d(G)
    P = np.diag(np.atleast_1d(powers).astype(float))
    return G @ P @ G.conj().T + sigma2 * np.eye(G.shape[0])


def crandn(shape, rng):
    """Circularly-symmetric complex Gaussian samples with unit variance."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def simulate_snapshots(G, powers, sigma2, N, rng=None):
    """``N`` snapshots of ``x = G s + n`` (columns of an (M, N) matrix).

    Sources are independent complex Gaussian with powers ``powers``; noise is white with
    variance ``sigma2`` per element.
    """
    rng = np.random.default_rng(rng)
    G = np.atleast_2d(G)
    M, L = G.shape
    S = np.sqrt(np.atleast_1d(powers))[:, None] * crandn((L, N), rng)
    return G @ S + np.sqrt(sigma2) * crandn((M, N), rng)


def sample_covariance(X):
    """Sample estimate ``R_hat = (1/N) sum_i x(i T_s) x^H(i T_s)`` from an (M, N) snapshot matrix."""
    return X @ X.conj().T / X.shape[1]


def eig_desc(R):
    """Eigen-decomposition ``R = U Lambda U^H`` with eigenvalues sorted high to low."""
    lam, U = np.linalg.eigh(R)
    idx = np.argsort(lam)[::-1]
    return lam[idx], U[:, idx]


def subspaces(R, L):
    """Return ``(eigenvalues, U_s, U_n)``: the ``L`` dominant eigenvectors and the ``M-L`` others."""
    lam, U = eig_desc(R)
    return lam, U[:, :L], U[:, L:]


def projector(U):
    """Orthogonal projection ``Pi = U U^H`` onto the span of the orthonormal columns of ``U``."""
    return U @ U.conj().T


def mdl_order(lam, N):
    """Minimum-description-length estimate of the number of sources from sorted eigenvalues.

    (Wax & Kailath, 1985.) Not in the lecture; used to show that the eigenvalue gap reveals ``L``.
    """
    lam = np.asarray(lam, dtype=float)
    M = lam.size
    mdl = np.empty(M)
    for k in range(M):
        tail = np.maximum(lam[k:], 1e-300)
        ratio = np.exp(np.mean(np.log(tail))) / np.mean(tail)
        mdl[k] = -N * (M - k) * np.log(ratio) + 0.5 * k * (2 * M - k) * np.log(N)
    return int(np.argmin(mdl))


# --------------------------------------------------------------------------------------
# DOA spectra (A = steering matrix on a scan grid, columns a(theta))
# --------------------------------------------------------------------------------------
def _quad(A, B):
    """``a^H(theta) B a(theta)`` for every column of ``A`` (real part)."""
    return np.real(np.sum(A.conj() * (B @ A), axis=0))


def bartlett_spectrum(R, A):
    """Bartlett (conventional beamformer) ``P_BF = a^H R a / a^H a``."""
    return _quad(A, R) / np.real(np.sum(np.abs(A) ** 2, axis=0))


def capon_spectrum(R, A):
    """Capon / MVDR spectrum ``P_MVDR = 1 / (a^H R^{-1} a)`` (not in the slides; for comparison)."""
    return 1.0 / _quad(A, np.linalg.inv(R))


def music_spectrum(R, A, L):
    """MUSIC pseudo-spectrum ``P_MUSIC = a^H a / (a^H Pi_n a)`` with ``Pi_n = U_n U_n^H``."""
    _, _, Un = subspaces(R, L)
    den = _quad(A, projector(Un))
    return np.real(np.sum(np.abs(A) ** 2, axis=0)) / np.maximum(den, 1e-15)


def find_peaks(spectrum, theta, n_peaks):
    """Angles of the ``n_peaks`` largest local maxima of ``spectrum`` on the grid ``theta``.

    Returns an array sorted by angle; it can be shorter than ``n_peaks`` if the spectrum has
    fewer local maxima (i.e. the sources are not resolved).
    """
    p = np.asarray(spectrum)
    interior = (p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])
    idx = np.flatnonzero(interior) + 1
    if p[0] > p[1]:
        idx = np.append(idx, 0)
    if p[-1] > p[-2]:
        idx = np.append(idx, p.size - 1)
    best = idx[np.argsort(p[idx])[::-1][:n_peaks]]
    return np.sort(np.asarray(theta)[best])


def resolved(estimates, truth, tol):
    """True if there are as many estimates as sources and each is within ``tol`` of its source."""
    est, tr = np.sort(np.atleast_1d(estimates)), np.sort(np.atleast_1d(truth))
    return est.size == tr.size and bool(np.all(np.abs(est - tr) <= tol))


# --------------------------------------------------------------------------------------
# Downlink beamforming weights
# --------------------------------------------------------------------------------------
def dominant_doa_weights(a_dom):
    """Dominant-DOA method: ``w = a(theta_DOM)``."""
    return np.asarray(a_dom)


def pinv_weights(A, col=0):
    """Pseudoinverse method: ``w = [row(pinv(A), col+1)]^H``.

    Because ``A^dagger A = I`` (full column rank), ``w^H a_j = 1`` for ``j = col`` and ``0``
    for all other columns: unit gain on the desired vector, nulls on the rest.
    """
    return np.linalg.pinv(A)[col].conj()


def unit_norm(w):
    """Scale ``w`` to unit norm (fixed total transmit power)."""
    w = np.asarray(w)
    return w / np.linalg.norm(w)


def downlink_sinr(W, G, powers, sigma2):
    """Per-user downlink SINR for beams ``W`` (M, L) and user spatial signatures ``G`` (M, L).

    User ``l`` receives gain ``|w_k^H g_l|^2`` from beam ``k`` (same convention as the pattern), so

        SINR_l = P_l |w_l^H g_l|^2 / ( sum_{k != l} P_k |w_k^H g_l|^2 + sigma^2 ).
    """
    C = np.abs(W.conj().T @ G) ** 2  # C[k, l] = |w_k^H g_l|^2
    p = np.asarray(powers, dtype=float)[:, None]
    rx = p * C
    sig = np.diag(rx)
    return sig / (rx.sum(axis=0) - sig + sigma2)
