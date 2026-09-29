"""Diversity combining and a first look at MIMO (Week 8).

Notation follows the Week 8 deck: ``M`` diversity branches, each with
instantaneous SNR gamma_i and mean SNR Gamma = E[gamma_i] (i.i.d. Rayleigh, so
gamma_i is exponential), per-branch noise power ``N``, combiner output SNR
gamma_OUTPUT, and ``M_T`` / ``M_R`` transmit / receive antennas.

Pure math only -- plotting lives in the Week 8 notebook.
"""

from math import comb, factorial

import numpy as np
from scipy import integrate, special

from . import fading
from .modulation import qfunc

__all__ = [
    "clarke_correlation", "frequency_correlation", "spatial_branches", "wideband_sos",
    "polarization_branches", "branch_snrs",
    "selection_combine", "mrc_combine", "egc_combine",
    "sc_cdf", "sc_pdf", "sc_mean_snr", "mrc_cdf", "mrc_pdf", "mrc_mean_snr", "egc_mean_snr",
    "outage_probability", "ber_bpsk_mrc", "ber_bpsk_sc", "ber_bpsk_egc2", "ber_bpsk_alamouti",
    "simulate_ber_bpsk", "simulate_alamouti_bpsk", "diversity_order",
    "mimo_capacity", "ergodic_capacity",
]


# ---------------------------------------------------------------------------
# Diversity branches and their correlation
# ---------------------------------------------------------------------------

def clarke_correlation(d_wavelengths):
    """Complex-gain correlation of two points ``d`` wavelengths apart in Clarke's
    isotropic scattering field: rho = J0(2 pi d / lambda).

    Equivalently, samples of one mobile tau seconds apart: rho = J0(2 pi f_m tau).
    The envelope-squared (power) correlation is rho**2.
    """
    return special.j0(2 * np.pi * np.asarray(d_wavelengths, dtype=float))


def frequency_correlation(delta_f_hz, sigma_tau_s):
    """Magnitude of the frequency correlation of an exponential power delay
    profile with rms delay spread sigma_tau:  |rho(Delta f)| = 1 / sqrt(1 + (2 pi Delta f sigma_tau)^2).

    Branches spaced by more than a coherence bandwidth B_c ~ 1/(2 pi sigma_tau)
    are nearly uncorrelated.
    """
    x = 2 * np.pi * np.asarray(delta_f_hz, dtype=float) * sigma_tau_s
    return 1 / np.sqrt(1 + x**2)


def spatial_branches(fm_hz, t, spacings_wl, n_paths=64, rng=None):
    """Complex gains seen by antennas placed ``spacings_wl`` wavelengths apart
    along the direction of motion, in one Clarke scattering field.

    A mobile moving at v = f_m lambda covers d wavelengths in d / f_m seconds, so
    antenna k sees the same field as antenna 0 shifted in time.  Returns an
    array of shape (len(spacings_wl), len(t)); the same trick models *time*
    diversity (repeat after tau seconds  <->  spacing f_m tau wavelengths).
    """
    seed = np.random.default_rng(rng).integers(2**63)
    t = np.asarray(t, dtype=float)
    return np.array([fading.rayleigh_sos(fm_hz, t + d / fm_hz, n_paths, seed)
                     for d in np.atleast_1d(spacings_wl)])


def wideband_sos(fm_hz, t, freqs_hz, sigma_tau_s, n_paths=128, rng=None):
    """Time-varying frequency response H(f, t) of a wideband Rayleigh channel.

    Sum of sinusoids with exponentially distributed path delays (rms delay
    spread ``sigma_tau_s``) and Clarke Doppler shifts:
        H(f, t) = (1/sqrt(n)) sum_n exp(j(2 pi f_m cos(theta_n) t - 2 pi f tau_n + phi_n))
    Returns shape (len(freqs_hz), len(t)) with E[|H|^2] = 1.
    """
    rng = np.random.default_rng(rng)
    t = np.asarray(t, dtype=float)
    theta = rng.uniform(0, 2 * np.pi, n_paths)
    phi = rng.uniform(0, 2 * np.pi, n_paths)
    tau = rng.exponential(sigma_tau_s, n_paths)
    f = np.atleast_1d(np.asarray(freqs_hz, dtype=float))
    phase = (2 * np.pi * fm_hz * np.multiply.outer(t, np.cos(theta))[None, :, :]
             - 2 * np.pi * np.multiply.outer(f, tau)[:, None, :] + phi)
    return np.exp(1j * phase).sum(axis=-1) / np.sqrt(n_paths)


def polarization_branches(fm_hz, t, rho, xpd_db=0.0, rng=None):
    """Two co-located, differently polarised branches (e.g. V and H).

    Branch 2 = rho * branch1 + sqrt(1 - rho^2) * independent fading, then scaled
    by the cross-polar discrimination XPD (mean power of branch 2 is 10^(-XPD/10)).
    Returns shape (2, len(t)).
    """
    rng = np.random.default_rng(rng)
    h1 = fading.rayleigh_sos(fm_hz, t, rng=rng)
    h_ind = fading.rayleigh_sos(fm_hz, t, rng=rng)
    h2 = (rho * h1 + np.sqrt(1 - rho**2) * h_ind) * 10 ** (-xpd_db / 20)
    return np.array([h1, h2])


def branch_snrs(n, m, gamma_mean, rng=None):
    """``n`` realisations of ``m`` i.i.d. Rayleigh branch SNRs gamma_i (exponential,
    mean Gamma). Returns shape (n, m)."""
    rng = np.random.default_rng(rng)
    return rng.exponential(gamma_mean, size=(n, m))


# ---------------------------------------------------------------------------
# Combiners (operate on branch SNRs along the last axis)
# ---------------------------------------------------------------------------

def selection_combine(gamma):
    """Selection diversity: gamma_OUTPUT = max_i gamma_i."""
    return np.max(gamma, axis=-1)


def mrc_combine(gamma):
    """Maximal ratio combining (G_i proportional to r_i): gamma_OUTPUT = sum_i gamma_i."""
    return np.sum(gamma, axis=-1)


def egc_combine(gamma):
    """Equal gain combining (unit gains, co-phased): gamma_OUTPUT = (sum_i sqrt(gamma_i))^2 / M."""
    gamma = np.asarray(gamma)
    return np.sum(np.sqrt(gamma), axis=-1) ** 2 / gamma.shape[-1]


# ---------------------------------------------------------------------------
# Output-SNR statistics (i.i.d. Rayleigh branches, mean Gamma)
# ---------------------------------------------------------------------------

def sc_cdf(gamma, gamma_mean, m):
    """Selection diversity: P_M(gamma) = Pr[gamma_1..gamma_M <= gamma] = (1 - e^{-gamma/Gamma})^M."""
    return (-np.expm1(-np.asarray(gamma, dtype=float) / gamma_mean)) ** m


def sc_pdf(gamma, gamma_mean, m):
    """d P_M / d gamma = (M / Gamma) (1 - e^{-gamma/Gamma})^{M-1} e^{-gamma/Gamma}."""
    x = np.asarray(gamma, dtype=float) / gamma_mean
    return m / gamma_mean * (-np.expm1(-x)) ** (m - 1) * np.exp(-x)


def sc_mean_snr(gamma_mean, m):
    """Mean selection-combiner SNR: Gamma sum_{k=1}^{M} 1/k."""
    return gamma_mean * np.sum(1.0 / np.arange(1, m + 1))


def mrc_pdf(gamma, gamma_mean, m):
    """MRC output SNR density (Erlang / chi-square with 2M degrees of freedom):
    p(gamma_OUTPUT) = gamma^{M-1} e^{-gamma/Gamma} / (Gamma^M (M-1)!)."""
    g = np.asarray(gamma, dtype=float)
    return g ** (m - 1) * np.exp(-g / gamma_mean) / (gamma_mean**m * factorial(m - 1))


def mrc_cdf(gamma, gamma_mean, m):
    """Pr(gamma_OUTPUT <= gamma) = 1 - e^{-gamma/Gamma} sum_{k=1}^{M} (gamma/Gamma)^{k-1} / (k-1)!."""
    return special.gammainc(m, np.asarray(gamma, dtype=float) / gamma_mean)


def mrc_mean_snr(gamma_mean, m):
    """Mean MRC output SNR: M Gamma."""
    return m * gamma_mean


def egc_mean_snr(gamma_mean, m):
    """Mean EGC output SNR for Rayleigh branches: Gamma (1 + (M - 1) pi / 4).

    From E[sqrt(gamma_i)] = sqrt(pi Gamma)/2 and E[gamma_i] = Gamma."""
    return gamma_mean * (1 + (m - 1) * np.pi / 4)


def outage_probability(gamma_th, gamma_mean, m, scheme="mrc"):
    """Pr(gamma_OUTPUT < gamma_th) for ``scheme`` in {"sc", "mrc"} (closed forms).
    For M = 1 all schemes reduce to 1 - e^{-gamma_th/Gamma}."""
    if scheme == "sc":
        return sc_cdf(gamma_th, gamma_mean, m)
    if scheme == "mrc":
        return mrc_cdf(gamma_th, gamma_mean, m)
    raise ValueError("closed form only for 'sc' and 'mrc'; simulate EGC")


# ---------------------------------------------------------------------------
# BPSK bit error rate with diversity (Gamma = mean Eb/N0 per branch, linear)
# ---------------------------------------------------------------------------

def ber_bpsk_mrc(gamma_mean, m):
    """Coherent BPSK with M-branch MRC in i.i.d. Rayleigh fading (Proakis):
        P_b = ((1 - mu)/2)^M sum_{k=0}^{M-1} C(M-1+k, k) ((1 + mu)/2)^k,  mu = sqrt(Gamma/(1+Gamma)).
    At high SNR P_b ~ C(2M-1, M) (4 Gamma)^{-M}: diversity order M."""
    g = np.asarray(gamma_mean, dtype=float)
    mu = np.sqrt(g / (1 + g))
    # (1 - mu)/2 written stably for large Gamma
    p = 0.5 / (np.sqrt(1 + g) * (np.sqrt(1 + g) + np.sqrt(g)))
    s = sum(comb(m - 1 + k, k) * ((1 + mu) / 2) ** k for k in range(m))
    return p**m * s


def ber_bpsk_sc(gamma_mean, m):
    """Coherent BPSK with M-branch selection combining in i.i.d. Rayleigh fading:
        P_b = int_0^inf Q(sqrt(2 gamma)) p_M(gamma) d gamma
            = (1/2) sum_{k=0}^{M} (-1)^k C(M, k) (1 + k/Gamma)^{-1/2}.
    Evaluated by quadrature (the alternating sum cancels badly at high SNR)."""
    g = np.atleast_1d(np.asarray(gamma_mean, dtype=float))

    def one(gm):
        f = lambda x: qfunc(np.sqrt(2 * gm * x)) * m * (-np.expm1(-x)) ** (m - 1) * np.exp(-x)
        # most of the mass sits at x ~ 1/Gamma; split the range there
        brk = min(1.0, 20.0 / gm)
        return (integrate.quad(f, 0, brk, epsabs=0, epsrel=1e-10, limit=200)[0]
                + integrate.quad(f, brk, np.inf, epsabs=0, epsrel=1e-10, limit=200)[0])

    out = np.array([one(x) for x in g])
    return out.reshape(np.shape(gamma_mean)) if np.ndim(gamma_mean) else float(out[0])


def ber_bpsk_egc2(gamma_mean):
    """Coherent BPSK with 2-branch EGC in i.i.d. Rayleigh fading (closed form):
        P_b = (1/2) [1 - sqrt(1 - 1/(1 + Gamma)^2)]."""
    g = np.asarray(gamma_mean, dtype=float)
    return 0.5 * (1 - np.sqrt(1 - 1 / (1 + g) ** 2))


def ber_bpsk_alamouti(gamma_mean):
    """Alamouti 2x1 with total transmit power split over two antennas: behaves as
    2-branch MRC with per-branch mean SNR Gamma/2 (3 dB array-gain penalty)."""
    return ber_bpsk_mrc(np.asarray(gamma_mean, dtype=float) / 2, 2)


def diversity_order(snr_db, ber):
    """Negative slope of log10(P_b) vs. SNR in decades: d = -d log10 P_b / d log10 Gamma
    (least-squares fit over the given points)."""
    x = np.asarray(snr_db, dtype=float) / 10
    y = np.log10(np.asarray(ber, dtype=float))
    return -np.polyfit(x, y, 1)[0]


# ---------------------------------------------------------------------------
# Monte Carlo (signal level: complex gains h_i, AWGN with per-branch power N)
# ---------------------------------------------------------------------------

def simulate_ber_bpsk(ebno_db, m, scheme="mrc", n_bits=100_000, rng=None):
    """Monte Carlo BER of coherent BPSK over M i.i.d. flat Rayleigh branches.

    r_i = h_i s + n_i with E|h_i|^2 = 1, E_b = 1 and per-branch noise power
    N = N_0 = 1/Gamma.  Combiners:
      "sc"  pick the branch with the largest |h_i|,
      "mrc" G_i = h_i^*  (sum_i h_i^* r_i),
      "egc" G_i = e^{-j arg h_i}.
    Returns an array of BERs, one per entry of ``ebno_db``.
    """
    rng = np.random.default_rng(rng)
    out = []
    for snr in np.atleast_1d(ebno_db):
        n0 = 10 ** (-snr / 10)
        bits = rng.integers(0, 2, n_bits)
        s = 2.0 * bits - 1
        h = fading.iid_rayleigh((n_bits, m), rng)
        noise = np.sqrt(n0) * fading.iid_rayleigh((n_bits, m), rng)
        r = h * s[:, None] + noise
        if scheme == "mrc":
            z = np.sum(np.conj(h) * r, axis=1)
        elif scheme == "egc":
            z = np.sum(np.exp(-1j * np.angle(h)) * r, axis=1)
        elif scheme == "sc":
            k = np.argmax(np.abs(h), axis=1)
            idx = np.arange(n_bits)
            z = np.conj(h[idx, k]) * r[idx, k]
        else:
            raise ValueError(scheme)
        out.append(np.mean((z.real > 0) != bits.astype(bool)))
    return np.array(out)


def simulate_alamouti_bpsk(ebno_db, n_bits=100_000, rng=None):
    """Monte Carlo BER of the Alamouti 2x1 space-time block code with BPSK.

    Slot 1 sends (s1, s2)/sqrt(2), slot 2 sends (-s2^*, s1^*)/sqrt(2) from antennas
    (1, 2): total transmit energy per symbol period equals the SISO case. The
    channel (h1, h2) is constant over the two slots. Linear combining
        s1_hat = h1^* r1 + h2 r2^*,  s2_hat = h2^* r1 - h1 r2^*
    gives (|h1|^2 + |h2|^2)/sqrt(2) * s + noise.
    """
    rng = np.random.default_rng(rng)
    n_pairs = n_bits // 2
    out = []
    for snr in np.atleast_1d(ebno_db):
        n0 = 10 ** (-snr / 10)
        bits = rng.integers(0, 2, (n_pairs, 2))
        s = 2.0 * bits - 1
        h = fading.iid_rayleigh((n_pairs, 2), rng)
        n = np.sqrt(n0) * fading.iid_rayleigh((n_pairs, 2), rng)
        h1, h2 = h[:, 0], h[:, 1]
        s1, s2 = s[:, 0], s[:, 1]
        r1 = (h1 * s1 + h2 * s2) / np.sqrt(2) + n[:, 0]
        r2 = (-h1 * np.conj(s2) + h2 * np.conj(s1)) / np.sqrt(2) + n[:, 1]
        z1 = np.conj(h1) * r1 + h2 * np.conj(r2)
        z2 = np.conj(h2) * r1 - h1 * np.conj(r2)
        det = np.column_stack([z1.real > 0, z2.real > 0])
        out.append(np.mean(det != bits.astype(bool)))
    return np.array(out)


# ---------------------------------------------------------------------------
# MIMO capacity teaser (Week 9 bridge)
# ---------------------------------------------------------------------------

def mimo_capacity(H, snr_linear):
    """Open-loop (equal power) MIMO capacity in bit/s/Hz for one channel matrix:
        I = log2 det(I_{M_R} + (E_s / (M_T N_0)) H H^H) = sum_i log2(1 + E_s lambda_i / (M_T N_0)),
    lambda_i the eigenvalues of H H^H.  ``H`` may be a stack (..., M_R, M_T)."""
    H = np.asarray(H)
    m_t = H.shape[-1]
    lam = np.linalg.eigvalsh(H @ np.conj(np.swapaxes(H, -1, -2)))
    lam = np.clip(lam, 0, None)
    return np.sum(np.log2(1 + snr_linear / m_t * lam), axis=-1)


def ergodic_capacity(m_t, m_r, snr_db, n_trials=2000, rng=None):
    """Average open-loop capacity E[I] over i.i.d. CN(0,1) channels H (M_R x M_T).
    Returns an array matching ``snr_db``."""
    rng = np.random.default_rng(rng)
    H = fading.iid_rayleigh((n_trials, m_r, m_t), rng)
    return np.array([mimo_capacity(H, 10 ** (s / 10)).mean() for s in np.atleast_1d(snr_db)])
