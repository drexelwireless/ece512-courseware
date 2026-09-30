"""Antennas and propagation mechanisms (Week 4 lecture).

Pure-math helpers for the Week 4 companion notebook:

* far field / Fraunhofer distance and aperture phase error,
* radiation patterns (small dipole, finite dipole, uniform linear array),
* numerical directivity and half-power beamwidth,
* Fresnel reflection coefficients and the Brewster angle,
* the two-ray ground-reflection model,
* knife-edge diffraction (Fresnel integral and Lee's approximation) and Fresnel zones.

Angles are in radians unless a name ends in ``_deg``.  Spherical coordinates follow
the lecture: ``theta`` is measured from the +z axis, ``phi`` in the xy plane.
"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.special import fresnel

from .units import wavelength

ETA_0 = 376.730313  # free-space impedance (ohm), ~377 on the slides


# ---------------------------------------------------------------------------
# Far field
# ---------------------------------------------------------------------------

def fraunhofer_distance(D_m, wavelength_m):
    """Far-field (Rayleigh / Fraunhofer) distance of an electrically large antenna,
    ``R_ff = 2 D^2 / lambda`` (slide "Far Field: Rayleigh Distance")."""
    return 2.0 * np.asarray(D_m, float) ** 2 / np.asarray(wavelength_m, float)


def far_field_distance(D_m, wavelength_m):
    """Far-field distance using the slide's two cases: ``max(2 D^2/lambda, 2 lambda)``
    (the ``2 lambda`` floor applies to electrically small antennas)."""
    lam = np.asarray(wavelength_m, float)
    return np.maximum(fraunhofer_distance(D_m, lam), 2.0 * lam)


def aperture_phase_error(D_m, r_m, wavelength_m):
    """Maximum phase error (rad) across an aperture of size ``D`` seen from distance ``r``
    on boresight: ``k (sqrt(r^2 + (D/2)^2) - r)``.  At ``r = 2D^2/lambda`` this is
    ~pi/8 (22.5 deg), the classical criterion behind the Fraunhofer distance."""
    r = np.asarray(r_m, float)
    k = 2 * np.pi / wavelength_m
    return k * (np.sqrt(r**2 + (D_m / 2.0) ** 2) - r)


def line_source_field(theta, D_m, wavelength_m, r_m=np.inf, n_points=None):
    """Normalized |field| of a uniform line source of length ``D`` on the z axis.

    With ``r_m = inf`` the far-field pattern ``|sinc(k D cos(theta) / 2)|`` is returned.
    For finite ``r_m`` the field is obtained by summing exact spherical waves
    ``exp(-j k R) / R`` from points along the source (near-field / Fresnel region).
    Returned array is normalized to its maximum over ``theta``.
    """
    theta = np.asarray(theta, float)
    k = 2 * np.pi / wavelength_m
    if np.isinf(r_m):
        x = k * D_m * np.cos(theta) / 2.0
        f = np.abs(np.sinc(x / np.pi))
    else:
        if n_points is None:
            n_points = max(int(np.ceil(20 * D_m / wavelength_m)), 21)
        # midpoint sampling: n_points sources, each representing a D/n_points segment
        z = (np.arange(n_points) + 0.5) * D_m / n_points - D_m / 2
        R = np.sqrt(r_m**2 + z[None, :] ** 2 - 2 * r_m * z[None, :] * np.cos(theta)[:, None])
        f = np.abs(np.sum(np.exp(-1j * k * R) / R, axis=1))
    return f / f.max()


# ---------------------------------------------------------------------------
# Radiation patterns (field pattern functions F(theta), normalized to max 1)
# ---------------------------------------------------------------------------

def small_dipole_field(theta):
    """Small (Hertzian) dipole on the z axis: ``F_theta = sin(theta)``."""
    return np.abs(np.sin(np.asarray(theta, float)))


def dipole_field(theta, length_wl=0.5):
    """Thin center-fed dipole of length ``L = length_wl * lambda`` on the z axis
    (sinusoidal current):

        F(theta) = [cos(k L/2 cos(theta)) - cos(k L/2)] / sin(theta)

    normalized to a maximum of 1.  ``length_wl = 0.5`` is the half-wave dipole.
    """
    theta = np.asarray(theta, float)
    kl2 = np.pi * length_wl
    s = np.sin(theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (np.cos(kl2 * np.cos(theta)) - np.cos(kl2)) / s
    f = np.where(np.abs(s) < 1e-12, 0.0, np.abs(f))
    # normalize with a fine grid so the result is independent of the input sampling
    tt = np.linspace(1e-6, np.pi - 1e-6, 4001)
    fmax = np.max(np.abs((np.cos(kl2 * np.cos(tt)) - np.cos(kl2)) / np.sin(tt)))
    return f / fmax


def half_wave_dipole_field(theta):
    """Half-wave dipole: ``F = cos(pi/2 cos(theta)) / sin(theta)``."""
    return dipole_field(theta, 0.5)


def steering_phase(theta0, d_wl):
    """Progressive phase ``beta = -k d cos(theta0)`` that steers a z-axis array's main
    beam to ``theta0`` (``theta0 = pi/2`` is broadside, ``beta = 0``)."""
    return -2 * np.pi * d_wl * np.cos(theta0)


def array_factor(theta, n_elements, d_wl, beta=0.0):
    """Normalized array factor of an N-element uniform linear array along z.

        psi = k d cos(theta) + beta
        |AF| = |sin(N psi / 2) / (N sin(psi / 2))|

    ``d_wl`` is the element spacing in wavelengths; ``beta`` the progressive phase (rad).
    """
    theta = np.asarray(theta, float)
    psi = 2 * np.pi * d_wl * np.cos(theta) + beta
    n = np.arange(n_elements)
    af = np.exp(1j * np.outer(psi, n)).sum(axis=1) / n_elements
    return np.abs(af).reshape(theta.shape)


def grating_lobe_free_spacing(theta0):
    """Largest spacing (in wavelengths) with no grating lobe in visible space when the
    beam is steered to ``theta0``: ``d/lambda < 1 / (1 + |cos(theta0)|)``."""
    return 1.0 / (1.0 + np.abs(np.cos(theta0)))


# ---------------------------------------------------------------------------
# Directivity and beamwidth
# ---------------------------------------------------------------------------

def directivity(U, n_theta=721, n_phi=721):
    """Numerical directivity of a radiation intensity ``U(theta, phi)``:

        D = 4 pi U_max / int_0^{2pi} int_0^pi U(theta, phi) sin(theta) dtheta dphi

    (slide "Directivity").  ``U`` must accept broadcast arrays ``(theta, phi)``.
    Uses the trapezoidal rule on a regular grid.
    """
    th = np.linspace(0, np.pi, n_theta)
    ph = np.linspace(0, 2 * np.pi, n_phi)
    T, P = np.meshgrid(th, ph, indexing="ij")
    u = np.broadcast_to(np.asarray(U(T, P), float), T.shape)
    inner = trapezoid(u * np.sin(T), th, axis=0)
    total = trapezoid(inner, ph)
    return 4 * np.pi * u.max() / total


def directivity_db(U, **kw):
    """Directivity in dBi, ``10 log10 D``."""
    return 10 * np.log10(directivity(U, **kw))


def hpbw(field, theta=None):
    """Half-power beamwidth (rad) of the main lobe of a sampled field pattern.

    ``field`` is either an array of |F(theta)| sampled on ``theta`` or a callable of
    theta.  The -3 dB crossings (|F|^2 = 1/2 of the peak) on both sides of the main-lobe
    maximum are found by linear interpolation.

    Note: the main lobe must lie inside the sampled range.  If the peak sits at an
    endpoint of ``theta`` (e.g. boresight at theta = 0 on a [0, pi] grid), only one
    side is found and the result is half the true beamwidth -- mirror the pattern or
    sample a symmetric range around the peak.
    """
    if theta is None:
        theta = np.linspace(0, np.pi, 20001)
    theta = np.asarray(theta, float)
    f = np.asarray(field(theta) if callable(field) else field, float)
    p = (f / f.max()) ** 2
    i0 = int(np.argmax(p))

    def crossing(step):
        i = i0
        while 0 < i < len(p) - 1 and p[i + step] >= 0.5:
            i += step
        j = i + step
        if not 0 <= j < len(p):
            return theta[i]
        # linear interpolation between samples i (>=0.5) and j (<0.5)
        return theta[i] + (0.5 - p[i]) * (theta[j] - theta[i]) / (p[j] - p[i])

    return abs(crossing(+1) - crossing(-1))


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------

def complex_permittivity(eps_r, sigma_s_per_m, freq_hz):
    """Relative complex permittivity of a lossy ground ``eps_r - j 60 sigma lambda``."""
    return eps_r - 1j * 60.0 * sigma_s_per_m * wavelength(freq_hz)


def fresnel_reflection(grazing, eps_r, polarization="TE"):
    """Fresnel reflection coefficient at a planar boundary (free space -> ground).

    ``grazing`` is the grazing angle psi measured from the surface (as in the two-ray
    geometry); ``eps_r`` may be complex (see :func:`complex_permittivity`).

    * ``"TE"`` / horizontal polarization (E perpendicular to the plane of incidence):
      ``Gamma = (sin psi - sqrt(eps_r - cos^2 psi)) / (sin psi + sqrt(eps_r - cos^2 psi))``
    * ``"TM"`` / vertical polarization (E in the plane of incidence):
      ``Gamma = (eps_r sin psi - sqrt(eps_r - cos^2 psi)) / (eps_r sin psi + sqrt(...))``

    Sign convention: both tend to -1 at grazing incidence (psi -> 0), which is the value
    the flat-earth two-ray model assumes.  (Rappaport's Gamma_parallel has the opposite
    sign because of a different reference direction for the reflected E field; the
    magnitudes are identical.)
    """
    psi = np.asarray(grazing, float)
    s, c2 = np.sin(psi), np.cos(psi) ** 2
    root = np.sqrt(np.asarray(eps_r, complex) - c2)
    pol = polarization.upper()
    if pol in ("TE", "H", "HORIZONTAL", "PERP"):
        return (s - root) / (s + root)
    if pol in ("TM", "V", "VERTICAL", "PARALLEL"):
        return (eps_r * s - root) / (eps_r * s + root)
    raise ValueError("polarization must be 'TE' (horizontal) or 'TM' (vertical)")


def brewster_angle(eps_r):
    """Brewster angle as a grazing angle (rad) for a lossless dielectric:
    ``sin psi_B = 1 / sqrt(eps_r + 1)``; equivalently ``tan theta_B = sqrt(eps_r)`` from
    the normal.  Only TM (vertical) polarization has a reflection null."""
    return np.arcsin(1.0 / np.sqrt(np.asarray(eps_r, float) + 1.0))


# ---------------------------------------------------------------------------
# Two-ray ground reflection
# ---------------------------------------------------------------------------

def two_ray_geometry(d_m, h_b, h_m):
    """Direct and reflected path lengths and the grazing angle for ground distance ``d``."""
    d = np.asarray(d_m, float)
    d_los = np.sqrt(d**2 + (h_b - h_m) ** 2)
    d_ref = np.sqrt(d**2 + (h_b + h_m) ** 2)
    grazing = np.arctan2(h_b + h_m, d)
    return d_los, d_ref, grazing


def two_ray_gain(d_m, freq_hz, h_b, h_m, gamma=-1.0, eps_r=None, polarization="TE"):
    """Exact two-ray received-to-transmitted power ratio ``P_RX / (P_TX G_TX G_RX)``:

        (lambda / 4 pi)^2 | exp(-j k d_los) / d_los + Gamma exp(-j k d_ref) / d_ref |^2

    ``gamma`` is a fixed reflection coefficient (-1 = perfectly conducting earth, the
    lecture's assumption).  If ``eps_r`` is given, Gamma is instead the angle-dependent
    Fresnel coefficient for that ground and ``polarization``.
    """
    lam = wavelength(freq_hz)
    k = 2 * np.pi / lam
    d_los, d_ref, psi = two_ray_geometry(d_m, h_b, h_m)
    g = fresnel_reflection(psi, eps_r, polarization) if eps_r is not None else gamma
    field = np.exp(-1j * k * d_los) / d_los + g * np.exp(-1j * k * d_ref) / d_ref
    return (lam / (4 * np.pi)) ** 2 * np.abs(field) ** 2


def two_ray_gain_slide(d_m, freq_hz, h_b, h_m):
    """Lecture's two-ray formula (flat perfectly conducting earth, equal path amplitudes):
    ``4 (lambda / 4 pi d)^2 sin^2(2 pi h_b h_m / (lambda d))``."""
    d = np.asarray(d_m, float)
    lam = wavelength(freq_hz)
    return 4 * (lam / (4 * np.pi * d)) ** 2 * np.sin(2 * np.pi * h_b * h_m / (lam * d)) ** 2


def two_ray_gain_asymptotic(d_m, h_b, h_m):
    """Large-distance two-ray limit ``h_b^2 h_m^2 / d^4`` (frequency independent)."""
    d = np.asarray(d_m, float)
    return (h_b * h_m) ** 2 / d**4


def free_space_gain(d_m, freq_hz):
    """Friis free-space ratio ``(lambda / 4 pi d)^2``."""
    return (wavelength(freq_hz) / (4 * np.pi * np.asarray(d_m, float))) ** 2


def two_ray_breakpoint(freq_hz, h_b, h_m):
    """Breakpoint (critical) distance ``d_c = 4 h_b h_m / lambda``: the last maximum of
    the two-ray pattern; beyond it power falls as d^-4."""
    return 4 * h_b * h_m / wavelength(freq_hz)


def two_ray_asymptote_crossing(freq_hz, h_b, h_m):
    """Distance where the free-space (d^-2) and d^-4 asymptotes meet:
    ``4 pi h_b h_m / lambda`` (= pi times the breakpoint distance)."""
    return 4 * np.pi * h_b * h_m / wavelength(freq_hz)


# ---------------------------------------------------------------------------
# Knife-edge diffraction and Fresnel zones
# ---------------------------------------------------------------------------

def fresnel_kirchhoff_parameter(h_m, d1_m, d2_m, wavelength_m):
    """Fresnel-Kirchhoff diffraction parameter ``nu = h sqrt(2 (d1 + d2) / (lambda d1 d2))``.
    ``h > 0``: the edge blocks the line of sight."""
    return np.asarray(h_m, float) * np.sqrt(2 * (d1_m + d2_m) / (wavelength_m * d1_m * d2_m))


def knife_edge_field(nu):
    """Complex diffraction coefficient ``F(nu) = E_d / E_o``:

        F(nu) = (1 + j)/2 * int_nu^inf exp(-j pi t^2 / 2) dt

    evaluated with the Fresnel integrals S, C:  the integral equals
    ``(1/2 - C(nu)) - j (1/2 - S(nu))``.
    """
    S, C = fresnel(np.asarray(nu, float))
    return (1 + 1j) / 2 * ((0.5 - C) - 1j * (0.5 - S))


def knife_edge_gain_db(nu):
    """Exact knife-edge diffraction gain ``G_d = 20 log10 |F(nu)|`` (dB, <= ~1.2 dB)."""
    return 20 * np.log10(np.abs(knife_edge_field(nu)))


def knife_edge_gain_lee_db(nu):
    """Lee's piecewise approximation to the knife-edge diffraction gain (dB)."""
    nu = np.asarray(nu, float)
    out = np.empty_like(nu)
    a = nu <= -1
    b = (nu > -1) & (nu <= 0)
    c = (nu > 0) & (nu <= 1)
    e = (nu > 1) & (nu <= 2.4)
    f = nu > 2.4
    out[a] = 0.0
    out[b] = 20 * np.log10(0.5 - 0.62 * nu[b])
    out[c] = 20 * np.log10(0.5 * np.exp(-0.95 * nu[c]))
    out[e] = 20 * np.log10(0.4 - np.sqrt(0.1184 - (0.38 - 0.1 * nu[e]) ** 2))
    out[f] = 20 * np.log10(0.225 / nu[f])
    return out


def fresnel_zone_radius(n, d1_m, d2_m, wavelength_m):
    """Radius of the n-th Fresnel zone at distances d1, d2 from the terminals:
    ``r_n = sqrt(n lambda d1 d2 / (d1 + d2))``.  Note ``nu = sqrt(2) h / r_1``."""
    d1 = np.asarray(d1_m, float)
    d2 = np.asarray(d2_m, float)
    return np.sqrt(n * wavelength_m * d1 * d2 / (d1 + d2))
