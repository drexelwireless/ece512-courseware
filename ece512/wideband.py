"""Small, explicit wideband-channel models for ECE 512.

The routines here are intentionally transparent: students can move from a
delay/power table to an impulse response and frequency response without first
learning a standards toolbox.  The bundled profiles are *TDL-style teaching
examples*, not claims of 3GPP conformance.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TappedDelayLine:
    """One realization of a time-varying tapped-delay-line channel."""

    delays_s: np.ndarray
    average_powers_db: np.ndarray
    times_s: np.ndarray
    taps: np.ndarray

    def __post_init__(self):
        delays = np.asarray(self.delays_s, dtype=float)
        powers = np.asarray(self.average_powers_db, dtype=float)
        times = np.asarray(self.times_s, dtype=float)
        taps = np.asarray(self.taps, dtype=complex)
        if delays.ndim != 1 or len(delays) == 0 or np.any(delays < 0):
            raise ValueError("delays_s must be a nonempty vector of nonnegative delays")
        if powers.shape != delays.shape:
            raise ValueError("average_powers_db must match delays_s")
        if times.ndim != 1 or taps.shape != (len(times), len(delays)):
            raise ValueError("taps must have shape (len(times_s), len(delays_s))")


def teaching_profile(name="urban"):
    """Return delays and relative powers for a compact TDL-style example."""

    profiles = {
        "indoor": ([0, 15, 40, 85, 150], [0, -1, -4, -8, -13]),
        "urban": ([0, 80, 210, 470, 950, 1700], [0, -1.5, -3, -6, -9, -14]),
        "hilly": ([0, 120, 380, 1100, 2600, 4100], [0, -2, -4, -7, -10, -15]),
    }
    try:
        delays_ns, powers_db = profiles[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown profile {name!r}; choose {', '.join(profiles)}") from exc
    return np.asarray(delays_ns, dtype=float) * 1e-9, np.asarray(powers_db, dtype=float)


def create_wideband_channel(
    delays_s,
    average_powers_db,
    *,
    n_times=1,
    sample_interval_s=1e-3,
    max_doppler_hz=0.0,
    rng=0,
):
    """Create a normalized Rayleigh TDL with a sum of Doppler tones per tap.

    The simple Doppler evolution is chosen for interpretability, not as a full
    Clarke/Jakes simulator.  Week 5 can inspect the taps; Week 7 can reuse the
    exact object to study frequency selectivity and cyclic prefix length.
    """

    delays = np.asarray(delays_s, dtype=float)
    powers_db = np.asarray(average_powers_db, dtype=float)
    if delays.ndim != 1 or len(delays) == 0 or np.any(delays < 0):
        raise ValueError("delays_s must be a nonempty vector of nonnegative delays")
    if powers_db.shape != delays.shape:
        raise ValueError("average_powers_db must match delays_s")
    if n_times < 1 or sample_interval_s <= 0 or max_doppler_hz < 0:
        raise ValueError("use n_times >= 1, sample_interval_s > 0, and max_doppler_hz >= 0")

    generator = np.random.default_rng(rng)
    linear = 10.0 ** (powers_db / 10.0)
    linear /= linear.sum()
    n_sinusoids = 24
    dopplers = generator.uniform(-max_doppler_hz, max_doppler_hz, (len(delays), n_sinusoids))
    phases = generator.uniform(0, 2 * np.pi, (len(delays), n_sinusoids))
    times = np.arange(n_times, dtype=float) * sample_interval_s
    oscillators = np.exp(1j * (2 * np.pi * times[:, None, None] * dopplers[None, :, :] + phases[None, :, :]))
    taps = np.sqrt(linear / n_sinusoids)[None, :] * oscillators.sum(axis=2)
    return TappedDelayLine(delays, powers_db, times, taps)


def rms_delay_spread(delays_s, powers_db):
    """Return the RMS delay spread of a power-delay profile in seconds."""

    delays = np.asarray(delays_s, dtype=float)
    powers = 10.0 ** (np.asarray(powers_db, dtype=float) / 10.0)
    if delays.shape != powers.shape or delays.ndim != 1 or powers.sum() <= 0:
        raise ValueError("delays_s and powers_db must be same-length vectors")
    weights = powers / powers.sum()
    mean = np.sum(weights * delays)
    return float(np.sqrt(np.sum(weights * (delays - mean) ** 2)))


def mean_excess_delay(delays_s, powers_db):
    """Return the power-weighted mean excess delay in seconds."""

    delays = np.asarray(delays_s, dtype=float)
    powers = 10.0 ** (np.asarray(powers_db, dtype=float) / 10.0)
    if delays.shape != powers.shape or delays.ndim != 1 or powers.sum() <= 0:
        raise ValueError("delays_s and powers_db must be same-length vectors")
    return float(np.sum(delays * powers) / powers.sum())


def coherence_bandwidth(delays_s, powers_db, correlation=0.5):
    """Rule-of-thumb coherence bandwidth (Hz) from RMS delay spread.

    ``correlation=0.5`` uses roughly ``1/(5 sigma_tau)``; ``0.9`` uses the
    more conservative ``1/(50 sigma_tau)`` convention.
    """

    sigma = rms_delay_spread(delays_s, powers_db)
    if sigma == 0:
        return np.inf
    if correlation == 0.5:
        return 1.0 / (5.0 * sigma)
    if correlation == 0.9:
        return 1.0 / (50.0 * sigma)
    raise ValueError("correlation must be 0.5 or 0.9")


def frequency_response(channel, frequencies_hz, time_index=0):
    """Evaluate H(f) for one time sample of a :class:`TappedDelayLine`."""

    frequencies = np.asarray(frequencies_hz, dtype=float)
    if not 0 <= time_index < len(channel.times_s):
        raise IndexError("time_index is outside the channel realization")
    phase = np.exp(-1j * 2 * np.pi * frequencies[..., None] * channel.delays_s)
    return np.sum(phase * channel.taps[time_index], axis=-1)
