"""Unit conversions used throughout the course."""

import numpy as np

SPEED_OF_LIGHT = 299_792_458.0  # m/s


def db_to_linear(x_db):
    """Power ratio in dB -> linear."""
    return 10.0 ** (np.asarray(x_db) / 10.0)


def linear_to_db(x):
    """Linear power ratio -> dB."""
    return 10.0 * np.log10(np.asarray(x))


def dbm_to_watts(p_dbm):
    """Power in dBm -> watts."""
    return 10.0 ** ((np.asarray(p_dbm) - 30.0) / 10.0)


def watts_to_dbm(p_w):
    """Power in watts -> dBm."""
    return 10.0 * np.log10(np.asarray(p_w)) + 30.0


def wavelength(freq_hz):
    """Free-space wavelength (m) of a carrier at ``freq_hz``."""
    return SPEED_OF_LIGHT / np.asarray(freq_hz)
