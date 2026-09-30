"""Propagation models.

Public course module (used by the Week 1 lecture notebook and HW1).
"""

import numpy as np

from .units import wavelength


def free_space_path_loss_db(distance_m, freq_hz):
    """Free-space path loss 20 log10(4 pi d / lambda) in dB."""
    return 20 * np.log10(4 * np.pi * np.asarray(distance_m) / wavelength(freq_hz))


def friis_received_power_dbm(tx_power_dbm, distance_m, freq_hz,
                             tx_gain_dbi=0.0, rx_gain_dbi=0.0, system_loss_db=0.0):
    """Friis free-space received power (dBm):

    P_RX = P_TX + G_TX + G_RX - L_sys - 20 log10(4 pi d / lambda)
    """
    return (tx_power_dbm + tx_gain_dbi + rx_gain_dbi - system_loss_db
            - free_space_path_loss_db(distance_m, freq_hz))
