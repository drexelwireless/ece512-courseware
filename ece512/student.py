"""Per-student assignment parameters.

Each student receives their own parameter set, derived deterministically from
their student ID and the assignment name, so results differ between students
but are reproducible by the autograder.
"""

import hashlib

import numpy as np


def _check_id(student_id):
    if student_id is Ellipsis or not str(student_id).strip() or str(student_id).strip() == "...":
        raise ValueError("Set STUDENT_ID to your Drexel user ID (e.g. 'abc123') first.")


def student_seed(student_id, assignment):
    """Deterministic 64-bit seed for ``student_id`` on ``assignment``."""
    _check_id(student_id)
    key = f"ece512:{assignment}:{str(student_id).strip().lower()}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def student_rng(student_id, assignment):
    """NumPy random generator unique to this student and assignment.

    Homework notebooks derive every personal parameter from this generator, and the
    autograder's hidden tests re-derive them from STUDENT_ID, so results can be checked
    against the student's own parameters.
    """
    return np.random.default_rng(student_seed(student_id, assignment))


def _rng(student_id, assignment):
    return student_rng(student_id, assignment)


def student_params(student_id, assignment):
    """Return the parameter dictionary for ``student_id`` on ``assignment``."""
    rng = _rng(student_id, assignment)
    if assignment == "hw01":
        radius = float(rng.choice([100, 150, 200, 250, 300, 400, 500]))
        start_angle = rng.uniform(0, 2 * np.pi)
        # Trajectory starts in a first-tier cell and crosses the central cluster.
        start = 2.5 * radius * np.exp(1j * start_angle)
        end = -start * np.exp(1j * rng.uniform(-0.6, 0.6))
        return {
            "cell_radius_m": radius,
            "carrier_mhz": float(rng.choice([700, 850, 1900, 2100, 2600, 3500])),
            "tx_power_dbm": float(rng.choice([30, 33, 36, 40, 43, 46])),
            "tx_gain_dbi": float(rng.choice([8, 10, 12, 15, 17])),
            "rx_gain_dbi": float(rng.choice([0, 2])),
            "system_loss_db": float(rng.choice([1, 2, 3])),
            "trajectory_start": complex(np.round(start, 1)),
            "trajectory_end": complex(np.round(end, 1)),
            "n_positions": 200,
        }
    raise KeyError(f"No parameters defined for assignment {assignment!r}")
