"""Pedagogical NR-inspired reference radio used across ECE 512.

This module deliberately implements a *link abstraction*, not a standards-
compliant NR modem.  The fixed-MCS BLER curve turns post-processing SINR into
packet outcomes so propagation and system effects remain the focus.
"""

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class RadioScenario:
    """Small multi-cell scene for the recurring course example."""

    serving_gnb: complex
    interferers: np.ndarray
    ue: complex
    image: np.ndarray


@dataclass(frozen=True)
class LinkResult:
    """Observable outputs of one reference-radio evaluation."""

    desired_power_dbm: float
    interference_power_dbm: float
    noise_power_dbm: float
    processing_gain_db: float
    sinr_db: float
    bler: float
    outage: bool
    resource_available: bool
    goodput_fraction: float
    packet_success: np.ndarray
    received_image: np.ndarray


@dataclass(frozen=True)
class LinkTrace:
    """A course-model trajectory translated into radio/application metrics."""

    driver: np.ndarray
    desired_power_dbm: np.ndarray
    interference_power_dbm: np.ndarray
    noise_power_dbm: np.ndarray
    processing_gain_db: np.ndarray
    availability: np.ndarray
    sinr_db: np.ndarray
    bler: np.ndarray
    expected_goodput: np.ndarray
    driver_label: str
    scene_kind: str = "generic"
    scene_data: dict = field(default_factory=dict)


def dbm_to_mw(power_dbm):
    """Convert dBm to mW, accepting scalars or arrays."""

    return 10.0 ** (np.asarray(power_dbm, dtype=float) / 10.0)


def mw_to_dbm(power_mw):
    """Convert positive mW to dBm."""

    p = np.asarray(power_mw, dtype=float)
    if np.any(p <= 0):
        raise ValueError("power_mw must be positive")
    return 10.0 * np.log10(p)


def fixed_mcs_bler(sinr_db, midpoint_db=5.0, transition_db=1.35):
    """Smooth fixed-MCS BLER abstraction.

    ``midpoint_db`` is the SINR where BLER is 0.5; ``transition_db`` controls
    the waterfall width.  These are pedagogical parameters, not an NR MCS table.
    """

    if transition_db <= 0:
        raise ValueError("transition_db must be positive")
    z = np.clip((np.asarray(sinr_db, dtype=float) - midpoint_db) / transition_db, -60, 60)
    out = 1.0 / (1.0 + np.exp(z))
    return float(out) if out.ndim == 0 else out


def reference_image(height=96, width=128):
    """Load the course's Drexel space-opera image as a small RGB array."""

    if height < 16 or width < 16:
        raise ValueError("reference image must be at least 16 by 16 pixels")
    from importlib import resources
    import matplotlib.image as mpimg

    asset = resources.files("ece512").joinpath("assets/reference-radio-drexel-space.png")
    with asset.open("rb") as stream:
        source = np.asarray(mpimg.imread(stream, format="png"), dtype=float)[..., :3]

    # Center-crop to the requested aspect ratio, then use deterministic nearest-
    # neighbour sampling. Bold shapes survive the intentionally tiny radio image.
    src_h, src_w = source.shape[:2]
    target_ratio = width / height
    if src_w / src_h > target_ratio:
        cropped_w = max(1, int(round(src_h * target_ratio)))
        left = (src_w - cropped_w) // 2
        source = source[:, left:left + cropped_w]
    else:
        cropped_h = max(1, int(round(src_w / target_ratio)))
        top = (src_h - cropped_h) // 2
        source = source[top:top + cropped_h]
    rows = np.linspace(0, source.shape[0] - 1, height).astype(int)
    cols = np.linspace(0, source.shape[1] - 1, width).astype(int)
    return source[rows[:, None], cols[None, :], :]


def campus_scenario(ue=350 + 180j):
    """Return the deterministic scene shared by slides and notebooks."""

    return RadioScenario(
        serving_gnb=0 + 0j,
        interferers=np.array([-650 + 420j, 720 + 360j, 180 - 760j], dtype=complex),
        ue=complex(ue),
        image=reference_image(),
    )


def _packetize_image(image, bler, rng, packet_rows=8):
    image = np.asarray(image, dtype=float)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("image must have shape (height, width, 3 or 4)")
    if packet_rows <= 0:
        raise ValueError("packet_rows must be positive")
    n_packets = int(np.ceil(image.shape[0] / packet_rows))
    success = rng.random(n_packets) >= bler
    received = image.copy()
    for packet, ok in enumerate(success):
        if not ok:
            lo = packet * packet_rows
            hi = min((packet + 1) * packet_rows, image.shape[0])
            received[lo:hi, :, :3] = 0.16
    return success, received


def evaluate_link(
    desired_power_dbm,
    interference_power_dbm=(),
    noise_power_dbm=-100.0,
    *,
    processing_gain_db=0.0,
    resource_available=True,
    image=None,
    packet_rows=8,
    rng=0,
    bler_midpoint_db=5.0,
    bler_transition_db=1.35,
    outage_bler=0.10,
):
    """Evaluate one fixed-MCS reference-radio snapshot.

    Interferers may be a scalar or a sequence of received powers in dBm.
    ``processing_gain_db`` represents beamforming, combining, or another
    post-processing SINR improvement introduced by the week's topic.
    """

    interference = np.asarray(interference_power_dbm, dtype=float)
    interference_mw = float(np.sum(dbm_to_mw(interference))) if interference.size else 0.0
    interference_dbm = float(mw_to_dbm(interference_mw)) if interference_mw > 0 else -np.inf
    denominator_mw = interference_mw + float(dbm_to_mw(noise_power_dbm))
    sinr_db = float(desired_power_dbm - mw_to_dbm(denominator_mw) + processing_gain_db)
    bler = fixed_mcs_bler(sinr_db, bler_midpoint_db, bler_transition_db)
    if not resource_available:
        bler = 1.0
    if image is None:
        image = reference_image()
    generator = np.random.default_rng(rng)
    packet_success, received = _packetize_image(image, bler, generator, packet_rows)
    goodput = float(np.mean(packet_success)) if resource_available else 0.0
    return LinkResult(
        desired_power_dbm=float(desired_power_dbm),
        interference_power_dbm=interference_dbm,
        noise_power_dbm=float(noise_power_dbm),
        processing_gain_db=float(processing_gain_db),
        sinr_db=sinr_db,
        bler=float(bler),
        outage=bool((bler > outage_bler) or not resource_available),
        resource_available=bool(resource_available),
        goodput_fraction=goodput,
        packet_success=packet_success,
        received_image=received,
    )


def aggregate_interference_dbm(powers_dbm, axis=-1):
    """Add independent interference powers in linear units and return dBm."""

    powers = np.asarray(powers_dbm, dtype=float)
    total = np.sum(dbm_to_mw(powers), axis=axis)
    with np.errstate(divide="ignore"):
        return np.where(total > 0, 10.0 * np.log10(total), -np.inf)


def evaluate_trace(
    driver,
    desired_power_dbm,
    interference_power_dbm=-np.inf,
    noise_power_dbm=-100.0,
    *,
    processing_gain_db=0.0,
    availability=1.0,
    driver_label="course-model input",
    bler_midpoint_db=5.0,
    bler_transition_db=1.35,
):
    """Translate a trajectory from another course model into link outcomes.

    ``availability`` may be Boolean or a probability/fraction in [0, 1]. It
    represents admission, scheduling, or supportable-rate limitations outside
    the fixed-MCS physical link. All inputs broadcast against ``driver``.
    """

    x = np.asarray(driver, dtype=float)
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("driver must be a one-dimensional trajectory with at least two points")
    try:
        desired, interference, noise, gain, available, _ = np.broadcast_arrays(
            np.asarray(desired_power_dbm, dtype=float),
            np.asarray(interference_power_dbm, dtype=float),
            np.asarray(noise_power_dbm, dtype=float),
            np.asarray(processing_gain_db, dtype=float),
            np.asarray(availability, dtype=float),
            x,
        )
    except ValueError as exc:
        raise ValueError("trace inputs must be scalar or broadcast to the driver shape") from exc
    if np.any((available < 0) | (available > 1)):
        raise ValueError("availability must lie between 0 and 1")
    interference_mw = np.where(np.isfinite(interference), dbm_to_mw(interference), 0.0)
    denominator = interference_mw + dbm_to_mw(noise)
    sinr = desired - mw_to_dbm(denominator) + gain
    bler = fixed_mcs_bler(sinr, bler_midpoint_db, bler_transition_db)
    goodput = available * (1.0 - bler)
    return LinkTrace(
        driver=x,
        desired_power_dbm=np.asarray(desired, dtype=float),
        interference_power_dbm=np.asarray(interference, dtype=float),
        noise_power_dbm=np.asarray(noise, dtype=float),
        processing_gain_db=np.asarray(gain, dtype=float),
        availability=np.asarray(available, dtype=float),
        sinr_db=np.asarray(sinr, dtype=float),
        bler=np.asarray(bler, dtype=float),
        expected_goodput=np.asarray(goodput, dtype=float),
        driver_label=str(driver_label),
    )


def trace_from_sinr(
    driver,
    sinr_db,
    *,
    availability=1.0,
    driver_label="course-model input",
    nominal_desired_dbm=-80.0,
    bler_midpoint_db=5.0,
    bler_transition_db=1.35,
):
    """Build a trace when the preceding analysis directly produced SINR/SIR."""

    sinr = np.asarray(sinr_db, dtype=float)
    desired = np.broadcast_to(float(nominal_desired_dbm), sinr.shape)
    equivalent_interference = desired - sinr
    return evaluate_trace(
        driver, desired, equivalent_interference, noise_power_dbm=-300.0,
        availability=availability, driver_label=driver_label,
        bler_midpoint_db=bler_midpoint_db, bler_transition_db=bler_transition_db,
    )


def attach_scene(trace, kind, **data):
    """Attach a lightweight physical-situation diagram to a link trace.

    Supported kinds are ``cellular``, ``traffic``, ``angular``, ``multipath``,
    ``branches`` and ``mimo``. The diagram is explanatory rather than a second
    propagation model: its animation index is locked to the trace index.
    """

    supported = {"cellular", "traffic", "angular", "multipath", "branches", "mimo"}
    if kind not in supported:
        raise ValueError(f"unknown physical scene {kind!r}; choose from {sorted(supported)}")
    return replace(trace, scene_kind=kind, scene_data=dict(data))


def _scene_value(value, index):
    values = np.asarray(value)
    if values.ndim and len(values) > 1:
        return values[min(index, len(values) - 1)]
    return values.item() if values.ndim == 0 else values[0]


def _draw_physical_scene(axis, trace, index):
    """Draw the physical interpretation corresponding to one trace sample."""

    from matplotlib.patches import Circle

    axis.clear()
    data = trace.scene_data
    kind = trace.scene_kind
    axis.set_title("1. Physical situation")

    if kind == "cellular":
        gnb = np.asarray(data.get("gnb_xy", [0 + 0j]), dtype=complex).ravel()
        ue = complex(_scene_value(data.get("ue_xy", trace.driver + 0j), index))
        serving = data.get("serving_index", 0)
        serving = int(_scene_value(serving, index)) if np.asarray(serving).size else 0
        serving = int(np.clip(serving, 0, len(gnb) - 1))
        axis.scatter(gnb.real, gnb.imag, marker="^", s=70, facecolors="none",
                     edgecolors="#D55E00", label="gNB")
        axis.scatter([gnb[serving].real], [gnb[serving].imag], marker="^", s=110,
                     color="#0072B2", label="serving gNB")
        axis.plot([gnb[serving].real, ue.real], [gnb[serving].imag, ue.imag], color="#0072B2", lw=2)
        for site in gnb:
            if site != gnb[serving]:
                axis.plot([site.real, ue.real], [site.imag, ue.imag], "--", color="#D55E00", alpha=.28)
        route = np.asarray(data.get("ue_xy", []), dtype=complex).ravel()
        if len(route) > 1:
            axis.plot(route.real, route.imag, color="black", alpha=.25, lw=1)
        axis.scatter([ue.real], [ue.imag], s=70, color="#E69F00", edgecolor="black", label="UE", zorder=5)
        axis.set_aspect("equal", adjustable="datalim"); axis.legend(fontsize=7, loc="best")
        axis.set(xlabel="east", ylabel="north")

    elif kind == "traffic":
        max_users = int(data.get("max_users", 18))
        theta = np.linspace(0, 2*np.pi, max_users, endpoint=False)
        radii = .55 + .32*(np.arange(max_users) % 3)
        xy = radii * np.exp(1j*theta)
        offered = max(1, int(round(max_users * index / max(len(trace.driver) - 1, 1))))
        admitted = int(round(offered * trace.availability[index]))
        colors = ["#0072B2" if k < admitted else "#D55E00" for k in range(offered)]
        axis.add_patch(Circle((0, 0), 1.05, fill=False, color="#0072B2"))
        axis.scatter([0], [0], marker="^", s=100, color="#0072B2")
        axis.scatter(xy.real[:offered], xy.imag[:offered], s=40, c=colors, edgecolor="black", linewidth=.3)
        axis.text(0, -1.28, f"{admitted} admitted / {offered} offered", ha="center", fontsize=9)
        axis.set(xlim=(-1.45, 1.45), ylim=(-1.45, 1.25), aspect="equal"); axis.axis("off")

    elif kind == "angular":
        angle = np.radians(float(_scene_value(data.get("angle_deg", trace.driver), index)))
        moving = complex(data.get("center", 0+0j)) + float(data.get("radius", 1.0))*np.exp(1j*angle)
        fixed = complex(data.get("fixed_ue", 1+0j))
        moving_is_ue = data.get("moving_role", "interferer") == "ue"
        fixed_color = "#D55E00" if moving_is_ue else "#E69F00"
        moving_color = "#E69F00" if moving_is_ue else "#D55E00"
        axis.scatter([0], [0], marker="^", s=110, color="#0072B2", label="array / gNB")
        axis.plot([0, fixed.real], [0, fixed.imag], "--" if moving_is_ue else "-", color=fixed_color,
                  lw=2, label="interferer" if moving_is_ue else "desired UE")
        axis.plot([0, moving.real], [0, moving.imag], "-" if moving_is_ue else "--", color=moving_color,
                  lw=2, label="moving UE" if moving_is_ue else "moving interferer")
        axis.scatter([fixed.real], [fixed.imag], color=fixed_color, s=60)
        axis.scatter([moving.real], [moving.imag], color=moving_color, s=60)
        axis.add_patch(Circle((0, 0), float(data.get("radius", 1.0)),
                              fill=False, color="black", alpha=.15))
        axis.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35), aspect="equal"); axis.axis("off")
        axis.legend(fontsize=7, loc="lower left")

    elif kind == "multipath":
        tx, rx = -1.15+0j, 1.15+0j
        scatterers = np.asarray(data.get("scatterers", [-.35+.75j, .35-.7j]), dtype=complex)
        axis.scatter([tx.real], [tx.imag], marker="^", s=100, color="#0072B2")
        axis.scatter([rx.real], [rx.imag], s=70, color="#E69F00", edgecolor="black")
        axis.plot([tx.real, rx.real], [0, 0], color="#0072B2", lw=2)
        for number, point in enumerate(scatterers):
            alpha = .25 + .7*((index + number) % 6)/5
            axis.plot([tx.real, point.real, rx.real], [0, point.imag, 0], color="#D55E00", alpha=alpha)
            axis.scatter([point.real], [point.imag], marker="s", s=35, color="0.45")
        axis.text(tx.real, -.2, "gNB", ha="center", fontsize=8); axis.text(rx.real, -.2, "UE", ha="center", fontsize=8)
        axis.set(xlim=(-1.45, 1.45), ylim=(-1.05, 1.05), aspect="equal"); axis.axis("off")

    elif kind == "branches":
        count = max(1, int(round(float(_scene_value(data.get("count", trace.driver), index)))))
        axis.scatter([-1], [0], marker="^", s=110, color="#0072B2")
        ys = np.linspace(-.8, .8, max(count, 2))[:count]
        for y in ys:
            axis.plot([-1, .8], [0, y], color="#009E73", alpha=.75)
            axis.scatter([.8], [y], marker="|", s=160, color="#E69F00")
        axis.text(.8, -.98, f"{count} diversity branch{'es' if count != 1 else ''}", ha="center", fontsize=9)
        axis.set(xlim=(-1.3, 1.2), ylim=(-1.15, 1.15)); axis.axis("off")

    elif kind == "mimo":
        ratio = float(_scene_value(data.get("mode_ratio", trace.driver), index))
        ratio_label = data.get("ratio_label", "weak/strong mode")
        tx_y, rx_y = np.array([.5, -.5]), np.array([.5, -.5])
        axis.scatter(np.full(2, -1.0), tx_y, marker="^", s=90, color="#0072B2")
        axis.scatter(np.full(2, 1.0), rx_y, marker="|", s=170, color="#E69F00")
        for i, ty in enumerate(tx_y):
            for j, ry in enumerate(rx_y):
                strength = 1.8 if i == j else .4 + 1.4*ratio
                axis.plot([-1, 1], [ty, ry], color="#009E73", lw=strength, alpha=.35+.55*ratio)
        axis.text(0, -.93, f"{ratio_label} = {ratio:.2f}", ha="center", fontsize=9)
        axis.set(xlim=(-1.35, 1.35), ylim=(-1.1, 1.1)); axis.axis("off")

    else:
        axis.text(.5, .58, "physical scenario", ha="center", va="center", transform=axis.transAxes,
                  fontweight="bold", color="#0072B2")
        axis.text(.5, .38, f"{trace.driver_label}\n= {trace.driver[index]:.3g}", ha="center", va="center",
                  transform=axis.transAxes)
        axis.axis("off")


def _trace_image(trace, index, image, rng=0, packet_rows=8):
    failure_probability = float(1.0 - trace.expected_goodput[index])
    return _packetize_image(image, failure_probability, np.random.default_rng(rng + int(index)), packet_rows)[1]


def plot_trace(trace, *, title="Course model → reference-radio performance", image=None, snapshot_indices=None):
    """Show the complete causal bridge and three application snapshots."""

    import matplotlib.pyplot as plt

    if image is None:
        image = reference_image()
    n = len(trace.driver)
    if snapshot_indices is None:
        snapshot_indices = [0, n // 2, n - 1]
    snapshot_indices = [int(np.clip(i, 0, n - 1)) for i in snapshot_indices]

    fig = plt.figure(figsize=(13, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.15, 1.0))
    ax_power = fig.add_subplot(grid[0, 0])
    ax_quality = fig.add_subplot(grid[0, 1])
    ax_chain = fig.add_subplot(grid[0, 2])
    image_axes = [fig.add_subplot(grid[1, i]) for i in range(3)]

    x = trace.driver
    if np.any(np.isfinite(trace.desired_power_dbm)):
        ax_power.plot(x, trace.desired_power_dbm, color="#0072B2", label="desired power")
    if np.any(np.isfinite(trace.interference_power_dbm)):
        ax_power.plot(x, trace.interference_power_dbm, "--", color="#D55E00", label="aggregate interference")
    if np.ptp(trace.processing_gain_db) > 1e-9 or np.any(trace.processing_gain_db != 0):
        ax_power.plot(x, trace.processing_gain_db, ":", color="#CC79A7", label="processing gain (dB)")
    ax_avail = None
    if np.ptp(trace.availability) > 1e-9 or np.any(trace.availability < 1):
        ax_avail = ax_power.twinx()
        ax_avail.plot(x, trace.availability, color="#E69F00", alpha=.75, label="resource/rate availability")
        ax_avail.set_ylim(-.03, 1.03); ax_avail.set_ylabel("availability")
    ax_power.set(xlabel=trace.driver_label, ylabel="power (dBm) or gain (dB)", title="1. Topic model sets receiver inputs")
    ax_power.grid(alpha=.25)
    input_lines = ax_power.lines + ([] if ax_avail is None else ax_avail.lines)
    ax_power.legend(input_lines, [line.get_label() for line in input_lines], fontsize=8, loc="best")

    ax_quality.plot(x, trace.sinr_db, color="#009E73", lw=2, label="post-processing SINR (dB)")
    ax_q2 = ax_quality.twinx()
    ax_q2.plot(x, trace.bler, "--", color="#D55E00", label="BLER")
    ax_q2.plot(x, trace.expected_goodput, color="#0072B2", label="expected goodput fraction")
    ax_q2.set_ylim(-.03, 1.03); ax_q2.set_ylabel("probability / fraction")
    ax_quality.set(xlabel=trace.driver_label, ylabel="SINR (dB)", title="2. Link abstraction predicts service")
    ax_quality.grid(alpha=.25)
    lines = ax_quality.lines + ax_q2.lines
    ax_quality.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="best")

    ax_chain.axis("off")
    ax_chain.text(.5, .88, trace.driver_label.upper(), ha="center", va="center", fontweight="bold",
                  color="#0072B2", wrap=True)
    ax_chain.text(.5, .67, "↓", ha="center", va="center", fontsize=20)
    ax_chain.text(.5, .52, "received powers / processing / resources", ha="center", va="center", wrap=True)
    ax_chain.text(.5, .34, "↓", ha="center", va="center", fontsize=20)
    ax_chain.text(.5, .17, "SINR → BLER → packets → image", ha="center", va="center", fontweight="bold", color="#009E73")
    ax_chain.set_title("The connective tissue")

    for axis, index in zip(image_axes, snapshot_indices):
        axis.imshow(_trace_image(trace, index, image, rng=512))
        axis.set_title(
            f"{trace.driver_label} = {trace.driver[index]:.3g}\n"
            f"SINR {trace.sinr_db[index]:.1f} dB; goodput {100*trace.expected_goodput[index]:.0f}%",
            fontsize=9,
        )
        axis.axis("off")
    fig.suptitle(title, fontweight="bold")
    return fig


def animate_trace(trace, *, title="Reference-radio trajectory", image=None, max_frames=36, interval_ms=180):
    """Animate topic inputs, radio metrics, and delivered image in lockstep."""

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if image is None:
        image = reference_image()
    frame_indices = np.unique(np.linspace(0, len(trace.driver) - 1, min(max_frames, len(trace.driver))).astype(int))
    fig, (ax_scene, ax_topic, ax_curve, ax_image) = plt.subplots(1, 4, figsize=(18, 4.3), constrained_layout=True)
    _draw_physical_scene(ax_scene, trace, 0)

    if np.any(np.isfinite(trace.desired_power_dbm)):
        ax_topic.plot(trace.driver, trace.desired_power_dbm, color="#0072B2", label="desired power")
    if np.any(np.isfinite(trace.interference_power_dbm)):
        ax_topic.plot(trace.driver, trace.interference_power_dbm, "--", color="#D55E00",
                      label="aggregate interference")
    if np.ptp(trace.processing_gain_db) > 1e-9 or np.any(trace.processing_gain_db != 0):
        ax_topic.plot(trace.driver, trace.processing_gain_db, ":", color="#CC79A7", label="processing gain")
    ax_avail = None
    if np.ptp(trace.availability) > 1e-9 or np.any(trace.availability < 1):
        ax_avail = ax_topic.twinx()
        ax_avail.plot(trace.driver, trace.availability, color="#E69F00", label="availability")
        ax_avail.set(ylim=(-.03, 1.03), ylabel="availability")
    topic_lines = ax_topic.lines + ([] if ax_avail is None else ax_avail.lines)
    ax_topic.legend(topic_lines, [line.get_label() for line in topic_lines], fontsize=8, loc="best")
    topic_cursor = ax_topic.axvline(trace.driver[0], color="#009E73", lw=2)
    ax_topic.set(xlabel=trace.driver_label, ylabel="power (dBm) / gain (dB)",
                 title="2. Notebook model changes")
    ax_topic.grid(alpha=.25)

    ax_curve.plot(trace.driver, trace.sinr_db, color="#009E73", lw=2, label="SINR (dB)")
    ax_fraction = ax_curve.twinx()
    ax_fraction.plot(trace.driver, trace.expected_goodput, color="#0072B2", label="expected goodput")
    ax_fraction.plot(trace.driver, trace.bler, "--", color="#D55E00", label="BLER")
    ax_fraction.set(ylim=(-.03, 1.03), ylabel="fraction")
    curve_lines = ax_curve.lines + ax_fraction.lines
    ax_curve.legend(curve_lines, [line.get_label() for line in curve_lines], fontsize=8, loc="best")
    curve_cursor = ax_curve.axvline(trace.driver[0], color="#009E73", lw=2)
    marker, = ax_fraction.plot([], [], "o", color="#0072B2", ms=8)
    ax_curve.set(xlabel=trace.driver_label, ylabel="SINR (dB)", title="3. Reference radio responds")
    ax_curve.grid(alpha=.25)
    shown = ax_image.imshow(image)
    ax_image.axis("off")
    fig.suptitle(title, fontweight="bold")

    def update(frame):
        index = int(frame_indices[frame])
        _draw_physical_scene(ax_scene, trace, index)
        topic_cursor.set_xdata([trace.driver[index], trace.driver[index]])
        curve_cursor.set_xdata([trace.driver[index], trace.driver[index]])
        marker.set_data([trace.driver[index]], [trace.expected_goodput[index]])
        shown.set_data(_trace_image(trace, index, image, rng=512))
        ax_image.set_title(
            f"4. Delivered image\n{trace.driver_label} = {trace.driver[index]:.3g}\n"
            f"SINR {trace.sinr_db[index]:.1f} dB | BLER {trace.bler[index]:.2f} | "
            f"goodput {100*trace.expected_goodput[index]:.0f}%"
        )
        return topic_cursor, curve_cursor, marker, shown

    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=interval_ms, blit=False)
    plt.close(fig)
    return animation


def dashboard(result, scenario=None, *, title="NR-inspired reference radio"):
    """Plot the common network, metric, and application dashboard."""

    import matplotlib.pyplot as plt

    if scenario is None:
        scenario = campus_scenario()
    fig = plt.figure(figsize=(11.5, 5.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.0, 1.15))
    ax_map = fig.add_subplot(grid[:, 0])
    ax_metric = fig.add_subplot(grid[0, 1:])
    ax_tx = fig.add_subplot(grid[1, 1])
    ax_rx = fig.add_subplot(grid[1, 2])

    ax_map.scatter([scenario.serving_gnb.real], [scenario.serving_gnb.imag], marker="^", s=130,
                   color="#0072B2", label="serving gNB")
    ax_map.scatter(scenario.interferers.real, scenario.interferers.imag, marker="^", s=90,
                   facecolors="none", edgecolors="#D55E00", label="interferers")
    ax_map.scatter([scenario.ue.real], [scenario.ue.imag], marker="o", s=75,
                   color="#E69F00", edgecolor="black", label="UE")
    ax_map.plot([scenario.serving_gnb.real, scenario.ue.real],
                [scenario.serving_gnb.imag, scenario.ue.imag], color="#0072B2")
    for gnb in scenario.interferers:
        ax_map.plot([gnb.real, scenario.ue.real], [gnb.imag, scenario.ue.imag], "--", color="#D55E00", alpha=.55)
    ax_map.set_aspect("equal"); ax_map.grid(alpha=.2); ax_map.legend(fontsize=8, loc="best")
    ax_map.set_title("Network snapshot"); ax_map.set_xlabel("east (m)"); ax_map.set_ylabel("north (m)")

    labels = ["desired", "interference", "noise", "post-processing SINR"]
    values = [result.desired_power_dbm, result.interference_power_dbm, result.noise_power_dbm, result.sinr_db]
    shown = [v if np.isfinite(v) else -125 for v in values]
    colors = ["#0072B2", "#D55E00", "#6B7280", "#009E73"]
    bars = ax_metric.bar(labels, shown, color=colors)
    ax_metric.axhline(0, color="black", lw=.5)
    ax_metric.set_ylabel("dBm for powers; dB for SINR")
    ax_metric.set_title(
        f"SINR {result.sinr_db:.1f} dB  |  BLER {result.bler:.2f}  |  "
        f"goodput {100 * result.goodput_fraction:.0f}%  |  "
        f"{'OUTAGE' if result.outage else 'service available'}"
    )
    for bar, value in zip(bars, values):
        ax_metric.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       "none" if not np.isfinite(value) else f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    ax_tx.imshow(scenario.image); ax_tx.set_title("Transmitted image"); ax_tx.axis("off")
    ax_rx.imshow(result.received_image); ax_rx.set_title("Image delivered to UE"); ax_rx.axis("off")
    fig.suptitle(title, fontweight="bold")
    return fig, (ax_map, ax_metric, ax_tx, ax_rx)
