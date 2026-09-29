"""Animations that play inline in Jupyter/Colab (replaces MATLAB movies)."""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate(draw_frame, n_frames, figsize=(6, 6), interval_ms=100):
    """Build an animation by calling ``draw_frame(ax, k)`` for k = 0..n_frames-1.

    ``draw_frame`` should redraw the whole frame on ``ax`` (the axes are cleared
    before each call).  Display the result in a notebook with ``show(anim)``.
    """
    fig, ax = plt.subplots(figsize=figsize)

    def update(k):
        ax.clear()
        draw_frame(ax, k)
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=False)
    plt.close(fig)  # show only the animation, not a stray static figure
    return anim


def show(anim):
    """Render an animation as interactive HTML (works in Colab and Jupyter)."""
    from IPython.display import HTML

    return HTML(anim.to_jshtml())


def save_mp4(anim, path, fps=10):
    """Save an animation as an MP4 file (requires ffmpeg, available on Colab)."""
    anim.save(path, fps=fps)
