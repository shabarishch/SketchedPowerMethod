import time
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def _plot_xy_curve(ax, label, x_values, y_values, color, alpha=1.0, linestyle="solid"):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    if x_values.size == 0 or y_values.size == 0:
        return False
    if x_values.shape != y_values.shape:
        raise ValueError(
            f"Curve {label!r} has mismatched x/y shapes: "
            f"{x_values.shape} vs {y_values.shape}."
        )

    ax.plot(
        x_values,
        y_values,
        color=color,
        alpha=alpha,
        linestyle=linestyle,
        label=label,
    )
    ax.plot(
        x_values,
        y_values,
        linestyle="None",
        marker="o",
        color=color,
        alpha=1.0,
        label="_nolegend_",
    )
    return True


def _curve_entry_parts(curve_entry):
    if len(curve_entry) == 2:
        label, curve_data = curve_entry
        linestyle = "solid"
    elif len(curve_entry) == 3:
        label, curve_data, linestyle = curve_entry
    else:
        raise ValueError(
            "curve entries must be (label, curve_data) or "
            "(label, curve_data, linestyle)"
        )

    if linestyle not in {"solid", "dashed"}:
        raise ValueError("curve linestyle must be 'solid' or 'dashed'.")
    return label, curve_data, linestyle


def _format_plot_metadata(metadata):
    if metadata is None:
        return None
    if isinstance(metadata, str):
        metadata_text = metadata.strip()
        return metadata_text or None
    if isinstance(metadata, Mapping):
        if not metadata:
            return None
        return ", ".join(f"{key}: {value}" for key, value in metadata.items())
    return str(metadata)


def _trial_curve_xy_values(trial_curve):
    x_values = []
    y_values = []
    for point in trial_curve:
        if len(point) == 2:
            elapsed, error = point
        elif len(point) == 3:
            _, elapsed, error = point
        else:
            raise ValueError(
                "trial curve points must be (elapsed, error) or "
                "(iteration, elapsed, error)"
            )
        x_values.append(elapsed)
        y_values.append(error)
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def plot_error_curves(
    curves,
    metadata=None,
    per_iteration_errors=True,
    save_png=False,
    output_filename=None,
    output_dir="Plots",
    dpi=200,
):
    """
    Plot grouped error curves where each entry is
    (label, curve_data) or (label, curve_data, linestyle), where linestyle
    is either "solid" or "dashed". curve_data may be either:
    - a list of trial curves, where each trial curve is a list of
      (elapsed_time, error) pairs or (iteration, elapsed_time, error)
      triples, or
    - a single averaged curve given as (x_values, y_values).
    metadata may be a string or mapping that is rendered beneath the title.
    per_iteration_errors controls whether the x-axis uses automatic major
    ticks for per-iteration timings or exact checkpoint ticks for averaged
    checkpoint curves.
    If save_png is True, the figure is also saved as a PNG under output_dir.
    """
    if not curves:
        raise ValueError("curves must contain at least one curve to plot.")

    legend_rows = max(1, (len(curves) + 1) // 2)
    fig = plt.figure(figsize=(6, 6 + 0.6 * legend_rows), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[6, max(1.2, 0.8 * legend_rows)], hspace=0.02)
    ax = fig.add_subplot(gs[0])
    ax.set_box_aspect(1)
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis("off")

    has_points = False
    time_points = None
    for scheme_idx, curve_entry in enumerate(curves):
        label, curve_data, linestyle = _curve_entry_parts(curve_entry)
        color = f"C{scheme_idx}"

        if isinstance(curve_data, tuple) and len(curve_data) == 2:
            if time_points is None:
                time_points = np.asarray(curve_data[0], dtype=float)
            has_points |= _plot_xy_curve(
                ax,
                label,
                curve_data[0],
                curve_data[1],
                color=color,
                alpha=1.0,
                linestyle=linestyle,
            )
            continue

        trial_curves = curve_data
        alpha = 1.0 if len(trial_curves) == 1 else 0.3

        for trial_idx, trial_curve in enumerate(trial_curves):
            if len(trial_curve) == 0:
                continue

            x_values, y_values = _trial_curve_xy_values(trial_curve)
            if time_points is None:
                time_points = x_values
            trial_label = label if trial_idx == 0 else "_nolegend_"
            has_points |= _plot_xy_curve(
                ax,
                trial_label,
                x_values,
                y_values,
                color=color,
                alpha=alpha,
                linestyle=linestyle,
            )

    if not has_points:
        raise ValueError("curves must contain at least one non-empty trial curve.")

    ax.set_xlabel("Time Taken", fontsize=14)
    ax.set_ylabel("Error", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=14)
    ax.set_yscale("log")
    metadata_text = _format_plot_metadata(metadata)
    title = f"{metadata_text}"
    ax.set_title(title, fontsize=14)
    if per_iteration_errors:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    elif time_points is not None:
        ax.set_xticks(time_points)
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", ncol=1, frameon=False, fontsize=14)
    output_path = None
    if save_png:
        if output_filename is None:
            output_filename = "error_curves.png"
        output_path = Path(output_filename)
        if output_path.suffix.lower() != ".png":
            output_path = output_path.with_suffix(".png")
        if not output_path.is_absolute():
            output_path = Path(output_dir) / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    return output_path
