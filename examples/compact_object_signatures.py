#!/usr/bin/env python3
"""Compare predicted photometric signatures of compact-object companions."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np

import troia


def run_example(output_path: Path) -> dict[str, np.ndarray]:
    """Calculate and plot signatures for a solar-type source star."""

    period_days = np.geomspace(0.3, 30.0, 200)[:, None]  # [day]
    companion_mass_solar = np.array([5.0, 30.0, 180.0])[None, :]  # [M_Sun]
    signatures = troia.compute_photometric_signatures(
        period_days,
        companion_mass_solar,
        stellar_radius_solar=1.0,
        stellar_mass_solar=1.0,
        stellar_density_cgs=1.41,
    )

    panel_titles = {
        "beaming": "Doppler beaming",
        "ellipsoidal": "Ellipsoidal variation",
        "self_lensing": "Self-lensing",
    }
    colors = ["#1B6CA8", "#D18B00", "#2E7D32"]
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(12.5, 4.0),
        sharex=True,
        sharey=True,
        facecolor="white",
        constrained_layout=True,
    )
    for axis, (name, title) in zip(axes, panel_titles.items()):
        for mass_index, mass_solar in enumerate(companion_mass_solar.ravel()):
            axis.plot(
                period_days.ravel(),
                signatures[name][:, mass_index],
                color=colors[mass_index],
                linewidth=2.2,
                label=rf"{mass_solar:g} $M_\odot$",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(period_days.min(), period_days.max())
        axis.set_ylim(5e-3, 1e3)
        axis.set_xlabel("Orbital period [day]")
        axis.set_title(title)
        axis.grid(False)

    axes[0].set_ylabel("Predicted amplitude [ppt]")
    legend = axes[2].legend(
        title="Companion mass",
        frameon=True,
        fancybox=True,
        framealpha=1.0,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    figure.suptitle(
        r"Compact-object signals for a $1\,M_\odot$, $1\,R_\odot$ source",
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {output_path}...")
    figure.savefig(
        output_path,
        dpi=300 if output_path.suffix == ".png" else None,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return signatures


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--typefileplot",
        choices=("png", "pdf"),
        default="png",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    output_path = Path(__file__).with_name(
        f"compact_object_signatures.{arguments.typefileplot}"
    )
    run_example(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())