#!/usr/bin/env python3
#
# Generate a recover-by comparison figure for each glider.
# Each subplot shows the battery data with linear fits computed over
# different time windows (1 day, 3 days, 7 days, and full deployment)
# plus tau-weighted (exponential downweighting) fits.
#
# Mar-2026, Pat Welch, pat@mousebrains.com

import argparse
import glob
import os

import matplotlib.pyplot as plt

from slocum_tpw.recover_by import FIT_COLORS, fit_recovery, prepare_dataset

SENSOR = "m_lithium_battery_relative_charge"
THRESHOLD = 15
CONFIDENCE = 0.95

# (ndays, tau, label)
WINDOWS = [
    (None, 0.1, "\u03c4=0.1d"),
    (None, 0.5, "\u03c4=0.5d"),
    (None, 1, "\u03c4=1d"),
    (None, 7, "\u03c4=7d"),
    (None, 14, "\u03c4=14d"),
    (None, None, "Full"),
]


def process_glider(ax, glider, basedir):
    """Plot recover-by fits for a single glider onto the given axes."""
    fn = os.path.join(basedir, f"{glider}.logs.nc")
    if not os.path.exists(fn):
        ax.set_title(f"{glider} \u2014 no data")
        ax.text(
            0.5,
            0.5,
            f"{fn} not found",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return

    try:
        ds = prepare_dataset(fn, sensor=SENSOR)
    except KeyError:
        ax.set_title(f"{glider} \u2014 missing variables")
        return

    if ds.time.size < 3:
        ax.set_title(f"{glider} \u2014 insufficient data")
        return

    # Plot raw data
    ax.plot(
        ds.time,
        ds[SENSOR],
        ".",
        color="tab:blue",
        markersize=3,
        alpha=0.5,
        label="data",
        zorder=1,
    )

    # Threshold line
    ax.axhline(
        y=THRESHOLD,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"threshold ({THRESHOLD}%)",
    )

    # Fit each time window
    for win_idx, (ndays, tau, label) in enumerate(WINDOWS):
        result = fit_recovery(
            ds,
            sensor=SENSOR,
            threshold=THRESHOLD,
            confidence=CONFIDENCE,
            ndays=ndays,
            tau=tau,
        )
        if result is None:
            continue

        r = result
        ci_days = r["recovery_ci_days"]
        ci_str = f"\u00b1{ci_days:.1f}d" if ci_days is not None else ""
        r_sq = r["r_squared"]
        r_sq_str = f" R\u00b2={r_sq:.3f}" if r_sq is not None else ""
        dof = r["dof"]
        dof_str = f"{dof:.1f}" if dof != int(dof) else f"{int(dof)}"
        fit_label = (
            f"{label} {r['recovery_date']}{ci_str}"
            f"{r_sq_str} (dof={dof_str})"
        )

        color = FIT_COLORS[win_idx % len(FIT_COLORS)]

        # Plot fit line over the data range used
        ax.plot(
            r["time"],
            r["intercept"] + r["slope"] * r["dDays"],
            color=color,
            linewidth=1.5,
            label=fit_label,
            zorder=2,
        )

        # Extend fit to recovery date with dashed line
        if r["recovery_date"] > r["time"][-1].values:
            last_val = float(r["intercept"] + r["slope"] * r["dDays"][-1].item())
            ax.plot(
                [r["time"][-1].values, r["recovery_date"]],
                [last_val, THRESHOLD],
                color=color,
                linestyle="--",
                alpha=0.5,
                linewidth=1.5,
                zorder=2,
            )

    ax.set_ylabel("Battery (%)")
    ax.set_title(f"{glider} (n={ds.time.size})")
    ax.legend(fontsize="x-small", loc="upper right")
    ax.grid(True, alpha=0.3)


def _discover_gliders(basedir):
    """Find glider names from *.logs.nc files in basedir."""
    pattern = os.path.join(basedir, "*.logs.nc")
    names = sorted(
        os.path.basename(f).removesuffix(".logs.nc") for f in glob.glob(pattern)
    )
    return names


def main():
    parser = argparse.ArgumentParser(
        description="Generate recover-by comparison figures for RIOT gliders"
    )
    parser.add_argument(
        "--gliders",
        nargs="+",
        help="Glider names to process (default: all *.logs.nc in basedir)",
    )
    parser.add_argument(
        "--basedir",
        default=".",
        help="Base directory containing .logs.nc files (default: .)",
    )
    parser.add_argument(
        "--output", type=str, help="Save figure to file instead of displaying"
    )
    args = parser.parse_args()

    if args.gliders is None:
        args.gliders = _discover_gliders(args.basedir)
        if not args.gliders:
            parser.error(f"No *.logs.nc files found in {args.basedir}")

    n = len(args.gliders)
    fig, axes = plt.subplots(
        n, 1, figsize=(12, 5 * n), constrained_layout=True, squeeze=False
    )

    for i, glider in enumerate(args.gliders):
        process_glider(axes[i, 0], glider, args.basedir)

    axes[-1, 0].set_xlabel("Time (UTC)")
    fig.suptitle(f"Recover-By Estimates \u2014 {SENSOR}", fontsize=12)

    for label in axes[-1, 0].get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")

    if args.output:
        fig.savefig(args.output)
        print(f"Saved to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
