#!/usr/bin/env python3
#
# Generate diagnostic figures for RIOT gliders, mirroring osu685_examine.m
#
# Mar-2026, Pat Welch, pat@mousebrains.com

import argparse
import os
import pickle
import numpy as np
import netCDF4 as nc
import gsw
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.colors import SymLogNorm
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import find_peaks
from matplotlib.widgets import Button
from matplotlib.lines import Line2D
import cmocean

_EPOCH = np.datetime64(0, "s")
_ONE_SEC = np.timedelta64(1, "s")


# =========================================================================
# Utility helpers
# =========================================================================

def _dt64_to_epoch(arr):
    """Convert datetime64 array to float64 POSIX seconds."""
    return (arr - _EPOCH) / _ONE_SEC


def _unmask(arr):
    """Convert a possibly-masked array to a plain ndarray (NaN for masked floats)."""
    if hasattr(arr, "filled"):
        if np.issubdtype(arr.dtype, np.floating):
            return arr.filled(np.nan)
        return arr.data
    return arr



def _find_profiles(epoch, factor=3):
    """Find profile boundaries from epoch timestamps. Returns (starts, ends)."""
    dt = np.diff(epoch)
    thresh = np.median(dt[dt > 0]) * factor
    gaps = np.where(dt > thresh)[0] + 1
    starts = np.concatenate([[0], gaps])
    ends = np.concatenate([gaps, [len(epoch)]])
    return starts, ends


def _rotate_xlabels(axes):
    """Rotate x-tick labels 30 degrees right-aligned."""
    for ax in np.atleast_1d(axes):
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")


def _plot_epsilon_pair(fig, ax1, ax2, t, y, ylabel, e1_med, e2_med,
                       norm, max_gap_s):
    """Plot paired e1/e2 panels with shared axes and joint colorbar."""
    lc1 = colored_line(ax1, fig, t, y, e1_med, "viridis",
                       None, norm=norm, max_gap_s=max_gap_s, colorbar=False)
    ax1.set_xlabel("Time (UTC)")
    ax1.set_ylabel(ylabel)
    ax1.set_title(r"$\epsilon_1$ median")

    colored_line(ax2, fig, t, y, e2_med, "viridis",
                 None, norm=norm, max_gap_s=max_gap_s, colorbar=False)
    ax2.set_xlabel("Time (UTC)")
    ax2.set_ylabel(ylabel)
    ax2.set_title(r"$\epsilon_2$ median")

    ax2.sharex(ax1)
    ax2.sharey(ax1)

    if lc1 is not None:
        cb = fig.colorbar(lc1, ax=[ax1, ax2], location="right", shrink=0.8)
        cb.set_label(r"$\log_{10}(\epsilon)$ median (W/kg)")
    return lc1


# =========================================================================
# Core data functions
# =========================================================================

def mk_degrees(vals):
    """Convert Slocum DDMM.MMMM format to decimal degrees."""
    deg = np.trunc(vals / 100)
    minutes = vals - deg * 100
    return deg + minutes / 60


def load_nc(fn, time_var, variables=None):
    """Load a netCDF file into a dict of arrays, filtering to valid times.

    variables: if given, only read these variable names (time_var is always included).
    """
    ds = nc.Dataset(fn)
    if variables is not None:
        to_read = set(variables) | {time_var}
    else:
        to_read = {v for v in ds.variables if not v.startswith("hdr_")}
    data = {}
    for v in to_read:
        if v not in ds.variables:
            continue
        arr = _unmask(ds[v][:])
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.float64)
        data[v] = arr
    ds.close()

    t = data[time_var]
    valid = t > 946684800  # after 2000-01-01
    for k in data:
        if data[k].shape == t.shape:
            data[k] = data[k][valid]
    data["t_dt"] = (data[time_var] * 1e6).astype("datetime64[us]")
    return data


def sci_update(sci, gps_t, gps_lat, gps_lon):
    """Compute oceanographic derived quantities, matching MATLAB sciUpdate."""
    pressure = sci["sci_water_pressure"] * 10  # bar -> dbar
    cond = sci["sci_water_cond"] * 10           # S/m -> mS/cm
    temp = sci["sci_water_temp"]

    gps_epoch = _dt64_to_epoch(gps_t)
    sci_epoch = _dt64_to_epoch(sci["t_dt"])
    lat = np.interp(sci_epoch, gps_epoch, gps_lat)
    lon = np.interp(sci_epoch, gps_epoch, gps_lon)

    sci["pressure"] = pressure
    sci["cond"] = cond
    sci["temp"] = temp
    sci["lat"] = lat
    sci["lon"] = lon
    sci["depth"] = -gsw.z_from_p(pressure, lat)
    sci["SP"] = gsw.SP_from_C(cond, temp, pressure)
    sci["SA"] = gsw.SA_from_SP(sci["SP"], pressure, lon, lat)
    sci["rho"] = gsw.rho_t_exact(sci["SA"], temp, pressure)
    return sci


def make_ctd(sci):
    """Extract CTD subset with no NaNs in key fields."""
    mask = (np.isfinite(sci["depth"]) & np.isfinite(sci["temp"])
            & np.isfinite(sci["SP"]) & np.isfinite(sci["rho"]))
    return {k: sci[k][mask] for k in ("t_dt", "depth", "temp", "SP", "SA", "rho")}


# =========================================================================
# Bathymetry (cached OPeNDAP fetch)
# =========================================================================

def _fetch_bathymetry_remote(lon_min, lon_max, lat_min, lat_max):
    """Fetch ETOPO2022 bathymetry via OPeNDAP for already-padded bounds."""
    url = ("https://www.ngdc.noaa.gov/thredds/dodsC/global/"
           "ETOPO2022/60s/60s_bed_elev_netcdf/ETOPO_2022_v1_60s_N90W180_bed.nc")
    ds = nc.Dataset(url)
    lat = ds["lat"][:]
    lon = ds["lon"][:]
    lat_ix = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    lon_ix = np.where((lon >= lon_min) & (lon <= lon_max))[0]
    bathy_lat = lat[lat_ix]
    bathy_lon = lon[lon_ix]
    bathy_z = ds["z"][lat_ix.min():lat_ix.max()+1, lon_ix.min():lon_ix.max()+1]
    ds.close()
    return bathy_lon, bathy_lat, bathy_z


def fetch_bathymetry(lon_min, lon_max, lat_min, lat_max, pad=0.1,
                     cache_dir="."):
    """Fetch ETOPO2022 bathymetry, caching locally with +/-0.2 deg margin.

    If the cache exists but doesn't cover the requested region, it is
    extended (union of old and new bounds) rather than replaced.
    """
    margin = 0.2
    req_lon_min, req_lon_max = lon_min - pad, lon_max + pad
    req_lat_min, req_lat_max = lat_min - pad, lat_max + pad

    cache_path = os.path.join(cache_dir, ".bathy_cache.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        c_lon_min, c_lon_max = cached["lon"].min(), cached["lon"].max()
        c_lat_min, c_lat_max = cached["lat"].min(), cached["lat"].max()

        if (c_lon_min <= req_lon_min and c_lon_max >= req_lon_max
                and c_lat_min <= req_lat_min and c_lat_max >= req_lat_max):
            lon_ix = ((cached["lon"] >= req_lon_min)
                      & (cached["lon"] <= req_lon_max))
            lat_ix = ((cached["lat"] >= req_lat_min)
                      & (cached["lat"] <= req_lat_max))
            return (cached["lon"][lon_ix], cached["lat"][lat_ix],
                    cached["z"][np.ix_(lat_ix, lon_ix)])

        fetch_lon_min = min(c_lon_min, req_lon_min - margin)
        fetch_lon_max = max(c_lon_max, req_lon_max + margin)
        fetch_lat_min = min(c_lat_min, req_lat_min - margin)
        fetch_lat_max = max(c_lat_max, req_lat_max + margin)
    else:
        fetch_lon_min = req_lon_min - margin
        fetch_lon_max = req_lon_max + margin
        fetch_lat_min = req_lat_min - margin
        fetch_lat_max = req_lat_max + margin

    blon, blat, bz = _fetch_bathymetry_remote(
        fetch_lon_min, fetch_lon_max, fetch_lat_min, fetch_lat_max)

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"lon": blon, "lat": blat, "z": bz}, f)

    lon_ix = (blon >= req_lon_min) & (blon <= req_lon_max)
    lat_ix = (blat >= req_lat_min) & (blat <= req_lat_max)
    return blon[lon_ix], blat[lat_ix], bz[np.ix_(lat_ix, lon_ix)]


# =========================================================================
# Colored-line plotting
# =========================================================================

def colored_line(ax, fig, t_dt, y, z, cmap, clabel, vmin=None, vmax=None,
                 lw=3, max_gap_s=None, norm=None, colorbar=True,
                 invert_yaxis=True):
    """Plot a line colored by z values using LineCollection.

    max_gap_s: if set, suppress segments spanning time gaps larger than this (seconds).
    norm: optional shared Normalize instance (overrides vmin/vmax).
    colorbar: whether to add a colorbar to this axes.
    invert_yaxis: whether to invert the y-axis (default True for depth plots).
    """
    t_num = mdates.date2num(t_dt)
    mask = np.isfinite(t_num) & np.isfinite(y) & np.isfinite(z)
    t_num, y, z = t_num[mask], y[mask], z[mask]
    if len(t_num) == 0:
        return None

    order = np.argsort(t_num)
    t_num, y, z = t_num[order], y[order], z[order]

    if norm is None:
        if vmin is None:
            vmin = np.nanquantile(z, 0.05)
        if vmax is None:
            vmax = np.nanquantile(z, 0.95)
        norm = plt.Normalize(vmin, vmax)

    points = np.column_stack([t_num, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    seg_colors = 0.5 * (z[:-1] + z[1:])

    if max_gap_s is not None:
        dt_days = np.diff(t_num)
        keep = dt_days < (max_gap_s / 86400.0)
        segments = segments[keep]
        seg_colors = seg_colors[keep]

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(seg_colors)
    lc.set_linewidth(lw)
    ax.add_collection(lc)
    ax.set_xlim(t_num.min(), t_num.max())
    ax.set_ylim(y.min(), y.max())
    ax.xaxis_date()
    if invert_yaxis:
        ax.invert_yaxis()
    ax.grid(True)
    if colorbar:
        cb = fig.colorbar(lc, ax=ax)
        cb.set_label(clabel)
    return lc


# =========================================================================
# Figure functions — each returns its Figure object (or None)
# =========================================================================

def _figure_ctd_flight(glider, basedir, merged, ctd_gap_s,
                       wd_t, wd_depth, pitch_deg, roll_deg, sci):
    """Figure 1: CTD & Flight Diagnostics (3x2 grid).

    Left column:  Temperature (+ water depth), Salinity, Density
    Right column: Map, Pitch histogram, Roll histogram
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    fig.suptitle(f"{glider} — CTD & Flight Diagnostics", fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(3, 2)

    # --- Left column: CTD time series (shared axes) ---

    # (0,0) Temperature + water depth
    ax_temp = fig.add_subplot(gs[0, 0])
    colored_line(ax_temp, fig, merged["t_dt"], merged["depth"], merged["temp"],
                 "viridis", "Temperature (C)", max_gap_s=ctd_gap_s)
    ax_temp.plot(wd_t, wd_depth, ".", color="0.4", markersize=2, label="Water depth")
    ax_temp.legend(fontsize=7, loc="lower right")
    ax_temp.set_xlabel("Time (UTC)")
    ax_temp.set_ylabel("Depth (m)")

    # (1,0) Salinity
    ax = fig.add_subplot(gs[1, 0], sharex=ax_temp, sharey=ax_temp)
    colored_line(ax, fig, merged["t_dt"], merged["depth"], merged["SP"],
                 "viridis_r", "Salinity (PSU)", max_gap_s=ctd_gap_s)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Depth (m)")

    # (2,0) Density
    ax = fig.add_subplot(gs[2, 0], sharex=ax_temp, sharey=ax_temp)
    colored_line(ax, fig, merged["t_dt"], merged["depth"], merged["rho"],
                 "viridis_r", "Density (kg/m$^3$)", max_gap_s=ctd_gap_s)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Depth (m)")

    # --- Right column: Map, Pitch, Roll ---

    # (0,1) Geographic track with coastline and bathymetry
    ax = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())

    sci_mask = np.isfinite(sci["lon"]) & np.isfinite(sci["lat"])
    track_lon = sci["lon"][sci_mask]
    track_lat = sci["lat"][sci_mask]

    try:
        blon, blat, bz = fetch_bathymetry(
            track_lon.min(), track_lon.max(), track_lat.min(), track_lat.max(),
            cache_dir=basedir,
        )
        blevels = np.arange(np.floor(bz.min() / 100) * 100, 1, 100)
        ax.contourf(blon, blat, bz, levels=blevels, cmap=cmocean.cm.deep_r,
                    transform=ccrs.PlateCarree())
        ax.contour(blon, blat, bz, levels=blevels, colors="0.5", linewidths=0.3,
                   transform=ccrs.PlateCarree())
    except Exception as e:
        print(f"  Warning: could not fetch bathymetry: {e}")

    ax.plot(track_lon, track_lat, "-", color="red", linewidth=1.5,
            transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="tan")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_xlabel("Longitude (deg East)")
    ax.set_ylabel("Latitude (deg North)")

    # (1,1) Pitch histogram
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(pitch_deg, bins=100)
    ax.grid(True)
    ax.set_xlabel("Pitch (deg)")
    ax.set_ylabel("Count")

    # (2,1) Roll histogram
    ax = fig.add_subplot(gs[2, 1])
    ax.hist(roll_deg, bins=100, color="tab:orange")
    ax.grid(True)
    ax.set_xlabel("Roll (deg)")
    ax.set_ylabel("Count")

    return fig


def _figure_turbulence(glider, mri):
    """Figure 2: MRI/Turbulence Diagnostics."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"{glider} — Turbulence Diagnostics", fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_epsilon_pair(fig, ax1, ax2, mri["t"], mri["pressure"],
                       "Pressure (dbar)", mri["e1_med"], mri["e2_med"],
                       mri["eps_norm"], mri["max_gap_s"])

    ax3 = fig.add_subplot(gs[1, :])
    for s, e in zip(mri["profile_starts"], mri["profile_ends"]):
        sl = slice(s, e)
        ax3.plot(mri["t"][sl], mri["e1"][sl], ".-", markersize=3, color="tab:blue",
                 label=r"$\log_{10}(\epsilon_1)$" if s == 0 else None)
        ax3.plot(mri["t"][sl], mri["e2"][sl], ".-", markersize=3, color="tab:orange",
                 label=r"$\log_{10}(\epsilon_2)$" if s == 0 else None)
        ax3.plot(mri["t"][sl], mri["e1_med"][sl], "--",
                 color="tab:blue", linewidth=1.5,
                 label=r"$\log_{10}(\epsilon_1)$ median" if s == 0 else None)
        ax3.plot(mri["t"][sl], mri["e2_med"][sl], "--",
                 color="tab:orange", linewidth=1.5,
                 label=r"$\log_{10}(\epsilon_2)$ median" if s == 0 else None)

    ax3.grid(True)
    ax3.set_xlabel("Time (UTC)")
    ax3.set_ylabel(r"$\log_{10}(\epsilon)$ (W/kg)")
    ax3.legend(loc="upper left")

    spike_parts = []
    for name, eps in [(r"$\epsilon_1$", mri["e1"]), (r"$\epsilon_2$", mri["e2"])]:
        peaks, _ = find_peaks(eps, prominence=5, distance=3)
        if len(peaks) > 1:
            peak_times = mri["epoch"][peaks]
            dt_peaks = np.diff(peak_times)
            dt_intra = dt_peaks[dt_peaks < 500]
            if len(dt_intra) > 0:
                mean_s = np.mean(dt_intra)
                std_s = np.std(dt_intra)
                spike_parts.append(
                    f"{name}: {mean_s:.0f} \u00b1 {std_s:.0f} s (n={len(dt_intra)})"
                )
    if spike_parts:
        ax3.set_title("Spike interval:  " + ",   ".join(spike_parts), fontsize=10)

    _rotate_xlabels((ax1, ax2, ax3))
    return fig


def _figure_modem_filter(glider, mri):
    """Figure 8: Despike — tagged vs cleaned epsilon."""
    t = mri["t"]
    e1 = mri["e1"]
    e2 = mri["e2"]
    dirty = mri["contaminated"]
    clean = ~dirty
    n_dirty = dirty.sum()
    n_total = len(dirty)

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        f"{glider} — Despike "
        f"({n_dirty} of {n_total} tagged, {100 * n_dirty / n_total:.1f}%)",
        fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
    ax_tl = fig.add_subplot(gs[0, 0])
    ax_tr = fig.add_subplot(gs[0, 1], sharex=ax_tl, sharey=ax_tl)
    ax_bot = fig.add_subplot(gs[1, :], sharex=ax_tl, sharey=ax_tl)

    for s, e in zip(mri["profile_starts"], mri["profile_ends"]):
        sl = slice(s, e)
        first = s == 0

        # Top-left: e1 all data with dirty highlighted
        ax_tl.plot(t[sl], e1[sl], ".-", markersize=2, color="tab:blue",
                   label="all" if first else None)
        d = dirty[sl]
        if d.any():
            ax_tl.plot(t[sl][d], e1[sl][d], "o",
                       markersize=3, color="tab:red", alpha=0.6,
                       label="tagged" if first else None)

        # Top-right: e2 all data with dirty highlighted
        ax_tr.plot(t[sl], e2[sl], ".-", markersize=2, color="tab:orange",
                   label="all" if first else None)
        if d.any():
            ax_tr.plot(t[sl][d], e2[sl][d], "o",
                       markersize=3, color="tab:red", alpha=0.6,
                       label="tagged" if first else None)

        # Bottom: both epsilons cleaned
        c = clean[sl]
        if c.any():
            ax_bot.plot(t[sl][c], e1[sl][c], ".-", markersize=2,
                        color="tab:blue",
                        label=r"$\epsilon_1$" if first else None)
            ax_bot.plot(t[sl][c], e2[sl][c], ".-", markersize=2,
                        color="tab:orange",
                        label=r"$\epsilon_2$" if first else None)

    ax_tl.set_title(r"$\epsilon_1$ — Tagged")
    ax_tr.set_title(r"$\epsilon_2$ — Tagged")
    ax_bot.set_title("Filtered")

    for ax in (ax_tl, ax_tr, ax_bot):
        ax.grid(True)
    ax_bot.set_xlabel("Time (UTC)")
    for ax in (ax_tl, ax_bot):
        ax.set_ylabel(r"$\log_{10}(\epsilon)$ (W/kg)")
    ax_tl.legend(loc="upper left", fontsize=8)
    ax_tr.legend(loc="upper left", fontsize=8)
    ax_bot.legend(loc="upper left", fontsize=8)

    _rotate_xlabels((ax_tl, ax_tr, ax_bot))
    return fig


def _figure_profile_walker(glider, mri_t, mri_depth, mri_e1_med, mri_e2_med,
                           prof_data, eps_norm, max_gap_s):
    """Figure 3: Profile Walker — interactive profile-by-profile viewer."""
    n_prof = len(prof_data)
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"{glider} \u2014 Profile Walker  [1/{n_prof}]",
                 fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2],
                          hspace=0.35, wspace=0.3)

    ax_top1 = fig.add_subplot(gs[0, 0])
    ax_top2 = fig.add_subplot(gs[0, 1])
    _plot_epsilon_pair(fig, ax_top1, ax_top2, mri_t, mri_depth, "Depth (m)",
                       mri_e1_med, mri_e2_med, eps_norm, max_gap_s)

    ax_bt = fig.add_subplot(gs[1, :])
    ax_bt.sharey(ax_top1)
    ax_bt.set_ylabel("Depth (m)")
    ax_bt.set_xlabel("Temperature (\u00b0C)", color="tab:red")
    ax_bt.tick_params(axis="x", labelcolor="tab:red")
    ax_bt.grid(True, alpha=0.3)

    ax_bs = ax_bt.twiny()
    ax_bs.set_xlabel("Salinity (PSU)", color="tab:green")
    ax_bs.tick_params(axis="x", labelcolor="tab:green")

    ax_be = ax_bt.twiny()
    ax_be.xaxis.set_ticks_position("bottom")
    ax_be.xaxis.set_label_position("bottom")
    ax_be.spines["bottom"].set_position(("outward", 45))
    ax_be.set_xlabel(r"$\log_{10}(\epsilon)$ (W/kg)", color="tab:blue")
    ax_be.tick_params(axis="x", labelcolor="tab:blue")

    ax_be.legend(handles=[
        Line2D([0], [0], color="tab:blue", marker="o", ls="none",
               label=r"$\log_{10}(\epsilon_1)$"),
        Line2D([0], [0], color="tab:orange", marker="s", ls="none",
               label=r"$\log_{10}(\epsilon_2)$"),
    ], loc="lower right", fontsize=8)

    _st = dict(idx=0, playing=False, timer=None,
               span1=None, span2=None, lines=[], updating=False)

    def _update(idx):
        if _st["updating"]:
            return
        _st["updating"] = True
        idx = max(0, min(idx, n_prof - 1))
        _st["idx"] = idx
        p = prof_data[idx]
        fig.suptitle(f"{glider} \u2014 Profile Walker  [{idx + 1}/{n_prof}]",
                     fontsize=14, fontweight="bold")

        for key in ("span1", "span2"):
            if _st[key] is not None:
                _st[key].remove()
        ts = mdates.date2num(p["t_start"])
        te = mdates.date2num(p["t_end"])
        _st["span1"] = ax_top1.axvspan(ts, te, alpha=0.3, color="yellow", zorder=0)
        _st["span2"] = ax_top2.axvspan(ts, te, alpha=0.3, color="yellow", zorder=0)

        for ln in _st["lines"]:
            ln.remove()
        _st["lines"] = []

        if len(p["temp"]) > 0:
            ln, = ax_bt.plot(p["temp"], p["ctd_depth"], "-o",
                             color="tab:red", ms=3, lw=1.5)
            _st["lines"].append(ln)
            rng = (p["temp"].max() - p["temp"].min()) * 0.05 or 0.1
            ax_bt.set_xlim(p["temp"].min() - rng, p["temp"].max() + rng)
        else:
            ax_bt.set_xlim(0, 1)

        if len(p["SP"]) > 0:
            ln, = ax_bs.plot(p["SP"], p["ctd_depth"], "-s",
                             color="tab:green", ms=3, lw=1.5)
            _st["lines"].append(ln)
            rng = (p["SP"].max() - p["SP"].min()) * 0.05 or 0.01
            ax_bs.set_xlim(p["SP"].min() - rng, p["SP"].max() + rng)
        else:
            ax_bs.set_xlim(0, 1)

        if len(p["e1"]) > 0:
            l1, = ax_be.plot(p["e1"], p["depth"], "o",
                             color="tab:blue", ms=4)
            l2, = ax_be.plot(p["e2"], p["depth"], "s",
                             color="tab:orange", ms=4)
            _st["lines"].extend([l1, l2])
            ae = np.concatenate([p["e1"], p["e2"]])
            ae = ae[np.isfinite(ae)]
            if len(ae) > 0:
                rng = (ae.max() - ae.min()) * 0.05 or 0.1
                ax_be.set_xlim(ae.min() - rng, ae.max() + rng)
        else:
            ax_be.set_xlim(-12, -6)

        fig.canvas.draw_idle()
        _st["updating"] = False

    def _prev(event):
        _update(_st["idx"] - 1)

    def _next(event):
        _update(_st["idx"] + 1)

    def _advance():
        if _st["playing"] and _st["idx"] < n_prof - 1:
            _update(_st["idx"] + 1)
        else:
            _st["playing"] = False
            if _st["timer"]:
                _st["timer"].stop()
            btn_play.label.set_text("Play")
            fig.canvas.draw_idle()

    def _play(event):
        if _st["playing"]:
            _st["playing"] = False
            if _st["timer"]:
                _st["timer"].stop()
            btn_play.label.set_text("Play")
            fig.canvas.draw_idle()
        else:
            _st["playing"] = True
            btn_play.label.set_text("Pause")
            fig.canvas.draw_idle()
            _st["timer"] = fig.canvas.new_timer(interval=1000)
            _st["timer"].add_callback(_advance)
            _st["timer"].start()

    def _on_xlim(ax):
        if _st["updating"]:
            return
        lo, hi = ax.get_xlim()
        for i, p in enumerate(prof_data):
            ts = mdates.date2num(p["t_start"])
            te = mdates.date2num(p["t_end"])
            if ts >= lo and te <= hi:
                if i != _st["idx"]:
                    _update(i)
                return

    ax_top1.callbacks.connect("xlim_changed", _on_xlim)

    btn_prev = Button(fig.add_axes([0.62, 0.94, 0.08, 0.035]), "Previous")
    btn_play = Button(fig.add_axes([0.71, 0.94, 0.08, 0.035]), "Play")
    btn_next = Button(fig.add_axes([0.80, 0.94, 0.08, 0.035]), "Next")
    btn_prev.on_clicked(_prev)
    btn_play.on_clicked(_play)
    btn_next.on_clicked(_next)

    _rotate_xlabels((ax_top1, ax_top2))
    _update(0)
    return fig


def _figure_ctd_derivatives(glider, merged):
    """Figure 4: CTD Derivatives — dT/dz, dS/dz, d-rho/dz."""
    ctd_order = np.argsort(merged["t_dt"])
    ctd_t = merged["t_dt"][ctd_order]
    ctd_depth = merged["depth"][ctd_order]
    ctd_temp = merged["temp"][ctd_order]
    ctd_SP = merged["SP"][ctd_order]
    ctd_rho = merged["rho"][ctd_order]

    ctd_epoch = _dt64_to_epoch(ctd_t)
    starts, ends = _find_profiles(ctd_epoch)
    ctd_dt = np.diff(ctd_epoch)
    gap_s = np.median(ctd_dt[ctd_dt > 0]) * 3

    dt_list, dz_list = [], []
    dTdz_list, dSdz_list, drhodz_list = [], [], []

    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        dz = np.diff(ctd_depth[s:e])
        ok = np.abs(dz) > 0.01
        mid_t = ctd_t[s:e - 1]
        mid_z = 0.5 * (ctd_depth[s:e - 1] + ctd_depth[s + 1:e])
        safe_dz = np.where(ok, dz, np.nan)
        dTdz = np.diff(ctd_temp[s:e]) / safe_dz
        dSdz = np.diff(ctd_SP[s:e]) / safe_dz
        drhodz = np.diff(ctd_rho[s:e]) / safe_dz
        dt_list.append(mid_t)
        dz_list.append(mid_z)
        dTdz_list.append(dTdz)
        dSdz_list.append(dSdz)
        drhodz_list.append(drhodz)

    if not dt_list:
        return None

    d_t_all = np.concatenate(dt_list)
    d_z_all = np.concatenate(dz_list)
    dTdz_all = np.concatenate(dTdz_list)
    dSdz_all = np.concatenate(dSdz_list)
    drhodz_all = np.concatenate(drhodz_list)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), constrained_layout=True,
                              sharex=True, sharey=True)
    fig.suptitle(f"{glider} \u2014 CTD Derivatives (per meter)",
                 fontsize=14, fontweight="bold")

    deep = d_z_all > 125
    derivatives = [
        (dTdz_all * 1000, "viridis", "$dT/dz$ (m\N{DEGREE SIGN}C/m)"),
        (dSdz_all * 1000, "viridis", r"$dS_{P}/dz$ (mPSU/m)"),
        (drhodz_all * 1000, "viridis", r"$d\rho/dz$ (g/m$^3$/m)"),
    ]

    for ax, (ddata, cmap, clabel) in zip(axes, derivatives):
        deep_vals = ddata[deep]
        deep_vals = deep_vals[np.isfinite(deep_vals)]
        if len(deep_vals) > 0:
            vmin = np.nanquantile(deep_vals, 0.05)
            vmax = np.nanquantile(deep_vals, 0.95)
            linthresh = np.nanmedian(np.abs(deep_vals))
            norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
        else:
            norm = None
        colored_line(ax, fig, d_t_all, d_z_all, ddata, cmap, clabel,
                     norm=norm, max_gap_s=gap_s)
        ax.set_ylabel("Depth (m)")

    axes[-1].set_xlabel("Time (UTC)")
    _rotate_xlabels(axes)
    return fig


def _figure_flight_health(glider, flt):
    """Figure 5: Pitch, Roll, and Battery voltage vs time."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), constrained_layout=True,
                              sharex=True)
    fig.suptitle(f"{glider} \u2014 Flight Health", fontsize=14, fontweight="bold")

    t = flt["t_dt"]

    # Pitch
    ax = axes[0]
    mask = np.isfinite(flt["m_pitch"])
    ax.plot(t[mask], np.degrees(flt["m_pitch"][mask]), ".", markersize=1.5)
    ax.grid(True)
    ax.set_ylabel("Pitch (deg)")

    # Roll
    ax = axes[1]
    mask = np.isfinite(flt["m_roll"])
    ax.plot(t[mask], np.degrees(flt["m_roll"][mask]), ".", markersize=1.5, color="tab:orange")
    ax.grid(True)
    ax.set_ylabel("Roll (deg)")

    # Thruster power on right y-axis
    ax2 = ax.twinx()
    tp_mask = np.isfinite(flt["m_thruster_power"])
    ax2.plot(t[tp_mask], flt["m_thruster_power"][tp_mask], ".", markersize=1.5, color="tab:red")
    ax2.set_ylabel("Thruster Power (W)")

    # Battery
    ax = axes[2]
    mask = np.isfinite(flt["m_battery"])
    ax.plot(t[mask], flt["m_battery"][mask], ".", markersize=1.5, color="tab:green")
    ax.grid(True)
    ax.set_ylabel("Battery (V)")
    ax.set_xlabel("Time (UTC)")

    _rotate_xlabels(axes)
    return fig


def _color_yaxis(ax, color):
    """Color an axis's y-label and tick labels."""
    ax.yaxis.label.set_color(color)
    ax.tick_params(axis="y", colors=color)


def _figure_flight_control(glider, flt, sci):
    """Figure 6: Pitch/Roll vs battery position, and oil volume vs vertical speed."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), constrained_layout=True,
                              sharex=True)
    fig.suptitle(f"{glider} \u2014 Flight Control", fontsize=14, fontweight="bold")

    t = flt["t_dt"]

    # Pitch (left) and battery position (right)
    ax = axes[0]
    mask = np.isfinite(flt["m_pitch"])
    ax.plot(t[mask], np.degrees(flt["m_pitch"][mask]), ".", markersize=1.5, color="tab:blue")
    ax.grid(True)
    ax.set_ylabel("Pitch (deg)")
    _color_yaxis(ax, "tab:blue")
    ax2 = ax.twinx()
    bp_mask = np.isfinite(flt["m_battpos"])
    ax2.plot(t[bp_mask], flt["m_battpos"][bp_mask], ".", markersize=1.5, color="tab:red")
    ax2.set_ylabel("Battery Position (in)")
    _color_yaxis(ax2, "tab:red")

    # Roll (left) and depth (right)
    ax = axes[1]
    mask = np.isfinite(flt["m_roll"])
    ax.plot(t[mask], np.degrees(flt["m_roll"][mask]), ".", markersize=1.5, color="tab:orange")
    ax.grid(True)
    ax.set_ylabel("Roll (deg)")
    roll_deg = np.degrees(flt["m_roll"][mask])
    rlim = max(abs(np.nanpercentile(roll_deg, 1)), abs(np.nanpercentile(roll_deg, 99)))
    ax.set_ylim(-rlim, rlim)
    _color_yaxis(ax, "tab:orange")
    ax2 = ax.twinx()
    depth_mask = np.isfinite(sci["depth"])
    ax2.plot(sci["t_dt"][depth_mask], sci["depth"][depth_mask], ".", markersize=1.5, color="tab:red")
    ax2.set_ylabel("Depth (m)")
    ax2.invert_yaxis()
    _color_yaxis(ax2, "tab:red")

    # Vertical speed (left) and oil volume (right)
    # Compute dz/dt from CTD depth and time
    depth = sci["depth"]
    sci_epoch = _dt64_to_epoch(sci["t_dt"])
    dmask = np.isfinite(depth) & np.isfinite(sci_epoch)
    d_ok = depth[dmask]
    t_ok = sci_epoch[dmask]
    dzdt = np.diff(d_ok) / np.diff(t_ok)
    dzdt_t = sci["t_dt"][dmask][1:]  # midpoint times (use end of each interval)

    ax = axes[2]
    ax.plot(dzdt_t, dzdt, ".", markersize=1.5, color="tab:purple")
    ax.grid(True)
    ax.set_ylabel("Vertical Speed (m/s)")
    ax.set_xlabel("Time (UTC)")
    _color_yaxis(ax, "tab:purple")
    ax2 = ax.twinx()
    oil_mask = np.isfinite(flt["m_de_oil_vol"])
    ax2.plot(t[oil_mask], flt["m_de_oil_vol"][oil_mask], ".", markersize=1.5, color="tab:green")
    ax2.set_ylabel("Oil Volume (cc)")
    _color_yaxis(ax2, "tab:green")

    _rotate_xlabels(axes)
    return fig


def _figure_depth_overview(glider, flt, sci, mri, mri_depth, mean_lat):
    """Figure 7: Depth panels colored by density, oil volume, and epsilon."""
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    fig.suptitle(f"{glider} \u2014 Depth Overview", fontsize=14, fontweight="bold")
    gs = GridSpec(3, 2, figure=fig)

    ax_top = fig.add_subplot(gs[0, :])
    ax_mid = fig.add_subplot(gs[1, :], sharex=ax_top, sharey=ax_top)
    ax_e1 = fig.add_subplot(gs[2, 0], sharex=ax_top, sharey=ax_top)
    ax_e2 = fig.add_subplot(gs[2, 1], sharex=ax_top, sharey=ax_top)
    depth_axes = [ax_top, ax_mid, ax_e1, ax_e2]

    t = flt["t_dt"]

    # --- Top: depth colored by density, battery position (right) ---
    rho_mask = np.isfinite(sci["rho"]) & (sci["rho"] >= 1020)
    colored_line(ax_top, fig, sci["t_dt"][rho_mask], sci["depth"][rho_mask],
                 sci["rho"][rho_mask] - 1000, "viridis",
                 r"$\rho$ − 1000 (kg/m³)", lw=2, invert_yaxis=False)
    ax_top.set_ylabel("Depth (m)")
    ax_top_r = ax_top.twinx()
    bp_mask = np.isfinite(flt["m_battpos"])
    ax_top_r.plot(t[bp_mask], flt["m_battpos"][bp_mask], ".", markersize=1.5, color="tab:red")
    ax_top_r.set_ylabel("Battery Position (in)")
    _color_yaxis(ax_top_r, "tab:red")

    # --- Middle: depth colored by oil volume, pitch (right) ---
    oil = flt["m_de_oil_vol"]
    flt_epoch = _dt64_to_epoch(t)
    sci_epoch = _dt64_to_epoch(sci["t_dt"])
    sci_depth_valid = np.isfinite(sci["depth"]) & np.isfinite(sci_epoch)
    flt_depth_interp = np.interp(flt_epoch, sci_epoch[sci_depth_valid],
                                  sci["depth"][sci_depth_valid])
    oil_depth_mask = np.isfinite(oil) & np.isfinite(flt_depth_interp)
    colored_line(ax_mid, fig, t[oil_depth_mask], flt_depth_interp[oil_depth_mask],
                 oil[oil_depth_mask], "viridis", "Oil Volume (cc)", lw=2,
                 invert_yaxis=False)
    ax_mid.set_ylabel("Depth (m)")
    ax_mid_r = ax_mid.twinx()
    pitch_mask = np.isfinite(flt["m_pitch"])
    ax_mid_r.plot(t[pitch_mask], np.degrees(flt["m_pitch"][pitch_mask]), ".",
                  markersize=1.5, color="tab:red")
    ax_mid_r.set_ylabel("Pitch (deg)")
    _color_yaxis(ax_mid_r, "tab:red")

    # Epsilon color normalization: q05–q95 for data below 125 m
    deep = mri_depth > 125
    deep_eps = np.concatenate([mri["e1_med"][deep], mri["e2_med"][deep]])
    deep_eps = deep_eps[np.isfinite(deep_eps)]
    eps_norm = plt.Normalize(np.nanquantile(deep_eps, 0.05),
                              np.nanquantile(deep_eps, 0.95))

    # --- Bottom left: depth colored by epsilon_1 ---
    lc1 = colored_line(ax_e1, fig, mri["t"], mri_depth, mri["e1_med"], "viridis",
                        None, norm=eps_norm, max_gap_s=mri["max_gap_s"],
                        colorbar=False, lw=2, invert_yaxis=False)
    ax_e1.set_ylabel("Depth (m)")
    ax_e1.set_xlabel("Time (UTC)")
    ax_e1.set_title(r"$\epsilon_1$ median")

    # --- Bottom right: depth colored by epsilon_2 ---
    lc2 = colored_line(ax_e2, fig, mri["t"], mri_depth, mri["e2_med"], "viridis",
                        None, norm=eps_norm, max_gap_s=mri["max_gap_s"],
                        colorbar=False, lw=2, invert_yaxis=False)
    ax_e2.set_xlabel("Time (UTC)")
    ax_e2.set_title(r"$\epsilon_2$ median")

    if lc1 is not None:
        cb = fig.colorbar(lc1, ax=[ax_e1, ax_e2], location="right", shrink=0.8)
        cb.set_label(r"$\log_{10}(\epsilon)$ median (W/kg)")

    # Invert depth axis (shared, so only need to do it once)
    ax_top.invert_yaxis()

    _rotate_xlabels(depth_axes)
    return fig


# =========================================================================
# MRI data loading
# =========================================================================

_SPIKE_FACTOR = np.log10(2)  # factor of 2 in linear space
_MODEM_WINDOW = 24  # seconds: points within this of a modem ping are non-anchor
_MODEM_STATES = (4, 5, 10)


def _modem_active_times(sci):
    """Extract sorted unique POSIX times when the acoustic modem is active."""
    for var in ("sci_generic_a", "sci_generic_b", "sci_generic_c", "sci_generic_k"):
        if var not in sci:
            return np.array([])
    a, b, c, k = (sci[v] for v in
                   ("sci_generic_a", "sci_generic_b", "sci_generic_c", "sci_generic_k"))
    valid = (a > 100) & (b >= 0) & (c < 1000)
    if not np.any(valid):
        return np.array([])
    ping_posix = a[valid] * 86400.0 + b[valid] + c[valid] / 1000.0
    active = np.isin(k[valid], _MODEM_STATES)
    if not np.any(active):
        return np.array([])
    return np.unique(ping_posix[active])


def _near_modem(mri_posix, modem_times, half_window):
    """Boolean mask: True if measurement is within half_window of any modem ping."""
    idx = np.searchsorted(modem_times, mri_posix)
    idx_l = np.clip(idx - 1, 0, len(modem_times) - 1)
    idx_r = np.clip(idx, 0, len(modem_times) - 1)
    return np.minimum(np.abs(mri_posix - modem_times[idx_l]),
                      np.abs(mri_posix - modem_times[idx_r])) <= half_window


def _despike_anchors(e1, e2, is_anchor, profile_starts, profile_ends,
                     factor_log):
    """Iteratively despike anchor points among themselves."""
    n = len(e1)
    bad = np.zeros(n, dtype=bool)
    for _ in range(100):
        new_flags = np.zeros(n, dtype=bool)
        for s, e in zip(profile_starts, profile_ends):
            valid = [i for i in range(s, e) if is_anchor[i] and not bad[i]]
            if len(valid) < 3:
                continue
            for eps in [e1, e2]:
                vals = np.array([eps[i] for i in valid])
                for j in range(1, len(valid) - 1):
                    if (vals[j] - vals[j - 1] > factor_log
                            and vals[j] - vals[j + 1] > factor_log):
                        new_flags[valid[j]] = True
        if not np.any(new_flags & ~bad):
            break
        bad |= new_flags
    return bad


def _flood_fill(e1, e2, clean_anchor, profile_starts, profile_ends,
                factor_log):
    """Walk outward from clean anchors, accepting points within factor_log.

    For each profile, walk forward and backward from anchor points.  A
    non-anchor point is accepted if **both** its e1 and e2 values are
    within ``factor_log`` of the last accepted point on that side.
    Points not reached (or that spike above the walk) are flagged.
    """
    n = len(e1)
    accepted = clean_anchor.copy()

    for s, e in zip(profile_starts, profile_ends):
        indices = list(range(s, e))
        # Forward walk
        last = -1
        for i in indices:
            if accepted[i]:
                last = i
                continue
            if last < 0:
                continue
            if all(eps[i] - eps[last] <= factor_log for eps in [e1, e2]):
                accepted[i] = True
                last = i
        # Backward walk
        last = -1
        for i in reversed(indices):
            if accepted[i]:
                last = i
                continue
            if last < 0:
                continue
            if all(eps[i] - eps[last] <= factor_log for eps in [e1, e2]):
                accepted[i] = True
                last = i

    return ~accepted


def _despike(e1, e2, mri_posix, modem_times, profile_starts, profile_ends,
             factor_log=_SPIKE_FACTOR, modem_window=_MODEM_WINDOW):
    """Combined modem-anchor + factor-of-2 despike.

    1. Points far from modem activity are *anchors* (high-confidence clean).
       Anchors are despiked among themselves.
    2. Walk outward from anchors through the modem-adjacent data, accepting
       points whose epsilon is within ``factor_log`` of the last accepted
       neighbor.  Spikes that exceed this are flagged.

    If no modem times are available, all points are treated as anchors
    and despiked directly.
    """
    if len(modem_times) == 0:
        # No modem data — pure despike (all points are anchors)
        is_anchor = np.ones(len(e1), dtype=bool)
    else:
        is_anchor = ~_near_modem(mri_posix, modem_times, modem_window)

    anchor_bad = _despike_anchors(e1, e2, is_anchor, profile_starts,
                                  profile_ends, factor_log)
    clean_anchor = is_anchor & ~anchor_bad

    return _flood_fill(e1, e2, clean_anchor, profile_starts, profile_ends,
                       factor_log)


def _load_mri(basedir, glider, mission_start, modem_times):
    """Load and process MRI data. Returns dict or None if unavailable."""
    mri_fn = os.path.join(basedir, f"{glider}.mri.nc")
    if not os.path.exists(mri_fn):
        print(f"  No MRI file for {glider}, skipping turbulence figures")
        return None

    ds = nc.Dataset(mri_fn)
    mri_time_ms = _unmask(ds["time"][:])
    mri_pressure = _unmask(ds["pressure"][:])
    mri_e1 = _unmask(ds["e_1"][:])
    mri_e2 = _unmask(ds["e_2"][:])
    ds.close()

    mask = (np.isfinite(mri_time_ms) & np.isfinite(mri_pressure)
            & np.isfinite(mri_e1) & np.isfinite(mri_e2))
    mri_time_ms = mri_time_ms[mask]
    mri_pressure = mri_pressure[mask]
    mri_e1 = mri_e1[mask]
    mri_e2 = mri_e2[mask]

    if len(mri_time_ms) == 0:
        print(f"  No valid MRI data for {glider}")
        return None

    mri_posix = mission_start + mri_time_ms / 1000.0
    mri_t = (mri_posix * 1e6).astype("datetime64[us]")

    dt_ms = np.diff(np.sort(mri_time_ms))
    max_gap_s = np.median(dt_ms[dt_ms > 0]) / 1000.0 * 3

    mri_epoch = _dt64_to_epoch(mri_t)
    profile_starts, profile_ends = _find_profiles(mri_epoch)

    # Despike: modem-anchored flood-fill with factor-of-2 neighbor test
    contaminated = _despike(mri_e1, mri_e2, mri_posix, modem_times,
                            profile_starts, profile_ends)

    mri_e1_med = np.copy(mri_e1)
    mri_e2_med = np.copy(mri_e2)
    mri_e1_med[contaminated] = np.nan
    mri_e2_med[contaminated] = np.nan

    all_eps = np.concatenate([mri_e1_med, mri_e2_med])
    all_eps = all_eps[np.isfinite(all_eps)]
    eps_norm = plt.Normalize(np.nanquantile(all_eps, 0.05),
                             np.nanquantile(all_eps, 0.95))

    return {
        "t": mri_t, "pressure": mri_pressure,
        "e1": mri_e1, "e2": mri_e2,
        "e1_med": mri_e1_med, "e2_med": mri_e2_med,
        "epoch": mri_epoch,
        "contaminated": contaminated,
        "profile_starts": profile_starts, "profile_ends": profile_ends,
        "eps_norm": eps_norm, "max_gap_s": max_gap_s,
    }


# =========================================================================
# Main orchestration
# =========================================================================

def generate_figures(glider, basedir, figures=None):
    """Generate diagnostic figures for a single glider.

    figures: set of figure numbers to generate, or None for all.
    """
    # Dataset requirements per figure
    _NEEDS_MRI = {2, 3, 7, 8}
    ALL_FIGURES = {1, 2, 3, 4, 5, 6, 7, 8}

    if figures is None:
        figures = ALL_FIGURES
    print(f"Processing {glider}...")

    # --- Load data (parallel I/O) ---
    flt_vars = ("m_present_secs_into_mission", "m_lat", "m_lon",
                "m_water_depth", "m_pitch", "m_roll", "m_battery",
                "m_thruster_power", "m_battpos", "m_de_oil_vol")
    sci_vars = ("sci_water_pressure", "sci_water_cond", "sci_water_temp",
                "sci_generic_a", "sci_generic_b", "sci_generic_c", "sci_generic_k")
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_flt = executor.submit(load_nc, os.path.join(basedir, f"{glider}.flt.nc"),
                                "m_present_time", flt_vars)
        f_sci = executor.submit(load_nc, os.path.join(basedir, f"{glider}.sci.nc"),
                                "sci_m_present_time", sci_vars)
    flt = f_flt.result()
    sci = f_sci.result()

    mission_start = flt["m_present_time"][0] - flt["m_present_secs_into_mission"][0]

    # GPS table
    gps_mask = np.isfinite(flt["m_lat"]) & np.isfinite(flt["m_lon"])
    gps_t = flt["t_dt"][gps_mask]
    gps_lat = mk_degrees(flt["m_lat"][gps_mask])
    gps_lon = mk_degrees(flt["m_lon"][gps_mask])

    # Process science data
    sci = sci_update(sci, gps_t, gps_lat, gps_lon)
    ctd = make_ctd(sci)

    # CTD: unique by time, filter temp > 0
    ctd_keys = ("t_dt", "depth", "temp", "SP", "SA", "rho")
    _, ix = np.unique(ctd["t_dt"], return_index=True)
    merged = {k: ctd[k][ix] for k in ctd_keys}
    tmask = merged["temp"] > 0
    merged = {k: merged[k][tmask] for k in merged}

    # CTD gap threshold for discontinuous lines (4x median sample interval)
    ctd_epoch = _dt64_to_epoch(merged["t_dt"])
    ctd_dt = np.diff(ctd_epoch)
    ctd_gap_s = np.median(ctd_dt[ctd_dt > 0]) * 4

    # Water depth
    wd_mask = np.isfinite(flt["m_water_depth"]) & (flt["m_water_depth"] > 1)
    wd_t = flt["t_dt"][wd_mask]
    wd_depth = flt["m_water_depth"][wd_mask]

    # Pitch & roll
    pitch_deg = np.degrees(flt["m_pitch"])
    roll_deg = np.degrees(flt["m_roll"])
    pitch_deg = pitch_deg[np.isfinite(pitch_deg)]
    roll_deg = roll_deg[np.isfinite(roll_deg)]

    # --- Load MRI data if any requested figure needs it ---
    mri = None
    mri_depth = None
    mean_lat = None
    if figures & _NEEDS_MRI:
        modem_times = _modem_active_times(sci)
        mri = _load_mri(basedir, glider, mission_start, modem_times)
        if mri is not None:
            mean_lat = np.nanmean(gps_lat)
            mri_depth = -gsw.z_from_p(mri["pressure"], mean_lat)

    # --- Figure 1: CTD & Flight ---
    if 1 in figures:
        _figure_ctd_flight(glider, basedir, merged, ctd_gap_s,
                           wd_t, wd_depth, pitch_deg, roll_deg, sci)

    # --- Figure 2: Turbulence ---
    if 2 in figures and mri is not None:
        _figure_turbulence(glider, mri)

    # --- Figure 3: Profile Walker ---
    if 3 in figures and mri is not None:
        prof_data = []
        for s, e in zip(mri["profile_starts"], mri["profile_ends"]):
            sl = slice(s, e)
            t_s, t_e = mri["t"][s], mri["t"][e - 1]
            cmask = (merged["t_dt"] >= t_s) & (merged["t_dt"] <= t_e)
            prof_data.append({
                "t_start": t_s, "t_end": t_e,
                "depth": mri_depth[sl],
                "e1": mri["e1_med"][sl], "e2": mri["e2_med"][sl],
                "ctd_depth": merged["depth"][cmask] if cmask.any() else np.array([]),
                "temp":      merged["temp"][cmask]  if cmask.any() else np.array([]),
                "SP":        merged["SP"][cmask]    if cmask.any() else np.array([]),
            })

        if len(prof_data) == 0:
            print(f"  No MRI profiles for {glider}, skipping profile walker")
        else:
            _figure_profile_walker(glider, mri["t"], mri_depth,
                                   mri["e1_med"], mri["e2_med"],
                                   prof_data, mri["eps_norm"],
                                   mri["max_gap_s"])

    # --- Figure 4: CTD Derivatives ---
    if 4 in figures:
        _figure_ctd_derivatives(glider, merged)

    # --- Figure 5: Flight Health ---
    if 5 in figures:
        _figure_flight_health(glider, flt)

    # --- Figure 6: Flight Control ---
    if 6 in figures:
        _figure_flight_control(glider, flt, sci)

    # --- Figure 7: Depth Overview ---
    if 7 in figures and mri is not None:
        _figure_depth_overview(glider, flt, sci, mri, mri_depth, mean_lat)

    # --- Figure 8: Modem Filter ---
    if 8 in figures and mri is not None:
        _figure_modem_filter(glider, mri)

    plt.show()


def _parse_figures(value):
    """Parse a figure spec: e.g. '1', '1,3', '1-3', '1 2 4'."""
    figs = set()
    for part in value.replace(",", " ").split():
        if "-" in part:
            lo, hi = part.split("-", 1)
            figs.update(range(int(lo), int(hi) + 1))
        else:
            figs.add(int(part))
    return figs


def main():
    parser = argparse.ArgumentParser(description="Generate RIOT glider diagnostic figures")
    parser.add_argument("--gliders", nargs="+", default=["osu684", "osu685"],
                        help="Glider names to process (default: osu684 osu685)")
    parser.add_argument("--basedir", default=".",
                        help="Base directory containing NetCDF files (default: .)")
    parser.add_argument("--figure", nargs="+",
                        help="Figures to display: 1-7, e.g. --figure 1 3 or --figure 1-3 or --figure 2,4")
    args = parser.parse_args()

    if args.figure:
        figures = set()
        for v in args.figure:
            figures |= _parse_figures(v)
    else:
        figures = None

    for glider in args.gliders:
        generate_figures(glider, args.basedir, figures=figures)


if __name__ == "__main__":
    main()
