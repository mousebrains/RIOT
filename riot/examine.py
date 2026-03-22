#!/usr/bin/env python3
#
# Generate diagnostic figures for RIOT gliders, mirroring osu685_examine.m
#
# Mar-2026, Pat Welch, pat@mousebrains.com

import argparse
import os
import numpy as np
import netCDF4 as nc
import gsw
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.colors import SymLogNorm
from datetime import datetime, timezone
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from matplotlib.widgets import Button
from matplotlib.lines import Line2D
import cmocean


def mk_degrees(vals):
    """Convert Slocum DDMM.MMMM format to decimal degrees."""
    deg = np.trunc(vals / 100)
    minutes = vals - deg * 100
    return deg + minutes / 60


def load_nc(fn, time_var):
    """Load a netCDF file into a dict of arrays, filtering to valid times."""
    ds = nc.Dataset(fn)
    data = {}
    for v in ds.variables:
        if v.startswith("hdr_"):
            continue
        arr = ds[v][:]
        if hasattr(arr, "filled"):
            if np.issubdtype(arr.dtype, np.floating):
                arr = arr.filled(np.nan)
            else:
                arr = arr.data
        data[v] = arr.astype(np.float64) if np.issubdtype(arr.dtype, np.integer) else arr
    ds.close()

    t = data[time_var]
    valid = t > 946684800  # after 2000-01-01
    for k in data:
        if data[k].shape == t.shape:
            data[k] = data[k][valid]
    data["t_dt"] = np.array(
        [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in data[time_var]],
    )
    return data


def sci_update(sci, gps_t, gps_lat, gps_lon):
    """Compute oceanographic derived quantities, matching MATLAB sciUpdate."""
    pressure = sci["sci_water_pressure"] * 10  # bar -> dbar
    cond = sci["sci_water_cond"] * 10           # S/m -> mS/cm
    temp = sci["sci_water_temp"]

    gps_epoch = np.array([g.timestamp() for g in gps_t])
    sci_epoch = np.array([s.timestamp() for s in sci["t_dt"]])
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


def fetch_bathymetry(lon_min, lon_max, lat_min, lat_max, pad=0.1):
    """Fetch ETOPO2022 bathymetry via OPeNDAP for a region."""
    lon_min, lon_max = lon_min - pad, lon_max + pad
    lat_min, lat_max = lat_min - pad, lat_max + pad

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


def colored_line(ax, fig, t_dt, y, z, cmap, clabel, vmin=None, vmax=None,
                 lw=3, max_gap_s=None, norm=None, colorbar=True):
    """Plot a line colored by z values using LineCollection.

    max_gap_s: if set, suppress segments spanning time gaps larger than this (seconds).
    norm: optional shared Normalize instance (overrides vmin/vmax).
    colorbar: whether to add a colorbar to this axes.
    """
    t_num = mdates.date2num(t_dt)
    mask = np.isfinite(t_num) & np.isfinite(y) & np.isfinite(z)
    t_num, y, z = t_num[mask], y[mask], z[mask]
    if len(t_num) == 0:
        return None

    # Sort by time
    order = np.argsort(t_num)
    t_num, y, z = t_num[order], y[order], z[order]

    if norm is None:
        if vmin is None:
            vmin = np.nanquantile(z, 0.05)
        if vmax is None:
            vmax = np.nanquantile(z, 0.95)
        norm = plt.Normalize(vmin, vmax)

    # Build line segments: each segment connects consecutive points
    points = np.column_stack([t_num, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    seg_colors = 0.5 * (z[:-1] + z[1:])

    # Remove segments that span large time gaps
    if max_gap_s is not None:
        dt_days = np.diff(t_num)  # date2num is in days
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
    ax.invert_yaxis()
    ax.grid(True)
    if colorbar:
        cb = fig.colorbar(lc, ax=ax)
        cb.set_label(clabel)
    return lc


def generate_figures(glider, basedir):
    """Generate diagnostic figures for a single glider."""
    print(f"Processing {glider}...")

    # --- Load data ---
    flt = load_nc(os.path.join(basedir, f"{glider}.flt.nc"), "m_present_time")
    sci = load_nc(os.path.join(basedir, f"{glider}.sci.nc"), "sci_m_present_time")
    fltFull = load_nc(os.path.join(basedir, f"{glider}.fltFull.nc"), "m_present_time")
    sciFull = load_nc(os.path.join(basedir, f"{glider}.sciFull.nc"), "sci_m_present_time")

    # Mission start time (MRI times are ms since mission start)
    mission_start = flt["m_present_time"][0] - flt["m_present_secs_into_mission"][0]

    # GPS table
    gps_mask = np.isfinite(flt["m_lat"]) & np.isfinite(flt["m_lon"])
    gps_t = flt["t_dt"][gps_mask]
    gps_lat = mk_degrees(flt["m_lat"][gps_mask])
    gps_lon = mk_degrees(flt["m_lon"][gps_mask])

    # Process science data
    sci = sci_update(sci, gps_t, gps_lat, gps_lon)
    sciFull = sci_update(sciFull, gps_t, gps_lat, gps_lon)
    ctd = make_ctd(sci)
    ctdFull = make_ctd(sciFull)

    # Merge CTD, unique by time, filter temp > 0
    merged = {}
    for k in ("t_dt", "depth", "temp", "SP", "rho"):
        merged[k] = np.concatenate([ctd[k], ctdFull[k]])
    _, ix = np.unique(merged["t_dt"], return_index=True)
    for k in merged:
        merged[k] = merged[k][ix]
    tmask = merged["temp"] > 0
    for k in merged:
        merged[k] = merged[k][tmask]

    # Merged depth timeseries
    d_parts_t, d_parts_d = [], []
    flt_dm = np.isfinite(flt["m_depth"])
    d_parts_t.append(flt["t_dt"][flt_dm])
    d_parts_d.append(flt["m_depth"][flt_dm])
    for s in (sci, sciFull):
        dm = np.isfinite(s["depth"])
        d_parts_t.append(s["t_dt"][dm])
        d_parts_d.append(s["depth"][dm])
    d_t = np.concatenate(d_parts_t)
    d_depth = np.concatenate(d_parts_d)
    _, ix = np.unique(d_t, return_index=True)
    d_t, d_depth = d_t[ix], d_depth[ix]

    # Water depth
    wd_mask = np.isfinite(flt["m_water_depth"]) & (flt["m_water_depth"] > 1)
    wd_t = flt["t_dt"][wd_mask]
    wd_depth = flt["m_water_depth"][wd_mask]

    # Pitch & roll (merged flt + fltFull, unique, to degrees)
    pr_t = np.concatenate([flt["t_dt"], fltFull["t_dt"]])
    pr_pitch = np.concatenate([flt["m_pitch"], fltFull["m_pitch"]])
    pr_roll = np.concatenate([flt["m_roll"], fltFull["m_roll"]])
    _, ix = np.unique(pr_t, return_index=True)
    pitch_deg = np.degrees(pr_pitch[ix])
    roll_deg = np.degrees(pr_roll[ix])
    pitch_deg = pitch_deg[np.isfinite(pitch_deg)]
    roll_deg = roll_deg[np.isfinite(roll_deg)]

    # =========================================================
    # Figure 1: CTD & Flight — 3x2 grid
    # =========================================================
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), constrained_layout=True)
    fig.suptitle(f"{glider} — CTD & Flight Diagnostics", fontsize=14, fontweight="bold")

    # (0,0) Temperature
    ax = axes[0, 0]
    colored_line(ax, fig, merged["t_dt"], merged["depth"], merged["temp"],
                 "viridis", "Temperature (C)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Depth (m)")

    # (0,1) Salinity
    ax = axes[0, 1]
    colored_line(ax, fig, merged["t_dt"], merged["depth"], merged["SP"],
                 "viridis_r", "Salinity (PSU)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Depth (m)")

    # (1,0) Density
    ax = axes[1, 0]
    colored_line(ax, fig, merged["t_dt"], merged["depth"], merged["rho"],
                 "viridis_r", "Density (kg/m$^3$)")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Depth (m)")

    # (1,1) Depth + water depth
    ax = axes[1, 1]
    ax.plot(d_t, d_depth, "-", label="Glider depth")
    ax.plot(wd_t, wd_depth, "o", markersize=3, label="Water depth")
    ax.invert_yaxis()
    ax.grid(True)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Depth (m)")
    ax.legend(fontsize=8)

    # Link x and y axes on the top four panels
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.sharex(axes[0, 0])
        ax.sharey(axes[0, 0])

    # (2,0) Geographic track with coastline and bathymetry
    # Replace the plain axes with a cartopy GeoAxes
    pos = axes[2, 0].get_position()
    axes[2, 0].remove()
    ax = fig.add_axes(pos, projection=ccrs.PlateCarree())

    sci_mask = np.isfinite(sci["lon"]) & np.isfinite(sci["lat"])
    track_lon = sci["lon"][sci_mask]
    track_lat = sci["lat"][sci_mask]

    # Fetch and plot bathymetry
    try:
        blon, blat, bz = fetch_bathymetry(
            track_lon.min(), track_lon.max(), track_lat.min(), track_lat.max(),
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

    # (2,1) Pitch & Roll histograms (stacked)
    ax_top = axes[2, 1]
    ax_top.hist(pitch_deg, bins=100)
    ax_top.grid(True)
    ax_top.set_xlabel("Pitch (deg)")
    ax_top.set_ylabel("Count")
    # Add roll as inset below pitch by using a twin-x with offset
    ax_bot = ax_top.inset_axes([0, -1.1, 1, 0.9])
    ax_bot.hist(roll_deg, bins=100, color="tab:orange")
    ax_bot.grid(True)
    ax_bot.set_xlabel("Roll (deg)")
    ax_bot.set_ylabel("Count")

    # =========================================================
    # Figure 2: MRI/Turbulence — 2 rows (top: e1, e2 side-by-side; bottom: timeseries spanning both)
    # =========================================================
    mri_fn = os.path.join(basedir, f"{glider}.mri.nc")
    if not os.path.exists(mri_fn):
        print(f"  No MRI file for {glider}, skipping turbulence figure")
        plt.show()
        return

    mri_ds = nc.Dataset(mri_fn)
    mri_time_ms = mri_ds["time"][:]
    mri_pressure = mri_ds["pressure"][:]
    mri_e1 = mri_ds["e_1"][:]
    mri_e2 = mri_ds["e_2"][:]
    mri_ds.close()
    for name in ("mri_time_ms", "mri_pressure", "mri_e1", "mri_e2"):
        arr = locals()[name]
        if hasattr(arr, "filled"):
            locals()[name] = arr.filled(np.nan)
    # Re-bind after filling
    mri_time_ms = mri_time_ms.filled(np.nan) if hasattr(mri_time_ms, "filled") else mri_time_ms
    mri_pressure = mri_pressure.filled(np.nan) if hasattr(mri_pressure, "filled") else mri_pressure
    mri_e1 = mri_e1.filled(np.nan) if hasattr(mri_e1, "filled") else mri_e1
    mri_e2 = mri_e2.filled(np.nan) if hasattr(mri_e2, "filled") else mri_e2

    mri_mask = (np.isfinite(mri_time_ms) & np.isfinite(mri_pressure)
                & np.isfinite(mri_e1) & np.isfinite(mri_e2))
    mri_time_ms = mri_time_ms[mri_mask]
    mri_pressure = mri_pressure[mri_mask]
    mri_e1 = mri_e1[mri_mask]
    mri_e2 = mri_e2[mri_mask]

    if len(mri_time_ms) == 0:
        print(f"  No valid MRI data for {glider}, skipping turbulence figure")
        plt.show()
        return

    # MRI times are milliseconds since mission start — convert to POSIX then datetime
    mri_posix = mission_start + mri_time_ms / 1000.0
    mri_t = np.array(
        [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in mri_posix],
    )

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"{glider} — Turbulence Diagnostics", fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)

    # Estimate max gap: median time step * 3 to detect downcast gaps
    dt_ms = np.diff(np.sort(mri_time_ms))
    max_gap_s = np.median(dt_ms[dt_ms > 0]) / 1000.0 * 3

    # Identify surface gaps and split into profile segments
    mri_epoch = np.array([t.timestamp() for t in mri_t])
    dt_s = np.diff(mri_epoch)
    gap_thresh = np.median(dt_s[dt_s > 0]) * 3  # same logic as max_gap_s
    gap_ix = np.where(dt_s > gap_thresh)[0] + 1
    profile_starts = np.concatenate([[0], gap_ix])
    profile_ends = np.concatenate([gap_ix, [len(mri_t)]])

    # Per-profile moving median (window=3) of epsilon
    mri_e1_med = np.copy(mri_e1)
    mri_e2_med = np.copy(mri_e2)
    for s, e in zip(profile_starts, profile_ends):
        sl = slice(s, e)
        mri_e1_med[sl] = median_filter(mri_e1[sl], size=3)
        mri_e2_med[sl] = median_filter(mri_e2[sl], size=3)

    # Joint colorbar limits: 5%-95% quantile of combined median-filtered e1 and e2
    all_eps = np.concatenate([mri_e1_med, mri_e2_med])
    all_eps = all_eps[np.isfinite(all_eps)]
    eps_vmin = np.nanquantile(all_eps, 0.05)
    eps_vmax = np.nanquantile(all_eps, 0.95)
    eps_norm = plt.Normalize(eps_vmin, eps_vmax)

    # Top-left: Epsilon_1 median vs pressure
    ax1 = fig.add_subplot(gs[0, 0])
    lc1 = colored_line(ax1, fig, mri_t, mri_pressure, mri_e1_med, "viridis",
                        None, norm=eps_norm, max_gap_s=max_gap_s, colorbar=False)
    ax1.set_xlabel("Time (UTC)")
    ax1.set_ylabel("Pressure (dbar)")
    ax1.set_title(r"$\epsilon_1$ median")

    # Top-right: Epsilon_2 median vs pressure
    ax2 = fig.add_subplot(gs[0, 1])
    lc2 = colored_line(ax2, fig, mri_t, mri_pressure, mri_e2_med, "viridis",
                        None, norm=eps_norm, max_gap_s=max_gap_s, colorbar=False)
    ax2.set_xlabel("Time (UTC)")
    ax2.set_ylabel("Pressure (dbar)")
    ax2.set_title(r"$\epsilon_2$ median")

    # Link axes on the two epsilon panels
    ax2.sharex(ax1)
    ax2.sharey(ax1)

    # Joint colorbar for both epsilon panels
    cb = fig.colorbar(lc1, ax=[ax1, ax2], location="right", shrink=0.8)
    cb.set_label(r"$\log_{10}(\epsilon)$ median (W/kg)")

    # Bottom: Epsilon time series (spanning both columns), breaking at surface gaps
    ax3 = fig.add_subplot(gs[1, :])

    for s, e in zip(profile_starts, profile_ends):
        sl = slice(s, e)
        ax3.plot(mri_t[sl], mri_e1[sl], ".-", markersize=3, color="tab:blue",
                 label=r"$\log_{10}(\epsilon_1)$" if s == 0 else None)
        ax3.plot(mri_t[sl], mri_e2[sl], ".-", markersize=3, color="tab:orange",
                 label=r"$\log_{10}(\epsilon_2)$" if s == 0 else None)
        ax3.plot(mri_t[sl], median_filter(mri_e1[sl], size=3), "--",
                 color="tab:blue", linewidth=1.5,
                 label=r"$\log_{10}(\epsilon_1)$ median" if s == 0 else None)
        ax3.plot(mri_t[sl], median_filter(mri_e2[sl], size=3), "--",
                 color="tab:orange", linewidth=1.5,
                 label=r"$\log_{10}(\epsilon_2)$ median" if s == 0 else None)

    ax3.grid(True)
    ax3.set_xlabel("Time (UTC)")
    ax3.set_ylabel(r"$\log_{10}(\epsilon)$ (W/kg)")
    ax3.legend(loc="upper left")

    # Detect anomalous spikes and compute their periodicity
    spike_parts = []
    for name, eps in [(r"$\epsilon_1$", mri_e1), (r"$\epsilon_2$", mri_e2)]:
        peaks, _ = find_peaks(eps, prominence=5, distance=3)
        if len(peaks) > 1:
            peak_times = mri_epoch[peaks]
            dt_peaks = np.diff(peak_times)
            # Keep only within-profile intervals; surface gaps are typically
            # > 500s while spike intervals are ~100-300s
            dt_intra = dt_peaks[dt_peaks < 500]
            if len(dt_intra) > 0:
                mean_s = np.mean(dt_intra)
                std_s = np.std(dt_intra)
                spike_parts.append(
                    f"{name}: {mean_s:.0f} \u00b1 {std_s:.0f} s (n={len(dt_intra)})"
                )
    if spike_parts:
        ax3.set_title("Spike interval:  " + ",   ".join(spike_parts), fontsize=10)

    for ax in (ax1, ax2, ax3):
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")

    # =========================================================
    # Figure 3: Profile Walker — interactive profile-by-profile viewer
    # Top row: ε₁ and ε₂ vs depth (same as fig 2 top, but with depth).
    # Bottom row: single profile with depth on y-axis and three x-axes
    #             (temperature, salinity, epsilon).
    # =========================================================

    # Convert MRI pressure to depth
    mean_lat = np.nanmean(gps_lat)
    mri_depth = -gsw.z_from_p(mri_pressure, mean_lat)

    # Pre-compute per-profile data (MRI + matching CTD)
    prof_data = []
    for s, e in zip(profile_starts, profile_ends):
        sl = slice(s, e)
        t_s, t_e = mri_t[s], mri_t[e - 1]
        cmask = (merged["t_dt"] >= t_s) & (merged["t_dt"] <= t_e)
        prof_data.append({
            "t_start": t_s, "t_end": t_e,
            "depth": mri_depth[sl],
            "e1": mri_e1_med[sl], "e2": mri_e2_med[sl],
            "ctd_depth": merged["depth"][cmask] if cmask.any() else np.array([]),
            "temp":      merged["temp"][cmask]  if cmask.any() else np.array([]),
            "SP":        merged["SP"][cmask]    if cmask.any() else np.array([]),
        })
    n_prof = len(prof_data)

    if n_prof == 0:
        print(f"  No MRI profiles detected for {glider}, skipping profile walker")
        plt.show()
        return

    fig3 = plt.figure(figsize=(16, 10))
    fig3.suptitle(f"{glider} \u2014 Profile Walker  [1/{n_prof}]",
                  fontsize=14, fontweight="bold")
    gs3 = fig3.add_gridspec(2, 2, height_ratios=[1, 1.2],
                            hspace=0.35, wspace=0.3)

    # --- Top panels: ε₁ and ε₂ vs depth, colored by epsilon ---
    ax_top1 = fig3.add_subplot(gs3[0, 0])
    lc3_1 = colored_line(ax_top1, fig3, mri_t, mri_depth, mri_e1_med, "viridis",
                         None, norm=eps_norm, max_gap_s=max_gap_s, colorbar=False)
    ax_top1.set_xlabel("Time (UTC)")
    ax_top1.set_ylabel("Depth (m)")
    ax_top1.set_title(r"$\epsilon_1$ median")

    ax_top2 = fig3.add_subplot(gs3[0, 1])
    lc3_2 = colored_line(ax_top2, fig3, mri_t, mri_depth, mri_e2_med, "viridis",
                         None, norm=eps_norm, max_gap_s=max_gap_s, colorbar=False)
    ax_top2.set_xlabel("Time (UTC)")
    ax_top2.set_ylabel("Depth (m)")
    ax_top2.set_title(r"$\epsilon_2$ median")

    # Link top panel time and depth axes
    ax_top2.sharex(ax_top1)
    ax_top2.sharey(ax_top1)

    # Joint colorbar for top panels
    if lc3_1 is not None:
        fig3.colorbar(lc3_1, ax=[ax_top1, ax_top2], location="right",
                      shrink=0.8).set_label(r"$\log_{10}(\epsilon)$ median (W/kg)")

    # --- Bottom panel: multi-x-axis profile view ---
    ax_bt = fig3.add_subplot(gs3[1, :])
    ax_bt.sharey(ax_top1)  # link depth axis with top panels
    ax_bt.set_ylabel("Depth (m)")
    ax_bt.set_xlabel("Temperature (\u00b0C)", color="tab:red")
    ax_bt.tick_params(axis="x", labelcolor="tab:red")
    ax_bt.grid(True, alpha=0.3)

    ax_bs = ax_bt.twiny()  # salinity x-axis (top)
    ax_bs.set_xlabel("Salinity (PSU)", color="tab:green")
    ax_bs.tick_params(axis="x", labelcolor="tab:green")

    ax_be = ax_bt.twiny()  # epsilon x-axis (offset to bottom)
    ax_be.xaxis.set_ticks_position("bottom")
    ax_be.xaxis.set_label_position("bottom")
    ax_be.spines["bottom"].set_position(("outward", 45))
    ax_be.set_xlabel(r"$\log_{10}(\epsilon)$ (W/kg)", color="tab:blue")
    ax_be.tick_params(axis="x", labelcolor="tab:blue")

    # Static epsilon legend (independent of line objects)
    ax_be.legend(handles=[
        Line2D([0], [0], color="tab:blue", marker="o", ls="none",
               label=r"$\log_{10}(\epsilon_1)$"),
        Line2D([0], [0], color="tab:orange", marker="s", ls="none",
               label=r"$\log_{10}(\epsilon_2)$"),
    ], loc="lower right", fontsize=8)

    # --- Interactive state & update logic ---
    _st = dict(idx=0, playing=False, timer=None,
               span1=None, span2=None, lines=[], updating=False)

    def _update(idx):
        if _st["updating"]:
            return
        _st["updating"] = True
        idx = max(0, min(idx, n_prof - 1))
        _st["idx"] = idx
        p = prof_data[idx]
        fig3.suptitle(f"{glider} \u2014 Profile Walker  [{idx + 1}/{n_prof}]",
                      fontsize=14, fontweight="bold")

        # Update highlight spans on top panels
        for key in ("span1", "span2"):
            if _st[key] is not None:
                _st[key].remove()
        ts = mdates.date2num(p["t_start"])
        te = mdates.date2num(p["t_end"])
        _st["span1"] = ax_top1.axvspan(ts, te, alpha=0.3, color="yellow", zorder=0)
        _st["span2"] = ax_top2.axvspan(ts, te, alpha=0.3, color="yellow", zorder=0)

        # Clear previous profile lines
        for ln in _st["lines"]:
            ln.remove()
        _st["lines"] = []

        # Temperature
        if len(p["temp"]) > 0:
            ln, = ax_bt.plot(p["temp"], p["ctd_depth"], "-o",
                             color="tab:red", ms=3, lw=1.5)
            _st["lines"].append(ln)
            rng = (p["temp"].max() - p["temp"].min()) * 0.05 or 0.1
            ax_bt.set_xlim(p["temp"].min() - rng, p["temp"].max() + rng)
        else:
            ax_bt.set_xlim(0, 1)

        # Salinity
        if len(p["SP"]) > 0:
            ln, = ax_bs.plot(p["SP"], p["ctd_depth"], "-s",
                             color="tab:green", ms=3, lw=1.5)
            _st["lines"].append(ln)
            rng = (p["SP"].max() - p["SP"].min()) * 0.05 or 0.01
            ax_bs.set_xlim(p["SP"].min() - rng, p["SP"].max() + rng)
        else:
            ax_bs.set_xlim(0, 1)

        # Epsilon (both ε₁ and ε₂) — markers only
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

        fig3.canvas.draw_idle()
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
            fig3.canvas.draw_idle()

    def _play(event):
        if _st["playing"]:
            _st["playing"] = False
            if _st["timer"]:
                _st["timer"].stop()
            btn_play.label.set_text("Play")
            fig3.canvas.draw_idle()
        else:
            _st["playing"] = True
            btn_play.label.set_text("Pause")
            fig3.canvas.draw_idle()
            _st["timer"] = fig3.canvas.new_timer(interval=1000)
            _st["timer"].add_callback(_advance)
            _st["timer"].start()

    def _on_xlim(ax):
        """When a top panel is zoomed, jump to the first full profile in view."""
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

    # Navigation buttons (northeast corner)
    btn_prev = Button(fig3.add_axes([0.62, 0.94, 0.08, 0.035]), "Previous")
    btn_play = Button(fig3.add_axes([0.71, 0.94, 0.08, 0.035]), "Play")
    btn_next = Button(fig3.add_axes([0.80, 0.94, 0.08, 0.035]), "Next")
    btn_prev.on_clicked(_prev)
    btn_play.on_clicked(_play)
    btn_next.on_clicked(_next)

    # Rotate top-panel tick labels
    for ax in (ax_top1, ax_top2):
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")

    # Initialize with first profile
    _update(0)

    # =========================================================
    # Figure 4: CTD Derivatives — dT/dz, dS/dz, dρ/dz
    # =========================================================

    # Sort merged CTD by time
    ctd_order = np.argsort(merged["t_dt"])
    ctd_t4 = merged["t_dt"][ctd_order]
    ctd_depth4 = merged["depth"][ctd_order]
    ctd_temp4 = merged["temp"][ctd_order]
    ctd_SP4 = merged["SP"][ctd_order]
    ctd_rho4 = merged["rho"][ctd_order]

    # Detect profile boundaries via time gaps
    ctd_epoch4 = np.array([t.timestamp() for t in ctd_t4])
    ctd_dt4 = np.diff(ctd_epoch4)
    ctd_gap4 = np.median(ctd_dt4[ctd_dt4 > 0]) * 3
    ctd_gix4 = np.where(ctd_dt4 > ctd_gap4)[0] + 1
    ctd_ps4 = np.concatenate([[0], ctd_gix4])
    ctd_pe4 = np.concatenate([ctd_gix4, [len(ctd_t4)]])

    # Compute per-meter derivatives within each profile
    dt_list, dz_list = [], []
    dTdz_list, dSdz_list, drhodz_list = [], [], []

    for s, e in zip(ctd_ps4, ctd_pe4):
        if e - s < 2:
            continue
        dz = np.diff(ctd_depth4[s:e])
        ok = np.abs(dz) > 0.01  # need at least 1 cm depth change
        mid_t = ctd_t4[s:e - 1]
        mid_z = 0.5 * (ctd_depth4[s:e - 1] + ctd_depth4[s + 1:e])
        safe_dz = np.where(ok, dz, np.nan)
        dTdz = np.diff(ctd_temp4[s:e]) / safe_dz
        dSdz = np.diff(ctd_SP4[s:e]) / safe_dz
        drhodz = np.diff(ctd_rho4[s:e]) / safe_dz
        dt_list.append(mid_t)
        dz_list.append(mid_z)
        dTdz_list.append(dTdz)
        dSdz_list.append(dSdz)
        drhodz_list.append(drhodz)

    if dt_list:
        d_t_all = np.concatenate(dt_list)
        d_z_all = np.concatenate(dz_list)
        dTdz_all = np.concatenate(dTdz_list)
        dSdz_all = np.concatenate(dSdz_list)
        drhodz_all = np.concatenate(drhodz_list)

        fig4, axes4 = plt.subplots(3, 1, figsize=(16, 10), constrained_layout=True,
                                   sharex=True, sharey=True)
        fig4.suptitle(f"{glider} \u2014 CTD Derivatives (per meter)",
                      fontsize=14, fontweight="bold")

        # Scale to milli-units; compute color limits from data below 125 m
        deep = d_z_all > 125

        derivatives = [
            (dTdz_all * 1000, "viridis", "$dT/dz$ (m\N{DEGREE SIGN}C/m)"),
            (dSdz_all * 1000, "viridis", r"$dS_{P}/dz$ (mPSU/m)"),
            (drhodz_all * 1000, "viridis", r"$d\rho/dz$ (g/m$^3$/m)"),
        ]

        for ax4, (ddata, cmap, clabel) in zip(axes4, derivatives):
            deep_vals = ddata[deep]
            deep_vals = deep_vals[np.isfinite(deep_vals)]
            if len(deep_vals) > 0:
                vmin = np.nanquantile(deep_vals, 0.05)
                vmax = np.nanquantile(deep_vals, 0.95)
                linthresh = np.nanmedian(np.abs(deep_vals))
                norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
            else:
                norm = None
            colored_line(ax4, fig4, d_t_all, d_z_all, ddata, cmap, clabel,
                         norm=norm, max_gap_s=ctd_gap4)
            ax4.set_ylabel("Depth (m)")

        axes4[-1].set_xlabel("Time (UTC)")

        # Link figure 4 axes to figure 3 top panels (time and depth)
        axes4[0].sharex(ax_top1)
        axes4[0].sharey(ax_top1)

        for ax4 in axes4:
            for lbl in ax4.get_xticklabels():
                lbl.set_rotation(30)
                lbl.set_ha("right")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate RIOT glider diagnostic figures")
    parser.add_argument("--gliders", nargs="+", default=["osu684", "osu685"],
                        help="Glider names to process (default: osu684 osu685)")
    parser.add_argument("--basedir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Base directory containing NetCDF files")
    args = parser.parse_args()

    for glider in args.gliders:
        generate_figures(glider, args.basedir)


if __name__ == "__main__":
    main()
