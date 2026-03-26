# RIOT

[![License: GPLv3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/mousebrains/RIOT/actions/workflows/ci.yml/badge.svg)](https://github.com/mousebrains/RIOT/actions/workflows/ci.yml)

Data acquisition and diagnostic tools for the RIOT (Rapid Investigation of Turbulence) Slocum glider deployment.

## Installation

### Python dependencies

```sh
python3 -m pip install -r requirements.txt
```

This installs the required Python dependencies. The scripts are run directly from this directory.

### dbd2netcdf (external)

`dbd2netcdf` is a compiled C++ tool required by `syncit` for converting Slocum binary data files. Install it from source:

```sh
git clone git@github.com:mousebrains/dbd2netcdf.git
cd dbd2netcdf && mkdir build && cd build && cmake .. && make && make install
```

## Commands

### syncit

Syncs raw glider data from a dockserver via rsync and converts it to NetCDF files in parallel.

```sh
./syncit.py [--hostname HOST] [--remotedir DIR] [--glider NAME ...] [--target DIR]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--hostname` | `gliderfs2.ceoas.oregonstate.edu` | rsync source hostname |
| `--remotedir` | `/data/Dockserver/gliderfmc0` | Remote directory containing glider folders |
| `--glider` | `osu684 osu685` | Glider name (repeatable) |
| `--target` | `.` | Local directory for synced data and output files |

**What it does:**

1. **rsync** — pulls each glider's `from-glider/` and `logs/` directories from the remote directory on the source host.
2. **Parallel conversion** — converts raw files to NetCDF using:

   | Source files | Tool | Output |
   |-------------|------|--------|
   | `*.s?d` (subset flight) | `dbd2netcdf` | `{glider}.flt.nc` |
   | `*.t?d` (subset science) | `dbd2netcdf` | `{glider}.sci.nc` |
   | `*.d?d` (full flight) | `dbd2netcdf` | `{glider}.fltFull.nc` |
   | `*.e?d` (full science) | `dbd2netcdf` | `{glider}.sciFull.nc` |
   | `*.mri` (MicroRider) | `q2netcdf` | `{glider}.mri.nc` |
   | `*.log` (glider logs) | `slocum-tpw log-harvest` | `{glider}.logs.nc` |

**Example:**

```sh
./syncit.py --hostname myhost.example.com --glider osu684 --glider osu685 --target /data/riot
```

### examine

Generates interactive diagnostic figures from the NetCDF files produced by `syncit`.

```sh
./examine.py [--gliders NAME ...] [--basedir DIR] [--figure FIGS ...]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--gliders` | `osu684 osu685` | Glider names to process |
| `--basedir` | `.` | Directory containing the NetCDF files |
| `--figure` | all | Figures to display (1-8). Accepts multiple values, ranges, or comma-separated lists (e.g. `--figure 1 3`, `--figure 1-3`, `--figure 2,4`) |

**Figures produced (per glider):**

1. **CTD & Flight Diagnostics** (3x2 grid)
   - Temperature, salinity, and density vs depth over time (colored lines, shared axes)
   - Geographic track with ETOPO2022 bathymetry, colored by age with last-position marker
   - Pitch and roll histograms

2. **Turbulence Diagnostics** (requires MRI data, 2x2 grid)
   - Epsilon-1 and epsilon-2 median-filtered dissipation rates vs pressure, colored by magnitude
   - Raw and median-filtered epsilon time series with spike periodicity analysis

3. **Profile Walker** (interactive, requires MRI data)
   - Overview panels of epsilon-1 and epsilon-2 vs depth over time
   - Single-profile view with temperature, salinity, and epsilon vs depth
   - Previous/Next/Play buttons to step or animate through profiles
   - Zoom-linked: zooming the overview panels jumps the profile view

4. **CTD Derivatives** (3x1 grid)
   - dT/dz, dS/dz, and d-rho/dz per meter vs time and depth, using symmetric log color scaling based on data below 125 m

5. **Flight Health** (3x1 grid)
   - Pitch, roll, and battery voltage vs time, with thruster power on roll panel

6. **Flight Control** (3x1 grid)
   - Pitch vs battery position, roll vs depth, vertical speed vs oil volume (dual-axis panels)

7. **Depth Overview** (requires MRI data, 3x2 grid)
   - Depth colored by density with battery position, depth colored by oil volume with pitch
   - Epsilon-1 and epsilon-2 median vs depth (bottom panels)

8. **Modem Despike Filter** (requires MRI data, 2x2 grid)
   - Epsilon-1 and epsilon-2 with contaminated samples highlighted in red
   - Filtered (clean) epsilon time series

### recoverBy

Generates a recover-by comparison figure showing battery decay fits across multiple exponential weighting windows for each glider.

```sh
./recoverBy.py [--gliders NAME ...] [--basedir DIR] [--output FILE]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--gliders` | all `*.logs.nc` in basedir | Glider names to process |
| `--basedir` | `.` | Directory containing `.logs.nc` files |
| `--output` | (interactive) | Save figure to file instead of displaying |

**What it does:**

Produces one subplot per glider, each showing the raw battery data with linear fits computed using `slocum-tpw recover-by` at multiple tau values (0.1, 0.5, 1, 7, 14 days) plus an unweighted full-deployment fit. Each fit line extends to the projected recovery date (battery at 15%) with 95% confidence intervals.

## Cache files

The `cache/` directory contains `.cac` sensor-list cache files used by `dbd2netcdf` to decode Slocum binary data files. These must be present for conversion to succeed.

`examine` caches ETOPO2022 bathymetry in `.bathy_cache.pkl` (inside `--basedir`) after the first OPeNDAP fetch to avoid repeated network requests. The cache is automatically extended if a future run covers a larger geographic area. Delete `.bathy_cache.pkl` to force a fresh download.
