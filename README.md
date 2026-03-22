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
./examine.py [--gliders NAME ...] [--basedir DIR]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--gliders` | `osu684 osu685` | Glider names to process |
| `--basedir` | `.` | Directory containing the NetCDF files |

**Figures produced (per glider):**

1. **CTD & Flight Diagnostics** (3x2 grid)
   - Temperature vs depth (colored by value)
   - Salinity vs depth
   - Density vs depth
   - Glider depth and water depth vs time
   - Geographic track with ETOPO2022 bathymetry
   - Pitch and roll histograms

2. **Turbulence Diagnostics** (requires MRI data)
   - Epsilon-1 and epsilon-2 median-filtered dissipation rates vs pressure, colored by magnitude
   - Raw and median-filtered epsilon time series with spike periodicity analysis

3. **Profile Walker** (interactive)
   - Overview panels of epsilon-1 and epsilon-2 vs depth over time
   - Single-profile view with temperature, salinity, and epsilon vs depth
   - Previous/Next/Play buttons to step or animate through profiles
   - Zoom-linked: zooming the overview panels jumps the profile view

4. **CTD Derivatives**
   - dT/dz, dS/dz, and d-rho/dz per meter vs time and depth, using symmetric log color scaling based on data below 125 m

## cache directory

The `cache/` directory contains `.cac` sensor-list cache files used by `dbd2netcdf` to decode Slocum binary data files. These must be present for conversion to succeed.
