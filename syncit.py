#!/usr/bin/env python3

import argparse
import glob
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

CONVERSIONS = [
    # (subdir, glob_pattern, tool, output_suffix)
    ("from-glider", "*.s?d", "dbd2netcdf", "flt"),
    ("from-glider", "*.t?d", "dbd2netcdf", "sci"),
    ("from-glider", "*.d?d", "dbd2netcdf", "fltFull"),
    ("from-glider", "*.e?d", "dbd2netcdf", "sciFull"),
    ("from-glider", "*.mri", "q2netcdf", "mri"),
    ("logs", "*.log", "log-harvest", "logs"),
]


def read_manifest(path):
    """Read a manifest file (one filename per line) and return as a list."""
    try:
        with open(path) as f:
            return f.read().splitlines()
    except FileNotFoundError:
        return None


def write_manifest(path, files):
    """Write sorted filenames to a manifest file."""
    with open(path, "w") as f:
        f.write("\n".join(files) + "\n")


def run_tool(tool, out, cache, files):
    """Build and run the conversion command, return (returncode, stderr)."""
    if tool == "dbd2netcdf":
        cmd = ["dbd2netcdf", "--skipFirst", "-o", out, "--cache", cache] + files
    elif tool == "log-harvest":
        cmd = ["slocum-tpw", "log-harvest", f"--nc={out}"] + files
    else:  # q2netcdf
        cmd = ["q2netcdf", "--nc", out] + files
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode, result.stderr.decode("utf-8", errors="replace")



def convert(target, glider, subdir, pattern, tool, suffix, force=False):
    """Run a single conversion and return (desc, returncode, stderr, elapsed, nfiles, mode)."""
    src = os.path.join(target, glider, subdir)
    files = sorted(glob.glob(os.path.join(src, pattern)))
    desc = f"{glider} {pattern}"
    if not files:
        return (desc, None, "no files", 0.0, 0, "skip")

    out = os.path.join(target, f"{glider}.{suffix}.nc")
    manifest_path = out + ".files"
    cache = os.path.join(target, "cache")
    desc = f"{glider} {pattern} -> {out}"

    # Check manifest for changes
    old_files = read_manifest(manifest_path)
    if not force and old_files is not None and old_files == files:
        return (desc, 0, "", 0.0, len(files), "up-to-date")

    t0 = time.monotonic()

    # Full rebuild
    rc, stderr = run_tool(tool, out, cache, files)
    elapsed = time.monotonic() - t0
    if rc == 0:
        write_manifest(manifest_path, files)
    return (desc, rc, stderr, elapsed, len(files), "full")



def main():
    parser = argparse.ArgumentParser(description="Sync glider data and convert to NetCDF")
    parser.add_argument(
        "--hostname", default="gliderfs2.ceoas.oregonstate.edu",
        help="rsync source hostname (default: gliderfs2.ceoas.oregonstate.edu)",
    )
    parser.add_argument(
        "--glider", action="append", dest="gliders",
        help="glider name (repeatable; default: osu684 osu685)",
    )
    parser.add_argument(
        "--remotedir", default="/data/Dockserver/gliderfmc0",
        help="remote directory containing glider folders (default: /data/Dockserver/gliderfmc0)",
    )
    parser.add_argument(
        "--target", default=".",
        help="local target directory for rsync and output files (default: .)",
    )
    fetch_group = parser.add_mutually_exclusive_group()
    fetch_group.add_argument(
        "--fetch", action="store_true", default=True, dest="fetch",
        help="run rsync before conversions (default)",
    )
    fetch_group.add_argument(
        "--nofetch", action="store_false", dest="fetch",
        help="skip rsync, only run conversions",
    )
    parser.add_argument(
        "--convert", action="store_true",
        help="force full conversion, ignoring manifests",
    )
    args = parser.parse_args()

    gliders = args.gliders if args.gliders else ["osu684", "osu685"]
    target = args.target

    # Step 1: rsync (must complete before conversions)
    if args.fetch:
        print(f"Syncing from {args.hostname} ...")
        sources = [f"{args.hostname}:{args.remotedir}/{g}" for g in gliders]
        rc = subprocess.call(["rsync", "--archive", "--verbose", "--compress"] + sources + [target])
        if rc != 0:
            sys.exit(f"rsync failed with exit code {rc}")

    # Step 2: parallel conversions
    jobs = []
    for glider in gliders:
        for subdir, pattern, tool, suffix in CONVERSIONS:
            src = os.path.join(target, glider, subdir)
            if not glob.glob(os.path.join(src, "")):
                print(f"{src} does not exist, skipping")
                continue
            jobs.append((target, glider, subdir, pattern, tool, suffix, args.convert))

    errors = []
    t_start = time.monotonic()
    with ThreadPoolExecutor() as pool:
        futures = {pool.submit(convert, *job): job for job in jobs}
        for future in as_completed(futures):
            desc, rc, stderr, elapsed, nfiles, mode = future.result()
            if rc is None:
                print(f"  {desc}: skipped ({stderr})")
            elif mode == "up-to-date":
                print(f"  {desc}: up-to-date ({nfiles} files)")
            elif rc == 0:
                print(f"  {desc}: ok ({nfiles} files, {elapsed:.1f}s)")
            else:
                print(f"  {desc}: FAILED (rc={rc}, {elapsed:.1f}s)")
                if stderr:
                    print(f"    {stderr.rstrip()}")
                errors.append(desc)
    t_total = time.monotonic() - t_start
    print(f"\nConversions finished in {t_total:.1f}s (wall clock)")

    if errors:
        print(f"\n{len(errors)} conversion(s) failed:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
