#!/usr/bin/env python3

import argparse
import glob
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

CONVERSIONS = [
    # (subdir, glob_pattern, tool, output_suffix)
    ("from-glider", "*.s?d", "dbd2netcdf", "flt"),
    ("from-glider", "*.t?d", "dbd2netcdf", "sci"),
    ("from-glider", "*.d?d", "dbd2netcdf", "fltFull"),
    ("from-glider", "*.e?d", "dbd2netcdf", "sciFull"),
    ("from-glider", "*.mri", "q2netcdf", "mri"),
    ("logs", "*.log", "log-harvest", "logs"),
]


def convert(target, glider, subdir, pattern, tool, suffix):
    """Run a single conversion and return (description, returncode, stderr)."""
    src = os.path.join(target, glider, subdir)
    files = sorted(glob.glob(os.path.join(src, pattern)))
    if not files:
        return (f"{glider} {pattern}", None, "no files")

    out = os.path.join(target, f"{glider}.{suffix}.nc")
    cache = os.path.join(target, "cache")
    if tool == "dbd2netcdf":
        cmd = ["dbd2netcdf", "--skipFirst", "-o", out, "--cache", cache] + files
    elif tool == "log-harvest":
        cmd = ["slocum-tpw", "log-harvest", f"--nc={out}"] + files
    else:  # q2netcdf
        cmd = ["q2netcdf", "--nc", out] + files

    result = subprocess.run(cmd, capture_output=True, text=True)
    return (f"{glider} {pattern} -> {out}", result.returncode, result.stderr)



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
    args = parser.parse_args()

    gliders = args.gliders if args.gliders else ["osu684", "osu685"]
    target = args.target

    # Step 1: rsync (must complete before conversions)
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
            jobs.append((target, glider, subdir, pattern, tool, suffix))

    errors = []
    with ProcessPoolExecutor() as pool:
        futures = {pool.submit(convert, *job): job for job in jobs}
        for future in as_completed(futures):
            desc, rc, stderr = future.result()
            if rc is None:
                print(f"  {desc}: skipped ({stderr})")
            elif rc == 0:
                print(f"  {desc}: ok")
            else:
                print(f"  {desc}: FAILED (rc={rc})")
                if stderr:
                    print(f"    {stderr.rstrip()}")
                errors.append(desc)

    if errors:
        print(f"\n{len(errors)} conversion(s) failed:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
