#!/usr/bin/env python3
"""
check_confound_headers.py — Verify DeepPrep confounds TSV has the columns
that BetaS2_deepprep.m expects (24-param HMP + 5 aCompCor = 29 total).

Usage (one file):
    python check_confound_headers.py path/to/sub-001_..._desc-confounds_timeseries.tsv

Usage (all subjects, one run):
    python check_confound_headers.py /blue/.../deepprep/BOLD/sub-*/func/*run-01*confounds*.tsv
"""

import sys
import glob

EXPECTED = [
    # 24-param HMP
    "trans_x",                    "trans_x_derivative1",
    "trans_x_power2",             "trans_x_derivative1_power2",
    "trans_y",                    "trans_y_derivative1",
    "trans_y_power2",             "trans_y_derivative1_power2",
    "trans_z",                    "trans_z_derivative1",
    "trans_z_power2",             "trans_z_derivative1_power2",
    "rot_x",                      "rot_x_derivative1",
    "rot_x_power2",               "rot_x_derivative1_power2",
    "rot_y",                      "rot_y_derivative1",
    "rot_y_power2",               "rot_y_derivative1_power2",
    "rot_z",                      "rot_z_derivative1",
    "rot_z_power2",               "rot_z_derivative1_power2",
    # 5 aCompCor
    "a_comp_cor_00", "a_comp_cor_01", "a_comp_cor_02",
    "a_comp_cor_03", "a_comp_cor_04",
]


def check_tsv(path):
    """Report whether one TSV contains every regressor expected by the GLM."""
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
    header_set = set(header)

    missing = [c for c in EXPECTED if c not in header_set]
    present = [c for c in EXPECTED if c in header_set]

    status = "OK" if not missing else "MISSING"
    print(f"\n[{status}] {path}")
    print(f"  Total columns in file : {len(header)}")
    print(f"  Expected present      : {len(present)}/{len(EXPECTED)}")
    if missing:
        print(f"  MISSING columns       : {missing}")

    return not missing


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    # Expand any globs passed as arguments
    paths = []
    for arg in sys.argv[1:]:
        expanded = glob.glob(arg)
        paths.extend(expanded if expanded else [arg])

    all_ok = True
    for p in sorted(paths):
        ok = check_tsv(p)
        all_ok = all_ok and ok

    print(f"\n{'All files OK' if all_ok else 'SOME FILES HAVE MISSING COLUMNS'}")
    sys.exit(0 if all_ok else 1)
