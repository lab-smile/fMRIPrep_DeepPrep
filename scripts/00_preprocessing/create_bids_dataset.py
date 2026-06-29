#!/usr/bin/env python3
"""
create_bids_dataset.py

Organizes raw NIfTI files (sub-001 to sub-016) and MATLAB onset files
(Sub##run#.mat) into a valid BIDS dataset ready for fMRIPrep / DeepPrep.

Study: Bo et al. (2021) - Decoding Neural Representations of Affective
       Scenes in Retinotopic Visual Cortex.
       3T Philips Achieva | TR=1.98s | 36 slices (ascending) | 5 runs

Usage:
    python create_bids_dataset.py \
        --input-dir  /path/to/raw/sub-XXX-folders \
        --output-dir /path/to/bids \
        --onset-dir  /path/to/NewStimuluesSetting

Subject renaming note:
    Original sub-010 to sub-013 had no T1 and were excluded. The remaining
    subjects were renamed so the folders are sub-001..sub-016, but the MATLAB
    onset files still use the original numbering (Sub01-Sub09, Sub14-Sub20).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.io

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TR = 1.98          # seconds
N_SLICES = 36
N_RUNS = 5
N_TRIALS_PER_COND = 20
STIMULUS_DURATION_S = round(1.51515151 * TR, 6)  # 3.0 seconds

# Mapping: new folder index (1-based) → original MATLAB subject number
# sub-001..sub-009 → Sub01..Sub09 (same), sub-010..sub-016 → Sub14..Sub20
ORIG_SUB_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20]


def orig_sub_id(new_sub_int: int) -> int:
    """Return the original MATLAB subject number for a new folder index."""
    return ORIG_SUB_IDS[new_sub_int - 1]


# ---------------------------------------------------------------------------
# Root-level BIDS files
# ---------------------------------------------------------------------------

DATASET_DESCRIPTION = {
    "Name": "Decoding Affective Scenes in Retinotopic Visual Cortex",
    "BIDSVersion": "1.9.0",
    "License": "CC0",
    "Authors": [
        "Ke Bo", "Siyang Yin", "Yuelu Liu", "Zhenhong Hu",
        "Sreenivasan Meyyappan", "Sungkean Kim", "Andreas Keil", "Mingzhou Ding"
    ],
    "ReferencesAndLinks": ["https://doi.org/10.1093/cercor/bhaa411"],
    "DatasetDOI": "n/a"
}

PARTICIPANTS_COLUMNS = {
    "participant_id": {"Description": "Unique participant identifier"},
    "sex":            {"Description": "Biological sex", "Levels": {"M": "Male", "F": "Female"}},
    "age":            {"Description": "Age in years at time of scan"}
}

README_TEXT = """\
# Decoding Affective Scenes in Retinotopic Visual Cortex

BIDS dataset for the Bo et al. (2021) replication/comparison study.

## Study description
EEG-fMRI data recorded while participants (N=16) viewed pleasant, neutral, and
unpleasant pictures from the International Affective Picture System (IAPS).
Five sessions of 60 pictures each (20 per valence category).

## Scanner
3T Philips Achieva | TR=1.98 s | TE=30 ms | Flip=80° | 36 ascending slices
Voxel size: 3.5×3.5×3.5 mm | Matrix: 64×64 | SENSE factor: 2

## Notes
- Subjects 10–13 in the original dataset (N=20) lacked T1 anatomical images
  and were excluded. Remaining subjects were renumbered sub-001..sub-016.
- First 5 volumes of each run are non-steady-state (excluded in preprocessing).
- No fieldmaps were acquired.

## Reference
Bo K, Yin S, Liu Y, Hu Z, Meyyappan S, Kim S, Keil A, Ding M (2021).
Decoding Neural Representations of Affective Scenes in Retinotopic Visual Cortex.
Cerebral Cortex, 31(6): 3047–3063. https://doi.org/10.1093/cercor/bhaa411
"""

BIDSIGNORE = """\
*.DS_Store
code/
derivatives/
stimuli/
"""

# Task-level BOLD sidecar — inherited by all runs of task-emotion
# SliceTiming: ascending acquisition (slice i acquired at i * TR/N_SLICES)
_slice_timing = [round(i * TR / N_SLICES, 6) for i in range(N_SLICES)]

TASK_BOLD_JSON = {
    "TaskName": "emotion",
    "TaskDescription": (
        "Passive viewing of pleasant, neutral, and unpleasant pictures "
        "from the IAPS library. 60 pictures per run (20 per valence), 5 runs."
    ),
    "RepetitionTime": TR,
    "EchoTime": 0.030001,
    "FlipAngle": 80,
    "SliceTiming": _slice_timing,
    "SliceEncodingDirection": "k",
    "PhaseEncodingDirection": "i",
    "TotalReadoutTime": 0.018935,
    "EffectiveEchoSpacing": 0.000300556,
    "MagneticFieldStrength": 3,
    "Manufacturer": "Philips",
    "ManufacturersModelName": "Achieva",
    "ScanningSequence": "GR",
    "PulseSequenceName": "FEEPI",
    "ParallelReductionFactorInPlane": 2,
    "ParallelAcquisitionTechnique": "SENSE",
    "NumberOfVolumesDiscardedByUser": 5,
    "Instructions": (
        "Maintain central fixation throughout. Rate 12 representative pictures "
        "for hedonic valence and arousal after the scan."
    )
}

# Minimal T1w sidecar — full params unknown; scanner fields from BOLD JSON
T1W_JSON = {
    "MagneticFieldStrength": 3,
    "Manufacturer": "Philips",
    "ManufacturersModelName": "Achieva"
}


# ---------------------------------------------------------------------------
# Events TSV helpers
# ---------------------------------------------------------------------------

def load_events(mat_path: Path) -> list[dict]:
    """
    Load a Sub##run#.mat file and return a list of event dicts.

    Onset matrix layout (60×4, scan units):
        col 0 : all 60 trial onsets in presentation order (not used)
        col 1 : Pleasant onsets   (rows 0–19; rows 20–59 are zero-padding)
        col 2 : Neutral onsets    (rows 0–19)
        col 3 : Unpleasant onsets (rows 0–19)

    Duration = 1.51515… scans × TR ≈ 3.0 s
    """
    mat = scipy.io.loadmat(str(mat_path))
    onset_mat = mat["Onset"]  # (60, 4)

    rows = []
    for col_idx, prefix in [(1, "Pl"), (2, "Nt"), (3, "Up")]:
        for trial_i in range(N_TRIALS_PER_COND):
            onset_scans = float(onset_mat[trial_i, col_idx])
            rows.append({
                "onset":      round(onset_scans * TR, 4),
                "duration":   STIMULUS_DURATION_S,
                "trial_type": f"{prefix}{trial_i + 1:02d}"
            })

    rows.sort(key=lambda r: r["onset"])
    return rows


def write_events_tsv(rows: list[dict], out_path: Path) -> None:
    """Write BIDS event rows with stable decimal formatting."""
    header = "onset\tduration\ttrial_type\n"
    lines = [
        f"{r['onset']:.4f}\t{r['duration']:.4f}\t{r['trial_type']}\n"
        for r in rows
    ]
    out_path.write_text(header + "".join(lines))


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def find_t1w(anat_dir: Path) -> Path:
    """Return the T1w NIfTI in anat_dir. Raises if not found."""
    candidates = sorted(anat_dir.glob("*.nii.gz"))
    if not candidates:
        candidates = sorted(anat_dir.glob("*.nii"))
    if not candidates:
        raise FileNotFoundError(f"No NIfTI found in {anat_dir}")
    return candidates[0]


def find_bold_runs(func_dir: Path) -> list[Path]:
    """Return sorted list of BOLD NIfTIs in func_dir. Must be exactly N_RUNS.

    Files ending in _bolda.nii.gz or _boldb.nii.gz (extra acquisitions present
    in sub-001 and sub-002) are ignored.
    """
    candidates = sorted(
        f for f in func_dir.glob("*.nii.gz")
        if not (f.stem.endswith("_bolda") or f.stem.endswith("_boldb")
                or f.name.endswith("_bolda.nii.gz") or f.name.endswith("_boldb.nii.gz"))
    )
    if not candidates:
        candidates = sorted(
            f for f in func_dir.glob("*.nii")
            if not (f.stem.endswith("_bolda") or f.stem.endswith("_boldb"))
        )
    if len(candidates) != N_RUNS:
        raise ValueError(
            f"{func_dir}: expected {N_RUNS} BOLD files, found {len(candidates)}"
        )
    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    """Parse source, destination, onset, and optional subject arguments."""
    p = argparse.ArgumentParser(
        description="Create a BIDS dataset from raw NIfTI + MATLAB onset files."
    )
    p.add_argument("--input-dir",  required=True, type=Path,
                   help="Directory containing sub-001 … sub-016 folders")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Destination BIDS root (created if absent)")
    p.add_argument("--onset-dir",  required=True, type=Path,
                   help="Directory containing Sub##run#.mat onset files")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Subset of subject IDs to process, e.g. 001 002 "
                        "(default: 001..016)")
    return p.parse_args()


def write_json(path: Path, data: dict) -> None:
    """Write human-readable JSON with a trailing newline."""
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    """Create root metadata and copy each subject into BIDS layout."""
    args = parse_args()
    input_dir: Path  = args.input_dir
    output_dir: Path = args.output_dir
    onset_dir: Path  = args.onset_dir

    # Validate inputs
    if not input_dir.is_dir():
        sys.exit(f"ERROR: --input-dir does not exist: {input_dir}")
    if not onset_dir.is_dir():
        sys.exit(f"ERROR: --onset-dir does not exist: {onset_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Root-level BIDS files
    # -----------------------------------------------------------------------
    print("Writing root BIDS files …")
    write_json(output_dir / "dataset_description.json", DATASET_DESCRIPTION)
    write_json(output_dir / "participants.json",        PARTICIPANTS_COLUMNS)
    write_json(output_dir / "task-emotion_bold.json",   TASK_BOLD_JSON)
    (output_dir / "README").write_text(README_TEXT)
    (output_dir / ".bidsignore").write_text(BIDSIGNORE)

    # Determine subject list
    if args.subjects:
        sub_ids = [s.zfill(3) for s in args.subjects]
    else:
        sub_ids = [f"{i:03d}" for i in range(1, 17)]   # 001 … 016

    # Write participants.tsv
    participants_rows = ["participant_id\tsex\tage"]
    for sid in sub_ids:
        participants_rows.append(f"sub-{sid}\tn/a\tn/a")
    (output_dir / "participants.tsv").write_text(
        "\n".join(participants_rows) + "\n"
    )

    # -----------------------------------------------------------------------
    # Per-subject processing
    # -----------------------------------------------------------------------
    errors = []

    for sid in sub_ids:
        sub_int = int(sid)          # 1-based new index
        orig_id = orig_sub_id(sub_int)   # original MATLAB subject number

        print(f"\n[sub-{sid}] (original: Sub{orig_id:02d})")

        src_sub = input_dir / f"sub-{sid}"
        if not src_sub.is_dir():
            msg = f"  SKIP — source folder not found: {src_sub}"
            print(msg); errors.append(msg)
            continue

        dst_sub  = output_dir / f"sub-{sid}"
        dst_anat = dst_sub / "anat"
        dst_func = dst_sub / "func"
        dst_anat.mkdir(parents=True, exist_ok=True)
        dst_func.mkdir(parents=True, exist_ok=True)

        # --- Anatomical ---
        src_anat = src_sub / "anat"
        try:
            t1_src = find_t1w(src_anat)
            t1_dst = dst_anat / f"sub-{sid}_T1w.nii.gz"
            shutil.copy2(t1_src, t1_dst)
            write_json(dst_anat / f"sub-{sid}_T1w.json", T1W_JSON)
            print(f"  anat: {t1_src.name} → {t1_dst.name}")
        except (FileNotFoundError, NotADirectoryError) as e:
            msg = f"  ERROR (anat): {e}"
            print(msg); errors.append(msg)

        # --- Functional ---
        src_func = src_sub / "func"
        try:
            bold_files = find_bold_runs(src_func)
        except (ValueError, NotADirectoryError) as e:
            msg = f"  ERROR (func): {e}"
            print(msg); errors.append(msg)
            continue

        for run_idx, bold_src in enumerate(bold_files, start=1):
            run_label = f"{run_idx:02d}"
            bold_base = f"sub-{sid}_task-emotion_run-{run_label}_bold"

            # Copy BOLD
            bold_dst = dst_func / f"{bold_base}.nii.gz"
            shutil.copy2(bold_src, bold_dst)

            # Load onset file.
            # orig sub 1–11  (new sub-001..sub-009): no underscore → Sub01run1.mat
            # orig sub 12–20 (new sub-010..sub-016): underscore    → Sub14_run1.mat
            if orig_id >= 12:
                onset_name = f"Sub{orig_id:02d}_run{run_idx}.mat"
            else:
                onset_name = f"Sub{orig_id:02d}run{run_idx}.mat"
            onset_path = onset_dir / onset_name
            if not onset_path.exists():
                msg = f"  ERROR: onset file not found: {onset_path}"
                print(msg); errors.append(msg)
                continue

            events = load_events(onset_path)
            events_dst = dst_func / f"{bold_base[:-5]}_events.tsv"
            write_events_tsv(events, events_dst)

            print(f"  run-{run_label}: {bold_src.name} + {onset_name} → OK")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"BIDS dataset written to: {output_dir}")
    print(f"SliceTiming ({N_SLICES} values, ascending): "
          f"{_slice_timing[0]:.4f} … {_slice_timing[-1]:.4f} s")
    print(f"Stimulus duration: {STIMULUS_DURATION_S} s")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")
    else:
        print("\nAll subjects processed without errors.")

    print("\nNext steps:")
    print(f"  bids-validator {output_dir} --ignoreWarnings")
    print("  fmriprep <bids_dir> <output_dir> participant \\")
    print("    --participant-label 001 002 ... \\")
    print("    --ignore slicetiming  # remove if slice timing correction desired")


if __name__ == "__main__":
    main()
