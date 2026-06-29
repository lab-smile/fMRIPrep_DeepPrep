#!/usr/bin/env python3
"""
merge_results.py — Combine per-subject decoding outputs into the final .mat

Mirrors the SUMMARY STATS + PLOTS + save block of SingleTrialDecodingv3.m
(lines 130-179).  Run this after all SLURM array jobs have finished.

Expected per-subject files (written by decode_subject.py):
    {output_dir}/results_sub01.mat
    {output_dir}/results_sub02.mat
    …

Produces (matching MATLAB output exactly):
    {output_dir}/decoding_results_k5x100_v4.mat
    {output_dir}/PlNtROI_k5x100_v4.png
    {output_dir}/UpNtROI_k5x100_v4.png

Usage:
    python merge_results.py
    python merge_results.py --output-dir /path/to/results --num-subjects 20
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")                   # non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

# ── defaults matching SingleTrialDecodingv3.m CONFIG ──────────────────────────
OUTPUT_DIR   = (
    "/orange/ruogu.fang/pateld3/SPM_Preprocessed_fMRI_20Subjects"
    "/single_mvpa_results/"
)
NUM_SUBJECTS = 20

# Must match the ROI_NAMES list in decode_subject.py
ROI_NAMES = [
    "whole_brain",
    "V1v", "V1d",
    "V2v", "V2d",
    "V3v", "V3d",
    "hV4",
    "VO1", "VO2",
    "PHC1", "PHC2",
    "hMT",
    "LO1", "LO2",
    "V3a", "V3b",
    "IPS",
]


def load_per_subject_files(output_dir, num_subjects):
    """
    Load results_sub{N:02d}.mat for each subject and stack into arrays.

    Returns
    -------
    PlNt_acc  : [num_subjects, n_rois]  — NaN where subject file missing
    UpNt_acc  : [num_subjects, n_rois]
    PlNt_null : [num_subjects, n_rois, n_shuffles]
    UpNt_null : [num_subjects, n_rois, n_shuffles]
    k_folds, n_reps, n_shuffles : scalars (taken from first available file)
    """
    n_shuffles = None
    k_folds    = None
    n_reps     = None

    # First pass: discover n_shuffles and check if any file is missing whole_brain
    any_missing_whole_brain = False
    for subj in range(1, num_subjects + 1):
        path = os.path.join(output_dir, f"results_sub{subj:02d}.mat")
        if not os.path.isfile(path):
            continue
        d = sio.loadmat(path, squeeze_me=True)
        if n_shuffles is None:
            n_shuffles = int(d["n_shuffles"])
            k_folds    = int(d["k_folds"])
            n_reps     = int(d["n_reps"])
        saved_rois = list(d["roi_names"])
        if "whole_brain" not in saved_rois:
            any_missing_whole_brain = True

    if n_shuffles is None:
        raise FileNotFoundError(
            f"No results_sub*.mat files found in {output_dir}"
        )

    # If any subject is missing whole_brain, drop it from all to keep arrays aligned
    roi_names = list(ROI_NAMES)
    if any_missing_whole_brain and "whole_brain" in roi_names:
        print("  Note: at least one subject file is missing 'whole_brain' ROI — "
              "skipping whole_brain for all subjects.")
        roi_names.remove("whole_brain")

    n_rois = len(roi_names)

    # MATLAB: PlNt_acc  = NaN(num_subjects, numel(roi_names_of_interest))
    PlNt_acc  = np.full((num_subjects, n_rois),                np.nan)
    UpNt_acc  = np.full((num_subjects, n_rois),                np.nan)
    PlNt_null = np.full((num_subjects, n_rois, n_shuffles),    np.nan)
    UpNt_null = np.full((num_subjects, n_rois, n_shuffles),    np.nan)

    loaded = []
    for subj in range(1, num_subjects + 1):
        path = os.path.join(output_dir, f"results_sub{subj:02d}.mat")
        if not os.path.isfile(path):
            print(f"  Missing: results_sub{subj:02d}.mat — subject skipped (NaN)")
            continue

        d = sio.loadmat(path, squeeze_me=True)

        # Build column indices into this file's arrays that correspond to roi_names
        saved_rois = list(d["roi_names"])
        try:
            col_idx = [saved_rois.index(r) for r in roi_names]
        except ValueError as e:
            warnings.warn(
                f"Sub {subj}: expected ROI not found in file ({e}). "
                "Skipping subject."
            )
            continue

        # subj is 1-based; array index is 0-based  →  row = subj - 1
        row = subj - 1
        PlNt_acc[row]  = d["PlNt_acc"][col_idx]          # [n_rois]
        UpNt_acc[row]  = d["UpNt_acc"][col_idx]
        PlNt_null[row] = d["PlNt_null"][col_idx, :]      # [n_rois, n_shuffles]
        UpNt_null[row] = d["UpNt_null"][col_idx, :]
        loaded.append(subj)

    print(f"\nLoaded {len(loaded)}/{num_subjects} subjects: {loaded}")
    return PlNt_acc, UpNt_acc, PlNt_null, UpNt_null, k_folds, n_reps, n_shuffles, roi_names


def compute_summary_stats(PlNt_acc, UpNt_acc):
    """
    MATLAB lines 131-140:
        mean_PN_acc = nanmean(PlNt_acc);
        mean_UN_acc = nanmean(UpNt_acc);
        sem_PN_acc  = nanstd(PlNt_acc) ./ sqrt(sum(~isnan(PlNt_acc)));
        sem_UN_acc  = nanstd(UpNt_acc) ./ sqrt(sum(~isnan(UpNt_acc)));

    nanmean/nanstd operate along dim 1 (subjects) → output shape [n_rois].
    """
    # MATLAB nanmean(X) with no dim arg = mean along first non-singleton dim = rows
    mean_PN_acc = np.nanmean(PlNt_acc, axis=0)
    mean_UN_acc = np.nanmean(UpNt_acc, axis=0)

    n_valid_pn  = np.sum(~np.isnan(PlNt_acc), axis=0)
    n_valid_un  = np.sum(~np.isnan(UpNt_acc), axis=0)

    # MATLAB nanstd uses N-1 (ddof=1) normalisation by default
    sem_PN_acc = np.nanstd(PlNt_acc, axis=0, ddof=1) / np.sqrt(n_valid_pn)
    sem_UN_acc = np.nanstd(UpNt_acc, axis=0, ddof=1) / np.sqrt(n_valid_un)

    return mean_PN_acc, mean_UN_acc, sem_PN_acc, sem_UN_acc


def print_summary(roi_names, mean_PN, sem_PN, mean_UN, sem_UN):
    """
    Mirrors MATLAB lines 137-141:
        fprintf('%s - PvsN: %.3f ± %.3f | UvsN: %.3f ± %.3f\n', …)
    """
    print("\n── Decoding accuracy summary ──────────────────────────────────────")
    for r, name in enumerate(roi_names):
        print(
            f"  {name:<15s}  PvsN: {mean_PN[r]:.3f} ± {sem_PN[r]:.3f}"
            f"  |  UvsN: {mean_UN[r]:.3f} ± {sem_UN[r]:.3f}"
        )


def save_results(output_dir, tag, PlNt_acc, UpNt_acc, PlNt_null, UpNt_null,
                 mean_PN, sem_PN, mean_UN, sem_UN,
                 roi_names, k_folds, n_reps, n_shuffles):
    """
    MATLAB lines 143-145:
        save(fullfile(output_dir,'decoding_results_k5x100_v4.mat'), …)

    Variable names match MATLAB exactly so downstream scripts (step 5) are
    compatible without modification.
    """
    out_path = os.path.join(output_dir, f"decoding_results_{tag}.mat")
    sio.savemat(
        out_path,
        {
            # Core per-subject accuracy arrays — shape [num_subjects, n_rois]
            "PlNt_acc":              PlNt_acc,
            "UpNt_acc":              UpNt_acc,
            # Null distributions — shape [num_subjects, n_rois, n_shuffles]
            "PlNt_null":             PlNt_null,
            "UpNt_null":             UpNt_null,
            # Group summary statistics — shape [n_rois]
            "mean_PN_acc":           mean_PN,
            "sem_PN_acc":            sem_PN,
            "mean_UN_acc":           mean_UN,
            "sem_UN_acc":            sem_UN,
            # Metadata
            "roi_names_of_interest": np.array(roi_names, dtype=object),
            "k_folds":               float(k_folds),
            "n_reps":                float(n_reps),
            "n_shuffles":            float(n_shuffles),
        },
    )
    print(f"\nSaved → {out_path}")
    return out_path


def make_boxplot(acc_matrix, roi_names, title_str, ylabel, out_path):
    """
    Reproduces the MATLAB boxplot style (lines 152-179).

    MATLAB:
        boxplot(PlNt_acc, 'Labels', roi_names, 'Whisker', 1.5);
        title(…); xlabel('ROI'); ylabel('Decoding accuracy (%)');
        ylim([.45 .8]); grid on;
        colors = lines(numel(roi_names));
        patch each box with distinct color (FaceAlpha 0.8).
        savefig(…)

    matplotlib.boxplot with whis=1.5 matches MATLAB's Whisker=1.5.
    Box colours are taken from matplotlib's 'tab10' colormap, which is the
    closest equivalent to MATLAB's lines() colormap.
    """
    n_rois = len(roi_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_rois))

    fig, ax = plt.subplots(figsize=(14, 5))

    # MATLAB: boxplot(acc_matrix, 'Labels', roi_names, 'Whisker', 1.5)
    # acc_matrix shape [n_subjects, n_rois]; matplotlib expects each column
    # as one group — transpose so rows = observations per group.
    bp = ax.boxplot(
        acc_matrix,                     # shape [n_subjects, n_rois]; columns = groups
        labels=roi_names,
        whis=1.5,                       # MATLAB Whisker=1.5
        patch_artist=True,              # needed to fill boxes with color
        medianprops=dict(color="black", linewidth=1.5),
    )

    # Apply distinct colour per box (MATLAB: patch … colors(j,:) FaceAlpha 0.8)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_title(title_str, fontsize=13)
    ax.set_xlabel("ROI", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_ylim([0.45, 0.80])          # MATLAB: ylim([.45 .8])
    ax.grid(True)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    """Merge subject MAT files, print summaries, and create group plots."""
    parser = argparse.ArgumentParser(
        description="Merge per-subject decoding results (port of SingleTrialDecodingv3.m summary block)"
    )
    parser.add_argument("--output-dir",    default=OUTPUT_DIR)
    parser.add_argument("--num-subjects",  type=int, default=NUM_SUBJECTS)
    parser.add_argument("--tag",           default="k5x100_v4",
                        help="Suffix used in output filenames, e.g. k5x100_v4")
    args = parser.parse_args()

    output_dir   = args.output_dir
    num_subjects = args.num_subjects
    tag          = args.tag

    print(f"Merging results from {output_dir} …")

    # ── Load and assemble ────────────────────────────────────────────────────
    PlNt_acc, UpNt_acc, PlNt_null, UpNt_null, k, n_reps, n_shuffles, roi_names = \
        load_per_subject_files(output_dir, num_subjects)

    # ── Summary statistics ───────────────────────────────────────────────────
    mean_PN, mean_UN, sem_PN, sem_UN = compute_summary_stats(PlNt_acc, UpNt_acc)
    print_summary(roi_names, mean_PN, sem_PN, mean_UN, sem_UN)

    # ── Save .mat ────────────────────────────────────────────────────────────
    save_results(
        output_dir, tag,
        PlNt_acc, UpNt_acc, PlNt_null, UpNt_null,
        mean_PN, sem_PN, mean_UN, sem_UN,
        roi_names, k, n_reps, n_shuffles,
    )

    # ── Plots (mirrors MATLAB lines 147-179) ─────────────────────────────────
    # MATLAB: figure('Name','Pleasant vs Neutral'); boxplot(PlNt_acc, …)
    make_boxplot(
        PlNt_acc,
        roi_names,
        title_str=f"Pleasant vs Neutral ({k}×{n_reps})",
        ylabel="Decoding accuracy (%)",
        out_path=os.path.join(output_dir, f"PlNtROI_{tag}.png"),
    )

    # MATLAB: figure('Name','Unpleasant vs Neutral'); boxplot(UpNt_acc, …)
    make_boxplot(
        UpNt_acc,
        roi_names,
        title_str=f"Unpleasant vs Neutral ({k}×{n_reps})",
        ylabel="Decoding accuracy (%)",
        out_path=os.path.join(output_dir, f"UpNtROI_{tag}.png"),
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
