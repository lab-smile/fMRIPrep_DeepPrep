#!/usr/bin/env python3
"""
decode_subject.py — Step 4: SVM decoding for one subject.

Faithful Python port of SingleTrialDecodingv3.m.
Every MATLAB→Python mapping is annotated inline.

Key equivalences (verified against MATLAB source):
  MATLAB fitcsvm(…,'KernelFunction','linear','Standardize',true)
      ≡  StandardScaler (fit on train) + LinearSVC(C=1.0)

  MATLAB cvpartition(labels,'KFold',k)          [stratified by default]
      ≡  StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

  MATLAB randperm(n)
      ≡  np.random.RandomState(seed).permutation(n)

  MATLAB nanmean / nanstd
      ≡  np.nanmean / np.nanstd

  MATLAB roi_masks.(name)(:) > 0               [Fortran-order ravel]
      ≡  masks[name].ravel(order='F').astype(bool)   (scipy.io.loadmat
         preserves MATLAB column-major layout, so order='F' matches (:))

  MATLAB load(file).Pl  →  shape [nVoxels × nTrials], dtype single
      ≡  scipy.io.loadmat(file)['Pl']           shape (nVoxels, nTrials)

NaN-filter correction (MATLAB lines 191-192 silent bug):
  MATLAB:  Xtr = Xtr(mask,:);  ytr = ytr(mask_on_NEW_Xtr);
  The second mask is computed on the already-filtered Xtr, so it is all-True
  and ytr is never actually filtered.  This means if any training row had NaN,
  Xtr and ytr would have mismatched lengths → MATLAB error.  Since the script
  runs without error, this case never fires in practice (valid_voxels already
  removes the worst offenders).  We implement the CORRECT intent: compute the
  mask once on the original Xtr, then apply it to both Xtr and ytr.

MATLAB isempty(model.SupportVectors) guard:
  LinearSVC has no support-vector concept.  The MATLAB guard catches degenerate
  fits (all predictions same class).  We replicate with a unique-prediction
  check after clf.predict().

Usage:
    python decode_subject.py --subject 3 --n-jobs 8
    python decode_subject.py --subject 3 --n-jobs 8 --k-folds 10  # paper value
"""

import argparse
import os
import sys
import warnings

import h5py
import numpy as np
import scipy.io as sio
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ─────────────────────────────── CONFIG ──────────────────────────────────────
# Mirrors the MATLAB CONFIG block.  Override via CLI args.

DATA_DIR = (
    "/orange/ruogu.fang/pateld3/data/deepprep-betas-extracted"
)
ROI_MASKS_FILE = (
    "/orange/ruogu.fang/pateld3/data/output_mats/roi_masks.mat"
)
OUTPUT_DIR = (
    "/orange/ruogu.fang/pateld3/data/deepprep-mvpa-results-py-9/"
)

# MATLAB: k_folds = 5 / n_reps = 100 / n_shuffles = 100
K_FOLDS = 5
N_REPS = 100
N_SHUFFLES = 100

# MATLAB: roi_names_of_interest
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

# MATLAB: ips_parts = {'IPS0','IPS1','IPS2','IPS3','IPS4','IPS5'}
IPS_PARTS = ["IPS0", "IPS1", "IPS2", "IPS3", "IPS4", "IPS5"]


# ─────────────────────────────── ROI MASK LOADING ───────────────────────────

def load_roi_masks(roi_masks_file):
    """
    Load roi_masks.mat and combine IPS sub-ROIs.

    Mirrors MATLAB CONFIG lines 13-24:
        roi_masks = load(roi_masks_file);
        ips_parts = {'IPS0',...,'IPS5'};
        have_all  = all(isfield(roi_masks, ips_parts));
        if have_all
            m = false(size(...));
            for p: m = m | (roi_masks.(ips_parts{p}) > 0); end
            roi_masks.IPS = m;
        end

    scipy.io.loadmat returns a dict; MATLAB struct fields become dict keys.
    The save was done with  save(file, '-struct', 'roi_masks')  so each ROI
    is a top-level key in the .mat file.

    Masks are 3D logical arrays stored in MATLAB column-major order.
    We keep them 3D here; raveling to 1D is done at usage time so the ravel
    order (Fortran, matching MATLAB's (:)) is applied consistently.
    """
    # squeeze_me=False: preserve the exact MATLAB array layout (no singleton dims removed).
    # squeeze_me=True would silently reshape e.g. [97,115,1] → [97,115], misaligning the
    # subsequent ravel(order='F') with MATLAB's (:) Fortran-order flattening.
    raw = sio.loadmat(roi_masks_file, squeeze_me=False)

    masks = {}
    for key, val in raw.items():
        if key.startswith("_"):          # skip __header__, __version__, etc.
            continue
        masks[key] = np.asarray(val)     # keep original 3D shape and dtype

    # ── Combine IPS0–IPS5 → IPS ──────────────────────────────────────────────
    # MATLAB lines 17-24
    have_all_ips = all(p in masks for p in IPS_PARTS)
    if have_all_ips:
        ref = masks[IPS_PARTS[0]]
        combined = np.zeros(ref.shape, dtype=bool)
        for p in IPS_PARTS:
            # MATLAB: m = m | (roi_masks.(ips_parts{p}) > 0)
            combined |= masks[p] > 0
        masks["IPS"] = combined
        print("  IPS0–IPS5 combined into IPS mask.")
    else:
        present = [p for p in IPS_PARTS if p in masks]
        if present:
            print(f"  WARNING: only found {present} of IPS sub-parts; IPS not combined.")

    return masks


def get_roi_mask_flat(masks, roi_name, n_voxels_expected):
    """
    Return a flat boolean mask of length n_voxels_expected.

    MATLAB lines 52-54:
        roi_mask = roi_masks.(roi_name)(:) > 0;
        % (:) ravels in Fortran (column-major) order

    scipy.io.loadmat preserves MATLAB's column-major memory layout, so
    ravel(order='F') reproduces MATLAB's (:) indexing exactly.
    """
    arr = masks[roi_name]
    flat = arr.ravel(order="F").astype(bool)       # MATLAB: (:) > 0
    if flat.shape[0] != n_voxels_expected:
        # MATLAB lines 73-77: size mismatch check (tries reshape first)
        raise ValueError(
            f"ROI mask size {flat.shape[0]} != nVoxels {n_voxels_expected}"
        )
    return flat


# ─────────────────────────────── BETA LOADING ───────────────────────────────

def _load_mat_variable(path, var_name):
    """
    Load a single variable from a .mat file, supporting both legacy and v7.3 formats.

    scipy.io.loadmat handles MATLAB ≤ v7.2 (default save format).
    MATLAB v7.3 files are HDF5; scipy raises NotImplementedError for these —
    fall back to h5py.

    CRITICAL dimension note for v7.3 / h5py:
        MATLAB stores arrays in column-major (Fortran) order.
        HDF5 uses row-major (C) order and reverses the dimension indices.
        A MATLAB array of shape [nVoxels, nTrials] is stored in HDF5 as
        [nTrials, nVoxels].  h5py reads it in that transposed shape, so we
        must apply .T to recover the original [nVoxels, nTrials] layout that
        matches scipy.io.loadmat's output and the MATLAB source.
    """
    try:
        data = sio.loadmat(path, squeeze_me=False)[var_name]
    except NotImplementedError:
        # v7.3 HDF5 format
        with h5py.File(path, "r") as f:
            data = f[var_name][:]   # shape [nTrials, nVoxels] due to HDF5 dim reversal
            data = data.T           # → [nVoxels, nTrials], matching MATLAB layout
    return data.astype(np.float64)


def load_betas(data_dir, subj):
    """
    Load Pl{subj}.mat, Nt{subj}.mat, Up{subj}.mat.

    MATLAB lines 58-69:
        pl_file = fullfile(data_dir, sprintf('Pl%d.mat', subj));
        …
        pl_trials = load(pl_file); pl_trials = pl_trials.Pl;
        nt_trials = load(nt_file); nt_trials = nt_trials.Nt;
        up_trials = load(up_file); up_trials = up_trials.Up;

    Each .mat contains one variable (Pl / Nt / Up), shape [nVoxels × nTrials],
    dtype single (float32).  Loaded as float64 for sklearn compatibility
    (MATLAB's fitcsvm also converts single→double internally).
    """
    pl_path = os.path.join(data_dir, f"Pl{subj}.mat")
    nt_path = os.path.join(data_dir, f"Nt{subj}.mat")
    up_path = os.path.join(data_dir, f"Up{subj}.mat")

    for p in [pl_path, nt_path, up_path]:
        if not os.path.isfile(p):
            return None, None, None

    pl = _load_mat_variable(pl_path, "Pl")
    nt = _load_mat_variable(nt_path, "Nt")
    up = _load_mat_variable(up_path, "Up")

    return pl, nt, up


# ─────────────────────────────── SVM HELPER ─────────────────────────────────

def _run_one_rep(X, y, k, seed):
    """
    One repetition of stratified k-fold SVM decoding.

    Mirrors the inner loop of MATLAB svm_kfold_repeated() (lines 184-200):

        cv = cvpartition(labels, 'KFold', k);    % new random stratified split
        for f = 1:k
            [Xtr/ytr, Xte/yte] = split by fold f
            Xtr = Xtr(all(~isnan(Xtr),2),:);  ytr = ytr(all(~isnan(Xtr),2));
            Xte = Xte(all(~isnan(Xte),2),:);  yte = yte(all(~isnan(Xte),2));
            if numel(unique(ytr))<2 || isempty(Xte), continue; end
            model = fitcsvm(Xtr,ytr,'KernelFunction','linear','Standardize',true);
            if isempty(model.SupportVectors), continue; end
            yhat = predict(model, Xte);
            fold_acc(f) = mean(yhat == yte);
        end
        acc_all(rep) = mean(fold_acc, 'omitnan');

    NaN-filter note: see module docstring for bug correction explanation.
    """
    rng = np.random.RandomState(seed)
    skf = StratifiedKFold(
        n_splits=k,
        shuffle=True,
        random_state=int(rng.randint(0, 2**31 - 1)),
    )

    fold_accs = []
    for train_idx, test_idx in skf.split(X, y):
        Xtr, ytr = X[train_idx], y[train_idx]
        Xte, yte = X[test_idx],  y[test_idx]

        # ── NaN-row removal ──────────────────────────────────────────────────
        # MATLAB lines 191-192 (corrected — see module docstring):
        #   mask computed BEFORE modifying Xtr/Xte so ytr/yte are filtered
        #   consistently.  In practice identical to MATLAB since NaN rows are
        #   already absent after the valid_voxels filter upstream.
        tr_valid = ~np.any(np.isnan(Xtr), axis=1)
        te_valid = ~np.any(np.isnan(Xte), axis=1)
        Xtr, ytr = Xtr[tr_valid], ytr[tr_valid]
        Xte, yte = Xte[te_valid], yte[te_valid]

        # MATLAB: if numel(unique(ytr)) < 2 || isempty(Xte), continue; end
        if len(np.unique(ytr)) < 2 or len(Xte) == 0:
            continue

        # ── Feature standardisation ──────────────────────────────────────────
        # MATLAB fitcsvm(…,'Standardize',true):
        #   fits mean/std on training data; std=0 features → 0 (no /0).
        # StandardScaler does the same; zero-variance columns become 0.
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        # ── Linear SVM ───────────────────────────────────────────────────────
        # MATLAB: fitcsvm(…,'KernelFunction','linear')
        #   default BoxConstraint (C) = 1.0
        # LinearSVC: C=1.0 (default), liblinear solver, same decision boundary.
        # dual=True preferred when n_features > n_samples (high-dim ROI data).
        #
        # MATLAB guard: if isempty(model.SupportVectors), continue; end
        #   → catches degenerate fits where the model predicts only one class.
        #   We replicate: skip fold if all predictions are identical.
        clf = LinearSVC(C=1.0, max_iter=5000, dual=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")    # suppress ConvergenceWarning
            clf.fit(Xtr, ytr)

        yhat = clf.predict(Xte)

        # Guard equivalent to MATLAB isempty(model.SupportVectors)
        if len(np.unique(yhat)) < 2:
            continue

        fold_accs.append(np.mean(yhat == yte))

    # MATLAB: acc_all(rep) = mean(fold_acc, 'omitnan')
    return np.nanmean(fold_accs) if fold_accs else np.nan


def svm_kfold_repeated(X, y, k, n_reps, n_jobs=1, base_seed=0):
    """
    Repeated stratified k-fold SVM decoding.

    Mirrors MATLAB function svm_kfold_repeated(data, labels, k, n_reps).

    MATLAB (lines 182-202):
        for rep = 1:n_reps
            … (see _run_one_rep)
        end
        macc = mean(acc_all, 'omitnan');

    n_jobs > 1 parallelises reps via joblib (loky backend).
    Results are numerically identical to serial (n_jobs=1) because each rep
    uses a distinct, independent random seed.
    """
    seeds = [base_seed + r for r in range(n_reps)]
    rep_accs = Parallel(n_jobs=n_jobs)(
        delayed(_run_one_rep)(X, y, k, s) for s in seeds
    )
    # MATLAB: macc = mean(acc_all, 'omitnan')
    return float(np.nanmean(rep_accs))


# ─────────────────────────────── NULL-SHUFFLE WORKER ────────────────────────

def _run_null_shuffle(
    sh_idx,
    PlNt_X, PlNt_y,
    UpNt_X, UpNt_y,
    k, n_reps,
    roi_idx, subj,
):
    """
    One shuffle of the null distribution.

    Mirrors MATLAB lines 115-124:
        pn_labels_shuf = PlNt_labels(randperm(length(PlNt_labels)));
        un_labels_shuf = UpNt_labels(randperm(length(UpNt_labels)));
        if numel(unique(pn_labels_shuf)) > 1
            pn_null(sh) = svm_kfold_repeated(PlNt_data, pn_labels_shuf, …);
        end
        if numel(unique(un_labels_shuf)) > 1
            un_null(sh) = svm_kfold_repeated(UpNt_data, un_labels_shuf, …);
        end

    Module-level (not a closure) so joblib can pickle it for multiprocessing.
    Seed encodes (roi_idx, subj, sh_idx) to guarantee independent randomness.
    """
    # Unique seed per (roi, subject, shuffle) — no collisions across jobs
    seed_base = roi_idx * 10_000_000 + subj * 100_000 + sh_idx * 100

    rng = np.random.RandomState(seed_base)

    # MATLAB: randperm(length(PlNt_labels))  → permuted index vector
    pn_shuf = rng.permutation(PlNt_y)        # label shuffle (not index shuffle)
    un_shuf = rng.permutation(UpNt_y)        # use same rng to match MATLAB style

    # MATLAB: pn_null = zeros(1, n_shuffles) — uncomputed shuffles default to 0.0, not NaN.
    # Using np.nan here would cause step 5 to silently exclude those entries from the
    # null-distribution percentile, diverging from MATLAB's behaviour.
    pn_val = 0.0   # MATLAB: zeros(1, n_shuffles) default
    un_val = 0.0   # MATLAB: zeros(1, n_shuffles) default

    # MATLAB: if numel(unique(pn_labels_shuf)) > 1
    if len(np.unique(pn_shuf)) > 1:
        pn_val = svm_kfold_repeated(
            PlNt_X, pn_shuf, k, n_reps,
            n_jobs=1,
            base_seed=seed_base + 1,
        )
    if len(np.unique(un_shuf)) > 1:
        un_val = svm_kfold_repeated(
            UpNt_X, un_shuf, k, n_reps,
            n_jobs=1,
            base_seed=seed_base + 50_001,
        )

    return pn_val, un_val


# ─────────────────────────────── MAIN ───────────────────────────────────────

def main():
    """Decode both affective contrasts for one subject and save a MAT file."""
    parser = argparse.ArgumentParser(
        description="Step 4 SVM decoding — one subject (port of SingleTrialDecodingv3.m)"
    )
    parser.add_argument("--subject",        type=int,   required=True,
                        help="1-based subject index (matches Pl{N}.mat naming)")
    parser.add_argument("--data-dir",       default=DATA_DIR)
    parser.add_argument("--roi-masks-file", default=ROI_MASKS_FILE)
    parser.add_argument("--output-dir",     default=OUTPUT_DIR)
    parser.add_argument("--k-folds",        type=int,   default=K_FOLDS,
                        help=f"CV folds (MATLAB default {K_FOLDS}; paper uses 10)")
    parser.add_argument("--n-reps",         type=int,   default=N_REPS)
    parser.add_argument("--n-shuffles",     type=int,   default=N_SHUFFLES)
    parser.add_argument("--n-jobs",         type=int,   default=1,
                        help="CPUs for parallel null-shuffle computation (outer loop)")
    parser.add_argument("--skip-existing",  action="store_true",
                        help="Skip subject if output file already exists")
    args = parser.parse_args()

    subj       = args.subject
    k          = args.k_folds
    n_reps     = args.n_reps
    n_shuffles = args.n_shuffles
    n_jobs     = args.n_jobs

    os.makedirs(args.output_dir, exist_ok=True)

    out_path = os.path.join(args.output_dir, f"results_sub{subj:02d}.mat")
    if args.skip_existing and os.path.isfile(out_path):
        print(f"Output already exists, skipping: {out_path}")
        sys.exit(0)

    print(
        f"=== Subject {subj} | k={k} folds | {n_reps} reps "
        f"| {n_shuffles} shuffles | {n_jobs} CPUs ==="
    )

    # ── Load ROI masks (MATLAB CONFIG lines 13-24) ───────────────────────────
    print("Loading ROI masks …")
    roi_masks = load_roi_masks(args.roi_masks_file)

    n_rois = len(ROI_NAMES)

    # Result arrays — one entry per ROI (one subject).
    # Shape mirrors the per-subject slice of the MATLAB arrays:
    #   MATLAB PlNt_acc[subj, roi_idx]        →  PlNt_acc[roi_idx]
    #   MATLAB PlNt_null[subj, roi_idx, sh]   →  PlNt_null[roi_idx, sh]
    PlNt_acc  = np.full(n_rois,                np.nan)   # [n_rois]
    UpNt_acc  = np.full(n_rois,                np.nan)
    PlNt_null = np.full((n_rois, n_shuffles),  np.nan)   # [n_rois, n_shuffles]
    UpNt_null = np.full((n_rois, n_shuffles),  np.nan)

    # ── Load beta files (MATLAB lines 58-69) ────────────────────────────────
    print(f"Loading beta files for subject {subj} …")
    pl, nt, up = load_betas(args.data_dir, subj)

    # MATLAB lines 62-65: if ~isfile(…), warning(…); continue; end
    if pl is None:
        print(f"  WARNING: Missing beta .mat files for subject {subj}. Exiting.")
        sys.exit(0)

    print(f"  Beta shapes — Pl: {pl.shape}, Nt: {nt.shape}, Up: {up.shape}")
    n_voxels_total = pl.shape[0]

    # ── ROI loop (MATLAB line 41: for roi_idx = 1:numel(roi_names_of_interest)) ──
    for roi_idx, roi_name in enumerate(ROI_NAMES):
        print(f"\n─── ROI [{roi_idx+1}/{n_rois}]: {roi_name} ───")

        use_whole_brain = roi_name.lower() == "whole_brain"

        # ── ROI mask (MATLAB lines 46-55) ────────────────────────────────────
        if use_whole_brain:
            # MATLAB: roi_mask = []  (no masking; use all voxels)
            pl_roi = pl.copy()
            nt_roi = nt.copy()
            up_roi = up.copy()
        else:
            if roi_name not in roi_masks:
                # MATLAB: warning('ROI "%s" not found. Skipping...', roi_name)
                print(f"  WARNING: ROI '{roi_name}' not found in masks. Skipping.")
                continue

            try:
                roi_mask = get_roi_mask_flat(roi_masks, roi_name, n_voxels_total)
            except ValueError as e:
                # MATLAB lines 73-77: size mismatch warning → skip
                print(f"  WARNING: {e}. Skipping.")
                continue

            # MATLAB lines 79-81:
            #   pl_trials = pl_trials(roi_mask,:);
            #   nt_trials = nt_trials(roi_mask,:);
            #   up_trials = up_trials(roi_mask,:);
            pl_roi = pl[roi_mask, :]
            nt_roi = nt[roi_mask, :]
            up_roi = up[roi_mask, :]

        # ── Remove voxels where ALL conditions are fully NaN ─────────────────
        # MATLAB lines 84-87:
        #   valid_voxels = ~(all(isnan(pl),2) & all(isnan(nt),2) & all(isnan(up),2));
        #   pl_trials = pl_trials(valid_voxels,:);  etc.
        #
        # axis=1: "across columns" = across trials (shape is [nVoxels, nTrials])
        all_nan_pl = np.all(np.isnan(pl_roi), axis=1)
        all_nan_nt = np.all(np.isnan(nt_roi), axis=1)
        all_nan_up = np.all(np.isnan(up_roi), axis=1)
        valid_voxels = ~(all_nan_pl & all_nan_nt & all_nan_up)

        n_removed = int(np.sum(~valid_voxels))
        n_kept    = int(np.sum(valid_voxels))
        # MATLAB line 89-91: fprintf log
        print(
            f"  Subj {subj} ({roi_name}): removed {n_removed} empty voxels, "
            f"kept {n_kept}"
        )

        pl_v = pl_roi[valid_voxels, :]
        nt_v = nt_roi[valid_voxels, :]
        up_v = up_roi[valid_voxels, :]

        # MATLAB lines 92-95: if any([isempty(pl_trials), …])
        if pl_v.size == 0 or nt_v.size == 0 or up_v.size == 0:
            print(f"  WARNING: No valid voxels for subj {subj}, {roi_name}. Skipping.")
            continue

        # ── Build classification datasets ─────────────────────────────────────
        # MATLAB lines 98-103:
        #   PlNt_data   = [pl_trials, nt_trials]'   → transpose: [n_trials × n_voxels]
        #   PlNt_labels = [ones(nPl,1); -ones(nNt,1)]
        #
        # np.hstack joins along axis=1 (trials axis), then .T → [n_trials, n_voxels]
        PlNt_X = np.hstack([pl_v, nt_v]).T                    # [n_PlNt_trials, n_voxels]
        PlNt_y = np.concatenate([
            np.ones(pl_v.shape[1]),                           # +1 for Pleasant
            -np.ones(nt_v.shape[1]),                          # -1 for Neutral
        ])

        UpNt_X = np.hstack([up_v, nt_v]).T                    # [n_UpNt_trials, n_voxels]
        UpNt_y = np.concatenate([
            np.ones(up_v.shape[1]),                           # +1 for Unpleasant
            -np.ones(nt_v.shape[1]),                          # -1 for Neutral
        ])

        # ── Real decoding accuracy (MATLAB lines 105-110) ────────────────────
        # MATLAB: if numel(unique(PlNt_labels)) > 1
        if len(np.unique(PlNt_y)) > 1:
            acc = svm_kfold_repeated(
                PlNt_X, PlNt_y, k, n_reps,
                n_jobs=1,
                base_seed=subj * 1_000_000 + roi_idx * 1_000,
            )
            PlNt_acc[roi_idx] = acc
            print(f"  PlNt accuracy = {acc:.4f}")

        if len(np.unique(UpNt_y)) > 1:
            acc = svm_kfold_repeated(
                UpNt_X, UpNt_y, k, n_reps,
                n_jobs=1,
                base_seed=subj * 1_000_000 + roi_idx * 1_000 + 500,
            )
            UpNt_acc[roi_idx] = acc
            print(f"  UpNt accuracy = {acc:.4f}")

        # ── Null distribution (MATLAB lines 112-126) ─────────────────────────
        # MATLAB:
        #   pn_null = zeros(1, n_shuffles);
        #   un_null = zeros(1, n_shuffles);
        #   for sh = 1:n_shuffles
        #       pn_labels_shuf = PlNt_labels(randperm(…));
        #       un_labels_shuf = UpNt_labels(randperm(…));
        #       pn_null(sh) = svm_kfold_repeated(PlNt_data, pn_labels_shuf, …);
        #       un_null(sh) = svm_kfold_repeated(UpNt_data, un_labels_shuf, …);
        #   end
        #   PlNt_null(subj, roi_idx, :) = pn_null;
        #   UpNt_null(subj, roi_idx, :) = un_null;
        #
        # Outer shuffle loop parallelised — identical results to serial because
        # each shuffle uses a deterministic, independent seed.
        print(
            f"  Null distribution: {n_shuffles} shuffles "
            f"({n_jobs} CPUs) …"
        )
        null_results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_run_null_shuffle)(
                sh,
                PlNt_X, PlNt_y,
                UpNt_X, UpNt_y,
                k, n_reps,
                roi_idx, subj,
            )
            for sh in range(n_shuffles)
        )

        for sh, (pn_val, un_val) in enumerate(null_results):
            # MATLAB: PlNt_null(subj, roi_idx, sh) = pn_null(sh)
            PlNt_null[roi_idx, sh] = pn_val
            UpNt_null[roi_idx, sh] = un_val

    # ── Save per-subject results ──────────────────────────────────────────────
    # merge_results.py assembles these into the full [n_subjects × n_rois] arrays
    # matching the MATLAB output layout.
    sio.savemat(
        out_path,
        {
            "PlNt_acc":   PlNt_acc,      # [n_rois]            — matches PlNt_acc[subj,:]
            "UpNt_acc":   UpNt_acc,      # [n_rois]
            "PlNt_null":  PlNt_null,     # [n_rois, n_shuffles] — matches PlNt_null[subj,:,:]
            "UpNt_null":  UpNt_null,     # [n_rois, n_shuffles]
            "roi_names":  np.array(ROI_NAMES, dtype=object),
            "subject":    float(subj),
            "k_folds":    float(k),
            "n_reps":     float(n_reps),
            "n_shuffles": float(n_shuffles),
        },
    )
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
