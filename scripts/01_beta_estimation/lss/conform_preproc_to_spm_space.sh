#!/bin/bash
# Conform DeepPrep/fMRIPrep MNI152NLin2009cAsym 2 mm preprocessed BOLD,
# brain masks, and run mean images to the original SPM/Wang-mask 3 mm grid
# before LSS beta estimation and SPM-space QC/FC.
#
# Usage:
#   LSS_PIPELINE=deepprep sbatch run_conform_spmspace.sbatch
#   bash conform_preproc_to_spm_space.sh deepprep 1 2 3
#
# Required on the cluster:
#   module load ants
#   python with templateflow installed, or set TEMPLATE_XFM explicitly.

set -euo pipefail

pipeline="${1:-${LSS_PIPELINE:-}}"
if [[ -z "${pipeline}" ]]; then
    echo "Usage: $0 {deepprep|fmriprep} [subject_index ...]" >&2
    echo "Or set LSS_PIPELINE={deepprep|fmriprep}." >&2
    exit 2
fi
if [[ "${pipeline}" != "deepprep" && "${pipeline}" != "fmriprep" ]]; then
    echo "Unknown pipeline: ${pipeline}" >&2
    exit 2
fi

DATA_ROOT="${DATA_ROOT:-/orange/ruogu.fang/pateld3/data}"
LSS_ROOT="${LSS_ROOT:-${DATA_ROOT}/LSS}"
SPM_SPACE_REF="${SPM_SPACE_REF:-${LSS_ROOT}/masks/maxprob_vol_lh_1.nii}"
OUT_ROOT="${OUT_ROOT:-${LSS_ROOT}/conformed_spmspace/${pipeline}}"
FORCE="${FORCE:-0}"
CONFORM_MASKS="${CONFORM_MASKS:-1}"
MAKE_RUN_MEANS="${MAKE_RUN_MEANS:-1}"
USE_DEEPPREP_T1W_MASK_XFM="${USE_DEEPPREP_T1W_MASK_XFM:-0}"

shopt -s nullglob

if [[ ! -f "${SPM_SPACE_REF}" ]]; then
    echo "SPM-space reference not found: ${SPM_SPACE_REF}" >&2
    echo "Set SPM_SPACE_REF to a 53x63x46 @3mm image on the native SPM/mask grid." >&2
    exit 1
fi

if [[ -z "${TEMPLATE_XFM:-}" ]]; then
    # TemplateFlow supplies the nonlinear bridge between the derivatives'
    # 2009c template and the MNI6 space underlying the legacy SPM grid.
    TEMPLATE_XFM="$(python -c "import templateflow.api as t; p=t.get('MNI152NLin6Asym', suffix='xfm', extension='.h5', mode='image', **{'from':'MNI152NLin2009cAsym'}); print(p[0] if isinstance(p, (list, tuple)) and len(p) else (p if p else ''))")"
fi
if [[ -z "${TEMPLATE_XFM}" ]]; then
    echo "TemplateFlow did not find a direct 2009c -> NLin6 image transform." >&2
    echo "Try setting TEMPLATE_XFM explicitly, or use the opposite transform with TEMPLATE_XFM_INVERT=1." >&2
    echo "Direct query:" >&2
    echo "  python -c \"import templateflow.api as t; print(t.get('MNI152NLin6Asym', suffix='xfm', extension='.h5', mode='image', **{'from':'MNI152NLin2009cAsym'}))\"" >&2
    exit 1
fi
if [[ ! -f "${TEMPLATE_XFM}" ]]; then
    echo "TemplateFlow transform not found: ${TEMPLATE_XFM}" >&2
    echo "Set TEMPLATE_XFM to tpl-MNI152NLin6Asym_from-MNI152NLin2009cAsym_*_xfm.h5." >&2
    exit 1
fi
if [[ ! -s "${TEMPLATE_XFM}" ]]; then
    echo "TemplateFlow transform is empty: ${TEMPLATE_XFM}" >&2
    echo "Remove/refetch this cache file or set TEMPLATE_XFM to a valid non-empty .h5 transform." >&2
    exit 1
fi
echo "Transform file : $(ls -lh "${TEMPLATE_XFM}")"
if command -v file >/dev/null 2>&1; then
    echo "Transform type : $(file "${TEMPLATE_XFM}")"
fi

transform_arg="${TEMPLATE_XFM}"
if [[ "${TEMPLATE_XFM_INVERT:-0}" == "1" ]]; then
    transform_arg="[${TEMPLATE_XFM},1]"
    echo "Transform mode : inverse"
else
    echo "Transform mode : forward"
fi

if [[ $# -gt 1 ]]; then
    subjects=("${@:2}")
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    subjects=("${SLURM_ARRAY_TASK_ID}")
else
    subjects=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
fi

echo "Pipeline      : ${pipeline}"
echo "Data root     : ${DATA_ROOT}"
echo "LSS root      : ${LSS_ROOT}"
echo "Reference     : ${SPM_SPACE_REF}"
echo "Transform     : ${TEMPLATE_XFM}"
echo "Output root   : ${OUT_ROOT}"
echo "Subjects      : ${subjects[*]}"
echo "Conform masks : ${CONFORM_MASKS}"
echo "Run means     : ${MAKE_RUN_MEANS}"
echo "DeepPrep T1w mask transform : ${USE_DEEPPREP_T1W_MASK_XFM}"

first_existing_file() {
    # Print the first usable candidate so callers can capture it directly.
    local path
    for path in "$@"; do
        if [[ -f "${path}" ]]; then
            printf '%s\n' "${path}"
            return 0
        fi
    done
    return 1
}

make_4d_mean() {
    # Prefer FSL on the cluster, with nibabel as a portable fallback.
    local in_img="$1"
    local out_img="$2"

    if [[ -s "${out_img}" && "${FORCE}" != "1" ]]; then
        echo "SKIP existing: ${out_img}"
        return 0
    fi

    if command -v fslmaths >/dev/null 2>&1; then
        fslmaths "${in_img}" -Tmean "${out_img}"
        return 0
    fi

    python -c "import nibabel as nib, numpy as np, sys; img=nib.load(sys.argv[1]); data=np.asanyarray(img.dataobj, dtype=np.float32); mean=data if data.ndim == 3 else np.nanmean(data, axis=3); nib.Nifti1Image(mean.astype(np.float32), img.affine, img.header).to_filename(sys.argv[2])" "${in_img}" "${out_img}"
}

binarize_mask() {
    # Label interpolation can leave fractional values at mask boundaries.
    local in_img="$1"
    local out_img="$2"

    if command -v ThresholdImage >/dev/null 2>&1; then
        ThresholdImage 3 "${in_img}" "${out_img}" 0.5 Inf 1 0
        return 0
    fi

    python -c "import nibabel as nib, numpy as np, sys; img=nib.load(sys.argv[1]); data=(np.asanyarray(img.dataobj) > 0.5).astype(np.uint8); nib.Nifti1Image(data, img.affine, img.header).to_filename(sys.argv[2])" "${in_img}" "${out_img}"
}

derive_mask_from_bold() {
    # Last-resort mask when a derivative did not provide a transformable mask.
    local in_img="$1"
    local out_img="$2"

    if [[ -s "${out_img}" && "${FORCE}" != "1" ]]; then
        echo "SKIP existing: ${out_img}"
        return 0
    fi

    python -c "import nibabel as nib, numpy as np, sys; img=nib.load(sys.argv[1]); data=np.asanyarray(img.dataobj); mask=np.isfinite(data).any(axis=3) & np.any(data != 0, axis=3) if data.ndim == 4 else (np.isfinite(data) & (data != 0)); nib.Nifti1Image(mask.astype(np.uint8), img.affine, img.header).to_filename(sys.argv[2])" "${in_img}" "${out_img}"
}

for subj_idx in "${subjects[@]}"; do
    sid="$(printf "%03d" "${subj_idx}")"

    if [[ "${pipeline}" == "deepprep" ]]; then
        in_func="${DATA_ROOT}/deepprep/BOLD/sub-${sid}/func"
    else
        in_func="${DATA_ROOT}/fmriprep/sub-${sid}/func"
    fi
    out_func="${OUT_ROOT}/sub-${sid}/func"
    mkdir -p "${out_func}"

    for run in 01 02 03 04 05; do
        in_bold="${in_func}/sub-${sid}_task-emotion_run-${run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
        out_bold="${out_func}/sub-${sid}_task-emotion_run-${run}_space-SPM3mm_desc-preproc_bold.nii.gz"
        out_mean="${out_func}/sub-${sid}_task-emotion_run-${run}_space-SPM3mm_desc-mean_bold.nii.gz"
        out_mask="${out_func}/sub-${sid}_task-emotion_run-${run}_space-SPM3mm_desc-brain_mask.nii.gz"

        if [[ ! -f "${in_bold}" ]]; then
            echo "Missing input: ${in_bold}" >&2
            exit 1
        fi

        if [[ -s "${out_bold}" && "${FORCE}" != "1" ]]; then
            echo "SKIP existing: ${out_bold}"
        else
            echo "Conforming BOLD ${pipeline} sub-${sid} run-${run}"
            antsApplyTransforms \
                -d 3 \
                -e 3 \
                -i "${in_bold}" \
                -r "${SPM_SPACE_REF}" \
                -o "${out_bold}" \
                -n Linear \
                -t "${transform_arg}"
        fi

        if [[ "${CONFORM_MASKS}" == "1" ]]; then
            # Derivative naming differs slightly by pipeline/version, so check
            # known filenames before trying a constrained wildcard.
            mask_candidates=(
                "${in_func}/sub-${sid}_task-emotion_run-${run}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
                "${in_func}/sub-${sid}_task-emotion_run-${run}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
            )
            in_mask="$(first_existing_file "${mask_candidates[@]}" || true)"
            if [[ -z "${in_mask}" ]]; then
                for candidate in "${in_func}/sub-${sid}_task-emotion_run-${run}"*_space-MNI152NLin2009cAsym*"_desc-brain_mask.nii.gz"; do
                    in_mask="${candidate}"
                    break
                done
            fi
            if [[ -n "${in_mask}" ]]; then
                if [[ -s "${out_mask}" && "${FORCE}" != "1" ]]; then
                    echo "SKIP existing: ${out_mask}"
                else
                    tmp_mask="${out_mask%.nii.gz}_float.nii.gz"
                    echo "Conforming mask ${pipeline} sub-${sid} run-${run}"
                    antsApplyTransforms \
                        -d 3 \
                        -i "${in_mask}" \
                        -r "${SPM_SPACE_REF}" \
                        -o "${tmp_mask}" \
                        -n GenericLabel \
                        -t "${transform_arg}"
                    binarize_mask "${tmp_mask}" "${out_mask}"
                    rm -f "${tmp_mask}"
                fi
                else
                    # DeepPrep may expose only a T1w-space mask. Applying the
                    # T1w-to-MNI transform before the template bridge preserves
                    # a true anatomical mask when explicitly enabled.
                    t1w_mask="${in_func}/sub-${sid}_task-emotion_run-${run}_space-T1w_desc-brain_mask.nii.gz"
                t1w_to_mni="${DATA_ROOT}/deepprep/BOLD/sub-${sid}/anat/sub-${sid}_from-T1w_to-MNI152NLin2009cAsym_desc-joint_trans.nii.gz"
                if [[ "${pipeline}" == "deepprep" && "${USE_DEEPPREP_T1W_MASK_XFM}" == "1" && -f "${t1w_mask}" && -f "${t1w_to_mni}" ]]; then
                    if [[ -s "${out_mask}" && "${FORCE}" != "1" ]]; then
                        echo "SKIP existing: ${out_mask}"
                    else
                        tmp_mask="${out_mask%.nii.gz}_float.nii.gz"
                        echo "Conforming T1w mask via T1w->MNI->SPM ${pipeline} sub-${sid} run-${run}"
                        antsApplyTransforms \
                            -d 3 \
                            -i "${t1w_mask}" \
                            -r "${SPM_SPACE_REF}" \
                            -o "${tmp_mask}" \
                            -n GenericLabel \
                            -t "${transform_arg}" \
                            -t "${t1w_to_mni}"
                        binarize_mask "${tmp_mask}" "${out_mask}"
                        rm -f "${tmp_mask}"
                    fi
                else
                    if [[ "${pipeline}" == "deepprep" && -f "${t1w_mask}" && "${USE_DEEPPREP_T1W_MASK_XFM}" != "1" ]]; then
                        echo "Found T1w mask but USE_DEEPPREP_T1W_MASK_XFM=0, so deriving mask from conformed BOLD: ${t1w_mask}" >&2
                    elif [[ -f "${t1w_mask}" ]]; then
                        echo "Found T1w mask but no usable T1w->MNI transform for ${pipeline} sub-${sid}: ${t1w_mask}" >&2
                    fi
                    echo "Missing source brain mask for ${pipeline} sub-${sid} run-${run}; deriving SPM-space mask from conformed BOLD." >&2
                    derive_mask_from_bold "${out_bold}" "${out_mask}"
                fi
            fi
        fi

        if [[ "${MAKE_RUN_MEANS}" == "1" ]]; then
            echo "Writing run mean ${pipeline} sub-${sid} run-${run}"
            make_4d_mean "${out_bold}" "${out_mean}"
        fi
    done
done

echo "Done: ${pipeline} conformed BOLD/masks/run means written under ${OUT_ROOT}"
