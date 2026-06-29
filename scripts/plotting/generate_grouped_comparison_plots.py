"""Generate grouped ROI boxplots from the final collected result sources.

Run from the repository root:
    python scripts/plotting/generate_grouped_comparison_plots.py
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-smile-lab")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "results"
LSS_DIR = RESULTS_ROOT / "01_our_deepprep_fmriprep_lss_spmspace"
MAX_DIR = RESULTS_ROOT / "02_max_spm_results"
OUR_SPM_DIR = RESULTS_ROOT / "03_our_spm_results"
PAPER_DIR = RESULTS_ROOT / "04_papers_results"
OUTPUT_DIR = RESULTS_ROOT / "plots"

ROI_NAMES = [
    "V1v",
    "V1d",
    "V2v",
    "V2d",
    "V3v",
    "V3d",
    "hV4",
    "VO1",
    "VO2",
    "PHC1",
    "PHC2",
    "hMT",
    "LO1",
    "LO2",
    "V3a",
    "V3b",
    "IPS",
]
CONTRASTS = [
    ("pleasant_vs_neutral", "Pleasant vs Neutral", "pleasant_vs_neutral"),
    ("unpleasant_vs_neutral", "Unpleasant vs Neutral", "unpleasant_vs_neutral"),
]
SUB16_ORIGINAL_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19, 20)

COLORS = {
    "paper": "#8E8E93",
    "max_spm": "#F2994A",
    "our_spm": "#27AE60",
    "deepprep": "#2F80ED",
    "fmriprep": "#9B51E0",
}
YLIM = (34, 85)
YTICKS = np.arange(35, 86, 5)
CHANCE = 50
THRESHOLD = 54
BOX_WIDTH = 0.22
BOX_OFFSET = 0.25

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
    }
)


@dataclass(frozen=True)
class Source:
    """Describe one result source included in a grouped comparison."""

    label: str
    color_key: str
    path: Path | None = None
    subject_ids: tuple[int, ...] | None = None
    is_paper: bool = False


@dataclass(frozen=True)
class Comparison:
    """Define the three sources and output naming for one comparison figure."""

    number: int
    stem: str
    title: str
    sources: tuple[Source, Source, Source]


def sorted_subjects(results: dict) -> list[str]:
    """Return MATLAB subject fields in numeric rather than lexical order."""
    return sorted(results, key=lambda name: int(re.sub(r"\D", "", name)))


def load_subject_matrix(source: Source, contrast: str) -> np.ndarray:
    """Load one source as a subject-by-ROI matrix expressed in percent."""
    if source.path is None:
        raise ValueError(f"No data path configured for {source.label}")

    mat = sio.loadmat(source.path, simplify_cells=True)
    results = mat["results"]
    subjects = sorted_subjects(results)
    if source.subject_ids is not None:
        keep = {f"Subj{subject_id}" for subject_id in source.subject_ids}
        subjects = [subject for subject in subjects if subject in keep]

    matrix = np.full((len(subjects), len(ROI_NAMES)), np.nan, dtype=float)
    for subject_index, subject in enumerate(subjects):
        subject_rois = results[subject]
        for roi_index, roi in enumerate(ROI_NAMES):
            if roi not in subject_rois:
                continue
            value = subject_rois[roi].get(contrast)
            if value is None:
                continue
            value = float(value)
            # Result files mix proportions and percentages; values near one are
            # proportions, while published/legacy values are already percent.
            matrix[subject_index, roi_index] = value * 100.0 if value <= 1.5 else value

    if not np.isfinite(matrix).any():
        raise ValueError(f"No finite {contrast} values in {source.path}")
    return matrix


def paper_path(contrast: str) -> Path:
    """Return the digitized-paper estimate file for a contrast."""
    return PAPER_DIR / f"Bo2021_Figure3B_{contrast}_estimates.mat"


def load_paper_estimates(contrast: str) -> np.ndarray:
    """Load published five-number summaries in the canonical ROI order."""
    mat = sio.loadmat(paper_path(contrast), simplify_cells=True)
    file_rois = np.atleast_1d(mat["roi_names"]).tolist()
    estimates = np.asarray(mat["estimates_percent"], dtype=float)
    index = {roi: i for i, roi in enumerate(file_rois)}
    return np.vstack([estimates[index[roi], :] for roi in ROI_NAMES])


def subject_count(matrix: np.ndarray) -> int:
    """Count subjects with at least one finite ROI estimate."""
    return int(np.sum(~np.isnan(matrix).all(axis=1)))


def draw_subject_boxes(
    ax: plt.Axes,
    matrix: np.ndarray,
    positions: np.ndarray,
    color: str,
) -> None:
    """Draw ordinary boxplots from subject-level decoding accuracies."""
    columns = [
        matrix[np.isfinite(matrix[:, roi_index]), roi_index]
        for roi_index in range(len(ROI_NAMES))
    ]
    boxplot = ax.boxplot(
        columns,
        positions=positions,
        widths=BOX_WIDTH,
        patch_artist=True,
        whis=1.5,
        showfliers=True,
        manage_ticks=False,
        medianprops={"color": "black", "linewidth": 1.35},
        whiskerprops={"color": "#222222", "linewidth": 0.85},
        capprops={"color": "#222222", "linewidth": 0.85},
        flierprops={
            "marker": "+",
            "markersize": 3.5,
            "markeredgecolor": "#222222",
            "alpha": 0.8,
        },
    )
    for box in boxplot["boxes"]:
        box.set_facecolor(color)
        box.set_edgecolor("black")
        box.set_linewidth(0.7)
        box.set_alpha(0.78)


def draw_paper_boxes(
    ax: plt.Axes,
    estimates: np.ndarray,
    positions: np.ndarray,
    color: str,
) -> None:
    """Reconstruct boxplots from the paper's digitized five-number summaries."""
    for position, (lower_whisker, q1, median, q3, upper_whisker) in zip(
        positions,
        estimates,
    ):
        ax.plot(
            [position, position],
            [lower_whisker, q1],
            color="#222222",
            linewidth=0.9,
            linestyle=(0, (3, 2)),
            zorder=2,
        )
        ax.plot(
            [position, position],
            [q3, upper_whisker],
            color="#222222",
            linewidth=0.9,
            linestyle=(0, (3, 2)),
            zorder=2,
        )
        cap_half_width = BOX_WIDTH * 0.28
        ax.plot(
            [position - cap_half_width, position + cap_half_width],
            [lower_whisker, lower_whisker],
            color="#222222",
            linewidth=1.1,
            zorder=3,
        )
        ax.plot(
            [position - cap_half_width, position + cap_half_width],
            [upper_whisker, upper_whisker],
            color="#222222",
            linewidth=1.1,
            zorder=3,
        )
        box = Rectangle(
            (position - BOX_WIDTH / 2.0, q1),
            BOX_WIDTH,
            q3 - q1,
            facecolor=color,
            edgecolor="black",
            linewidth=0.9,
            alpha=0.78,
            zorder=3,
        )
        ax.add_patch(box)
        ax.plot(
            [position - BOX_WIDTH / 2.0, position + BOX_WIDTH / 2.0],
            [median, median],
            color="black",
            linewidth=1.35,
            zorder=4,
        )


def style_axis(ax: plt.Axes) -> None:
    """Apply the shared ROI labels, reference lines, and accuracy limits."""
    centers = np.arange(1, len(ROI_NAMES) + 1)
    ax.set_xlim(0.45, len(ROI_NAMES) + 0.55)
    ax.set_ylim(*YLIM)
    ax.set_yticks(YTICKS)
    ax.set_xticks(centers)
    ax.set_xticklabels(ROI_NAMES, rotation=45, ha="right", fontsize=12)
    ax.set_ylabel("Decoding Accuracy (%)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Retinotopic ROI", fontsize=14, fontweight="bold")
    ax.axhline(
        CHANCE,
        color="black",
        linestyle="--",
        linewidth=1.2,
        zorder=1,
    )
    ax.axhline(
        THRESHOLD,
        color="#666666",
        linestyle="--",
        linewidth=1.0,
        zorder=1,
    )
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.tick_params(axis="y", labelsize=11)


def validate_sources(comparisons: list[Comparison]) -> None:
    """Fail early when a configured local result file is unavailable."""
    missing = []
    for comparison in comparisons:
        for source in comparison.sources:
            if source.is_paper:
                paths = [paper_path(contrast) for contrast, _, _ in CONTRASTS]
            else:
                paths = [source.path]
            for path in paths:
                if path is None or not path.is_file():
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing plot inputs:\n" + "\n".join(sorted(set(missing))))


def plot_comparison(comparison: Comparison, contrast: str, contrast_label: str) -> Path:
    """Render and save one three-source comparison for one contrast."""
    centers = np.arange(1, len(ROI_NAMES) + 1)
    offsets = np.array([-BOX_OFFSET, 0.0, BOX_OFFSET])
    loaded: list[np.ndarray] = []
    legend_labels: list[str] = []

    fig, ax = plt.subplots(figsize=(25, 10))
    style_axis(ax)

    # Each ROI owns a three-position cluster centered on its integer x value.
    for source_index, source in enumerate(comparison.sources):
        positions = centers + offsets[source_index]
        color = COLORS[source.color_key]
        if source.is_paper:
            values = load_paper_estimates(contrast)
            draw_paper_boxes(ax, values, positions, color)
            legend_labels.append(f"{source.label} (digitized box + whiskers)")
        else:
            values = load_subject_matrix(source, contrast)
            draw_subject_boxes(ax, values, positions, color)
            legend_labels.append(f"{source.label} (n={subject_count(values)})")
        loaded.append(values)

    ax.set_title(
        f"{comparison.number}. {comparison.title}\n{contrast_label}",
        fontsize=22,
        pad=15,
    )
    handles = [
        Patch(
            facecolor=COLORS[source.color_key],
            edgecolor="black",
            alpha=0.78,
            label=label,
        )
        for source, label in zip(comparison.sources, legend_labels)
    ]
    handles.extend(
        [
            plt.Line2D(
                [0],
                [0],
                color="black",
                linestyle="--",
                linewidth=1.3,
                label="Chance = 50%",
            ),
            plt.Line2D(
                [0],
                [0],
                color="#666666",
                linestyle="--",
                linewidth=1.1,
                label="Paper threshold = 54%",
            ),
        ]
    )
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        frameon=False,
        fontsize=12,
    )

    has_paper = any(source.is_paper for source in comparison.sources)
    footer = "Subject-data boxes: 1.5-IQR whiskers and '+' outliers."
    if has_paper:
        footer += " Paper boxes and visible whisker caps are digitized estimates."
    fig.text(
        0.5,
        0.015,
        footer,
        ha="center",
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    output_path = OUTPUT_DIR / f"{comparison.number:02d}_{comparison.stem}_{contrast}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def build_comparisons() -> list[Comparison]:
    """Build the fixed set of paper, SPM, and preprocessing comparisons."""
    deep = Source(
        "DeepPrep",
        "deepprep",
        LSS_DIR / "DecodingResults_deepprep_spmspace_6conf_libsvm_balanced_sub16.mat",
    )
    fmri = Source(
        "fMRIPrep",
        "fmriprep",
        LSS_DIR / "DecodingResults_fmriprep_spmspace_6conf_libsvm_balanced_sub16.mat",
    )
    paper = Source("Paper Figure 3B", "paper", is_paper=True)
    max_all20 = Source(
        "Max's SPM",
        "max_spm",
        MAX_DIR / "DecodingResults_LIBSVM_v5.mat",
    )
    max_sub16 = Source(
        "Max's SPM",
        "max_spm",
        MAX_DIR / "DecodingResults_LIBSVM_v5.mat",
        SUB16_ORIGINAL_IDS,
    )
    our_all20 = Source(
        "Our SPM",
        "our_spm",
        OUR_SPM_DIR / "DecodingResults_spm_6conf_libsvm_balanced_all20.mat",
    )
    our_sub16 = Source(
        "Our SPM",
        "our_spm",
        OUR_SPM_DIR / "DecodingResults_spm_6conf_libsvm_balanced_sub16.mat",
    )

    return [
        Comparison(
            1,
            "paper_vs_max_vs_our_spm_n20",
            "Paper vs Max vs Our SPM, n=20",
            (paper, max_all20, our_all20),
        ),
        Comparison(
            2,
            "paper_vs_max_vs_our_spm_n16",
            "Paper vs Max vs Our SPM, n=16",
            (paper, max_sub16, our_sub16),
        ),
        Comparison(
            3,
            "deepprep_vs_fmriprep_vs_max_spm_n16",
            "DeepPrep vs fMRIPrep vs Max's SPM, n=16",
            (deep, fmri, max_sub16),
        ),
        Comparison(
            4,
            "deepprep_vs_fmriprep_vs_our_spm_n16",
            "DeepPrep vs fMRIPrep vs Our SPM, n=16",
            (deep, fmri, our_sub16),
        ),
        Comparison(
            5,
            "deepprep_vs_fmriprep_vs_paper",
            "DeepPrep vs fMRIPrep vs Paper",
            (deep, fmri, paper),
        ),
    ]


def main() -> None:
    """Generate both contrasts for every configured comparison."""
    comparisons = build_comparisons()
    validate_sources(comparisons)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for comparison in comparisons:
        for contrast, contrast_label, _ in CONTRASTS:
            output_path = plot_comparison(comparison, contrast, contrast_label)
            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
