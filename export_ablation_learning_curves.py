"""Export publication-ready learning curves from saved nested-CV histories.

The script is read-only with respect to trained models. It aggregates the 25
inner-fold histories (5 outer development sets x 5 inner folds) for each
residual-free ablation configuration and writes figures/tables used by the
Supplementary Material.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT.parent / "supplementary_figures" / "learning_curves"

VARIANTS = {
    "Selected residual-free": ROOT
    / "nested_cv_ablation_results"
    / "split_ablation"
    / "without_residual"
    / "models",
    "Add residual shortcuts": ROOT
    / "nested_cv_results"
    / "split_554d062a5bd7"
    / "models",
    "Remove vertical flip": ROOT
    / "nested_cv_candidate_results"
    / "DSSE_NET_No_Residual"
    / "models",
    "Remove SE attention": ROOT
    / "nested_cv_ablation_results"
    / "split_ablation_without_residual_reference"
    / "no_residual_without_se"
    / "models",
    "Use standard 3x3 convolution": ROOT
    / "nested_cv_ablation_results"
    / "split_ablation_without_residual_reference"
    / "no_residual_standard_convolution"
    / "models",
    "Remove dropout": ROOT
    / "nested_cv_ablation_results"
    / "split_ablation_without_residual_reference"
    / "no_residual_without_dropout"
    / "models",
    "Remove all augmentation": ROOT
    / "nested_cv_ablation_results"
    / "split_ablation_without_residual_reference"
    / "no_residual_without_augmentation"
    / "models",
}

METRICS = [
    "train_loss",
    "train_accuracy",
    "validation_loss",
    "validation_accuracy",
    "validation_macro_f1",
]


def load_inner_histories(label: str, models_dir: Path) -> pd.DataFrame:
    # The frozen full model includes an additional ``paper_default`` directory;
    # the ablation runs store ``inner_fold_*`` directly under inner_selection.
    pattern = "outer_fold_*/inner_selection/**/inner_fold_*/history.csv"
    paths = sorted(models_dir.glob(pattern))
    if len(paths) != 25:
        raise RuntimeError(
            f"{label}: expected 25 inner-fold histories, found {len(paths)} in {models_dir}"
        )

    frames: list[pd.DataFrame] = []
    for path in paths:
        outer_part = next(part for part in path.parts if part.startswith("outer_fold_"))
        inner_part = next(part for part in path.parts if part.startswith("inner_fold_"))
        outer_fold = int(outer_part.split("_")[-1])
        inner_fold = int(inner_part.split("_")[-1])
        frame = pd.read_csv(path)
        missing = {"epoch", *METRICS} - set(frame.columns)
        if missing:
            raise RuntimeError(f"{path} is missing columns: {sorted(missing)}")
        frame = frame[["epoch", *METRICS]].copy()
        frame.insert(0, "inner_fold", inner_fold)
        frame.insert(0, "outer_fold", outer_fold)
        frame.insert(0, "variant", label)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def aggregate(histories: pd.DataFrame) -> pd.DataFrame:
    summary = (
        histories.groupby(["variant", "epoch"], sort=False)[METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]
    return summary


def draw_all_validation_curves(histories: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.2, 6.8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, label in zip(axes, VARIANTS):
        subset = histories[histories["variant"] == label]
        for (_, _), run in subset.groupby(["outer_fold", "inner_fold"]):
            ax.plot(
                run["epoch"],
                run["validation_macro_f1"],
                color="#8fa8c7",
                alpha=0.20,
                linewidth=0.7,
            )
        mean_curve = subset.groupby("epoch")["validation_macro_f1"].mean()
        ax.plot(
            mean_curve.index,
            mean_curve.values,
            color="#173f6b",
            linewidth=2.0,
            label="Mean of 25 inner runs",
        )
        ax.set_title(label, fontsize=9)
        ax.set_ylim(0.15, 1.01)
        ax.grid(alpha=0.22, linewidth=0.5)

    axes[-1].axis("off")
    for ax in axes[:4]:
        ax.set_ylabel("Validation macro F1")
    for ax in axes[4:7]:
        ax.set_xlabel("Epoch")
    axes[0].legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_selected_vs_standard(summary: pd.DataFrame, output_path: Path) -> None:
    labels = ["Selected residual-free", "Use standard 3x3 convolution"]
    colors = {
        "Selected residual-free": "#0b6e4f",
        "Use standard 3x3 convolution": "#b23a48",
    }
    panels = [
        ("train_loss", "Training loss"),
        ("validation_loss", "Validation loss"),
        ("validation_accuracy", "Validation accuracy"),
        ("validation_macro_f1", "Validation macro F1"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6), sharex=True)
    for ax, (metric, title) in zip(axes.ravel(), panels):
        for label in labels:
            curve = summary[summary["variant"] == label].sort_values("epoch")
            x = curve["epoch"].to_numpy(dtype=float)
            mean = curve[f"{metric}_mean"].to_numpy(dtype=float)
            std = curve[f"{metric}_std"].fillna(0).to_numpy(dtype=float)
            ax.plot(x, mean, color=colors[label], linewidth=2.0, label=label)
            ax.fill_between(x, mean - std, mean + std, color=colors[label], alpha=0.14)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.22, linewidth=0.5)
        ax.set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Cross-entropy")
    axes[0, 1].set_ylabel("Cross-entropy")
    axes[1, 0].set_ylabel("Proportion")
    axes[1, 1].set_ylabel("Score")
    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    histories = pd.concat(
        [load_inner_histories(label, path) for label, path in VARIANTS.items()],
        ignore_index=True,
    )
    summary = aggregate(histories)

    histories.to_csv(OUTPUT_DIR / "ablation_inner_fold_histories.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "ablation_inner_fold_learning_curve_summary.csv", index=False)

    inventory = (
        histories.groupby("variant")
        .agg(
            inner_runs=("inner_fold", "size"),
            unique_inner_runs=("outer_fold", lambda x: len(x)),
            min_epochs=("epoch", "min"),
            max_epochs=("epoch", "max"),
        )
        .reset_index()
    )
    inventory["unique_inner_runs"] = 25
    inventory["history_rows"] = histories.groupby("variant").size().values
    inventory.to_csv(OUTPUT_DIR / "learning_curve_inventory.csv", index=False)

    draw_all_validation_curves(
        histories, OUTPUT_DIR / "FigS4_all_ablation_validation_macro_f1.png"
    )
    draw_selected_vs_standard(
        summary, OUTPUT_DIR / "FigS5_selected_vs_standard_learning_curves.png"
    )

    print(f"Saved learning-curve artifacts to: {OUTPUT_DIR}")
    print(histories.groupby("variant")[["outer_fold", "inner_fold"]].apply(lambda x: len(x.drop_duplicates())))


if __name__ == "__main__":
    main()
