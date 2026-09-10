"""Compact reporting and explainability helpers for the selected DSSE-Net model.

The primary model is the completed ``without_residual`` ablation variant.  This
module intentionally reads frozen artifacts and never starts model training.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from statsmodels.stats.contingency_tables import mcnemar
from torch import nn
from torchvision import transforms


PRIMARY_VARIANT = "without_residual"
CLASS_TO_INDEX = {"meningioma": 0, "glioma": 1, "pituitary": 2}
CLASS_NAMES = ["meningioma", "glioma", "pituitary"]
IMAGE_SIZE = (224, 224)


def read_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def primary_result_root(project_root: Path) -> Path:
    return (
        Path(project_root)
        / "nested_cv_ablation_results"
        / "split_ablation"
        / PRIMARY_VARIANT
    )


def holm_adjust_pvalues(p_values: list[float]) -> np.ndarray:
    """Return monotone Holm-adjusted p-values in the original order."""
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        return values
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_maximum = 0.0
    number_of_tests = len(values)
    for rank, original_index in enumerate(order):
        candidate = min(
            1.0,
            (number_of_tests - rank) * values[original_index],
        )
        running_maximum = max(running_maximum, candidate)
        adjusted[original_index] = running_maximum
    return adjusted


def build_primary_baseline_mcnemar(
    project_root: Path,
) -> tuple[pd.DataFrame, Path]:
    """Compare selected-model and baseline OOF predictions by patient.

    The function reads frozen predictions only. It uses the exact two-sided
    McNemar test for each paired comparison and controls the family-wise error
    rate across the five baselines with the Holm procedure.
    """
    project_root = Path(project_root)
    primary_predictions_path = (
        primary_result_root(project_root)
        / "oof_evaluation"
        / "oof_patient_level_predictions.csv"
    )
    baseline_root = (
        project_root
        / "nested_cv_baseline_results"
        / "split_baseline"
    )
    baseline_models = {
        "resnet50": "ResNet-50",
        "densenet121": "DenseNet-121",
        "mobilenet_v3_large": "MobileNetV3-Large",
        "efficientnet_b0": "EfficientNet-B0",
        "regnet_y_400mf": "RegNet-Y-400MF",
    }
    primary = pd.read_csv(
        primary_predictions_path,
        dtype={"patient_id": str},
    )[["patient_id", "true_label", "predicted_label"]].rename(
        columns={"predicted_label": "primary_prediction"}
    )
    if primary["patient_id"].duplicated().any():
        raise RuntimeError("Selected-model OOF predictions contain duplicate patients")

    rows = []
    for model_name, model_label in baseline_models.items():
        baseline_predictions_path = (
            baseline_root
            / model_name
            / "oof_evaluation"
            / "oof_patient_level_predictions.csv"
        )
        baseline = pd.read_csv(
            baseline_predictions_path,
            dtype={"patient_id": str},
        )[["patient_id", "true_label", "predicted_label"]].rename(
            columns={
                "true_label": "baseline_true_label",
                "predicted_label": "baseline_prediction",
            }
        )
        paired = primary.merge(
            baseline,
            on="patient_id",
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != len(primary):
            raise RuntimeError(f"Incomplete patient pairing for {model_name}")
        if not paired["true_label"].eq(
            paired["baseline_true_label"]
        ).all():
            raise RuntimeError(f"True labels disagree for {model_name}")

        primary_correct = paired["primary_prediction"].eq(
            paired["true_label"]
        )
        baseline_correct = paired["baseline_prediction"].eq(
            paired["true_label"]
        )
        both_correct = int((primary_correct & baseline_correct).sum())
        primary_only = int((primary_correct & ~baseline_correct).sum())
        baseline_only = int((~primary_correct & baseline_correct).sum())
        both_wrong = int((~primary_correct & ~baseline_correct).sum())
        contingency = np.asarray(
            [[both_correct, primary_only], [baseline_only, both_wrong]],
            dtype=int,
        )
        test_result = mcnemar(contingency, exact=True, correction=False)
        rows.append(
            {
                "comparison": f"without_residual_vs_{model_name}",
                "baseline_model": model_name,
                "baseline_label": model_label,
                "patients": len(paired),
                "both_correct": both_correct,
                "primary_correct_baseline_wrong": primary_only,
                "primary_wrong_baseline_correct": baseline_only,
                "both_wrong": both_wrong,
                "discordant_pairs": primary_only + baseline_only,
                "test": "exact_two_sided_mcnemar",
                "mcnemar_statistic": float(test_result.statistic),
                "raw_p_value": float(test_result.pvalue),
            }
        )

    table = pd.DataFrame(rows)
    table["holm_adjusted_p_value"] = holm_adjust_pvalues(
        table["raw_p_value"].tolist()
    )
    table["significant_after_holm_0_05"] = (
        table["holm_adjusted_p_value"] < 0.05
    )
    output_path = (
        baseline_root
        / "tables"
        / "paired_patient_mcnemar_holm_vs_without_residual.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table, output_path


class SEBlock(nn.Module):
    """Squeeze-and-excitation attention used by the selected architecture."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = self.fc(inputs).unsqueeze(-1).unsqueeze(-1)
        return inputs * weights


class DSSEBlockWithoutResidual(nn.Module):
    """Depthwise-SE block with dropout and no shortcut addition."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout_probability: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_residual = False
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout2d(dropout_probability)
        # Kept for state-structure compatibility with the completed experiment.
        self.shortcut = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv1(inputs)
        outputs = self.dwconv(outputs)
        outputs = self.se(outputs)
        outputs = self.dropout(outputs)
        return self.relu(outputs)


class DSSEWithoutResidual(nn.Module):
    """Selected DSSE-Net architecture: SE + depthwise convolution, no residual."""

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        block = DSSEBlockWithoutResidual
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.stage1 = nn.Sequential(block(32, 32), block(32, 32))
        self.down1 = nn.Conv2d(32, 96, kernel_size=1, stride=2)
        self.stage2 = nn.Sequential(
            block(96, 96),
            block(96, 96),
            block(96, 96),
        )
        self.down2 = nn.Conv2d(96, 192, kernel_size=1, stride=2)
        self.stage3 = nn.Sequential(
            block(192, 192),
            block(192, 192),
            block(192, 192),
            block(192, 192),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        outputs = self.stage1(outputs)
        outputs = self.down1(outputs)
        outputs = self.stage2(outputs)
        outputs = self.down2(outputs)
        outputs = self.stage3(outputs)
        return self.classifier(outputs)


def load_primary_checkpoint(
    checkpoint_path: Path,
    device: str | torch.device | None = None,
) -> tuple[DSSEWithoutResidual, dict]:
    """Load and validate a completed outer-fold checkpoint."""

    execution_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location=execution_device,
        weights_only=False,
    )
    if checkpoint.get("variant_name") != PRIMARY_VARIANT:
        raise RuntimeError("Checkpoint is not the selected without_residual variant")
    variant_config = checkpoint.get("variant_config", {})
    if variant_config.get("use_residual") is not False:
        raise RuntimeError("Checkpoint unexpectedly enables residual additions")
    if checkpoint.get("class_to_index") != CLASS_TO_INDEX:
        raise RuntimeError("Checkpoint class mapping does not match the study")

    model = DSSEWithoutResidual(num_classes=len(CLASS_NAMES)).to(execution_device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _find_with_residual_result_root(project_root: Path, split_hash: str) -> Path:
    candidates = sorted((Path(project_root) / "nested_cv_results").glob("split_*"))
    for candidate in candidates:
        summary_path = candidate / "experiment_summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            if summary.get("split_assignment_sha256") == split_hash:
                return candidate
    raise FileNotFoundError("Could not locate the frozen with-residual DSSE-Net result")


def _result_row(label: str, result_root: Path) -> dict:
    summary = read_json(result_root / "experiment_summary.json")
    patient = summary["oof_patient_level_metrics"]
    slices = summary["oof_slice_level_metrics"]
    fold_metrics = pd.read_csv(result_root / "tables" / "outer_fold_metrics.csv")
    patient_folds = fold_metrics.loc[
        fold_metrics["evaluation_level"].eq("patient_level")
    ]
    return {
        "model": label,
        "correct_patients": int(round(patient["accuracy"] * patient["samples"])),
        "patients": int(patient["samples"]),
        "patient_accuracy": patient["accuracy"],
        "patient_balanced_accuracy": patient["balanced_accuracy"],
        "patient_macro_f1": patient["macro_f1"],
        "patient_macro_auc": patient["macro_roc_auc_ovr"],
        "patient_ece": patient["expected_calibration_error_10_bins"],
        "slice_accuracy": slices["accuracy"],
        "slice_macro_f1": slices["macro_f1"],
        "fold_accuracy_sd": patient_folds["accuracy"].std(),
    }


def load_primary_report(project_root: Path) -> dict[str, pd.DataFrame | Path]:
    """Load compact, publication-oriented results without retraining."""

    project_root = Path(project_root)
    selected_root = primary_result_root(project_root)
    selected_summary = read_json(selected_root / "experiment_summary.json")
    split_hash = selected_summary["split_assignment_sha256"]
    with_residual_root = _find_with_residual_result_root(project_root, split_hash)
    combined_root = (
        project_root
        / "nested_cv_candidate_results"
        / "DSSE_NET_No_Residual"
    )
    result_roots = {
        "With residual connections": with_residual_root,
        "Without residual (selected)": selected_root,
        "Without residual + no V-flip": combined_root,
    }
    missing = [
        root / "experiment_summary.json"
        for root in result_roots.values()
        if not (root / "experiment_summary.json").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing completed result artifacts: {missing}")

    comparison = pd.DataFrame(
        [_result_row(label, root) for label, root in result_roots.items()]
    )
    selected_folds = pd.read_csv(
        selected_root / "tables" / "outer_fold_metrics.csv"
    )
    selected_folds = selected_folds.loc[
        selected_folds["evaluation_level"].eq("patient_level"),
        [
            "outer_fold",
            "samples",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "macro_roc_auc_ovr",
        ],
    ].reset_index(drop=True)
    per_class = pd.read_csv(
        selected_root
        / "oof_evaluation"
        / "oof_patient_level_per_class_metrics.csv"
    )
    intervals = pd.read_csv(
        selected_root
        / "tables"
        / "oof_patient_cluster_bootstrap_95ci.csv"
    )
    intervals = intervals.loc[
        intervals["evaluation_level"].eq("patient_level")
    ].reset_index(drop=True)

    return {
        "comparison": comparison,
        "selected_folds": selected_folds,
        "per_class": per_class,
        "confidence_intervals": intervals,
        "selected_root": selected_root,
    }


def plot_primary_dashboard(
    report: dict[str, pd.DataFrame | Path],
    output_path: Path | None = None,
) -> Path:
    """Create one compact dashboard for the selected model and its references."""

    comparison = report["comparison"]
    selected_folds = report["selected_folds"]
    per_class = report["per_class"]
    selected_root = Path(report["selected_root"])
    if output_path is None:
        output_path = selected_root / "figures" / "primary_model_dashboard.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    colors = ["#9CA3AF", "#167D9A", "#D18B32"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    x = np.arange(len(comparison))
    width = 0.34
    axes[0, 0].bar(
        x - width / 2,
        comparison["patient_accuracy"] * 100,
        width,
        label="Accuracy",
        color=colors,
    )
    axes[0, 0].bar(
        x + width / 2,
        comparison["patient_macro_f1"] * 100,
        width,
        label="Macro-F1",
        color=colors,
        alpha=0.55,
        hatch="//",
    )
    axes[0, 0].set_xticks(x, ["With residual", "Selected", "No V-flip"])
    axes[0, 0].set_ylim(80, 100)
    axes[0, 0].set_ylabel("Patient-level score (%)")
    axes[0, 0].set_title("Pooled out-of-fold performance")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        selected_folds["outer_fold"] + 1,
        selected_folds["accuracy"] * 100,
        marker="o",
        linewidth=2,
        color=colors[1],
        label="Accuracy",
    )
    axes[0, 1].plot(
        selected_folds["outer_fold"] + 1,
        selected_folds["macro_f1"] * 100,
        marker="s",
        linewidth=2,
        color="#6B4C9A",
        label="Macro-F1",
    )
    axes[0, 1].set_xticks(range(1, 6))
    axes[0, 1].set_ylim(80, 100)
    axes[0, 1].set_xlabel("Outer fold")
    axes[0, 1].set_ylabel("Score (%)")
    axes[0, 1].set_title("Selected model across outer folds")
    axes[0, 1].legend(frameon=False)

    class_x = np.arange(len(per_class))
    class_width = 0.26
    for offset, column, label, color in [
        (-class_width, "precision", "Precision", "#167D9A"),
        (0, "sensitivity_recall", "Sensitivity", "#D18B32"),
        (class_width, "f1_score", "F1", "#6B4C9A"),
    ]:
        axes[1, 0].bar(
            class_x + offset,
            per_class[column] * 100,
            class_width,
            label=label,
            color=color,
        )
    axes[1, 0].set_xticks(class_x, per_class["class_name"].str.title())
    axes[1, 0].set_ylim(75, 100)
    axes[1, 0].set_ylabel("Patient-level score (%)")
    axes[1, 0].set_title("Selected model by class")
    axes[1, 0].legend(frameon=False, ncol=3, fontsize=8)

    axes[1, 1].bar(
        x,
        comparison["patient_ece"] * 100,
        color=colors,
    )
    axes[1, 1].set_xticks(x, ["With residual", "Selected", "No V-flip"])
    axes[1, 1].set_ylabel("ECE (%) — lower is better")
    axes[1, 1].set_title("Patient-level calibration")

    fig.suptitle(
        "Selected DSSE-Net without residual connections",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _representative_correct_slices(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    correct = predictions.loc[predictions["correct"].astype(bool)].copy()
    for class_name in CLASS_NAMES:
        group = correct.loc[correct["true_class"].eq(class_name)].copy()
        probability_column = f"probability_{class_name}"
        median_confidence = group[probability_column].median()
        group["distance_from_class_median"] = (
            group[probability_column] - median_confidence
        ).abs()
        selected = group.sort_values(
            ["distance_from_class_median", "file_name"],
            kind="stable",
        ).iloc[0]
        rows.append(selected)
    return pd.DataFrame(rows).reset_index(drop=True)


def build_primary_gradcam_gallery(
    project_root: Path,
    output_path: Path | None = None,
    *,
    overwrite: bool = False,
    device: str | torch.device | None = None,
) -> dict[str, Path | pd.DataFrame]:
    """Generate representative OOF Grad-CAMs for the selected model.

    One correctly classified slice per class is selected deterministically at
    the class-median confidence.  Its matching outer-fold model is used, so the
    shown image was not part of that model's training data.
    """

    project_root = Path(project_root)
    selected_root = primary_result_root(project_root)
    figures_root = selected_root / "figures"
    if output_path is None:
        output_path = figures_root / "gradcam_representative_oof.png"
    output_path = Path(output_path)
    selection_path = figures_root / "gradcam_representative_oof_selection.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_csv(
        selected_root
        / "oof_evaluation"
        / "oof_slice_level_predictions.csv",
        dtype={"patient_id": str},
    )
    selected = _representative_correct_slices(predictions)
    if output_path.exists() and selection_path.exists() and not overwrite:
        return {
            "figure_path": output_path,
            "selection": pd.read_csv(selection_path, dtype={"patient_id": str}),
        }

    execution_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(
                IMAGE_SIZE,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
    )
    model_cache: dict[int, DSSEWithoutResidual] = {}
    gallery_rows = []
    figure, axes = plt.subplots(
        len(selected),
        3,
        figsize=(10.5, 10),
        constrained_layout=True,
    )

    for row_index, row in selected.iterrows():
        outer_fold = int(row["outer_fold"])
        class_name = str(row["true_class"])
        image_path = (
            project_root
            / "nested_cv_cropped"
            / f"outer_fold_{outer_fold}"
            / "test"
            / class_name
            / Path(row["file_name"]).with_suffix(".png")
        )
        if not image_path.exists():
            raise FileNotFoundError(f"Missing OOF image for Grad-CAM: {image_path}")

        if outer_fold not in model_cache:
            checkpoint_path = (
                selected_root
                / "models"
                / f"outer_fold_{outer_fold}"
                / "final_refit"
                / "final_model.pt"
            )
            model_cache[outer_fold], _ = load_primary_checkpoint(
                checkpoint_path,
                device=execution_device,
            )
        model = model_cache[outer_fold]

        pil_image = Image.open(image_path).convert("L")
        input_tensor = evaluation_transform(pil_image).unsqueeze(0).to(execution_device)
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_index = int(logits.argmax(dim=1).item())
            confidence = float(probabilities[0, predicted_index].item())

        target_layers = [model.stage3[-1].dwconv[0]]
        targets = [ClassifierOutputTarget(predicted_index)]
        with GradCAM(model=model, target_layers=target_layers) as cam:
            activation = cam(input_tensor=input_tensor, targets=targets)[0]

        grayscale = input_tensor[0, 0].detach().cpu().numpy()
        grayscale = (grayscale - grayscale.min()) / (
            grayscale.max() - grayscale.min() + 1e-8
        )
        rgb_image = np.repeat(grayscale[..., None], 3, axis=2).astype(np.float32)
        overlay = show_cam_on_image(
            rgb_image,
            activation,
            use_rgb=True,
            image_weight=0.65,
        )

        axes[row_index, 0].imshow(grayscale, cmap="gray")
        axes[row_index, 1].imshow(activation, cmap="jet", vmin=0, vmax=1)
        axes[row_index, 2].imshow(overlay)
        for column in range(3):
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_ylabel(class_name.title(), fontsize=11)
        axes[row_index, 2].set_title(
            f"Predicted: {CLASS_NAMES[predicted_index]} | p={confidence:.3f}\n"
            f"Outer fold {outer_fold + 1}",
            fontsize=9,
        )
        gallery_rows.append(
            {
                "class_name": class_name,
                "file_name": row["file_name"],
                "patient_id": row["patient_id"],
                "outer_fold": outer_fold,
                "image_path": image_path.relative_to(project_root).as_posix(),
                "predicted_class": CLASS_NAMES[predicted_index],
                "confidence": confidence,
                "selection_rule": "correct OOF slice nearest class-median confidence",
                "target_layer": "stage3[-1].dwconv[0]",
            }
        )

    axes[0, 0].set_title("Cropped MRI")
    axes[0, 1].set_title("Grad-CAM activation")
    figure.suptitle(
        "DSSE-Net without residual connections: representative OOF Grad-CAM",
        fontsize=14,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    selection = pd.DataFrame(gallery_rows)
    selection.to_csv(selection_path, index=False)
    model_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"figure_path": output_path, "selection": selection}
