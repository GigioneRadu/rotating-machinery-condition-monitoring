from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_fault_severity_dataset.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "cwru_leave_one_severity_out"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_leave_one_severity_out_results.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_leave_one_severity_out_predictions.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SEVERITIES = [
    0.007,
    0.014,
    0.021,
]

RANDOM_STATE = 42
N_ESTIMATORS = 500

TARGET_COLUMN = "condition"


# ============================================================
# FEATURE GROUPS
# ============================================================

TIME_DOMAIN_FEATURES = [
    "rms",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
]

RAW_FREQUENCY_FEATURES = [
    "shaft_amplitude",
    "bpfo_raw",
    "bpfi_raw",
    "bsf_raw",
    "ftf_raw",
]

ENVELOPE_FEATURES = [
    "bpfo_envelope",
    "bpfi_envelope",
    "bsf_envelope",
    "ftf_envelope",
]

PHYSICS_FREQUENCY_FEATURES = (
    RAW_FREQUENCY_FEATURES
    + ENVELOPE_FEATURES
)

ALL_FEATURES = (
    TIME_DOMAIN_FEATURES
    + PHYSICS_FREQUENCY_FEATURES
)

FEATURE_GROUPS = {
    "time_domain_only": TIME_DOMAIN_FEATURES,
    "raw_frequency_only": RAW_FREQUENCY_FEATURES,
    "envelope_only": ENVELOPE_FEATURES,
    "physics_frequency_only": PHYSICS_FREQUENCY_FEATURES,
    "time_plus_envelope": (
        TIME_DOMAIN_FEATURES
        + ENVELOPE_FEATURES
    ),
    "all_features": ALL_FEATURES,
}


# ============================================================
# HELPERS
# ============================================================

def load_dataset() -> pd.DataFrame:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_FILE}"
        )

    dataframe = pd.read_csv(
        DATASET_FILE
    )

    required_columns = [
        TARGET_COLUMN,
        "severity_inch",
        "source_file",
        "load_hp",
        "rpm",
        "window_index",
        *ALL_FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{missing_columns}"
        )

    return dataframe


def severity_mask(
    dataframe: pd.DataFrame,
    severity: float,
):
    return np.isclose(
        dataframe["severity_inch"].to_numpy(),
        severity,
        rtol=0.0,
        atol=1e-9,
    )


def create_model():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )


def save_confusion_matrix(
    y_true,
    y_pred,
    labels,
    feature_group,
    test_severity,
):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        f"CWRU Leave-One-Severity-Out - {feature_group}\n"
        f"Test severity {test_severity:.3f} in"
    )

    fig.tight_layout()

    output_file = (
        FIGURES_DIR
        / (
            f"confusion_{feature_group}_"
            f"test_{test_severity:.3f}.png"
        )
    )

    fig.savefig(
        output_file,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# ONE FOLD
# ============================================================

def run_fold(
    dataframe: pd.DataFrame,
    feature_group: str,
    feature_columns: list[str],
    test_severity: float,
):
    train_severities = [
        severity
        for severity in SEVERITIES
        if not np.isclose(
            severity,
            test_severity,
            rtol=0.0,
            atol=1e-9,
        )
    ]

    train_mask = np.zeros(
        len(dataframe),
        dtype=bool,
    )

    for severity in train_severities:
        train_mask |= severity_mask(
            dataframe,
            severity,
        )

    test_mask = severity_mask(
        dataframe,
        test_severity,
    )

    train_df = dataframe[
        train_mask
    ].copy()

    test_df = dataframe[
        test_mask
    ].copy()

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        TARGET_COLUMN
    ]

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        TARGET_COLUMN
    ]

    model = create_model()

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    confidence = np.max(
        probabilities,
        axis=1,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    labels = sorted(
        dataframe[
            TARGET_COLUMN
        ]
        .unique()
        .tolist()
    )

    save_confusion_matrix(
        y_true=y_test,
        y_pred=predictions,
        labels=labels,
        feature_group=feature_group,
        test_severity=test_severity,
    )

    print()
    print(
        "Train severities : "
        + ", ".join(
            f"{severity:.3f} in"
            for severity in train_severities
        )
    )

    print(
        f"Test severity    : "
        f"{test_severity:.3f} in"
    )

    print(
        f"Accuracy         : "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro F1         : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Mean confidence  : "
        f"{np.mean(confidence):.4f}"
    )

    print(
        f"Min confidence   : "
        f"{np.min(confidence):.4f}"
    )

    print()

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    result_record = {
        "feature_group": feature_group,
        "n_features": len(
            feature_columns
        ),
        "train_severities": ",".join(
            f"{severity:.3f}"
            for severity in train_severities
        ),
        "test_severity_inch": test_severity,
        "train_samples": len(
            train_df
        ),
        "test_samples": len(
            test_df
        ),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "mean_confidence": float(
            np.mean(
                confidence
            )
        ),
        "min_confidence": float(
            np.min(
                confidence
            )
        ),
        "max_confidence": float(
            np.max(
                confidence
            )
        ),
    }

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(
            model.classes_
        )
    }

    prediction_records = []

    test_df_reset = test_df.reset_index(
        drop=True
    )

    for row_index in range(
        len(test_df_reset)
    ):
        row = test_df_reset.iloc[
            row_index
        ]

        record = {
            "feature_group": feature_group,
            "test_severity_inch": test_severity,
            "source_file": row["source_file"],
            "condition": row["condition"],
            "load_hp": row["load_hp"],
            "rpm": row["rpm"],
            "window_index": row["window_index"],
            "predicted_condition": predictions[
                row_index
            ],
            "correct": (
                row["condition"]
                == predictions[
                    row_index
                ]
            ),
            "prediction_confidence": confidence[
                row_index
            ],
        }

        for class_name in model.classes_:
            record[
                f"probability_{class_name}"
            ] = probabilities[
                row_index,
                class_to_index[
                    class_name
                ],
            ]

        prediction_records.append(
            record
        )

    return (
        result_record,
        prediction_records,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 120)
    print("CWRU LEAVE-ONE-SEVERITY-OUT EVALUATION")
    print("=" * 120)

    dataframe = load_dataset()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []
    all_predictions = []

    for (
        feature_group,
        feature_columns,
    ) in FEATURE_GROUPS.items():
        print()
        print("=" * 120)
        print(
            f"FEATURE GROUP: "
            f"{feature_group}"
        )
        print("=" * 120)

        print()
        print(
            f"Features ({len(feature_columns)}): "
            + ", ".join(
                feature_columns
            )
        )

        for test_severity in SEVERITIES:
            (
                result_record,
                prediction_records,
            ) = run_fold(
                dataframe=dataframe,
                feature_group=feature_group,
                feature_columns=feature_columns,
                test_severity=test_severity,
            )

            all_results.append(
                result_record
            )

            all_predictions.extend(
                prediction_records
            )

    results_df = pd.DataFrame(
        all_results
    )

    predictions_df = pd.DataFrame(
        all_predictions
    )

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print()
    print("=" * 120)
    print("LEAVE-ONE-SEVERITY-OUT SUMMARY")
    print("=" * 120)

    print()

    print(
        results_df[
            [
                "feature_group",
                "test_severity_inch",
                "accuracy",
                "macro_f1",
                "mean_confidence",
                "min_confidence",
            ]
        ]
        .sort_values(
            by=[
                "test_severity_inch",
                "accuracy",
                "macro_f1",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("MEAN PERFORMANCE ACROSS HELD-OUT SEVERITIES")
    print("=" * 120)

    print()

    mean_summary = (
        results_df
        .groupby(
            [
                "feature_group",
                "n_features",
            ],
            as_index=False,
        )
        .agg(
            mean_accuracy=(
                "accuracy",
                "mean",
            ),
            min_accuracy=(
                "accuracy",
                "min",
            ),
            mean_macro_f1=(
                "macro_f1",
                "mean",
            ),
            min_macro_f1=(
                "macro_f1",
                "min",
            ),
            mean_confidence=(
                "mean_confidence",
                "mean",
            ),
        )
        .sort_values(
            by=[
                "mean_accuracy",
                "mean_macro_f1",
            ],
            ascending=False,
        )
    )

    print(
        mean_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("MISCLASSIFICATION COUNTS")
    print("=" * 120)

    print()

    error_summary = (
        predictions_df
        .assign(
            error=lambda df: (
                ~df["correct"]
            ).astype(int)
        )
        .groupby(
            [
                "feature_group",
                "test_severity_inch",
                "condition",
            ]
        )[
            "error"
        ]
        .sum()
        .reset_index()
    )

    print(
        error_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("FILES SAVED")
    print("=" * 120)

    print()
    print("Results:")
    print(RESULTS_FILE)

    print()
    print("Predictions:")
    print(PREDICTIONS_FILE)

    print()
    print("Confusion matrices:")
    print(FIGURES_DIR)

    print()
    print("=" * 120)
    print("LEAVE-ONE-SEVERITY-OUT EVALUATION COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
