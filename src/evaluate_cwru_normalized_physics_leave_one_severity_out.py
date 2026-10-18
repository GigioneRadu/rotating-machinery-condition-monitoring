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
    / "cwru_normalized_physics_dataset.csv"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_normalized_physics_leave_one_severity_out_results.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_normalized_physics_leave_one_severity_out_predictions.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "cwru_normalized_physics_leave_one_severity_out"
)


# ============================================================
# SETTINGS
# ============================================================

SEVERITIES = [
    0.007,
    0.014,
    0.021,
]

TARGET_COLUMN = "condition"

RANDOM_STATE = 42
N_ESTIMATORS = 500


# ============================================================
# METADATA COLUMNS
# ============================================================

METADATA_COLUMNS = [
    "source_file",
    "condition",
    "severity_inch",
    "load_hp",
    "rpm",
    "rpm_source",
    "window_index",
    "sampling_frequency",
    "window_duration_seconds",
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_FILE}\n\n"
            "Run src/build_cwru_normalized_physics_dataset.py first."
        )

    dataframe = pd.read_csv(
        DATASET_FILE
    )

    missing_metadata = [
        column
        for column in METADATA_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_metadata:
        raise ValueError(
            "Dataset is missing metadata columns: "
            f"{missing_metadata}"
        )

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in METADATA_COLUMNS
    ]

    if not feature_columns:
        raise ValueError(
            "No feature columns found."
        )

    return (
        dataframe,
        feature_columns,
    )


# ============================================================
# HELPERS
# ============================================================

def severity_mask(
    dataframe,
    severity,
):
    return np.isclose(
        dataframe[
            "severity_inch"
        ].to_numpy(),
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
        "Normalized Physics Features\n"
        f"Leave-One-Severity-Out | Test {test_severity:.3f} in"
    )

    fig.tight_layout()

    output_file = (
        FIGURES_DIR
        / f"confusion_test_{test_severity:.3f}.png"
    )

    fig.savefig(
        output_file,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# ONE FOLD
# ============================================================

def run_fold(
    dataframe,
    feature_columns,
    test_severity,
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
        test_severity=test_severity,
    )

    print()
    print("-" * 120)

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
        f"Train samples    : "
        f"{len(train_df)}"
    )

    print(
        f"Test samples     : "
        f"{len(test_df)}"
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
        "train_severities": ",".join(
            f"{severity:.3f}"
            for severity in train_severities
        ),
        "test_severity_inch": test_severity,
        "n_features": len(
            feature_columns
        ),
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
            "test_severity_inch": test_severity,
            "source_file": row[
                "source_file"
            ],
            "condition": row[
                "condition"
            ],
            "load_hp": row[
                "load_hp"
            ],
            "rpm": row[
                "rpm"
            ],
            "window_index": row[
                "window_index"
            ],
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
    print("CWRU NORMALIZED PHYSICS - LEAVE-ONE-SEVERITY-OUT")
    print("=" * 120)

    (
        dataframe,
        feature_columns,
    ) = load_dataset()

    print()
    print(
        f"Dataset shape     : "
        f"{dataframe.shape}"
    )

    print(
        f"Physics features  : "
        f"{len(feature_columns)}"
    )

    print(
        f"Classes           : "
        f"{sorted(dataframe[TARGET_COLUMN].unique().tolist())}"
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []
    all_predictions = []

    for test_severity in SEVERITIES:
        (
            result_record,
            prediction_records,
        ) = run_fold(
            dataframe=dataframe,
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
    print("SUMMARY")
    print("=" * 120)

    print()

    print(
        results_df[
            [
                "test_severity_inch",
                "accuracy",
                "macro_f1",
                "mean_confidence",
                "min_confidence",
                "max_confidence",
            ]
        ]
        .sort_values(
            "test_severity_inch"
        )
        .to_string(
            index=False
        )
    )

    mean_accuracy = float(
        results_df[
            "accuracy"
        ]
        .mean()
    )

    mean_macro_f1 = float(
        results_df[
            "macro_f1"
        ]
        .mean()
    )

    min_accuracy = float(
        results_df[
            "accuracy"
        ]
        .min()
    )

    min_macro_f1 = float(
        results_df[
            "macro_f1"
        ]
        .min()
    )

    print()
    print(
        f"Mean accuracy     : "
        f"{mean_accuracy:.4f}"
    )

    print(
        f"Minimum accuracy  : "
        f"{min_accuracy:.4f}"
    )

    print(
        f"Mean Macro F1     : "
        f"{mean_macro_f1:.4f}"
    )

    print(
        f"Minimum Macro F1  : "
        f"{min_macro_f1:.4f}"
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
                ~df[
                    "correct"
                ]
            ).astype(int)
        )
        .groupby(
            [
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
    print(
        RESULTS_FILE
    )

    print()
    print("Predictions:")
    print(
        PREDICTIONS_FILE
    )

    print()
    print("Confusion matrices:")
    print(
        FIGURES_DIR
    )

    print()
    print("=" * 120)
    print("NORMALIZED PHYSICS EVALUATION COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
