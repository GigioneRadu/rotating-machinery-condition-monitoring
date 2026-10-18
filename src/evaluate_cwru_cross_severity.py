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
    / "cwru_cross_severity"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_cross_severity_results.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_cross_severity_predictions.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_SEVERITY = 0.007

TEST_SEVERITIES = [
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
# LOAD DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_FILE}\n\n"
            "Run src/build_cwru_severity_dataset.py first."
        )

    dataframe = pd.read_csv(DATASET_FILE)

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


# ============================================================
# MODEL
# ============================================================

def create_model():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )


# ============================================================
# FLOAT-SAFE SEVERITY FILTER
# ============================================================

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


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

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
        f"CWRU Cross-Severity - {feature_group}\n"
        f"Train 0.007 in -> Test {test_severity:.3f} in"
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
# ONE EXPERIMENT
# ============================================================

def run_experiment(
    dataframe: pd.DataFrame,
    feature_group: str,
    feature_columns: list[str],
    test_severity: float,
):
    train_df = dataframe[
        severity_mask(
            dataframe,
            TRAIN_SEVERITY,
        )
    ].copy()

    test_df = dataframe[
        severity_mask(
            dataframe,
            test_severity,
        )
    ].copy()

    if train_df.empty:
        raise ValueError(
            f"No rows found for training severity "
            f"{TRAIN_SEVERITY:.3f} in."
        )

    if test_df.empty:
        raise ValueError(
            f"No rows found for test severity "
            f"{test_severity:.3f} in."
        )

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

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    confidence = np.max(
        probabilities,
        axis=1,
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
        f"TRAIN {TRAIN_SEVERITY:.3f} in "
        f"-> TEST {test_severity:.3f} in"
    )

    print(
        f"Accuracy        : {accuracy:.4f}"
    )

    print(
        f"Macro F1        : {macro_f1:.4f}"
    )

    print(
        f"Mean confidence : "
        f"{np.mean(confidence):.4f}"
    )

    print(
        f"Min confidence  : "
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
            "train_severity_inch": TRAIN_SEVERITY,
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
            probability_index = (
                class_to_index[
                    class_name
                ]
            )

            record[
                f"probability_{class_name}"
            ] = probabilities[
                row_index,
                probability_index,
            ]

        prediction_records.append(
            record
        )

    result_record = {
        "feature_group": feature_group,
        "n_features": len(
            feature_columns
        ),
        "train_severity_inch": TRAIN_SEVERITY,
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

    return (
        result_record,
        prediction_records,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 115)
    print("CWRU CROSS-SEVERITY GENERALIZATION")
    print("=" * 115)

    print()
    print(
        f"Training severity : "
        f"{TRAIN_SEVERITY:.3f} in"
    )

    print(
        "Test severities   : "
        + ", ".join(
            f"{severity:.3f} in"
            for severity in TEST_SEVERITIES
        )
    )

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
        print("=" * 115)
        print(
            f"FEATURE GROUP: {feature_group}"
        )
        print("=" * 115)

        print()
        print(
            f"Features ({len(feature_columns)}): "
            + ", ".join(
                feature_columns
            )
        )

        for test_severity in TEST_SEVERITIES:
            (
                result_record,
                prediction_records,
            ) = run_experiment(
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
    print("=" * 115)
    print("CROSS-SEVERITY SUMMARY")
    print("=" * 115)

    print()

    summary_columns = [
        "feature_group",
        "test_severity_inch",
        "accuracy",
        "macro_f1",
        "mean_confidence",
        "min_confidence",
    ]

    print(
        results_df[
            summary_columns
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
    print("=" * 115)
    print("COMPARISON: TIME-DOMAIN VS ENVELOPE")
    print("=" * 115)

    comparison = (
        results_df[
            results_df[
                "feature_group"
            ]
            .isin(
                [
                    "time_domain_only",
                    "envelope_only",
                ]
            )
        ]
        [
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
                "feature_group",
            ]
        )
    )

    print()
    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print("=" * 115)
    print("MISCLASSIFICATION COUNTS")
    print("=" * 115)

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
    print("=" * 115)
    print("FILES SAVED")
    print("=" * 115)

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
    print("=" * 115)
    print("CROSS-SEVERITY EVALUATION COMPLETE")
    print("=" * 115)


if __name__ == "__main__":
    main()
