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
    / "cwru_real_bearing_dataset.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "cwru_leave_one_load_out"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_leave_one_load_out_results.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_leave_one_load_out_predictions.csv"
)

AGGREGATE_CONFUSION_MATRIX_FILE = (
    FIGURES_DIR
    / "aggregate_confusion_matrix.png"
)


# ============================================================
# SETTINGS
# ============================================================

LOADS = [0, 1, 2, 3]

RANDOM_STATE = 42
N_ESTIMATORS = 500


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "rms",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
    "shaft_amplitude",
    "bpfo_raw",
    "bpfi_raw",
    "bsf_raw",
    "ftf_raw",
    "bpfo_envelope",
    "bpfi_envelope",
    "bsf_envelope",
    "ftf_envelope",
]

TARGET_COLUMN = "condition"


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_FILE}\n\n"
            "Run src/build_cwru_dataset.py first."
        )

    dataframe = pd.read_csv(DATASET_FILE)

    required_columns = (
        FEATURE_COLUMNS
        + [
            TARGET_COLUMN,
            "source_file",
            "load_hp",
            "rpm",
            "window_index",
        ]
    )

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
# SAVE ONE FOLD CONFUSION MATRIX
# ============================================================

def save_fold_confusion_matrix(
    y_true,
    y_pred,
    labels,
    test_load,
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
        f"CWRU Leave-One-Load-Out - Test {test_load} HP"
    )

    fig.tight_layout()

    output_file = (
        FIGURES_DIR
        / f"confusion_matrix_test_{test_load}hp.png"
    )

    fig.savefig(
        output_file,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    return matrix


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 110)
    print("CWRU LEAVE-ONE-LOAD-OUT EVALUATION")
    print("=" * 110)

    dataframe = load_dataset()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_classes = sorted(
        dataframe[TARGET_COLUMN]
        .unique()
        .tolist()
    )

    fold_results = []
    prediction_records = []

    aggregate_confusion = np.zeros(
        (
            len(all_classes),
            len(all_classes),
        ),
        dtype=int,
    )

    # ========================================================
    # ONE FOLD PER HELD-OUT LOAD
    # ========================================================

    for test_load in LOADS:
        train_loads = [
            load
            for load in LOADS
            if load != test_load
        ]

        train_df = dataframe[
            dataframe["load_hp"].isin(train_loads)
        ].copy()

        test_df = dataframe[
            dataframe["load_hp"] == test_load
        ].copy()

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df[TARGET_COLUMN]

        X_test = test_df[FEATURE_COLUMNS]
        y_test = test_df[TARGET_COLUMN]

        print()
        print("-" * 110)
        print(
            f"FOLD - TEST LOAD {test_load} HP"
        )
        print("-" * 110)

        print(
            "Training loads : "
            + ", ".join(
                f"{load} HP"
                for load in train_loads
            )
        )

        print(
            f"Test load      : {test_load} HP"
        )

        print(
            f"Training samples: {len(train_df)}"
        )

        print(
            f"Test samples    : {len(test_df)}"
        )

        print()
        print("Test source files:")

        print(
            test_df[
                [
                    "source_file",
                    TARGET_COLUMN,
                    "load_hp",
                    "rpm",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                by=TARGET_COLUMN
            )
            .to_string(
                index=False
            )
        )

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

        print()
        print(
            f"Accuracy : {accuracy:.4f}"
        )

        print(
            f"Macro F1 : {macro_f1:.4f}"
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

        matrix = save_fold_confusion_matrix(
            y_true=y_test,
            y_pred=predictions,
            labels=all_classes,
            test_load=test_load,
        )

        aggregate_confusion += matrix

        class_to_index = {
            class_name: index
            for index, class_name in enumerate(
                model.classes_
            )
        }

        confidence = np.max(
            probabilities,
            axis=1,
        )

        fold_result = {
            "test_load_hp": test_load,
            "train_loads": ",".join(
                str(load)
                for load in train_loads
            ),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "mean_confidence": float(
                np.mean(confidence)
            ),
            "min_confidence": float(
                np.min(confidence)
            ),
            "max_confidence": float(
                np.max(confidence)
            ),
        }

        fold_results.append(
            fold_result
        )

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
                "test_load_hp": test_load,
                "source_file": row["source_file"],
                "condition": row["condition"],
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


    # ========================================================
    # SUMMARY DATAFRAMES
    # ========================================================

    results_df = pd.DataFrame(
        fold_results
    )

    predictions_df = pd.DataFrame(
        prediction_records
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


    # ========================================================
    # AGGREGATE CONFUSION MATRIX
    # ========================================================

    display = ConfusionMatrixDisplay(
        confusion_matrix=aggregate_confusion,
        display_labels=all_classes,
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        "CWRU Leave-One-Load-Out - Aggregate Confusion Matrix"
    )

    fig.tight_layout()

    fig.savefig(
        AGGREGATE_CONFUSION_MATRIX_FILE,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 110)
    print("LEAVE-ONE-LOAD-OUT SUMMARY")
    print("=" * 110)

    print()

    print(
        results_df[
            [
                "test_load_hp",
                "accuracy",
                "macro_f1",
                "mean_confidence",
                "min_confidence",
                "max_confidence",
            ]
        ]
        .to_string(
            index=False
        )
    )

    mean_accuracy = float(
        results_df[
            "accuracy"
        ].mean()
    )

    std_accuracy = float(
        results_df[
            "accuracy"
        ].std(
            ddof=0
        )
    )

    mean_macro_f1 = float(
        results_df[
            "macro_f1"
        ].mean()
    )

    std_macro_f1 = float(
        results_df[
            "macro_f1"
        ].std(
            ddof=0
        )
    )

    overall_accuracy = float(
        predictions_df[
            "correct"
        ].mean()
    )

    print()
    print(
        f"Mean fold accuracy : "
        f"{mean_accuracy:.4f}"
    )

    print(
        f"Std fold accuracy  : "
        f"{std_accuracy:.4f}"
    )

    print(
        f"Mean fold Macro F1 : "
        f"{mean_macro_f1:.4f}"
    )

    print(
        f"Std fold Macro F1  : "
        f"{std_macro_f1:.4f}"
    )

    print(
        f"Overall accuracy   : "
        f"{overall_accuracy:.4f}"
    )


    # ========================================================
    # ERRORS
    # ========================================================

    errors = predictions_df[
        ~predictions_df[
            "correct"
        ]
    ]

    print()
    print("=" * 110)
    print("MISCLASSIFIED WINDOWS")
    print("=" * 110)

    print()

    if errors.empty:
        print(
            "None"
        )
    else:
        print(
            errors.to_string(
                index=False
            )
        )


    # ========================================================
    # CONFIDENCE BY TRUE CLASS
    # ========================================================

    print()
    print("=" * 110)
    print("CONFIDENCE BY TRUE CLASS - ALL HELD-OUT LOADS")
    print("=" * 110)

    print()

    confidence_by_class = (
        predictions_df
        .groupby(
            "condition"
        )[
            "prediction_confidence"
        ]
        .agg(
            [
                "mean",
                "min",
                "max",
            ]
        )
    )

    print(
        confidence_by_class.to_string()
    )


    # ========================================================
    # OUTPUT FILES
    # ========================================================

    print()
    print("=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    print()
    print("Fold results:")
    print(RESULTS_FILE)

    print()
    print("All predictions:")
    print(PREDICTIONS_FILE)

    print()
    print("Aggregate confusion matrix:")
    print(AGGREGATE_CONFUSION_MATRIX_FILE)

    print()
    print("Per-fold confusion matrices:")
    print(FIGURES_DIR)

    print()
    print("=" * 110)
    print("LEAVE-ONE-LOAD-OUT EVALUATION COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
