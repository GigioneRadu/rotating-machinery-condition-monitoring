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
from sklearn.model_selection import StratifiedGroupKFold


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_n15_m07_f10_dataset.csv"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_grouped_cv_results.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_grouped_cv_predictions.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "paderborn_grouped_cv"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
N_ESTIMATORS = 500
N_SPLITS = 5

TARGET_COLUMN = "condition"
GROUP_COLUMN = "source_file"


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

def load_dataset():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_FILE}"
        )

    dataframe = pd.read_csv(
        DATASET_FILE
    )

    required_columns = [
        TARGET_COLUMN,
        GROUP_COLUMN,
        "bearing_code",
        "measurement_index",
        "window_index",
        *ALL_FEATURES,
    ]

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    return dataframe


def create_model():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )


def save_aggregate_confusion_matrix(
    y_true,
    y_pred,
    labels,
    feature_group,
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
        f"Paderborn grouped 5-fold CV\n"
        f"{feature_group}"
    )

    fig.tight_layout()

    output_file = (
        FIGURES_DIR
        / f"confusion_{feature_group}.png"
    )

    fig.savefig(
        output_file,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    print()
    print("=" * 120)
    print("PADERBORN GROUPED 5-FOLD CROSS-VALIDATION")
    print("=" * 120)

    print()
    print(
        "Grouping is by source_file, so the four 1-second windows "
        "from the same 4-second recording never appear in both "
        "training and test sets."
    )

    dataframe = load_dataset()

    labels = sorted(
        dataframe[
            TARGET_COLUMN
        ]
        .unique()
        .tolist()
    )

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

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
            f"FEATURE GROUP: {feature_group}"
        )
        print("=" * 120)

        X = dataframe[
            feature_columns
        ]

        y = dataframe[
            TARGET_COLUMN
        ]

        groups = dataframe[
            GROUP_COLUMN
        ]

        fold_accuracies = []
        fold_macro_f1s = []

        aggregate_true = []
        aggregate_pred = []

        for fold_index, (
            train_index,
            test_index,
        ) in enumerate(
            splitter.split(
                X,
                y,
                groups,
            ),
            start=1,
        ):
            train_df = dataframe.iloc[
                train_index
            ].copy()

            test_df = dataframe.iloc[
                test_index
            ].copy()

            train_groups = set(
                train_df[
                    GROUP_COLUMN
                ]
            )

            test_groups = set(
                test_df[
                    GROUP_COLUMN
                ]
            )

            overlap = (
                train_groups
                & test_groups
            )

            if overlap:
                raise RuntimeError(
                    f"Group leakage detected in fold {fold_index}: "
                    f"{sorted(overlap)}"
                )

            model = create_model()

            model.fit(
                train_df[
                    feature_columns
                ],
                train_df[
                    TARGET_COLUMN
                ],
            )

            predictions = model.predict(
                test_df[
                    feature_columns
                ]
            )

            probabilities = model.predict_proba(
                test_df[
                    feature_columns
                ]
            )

            confidence = np.max(
                probabilities,
                axis=1,
            )

            accuracy = accuracy_score(
                test_df[
                    TARGET_COLUMN
                ],
                predictions,
            )

            macro_f1 = f1_score(
                test_df[
                    TARGET_COLUMN
                ],
                predictions,
                average="macro",
            )

            fold_accuracies.append(
                accuracy
            )

            fold_macro_f1s.append(
                macro_f1
            )

            aggregate_true.extend(
                test_df[
                    TARGET_COLUMN
                ].tolist()
            )

            aggregate_pred.extend(
                predictions.tolist()
            )

            test_sources = (
                test_df[
                    [
                        "source_file",
                        "condition",
                    ]
                ]
                .drop_duplicates()
                .sort_values(
                    [
                        "condition",
                        "source_file",
                    ]
                )
            )

            print()
            print(
                f"Fold {fold_index}"
                f" | accuracy={accuracy:.4f}"
                f" | macro_f1={macro_f1:.4f}"
                f" | test recordings={len(test_sources)}"
            )

            print(
                "  Test recordings by class:"
            )

            print(
                test_sources
                .groupby(
                    "condition"
                )
                .size()
                .to_string()
            )

            class_to_index = {
                class_name: index
                for index, class_name
                in enumerate(
                    model.classes_
                )
            }

            test_reset = (
                test_df
                .reset_index(
                    drop=True
                )
            )

            for row_index in range(
                len(test_reset)
            ):
                row = test_reset.iloc[
                    row_index
                ]

                record = {
                    "feature_group": feature_group,
                    "fold": fold_index,
                    "source_file": row[
                        "source_file"
                    ],
                    "bearing_code": row[
                        "bearing_code"
                    ],
                    "measurement_index": row[
                        "measurement_index"
                    ],
                    "window_index": row[
                        "window_index"
                    ],
                    "condition": row[
                        "condition"
                    ],
                    "predicted_condition": predictions[
                        row_index
                    ],
                    "correct": (
                        row[
                            "condition"
                        ]
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

                all_predictions.append(
                    record
                )

            all_results.append(
                {
                    "feature_group": feature_group,
                    "fold": fold_index,
                    "n_features": len(
                        feature_columns
                    ),
                    "train_samples": len(
                        train_df
                    ),
                    "test_samples": len(
                        test_df
                    ),
                    "train_recordings": train_df[
                        GROUP_COLUMN
                    ].nunique(),
                    "test_recordings": test_df[
                        GROUP_COLUMN
                    ].nunique(),
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
            )

        mean_accuracy = float(
            np.mean(
                fold_accuracies
            )
        )

        std_accuracy = float(
            np.std(
                fold_accuracies
            )
        )

        mean_macro_f1 = float(
            np.mean(
                fold_macro_f1s
            )
        )

        std_macro_f1 = float(
            np.std(
                fold_macro_f1s
            )
        )

        print()
        print(
            f"Mean accuracy : "
            f"{mean_accuracy:.4f}"
            f" ± {std_accuracy:.4f}"
        )

        print(
            f"Mean Macro F1 : "
            f"{mean_macro_f1:.4f}"
            f" ± {std_macro_f1:.4f}"
        )

        print()
        print(
            "Aggregate classification report:"
        )

        print(
            classification_report(
                aggregate_true,
                aggregate_pred,
                labels=labels,
                digits=4,
                zero_division=0,
            )
        )

        save_aggregate_confusion_matrix(
            y_true=aggregate_true,
            y_pred=aggregate_pred,
            labels=labels,
            feature_group=feature_group,
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

    summary = (
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
            std_accuracy=(
                "accuracy",
                "std",
            ),
            min_accuracy=(
                "accuracy",
                "min",
            ),
            mean_macro_f1=(
                "macro_f1",
                "mean",
            ),
            std_macro_f1=(
                "macro_f1",
                "std",
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
            [
                "mean_accuracy",
                "mean_macro_f1",
            ],
            ascending=False,
        )
    )

    print()
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)

    print()
    print(
        summary.to_string(
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
    print("PADERBORN GROUPED CV COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
