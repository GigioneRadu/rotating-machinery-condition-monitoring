from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    recall_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_multi_bearing_n15_m07_f10_dataset.csv"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_leave_one_bearing_per_class_results.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_leave_one_bearing_per_class_predictions.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_leave_one_bearing_per_class_summary.csv"
)

BEARING_SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_leave_one_bearing_per_class_by_test_bearing.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "paderborn_leave_one_bearing_per_class"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
N_ESTIMATORS = 500

TARGET_COLUMN = "condition"
BEARING_COLUMN = "bearing_code"

CLASS_ORDER = [
    "healthy",
    "inner_race",
    "outer_race",
]

HEALTHY_BEARINGS = [
    "K001",
    "K002",
    "K003",
]

INNER_BEARINGS = [
    "KI01",
    "KI05",
    "KI07",
]

OUTER_BEARINGS = [
    "KA01",
    "KA05",
    "KA07",
]


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
        BEARING_COLUMN,
        "source_file",
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


def verify_bearing_labels(
    dataframe: pd.DataFrame,
):
    expected = {
        **{
            bearing: "healthy"
            for bearing in HEALTHY_BEARINGS
        },
        **{
            bearing: "inner_race"
            for bearing in INNER_BEARINGS
        },
        **{
            bearing: "outer_race"
            for bearing in OUTER_BEARINGS
        },
    }

    for bearing, expected_class in expected.items():
        rows = dataframe[
            dataframe[
                BEARING_COLUMN
            ] == bearing
        ]

        if rows.empty:
            raise ValueError(
                f"Dataset contains no rows for bearing {bearing}."
            )

        actual_classes = sorted(
            rows[
                TARGET_COLUMN
            ]
            .unique()
            .tolist()
        )

        if actual_classes != [
            expected_class
        ]:
            raise ValueError(
                f"{bearing}: expected class {expected_class!r}, "
                f"found {actual_classes}."
            )


def save_confusion_matrix(
    y_true,
    y_pred,
    feature_group: str,
):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_ORDER,
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        "Paderborn leave-one-bearing-per-class-out\n"
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
# MAIN
# ============================================================

def main():
    print()
    print("=" * 130)
    print("PADERBORN LEAVE-ONE-BEARING-PER-CLASS-OUT")
    print("=" * 130)

    print()
    print(
        "Each fold holds out ONE physical bearing from EACH class."
    )

    print(
        "With 3 healthy x 3 inner-race x 3 outer-race bearings, "
        "this creates 27 balanced bearing-level test combinations."
    )

    print()
    print(
        "No windows or recordings from any held-out bearing are used "
        "during training."
    )

    dataframe = load_dataset()

    verify_bearing_labels(
        dataframe
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_combinations = list(
        product(
            HEALTHY_BEARINGS,
            INNER_BEARINGS,
            OUTER_BEARINGS,
        )
    )

    print()
    print(
        f"Dataset rows      : {len(dataframe)}"
    )

    print(
        f"Physical bearings : "
        f"{dataframe[BEARING_COLUMN].nunique()}"
    )

    print(
        f"Test combinations : "
        f"{len(test_combinations)}"
    )

    all_results = []
    all_predictions = []

    for (
        feature_group,
        feature_columns,
    ) in FEATURE_GROUPS.items():
        print()
        print("=" * 130)
        print(
            f"FEATURE GROUP: {feature_group}"
        )
        print("=" * 130)

        feature_true = []
        feature_pred = []

        for fold_index, (
            healthy_test,
            inner_test,
            outer_test,
        ) in enumerate(
            test_combinations,
            start=1,
        ):
            test_bearings = {
                healthy_test,
                inner_test,
                outer_test,
            }

            test_mask = (
                dataframe[
                    BEARING_COLUMN
                ]
                .isin(
                    test_bearings
                )
            )

            train_df = dataframe[
                ~test_mask
            ].copy()

            test_df = dataframe[
                test_mask
            ].copy()

            train_bearings = set(
                train_df[
                    BEARING_COLUMN
                ]
                .unique()
            )

            overlap = (
                train_bearings
                & test_bearings
            )

            if overlap:
                raise RuntimeError(
                    f"Bearing leakage in fold {fold_index}: "
                    f"{sorted(overlap)}"
                )

            train_classes = set(
                train_df[
                    TARGET_COLUMN
                ]
                .unique()
            )

            if train_classes != set(
                CLASS_ORDER
            ):
                raise RuntimeError(
                    f"Fold {fold_index}: training set does not "
                    f"contain all three classes."
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

            probabilities = (
                model.predict_proba(
                    test_df[
                        feature_columns
                    ]
                )
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
                labels=CLASS_ORDER,
                average="macro",
                zero_division=0,
            )

            class_recalls = recall_score(
                test_df[
                    TARGET_COLUMN
                ],
                predictions,
                labels=CLASS_ORDER,
                average=None,
                zero_division=0,
            )

            recall_map = dict(
                zip(
                    CLASS_ORDER,
                    class_recalls,
                )
            )

            feature_true.extend(
                test_df[
                    TARGET_COLUMN
                ].tolist()
            )

            feature_pred.extend(
                predictions.tolist()
            )

            all_results.append(
                {
                    "feature_group": feature_group,
                    "fold": fold_index,
                    "test_healthy_bearing": healthy_test,
                    "test_inner_bearing": inner_test,
                    "test_outer_bearing": outer_test,
                    "n_features": len(
                        feature_columns
                    ),
                    "train_samples": len(
                        train_df
                    ),
                    "test_samples": len(
                        test_df
                    ),
                    "train_bearings": train_df[
                        BEARING_COLUMN
                    ].nunique(),
                    "test_bearings": test_df[
                        BEARING_COLUMN
                    ].nunique(),
                    "accuracy": accuracy,
                    "macro_f1": macro_f1,
                    "recall_healthy": recall_map[
                        "healthy"
                    ],
                    "recall_inner_race": recall_map[
                        "inner_race"
                    ],
                    "recall_outer_race": recall_map[
                        "outer_race"
                    ],
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
            )

            class_to_index = {
                class_name: index
                for index, class_name
                in enumerate(
                    model.classes_
                )
            }

            test_reset = test_df.reset_index(
                drop=True
            )

            for row_index in range(
                len(test_reset)
            ):
                row = test_reset.iloc[
                    row_index
                ]

                prediction_record = {
                    "feature_group": feature_group,
                    "fold": fold_index,
                    "test_healthy_bearing": healthy_test,
                    "test_inner_bearing": inner_test,
                    "test_outer_bearing": outer_test,
                    "bearing_code": row[
                        BEARING_COLUMN
                    ],
                    "condition": row[
                        TARGET_COLUMN
                    ],
                    "source_file": row[
                        "source_file"
                    ],
                    "window_index": row[
                        "window_index"
                    ],
                    "predicted_condition": predictions[
                        row_index
                    ],
                    "correct": (
                        row[
                            TARGET_COLUMN
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
                    prediction_record[
                        f"probability_{class_name}"
                    ] = probabilities[
                        row_index,
                        class_to_index[
                            class_name
                        ],
                    ]

                all_predictions.append(
                    prediction_record
                )

            print(
                f"Fold {fold_index:02d}"
                f" | test="
                f"{healthy_test}/{inner_test}/{outer_test}"
                f" | acc={accuracy:.4f}"
                f" | F1={macro_f1:.4f}"
                f" | recalls="
                f"H:{recall_map['healthy']:.3f} "
                f"I:{recall_map['inner_race']:.3f} "
                f"O:{recall_map['outer_race']:.3f}"
                f" | conf={np.mean(confidence):.3f}"
            )

        save_confusion_matrix(
            y_true=feature_true,
            y_pred=feature_pred,
            feature_group=feature_group,
        )

    results_df = pd.DataFrame(
        all_results
    )

    predictions_df = pd.DataFrame(
        all_predictions
    )

    summary_df = (
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
            max_accuracy=(
                "accuracy",
                "max",
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
            mean_recall_healthy=(
                "recall_healthy",
                "mean",
            ),
            mean_recall_inner_race=(
                "recall_inner_race",
                "mean",
            ),
            mean_recall_outer_race=(
                "recall_outer_race",
                "mean",
            ),
            mean_confidence=(
                "mean_confidence",
                "mean",
            ),
            min_confidence_seen=(
                "min_confidence",
                "min",
            ),
        )
        .sort_values(
            [
                "mean_macro_f1",
                "mean_accuracy",
            ],
            ascending=False,
        )
    )

    # Per-bearing performance.
    predictions_df[
        "correct_int"
    ] = (
        predictions_df[
            "correct"
        ]
        .astype(int)
    )

    bearing_summary_df = (
        predictions_df
        .groupby(
            [
                "feature_group",
                "condition",
                "bearing_code",
            ],
            as_index=False,
        )
        .agg(
            evaluated_windows=(
                "correct_int",
                "size",
            ),
            accuracy=(
                "correct_int",
                "mean",
            ),
            mean_confidence=(
                "prediction_confidence",
                "mean",
            ),
        )
        .sort_values(
            [
                "feature_group",
                "condition",
                "bearing_code",
            ]
        )
    )

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    predictions_df.drop(
        columns=[
            "correct_int"
        ]
    ).to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    bearing_summary_df.to_csv(
        BEARING_SUMMARY_FILE,
        index=False,
    )

    print()
    print("=" * 130)
    print("OVERALL SUMMARY")
    print("=" * 130)

    print()
    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("PER-BEARING SUMMARY")
    print("=" * 130)

    print()
    print(
        bearing_summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("FILES SAVED")
    print("=" * 130)

    print()
    print(
        f"Fold results:\n{RESULTS_FILE}"
    )

    print()
    print(
        f"Predictions:\n{PREDICTIONS_FILE}"
    )

    print()
    print(
        f"Overall summary:\n{SUMMARY_FILE}"
    )

    print()
    print(
        f"Per-bearing summary:\n{BEARING_SUMMARY_FILE}"
    )

    print()
    print(
        f"Confusion matrices:\n{FIGURES_DIR}"
    )

    print()
    print("=" * 130)
    print("PADERBORN BEARING-LEVEL VALIDATION COMPLETE")
    print("=" * 130)


if __name__ == "__main__":
    main()
