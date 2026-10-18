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
    recall_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFICIAL_DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_multi_bearing_n15_m07_f10_dataset.csv"
)

REAL_DAMAGE_DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_real_damage_n15_m07_f10_dataset.csv"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_artificial_to_real_damage_results.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_artificial_to_real_damage_predictions.csv"
)

PER_BEARING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_artificial_to_real_damage_by_bearing.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "paderborn_artificial_to_real_damage"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
N_ESTIMATORS = 500

TARGET_COLUMN = "condition"
BEARING_COLUMN = "bearing_code"

CLASS_ORDER = [
    "inner_race",
    "outer_race",
]

ARTIFICIAL_TRAIN_BEARINGS = {
    "KA01",
    "KA05",
    "KA07",
    "KI01",
    "KI05",
    "KI07",
}

REAL_TEST_BEARINGS = {
    "KA04",
    "KA16",
    "KI04",
    "KI14",
}


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

def load_datasets():
    if not ARTIFICIAL_DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Artificial-damage dataset not found:\n"
            f"{ARTIFICIAL_DATASET_FILE}"
        )

    if not REAL_DAMAGE_DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Real-damage dataset not found:\n"
            f"{REAL_DAMAGE_DATASET_FILE}"
        )

    artificial_df = pd.read_csv(
        ARTIFICIAL_DATASET_FILE
    )

    real_df = pd.read_csv(
        REAL_DAMAGE_DATASET_FILE
    )

    required_columns = [
        TARGET_COLUMN,
        BEARING_COLUMN,
        "source_file",
        "window_index",
        *ALL_FEATURES,
    ]

    for name, dataframe in [
        ("artificial", artificial_df),
        ("real", real_df),
    ]:
        missing = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing:
            raise ValueError(
                f"{name} dataset is missing columns: {missing}"
            )

    train_df = artificial_df[
        artificial_df[
            BEARING_COLUMN
        ].isin(
            ARTIFICIAL_TRAIN_BEARINGS
        )
    ].copy()

    test_df = real_df[
        real_df[
            BEARING_COLUMN
        ].isin(
            REAL_TEST_BEARINGS
        )
    ].copy()

    train_df = train_df[
        train_df[
            TARGET_COLUMN
        ].isin(
            CLASS_ORDER
        )
    ].copy()

    test_df = test_df[
        test_df[
            TARGET_COLUMN
        ].isin(
            CLASS_ORDER
        )
    ].copy()

    return train_df, test_df


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
    feature_group,
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
        figsize=(8, 6)
    )

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        "Paderborn artificial damage → real damage\n"
        f"{feature_group}"
    )

    fig.tight_layout()

    output_file = (
        FIGURES_DIR
        / f"confusion_{feature_group}.png"
    )

    fig.savefig(
        output_file,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 130)
    print("PADERBORN ARTIFICIAL-DAMAGE -> REAL-DAMAGE TRANSFER")
    print("=" * 130)

    print()
    print(
        "TRAIN: artificial-damage bearings only"
    )

    print(
        "  outer: KA01, KA05, KA07"
    )

    print(
        "  inner: KI01, KI05, KI07"
    )

    print()
    print(
        "TEST: accelerated-lifetime real-damage bearings only"
    )

    print(
        "  outer: KA04, KA16"
    )

    print(
        "  inner: KI04, KI14"
    )

    train_df, test_df = load_datasets()

    train_bearings = set(
        train_df[
            BEARING_COLUMN
        ].unique()
    )

    test_bearings = set(
        test_df[
            BEARING_COLUMN
        ].unique()
    )

    overlap = (
        train_bearings
        & test_bearings
    )

    if overlap:
        raise RuntimeError(
            f"Bearing leakage detected: {sorted(overlap)}"
        )

    print()
    print(
        f"Training rows      : {len(train_df)}"
    )

    print(
        f"Training bearings  : "
        f"{sorted(train_bearings)}"
    )

    print(
        f"Test rows          : {len(test_df)}"
    )

    print(
        f"Test bearings      : "
        f"{sorted(test_bearings)}"
    )

    print()
    print(
        "Training class counts:"
    )

    print(
        train_df
        .groupby(
            TARGET_COLUMN
        )
        .size()
        .to_string()
    )

    print()
    print(
        "Test class counts:"
    )

    print(
        test_df
        .groupby(
            TARGET_COLUMN
        )
        .size()
        .to_string()
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []
    predictions_records = []
    per_bearing_records = []

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

        results.append(
            {
                "feature_group": feature_group,
                "n_features": len(
                    feature_columns
                ),
                "train_samples": len(
                    train_df
                ),
                "test_samples": len(
                    test_df
                ),
                "train_bearings": len(
                    train_bearings
                ),
                "test_bearings": len(
                    test_bearings
                ),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "recall_inner_race": recall_map[
                    "inner_race"
                ],
                "recall_outer_race": recall_map[
                    "outer_race"
                ],
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

        print()
        print(
            f"Accuracy          : "
            f"{accuracy:.4f}"
        )

        print(
            f"Macro F1          : "
            f"{macro_f1:.4f}"
        )

        print(
            f"Recall inner_race : "
            f"{recall_map['inner_race']:.4f}"
        )

        print(
            f"Recall outer_race : "
            f"{recall_map['outer_race']:.4f}"
        )

        print(
            f"Mean confidence   : "
            f"{np.mean(confidence):.4f}"
        )

        print(
            f"Min confidence    : "
            f"{np.min(confidence):.4f}"
        )

        print()
        print(
            "Classification report:"
        )

        print(
            classification_report(
                test_df[
                    TARGET_COLUMN
                ],
                predictions,
                labels=CLASS_ORDER,
                digits=4,
                zero_division=0,
            )
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

            record = {
                "feature_group": feature_group,
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
                record[
                    f"probability_{class_name}"
                ] = probabilities[
                    row_index,
                    class_to_index[
                        class_name
                    ],
                ]

            predictions_records.append(
                record
            )

        for bearing_code in sorted(
            test_bearings
        ):
            mask = (
                test_df[
                    BEARING_COLUMN
                ].to_numpy()
                == bearing_code
            )

            bearing_true = (
                test_df.loc[
                    test_df[
                        BEARING_COLUMN
                    ] == bearing_code,
                    TARGET_COLUMN,
                ]
                .to_numpy()
            )

            bearing_pred = predictions[
                mask
            ]

            bearing_confidence = confidence[
                mask
            ]

            bearing_accuracy = accuracy_score(
                bearing_true,
                bearing_pred,
            )

            true_class = bearing_true[
                0
            ]

            per_bearing_records.append(
                {
                    "feature_group": feature_group,
                    "bearing_code": bearing_code,
                    "condition": true_class,
                    "samples": len(
                        bearing_true
                    ),
                    "accuracy": bearing_accuracy,
                    "mean_confidence": float(
                        np.mean(
                            bearing_confidence
                        )
                    ),
                    "min_confidence": float(
                        np.min(
                            bearing_confidence
                        )
                    ),
                }
            )

            print(
                f"  {bearing_code:<5}"
                f" | true={true_class:<10}"
                f" | accuracy={bearing_accuracy:.4f}"
                f" | mean_conf={np.mean(bearing_confidence):.4f}"
            )

        save_confusion_matrix(
            y_true=test_df[
                TARGET_COLUMN
            ],
            y_pred=predictions,
            feature_group=feature_group,
        )

    results_df = pd.DataFrame(
        results
    )

    predictions_df = pd.DataFrame(
        predictions_records
    )

    per_bearing_df = pd.DataFrame(
        per_bearing_records
    )

    results_df = results_df.sort_values(
        [
            "macro_f1",
            "accuracy",
        ],
        ascending=False,
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    per_bearing_df.to_csv(
        PER_BEARING_FILE,
        index=False,
    )

    print()
    print("=" * 130)
    print("OVERALL SUMMARY")
    print("=" * 130)

    print()
    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("PER-BEARING SUMMARY")
    print("=" * 130)

    print()
    print(
        per_bearing_df
        .sort_values(
            [
                "feature_group",
                "condition",
                "bearing_code",
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("FILES SAVED")
    print("=" * 130)

    print()
    print(
        RESULTS_FILE
    )

    print(
        PREDICTIONS_FILE
    )

    print(
        PER_BEARING_FILE
    )

    print(
        FIGURES_DIR
    )

    print()
    print("=" * 130)
    print("ARTIFICIAL -> REAL DAMAGE TRANSFER COMPLETE")
    print("=" * 130)


if __name__ == "__main__":
    main()
