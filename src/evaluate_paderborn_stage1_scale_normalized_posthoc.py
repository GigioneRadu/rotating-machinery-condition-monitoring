from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_multi_bearing_n15_m07_f10_dataset.csv"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_stage1_heldout_test_dataset.csv"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_stage1_scale_normalized_results.csv"
)

PER_BEARING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_stage1_scale_normalized_by_bearing.csv"
)


# ============================================================
# BEARING GROUPS
# ============================================================

TRAIN_HEALTHY = {
    "K001",
    "K002",
    "K003",
}

TRAIN_DAMAGED = {
    "KA01",
    "KA05",
    "KA07",
    "KI01",
    "KI05",
    "KI07",
}

CLASS_ORDER = [
    "healthy",
    "damaged",
]

EPS = 1e-12

RANDOM_STATE = 42
N_ESTIMATORS = 500


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_scale_normalized_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    df = dataframe.copy()

    # Amplitude-invariant time-domain features already available.
    df["log_crest_factor"] = np.log1p(
        np.clip(
            df["crest_factor"],
            a_min=0.0,
            a_max=None,
        )
    )

    df["log_kurtosis"] = np.log1p(
        np.clip(
            df["kurtosis"],
            a_min=0.0,
            a_max=None,
        )
    )

    # Normalize raw spectral amplitudes by broadband RMS.
    for column in [
        "shaft_amplitude",
        "bpfo_raw",
        "bpfi_raw",
        "bsf_raw",
        "ftf_raw",
    ]:
        df[
            f"{column}_over_rms"
        ] = (
            df[column]
            / (
                df["rms"]
                + EPS
            )
        )

    # Normalize envelope fault-frequency components by their
    # within-window total. This removes much of the absolute
    # envelope scale and keeps relative spectral composition.
    envelope_columns = [
        "bpfo_envelope",
        "bpfi_envelope",
        "bsf_envelope",
        "ftf_envelope",
    ]

    envelope_sum = (
        df[
            envelope_columns
        ]
        .sum(
            axis=1
        )
        + EPS
    )

    for column in envelope_columns:
        df[
            f"{column}_share"
        ] = (
            df[column]
            / envelope_sum
        )

    # Relative inner-vs-outer spectral indicators.
    df[
        "raw_bpfo_to_bpfi_ratio"
    ] = (
        df["bpfo_raw"]
        / (
            df["bpfi_raw"]
            + EPS
        )
    )

    df[
        "env_bpfo_to_bpfi_ratio"
    ] = (
        df["bpfo_envelope"]
        / (
            df["bpfi_envelope"]
            + EPS
        )
    )

    return df


FEATURE_GROUPS = {
    "shape_only": [
        "log_crest_factor",
        "log_kurtosis",
    ],
    "raw_over_rms": [
        "shaft_amplitude_over_rms",
        "bpfo_raw_over_rms",
        "bpfi_raw_over_rms",
        "bsf_raw_over_rms",
        "ftf_raw_over_rms",
    ],
    "envelope_composition": [
        "bpfo_envelope_share",
        "bpfi_envelope_share",
        "bsf_envelope_share",
        "ftf_envelope_share",
    ],
    "relative_physics": [
        "bpfo_raw_over_rms",
        "bpfi_raw_over_rms",
        "bsf_raw_over_rms",
        "ftf_raw_over_rms",
        "bpfo_envelope_share",
        "bpfi_envelope_share",
        "bsf_envelope_share",
        "ftf_envelope_share",
        "raw_bpfo_to_bpfi_ratio",
        "env_bpfo_to_bpfi_ratio",
    ],
    "all_scale_normalized": [
        "log_crest_factor",
        "log_kurtosis",
        "shaft_amplitude_over_rms",
        "bpfo_raw_over_rms",
        "bpfi_raw_over_rms",
        "bsf_raw_over_rms",
        "ftf_raw_over_rms",
        "bpfo_envelope_share",
        "bpfi_envelope_share",
        "bsf_envelope_share",
        "ftf_envelope_share",
        "raw_bpfo_to_bpfi_ratio",
        "env_bpfo_to_bpfi_ratio",
    ],
}


# ============================================================
# DATA
# ============================================================

def load_data():
    train_df = pd.read_csv(
        TRAIN_FILE
    )

    test_df = pd.read_csv(
        TEST_FILE
    )

    allowed_train = (
        TRAIN_HEALTHY
        | TRAIN_DAMAGED
    )

    train_df = train_df[
        train_df[
            "bearing_code"
        ].isin(
            allowed_train
        )
    ].copy()

    train_df[
        "binary_condition"
    ] = np.where(
        train_df[
            "bearing_code"
        ].isin(
            TRAIN_HEALTHY
        ),
        "healthy",
        "damaged",
    )

    train_df = add_scale_normalized_features(
        train_df
    )

    test_df = add_scale_normalized_features(
        test_df
    )

    return (
        train_df,
        test_df,
    )


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
# MAIN
# ============================================================

def main():
    print()
    print("=" * 130)
    print(
        "PADERBORN STAGE-1 POST-HOC SCALE-NORMALIZED FEATURE TEST"
    )
    print("=" * 130)

    print()
    print(
        "IMPORTANT: this is exploratory post-hoc analysis."
    )

    print(
        "K004/K005 and the held-out damaged bearings have already been inspected."
    )

    print(
        "Do NOT present this run as untouched final validation."
    )

    train_df, test_df = load_data()

    train_bearings = sorted(
        train_df[
            "bearing_code"
        ].unique()
    )

    test_bearings = sorted(
        test_df[
            "bearing_code"
        ].unique()
    )

    print()
    print(
        f"Train bearings: "
        f"{train_bearings}"
    )

    print(
        f"Test bearings : "
        f"{test_bearings}"
    )

    results = []
    bearing_rows = []

    for (
        feature_group,
        columns,
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
                columns
            ],
            train_df[
                "binary_condition"
            ],
        )

        predictions = model.predict(
            test_df[
                columns
            ]
        )

        probabilities = model.predict_proba(
            test_df[
                columns
            ]
        )

        confidence = np.max(
            probabilities,
            axis=1,
        )

        accuracy = accuracy_score(
            test_df[
                "binary_condition"
            ],
            predictions,
        )

        macro_f1 = f1_score(
            test_df[
                "binary_condition"
            ],
            predictions,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        )

        recalls = recall_score(
            test_df[
                "binary_condition"
            ],
            predictions,
            labels=CLASS_ORDER,
            average=None,
            zero_division=0,
        )

        recall_map = dict(
            zip(
                CLASS_ORDER,
                recalls,
            )
        )

        print()
        print(
            f"Accuracy       : "
            f"{accuracy:.4f}"
        )

        print(
            f"Macro F1       : "
            f"{macro_f1:.4f}"
        )

        print(
            f"Recall healthy : "
            f"{recall_map['healthy']:.4f}"
        )

        print(
            f"Recall damaged : "
            f"{recall_map['damaged']:.4f}"
        )

        print(
            f"Mean confidence: "
            f"{np.mean(confidence):.4f}"
        )

        print()
        print(
            classification_report(
                test_df[
                    "binary_condition"
                ],
                predictions,
                labels=CLASS_ORDER,
                digits=4,
                zero_division=0,
            )
        )

        results.append(
            {
                "feature_group": feature_group,
                "n_features": len(columns),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "recall_healthy": recall_map[
                    "healthy"
                ],
                "recall_damaged": recall_map[
                    "damaged"
                ],
                "mean_confidence": float(
                    np.mean(
                        confidence
                    )
                ),
            }
        )

        print(
            "Per-bearing:"
        )

        for bearing_code in test_bearings:
            mask = (
                test_df[
                    "bearing_code"
                ].to_numpy()
                == bearing_code
            )

            true_values = (
                test_df.loc[
                    mask,
                    "binary_condition",
                ]
                .to_numpy()
            )

            pred_values = predictions[
                mask
            ]

            conf_values = confidence[
                mask
            ]

            bearing_accuracy = accuracy_score(
                true_values,
                pred_values,
            )

            true_class = true_values[
                0
            ]

            print(
                f"  {bearing_code:<5}"
                f" | true={true_class:<7}"
                f" | accuracy="
                f"{bearing_accuracy:.4f}"
                f" | mean_conf="
                f"{np.mean(conf_values):.4f}"
            )

            bearing_rows.append(
                {
                    "feature_group": feature_group,
                    "bearing_code": bearing_code,
                    "binary_condition": true_class,
                    "samples": len(
                        true_values
                    ),
                    "accuracy": bearing_accuracy,
                    "mean_confidence": float(
                        np.mean(
                            conf_values
                        )
                    ),
                }
            )

    results_df = (
        pd.DataFrame(
            results
        )
        .sort_values(
            [
                "macro_f1",
                "accuracy",
            ],
            ascending=False,
        )
    )

    bearing_df = pd.DataFrame(
        bearing_rows
    )

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    bearing_df.to_csv(
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
    print("FILES SAVED")
    print("=" * 130)

    print()
    print(
        RESULTS_FILE
    )

    print(
        PER_BEARING_FILE
    )

    print()
    print("=" * 130)
    print(
        "SCALE-NORMALIZED POST-HOC TEST COMPLETE"
    )
    print("=" * 130)


if __name__ == "__main__":
    main()
