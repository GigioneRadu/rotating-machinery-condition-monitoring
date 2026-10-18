from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_multi_bearing_n15_m07_f10_dataset.csv"
)

HELDOUT_TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_stage1_heldout_test_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_stage1_healthy_domain_shift_diagnostic.csv"
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

HELDOUT_HEALTHY = {
    "K004",
    "K005",
}


# ============================================================
# FEATURES
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

def load_data():
    train_df = pd.read_csv(
        TRAIN_DATASET_FILE
    )

    test_df = pd.read_csv(
        HELDOUT_TEST_FILE
    )

    train_df = train_df[
        train_df[
            "bearing_code"
        ].isin(
            TRAIN_HEALTHY
            | TRAIN_DAMAGED
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

    heldout_healthy_df = test_df[
        test_df[
            "bearing_code"
        ].isin(
            HELDOUT_HEALTHY
        )
    ].copy()

    return (
        train_df,
        heldout_healthy_df,
    )


def centroid_dataframe(
    dataframe,
    feature_columns,
):
    return (
        dataframe
        .groupby(
            [
                "binary_condition",
                "bearing_code",
            ],
            as_index=False,
        )[
            feature_columns
        ]
        .mean()
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 130)
    print(
        "PADERBORN STAGE-1 HEALTHY DOMAIN-SHIFT DIAGNOSTIC"
    )
    print("=" * 130)

    train_df, test_df = load_data()

    print()
    print(
        f"Training bearings      : "
        f"{sorted(train_df['bearing_code'].unique())}"
    )

    print(
        f"Held-out healthy       : "
        f"{sorted(test_df['bearing_code'].unique())}"
    )

    all_records = []

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

        scaler = StandardScaler()

        train_scaled_values = scaler.fit_transform(
            train_df[
                feature_columns
            ]
        )

        test_scaled_values = scaler.transform(
            test_df[
                feature_columns
            ]
        )

        train_scaled = pd.DataFrame(
            train_scaled_values,
            columns=feature_columns,
            index=train_df.index,
        )

        train_scaled[
            "binary_condition"
        ] = train_df[
            "binary_condition"
        ].values

        train_scaled[
            "bearing_code"
        ] = train_df[
            "bearing_code"
        ].values

        test_scaled = pd.DataFrame(
            test_scaled_values,
            columns=feature_columns,
            index=test_df.index,
        )

        test_scaled[
            "binary_condition"
        ] = "healthy"

        test_scaled[
            "bearing_code"
        ] = test_df[
            "bearing_code"
        ].values

        train_bearing_centroids = (
            centroid_dataframe(
                train_scaled,
                feature_columns,
            )
        )

        train_class_centroids = (
            train_scaled
            .groupby(
                "binary_condition",
                as_index=False,
            )[
                feature_columns
            ]
            .mean()
        )

        test_bearing_centroids = (
            centroid_dataframe(
                test_scaled,
                feature_columns,
            )
        )

        train_bearing_matrix = (
            train_bearing_centroids[
                feature_columns
            ]
            .to_numpy()
        )

        train_bearing_codes = (
            train_bearing_centroids[
                "bearing_code"
            ]
            .tolist()
        )

        train_bearing_classes = (
            train_bearing_centroids[
                "binary_condition"
            ]
            .tolist()
        )

        class_matrix = (
            train_class_centroids[
                feature_columns
            ]
            .to_numpy()
        )

        class_names = (
            train_class_centroids[
                "binary_condition"
            ]
            .tolist()
        )

        for _, row in test_bearing_centroids.iterrows():
            vector = (
                row[
                    feature_columns
                ]
                .to_numpy(
                    dtype=float
                )
                .reshape(
                    1,
                    -1,
                )
            )

            bearing_distances = (
                pairwise_distances(
                    vector,
                    train_bearing_matrix,
                    metric="euclidean",
                )[0]
            )

            class_distances = (
                pairwise_distances(
                    vector,
                    class_matrix,
                    metric="euclidean",
                )[0]
            )

            nearest_index = int(
                np.argmin(
                    bearing_distances
                )
            )

            class_distance_map = {
                class_name: float(
                    class_distances[
                        index
                    ]
                )
                for (
                    index,
                    class_name,
                ) in enumerate(
                    class_names
                )
            }

            nearest_class_centroid = min(
                class_distance_map,
                key=class_distance_map.get,
            )

            record = {
                "feature_group": feature_group,
                "test_bearing": row[
                    "bearing_code"
                ],
                "true_class": "healthy",
                "nearest_train_bearing": train_bearing_codes[
                    nearest_index
                ],
                "nearest_train_class": train_bearing_classes[
                    nearest_index
                ],
                "nearest_bearing_distance": float(
                    bearing_distances[
                        nearest_index
                    ]
                ),
                "distance_to_healthy_centroid": class_distance_map[
                    "healthy"
                ],
                "distance_to_damaged_centroid": class_distance_map[
                    "damaged"
                ],
                "nearest_class_centroid": nearest_class_centroid,
                "healthy_centroid_closer": (
                    class_distance_map[
                        "healthy"
                    ]
                    < class_distance_map[
                        "damaged"
                    ]
                ),
            }

            all_records.append(
                record
            )

            print()
            print(
                f"{row['bearing_code']}"
                f" | nearest bearing="
                f"{train_bearing_codes[nearest_index]}"
                f" ({train_bearing_classes[nearest_index]})"
                f" | d="
                f"{bearing_distances[nearest_index]:.4f}"
            )

            print(
                f"    distance to HEALTHY centroid : "
                f"{class_distance_map['healthy']:.4f}"
            )

            print(
                f"    distance to DAMAGED centroid : "
                f"{class_distance_map['damaged']:.4f}"
            )

            print(
                f"    nearest class centroid       : "
                f"{nearest_class_centroid}"
            )

    results_df = pd.DataFrame(
        all_records
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 130)
    print("COMPACT SUMMARY")
    print("=" * 130)

    print()
    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("RAW MEAN FEATURE COMPARISON")
    print("=" * 130)

    raw_train_means = (
        train_df
        .groupby(
            [
                "binary_condition",
                "bearing_code",
            ]
        )[
            ALL_FEATURES
        ]
        .mean()
    )

    raw_test_means = (
        test_df
        .groupby(
            "bearing_code"
        )[
            ALL_FEATURES
        ]
        .mean()
    )

    print()
    print(
        "TRAIN:"
    )

    print(
        raw_train_means.to_string()
    )

    print()
    print(
        "HELD-OUT HEALTHY:"
    )

    print(
        raw_test_means.to_string()
    )

    print()
    print("=" * 130)
    print("SAVED")
    print("=" * 130)

    print()
    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 130)
    print(
        "HEALTHY DOMAIN-SHIFT DIAGNOSTIC COMPLETE"
    )
    print("=" * 130)


if __name__ == "__main__":
    main()
