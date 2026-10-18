from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_artificial_real_feature_distances.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_BEARINGS = {
    "KA01",
    "KA05",
    "KA07",
    "KI01",
    "KI05",
    "KI07",
}

TEST_BEARINGS = {
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

def load_data():
    artificial_df = pd.read_csv(
        ARTIFICIAL_DATASET_FILE
    )

    real_df = pd.read_csv(
        REAL_DAMAGE_DATASET_FILE
    )

    train_df = artificial_df[
        artificial_df[
            "bearing_code"
        ].isin(TRAIN_BEARINGS)
    ].copy()

    test_df = real_df[
        real_df[
            "bearing_code"
        ].isin(TEST_BEARINGS)
    ].copy()

    return train_df, test_df


def calculate_centroids(
    dataframe,
    feature_columns,
):
    return (
        dataframe
        .groupby(
            [
                "condition",
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
    print("PADERBORN ARTIFICIAL -> REAL DAMAGE FEATURE-DISTANCE ANALYSIS")
    print("=" * 130)

    train_df, test_df = load_data()

    print()
    print(
        f"Artificial training bearings: "
        f"{sorted(train_df['bearing_code'].unique())}"
    )

    print(
        f"Real-damage test bearings   : "
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

        # Fit scaling ONLY on artificial training data.
        scaler = StandardScaler()

        train_scaled = scaler.fit_transform(
            train_df[
                feature_columns
            ]
        )

        test_scaled = scaler.transform(
            test_df[
                feature_columns
            ]
        )

        train_scaled_df = pd.DataFrame(
            train_scaled,
            columns=feature_columns,
            index=train_df.index,
        )

        train_scaled_df[
            "condition"
        ] = train_df[
            "condition"
        ].values

        train_scaled_df[
            "bearing_code"
        ] = train_df[
            "bearing_code"
        ].values

        test_scaled_df = pd.DataFrame(
            test_scaled,
            columns=feature_columns,
            index=test_df.index,
        )

        test_scaled_df[
            "condition"
        ] = test_df[
            "condition"
        ].values

        test_scaled_df[
            "bearing_code"
        ] = test_df[
            "bearing_code"
        ].values

        train_centroids = calculate_centroids(
            train_scaled_df,
            feature_columns,
        )

        test_centroids = calculate_centroids(
            test_scaled_df,
            feature_columns,
        )

        # Class centroids from artificial training set.
        train_class_centroids = (
            train_scaled_df
            .groupby(
                "condition",
                as_index=False,
            )[
                feature_columns
            ]
            .mean()
        )

        train_bearing_matrix = (
            train_centroids[
                feature_columns
            ]
            .to_numpy()
        )

        train_bearing_codes = (
            train_centroids[
                "bearing_code"
            ]
            .tolist()
        )

        train_bearing_classes = (
            train_centroids[
                "condition"
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
                "condition"
            ]
            .tolist()
        )

        for _, row in test_centroids.iterrows():
            test_vector = (
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
                    test_vector,
                    train_bearing_matrix,
                    metric="euclidean",
                )[0]
            )

            class_distances = (
                pairwise_distances(
                    test_vector,
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
                for index, class_name
                in enumerate(
                    class_names
                )
            }

            predicted_by_nearest_class_centroid = min(
                class_distance_map,
                key=class_distance_map.get,
            )

            record = {
                "feature_group": feature_group,
                "test_bearing": row[
                    "bearing_code"
                ],
                "true_class": row[
                    "condition"
                ],
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
                "distance_to_inner_class_centroid": class_distance_map.get(
                    "inner_race",
                    np.nan,
                ),
                "distance_to_outer_class_centroid": class_distance_map.get(
                    "outer_race",
                    np.nan,
                ),
                "nearest_class_centroid": predicted_by_nearest_class_centroid,
                "class_centroid_correct": (
                    predicted_by_nearest_class_centroid
                    == row[
                        "condition"
                    ]
                ),
            }

            all_records.append(
                record
            )

            print()
            print(
                f"{row['bearing_code']:<5}"
                f" | true={row['condition']:<10}"
                f" | nearest bearing="
                f"{train_bearing_codes[nearest_index]:<5}"
                f" ({train_bearing_classes[nearest_index]:<10})"
                f" | d={bearing_distances[nearest_index]:.4f}"
            )

            print(
                f"      distance to INNER class centroid: "
                f"{class_distance_map.get('inner_race', np.nan):.4f}"
            )

            print(
                f"      distance to OUTER class centroid: "
                f"{class_distance_map.get('outer_race', np.nan):.4f}"
            )

            print(
                f"      nearest class centroid           : "
                f"{predicted_by_nearest_class_centroid}"
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
        results_df[
            [
                "feature_group",
                "test_bearing",
                "true_class",
                "nearest_train_bearing",
                "nearest_train_class",
                "nearest_bearing_distance",
                "distance_to_inner_class_centroid",
                "distance_to_outer_class_centroid",
                "nearest_class_centroid",
                "class_centroid_correct",
            ]
        ]
        .to_string(
            index=False
        )
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
    print("FEATURE-DISTANCE ANALYSIS COMPLETE")
    print("=" * 130)


if __name__ == "__main__":
    main()
