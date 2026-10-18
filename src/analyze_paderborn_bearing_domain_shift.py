from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "paderborn_bearing_domain_shift"
)

BEARING_MEANS_FILE = (
    OUTPUT_DIR
    / "paderborn_bearing_feature_means.csv"
)

STANDARDIZED_BEARING_MEANS_FILE = (
    OUTPUT_DIR
    / "paderborn_bearing_feature_means_standardized.csv"
)

DISTANCES_FILE = (
    OUTPUT_DIR
    / "paderborn_bearing_centroid_distances.csv"
)

PCA_FILE = (
    OUTPUT_DIR
    / "paderborn_bearing_centroid_pca.csv"
)


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


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 120)
    print("PADERBORN BEARING DOMAIN-SHIFT DIAGNOSTIC")
    print("=" * 120)

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_FILE}"
        )

    dataframe = pd.read_csv(
        DATASET_FILE
    )

    required_columns = [
        "bearing_code",
        "condition",
        *ALL_FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Per-bearing mean feature profiles
    # --------------------------------------------------------

    bearing_means = (
        dataframe
        .groupby(
            [
                "condition",
                "bearing_code",
            ],
            as_index=False,
        )[
            ALL_FEATURES
        ]
        .mean()
    )

    bearing_means.to_csv(
        BEARING_MEANS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Standardize using all WINDOW-LEVEL samples.
    # Then compute standardized bearing centroids.
    # --------------------------------------------------------

    scaler = StandardScaler()

    standardized_windows = scaler.fit_transform(
        dataframe[
            ALL_FEATURES
        ]
    )

    standardized_df = pd.DataFrame(
        standardized_windows,
        columns=ALL_FEATURES,
        index=dataframe.index,
    )

    standardized_df[
        "condition"
    ] = dataframe[
        "condition"
    ].values

    standardized_df[
        "bearing_code"
    ] = dataframe[
        "bearing_code"
    ].values

    standardized_bearing_means = (
        standardized_df
        .groupby(
            [
                "condition",
                "bearing_code",
            ],
            as_index=False,
        )[
            ALL_FEATURES
        ]
        .mean()
    )

    standardized_bearing_means.to_csv(
        STANDARDIZED_BEARING_MEANS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Pairwise distances between physical bearing centroids
    # --------------------------------------------------------

    centroid_matrix = (
        standardized_bearing_means[
            ALL_FEATURES
        ]
        .to_numpy()
    )

    bearing_codes = (
        standardized_bearing_means[
            "bearing_code"
        ]
        .tolist()
    )

    conditions = (
        standardized_bearing_means[
            "condition"
        ]
        .tolist()
    )

    distances = pairwise_distances(
        centroid_matrix,
        metric="euclidean",
    )

    distance_records = []

    for i in range(
        len(bearing_codes)
    ):
        for j in range(
            i + 1,
            len(bearing_codes),
        ):
            same_class = (
                conditions[i]
                == conditions[j]
            )

            distance_records.append(
                {
                    "bearing_a": bearing_codes[i],
                    "class_a": conditions[i],
                    "bearing_b": bearing_codes[j],
                    "class_b": conditions[j],
                    "same_class": same_class,
                    "distance": float(
                        distances[i, j]
                    ),
                }
            )

    distance_df = pd.DataFrame(
        distance_records
    )

    distance_df.to_csv(
        DISTANCES_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # PCA of standardized bearing centroids
    # --------------------------------------------------------

    pca = PCA(
        n_components=2,
    )

    pca_values = pca.fit_transform(
        centroid_matrix
    )

    pca_df = pd.DataFrame(
        {
            "bearing_code": bearing_codes,
            "condition": conditions,
            "pc1": pca_values[:, 0],
            "pc2": pca_values[:, 1],
        }
    )

    pca_df.to_csv(
        PCA_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Plot 1: PCA bearing centroids
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    for condition in sorted(
        pca_df[
            "condition"
        ].unique()
    ):
        subset = pca_df[
            pca_df[
                "condition"
            ] == condition
        ]

        ax.scatter(
            subset[
                "pc1"
            ],
            subset[
                "pc2"
            ],
            label=condition,
            s=90,
        )

        for _, row in subset.iterrows():
            ax.annotate(
                row[
                    "bearing_code"
                ],
                (
                    row[
                        "pc1"
                    ],
                    row[
                        "pc2"
                    ],
                ),
                xytext=(
                    5,
                    5,
                ),
                textcoords="offset points",
            )

    ax.set_title(
        "Paderborn physical-bearing centroids\n"
        "PCA of standardized 14-feature means"
    )

    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)"
    )

    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)"
    )

    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "bearing_centroids_pca.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Plot 2: standardized feature profiles
    # --------------------------------------------------------

    profile_df = (
        standardized_bearing_means
        .set_index(
            "bearing_code"
        )[
            ALL_FEATURES
        ]
    )

    fig, ax = plt.subplots(
        figsize=(15, 8)
    )

    image = ax.imshow(
        profile_df.to_numpy(),
        aspect="auto",
    )

    ax.set_title(
        "Standardized mean feature profile by physical bearing"
    )

    ax.set_yticks(
        np.arange(
            len(
                profile_df.index
            )
        )
    )

    ax.set_yticklabels(
        profile_df.index
    )

    ax.set_xticks(
        np.arange(
            len(
                ALL_FEATURES
            )
        )
    )

    ax.set_xticklabels(
        ALL_FEATURES,
        rotation=70,
        ha="right",
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Standardized bearing mean",
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "bearing_standardized_feature_profiles.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Console summaries
    # --------------------------------------------------------

    print()
    print(
        "PER-BEARING STANDARDIZED CENTROIDS"
    )
    print("-" * 120)

    print(
        standardized_bearing_means.to_string(
            index=False
        )
    )

    print()
    print(
        "PAIRWISE CENTROID DISTANCES"
    )
    print("-" * 120)

    print(
        distance_df
        .sort_values(
            "distance"
        )
        .to_string(
            index=False
        )
    )

    within_class = distance_df[
        distance_df[
            "same_class"
        ]
    ]

    between_class = distance_df[
        ~distance_df[
            "same_class"
        ]
    ]

    print()
    print(
        "DISTANCE SUMMARY"
    )
    print("-" * 120)

    print(
        f"Mean within-class distance : "
        f"{within_class['distance'].mean():.4f}"
    )

    print(
        f"Min within-class distance  : "
        f"{within_class['distance'].min():.4f}"
    )

    print(
        f"Max within-class distance  : "
        f"{within_class['distance'].max():.4f}"
    )

    print()

    print(
        f"Mean between-class distance: "
        f"{between_class['distance'].mean():.4f}"
    )

    print(
        f"Min between-class distance : "
        f"{between_class['distance'].min():.4f}"
    )

    print(
        f"Max between-class distance : "
        f"{between_class['distance'].max():.4f}"
    )

    print()
    print(
        "NEAREST OTHER BEARING FOR EACH PHYSICAL BEARING"
    )
    print("-" * 120)

    for i, bearing in enumerate(
        bearing_codes
    ):
        row = distances[
            i
        ].copy()

        row[
            i
        ] = np.inf

        nearest_index = int(
            np.argmin(
                row
            )
        )

        print(
            f"{bearing:<5}"
            f" ({conditions[i]:<10})"
            f" -> {bearing_codes[nearest_index]:<5}"
            f" ({conditions[nearest_index]:<10})"
            f" | distance={row[nearest_index]:.4f}"
        )

    print()
    print(
        "PCA EXPLAINED VARIANCE"
    )
    print("-" * 120)

    print(
        f"PC1: "
        f"{pca.explained_variance_ratio_[0] * 100:.2f}%"
    )

    print(
        f"PC2: "
        f"{pca.explained_variance_ratio_[1] * 100:.2f}%"
    )

    print()
    print("=" * 120)
    print("FILES SAVED")
    print("=" * 120)

    print()
    print(
        BEARING_MEANS_FILE
    )

    print(
        STANDARDIZED_BEARING_MEANS_FILE
    )

    print(
        DISTANCES_FILE
    )

    print(
        PCA_FILE
    )

    print(
        FIGURES_DIR
    )

    print()
    print("=" * 120)
    print("PADERBORN BEARING DOMAIN-SHIFT DIAGNOSTIC COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
