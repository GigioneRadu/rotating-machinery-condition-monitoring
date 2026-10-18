from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_rtf_b01_trend_diagnostic.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_rtf_b01_condition_matched_monitoring.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_rtf_b01_condition_matched_summary.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "paderborn_rtf_b01_condition_matched_monitoring"
)


# ============================================================
# REFERENCE / SETTINGS
# ============================================================

REFERENCE_START = 76
REFERENCE_END = 150

K_NEIGHBORS = 10

WARNING_THRESHOLD = 3.0
ANOMALY_THRESHOLD = 6.0
SUSTAINED_COUNT = 5

EPS = 1e-12


# ============================================================
# FEATURES
# ============================================================

OPERATING_FEATURES = [
    "mean_abs_speed_rpm",
    "mean_abs_static_load_n",
    "peak_dynamic_load_n",
]

SIGNAL_COLUMNS = [
    "frontal_rms",
    "rear_rms",
]


# ============================================================
# HELPERS
# ============================================================

def robust_sigma(values):
    values = np.asarray(values, dtype=float)

    median = float(
        np.median(values)
    )

    mad = float(
        np.median(
            np.abs(
                values - median
            )
        )
    )

    sigma = 1.4826 * mad

    if sigma < EPS:
        sigma = float(
            np.std(
                values,
                ddof=1,
            )
        )

    if sigma < EPS:
        sigma = 1.0

    return median, sigma


def first_sustained_event(
    mask,
    measurement_numbers,
    count,
):
    run = 0

    for index, flag in enumerate(mask):
        if flag:
            run += 1
        else:
            run = 0

        if run >= count:
            start_index = (
                index
                - count
                + 1
            )

            return int(
                measurement_numbers[
                    start_index
                ]
            )

    return None


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 130)
    print(
        "PADERBORN B01 CONDITION-MATCHED BASELINE MONITORING"
    )
    print("=" * 130)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    reference_mask = (
        (
            df[
                "measurement_number"
            ]
            >= REFERENCE_START
        )
        & (
            df[
                "measurement_number"
            ]
            <= REFERENCE_END
        )
    )

    reference = df[
        reference_mask
    ].copy()

    print()
    print(
        f"Reference interval: "
        f"{REFERENCE_START}-{REFERENCE_END}"
    )

    print(
        f"Reference rows    : "
        f"{len(reference)}"
    )

    # --------------------------------------------------------
    # Standardize operating conditions using reference only
    # --------------------------------------------------------

    scaler = StandardScaler()

    reference_conditions = (
        scaler.fit_transform(
            reference[
                OPERATING_FEATURES
            ]
        )
    )

    all_conditions = (
        scaler.transform(
            df[
                OPERATING_FEATURES
            ]
        )
    )

    # --------------------------------------------------------
    # Reference operating-condition support
    # --------------------------------------------------------

    support_model = NearestNeighbors(
        n_neighbors=2,
        metric="euclidean",
    )

    support_model.fit(
        reference_conditions
    )

    ref_distances, _ = (
        support_model.kneighbors(
            reference_conditions
        )
    )

    # second neighbor excludes self
    reference_nn_distance = (
        ref_distances[
            :,
            1
        ]
    )

    support_threshold = float(
        np.quantile(
            reference_nn_distance,
            0.99,
        )
    )

    nearest_ref_model = (
        NearestNeighbors(
            n_neighbors=1,
            metric="euclidean",
        )
    )

    nearest_ref_model.fit(
        reference_conditions
    )

    all_nearest_distance, _ = (
        nearest_ref_model.kneighbors(
            all_conditions
        )
    )

    df[
        "operating_condition_distance"
    ] = (
        all_nearest_distance[
            :,
            0
        ]
    )

    df[
        "operating_condition_in_support"
    ] = (
        df[
            "operating_condition_distance"
        ]
        <= support_threshold
    )

    # --------------------------------------------------------
    # Condition-matched expected vibration
    # --------------------------------------------------------

    k = min(
        K_NEIGHBORS,
        len(reference),
    )

    knn = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
    )

    knn.fit(
        reference_conditions
    )

    distances, indices = (
        knn.kneighbors(
            all_conditions
        )
    )

    for signal_column in SIGNAL_COLUMNS:
        reference_values = (
            reference[
                signal_column
            ]
            .to_numpy()
        )

        expected = []

        for neighbor_indices in indices:
            neighbor_values = (
                reference_values[
                    neighbor_indices
                ]
            )

            expected.append(
                float(
                    np.median(
                        neighbor_values
                    )
                )
            )

        expected = np.asarray(
            expected
        )

        df[
            f"{signal_column}_expected"
        ] = expected

        df[
            f"{signal_column}_log_ratio"
        ] = np.log(
            (
                df[
                    signal_column
                ].to_numpy()
                + EPS
            )
            / (
                expected
                + EPS
            )
        )

    # --------------------------------------------------------
    # Build reference residual distribution
    # using leave-one-out-like neighbor prediction
    # --------------------------------------------------------

    loo_neighbors = min(
        K_NEIGHBORS + 1,
        len(reference),
    )

    loo_model = NearestNeighbors(
        n_neighbors=loo_neighbors,
        metric="euclidean",
    )

    loo_model.fit(
        reference_conditions
    )

    _, loo_indices = (
        loo_model.kneighbors(
            reference_conditions
        )
    )

    baseline_stats = {}

    for signal_column in SIGNAL_COLUMNS:
        reference_values = (
            reference[
                signal_column
            ]
            .to_numpy()
        )

        loo_log_ratios = []

        for row_index, neighbor_indices in enumerate(
            loo_indices
        ):
            neighbor_indices = [
                idx
                for idx in neighbor_indices
                if idx != row_index
            ][:K_NEIGHBORS]

            if not neighbor_indices:
                continue

            expected_value = float(
                np.median(
                    reference_values[
                        neighbor_indices
                    ]
                )
            )

            actual_value = float(
                reference_values[
                    row_index
                ]
            )

            loo_log_ratios.append(
                np.log(
                    (
                        actual_value
                        + EPS
                    )
                    / (
                        expected_value
                        + EPS
                    )
                )
            )

        center, sigma = robust_sigma(
            loo_log_ratios
        )

        baseline_stats[
            signal_column
        ] = {
            "center": center,
            "sigma": sigma,
        }

        residual = (
            df[
                f"{signal_column}_log_ratio"
            ]
            - center
        )

        df[
            f"{signal_column}_positive_z"
        ] = np.maximum(
            residual / sigma,
            0.0,
        )

    df[
        "condition_matched_anomaly_score"
    ] = np.maximum(
        df[
            "frontal_rms_positive_z"
        ],
        df[
            "rear_rms_positive_z"
        ],
    )

    # --------------------------------------------------------
    # Monitoring states
    # --------------------------------------------------------

    df[
        "monitoring_state"
    ] = "NORMAL"

    df.loc[
        df[
            "condition_matched_anomaly_score"
        ]
        >= WARNING_THRESHOLD,
        "monitoring_state",
    ] = "WARNING"

    df.loc[
        df[
            "condition_matched_anomaly_score"
        ]
        >= ANOMALY_THRESHOLD,
        "monitoring_state",
    ] = "ANOMALY"

    df.loc[
        ~df[
            "operating_condition_in_support"
        ],
        "monitoring_state",
    ] = "OOD_CONDITION"

    # --------------------------------------------------------
    # Sustained event dates, only inside support
    # --------------------------------------------------------

    after_reference = df[
        df[
            "measurement_number"
        ]
        > REFERENCE_END
    ].copy()

    valid = (
        after_reference[
            "operating_condition_in_support"
        ]
        .to_numpy()
    )

    score = (
        after_reference[
            "condition_matched_anomaly_score"
        ]
        .to_numpy()
    )

    measurement_numbers = (
        after_reference[
            "measurement_number"
        ]
        .to_numpy()
    )

    warning_mask = (
        valid
        & (
            score
            >= WARNING_THRESHOLD
        )
    )

    anomaly_mask = (
        valid
        & (
            score
            >= ANOMALY_THRESHOLD
        )
    )

    first_warning = (
        first_sustained_event(
            warning_mask,
            measurement_numbers,
            SUSTAINED_COUNT,
        )
    )

    first_anomaly = (
        first_sustained_event(
            anomaly_mask,
            measurement_numbers,
            SUSTAINED_COUNT,
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows = []

    for block_name, block in df.groupby(
        "time_block",
        sort=False,
    ):
        summary_rows.append(
            {
                "time_block": block_name,
                "measurements": len(
                    block
                ),
                "in_support_fraction": float(
                    block[
                        "operating_condition_in_support"
                    ].mean()
                ),
                "median_score": float(
                    block[
                        "condition_matched_anomaly_score"
                    ].median()
                ),
                "mean_score": float(
                    block[
                        "condition_matched_anomaly_score"
                    ].mean()
                ),
                "max_score": float(
                    block[
                        "condition_matched_anomaly_score"
                    ].max()
                ),
                "anomaly_fraction_all": float(
                    (
                        block[
                            "condition_matched_anomaly_score"
                        ]
                        >= ANOMALY_THRESHOLD
                    ).mean()
                ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    ax.plot(
        df[
            "measurement_number"
        ],
        df[
            "condition_matched_anomaly_score"
        ],
        linewidth=1.3,
        label="Condition-matched anomaly score",
    )

    ax.axhline(
        WARNING_THRESHOLD,
        linestyle="--",
        label="Warning threshold",
    )

    ax.axhline(
        ANOMALY_THRESHOLD,
        linestyle="--",
        label="Anomaly threshold",
    )

    ax.axvspan(
        REFERENCE_START,
        REFERENCE_END,
        alpha=0.15,
        label="Reference interval",
    )

    ood = df[
        ~df[
            "operating_condition_in_support"
        ]
    ]

    if not ood.empty:
        ax.scatter(
            ood[
                "measurement_number"
            ],
            ood[
                "condition_matched_anomaly_score"
            ],
            s=14,
            label="OOD operating condition",
        )

    ax.set_xlabel(
        "Measurement number"
    )

    ax.set_ylabel(
        "Positive robust z-score"
    )

    ax.set_title(
        "B01 condition-matched baseline anomaly score"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "condition_matched_anomaly_score.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 130)
    print("REFERENCE RESIDUAL STATISTICS")
    print("=" * 130)

    print()
    for signal_column in SIGNAL_COLUMNS:
        stats = baseline_stats[
            signal_column
        ]

        print(
            f"{signal_column:<15}"
            f" center={stats['center']:.6f}"
            f" | robust_sigma={stats['sigma']:.6f}"
        )

    print()
    print(
        f"Operating-condition support threshold: "
        f"{support_threshold:.4f}"
    )

    print()
    print("=" * 130)
    print("FIRST SUSTAINED EVENTS")
    print("=" * 130)

    print()
    print(
        f"Warning ({SUSTAINED_COUNT} consecutive in-support): "
        f"{first_warning}"
    )

    print(
        f"Anomaly ({SUSTAINED_COUNT} consecutive in-support): "
        f"{first_anomaly}"
    )

    print()
    print("=" * 130)
    print("TIME-BLOCK SUMMARY")
    print("=" * 130)

    print()
    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("SELECTED MEASUREMENTS")
    print("=" * 130)

    selected_numbers = [
        150,
        175,
        190,
        200,
        205,
        208,
        210,
        215,
        225,
        250,
        300,
        350,
        377,
    ]

    selected = df[
        df[
            "measurement_number"
        ].isin(
            selected_numbers
        )
    ][
        [
            "measurement_number",
            "mean_abs_speed_rpm",
            "mean_abs_static_load_n",
            "peak_dynamic_load_n",
            "operating_condition_distance",
            "operating_condition_in_support",
            "frontal_rms",
            "frontal_rms_expected",
            "rear_rms",
            "rear_rms_expected",
            "condition_matched_anomaly_score",
            "monitoring_state",
        ]
    ]

    print()
    print(
        selected.to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("FILES SAVED")
    print("=" * 130)

    print()
    print(
        OUTPUT_FILE
    )

    print(
        SUMMARY_FILE
    )

    print(
        FIGURES_DIR
    )

    print()
    print("=" * 130)
    print(
        "CONDITION-MATCHED MONITORING COMPLETE"
    )
    print("=" * 130)


if __name__ == "__main__":
    main()
