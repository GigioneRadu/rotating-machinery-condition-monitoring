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
    / "paderborn_rtf_b01_post_restart_monitoring.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_rtf_b01_post_restart_summary.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "paderborn_rtf_b01_post_restart_monitoring"
)


# ============================================================
# SEGMENTS / REFERENCE
# ============================================================

# Based on the observed gaps:
# M0105 -> M0106 : 1147 s
# M0133 -> M0134 : 138 s
#
# We treat the experiment as three acquisition episodes.

SEGMENT_1_END = 105
SEGMENT_2_END = 133

# Post-restart reference chosen entirely AFTER the second gap,
# and before the clear vibration rise around M0205-M0208.
REFERENCE_START = 134
REFERENCE_END = 190

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

def robust_location_scale(values):
    values = np.asarray(values, dtype=float)

    median = float(
        np.median(values)
    )

    mad = float(
        np.median(
            np.abs(values - median)
        )
    )

    sigma = 1.4826 * mad

    if sigma < EPS:
        sigma = float(
            np.std(values, ddof=1)
        )

    if sigma < EPS:
        sigma = 1.0

    return median, mad, sigma


def first_sustained_event(mask, measurement_numbers, count):
    run = 0

    for index, flag in enumerate(mask):
        if flag:
            run += 1
        else:
            run = 0

        if run >= count:
            start_index = index - count + 1
            return int(
                measurement_numbers[start_index]
            )

    return None


def assign_segment(measurement_number):
    if measurement_number <= SEGMENT_1_END:
        return "episode_1_pre_restart"

    if measurement_number <= SEGMENT_2_END:
        return "episode_2_after_long_gap"

    return "episode_3_post_second_gap"


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 130)
    print("PADERBORN B01 POST-RESTART CONDITION-MATCHED MONITORING")
    print("=" * 130)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    df["episode"] = df[
        "measurement_number"
    ].apply(
        assign_segment
    )

    reference_mask = (
        (
            df["measurement_number"]
            >= REFERENCE_START
        )
        & (
            df["measurement_number"]
            <= REFERENCE_END
        )
    )

    reference = df[
        reference_mask
    ].copy()

    if reference.empty:
        raise ValueError(
            "Reference interval is empty."
        )

    print()
    print(
        f"Reference interval : "
        f"M{REFERENCE_START:04d}-M{REFERENCE_END:04d}"
    )

    print(
        f"Reference rows     : "
        f"{len(reference)}"
    )

    print(
        "Reference is entirely after both major acquisition gaps."
    )

    # --------------------------------------------------------
    # Standardize operating conditions using reference only
    # --------------------------------------------------------

    scaler = StandardScaler()

    reference_conditions = scaler.fit_transform(
        reference[
            OPERATING_FEATURES
        ]
    )

    all_conditions = scaler.transform(
        df[
            OPERATING_FEATURES
        ]
    )

    # --------------------------------------------------------
    # Reference support threshold
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

    reference_nn_distance = (
        ref_distances[:, 1]
    )

    support_threshold = float(
        np.quantile(
            reference_nn_distance,
            0.99,
        )
    )

    nearest_ref_model = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
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
    ] = all_nearest_distance[:, 0]

    df[
        "operating_condition_in_support"
    ] = (
        df[
            "operating_condition_distance"
        ]
        <= support_threshold
    )

    # --------------------------------------------------------
    # Expected vibration from matched reference neighbors
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

    _, all_neighbor_indices = knn.kneighbors(
        all_conditions
    )

    for signal_column in SIGNAL_COLUMNS:
        reference_values = reference[
            signal_column
        ].to_numpy()

        expected_values = []

        for neighbor_indices in all_neighbor_indices:
            expected_values.append(
                float(
                    np.median(
                        reference_values[
                            neighbor_indices
                        ]
                    )
                )
            )

        expected_values = np.asarray(
            expected_values
        )

        df[
            f"{signal_column}_expected"
        ] = expected_values

        df[
            f"{signal_column}_log_ratio"
        ] = np.log(
            (
                df[signal_column].to_numpy()
                + EPS
            )
            / (
                expected_values
                + EPS
            )
        )

    # --------------------------------------------------------
    # Leave-one-out-like reference residuals
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

    _, loo_indices = loo_model.kneighbors(
        reference_conditions
    )

    baseline_rows = []

    for signal_column in SIGNAL_COLUMNS:
        reference_values = reference[
            signal_column
        ].to_numpy()

        loo_log_ratios = []

        for row_index, neighbor_indices in enumerate(
            loo_indices
        ):
            neighbor_indices = [
                idx
                for idx in neighbor_indices
                if idx != row_index
            ][:K_NEIGHBORS]

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

        center, mad, sigma = (
            robust_location_scale(
                loo_log_ratios
            )
        )

        baseline_rows.append(
            {
                "signal": signal_column,
                "reference_start": REFERENCE_START,
                "reference_end": REFERENCE_END,
                "center": center,
                "mad": mad,
                "robust_sigma": sigma,
            }
        )

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

    baseline_df = pd.DataFrame(
        baseline_rows
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
    # States
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
    # Sustained events only after reference
    # --------------------------------------------------------

    evaluation = df[
        df[
            "measurement_number"
        ]
        > REFERENCE_END
    ].copy()

    in_support = (
        evaluation[
            "operating_condition_in_support"
        ]
        .to_numpy()
    )

    scores = (
        evaluation[
            "condition_matched_anomaly_score"
        ]
        .to_numpy()
    )

    measurement_numbers = (
        evaluation[
            "measurement_number"
        ]
        .to_numpy()
    )

    warning_mask = (
        in_support
        & (
            scores
            >= WARNING_THRESHOLD
        )
    )

    anomaly_mask = (
        in_support
        & (
            scores
            >= ANOMALY_THRESHOLD
        )
    )

    first_warning = first_sustained_event(
        warning_mask,
        measurement_numbers,
        SUSTAINED_COUNT,
    )

    first_anomaly = first_sustained_event(
        anomaly_mask,
        measurement_numbers,
        SUSTAINED_COUNT,
    )

    # --------------------------------------------------------
    # Episode summary
    # --------------------------------------------------------

    episode_summary = (
        df
        .groupby(
            "episode",
            sort=False,
        )
        .agg(
            measurements=(
                "measurement_number",
                "count",
            ),
            first_measurement=(
                "measurement_number",
                "min",
            ),
            last_measurement=(
                "measurement_number",
                "max",
            ),
            in_support_fraction=(
                "operating_condition_in_support",
                "mean",
            ),
            median_score=(
                "condition_matched_anomaly_score",
                "median",
            ),
            mean_score=(
                "condition_matched_anomaly_score",
                "mean",
            ),
            max_score=(
                "condition_matched_anomaly_score",
                "max",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Save plots
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
        label="Post-restart reference",
    )

    ax.axvline(
        106,
        linestyle=":",
        label="Long-gap restart boundary",
    )

    ax.axvline(
        134,
        linestyle=":",
        label="Second-gap boundary",
    )

    ax.set_xlabel(
        "Measurement number"
    )

    ax.set_ylabel(
        "Positive robust z-score"
    )

    ax.set_title(
        "B01 post-restart condition-matched monitoring"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "post_restart_condition_matched_score.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    episode_summary.to_csv(
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
    print(
        baseline_df.to_string(
            index=False
        )
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
    print("EPISODE SUMMARY")
    print("=" * 130)

    print()
    print(
        episode_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("SELECTED MEASUREMENTS")
    print("=" * 130)

    selected_numbers = [
        134,
        150,
        175,
        190,
        195,
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
            "episode",
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
    print("POST-RESTART MONITORING COMPLETE")
    print("=" * 130)


if __name__ == "__main__":
    main()
