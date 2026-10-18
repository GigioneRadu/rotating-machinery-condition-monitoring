from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

FIGURES_ROOT = (
    PROJECT_ROOT
    / "figures"
    / "paderborn_rtf_generic_monitoring"
)


# ============================================================
# FROZEN EXPERIMENT CONFIGURATION
# ============================================================

# These intervals are intentionally frozen from the previous
# B01-B03 analyses. Do not change them retrospectively based
# on detector output.

EXPERIMENT_CONFIG = {
    "B01": {
        "summary_file": "paderborn_rtf_b01_trend_diagnostic.csv",
        "reference_start": 134,
        "reference_end": 190,
        "notes": (
            "Post-restart reference, entirely after the two "
            "major acquisition gaps."
        ),
    },
    "B02": {
        "summary_file": "paderborn_rtf_b02_measurement_summary.csv",
        "reference_start": 52,
        "reference_end": 250,
        "notes": (
            "First ~200 measurements after recorded warm-up."
        ),
    },
    "B03": {
        "summary_file": "paderborn_rtf_b03_measurement_summary.csv",
        "reference_start": 1,
        "reference_end": 200,
        "notes": (
            "First 200 recorded measurements; B03 files begin "
            "directly in randomized operating regime."
        ),
    },
}


# ============================================================
# FROZEN DETECTOR SETTINGS
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

K_NEIGHBORS = 10

WARNING_THRESHOLD = 3.0
ANOMALY_THRESHOLD = 6.0
SUSTAINED_COUNT = 5

EPS = 1e-12


# ============================================================
# HELPERS
# ============================================================

def robust_location_scale(values):
    values = np.asarray(
        values,
        dtype=float,
    )

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

    return median, mad, sigma


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


def find_timestamp_column(df):
    for candidate in [
        "timestamp_operating",
        "timestamp",
    ]:
        if candidate in df.columns:
            return candidate

    return None


# ============================================================
# GENERIC MONITORING
# ============================================================

def run_monitoring(experiment):
    experiment = experiment.upper()

    if experiment not in EXPERIMENT_CONFIG:
        raise ValueError(
            f"Unsupported experiment: {experiment}. "
            f"Choose from: {', '.join(EXPERIMENT_CONFIG)}"
        )

    config = EXPERIMENT_CONFIG[
        experiment
    ]

    input_file = (
        PROCESSED_DIR
        / config["summary_file"]
    )

    output_file = (
        PROCESSED_DIR
        / f"paderborn_rtf_{experiment.lower()}_generic_monitoring.csv"
    )

    summary_file = (
        PROCESSED_DIR
        / f"paderborn_rtf_{experiment.lower()}_generic_monitoring_summary.csv"
    )

    figures_dir = (
        FIGURES_ROOT
        / experiment.lower()
    )

    if not input_file.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{input_file}"
        )

    df = pd.read_csv(
        input_file
    )

    timestamp_column = (
        find_timestamp_column(
            df
        )
    )

    if timestamp_column is not None:
        df[
            timestamp_column
        ] = pd.to_datetime(
            df[
                timestamp_column
            ],
            errors="coerce",
        )

    reference_start = int(
        config[
            "reference_start"
        ]
    )

    reference_end = int(
        config[
            "reference_end"
        ]
    )

    reference_mask = (
        (
            df[
                "measurement_number"
            ]
            >= reference_start
        )
        & (
            df[
                "measurement_number"
            ]
            <= reference_end
        )
    )

    reference = df[
        reference_mask
    ].copy()

    if reference.empty:
        raise ValueError(
            f"{experiment}: reference interval is empty."
        )

    # --------------------------------------------------------
    # Scale operating conditions from reference only
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
    # Operating-condition support
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

    support_threshold = float(
        np.quantile(
            ref_distances[:, 1],
            0.99,
        )
    )

    nearest_reference_model = (
        NearestNeighbors(
            n_neighbors=1,
            metric="euclidean",
        )
    )

    nearest_reference_model.fit(
        reference_conditions
    )

    all_distances, _ = (
        nearest_reference_model.kneighbors(
            all_conditions
        )
    )

    df[
        "operating_condition_distance"
    ] = all_distances[:, 0]

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

    _, all_neighbor_indices = (
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
                df[
                    signal_column
                ].to_numpy()
                + EPS
            )
            / (
                expected_values
                + EPS
            )
        )

    # --------------------------------------------------------
    # Reference residual distribution
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

    baseline_rows = []

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
            ][
                :K_NEIGHBORS
            ]

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
                "experiment": experiment,
                "signal": signal_column,
                "reference_start": reference_start,
                "reference_end": reference_end,
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

    # --------------------------------------------------------
    # Combined score
    # --------------------------------------------------------

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
    # Sustained events
    # --------------------------------------------------------

    evaluation = df[
        df[
            "measurement_number"
        ]
        > reference_end
    ].copy()

    measurement_numbers = (
        evaluation[
            "measurement_number"
        ]
        .to_numpy()
    )

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

    first_warning = (
        first_sustained_event(
            (
                in_support
                & (
                    scores
                    >= WARNING_THRESHOLD
                )
            ),
            measurement_numbers,
            SUSTAINED_COUNT,
        )
    )

    first_anomaly = (
        first_sustained_event(
            (
                in_support
                & (
                    scores
                    >= ANOMALY_THRESHOLD
                )
            ),
            measurement_numbers,
            SUSTAINED_COUNT,
        )
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    final_measurement = int(
        df[
            "measurement_number"
        ].max()
    )

    final_row = df.iloc[-1]

    summary = pd.DataFrame(
        [
            {
                "experiment": experiment,
                "measurements": len(df),
                "reference_start": reference_start,
                "reference_end": reference_end,
                "k_neighbors": K_NEIGHBORS,
                "warning_threshold": WARNING_THRESHOLD,
                "anomaly_threshold": ANOMALY_THRESHOLD,
                "sustained_count": SUSTAINED_COUNT,
                "support_threshold": support_threshold,
                "first_sustained_warning": first_warning,
                "first_sustained_anomaly": first_anomaly,
                "final_measurement": final_measurement,
                "final_anomaly_score": float(
                    final_row[
                        "condition_matched_anomaly_score"
                    ]
                ),
                "final_state": final_row[
                    "monitoring_state"
                ],
                "in_support_fraction": float(
                    df[
                        "operating_condition_in_support"
                    ].mean()
                ),
                "notes": config[
                    "notes"
                ],
            }
        ]
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        df[
            "measurement_number"
        ],
        df[
            "condition_matched_anomaly_score"
        ],
        linewidth=1.2,
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
        reference_start,
        reference_end,
        alpha=0.15,
        label="Frozen reference",
    )

    if first_anomaly is not None:
        ax.axvline(
            first_anomaly,
            linestyle=":",
            label="First sustained anomaly",
        )

    ax.set_xlabel(
        "Measurement number"
    )

    ax.set_ylabel(
        "Positive robust z-score"
    )

    ax.set_title(
        f"{experiment} generic condition-matched monitoring"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        figures_dir
        / f"{experiment.lower()}_generic_monitoring.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        output_file,
        index=False,
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 130)
    print(
        f"PADERBORN {experiment} GENERIC CONDITION-MATCHED MONITORING"
    )
    print("=" * 130)

    print()
    print(
        f"Input file              : {input_file.name}"
    )

    print(
        f"Frozen reference        : "
        f"M{reference_start:04d}-M{reference_end:04d}"
    )

    print(
        f"Reference rows          : {len(reference)}"
    )

    print(
        f"K neighbors             : {K_NEIGHBORS}"
    )

    print(
        f"Warning threshold       : {WARNING_THRESHOLD}"
    )

    print(
        f"Anomaly threshold       : {ANOMALY_THRESHOLD}"
    )

    print(
        f"Sustained measurements  : {SUSTAINED_COUNT}"
    )

    print(
        f"Operating support limit : {support_threshold:.4f}"
    )

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
    print("=" * 130)
    print("RESULT")
    print("=" * 130)

    print()
    print(
        f"First sustained warning : {first_warning}"
    )

    print(
        f"First sustained anomaly : {first_anomaly}"
    )

    print(
        f"Final measurement       : {final_measurement}"
    )

    print(
        f"Final anomaly score     : "
        f"{final_row['condition_matched_anomaly_score']:.4f}"
    )

    print(
        f"Final monitoring state  : "
        f"{final_row['monitoring_state']}"
    )

    print(
        f"In-support fraction     : "
        f"{df['operating_condition_in_support'].mean():.4f}"
    )

    print()
    print("=" * 130)
    print("FILES SAVED")
    print("=" * 130)

    print()
    print(output_file)
    print(summary_file)
    print(figures_dir)

    print()
    print("=" * 130)
    print("GENERIC MONITORING COMPLETE")
    print("=" * 130)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen condition-matched monitoring "
            "pipeline for Paderborn B01, B02, or B03."
        )
    )

    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            "B01",
            "B02",
            "B03",
            "b01",
            "b02",
            "b03",
        ],
        help="Experiment identifier.",
    )

    args = parser.parse_args()

    run_monitoring(
        args.experiment
    )
