from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    / "paderborn_rtf_b01_baseline_anomaly_scores.csv"
)

BASELINE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_rtf_b01_baseline_statistics.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "paderborn_rtf_b01_baseline_monitoring"
)


# ============================================================
# REFERENCE PERIOD
# ============================================================

# Exploratory reference window chosen because Q2 operates at
# speed/load levels much closer to Q3-Q5 than the very-low-speed
# beginning of the run.
#
# IMPORTANT:
# This is a reference period, not a certified "healthy" label.

REFERENCE_START = 76
REFERENCE_END = 150

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
                values
                - median
            )
        )
    )

    robust_sigma = (
        1.4826
        * mad
    )

    if robust_sigma < EPS:
        robust_sigma = float(
            np.std(
                values,
                ddof=1,
            )
        )

    if robust_sigma < EPS:
        robust_sigma = 1.0

    return (
        median,
        robust_sigma,
        mad,
    )


def positive_robust_z(
    values,
    median,
    robust_sigma,
):
    raw = (
        values
        - median
    ) / robust_sigma

    return np.maximum(
        raw,
        0.0,
    )


def first_sustained_event(
    mask,
    measurement_numbers,
    count,
):
    mask = np.asarray(
        mask,
        dtype=bool,
    )

    measurement_numbers = np.asarray(
        measurement_numbers,
        dtype=int,
    )

    if len(mask) < count:
        return None

    run_length = 0

    for index, value in enumerate(mask):
        if value:
            run_length += 1
        else:
            run_length = 0

        if run_length >= count:
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
        "PADERBORN RTF B01 BASELINE-BASED ANOMALY MONITORING"
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

    reference = df[
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
    ].copy()

    if reference.empty:
        raise ValueError(
            "Reference period is empty."
        )

    print()
    print(
        f"Reference measurements: "
        f"{REFERENCE_START}-{REFERENCE_END}"
    )

    print(
        f"Reference sample count : "
        f"{len(reference)}"
    )

    print()
    print(
        "NOTE: this reference period is exploratory and is not "
        "claimed to be a certified healthy interval."
    )

    # --------------------------------------------------------
    # Monitoring metrics
    # --------------------------------------------------------

    metric_columns = [
        "frontal_rms_per_krpm_kn",
        "rear_rms_per_krpm_kn",
        "frontal_kurtosis",
        "rear_kurtosis",
    ]

    baseline_records = []

    for column in metric_columns:
        median, robust_sigma, mad = (
            robust_location_scale(
                reference[
                    column
                ]
            )
        )

        baseline_records.append(
            {
                "metric": column,
                "reference_start": REFERENCE_START,
                "reference_end": REFERENCE_END,
                "median": median,
                "mad": mad,
                "robust_sigma": robust_sigma,
            }
        )

        df[
            f"{column}_positive_robust_z"
        ] = positive_robust_z(
            df[
                column
            ].to_numpy(),
            median,
            robust_sigma,
        )

    baseline_df = pd.DataFrame(
        baseline_records
    )

    # --------------------------------------------------------
    # Primary vibration anomaly score
    # --------------------------------------------------------

    df[
        "vibration_anomaly_score"
    ] = np.maximum(
        df[
            "frontal_rms_per_krpm_kn_positive_robust_z"
        ],
        df[
            "rear_rms_per_krpm_kn_positive_robust_z"
        ],
    )

    # Kurtosis is kept as a secondary late-stage indicator.
    df[
        "kurtosis_anomaly_score"
    ] = np.maximum(
        df[
            "frontal_kurtosis_positive_robust_z"
        ],
        df[
            "rear_kurtosis_positive_robust_z"
        ],
    )

    df[
        "monitoring_state"
    ] = "NORMAL"

    df.loc[
        df[
            "vibration_anomaly_score"
        ]
        >= WARNING_THRESHOLD,
        "monitoring_state",
    ] = "WARNING"

    df.loc[
        df[
            "vibration_anomaly_score"
        ]
        >= ANOMALY_THRESHOLD,
        "monitoring_state",
    ] = "ANOMALY"

    # --------------------------------------------------------
    # Sustained events
    # --------------------------------------------------------

    after_reference = df[
        df[
            "measurement_number"
        ]
        > REFERENCE_END
    ].copy()

    first_warning = (
        first_sustained_event(
            after_reference[
                "vibration_anomaly_score"
            ]
            >= WARNING_THRESHOLD,
            after_reference[
                "measurement_number"
            ],
            SUSTAINED_COUNT,
        )
    )

    first_anomaly = (
        first_sustained_event(
            after_reference[
                "vibration_anomaly_score"
            ]
            >= ANOMALY_THRESHOLD,
            after_reference[
                "measurement_number"
            ],
            SUSTAINED_COUNT,
        )
    )

    # --------------------------------------------------------
    # Summary by block
    # --------------------------------------------------------

    block_summary = (
        df
        .groupby(
            "time_block",
            sort=False,
        )
        .agg(
            measurements=(
                "measurement_number",
                "count",
            ),
            median_vibration_score=(
                "vibration_anomaly_score",
                "median",
            ),
            mean_vibration_score=(
                "vibration_anomaly_score",
                "mean",
            ),
            max_vibration_score=(
                "vibration_anomaly_score",
                "max",
            ),
            median_kurtosis_score=(
                "kurtosis_anomaly_score",
                "median",
            ),
            warning_or_anomaly_fraction=(
                "monitoring_state",
                lambda values: float(
                    np.mean(
                        np.asarray(
                            values
                        )
                        != "NORMAL"
                    )
                ),
            ),
            anomaly_fraction=(
                "monitoring_state",
                lambda values: float(
                    np.mean(
                        np.asarray(
                            values
                        )
                        == "ANOMALY"
                    )
                ),
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Plots
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
            "vibration_anomaly_score"
        ],
        linewidth=1.3,
        label="Vibration anomaly score",
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
        label="Reference period",
    )

    ax.set_xlabel(
        "Measurement number"
    )

    ax.set_ylabel(
        "Positive robust z-score"
    )

    ax.set_title(
        "B01 baseline-based vibration anomaly score"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "01_vibration_anomaly_score.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    ax.plot(
        df[
            "measurement_number"
        ],
        df[
            "kurtosis_anomaly_score"
        ],
        linewidth=1.3,
        label="Kurtosis anomaly score",
    )

    ax.axvspan(
        REFERENCE_START,
        REFERENCE_END,
        alpha=0.15,
        label="Reference period",
    )

    ax.set_xlabel(
        "Measurement number"
    )

    ax.set_ylabel(
        "Positive robust z-score"
    )

    ax.set_title(
        "B01 kurtosis deviation from reference period"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "02_kurtosis_anomaly_score.png",
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

    baseline_df.to_csv(
        BASELINE_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 130)
    print("REFERENCE STATISTICS")
    print("=" * 130)

    print()
    print(
        baseline_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("FIRST SUSTAINED EVENTS")
    print("=" * 130)

    print()
    print(
        f"Sustained warning "
        f"({SUSTAINED_COUNT} consecutive measurements): "
        f"{first_warning}"
    )

    print(
        f"Sustained anomaly "
        f"({SUSTAINED_COUNT} consecutive measurements): "
        f"{first_anomaly}"
    )

    print()
    print("=" * 130)
    print("TIME-BLOCK MONITORING SUMMARY")
    print("=" * 130)

    print()
    print(
        block_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("SELECTED MEASUREMENTS")
    print("=" * 130)

    selected_numbers = [
        76,
        100,
        125,
        150,
        175,
        200,
        225,
        250,
        275,
        300,
        325,
        350,
        360,
        370,
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
            "frontal_rms_per_krpm_kn",
            "rear_rms_per_krpm_kn",
            "vibration_anomaly_score",
            "kurtosis_anomaly_score",
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
        BASELINE_FILE
    )

    print(
        FIGURES_DIR
    )

    print()
    print("=" * 130)
    print(
        "B01 BASELINE MONITORING COMPLETE"
    )
    print("=" * 130)


if __name__ == "__main__":
    main()
