from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "paderborn_rtf_comparison"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "paderborn_rtf_b01_b03_comparison.csv"
)


# ============================================================
# EXPERIMENT METADATA
# ============================================================

# Official stop information from experiment logs.
# B01 is intentionally not used for lead-time validation because
# the experiment was stopped/restarted and the log chronology is
# not directly comparable with the later anomaly episode.

EXPERIMENTS = {
    "B01": {
        "monitoring_file": "paderborn_rtf_b01_generic_monitoring.csv",
        "summary_file": "paderborn_rtf_b01_generic_monitoring_summary.csv",
        "official_stop_time": None,
        "official_stop_reason": (
            "Temperature threshold exceeded; run contains restart(s), "
            "so no simple lead-time comparison is reported."
        ),
    },
    "B02": {
        "monitoring_file": "paderborn_rtf_b02_generic_monitoring.csv",
        "summary_file": "paderborn_rtf_b02_generic_monitoring_summary.csv",
        "official_stop_time": pd.Timestamp(
            "2021-10-25 14:36:00"
        ),
        "official_stop_reason": (
            "Accelerometer RMS exceeded 0.65 V four times."
        ),
    },
    "B03": {
        "monitoring_file": "paderborn_rtf_b03_generic_monitoring.csv",
        "summary_file": "paderborn_rtf_b03_generic_monitoring_summary.csv",
        "official_stop_time": pd.Timestamp(
            "2021-11-05 18:23:00"
        ),
        "official_stop_reason": (
            "Bearing housing temperature exceeded 110 C."
        ),
    },
}


# ============================================================
# HELPERS
# ============================================================

def find_timestamp_column(df):
    for candidate in [
        "timestamp_operating",
        "timestamp",
    ]:
        if candidate in df.columns:
            return candidate

    return None


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 130)
    print("PADERBORN RTF B01-B03 COMPARATIVE SUMMARY")
    print("=" * 130)

    rows = []

    for experiment, config in EXPERIMENTS.items():
        monitoring_path = (
            PROCESSED_DIR
            / config["monitoring_file"]
        )

        summary_path = (
            PROCESSED_DIR
            / config["summary_file"]
        )

        if not monitoring_path.exists():
            raise FileNotFoundError(
                f"Missing monitoring file:\n{monitoring_path}"
            )

        if not summary_path.exists():
            raise FileNotFoundError(
                f"Missing summary file:\n{summary_path}"
            )

        monitoring = pd.read_csv(
            monitoring_path
        )

        summary = pd.read_csv(
            summary_path
        ).iloc[0]

        timestamp_column = (
            find_timestamp_column(
                monitoring
            )
        )

        if timestamp_column is not None:
            monitoring[
                timestamp_column
            ] = pd.to_datetime(
                monitoring[
                    timestamp_column
                ],
                errors="coerce",
            )

        first_warning = int(
            summary[
                "first_sustained_warning"
            ]
        )

        first_anomaly = int(
            summary[
                "first_sustained_anomaly"
            ]
        )

        warning_timestamp = None
        anomaly_timestamp = None

        if timestamp_column is not None:
            warning_match = monitoring[
                monitoring[
                    "measurement_number"
                ]
                == first_warning
            ]

            anomaly_match = monitoring[
                monitoring[
                    "measurement_number"
                ]
                == first_anomaly
            ]

            if not warning_match.empty:
                warning_timestamp = (
                    warning_match.iloc[0][
                        timestamp_column
                    ]
                )

            if not anomaly_match.empty:
                anomaly_timestamp = (
                    anomaly_match.iloc[0][
                        timestamp_column
                    ]
                )

        official_stop_time = (
            config[
                "official_stop_time"
            ]
        )

        lead_minutes = None

        if (
            official_stop_time is not None
            and anomaly_timestamp is not None
            and not pd.isna(
                anomaly_timestamp
            )
        ):
            lead_minutes = (
                official_stop_time
                - anomaly_timestamp
            ).total_seconds() / 60.0

        rows.append(
            {
                "experiment": experiment,
                "measurements": int(
                    summary[
                        "measurements"
                    ]
                ),
                "reference_start": int(
                    summary[
                        "reference_start"
                    ]
                ),
                "reference_end": int(
                    summary[
                        "reference_end"
                    ]
                ),
                "first_sustained_warning": (
                    first_warning
                ),
                "warning_timestamp": (
                    warning_timestamp
                ),
                "first_sustained_anomaly": (
                    first_anomaly
                ),
                "anomaly_timestamp": (
                    anomaly_timestamp
                ),
                "official_stop_time": (
                    official_stop_time
                ),
                "lead_time_minutes_vs_official_stop": (
                    lead_minutes
                ),
                "final_anomaly_score": float(
                    summary[
                        "final_anomaly_score"
                    ]
                ),
                "final_state": summary[
                    "final_state"
                ],
                "in_support_fraction": float(
                    summary[
                        "in_support_fraction"
                    ]
                ),
                "official_stop_reason": (
                    config[
                        "official_stop_reason"
                    ]
                ),
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Plot final anomaly score
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        comparison[
            "experiment"
        ],
        comparison[
            "final_anomaly_score"
        ],
    )

    ax.set_xlabel(
        "Experiment"
    )

    ax.set_ylabel(
        "Final anomaly score"
    )

    ax.set_title(
        "Paderborn B01-B03 final anomaly scores"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "final_anomaly_scores.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Plot lead time where official stop comparison is valid
    # --------------------------------------------------------

    lead_df = comparison.dropna(
        subset=[
            "lead_time_minutes_vs_official_stop"
        ]
    )

    if not lead_df.empty:
        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.bar(
            lead_df[
                "experiment"
            ],
            lead_df[
                "lead_time_minutes_vs_official_stop"
            ],
        )

        ax.set_xlabel(
            "Experiment"
        )

        ax.set_ylabel(
            "Lead time before official stop (minutes)"
        )

        ax.set_title(
            "Detector lead time before official stop"
        )

        ax.grid(
            True,
            axis="y",
            alpha=0.3,
        )

        fig.tight_layout()

        fig.savefig(
            FIGURES_DIR
            / "lead_time_before_official_stop.png",
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(fig)

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print(
        comparison[
            [
                "experiment",
                "measurements",
                "reference_start",
                "reference_end",
                "first_sustained_warning",
                "first_sustained_anomaly",
                "anomaly_timestamp",
                "official_stop_time",
                "lead_time_minutes_vs_official_stop",
                "final_anomaly_score",
                "in_support_fraction",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("INTERPRETATION")
    print("=" * 130)

    print()
    print(
        "B01: retained as a development / robustness case; "
        "no simple official-stop lead time is reported because "
        "the experiment contains documented interruption/restart behavior."
    )

    print(
        "B02 and B03: lead time is calculated from the first sustained "
        "frozen-detector anomaly to the official experiment stop time."
    )

    print()
    print("=" * 130)
    print("SAVED")
    print("=" * 130)

    print()
    print(OUTPUT_FILE)
    print(FIGURES_DIR)

    print()
    print("=" * 130)
    print("COMPARATIVE SUMMARY COMPLETE")
    print("=" * 130)


if __name__ == "__main__":
    main()
