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
    / "paderborn_rtf_b01_b03_comparison_revised.csv"
)


# ============================================================
# OFFICIAL LOG EVENTS
# ============================================================

# IMPORTANT:
# B02 has TWO different official milestones:
# 1) 13:32 - test stopped after RMS > 0.65 V twice.
# 2) 14:36 - final completion after allowing 4 exceedances.
#
# The first milestone is the fairer comparison for "early warning".
#
# B03 has one documented stop milestone:
# 18:23 - bearing housing temperature exceeded 110 C.
#
# B01 is not used for lead-time validation because of the
# documented stop/restart chronology.

EXPERIMENTS = {
    "B01": {
        "monitoring_file": "paderborn_rtf_b01_generic_monitoring.csv",
        "summary_file": "paderborn_rtf_b01_generic_monitoring_summary.csv",
        "official_first_action_time": None,
        "official_final_stop_time": None,
        "official_first_action_reason": None,
        "official_final_stop_reason": (
            "Temperature threshold event with interruption/restart chronology; "
            "no simple lead-time comparison reported."
        ),
    },
    "B02": {
        "monitoring_file": "paderborn_rtf_b02_generic_monitoring.csv",
        "summary_file": "paderborn_rtf_b02_generic_monitoring_summary.csv",
        "official_first_action_time": pd.Timestamp(
            "2021-10-25 13:32:00"
        ),
        "official_final_stop_time": pd.Timestamp(
            "2021-10-25 14:36:00"
        ),
        "official_first_action_reason": (
            "Test stopped after accelerometer RMS exceeded 0.65 V twice."
        ),
        "official_final_stop_reason": (
            "Test completed after accelerometer RMS exceeded 0.65 V four times."
        ),
    },
    "B03": {
        "monitoring_file": "paderborn_rtf_b03_generic_monitoring.csv",
        "summary_file": "paderborn_rtf_b03_generic_monitoring_summary.csv",
        "official_first_action_time": pd.Timestamp(
            "2021-11-05 18:23:00"
        ),
        "official_final_stop_time": pd.Timestamp(
            "2021-11-05 18:23:00"
        ),
        "official_first_action_reason": (
            "Automatic stop after bearing housing temperature exceeded 110 C."
        ),
        "official_final_stop_reason": (
            "Automatic stop after bearing housing temperature exceeded 110 C."
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


def minutes_between(later, earlier):
    if (
        later is None
        or earlier is None
        or pd.isna(later)
        or pd.isna(earlier)
    ):
        return None

    return (
        later - earlier
    ).total_seconds() / 60.0


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 140)
    print("PADERBORN RTF B01-B03 REVISED COMPARATIVE SUMMARY")
    print("=" * 140)

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

        first_action_time = (
            config[
                "official_first_action_time"
            ]
        )

        final_stop_time = (
            config[
                "official_final_stop_time"
            ]
        )

        lead_to_first_action = (
            minutes_between(
                first_action_time,
                anomaly_timestamp,
            )
        )

        lead_to_final_stop = (
            minutes_between(
                final_stop_time,
                anomaly_timestamp,
            )
        )

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
                "first_sustained_warning": first_warning,
                "warning_timestamp": warning_timestamp,
                "first_sustained_anomaly": first_anomaly,
                "anomaly_timestamp": anomaly_timestamp,
                "official_first_action_time": first_action_time,
                "lead_minutes_to_first_official_action": lead_to_first_action,
                "official_final_stop_time": final_stop_time,
                "lead_minutes_to_final_stop": lead_to_final_stop,
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
                "official_first_action_reason": (
                    config[
                        "official_first_action_reason"
                    ]
                ),
                "official_final_stop_reason": (
                    config[
                        "official_final_stop_reason"
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
    # Plot lead time to first official action
    # --------------------------------------------------------

    lead_df = comparison.dropna(
        subset=[
            "lead_minutes_to_first_official_action"
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
                "lead_minutes_to_first_official_action"
            ],
        )

        ax.set_xlabel(
            "Experiment"
        )

        ax.set_ylabel(
            "Lead time before first official action (minutes)"
        )

        ax.set_title(
            "Detector lead time before first official intervention"
        )

        ax.grid(
            True,
            axis="y",
            alpha=0.3,
        )

        fig.tight_layout()

        fig.savefig(
            FIGURES_DIR
            / "lead_time_before_first_official_action.png",
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(fig)

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    display_columns = [
        "experiment",
        "measurements",
        "reference_start",
        "reference_end",
        "first_sustained_warning",
        "first_sustained_anomaly",
        "anomaly_timestamp",
        "official_first_action_time",
        "lead_minutes_to_first_official_action",
        "official_final_stop_time",
        "lead_minutes_to_final_stop",
        "final_anomaly_score",
        "in_support_fraction",
    ]

    print()
    print(
        comparison[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 140)
    print("INTERPRETATION")
    print("=" * 140)

    print()
    print(
        "B01: development/robustness case only; no simple lead-time validation."
    )

    print(
        "B02: compare early warning primarily with the FIRST official stop/action at 13:32, "
        "not only with the final completion at 14:36."
    )

    print(
        "B03: the first official action and final stop are the same event at 18:23."
    )

    print()
    print("=" * 140)
    print("SAVED")
    print("=" * 140)

    print()
    print(OUTPUT_FILE)
    print(FIGURES_DIR)

    print()
    print("=" * 140)
    print("REVISED COMPARATIVE SUMMARY COMPLETE")
    print("=" * 140)


if __name__ == "__main__":
    main()
