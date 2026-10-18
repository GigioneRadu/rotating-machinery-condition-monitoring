from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_rtf_b02_measurement_summary.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_rtf_b02_timeline_gap_diagnostic.csv"
)


# ============================================================
# SETTINGS
# ============================================================

GAP_THRESHOLD_SECONDS = 60

WINDOW_BEFORE = 5
WINDOW_AFTER = 5


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 130)
    print("PADERBORN B02 TIMELINE / GAP DIAGNOSTIC")
    print("=" * 130)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=[
            "timestamp_operating",
            "timestamp_temperature",
        ],
    )

    # --------------------------------------------------------
    # Warm-up heuristic
    # --------------------------------------------------------

    df[
        "warmup_like"
    ] = (
        df[
            "mean_abs_speed_rpm"
        ].between(
            700,
            800,
        )
        & df[
            "mean_abs_static_load_n"
        ].between(
            2300,
            2700,
        )
        & (
            df[
                "peak_dynamic_load_n"
            ]
            < 100
        )
    )

    warmup_rows = df[
        df[
            "warmup_like"
        ]
    ]

    last_warmup_measurement = None

    if not warmup_rows.empty:
        last_warmup_measurement = int(
            warmup_rows[
                "measurement_number"
            ].max()
        )

    # --------------------------------------------------------
    # Gaps
    # --------------------------------------------------------

    gaps = df[
        df[
            "seconds_since_previous"
        ]
        > GAP_THRESHOLD_SECONDS
    ].copy()

    diagnostic_rows = []

    print()
    print("=" * 130)
    print("RUN START / END")
    print("=" * 130)

    print()
    print(
        f"First timestamp : "
        f"{df['timestamp_operating'].iloc[0]}"
    )

    print(
        f"Last timestamp  : "
        f"{df['timestamp_operating'].iloc[-1]}"
    )

    print(
        f"Elapsed time    : "
        f"{df['minutes_since_start'].iloc[-1]:.2f} minutes"
    )

    print()
    print("=" * 130)
    print("WARM-UP HEURISTIC")
    print("=" * 130)

    print()
    print(
        f"Last warm-up-like measurement: "
        f"{last_warmup_measurement}"
    )

    if last_warmup_measurement is not None:
        row = df[
            df[
                "measurement_number"
            ]
            == last_warmup_measurement
        ].iloc[0]

        print(
            f"Timestamp: "
            f"{row['timestamp_operating']}"
        )

        print(
            f"Temp T1  : "
            f"{row['mean_temp_t1_c']:.3f} °C"
        )

    print()
    print("=" * 130)
    print(f"GAPS > {GAP_THRESHOLD_SECONDS} s")
    print("=" * 130)

    if gaps.empty:
        print()
        print("None.")
    else:
        for _, gap_row in gaps.iterrows():
            gap_measurement = int(
                gap_row[
                    "measurement_number"
                ]
            )

            previous_measurement = (
                gap_measurement - 1
            )

            previous_row = df[
                df[
                    "measurement_number"
                ]
                == previous_measurement
            ].iloc[0]

            current_row = df[
                df[
                    "measurement_number"
                ]
                == gap_measurement
            ].iloc[0]

            print()
            print("-" * 130)

            print(
                f"GAP BEFORE M{gap_measurement:04d}"
            )

            print("-" * 130)

            print()
            print(
                f"Previous measurement : "
                f"M{previous_measurement:04d}"
            )

            print(
                f"Previous timestamp   : "
                f"{previous_row['timestamp_operating']}"
            )

            print(
                f"Current timestamp    : "
                f"{current_row['timestamp_operating']}"
            )

            print(
                f"Gap                  : "
                f"{current_row['seconds_since_previous']:.1f} s"
            )

            print()
            print("Before -> after:")

            for column, label in [
                (
                    "mean_abs_speed_rpm",
                    "Speed rpm",
                ),
                (
                    "mean_abs_static_load_n",
                    "Static load N",
                ),
                (
                    "peak_dynamic_load_n",
                    "Dynamic peak N",
                ),
                (
                    "mean_temp_t1_c",
                    "Temp T1 C",
                ),
                (
                    "mean_temp_t2_c",
                    "Temp T2 C",
                ),
                (
                    "frontal_rms",
                    "Frontal RMS V",
                ),
                (
                    "rear_rms",
                    "Rear RMS V",
                ),
                (
                    "frontal_kurtosis",
                    "Frontal kurtosis",
                ),
                (
                    "rear_kurtosis",
                    "Rear kurtosis",
                ),
            ]:
                before = float(
                    previous_row[column]
                )

                after = float(
                    current_row[column]
                )

                delta = (
                    after - before
                )

                print(
                    f"  {label:<20}"
                    f"{before:>10.4f}"
                    f" -> "
                    f"{after:>10.4f}"
                    f" | delta={delta:>10.4f}"
                )

            start_measurement = max(
                1,
                gap_measurement
                - WINDOW_BEFORE,
            )

            end_measurement = min(
                int(
                    df[
                        "measurement_number"
                    ].max()
                ),
                gap_measurement
                + WINDOW_AFTER,
            )

            local = df[
                df[
                    "measurement_number"
                ].between(
                    start_measurement,
                    end_measurement,
                )
            ][
                [
                    "measurement_number",
                    "timestamp_operating",
                    "seconds_since_previous",
                    "mean_abs_speed_rpm",
                    "mean_abs_static_load_n",
                    "peak_dynamic_load_n",
                    "mean_temp_t1_c",
                    "frontal_rms",
                    "rear_rms",
                    "frontal_kurtosis",
                    "rear_kurtosis",
                ]
            ]

            print()
            print(
                f"Local window "
                f"M{start_measurement:04d}-"
                f"M{end_measurement:04d}:"
            )

            print()
            print(
                local.to_string(
                    index=False
                )
            )

            diagnostic_rows.append(
                {
                    "gap_before_measurement": gap_measurement,
                    "previous_measurement": previous_measurement,
                    "previous_timestamp": previous_row[
                        "timestamp_operating"
                    ],
                    "current_timestamp": current_row[
                        "timestamp_operating"
                    ],
                    "gap_seconds": current_row[
                        "seconds_since_previous"
                    ],
                    "temp_t1_before": previous_row[
                        "mean_temp_t1_c"
                    ],
                    "temp_t1_after": current_row[
                        "mean_temp_t1_c"
                    ],
                    "frontal_rms_before": previous_row[
                        "frontal_rms"
                    ],
                    "frontal_rms_after": current_row[
                        "frontal_rms"
                    ],
                    "rear_rms_before": previous_row[
                        "rear_rms"
                    ],
                    "rear_rms_after": current_row[
                        "rear_rms"
                    ],
                }
            )

    # --------------------------------------------------------
    # Trend checkpoints around visible rise
    # --------------------------------------------------------

    print()
    print("=" * 130)
    print("SELECTED TREND POINTS")
    print("=" * 130)

    selected_numbers = [
        1,
        25,
        50,
        75,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        850,
        875,
        900,
        910,
        920,
        921,
        922,
        923,
        950,
        1000,
        1050,
        1100,
        1116,
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
            "timestamp_operating",
            "mean_abs_speed_rpm",
            "mean_abs_static_load_n",
            "peak_dynamic_load_n",
            "mean_temp_t1_c",
            "frontal_rms",
            "rear_rms",
            "frontal_kurtosis",
            "rear_kurtosis",
        ]
    ]

    print()
    print(
        selected.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save compact gap diagnostic
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostic_df = pd.DataFrame(
        diagnostic_rows
    )

    diagnostic_df.to_csv(
        OUTPUT_FILE,
        index=False,
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
    print("B02 TIMELINE / GAP DIAGNOSTIC COMPLETE")
    print("=" * 130)


if __name__ == "__main__":
    main()
