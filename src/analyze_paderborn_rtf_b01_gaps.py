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
    / "paderborn_rtf_b01_timeline_diagnostic.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_rtf_b01_gap_diagnostic.csv"
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
    print("PADERBORN B01 GAP / RESTART DIAGNOSTIC")
    print("=" * 130)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"],
    )

    gaps = df[
        df["seconds_since_previous"]
        > GAP_THRESHOLD_SECONDS
    ].copy()

    if gaps.empty:
        print()
        print(
            f"No gaps > {GAP_THRESHOLD_SECONDS} s found."
        )
        return

    print()
    print(
        f"Gaps > {GAP_THRESHOLD_SECONDS} s: "
        f"{len(gaps)}"
    )

    diagnostic_rows = []

    for _, gap_row in gaps.iterrows():
        gap_measurement = int(
            gap_row["measurement_number"]
        )

        previous_measurement = (
            gap_measurement - 1
        )

        previous_row = df[
            df["measurement_number"]
            == previous_measurement
        ].iloc[0]

        current_row = df[
            df["measurement_number"]
            == gap_measurement
        ].iloc[0]

        print()
        print("=" * 130)
        print(
            f"GAP BEFORE M{gap_measurement:04d}"
        )
        print("=" * 130)

        print()
        print(
            f"Previous measurement : "
            f"M{previous_measurement:04d}"
        )

        print(
            f"Previous timestamp   : "
            f"{previous_row['timestamp']}"
        )

        print(
            f"Current timestamp    : "
            f"{current_row['timestamp']}"
        )

        print(
            f"Gap                  : "
            f"{current_row['seconds_since_previous']:.1f} s"
        )

        print()
        print(
            "Before -> after:"
        )

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
                f"  {label:<18}"
                f"{before:>10.3f}"
                f" -> "
                f"{after:>10.3f}"
                f" | delta={delta:>10.3f}"
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
                "timestamp",
                "seconds_since_previous",
                "mean_abs_speed_rpm",
                "mean_abs_static_load_n",
                "peak_dynamic_load_n",
                "mean_temp_t1_c",
                "mean_temp_t2_c",
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
                    "timestamp"
                ],
                "current_timestamp": current_row[
                    "timestamp"
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
                "temp_t1_delta": (
                    current_row[
                        "mean_temp_t1_c"
                    ]
                    - previous_row[
                        "mean_temp_t1_c"
                    ]
                ),
                "temp_t2_before": previous_row[
                    "mean_temp_t2_c"
                ],
                "temp_t2_after": current_row[
                    "mean_temp_t2_c"
                ],
                "temp_t2_delta": (
                    current_row[
                        "mean_temp_t2_c"
                    ]
                    - previous_row[
                        "mean_temp_t2_c"
                    ]
                ),
            }
        )

    diagnostic_df = pd.DataFrame(
        diagnostic_rows
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostic_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 130)
    print("COMPACT GAP SUMMARY")
    print("=" * 130)

    print()
    print(
        diagnostic_df.to_string(
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
    print(
        "GAP / RESTART DIAGNOSTIC COMPLETE"
    )
    print("=" * 130)


if __name__ == "__main__":
    main()
