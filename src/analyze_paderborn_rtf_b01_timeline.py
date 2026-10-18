from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

B01_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "paderborn_run_to_failure"
    / "B01"
)

OPERATING_CONDITIONS_FILE = (
    B01_DIR
    / "B01_operatingConditions.csv"
)

TEMPERATURES_FILE = (
    B01_DIR
    / "B01_meanTemperatures.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_rtf_b01_timeline_diagnostic.csv"
)


# ============================================================
# SETTINGS
# ============================================================

LOG_EVENT_TIME = pd.Timestamp(
    "2021-10-19 09:13:00"
)

TARGET_MEASUREMENTS = [
    1,
    50,
    75,
    76,
    100,
    125,
    150,
    175,
    200,
    205,
    208,
    210,
    225,
    250,
    300,
    350,
    377,
]


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 130)
    print("PADERBORN B01 TIMELINE / LOG CONSISTENCY DIAGNOSTIC")
    print("=" * 130)

    operating = pd.read_csv(
        OPERATING_CONDITIONS_FILE
    )

    temperatures = pd.read_csv(
        TEMPERATURES_FILE
    )

    if len(operating) != len(temperatures):
        raise ValueError(
            "Operating-condition and temperature CSV lengths differ."
        )

    operating[
        "timestamp"
    ] = pd.to_datetime(
        operating[
            "Time"
        ],
        format="%d-%b-%Y %H:%M:%S",
    )

    temperatures[
        "timestamp"
    ] = pd.to_datetime(
        temperatures[
            "Time"
        ],
        format="%d-%b-%Y %H:%M:%S",
    )

    if not operating[
        "timestamp"
    ].equals(
        temperatures[
            "timestamp"
        ]
    ):
        raise ValueError(
            "Operating-condition and temperature timestamps do not align."
        )

    df = pd.DataFrame(
        {
            "measurement_number": range(
                1,
                len(operating) + 1,
            ),
            "timestamp": operating[
                "timestamp"
            ],
            "set_dynamic_load_n": operating[
                "setDynLoad / N"
            ],
            "peak_dynamic_load_n": operating[
                "peak_dynLoad / N"
            ],
            "set_static_load_n": operating[
                "setStatLoad / N"
            ],
            "mean_abs_static_load_n": operating[
                "meanAbs_statLoad / N"
            ],
            "set_speed_rpm": operating[
                "setSpeed / rpm"
            ],
            "mean_abs_speed_rpm": operating[
                "meanAbs_speed / rpm"
            ],
            "mean_temp_t1_c": temperatures[
                "Mean Abs. Temp. T1 / °C"
            ],
            "mean_temp_t2_c": temperatures[
                "Mean Abs. Temp. T2 / °C"
            ],
            "room_temp_c": temperatures[
                "Mean Room Temp. / °C"
            ],
        }
    )

    df[
        "seconds_since_previous"
    ] = (
        df[
            "timestamp"
        ]
        .diff()
        .dt
        .total_seconds()
    )

    df[
        "minutes_since_start"
    ] = (
        (
            df[
                "timestamp"
            ]
            - df[
                "timestamp"
            ].iloc[0]
        )
        .dt
        .total_seconds()
        / 60.0
    )

    df[
        "seconds_from_log_0913"
    ] = (
        df[
            "timestamp"
        ]
        - LOG_EVENT_TIME
    ).dt.total_seconds()

    # Closest measurement to 09:13
    closest_index = (
        df[
            "seconds_from_log_0913"
        ]
        .abs()
        .idxmin()
    )

    closest = df.loc[
        closest_index
    ]

    # Largest time gaps
    largest_gaps = (
        df[
            [
                "measurement_number",
                "timestamp",
                "seconds_since_previous",
            ]
        ]
        .dropna()
        .sort_values(
            "seconds_since_previous",
            ascending=False,
        )
        .head(15)
    )

    # All gaps above 60s
    large_gap_rows = df[
        df[
            "seconds_since_previous"
        ]
        > 60
    ][
        [
            "measurement_number",
            "timestamp",
            "seconds_since_previous",
        ]
    ]

    # Warm-up / experiment regime heuristic from log ranges
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

    last_warmup_measurement = None

    warmup_rows = df[
        df[
            "warmup_like"
        ]
    ]

    if not warmup_rows.empty:
        last_warmup_measurement = int(
            warmup_rows[
                "measurement_number"
            ].max()
        )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 130)
    print("RUN START / END")
    print("=" * 130)

    print()
    print(
        f"First timestamp : "
        f"{df['timestamp'].iloc[0]}"
    )

    print(
        f"Last timestamp  : "
        f"{df['timestamp'].iloc[-1]}"
    )

    print(
        f"Elapsed time    : "
        f"{df['minutes_since_start'].iloc[-1]:.2f} minutes"
    )

    print()
    print("=" * 130)
    print("LOG EVENT 09:13")
    print("=" * 130)

    print()
    print(
        f"Closest measurement : "
        f"M{int(closest['measurement_number']):04d}"
    )

    print(
        f"Timestamp           : "
        f"{closest['timestamp']}"
    )

    print(
        f"Delta to 09:13      : "
        f"{closest['seconds_from_log_0913']:.1f} s"
    )

    print(
        f"Speed               : "
        f"{closest['mean_abs_speed_rpm']:.3f} rpm"
    )

    print(
        f"Static load         : "
        f"{closest['mean_abs_static_load_n']:.3f} N"
    )

    print(
        f"Dynamic load peak   : "
        f"{closest['peak_dynamic_load_n']:.3f} N"
    )

    print(
        f"Temp T1             : "
        f"{closest['mean_temp_t1_c']:.3f} °C"
    )

    print(
        f"Temp T2             : "
        f"{closest['mean_temp_t2_c']:.3f} °C"
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
            f"{row['timestamp']}"
        )

    print()
    print("=" * 130)
    print("LARGEST TIME GAPS")
    print("=" * 130)

    print()
    print(
        largest_gaps.to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("GAPS > 60 s")
    print("=" * 130)

    print()

    if large_gap_rows.empty:
        print(
            "None."
        )
    else:
        print(
            large_gap_rows.to_string(
                index=False
            )
        )

    print()
    print("=" * 130)
    print("SELECTED TIMELINE POINTS")
    print("=" * 130)

    selected = df[
        df[
            "measurement_number"
        ].isin(
            TARGET_MEASUREMENTS
        )
    ][
        [
            "measurement_number",
            "timestamp",
            "minutes_since_start",
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
        selected.to_string(
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
        "TIMELINE DIAGNOSTIC COMPLETE"
    )
    print("=" * 130)


if __name__ == "__main__":
    main()
