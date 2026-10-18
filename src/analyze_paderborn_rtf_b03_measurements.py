from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

B03_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "paderborn_run_to_failure"
    / "B03"
)

VIBRATION_DIR = B03_DIR / "vibrationData"
OPERATING_FILE = B03_DIR / "B03_operatingConditions.csv"
TEMPERATURE_FILE = B03_DIR / "B03_meanTemperatures.csv"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_rtf_b03_measurement_summary.csv"
)


# ============================================================
# HELPERS
# ============================================================

def extract_measurement_number(path: Path) -> int:
    return int(
        path.stem.split("_M")[-1]
    )


def signal_features(signal):
    signal = np.asarray(
        signal,
        dtype=float,
    )

    centered = (
        signal
        - np.mean(signal)
    )

    rms = float(
        np.sqrt(
            np.mean(
                centered ** 2
            )
        )
    )

    peak = float(
        np.max(
            np.abs(centered)
        )
    )

    peak_to_peak = float(
        np.ptp(centered)
    )

    crest_factor = (
        peak / rms
        if rms > 0
        else np.nan
    )

    kurt = float(
        kurtosis(
            centered,
            fisher=False,
            bias=False,
        )
    )

    std = float(
        np.std(
            centered,
            ddof=1,
        )
    )

    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "kurtosis": kurt,
        "std": std,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 130)
    print(
        "PADERBORN RUN-TO-FAILURE B03 MEASUREMENT SUMMARY"
    )
    print("=" * 130)

    mat_files = sorted(
        VIBRATION_DIR.glob(
            "data_B03_M*.mat"
        ),
        key=extract_measurement_number,
    )

    if not mat_files:
        raise FileNotFoundError(
            f"No B03 MAT files found in:\n"
            f"{VIBRATION_DIR}"
        )

    operating = pd.read_csv(
        OPERATING_FILE
    )

    temperatures = pd.read_csv(
        TEMPERATURE_FILE
    )

    if not (
        len(mat_files)
        == len(operating)
        == len(temperatures)
    ):
        raise ValueError(
            "MAT / operating / temperature counts do not match."
        )

    print()
    print(
        f"Measurements: {len(mat_files)}"
    )

    rows = []
    sampling_frequencies = []
    durations = []

    for index, mat_file in enumerate(
        mat_files,
        start=1,
    ):
        data = loadmat(
            mat_file,
            squeeze_me=True,
            struct_as_record=False,
        )

        frontal = np.asarray(
            data[
                "accHorizFrontal_C"
            ],
            dtype=float,
        )

        rear = np.asarray(
            data[
                "accHorizRear_A"
            ],
            dtype=float,
        )

        time = np.asarray(
            data[
                "measTime"
            ],
            dtype=float,
        )

        dt = float(
            np.median(
                np.diff(time)
            )
        )

        fs = 1.0 / dt

        duration = float(
            time[-1]
            - time[0]
            + dt
        )

        sampling_frequencies.append(fs)
        durations.append(duration)

        frontal_features = signal_features(
            frontal
        )

        rear_features = signal_features(
            rear
        )

        op_row = operating.iloc[
            index - 1
        ]

        temp_row = temperatures.iloc[
            index - 1
        ]

        rows.append(
            {
                "measurement_number": index,
                "mat_file": mat_file.name,
                "timestamp_operating": op_row["Time"],
                "timestamp_temperature": temp_row["Time"],
                "sampling_frequency_hz": fs,
                "duration_s": duration,
                "set_dynamic_load_n": op_row["setDynLoad / N"],
                "peak_dynamic_load_n": op_row["peak_dynLoad / N"],
                "set_static_load_n": op_row["setStatLoad / N"],
                "mean_abs_static_load_n": op_row["meanAbs_statLoad / N"],
                "set_speed_rpm": op_row["setSpeed / rpm"],
                "mean_abs_speed_rpm": op_row["meanAbs_speed / rpm"],
                "mean_temp_t1_c": temp_row["Mean Abs. Temp. T1 / °C"],
                "mean_temp_t2_c": temp_row["Mean Abs. Temp. T2 / °C"],
                "room_temp_c": temp_row["Mean Room Temp. / °C"],
                "frontal_rms": frontal_features["rms"],
                "frontal_peak": frontal_features["peak"],
                "frontal_peak_to_peak": frontal_features["peak_to_peak"],
                "frontal_crest_factor": frontal_features["crest_factor"],
                "frontal_kurtosis": frontal_features["kurtosis"],
                "frontal_std": frontal_features["std"],
                "rear_rms": rear_features["rms"],
                "rear_peak": rear_features["peak"],
                "rear_peak_to_peak": rear_features["peak_to_peak"],
                "rear_crest_factor": rear_features["crest_factor"],
                "rear_kurtosis": rear_features["kurtosis"],
                "rear_std": rear_features["std"],
            }
        )

        if (
            index == 1
            or index % 100 == 0
            or index == len(mat_files)
        ):
            print(
                f"Processed {index:4d}/{len(mat_files)}"
            )

    df = pd.DataFrame(
        rows
    )

    df[
        "timestamp_operating"
    ] = pd.to_datetime(
        df["timestamp_operating"],
        format="%d-%b-%Y %H:%M:%S",
    )

    df[
        "timestamp_temperature"
    ] = pd.to_datetime(
        df["timestamp_temperature"],
        format="%d-%b-%Y %H:%M:%S",
    )

    timestamps_aligned = bool(
        (
            df["timestamp_operating"]
            == df["timestamp_temperature"]
        ).all()
    )

    df[
        "seconds_since_previous"
    ] = (
        df["timestamp_operating"]
        .diff()
        .dt
        .total_seconds()
    )

    df[
        "minutes_since_start"
    ] = (
        (
            df["timestamp_operating"]
            - df["timestamp_operating"].iloc[0]
        )
        .dt
        .total_seconds()
        / 60.0
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 130)
    print("SAMPLING CHECK")
    print("=" * 130)

    print()
    print(
        f"Sampling frequency min : "
        f"{np.min(sampling_frequencies):.3f} Hz"
    )

    print(
        f"Sampling frequency max : "
        f"{np.max(sampling_frequencies):.3f} Hz"
    )

    print(
        f"Sampling frequency mean: "
        f"{np.mean(sampling_frequencies):.3f} Hz"
    )

    print(
        f"Duration min           : "
        f"{np.min(durations):.6f} s"
    )

    print(
        f"Duration max           : "
        f"{np.max(durations):.6f} s"
    )

    print()
    print("=" * 130)
    print("TIMELINE CHECK")
    print("=" * 130)

    print()
    print(
        f"Timestamps aligned     : "
        f"{timestamps_aligned}"
    )

    print(
        f"First timestamp        : "
        f"{df['timestamp_operating'].iloc[0]}"
    )

    print(
        f"Last timestamp         : "
        f"{df['timestamp_operating'].iloc[-1]}"
    )

    print(
        f"Elapsed time           : "
        f"{df['minutes_since_start'].iloc[-1]:.2f} min"
    )

    print()
    print("=" * 130)
    print("OPERATING CONDITION RANGES")
    print("=" * 130)

    for column in [
        "mean_abs_speed_rpm",
        "mean_abs_static_load_n",
        "peak_dynamic_load_n",
    ]:
        print()
        print(f"{column}:")
        print(f"  min  = {df[column].min():.3f}")
        print(f"  mean = {df[column].mean():.3f}")
        print(f"  max  = {df[column].max():.3f}")

    print()
    print("=" * 130)
    print("TEMPERATURE RANGE")
    print("=" * 130)

    for column in [
        "mean_temp_t1_c",
        "mean_temp_t2_c",
        "room_temp_c",
    ]:
        print(
            f"{column:<20}"
            f" min={df[column].min():.3f}"
            f" | mean={df[column].mean():.3f}"
            f" | max={df[column].max():.3f}"
        )

    print()
    print("=" * 130)
    print("LARGEST TIME GAPS")
    print("=" * 130)

    largest_gaps = (
        df[
            [
                "measurement_number",
                "timestamp_operating",
                "seconds_since_previous",
                "mean_temp_t1_c",
                "mean_temp_t2_c",
            ]
        ]
        .dropna()
        .sort_values(
            "seconds_since_previous",
            ascending=False,
        )
        .head(15)
    )

    print()
    print(
        largest_gaps.to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("VIBRATION TREND SNAPSHOTS")
    print("=" * 130)

    snapshot_numbers = sorted(
        set(
            [
                1,
                10,
                25,
                50,
                75,
                100,
                150,
                200,
                250,
                300,
                350,
                400,
                450,
                500,
                550,
                600,
                len(df),
            ]
        )
    )

    snapshot_numbers = [
        number
        for number in snapshot_numbers
        if number <= len(df)
    ]

    snapshots = df[
        df[
            "measurement_number"
        ].isin(
            snapshot_numbers
        )
    ][
        [
            "measurement_number",
            "timestamp_operating",
            "mean_abs_speed_rpm",
            "mean_abs_static_load_n",
            "mean_temp_t1_c",
            "frontal_rms",
            "frontal_kurtosis",
            "rear_rms",
            "rear_kurtosis",
        ]
    ]

    print()
    print(
        snapshots.to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("FIRST 10 MEASUREMENTS")
    print("=" * 130)

    print()
    print(
        df[
            [
                "measurement_number",
                "timestamp_operating",
                "mean_abs_speed_rpm",
                "mean_abs_static_load_n",
                "mean_temp_t1_c",
                "frontal_rms",
                "frontal_kurtosis",
                "rear_rms",
                "rear_kurtosis",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("LAST 10 MEASUREMENTS")
    print("=" * 130)

    print()
    print(
        df[
            [
                "measurement_number",
                "timestamp_operating",
                "mean_abs_speed_rpm",
                "mean_abs_static_load_n",
                "mean_temp_t1_c",
                "frontal_rms",
                "frontal_kurtosis",
                "rear_rms",
                "rear_kurtosis",
            ]
        ]
        .tail(10)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("SAVED")
    print("=" * 130)

    print()
    print(OUTPUT_FILE)

    print()
    print("=" * 130)
    print("B03 MEASUREMENT SUMMARY COMPLETE")
    print("=" * 130)


if __name__ == "__main__":
    main()
