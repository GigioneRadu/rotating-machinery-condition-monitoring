from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import kurtosis


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

VIBRATION_DIR = (
    B01_DIR
    / "vibrationData"
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
    / "paderborn_rtf_b01_measurement_summary.csv"
)


# ============================================================
# SETTINGS
# ============================================================

FRONTAL_CHANNEL = "accHorizFrontal_C"
REAR_CHANNEL = "accHorizRear_A"
TIME_CHANNEL = "measTime"


# ============================================================
# HELPERS
# ============================================================

def load_mat_measurement(file_path: Path):
    data = loadmat(
        file_path,
        squeeze_me=True,
        struct_as_record=False,
    )

    required = {
        FRONTAL_CHANNEL,
        REAR_CHANNEL,
        TIME_CHANNEL,
    }

    missing = (
        required
        - set(data.keys())
    )

    if missing:
        raise KeyError(
            f"{file_path.name}: missing variables "
            f"{sorted(missing)}"
        )

    frontal = np.asarray(
        data[FRONTAL_CHANNEL]
    ).reshape(-1).astype(float)

    rear = np.asarray(
        data[REAR_CHANNEL]
    ).reshape(-1).astype(float)

    time = np.asarray(
        data[TIME_CHANNEL]
    ).reshape(-1).astype(float)

    if not (
        len(frontal)
        == len(rear)
        == len(time)
    ):
        raise ValueError(
            f"{file_path.name}: channel lengths differ."
        )

    if len(time) < 2:
        raise ValueError(
            f"{file_path.name}: time vector too short."
        )

    dt = float(
        np.median(
            np.diff(time)
        )
    )

    fs = (
        1.0 / dt
    )

    duration = float(
        time[-1]
        - time[0]
        + dt
    )

    return (
        frontal,
        rear,
        time,
        fs,
        duration,
    )


def signal_features(
    signal: np.ndarray,
):
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
            np.abs(
                centered
            )
        )
    )

    peak_to_peak = float(
        np.ptp(
            centered
        )
    )

    crest_factor = (
        peak / rms
        if rms > 0.0
        else 0.0
    )

    kurtosis_value = float(
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
        "kurtosis": kurtosis_value,
        "std": std,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 130)
    print(
        "PADERBORN RUN-TO-FAILURE B01 MEASUREMENT SUMMARY"
    )
    print("=" * 130)

    mat_files = sorted(
        VIBRATION_DIR.glob(
            "data_B01_M*.mat"
        )
    )

    if not mat_files:
        raise FileNotFoundError(
            f"No B01 MAT files found in:\n"
            f"{VIBRATION_DIR}"
        )

    operating = pd.read_csv(
        OPERATING_CONDITIONS_FILE
    )

    temperatures = pd.read_csv(
        TEMPERATURES_FILE
    )

    if len(operating) != len(mat_files):
        raise ValueError(
            f"Operating-condition rows ({len(operating)}) "
            f"do not match MAT files ({len(mat_files)})."
        )

    if len(temperatures) != len(mat_files):
        raise ValueError(
            f"Temperature rows ({len(temperatures)}) "
            f"do not match MAT files ({len(mat_files)})."
        )

    print()
    print(
        f"Measurements: {len(mat_files)}"
    )

    records = []

    for index, file_path in enumerate(
        mat_files
    ):
        (
            frontal,
            rear,
            time,
            fs,
            duration,
        ) = load_mat_measurement(
            file_path
        )

        frontal_features = signal_features(
            frontal
        )

        rear_features = signal_features(
            rear
        )

        op_row = operating.iloc[
            index
        ]

        temp_row = temperatures.iloc[
            index
        ]

        record = {
            "measurement_number": (
                index + 1
            ),
            "source_file": file_path.name,
            "timestamp": op_row[
                "Time"
            ],
            "samples": len(
                frontal
            ),
            "sampling_frequency_hz": fs,
            "duration_seconds": duration,
            "set_dynamic_load_n": op_row[
                "setDynLoad / N"
            ],
            "peak_dynamic_load_n": op_row[
                "peak_dynLoad / N"
            ],
            "set_static_load_n": op_row[
                "setStatLoad / N"
            ],
            "mean_abs_static_load_n": op_row[
                "meanAbs_statLoad / N"
            ],
            "set_speed_rpm": op_row[
                "setSpeed / rpm"
            ],
            "mean_abs_speed_rpm": op_row[
                "meanAbs_speed / rpm"
            ],
            "mean_temp_t1_c": temp_row[
                "Mean Abs. Temp. T1 / °C"
            ],
            "mean_temp_t2_c": temp_row[
                "Mean Abs. Temp. T2 / °C"
            ],
            "room_temp_c": temp_row[
                "Mean Room Temp. / °C"
            ],
        }

        for key, value in frontal_features.items():
            record[
                f"frontal_{key}"
            ] = value

        for key, value in rear_features.items():
            record[
                f"rear_{key}"
            ] = value

        records.append(
            record
        )

        if (
            index == 0
            or (index + 1) % 50 == 0
            or index == len(mat_files) - 1
        ):
            print(
                f"Processed "
                f"{index + 1:>3}/{len(mat_files)}"
            )

    dataframe = pd.DataFrame(
        records
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
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
        f"{dataframe['sampling_frequency_hz'].min():.3f} Hz"
    )

    print(
        f"Sampling frequency max : "
        f"{dataframe['sampling_frequency_hz'].max():.3f} Hz"
    )

    print(
        f"Sampling frequency mean: "
        f"{dataframe['sampling_frequency_hz'].mean():.3f} Hz"
    )

    print(
        f"Duration min           : "
        f"{dataframe['duration_seconds'].min():.6f} s"
    )

    print(
        f"Duration max           : "
        f"{dataframe['duration_seconds'].max():.6f} s"
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
        print(
            f"{column}:"
        )

        print(
            f"  min  = "
            f"{dataframe[column].min():.3f}"
        )

        print(
            f"  mean = "
            f"{dataframe[column].mean():.3f}"
        )

        print(
            f"  max  = "
            f"{dataframe[column].max():.3f}"
        )

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
            f" min={dataframe[column].min():.3f}"
            f" | mean={dataframe[column].mean():.3f}"
            f" | max={dataframe[column].max():.3f}"
        )

    print()
    print("=" * 130)
    print("VIBRATION TREND SNAPSHOTS")
    print("=" * 130)

    snapshot_indices = sorted(
        set(
            [
                0,
                9,
                24,
                49,
                99,
                149,
                199,
                249,
                299,
                349,
                len(dataframe) - 1,
            ]
        )
    )

    snapshot_indices = [
        index
        for index in snapshot_indices
        if 0 <= index < len(dataframe)
    ]

    snapshot_columns = [
        "measurement_number",
        "mean_abs_speed_rpm",
        "mean_abs_static_load_n",
        "mean_temp_t1_c",
        "frontal_rms",
        "frontal_kurtosis",
        "rear_rms",
        "rear_kurtosis",
    ]

    print()
    print(
        dataframe.iloc[
            snapshot_indices
        ][
            snapshot_columns
        ]
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("FIRST 10 MEASUREMENTS")
    print("=" * 130)

    print()
    print(
        dataframe[
            snapshot_columns
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
        dataframe[
            snapshot_columns
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
    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 130)
    print(
        "B01 MEASUREMENT SUMMARY COMPLETE"
    )
    print("=" * 130)


if __name__ == "__main__":
    main()
