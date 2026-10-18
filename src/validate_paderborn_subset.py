from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PADERBORN_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "paderborn"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_n15_m07_f10_validation.csv"
)


# ============================================================
# SUBSET SETTINGS
# ============================================================

OPERATING_CONDITION = "N15_M07_F10"

BEARINGS = {
    "K001": "healthy",
    "KA01": "outer_race",
    "KI01": "inner_race",
}

EXPECTED_MEASUREMENTS_PER_BEARING = 20

VIBRATION_CHANNEL_NAME = "vibration_1"
SPEED_CHANNEL_NAME = "speed"


# ============================================================
# HELPERS
# ============================================================

def load_root_struct(
    file_path: Path,
):
    data = loadmat(
        file_path,
        squeeze_me=True,
        struct_as_record=False,
    )

    public_keys = [
        key
        for key in data.keys()
        if not key.startswith("__")
    ]

    if len(public_keys) != 1:
        raise ValueError(
            f"{file_path.name}: expected exactly one "
            f"top-level variable, found {public_keys}"
        )

    root_name = public_keys[0]

    return (
        root_name,
        data[root_name],
    )


def as_1d_array(
    value,
) -> np.ndarray:
    return np.asarray(
        value
    ).reshape(-1)


def find_y_channel(
    root,
    channel_name: str,
):
    y_channels = np.atleast_1d(
        root.Y
    )

    matches = []

    for channel in y_channels:
        name = str(
            getattr(
                channel,
                "Name",
                "",
            )
        )

        if name == channel_name:
            matches.append(
                channel
            )

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one Y channel named "
            f"{channel_name!r}, found {len(matches)}."
        )

    return matches[0]


def sampling_frequency_for_channel(
    root,
    channel,
) -> float:
    x_index = int(
        getattr(
            channel,
            "XIndex",
            0,
        )
    )

    x_channels = np.atleast_1d(
        root.X
    )

    if x_index < 1 or x_index > len(
        x_channels
    ):
        raise ValueError(
            f"Invalid XIndex={x_index}."
        )

    x_channel = x_channels[
        x_index - 1
    ]

    x_data = as_1d_array(
        x_channel.Data
    ).astype(float)

    if len(x_data) < 2:
        raise ValueError(
            "X axis contains fewer than 2 samples."
        )

    dt = float(
        np.median(
            np.diff(
                x_data
            )
        )
    )

    if dt <= 0:
        raise ValueError(
            f"Invalid sampling interval dt={dt}."
        )

    return 1.0 / dt


def inspect_measurement(
    file_path: Path,
    bearing_code: str,
    condition: str,
    measurement_index: int,
) -> dict:
    (
        root_name,
        root,
    ) = load_root_struct(
        file_path
    )

    vibration_channel = (
        find_y_channel(
            root,
            VIBRATION_CHANNEL_NAME,
        )
    )

    speed_channel = (
        find_y_channel(
            root,
            SPEED_CHANNEL_NAME,
        )
    )

    vibration = as_1d_array(
        vibration_channel.Data
    ).astype(float)

    speed = as_1d_array(
        speed_channel.Data
    ).astype(float)

    vibration_fs = (
        sampling_frequency_for_channel(
            root,
            vibration_channel,
        )
    )

    speed_fs = (
        sampling_frequency_for_channel(
            root,
            speed_channel,
        )
    )

    vibration_duration = (
        len(vibration)
        / vibration_fs
    )

    speed_duration = (
        len(speed)
        / speed_fs
    )

    vibration_rms = float(
        np.sqrt(
            np.mean(
                vibration ** 2
            )
        )
    )

    vibration_peak = float(
        np.max(
            np.abs(
                vibration
            )
        )
    )

    speed_mean = float(
        np.mean(
            speed
        )
    )

    speed_min = float(
        np.min(
            speed
        )
    )

    speed_max = float(
        np.max(
            speed
        )
    )

    return {
        "bearing_code": bearing_code,
        "condition": condition,
        "measurement_index": measurement_index,
        "source_file": file_path.name,
        "root_variable": root_name,
        "vibration_samples": len(
            vibration
        ),
        "vibration_fs_hz": vibration_fs,
        "vibration_duration_s": vibration_duration,
        "vibration_rms": vibration_rms,
        "vibration_peak": vibration_peak,
        "speed_samples": len(
            speed
        ),
        "speed_fs_hz": speed_fs,
        "speed_duration_s": speed_duration,
        "speed_mean_rpm": speed_mean,
        "speed_min_rpm": speed_min,
        "speed_max_rpm": speed_max,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 120)
    print("PADERBORN SUBSET VALIDATION")
    print("=" * 120)

    print()
    print(
        f"Operating condition : "
        f"{OPERATING_CONDITION}"
    )

    print(
        f"Expected files/class: "
        f"{EXPECTED_MEASUREMENTS_PER_BEARING}"
    )

    records = []

    for (
        bearing_code,
        condition,
    ) in BEARINGS.items():
        print()
        print("-" * 120)

        print(
            f"BEARING: {bearing_code}"
            f" | condition={condition}"
        )

        print("-" * 120)

        bearing_dir = (
            PADERBORN_DIR
            / bearing_code
        )

        if not bearing_dir.exists():
            raise FileNotFoundError(
                f"Missing bearing folder:\n"
                f"{bearing_dir}"
            )

        expected_files = [
            bearing_dir
            / (
                f"{OPERATING_CONDITION}_"
                f"{bearing_code}_"
                f"{measurement_index}.mat"
            )
            for measurement_index in range(
                1,
                EXPECTED_MEASUREMENTS_PER_BEARING + 1,
            )
        ]

        missing_files = [
            file_path
            for file_path in expected_files
            if not file_path.exists()
        ]

        if missing_files:
            print()
            print("Missing files:")

            for file_path in missing_files:
                print(
                    f"  {file_path.name}"
                )

            raise FileNotFoundError(
                f"{bearing_code}: "
                f"{len(missing_files)} expected files are missing."
            )

        for measurement_index, file_path in enumerate(
            expected_files,
            start=1,
        ):
            record = (
                inspect_measurement(
                    file_path=file_path,
                    bearing_code=bearing_code,
                    condition=condition,
                    measurement_index=measurement_index,
                )
            )

            records.append(
                record
            )

            print(
                f"{file_path.name:<34}"
                f" | vib_n={record['vibration_samples']:>6}"
                f" | vib_fs={record['vibration_fs_hz']:>9.2f}"
                f" | rpm={record['speed_mean_rpm']:>8.2f}"
                f" | vib_rms={record['vibration_rms']:.5f}"
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
    print("=" * 120)
    print("VALIDATION SUMMARY")
    print("=" * 120)

    print()
    print(
        f"Total measurements: "
        f"{len(dataframe)}"
    )

    print()
    print(
        "Measurements per class:"
    )

    print(
        dataframe
        .groupby(
            [
                "condition",
                "bearing_code",
            ]
        )
        .size()
        .to_string()
    )

    print()
    print(
        "Vibration sampling frequency by class:"
    )

    fs_summary = (
        dataframe
        .groupby(
            "condition"
        )[
            "vibration_fs_hz"
        ]
        .agg(
            [
                "min",
                "mean",
                "max",
            ]
        )
    )

    print(
        fs_summary.to_string()
    )

    print()
    print(
        "Speed by class [RPM]:"
    )

    speed_summary = (
        dataframe
        .groupby(
            "condition"
        )[
            "speed_mean_rpm"
        ]
        .agg(
            [
                "min",
                "mean",
                "max",
            ]
        )
    )

    print(
        speed_summary.to_string()
    )

    print()
    print(
        "Vibration RMS by class:"
    )

    rms_summary = (
        dataframe
        .groupby(
            "condition"
        )[
            "vibration_rms"
        ]
        .agg(
            [
                "min",
                "mean",
                "max",
            ]
        )
    )

    print(
        rms_summary.to_string()
    )

    print()
    print(
        "Vibration sample-count range:"
    )

    print(
        dataframe
        .groupby(
            "condition"
        )[
            "vibration_samples"
        ]
        .agg(
            [
                "min",
                "max",
            ]
        )
        .to_string()
    )

    print()
    print(
        "Saved validation CSV:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 120)
    print("PADERBORN SUBSET VALIDATION COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
