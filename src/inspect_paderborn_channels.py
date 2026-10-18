from pathlib import Path

import numpy as np
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

FILES_TO_INSPECT = [
    PADERBORN_DIR
    / "K001"
    / "N15_M07_F10_K001_1.mat",

    PADERBORN_DIR
    / "KA01"
    / "N15_M07_F10_KA01_1.mat",

    PADERBORN_DIR
    / "KI01"
    / "N15_M07_F10_KI01_1.mat",
]


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
        for key in data
        if not key.startswith("__")
    ]

    if len(public_keys) != 1:
        raise ValueError(
            f"Expected exactly one top-level variable in "
            f"{file_path.name}, found: {public_keys}"
        )

    root_name = public_keys[0]

    return (
        root_name,
        data[root_name],
    )


def safe_array(
    value,
):
    array = np.asarray(
        value
    ).reshape(-1)

    return array


def get_sampling_frequency_from_x(
    root,
    x_index: int,
):
    x_channels = np.atleast_1d(
        root.X
    )

    if x_index < 1 or x_index > len(
        x_channels
    ):
        return None

    x_channel = x_channels[
        x_index - 1
    ]

    x_data = safe_array(
        x_channel.Data
    )

    if len(x_data) < 2:
        return None

    differences = np.diff(
        x_data
    )

    median_step = float(
        np.median(
            differences
        )
    )

    if median_step <= 0:
        return None

    return (
        1.0
        / median_step
    )


# ============================================================
# INSPECT ONE FILE
# ============================================================

def inspect_file(
    file_path: Path,
):
    print()
    print("=" * 140)
    print(
        f"FILE: {file_path.name}"
    )
    print("=" * 140)

    if not file_path.exists():
        print(
            f"ERROR: file not found:\n{file_path}"
        )
        return

    (
        root_name,
        root,
    ) = load_root_struct(
        file_path
    )

    print()
    print(
        f"Top-level variable: "
        f"{root_name}"
    )

    x_channels = np.atleast_1d(
        root.X
    )

    y_channels = np.atleast_1d(
        root.Y
    )

    print()
    print(
        f"X channels: "
        f"{len(x_channels)}"
    )

    for index, channel in enumerate(
        x_channels,
        start=1,
    ):
        data = safe_array(
            channel.Data
        )

        name = getattr(
            channel,
            "Name",
            "",
        )

        raster = getattr(
            channel,
            "Raster",
            "",
        )

        print(
            f"  X[{index}]"
            f" | Name={name!r}"
            f" | Raster={raster!r}"
            f" | samples={len(data)}"
        )

        if len(data) >= 2:
            dt = float(
                np.median(
                    np.diff(
                        data
                    )
                )
            )

            if dt > 0:
                print(
                    f"       dt={dt:.10f} s"
                    f" | fs≈{1.0 / dt:.2f} Hz"
                )

    print()
    print(
        f"Y channels: "
        f"{len(y_channels)}"
    )

    for index, channel in enumerate(
        y_channels,
        start=1,
    ):
        name = getattr(
            channel,
            "Name",
            "",
        )

        unit = getattr(
            channel,
            "Unit",
            "",
        )

        raster = getattr(
            channel,
            "Raster",
            "",
        )

        device = getattr(
            channel,
            "Device",
            "",
        )

        x_index = int(
            getattr(
                channel,
                "XIndex",
                0,
            )
        )

        down_sampling = getattr(
            channel,
            "DownSampling",
            "",
        )

        data = safe_array(
            channel.Data
        )

        fs = (
            get_sampling_frequency_from_x(
                root,
                x_index,
            )
        )

        print()
        print(
            f"  Y[{index}]"
        )

        print(
            f"       Name         : "
            f"{name}"
        )

        print(
            f"       Unit         : "
            f"{unit}"
        )

        print(
            f"       Raster       : "
            f"{raster}"
        )

        print(
            f"       Device       : "
            f"{device}"
        )

        print(
            f"       XIndex       : "
            f"{x_index}"
        )

        print(
            f"       DownSampling : "
            f"{down_sampling}"
        )

        print(
            f"       Samples      : "
            f"{len(data)}"
        )

        if fs is not None:
            print(
                f"       fs from X    : "
                f"{fs:.2f} Hz"
            )

        preview_count = min(
            8,
            len(data),
        )

        print(
            f"       Preview      : "
            f"{data[:preview_count]}"
        )

        if len(data) > 0:
            numeric_data = data.astype(
                float,
                copy=False,
            )

            print(
                f"       Min          : "
                f"{np.min(numeric_data):.6g}"
            )

            print(
                f"       Max          : "
                f"{np.max(numeric_data):.6g}"
            )

            print(
                f"       Mean         : "
                f"{np.mean(numeric_data):.6g}"
            )

            print(
                f"       RMS          : "
                f"{np.sqrt(np.mean(numeric_data ** 2)):.6g}"
            )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 140)
    print(
        "PADERBORN CHANNEL INSPECTION"
    )
    print("=" * 140)

    print()
    print(
        "Goal: list every Y channel and determine which one "
        "contains the vibration signal and at what sampling rate."
    )

    for file_path in FILES_TO_INSPECT:
        inspect_file(
            file_path
        )

    print()
    print("=" * 140)
    print(
        "PADERBORN CHANNEL INSPECTION COMPLETE"
    )
    print("=" * 140)


if __name__ == "__main__":
    main()
