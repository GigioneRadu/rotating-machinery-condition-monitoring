from pathlib import Path

import numpy as np
from scipy.io import loadmat


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cwru"
)


# ============================================================
# CWRU FILE MAPPING
# ============================================================

CWRU_FILES = {
    "healthy": "97.mat",
    "inner_race": "105.mat",
    "ball_fault": "118.mat",
    "outer_race": "130.mat",
}


SAMPLING_FREQUENCY = 12000


# ============================================================
# HELPERS
# ============================================================

def find_key(
    data: dict,
    suffix: str,
    required: bool = True,
):
    """
    Find a MATLAB variable whose name ends with the given suffix.

    Example:
        suffix = "_DE_time"
        result = "X105_DE_time"

    If required=False and no key exists, return None.
    """

    matches = [
        key
        for key in data.keys()
        if key.endswith(suffix)
    ]

    if not matches:

        if required:
            raise KeyError(
                f"No key ending with {suffix!r} was found."
            )

        return None

    if len(matches) > 1:
        raise KeyError(
            f"Multiple keys ending with {suffix!r} were found: "
            f"{matches}"
        )

    return matches[0]


# ============================================================
# LOAD CWRU FILE
# ============================================================

def load_cwru_file(
    file_path: Path,
) -> dict:
    """
    Load one CWRU MATLAB file.

    Drive End vibration and RPM are required.

    Fan End and Base vibration channels are treated as optional
    because not all CWRU files contain exactly the same channels.
    """

    data = loadmat(
        file_path
    )


    # --------------------------------------------------------
    # FIND VARIABLES
    # --------------------------------------------------------

    de_key = find_key(
        data,
        "_DE_time",
        required=True,
    )

    fe_key = find_key(
        data,
        "_FE_time",
        required=False,
    )

    ba_key = find_key(
        data,
        "_BA_time",
        required=False,
    )

    rpm_key = find_key(
        data,
        "RPM",
        required=True,
    )


    # --------------------------------------------------------
    # DRIVE END SIGNAL
    # --------------------------------------------------------

    de_signal = (
        data[de_key]
        .squeeze()
        .astype(float)
    )


    # --------------------------------------------------------
    # FAN END SIGNAL
    # --------------------------------------------------------

    if fe_key is not None:

        fe_signal = (
            data[fe_key]
            .squeeze()
            .astype(float)
        )

    else:

        fe_signal = None


    # --------------------------------------------------------
    # BASE SIGNAL
    # --------------------------------------------------------

    if ba_key is not None:

        ba_signal = (
            data[ba_key]
            .squeeze()
            .astype(float)
        )

    else:

        ba_signal = None


    # --------------------------------------------------------
    # RPM
    # --------------------------------------------------------

    rpm_array = (
        data[rpm_key]
        .squeeze()
    )

    rpm = float(
        np.asarray(
            rpm_array
        ).reshape(-1)[0]
    )


    return {
        "de_signal": de_signal,
        "fe_signal": fe_signal,
        "ba_signal": ba_signal,
        "rpm": rpm,
        "de_key": de_key,
        "fe_key": fe_key,
        "ba_key": ba_key,
        "rpm_key": rpm_key,
    }


# ============================================================
# BASIC STATISTICS
# ============================================================

def calculate_signal_statistics(
    signal: np.ndarray,
) -> dict:
    """
    Calculate basic statistics for one vibration signal.
    """

    rms = float(
        np.sqrt(
            np.mean(
                signal ** 2
            )
        )
    )

    mean = float(
        np.mean(
            signal
        )
    )

    std = float(
        np.std(
            signal
        )
    )

    minimum = float(
        np.min(
            signal
        )
    )

    maximum = float(
        np.max(
            signal
        )
    )

    peak_to_peak = float(
        np.ptp(
            signal
        )
    )


    return {
        "mean": mean,
        "std": std,
        "rms": rms,
        "min": minimum,
        "max": maximum,
        "peak_to_peak": peak_to_peak,
    }


# ============================================================
# CHANNEL DESCRIPTION
# ============================================================

def print_channel_status(
    label: str,
    key,
):

    if key is None:

        print(
            f"{label:<11}: not available"
        )

    else:

        print(
            f"{label:<11}: {key}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("CWRU REAL DATA INSPECTION")
    print("=" * 80)

    print()

    print(
        f"Sampling frequency: "
        f"{SAMPLING_FREQUENCY} Hz"
    )


    # ========================================================
    # PROCESS EACH CONDITION
    # ========================================================

    for condition, filename in (
        CWRU_FILES.items()
    ):

        file_path = (
            DATA_DIR
            / filename
        )

        print()
        print("-" * 80)

        print(
            f"Condition : {condition}"
        )

        print(
            f"File      : {filename}"
        )


        # ----------------------------------------------------
        # FILE CHECK
        # ----------------------------------------------------

        if not file_path.exists():

            print(
                "ERROR: file not found"
            )

            continue


        # ----------------------------------------------------
        # LOAD DATA
        # ----------------------------------------------------

        result = load_cwru_file(
            file_path
        )


        de_signal = result[
            "de_signal"
        ]

        rpm = result[
            "rpm"
        ]


        # ----------------------------------------------------
        # SIGNAL DURATION
        # ----------------------------------------------------

        duration = (
            len(
                de_signal
            )
            / SAMPLING_FREQUENCY
        )


        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        statistics = (
            calculate_signal_statistics(
                de_signal
            )
        )


        # ----------------------------------------------------
        # CHANNEL INFORMATION
        # ----------------------------------------------------

        print()

        print_channel_status(
            "DE variable",
            result[
                "de_key"
            ],
        )

        print_channel_status(
            "FE variable",
            result[
                "fe_key"
            ],
        )

        print_channel_status(
            "BA variable",
            result[
                "ba_key"
            ],
        )

        print_channel_status(
            "RPM key",
            result[
                "rpm_key"
            ],
        )


        # ----------------------------------------------------
        # BASIC INFORMATION
        # ----------------------------------------------------

        print()

        print(
            f"RPM        : "
            f"{rpm:.0f}"
        )

        print(
            f"Samples    : "
            f"{len(de_signal)}"
        )

        print(
            f"Duration   : "
            f"{duration:.3f} s"
        )


        # ----------------------------------------------------
        # DRIVE END STATISTICS
        # ----------------------------------------------------

        print()

        print(
            "Drive End signal statistics:"
        )

        print(
            f"Mean       : "
            f"{statistics['mean']:.6f}"
        )

        print(
            f"Std        : "
            f"{statistics['std']:.6f}"
        )

        print(
            f"RMS        : "
            f"{statistics['rms']:.6f}"
        )

        print(
            f"Min        : "
            f"{statistics['min']:.6f}"
        )

        print(
            f"Max        : "
            f"{statistics['max']:.6f}"
        )

        print(
            f"Peak-to-peak: "
            f"{statistics['peak_to_peak']:.6f}"
        )


    # ========================================================
    # FINISH
    # ========================================================

    print()
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()