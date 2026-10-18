from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cwru"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_fault_severity_dataset.csv"
)


# ============================================================
# DATASET SETTINGS
# ============================================================

SAMPLING_FREQUENCY = 12000

WINDOW_DURATION_SECONDS = 1.0

WINDOW_SIZE = int(
    SAMPLING_FREQUENCY
    * WINDOW_DURATION_SECONDS
)

MAX_WINDOWS_PER_FILE = 10


# ============================================================
# CWRU DRIVE-END BEARING GEOMETRY
# SKF 6205-2RS JEM
# ============================================================

NUMBER_OF_ROLLING_ELEMENTS = 9

BALL_DIAMETER_INCH = 0.3126

PITCH_DIAMETER_INCH = 1.537

CONTACT_ANGLE_DEG = 0.0


# ============================================================
# ENVELOPE / FREQUENCY SETTINGS
# ============================================================

ENVELOPE_LOW_HZ = 1000

ENVELOPE_HIGH_HZ = 5000

SEARCH_BAND_HZ = 3.0


# ============================================================
# CWRU FAULT FILE METADATA
#
# This dataset intentionally contains FAULT CLASSES ONLY:
#
# inner_race
# ball_fault
# outer_race @ 6:00
#
# at three seeded fault diameters:
#
# 0.007 in
# 0.014 in
# 0.021 in
#
# Healthy recordings are intentionally excluded from this
# cross-severity dataset because "healthy" has no fault diameter.
# Reusing the same healthy source files in both training and test
# severity groups would create an avoidable leakage problem.
# ============================================================

CWRU_FILES = [

    # ========================================================
    # 0.007 inch
    # ========================================================

    # Inner race
    {"filename": "105.mat", "condition": "inner_race", "severity_inch": 0.007, "load_hp": 0, "rpm_expected": 1797},
    {"filename": "106.mat", "condition": "inner_race", "severity_inch": 0.007, "load_hp": 1, "rpm_expected": 1772},
    {"filename": "107.mat", "condition": "inner_race", "severity_inch": 0.007, "load_hp": 2, "rpm_expected": 1750},
    {"filename": "108.mat", "condition": "inner_race", "severity_inch": 0.007, "load_hp": 3, "rpm_expected": 1730},

    # Ball
    {"filename": "118.mat", "condition": "ball_fault", "severity_inch": 0.007, "load_hp": 0, "rpm_expected": 1797},
    {"filename": "119.mat", "condition": "ball_fault", "severity_inch": 0.007, "load_hp": 1, "rpm_expected": 1772},
    {"filename": "120.mat", "condition": "ball_fault", "severity_inch": 0.007, "load_hp": 2, "rpm_expected": 1750},
    {"filename": "121.mat", "condition": "ball_fault", "severity_inch": 0.007, "load_hp": 3, "rpm_expected": 1730},

    # Outer race @ 6:00
    {"filename": "130.mat", "condition": "outer_race", "severity_inch": 0.007, "load_hp": 0, "rpm_expected": 1797},
    {"filename": "131.mat", "condition": "outer_race", "severity_inch": 0.007, "load_hp": 1, "rpm_expected": 1772},
    {"filename": "132.mat", "condition": "outer_race", "severity_inch": 0.007, "load_hp": 2, "rpm_expected": 1750},
    {"filename": "133.mat", "condition": "outer_race", "severity_inch": 0.007, "load_hp": 3, "rpm_expected": 1730},

    # ========================================================
    # 0.014 inch
    # ========================================================

    # Inner race
    {"filename": "169.mat", "condition": "inner_race", "severity_inch": 0.014, "load_hp": 0, "rpm_expected": 1797},
    {"filename": "170.mat", "condition": "inner_race", "severity_inch": 0.014, "load_hp": 1, "rpm_expected": 1772},
    {"filename": "171.mat", "condition": "inner_race", "severity_inch": 0.014, "load_hp": 2, "rpm_expected": 1750},
    {"filename": "172.mat", "condition": "inner_race", "severity_inch": 0.014, "load_hp": 3, "rpm_expected": 1730},

    # Ball
    {"filename": "185.mat", "condition": "ball_fault", "severity_inch": 0.014, "load_hp": 0, "rpm_expected": 1797},
    {"filename": "186.mat", "condition": "ball_fault", "severity_inch": 0.014, "load_hp": 1, "rpm_expected": 1772},
    {"filename": "187.mat", "condition": "ball_fault", "severity_inch": 0.014, "load_hp": 2, "rpm_expected": 1750},
    {"filename": "188.mat", "condition": "ball_fault", "severity_inch": 0.014, "load_hp": 3, "rpm_expected": 1730},

    # Outer race @ 6:00
    {"filename": "197.mat", "condition": "outer_race", "severity_inch": 0.014, "load_hp": 0, "rpm_expected": 1797},
    {"filename": "198.mat", "condition": "outer_race", "severity_inch": 0.014, "load_hp": 1, "rpm_expected": 1772},
    {"filename": "199.mat", "condition": "outer_race", "severity_inch": 0.014, "load_hp": 2, "rpm_expected": 1750},
    {"filename": "200.mat", "condition": "outer_race", "severity_inch": 0.014, "load_hp": 3, "rpm_expected": 1730},

    # ========================================================
    # 0.021 inch
    # ========================================================

    # Inner race
    {"filename": "209.mat", "condition": "inner_race", "severity_inch": 0.021, "load_hp": 0, "rpm_expected": 1797},
    {"filename": "210.mat", "condition": "inner_race", "severity_inch": 0.021, "load_hp": 1, "rpm_expected": 1772},
    {"filename": "211.mat", "condition": "inner_race", "severity_inch": 0.021, "load_hp": 2, "rpm_expected": 1750},
    {"filename": "212.mat", "condition": "inner_race", "severity_inch": 0.021, "load_hp": 3, "rpm_expected": 1730},

    # Ball
    {"filename": "222.mat", "condition": "ball_fault", "severity_inch": 0.021, "load_hp": 0, "rpm_expected": 1797},
    {"filename": "223.mat", "condition": "ball_fault", "severity_inch": 0.021, "load_hp": 1, "rpm_expected": 1772},
    {"filename": "224.mat", "condition": "ball_fault", "severity_inch": 0.021, "load_hp": 2, "rpm_expected": 1750},
    {"filename": "225.mat", "condition": "ball_fault", "severity_inch": 0.021, "load_hp": 3, "rpm_expected": 1730},

    # Outer race @ 6:00
    {"filename": "234.mat", "condition": "outer_race", "severity_inch": 0.021, "load_hp": 0, "rpm_expected": 1797},
    {"filename": "235.mat", "condition": "outer_race", "severity_inch": 0.021, "load_hp": 1, "rpm_expected": 1772},
    {"filename": "236.mat", "condition": "outer_race", "severity_inch": 0.021, "load_hp": 2, "rpm_expected": 1750},
    {"filename": "237.mat", "condition": "outer_race", "severity_inch": 0.021, "load_hp": 3, "rpm_expected": 1730},
]


FEATURE_COLUMNS = [
    "rms",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
    "shaft_amplitude",
    "bpfo_raw",
    "bpfi_raw",
    "bsf_raw",
    "ftf_raw",
    "bpfo_envelope",
    "bpfi_envelope",
    "bsf_envelope",
    "ftf_envelope",
]


# ============================================================
# MATLAB VARIABLE HELPERS
# ============================================================

def cwru_prefix_from_filename(
    file_path: Path,
) -> str:
    experiment_id = int(
        file_path.stem
    )

    return f"X{experiment_id:03d}"


def find_exact_experiment_key(
    data: dict,
    prefix: str,
    suffix: str,
    required: bool = True,
):
    exact_key = (
        f"{prefix}{suffix}"
    )

    if exact_key in data:
        return exact_key

    matches = [
        key
        for key in data.keys()
        if key.endswith(
            suffix
        )
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        if required:
            raise KeyError(
                f"No MATLAB variable for "
                f"{prefix!r} ending with "
                f"{suffix!r} was found."
            )

        return None

    if required:
        raise KeyError(
            f"Could not uniquely select "
            f"{suffix!r} for {prefix!r}. "
            f"Candidates: {matches}"
        )

    return None


# ============================================================
# LOAD ONE CWRU FILE
# ============================================================

def load_cwru_file(
    file_path: Path,
    rpm_fallback: float,
):
    data = loadmat(
        file_path
    )

    prefix = (
        cwru_prefix_from_filename(
            file_path
        )
    )

    de_key = (
        find_exact_experiment_key(
            data=data,
            prefix=prefix,
            suffix="_DE_time",
            required=True,
        )
    )

    rpm_key = (
        find_exact_experiment_key(
            data=data,
            prefix=prefix,
            suffix="RPM",
            required=False,
        )
    )

    signal = (
        data[de_key]
        .squeeze()
        .astype(float)
    )

    if rpm_key is not None:
        rpm = float(
            np.asarray(
                data[rpm_key]
            )
            .reshape(-1)[0]
        )

        rpm_source = "MAT file"

    else:
        rpm = float(
            rpm_fallback
        )

        rpm_source = (
            "metadata fallback"
        )

    return {
        "signal": signal,
        "rpm": rpm,
        "rpm_source": rpm_source,
        "de_key": de_key,
        "rpm_key": rpm_key,
        "prefix": prefix,
    }


# ============================================================
# BEARING FREQUENCIES
# ============================================================

def calculate_bearing_frequencies(
    rpm: float,
) -> dict:
    shaft_frequency = (
        rpm / 60.0
    )

    theta = np.deg2rad(
        CONTACT_ANGLE_DEG
    )

    n = (
        NUMBER_OF_ROLLING_ELEMENTS
    )

    d = BALL_DIAMETER_INCH

    D = PITCH_DIAMETER_INCH

    ratio = (
        d / D
    ) * np.cos(
        theta
    )

    bpfo = (
        n
        / 2.0
        * shaft_frequency
        * (
            1.0
            - ratio
        )
    )

    bpfi = (
        n
        / 2.0
        * shaft_frequency
        * (
            1.0
            + ratio
        )
    )

    bsf = (
        D
        / (
            2.0 * d
        )
        * shaft_frequency
        * (
            1.0
            - ratio ** 2
        )
    )

    ftf = (
        shaft_frequency
        / 2.0
        * (
            1.0
            - ratio
        )
    )

    return {
        "shaft": shaft_frequency,
        "bpfo": bpfo,
        "bpfi": bpfi,
        "bsf": bsf,
        "ftf": ftf,
    }


# ============================================================
# FFT
# ============================================================

def calculate_fft(
    signal: np.ndarray,
):
    centered_signal = (
        signal
        - np.mean(
            signal
        )
    )

    n = len(
        centered_signal
    )

    window = np.hanning(
        n
    )

    spectrum = np.fft.rfft(
        centered_signal
        * window
    )

    frequencies = np.fft.rfftfreq(
        n,
        d=(
            1.0
            / SAMPLING_FREQUENCY
        ),
    )

    amplitude = (
        2.0
        * np.abs(
            spectrum
        )
        / np.sum(
            window
        )
    )

    return (
        frequencies,
        amplitude,
    )


def get_peak_amplitude_near_frequency(
    frequencies: np.ndarray,
    amplitude: np.ndarray,
    target_frequency: float,
) -> float:
    mask = (
        frequencies
        >= (
            target_frequency
            - SEARCH_BAND_HZ
        )
    ) & (
        frequencies
        <= (
            target_frequency
            + SEARCH_BAND_HZ
        )
    )

    if not np.any(
        mask
    ):
        return 0.0

    return float(
        np.max(
            amplitude[
                mask
            ]
        )
    )


# ============================================================
# ENVELOPE
# ============================================================

def calculate_envelope(
    signal: np.ndarray,
) -> np.ndarray:
    nyquist = (
        SAMPLING_FREQUENCY
        / 2.0
    )

    low = (
        ENVELOPE_LOW_HZ
        / nyquist
    )

    high = (
        ENVELOPE_HIGH_HZ
        / nyquist
    )

    b, a = butter(
        4,
        [
            low,
            high,
        ],
        btype="bandpass",
    )

    filtered = filtfilt(
        b,
        a,
        signal,
    )

    analytic_signal = hilbert(
        filtered
    )

    envelope = np.abs(
        analytic_signal
    )

    return (
        envelope
        - np.mean(
            envelope
        )
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    signal: np.ndarray,
    rpm: float,
) -> dict:
    bearing_frequencies = (
        calculate_bearing_frequencies(
            rpm
        )
    )

    rms = float(
        np.sqrt(
            np.mean(
                signal ** 2
            )
        )
    )

    peak = float(
        np.max(
            np.abs(
                signal
            )
        )
    )

    peak_to_peak = float(
        np.ptp(
            signal
        )
    )

    crest_factor = (
        float(
            peak / rms
        )
        if rms > 0.0
        else 0.0
    )

    kurtosis_value = float(
        kurtosis(
            signal,
            fisher=False,
            bias=False,
        )
    )

    (
        raw_frequencies,
        raw_amplitude,
    ) = calculate_fft(
        signal
    )

    shaft_amplitude = (
        get_peak_amplitude_near_frequency(
            raw_frequencies,
            raw_amplitude,
            bearing_frequencies[
                "shaft"
            ],
        )
    )

    bpfo_raw = (
        get_peak_amplitude_near_frequency(
            raw_frequencies,
            raw_amplitude,
            bearing_frequencies[
                "bpfo"
            ],
        )
    )

    bpfi_raw = (
        get_peak_amplitude_near_frequency(
            raw_frequencies,
            raw_amplitude,
            bearing_frequencies[
                "bpfi"
            ],
        )
    )

    bsf_raw = (
        get_peak_amplitude_near_frequency(
            raw_frequencies,
            raw_amplitude,
            bearing_frequencies[
                "bsf"
            ],
        )
    )

    ftf_raw = (
        get_peak_amplitude_near_frequency(
            raw_frequencies,
            raw_amplitude,
            bearing_frequencies[
                "ftf"
            ],
        )
    )

    envelope_signal = (
        calculate_envelope(
            signal
        )
    )

    (
        envelope_frequencies,
        envelope_amplitude,
    ) = calculate_fft(
        envelope_signal
    )

    bpfo_envelope = (
        get_peak_amplitude_near_frequency(
            envelope_frequencies,
            envelope_amplitude,
            bearing_frequencies[
                "bpfo"
            ],
        )
    )

    bpfi_envelope = (
        get_peak_amplitude_near_frequency(
            envelope_frequencies,
            envelope_amplitude,
            bearing_frequencies[
                "bpfi"
            ],
        )
    )

    bsf_envelope = (
        get_peak_amplitude_near_frequency(
            envelope_frequencies,
            envelope_amplitude,
            bearing_frequencies[
                "bsf"
            ],
        )
    )

    ftf_envelope = (
        get_peak_amplitude_near_frequency(
            envelope_frequencies,
            envelope_amplitude,
            bearing_frequencies[
                "ftf"
            ],
        )
    )

    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "kurtosis": kurtosis_value,
        "shaft_amplitude": shaft_amplitude,
        "bpfo_raw": bpfo_raw,
        "bpfi_raw": bpfi_raw,
        "bsf_raw": bsf_raw,
        "ftf_raw": ftf_raw,
        "bpfo_envelope": bpfo_envelope,
        "bpfi_envelope": bpfi_envelope,
        "bsf_envelope": bsf_envelope,
        "ftf_envelope": ftf_envelope,
    }


# ============================================================
# SEGMENTATION
# ============================================================

def segment_signal(
    signal: np.ndarray,
):
    complete_windows = (
        len(
            signal
        )
        // WINDOW_SIZE
    )

    number_of_windows = min(
        complete_windows,
        MAX_WINDOWS_PER_FILE,
    )

    windows = []

    for window_index in range(
        number_of_windows
    ):
        start = (
            window_index
            * WINDOW_SIZE
        )

        end = (
            start
            + WINDOW_SIZE
        )

        windows.append(
            signal[
                start:end
            ]
        )

    return windows


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(
    dataframe: pd.DataFrame,
):
    missing_values = int(
        dataframe
        .isna()
        .sum()
        .sum()
    )

    infinite_values = int(
        np.isinf(
            dataframe[
                FEATURE_COLUMNS
            ]
            .to_numpy()
        )
        .sum()
    )

    duplicate_windows = int(
        dataframe
        .duplicated(
            subset=[
                "source_file",
                "window_index",
            ]
        )
        .sum()
    )

    if missing_values > 0:
        raise ValueError(
            "Dataset contains missing values."
        )

    if infinite_values > 0:
        raise ValueError(
            "Dataset contains infinite feature values."
        )

    if duplicate_windows > 0:
        raise ValueError(
            "Dataset contains duplicate source windows."
        )

    return (
        missing_values,
        infinite_values,
        duplicate_windows,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 110)
    print("BUILD CWRU CROSS-SEVERITY FAULT DATASET")
    print("=" * 110)

    print()
    print(
        f"Sampling frequency : "
        f"{SAMPLING_FREQUENCY} Hz"
    )

    print(
        f"Window duration    : "
        f"{WINDOW_DURATION_SECONDS:.1f} s"
    )

    print(
        f"Samples per window : "
        f"{WINDOW_SIZE}"
    )

    print(
        f"Max windows/file   : "
        f"{MAX_WINDOWS_PER_FILE}"
    )

    records = []

    for file_info in CWRU_FILES:
        filename = (
            file_info[
                "filename"
            ]
        )

        condition = (
            file_info[
                "condition"
            ]
        )

        severity_inch = (
            file_info[
                "severity_inch"
            ]
        )

        load_hp = (
            file_info[
                "load_hp"
            ]
        )

        rpm_expected = (
            file_info[
                "rpm_expected"
            ]
        )

        file_path = (
            RAW_DATA_DIR
            / filename
        )

        print()
        print("-" * 110)

        print(
            f"File       : "
            f"{filename}"
        )

        print(
            f"Condition  : "
            f"{condition}"
        )

        print(
            f"Severity   : "
            f"{severity_inch:.3f} in"
        )

        print(
            f"Load       : "
            f"{load_hp} HP"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing CWRU file: "
                f"{file_path}"
            )

        loaded = (
            load_cwru_file(
                file_path=file_path,
                rpm_fallback=rpm_expected,
            )
        )

        signal = loaded[
            "signal"
        ]

        rpm = loaded[
            "rpm"
        ]

        windows = (
            segment_signal(
                signal
            )
        )

        print(
            f"MAT prefix : "
            f"{loaded['prefix']}"
        )

        print(
            f"DE key     : "
            f"{loaded['de_key']}"
        )

        print(
            f"RPM key    : "
            f"{loaded['rpm_key'] if loaded['rpm_key'] else 'not available'}"
        )

        print(
            f"RPM        : "
            f"{rpm:.0f}"
        )

        print(
            f"RPM source : "
            f"{loaded['rpm_source']}"
        )

        print(
            f"Samples    : "
            f"{len(signal)}"
        )

        print(
            f"Windows    : "
            f"{len(windows)}"
        )

        for (
            window_index,
            window,
        ) in enumerate(
            windows
        ):
            features = (
                extract_features(
                    signal=window,
                    rpm=rpm,
                )
            )

            record = {
                "source_file": filename,
                "condition": condition,
                "severity_inch": severity_inch,
                "load_hp": load_hp,
                "rpm": rpm,
                "rpm_source": loaded[
                    "rpm_source"
                ],
                "window_index": window_index,
                "sampling_frequency": SAMPLING_FREQUENCY,
                "window_duration_seconds": WINDOW_DURATION_SECONDS,
            }

            record.update(
                features
            )

            records.append(
                record
            )

    dataframe = pd.DataFrame(
        records
    )

    (
        missing_values,
        infinite_values,
        duplicate_windows,
    ) = validate_dataset(
        dataframe
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
    print("=" * 110)
    print("DATASET VALIDATION")
    print("=" * 110)

    print()
    print(
        f"Missing values          : "
        f"{missing_values}"
    )

    print(
        f"Infinite feature values : "
        f"{infinite_values}"
    )

    print(
        f"Duplicate source windows: "
        f"{duplicate_windows}"
    )

    print()
    print(
        "Dataset validation: OK"
    )

    print()
    print("=" * 110)
    print("DATASET SUMMARY")
    print("=" * 110)

    print()
    print(
        f"Dataset shape: "
        f"{dataframe.shape}"
    )

    print()
    print(
        "Samples per severity and condition:"
    )

    print(
        dataframe
        .groupby(
            [
                "severity_inch",
                "condition",
            ]
        )
        .size()
        .to_string()
    )

    print()
    print(
        "Samples per severity, condition and load:"
    )

    print(
        dataframe
        .groupby(
            [
                "severity_inch",
                "condition",
                "load_hp",
            ]
        )
        .size()
        .to_string()
    )

    print()
    print(
        "Mean feature values by severity and condition:"
    )

    means = (
        dataframe
        .groupby(
            [
                "severity_inch",
                "condition",
            ]
        )[
            FEATURE_COLUMNS
        ]
        .mean()
    )

    print(
        means.to_string()
    )

    print()
    print(
        "Saved dataset:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 110)
    print("CROSS-SEVERITY DATASET BUILD COMPLETE")
    print("=" * 110)


if __name__ == "__main__":
    main()
