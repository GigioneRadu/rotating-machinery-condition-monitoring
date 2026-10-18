from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert


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
    / "cwru_normalized_physics_dataset.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLING_FREQUENCY = 12000
WINDOW_DURATION_SECONDS = 1.0
WINDOW_SIZE = int(
    SAMPLING_FREQUENCY
    * WINDOW_DURATION_SECONDS
)
MAX_WINDOWS_PER_FILE = 10

SEARCH_BAND_HZ = 3.0
EPSILON = 1e-12

ENVELOPE_BANDS = [
    (500, 1500),
    (1000, 3000),
    (1000, 5000),
    (3000, 5000),
]


# ============================================================
# CWRU DRIVE-END BEARING GEOMETRY
# SKF 6205-2RS JEM
# ============================================================

NUMBER_OF_ROLLING_ELEMENTS = 9
BALL_DIAMETER_INCH = 0.3126
PITCH_DIAMETER_INCH = 1.537
CONTACT_ANGLE_DEG = 0.0


# ============================================================
# FILE METADATA
# ============================================================

CWRU_FILES = [
    # 0.007 inch
    {"filename": "105.mat", "condition": "inner_race", "severity_inch": 0.007, "load_hp": 0, "rpm_expected": 1797},
    {"filename": "106.mat", "condition": "inner_race", "severity_inch": 0.007, "load_hp": 1, "rpm_expected": 1772},
    {"filename": "107.mat", "condition": "inner_race", "severity_inch": 0.007, "load_hp": 2, "rpm_expected": 1750},
    {"filename": "108.mat", "condition": "inner_race", "severity_inch": 0.007, "load_hp": 3, "rpm_expected": 1730},

    {"filename": "118.mat", "condition": "ball_fault", "severity_inch": 0.007, "load_hp": 0, "rpm_expected": 1797},
    {"filename": "119.mat", "condition": "ball_fault", "severity_inch": 0.007, "load_hp": 1, "rpm_expected": 1772},
    {"filename": "120.mat", "condition": "ball_fault", "severity_inch": 0.007, "load_hp": 2, "rpm_expected": 1750},
    {"filename": "121.mat", "condition": "ball_fault", "severity_inch": 0.007, "load_hp": 3, "rpm_expected": 1730},

    {"filename": "130.mat", "condition": "outer_race", "severity_inch": 0.007, "load_hp": 0, "rpm_expected": 1797},
    {"filename": "131.mat", "condition": "outer_race", "severity_inch": 0.007, "load_hp": 1, "rpm_expected": 1772},
    {"filename": "132.mat", "condition": "outer_race", "severity_inch": 0.007, "load_hp": 2, "rpm_expected": 1750},
    {"filename": "133.mat", "condition": "outer_race", "severity_inch": 0.007, "load_hp": 3, "rpm_expected": 1730},

    # 0.014 inch
    {"filename": "169.mat", "condition": "inner_race", "severity_inch": 0.014, "load_hp": 0, "rpm_expected": 1797},
    {"filename": "170.mat", "condition": "inner_race", "severity_inch": 0.014, "load_hp": 1, "rpm_expected": 1772},
    {"filename": "171.mat", "condition": "inner_race", "severity_inch": 0.014, "load_hp": 2, "rpm_expected": 1750},
    {"filename": "172.mat", "condition": "inner_race", "severity_inch": 0.014, "load_hp": 3, "rpm_expected": 1730},

    {"filename": "185.mat", "condition": "ball_fault", "severity_inch": 0.014, "load_hp": 0, "rpm_expected": 1797},
    {"filename": "186.mat", "condition": "ball_fault", "severity_inch": 0.014, "load_hp": 1, "rpm_expected": 1772},
    {"filename": "187.mat", "condition": "ball_fault", "severity_inch": 0.014, "load_hp": 2, "rpm_expected": 1750},
    {"filename": "188.mat", "condition": "ball_fault", "severity_inch": 0.014, "load_hp": 3, "rpm_expected": 1730},

    {"filename": "197.mat", "condition": "outer_race", "severity_inch": 0.014, "load_hp": 0, "rpm_expected": 1797},
    {"filename": "198.mat", "condition": "outer_race", "severity_inch": 0.014, "load_hp": 1, "rpm_expected": 1772},
    {"filename": "199.mat", "condition": "outer_race", "severity_inch": 0.014, "load_hp": 2, "rpm_expected": 1750},
    {"filename": "200.mat", "condition": "outer_race", "severity_inch": 0.014, "load_hp": 3, "rpm_expected": 1730},

    # 0.021 inch
    {"filename": "209.mat", "condition": "inner_race", "severity_inch": 0.021, "load_hp": 0, "rpm_expected": 1797},
    {"filename": "210.mat", "condition": "inner_race", "severity_inch": 0.021, "load_hp": 1, "rpm_expected": 1772},
    {"filename": "211.mat", "condition": "inner_race", "severity_inch": 0.021, "load_hp": 2, "rpm_expected": 1750},
    {"filename": "212.mat", "condition": "inner_race", "severity_inch": 0.021, "load_hp": 3, "rpm_expected": 1730},

    {"filename": "222.mat", "condition": "ball_fault", "severity_inch": 0.021, "load_hp": 0, "rpm_expected": 1797},
    {"filename": "223.mat", "condition": "ball_fault", "severity_inch": 0.021, "load_hp": 1, "rpm_expected": 1772},
    {"filename": "224.mat", "condition": "ball_fault", "severity_inch": 0.021, "load_hp": 2, "rpm_expected": 1750},
    {"filename": "225.mat", "condition": "ball_fault", "severity_inch": 0.021, "load_hp": 3, "rpm_expected": 1730},

    {"filename": "234.mat", "condition": "outer_race", "severity_inch": 0.021, "load_hp": 0, "rpm_expected": 1797},
    {"filename": "235.mat", "condition": "outer_race", "severity_inch": 0.021, "load_hp": 1, "rpm_expected": 1772},
    {"filename": "236.mat", "condition": "outer_race", "severity_inch": 0.021, "load_hp": 2, "rpm_expected": 1750},
    {"filename": "237.mat", "condition": "outer_race", "severity_inch": 0.021, "load_hp": 3, "rpm_expected": 1730},
]


# ============================================================
# MATLAB HELPERS
# ============================================================

def cwru_prefix_from_filename(file_path: Path) -> str:
    experiment_id = int(file_path.stem)
    return f"X{experiment_id:03d}"


def find_exact_experiment_key(
    data: dict,
    prefix: str,
    suffix: str,
    required: bool = True,
):
    exact_key = f"{prefix}{suffix}"

    if exact_key in data:
        return exact_key

    matches = [
        key
        for key in data.keys()
        if key.endswith(suffix)
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        if required:
            raise KeyError(
                f"No MATLAB variable for {prefix!r} "
                f"ending with {suffix!r}."
            )
        return None

    if required:
        raise KeyError(
            f"Could not uniquely select {suffix!r} "
            f"for {prefix!r}. Candidates: {matches}"
        )

    return None


def load_cwru_file(
    file_path: Path,
    rpm_fallback: float,
):
    data = loadmat(file_path)

    prefix = cwru_prefix_from_filename(
        file_path
    )

    de_key = find_exact_experiment_key(
        data=data,
        prefix=prefix,
        suffix="_DE_time",
        required=True,
    )

    rpm_key = find_exact_experiment_key(
        data=data,
        prefix=prefix,
        suffix="RPM",
        required=False,
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
        rpm_source = "metadata fallback"

    return (
        signal,
        rpm,
        rpm_source,
    )


# ============================================================
# BEARING FREQUENCIES
# ============================================================

def calculate_bearing_frequencies(
    rpm: float,
) -> dict:
    shaft = rpm / 60.0

    theta = np.deg2rad(
        CONTACT_ANGLE_DEG
    )

    ratio = (
        BALL_DIAMETER_INCH
        / PITCH_DIAMETER_INCH
    ) * np.cos(theta)

    bpfo = (
        NUMBER_OF_ROLLING_ELEMENTS
        / 2.0
        * shaft
        * (1.0 - ratio)
    )

    bpfi = (
        NUMBER_OF_ROLLING_ELEMENTS
        / 2.0
        * shaft
        * (1.0 + ratio)
    )

    bsf = (
        PITCH_DIAMETER_INCH
        / (2.0 * BALL_DIAMETER_INCH)
        * shaft
        * (1.0 - ratio**2)
    )

    ftf = (
        shaft
        / 2.0
        * (1.0 - ratio)
    )

    return {
        "shaft": shaft,
        "bpfo": bpfo,
        "bpfi": bpfi,
        "bsf": bsf,
        "ftf": ftf,
    }


# ============================================================
# SIGNAL HELPERS
# ============================================================

def calculate_fft(
    signal: np.ndarray,
):
    centered = (
        signal
        - np.mean(signal)
    )

    n = len(centered)

    window = np.hanning(
        n
    )

    spectrum = np.fft.rfft(
        centered * window
    )

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / SAMPLING_FREQUENCY,
    )

    amplitude = (
        2.0
        * np.abs(spectrum)
        / np.sum(window)
    )

    return (
        frequencies,
        amplitude,
    )


def calculate_envelope(
    signal: np.ndarray,
    low_hz: float,
    high_hz: float,
):
    nyquist = (
        SAMPLING_FREQUENCY
        / 2.0
    )

    low = (
        low_hz
        / nyquist
    )

    high = (
        high_hz
        / nyquist
    )

    b, a = butter(
        4,
        [low, high],
        btype="bandpass",
    )

    filtered = filtfilt(
        b,
        a,
        signal,
    )

    analytic = hilbert(
        filtered
    )

    envelope = np.abs(
        analytic
    )

    return (
        envelope
        - np.mean(envelope)
    )


def get_peak_amplitude_near(
    frequencies: np.ndarray,
    amplitude: np.ndarray,
    target_frequency: float,
) -> float:
    mask = (
        frequencies
        >= target_frequency
        - SEARCH_BAND_HZ
    ) & (
        frequencies
        <= target_frequency
        + SEARCH_BAND_HZ
    )

    if not np.any(mask):
        return 0.0

    return float(
        np.max(
            amplitude[mask]
        )
    )


def segment_signal(
    signal: np.ndarray,
):
    complete_windows = (
        len(signal)
        // WINDOW_SIZE
    )

    number_of_windows = min(
        complete_windows,
        MAX_WINDOWS_PER_FILE,
    )

    return [
        signal[
            index * WINDOW_SIZE:
            (index + 1) * WINDOW_SIZE
        ]
        for index in range(
            number_of_windows
        )
    ]


# ============================================================
# NORMALIZED PHYSICS FEATURES
# ============================================================

def extract_normalized_physics_features(
    signal: np.ndarray,
    rpm: float,
) -> dict:
    bearing = (
        calculate_bearing_frequencies(
            rpm
        )
    )

    feature_values = {}

    fault_families = {
        "bpfo": bearing["bpfo"],
        "bpfi": bearing["bpfi"],
        "bsf": bearing["bsf"],
    }

    for (
        band_low,
        band_high,
    ) in ENVELOPE_BANDS:
        envelope = (
            calculate_envelope(
                signal=signal,
                low_hz=band_low,
                high_hz=band_high,
            )
        )

        (
            frequencies,
            amplitude,
        ) = calculate_fft(
            envelope
        )

        band_name = (
            f"{band_low}_{band_high}"
        )

        for (
            family_name,
            fundamental_frequency,
        ) in fault_families.items():
            fundamental = (
                get_peak_amplitude_near(
                    frequencies,
                    amplitude,
                    fundamental_frequency,
                )
            )

            harmonic_2 = (
                get_peak_amplitude_near(
                    frequencies,
                    amplitude,
                    2.0
                    * fundamental_frequency,
                )
            )

            harmonic_3 = (
                get_peak_amplitude_near(
                    frequencies,
                    amplitude,
                    3.0
                    * fundamental_frequency,
                )
            )

            minus_sideband = (
                get_peak_amplitude_near(
                    frequencies,
                    amplitude,
                    fundamental_frequency
                    - bearing["shaft"],
                )
            )

            plus_sideband = (
                get_peak_amplitude_near(
                    frequencies,
                    amplitude,
                    fundamental_frequency
                    + bearing["shaft"],
                )
            )

            harmonic_sum = (
                fundamental
                + harmonic_2
                + harmonic_3
            )

            sideband_sum = (
                minus_sideband
                + plus_sideband
            )

            total_pattern = (
                harmonic_sum
                + sideband_sum
            )

            harmonics_to_fundamental = (
                (
                    harmonic_2
                    + harmonic_3
                )
                / (
                    fundamental
                    + EPSILON
                )
            )

            sidebands_to_fundamental = (
                sideband_sum
                / (
                    fundamental
                    + EPSILON
                )
            )

            fundamental_fraction = (
                fundamental
                / (
                    total_pattern
                    + EPSILON
                )
            )

            harmonic_fraction = (
                (
                    harmonic_2
                    + harmonic_3
                )
                / (
                    total_pattern
                    + EPSILON
                )
            )

            sideband_fraction = (
                sideband_sum
                / (
                    total_pattern
                    + EPSILON
                )
            )

            prefix = (
                f"{family_name}_"
                f"{band_name}"
            )

            # Intentionally use normalized pattern-shape features.
            # Absolute fundamental/harmonic amplitudes are NOT included.
            feature_values[
                f"{prefix}_harmonics_to_fundamental"
            ] = harmonics_to_fundamental

            feature_values[
                f"{prefix}_sidebands_to_fundamental"
            ] = sidebands_to_fundamental

            feature_values[
                f"{prefix}_fundamental_fraction"
            ] = fundamental_fraction

            feature_values[
                f"{prefix}_harmonic_fraction"
            ] = harmonic_fraction

            feature_values[
                f"{prefix}_sideband_fraction"
            ] = sideband_fraction

    return feature_values


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 120)
    print("BUILD CWRU NORMALIZED PHYSICS DATASET")
    print("=" * 120)

    print()
    print(
        "This dataset intentionally removes absolute fault-pattern "
        "amplitudes and keeps normalized harmonic/sideband shape features."
    )

    print(
        "The same BPFO, BPFI and BSF feature families are computed "
        "for every sample, so no true-label information is used "
        "during feature extraction."
    )

    records = []

    for file_info in CWRU_FILES:
        file_path = (
            RAW_DATA_DIR
            / file_info[
                "filename"
            ]
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing file: {file_path}"
            )

        (
            signal,
            rpm,
            rpm_source,
        ) = load_cwru_file(
            file_path=file_path,
            rpm_fallback=file_info[
                "rpm_expected"
            ],
        )

        windows = segment_signal(
            signal
        )

        print(
            f"{file_info['filename']:>7}"
            f" | {file_info['condition']:<11}"
            f" | severity={file_info['severity_inch']:.3f}"
            f" | load={file_info['load_hp']}"
            f" | rpm={rpm:.0f}"
            f" | windows={len(windows)}"
        )

        for (
            window_index,
            window,
        ) in enumerate(
            windows
        ):
            features = (
                extract_normalized_physics_features(
                    signal=window,
                    rpm=rpm,
                )
            )

            record = {
                "source_file": file_info[
                    "filename"
                ],
                "condition": file_info[
                    "condition"
                ],
                "severity_inch": file_info[
                    "severity_inch"
                ],
                "load_hp": file_info[
                    "load_hp"
                ],
                "rpm": rpm,
                "rpm_source": rpm_source,
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

    metadata_columns = [
        "source_file",
        "condition",
        "severity_inch",
        "load_hp",
        "rpm",
        "rpm_source",
        "window_index",
        "sampling_frequency",
        "window_duration_seconds",
    ]

    feature_columns = [
        column
        for column in dataframe.columns
        if column not in metadata_columns
    ]

    missing_values = int(
        dataframe[
            feature_columns
        ]
        .isna()
        .sum()
        .sum()
    )

    infinite_values = int(
        np.isinf(
            dataframe[
                feature_columns
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

    if missing_values:
        raise ValueError(
            "Dataset contains missing feature values."
        )

    if infinite_values:
        raise ValueError(
            "Dataset contains infinite feature values."
        )

    if duplicate_windows:
        raise ValueError(
            "Dataset contains duplicate source windows."
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
    print("DATASET SUMMARY")
    print("=" * 120)

    print()
    print(
        f"Shape              : "
        f"{dataframe.shape}"
    )

    print(
        f"Metadata columns   : "
        f"{len(metadata_columns)}"
    )

    print(
        f"Physics features   : "
        f"{len(feature_columns)}"
    )

    print(
        f"Missing values     : "
        f"{missing_values}"
    )

    print(
        f"Infinite values    : "
        f"{infinite_values}"
    )

    print(
        f"Duplicate windows  : "
        f"{duplicate_windows}"
    )

    print()
    print(
        "Samples per severity / condition:"
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
        "Feature families:"
    )

    print(
        f"- 3 bearing frequencies "
        f"(BPFO, BPFI, BSF)"
    )

    print(
        f"- {len(ENVELOPE_BANDS)} envelope bands"
    )

    print(
        "- 5 normalized pattern features per frequency/band"
    )

    print()
    print(
        "Saved dataset:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print("=" * 120)
    print("NORMALIZED PHYSICS DATASET COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
