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
    / "paderborn_n15_m07_f10_dataset.csv"
)


# ============================================================
# DATASET SETTINGS
# ============================================================

OPERATING_CONDITION = "N15_M07_F10"

BEARINGS = {
    "K001": "healthy",
    "KA01": "outer_race",
    "KI01": "inner_race",
}

MEASUREMENTS_PER_BEARING = 20

# Paderborn documentation states 64 kHz for the two main signals.
SAMPLING_FREQUENCY = 64000

RECORDING_DURATION_SECONDS = 4
EXPECTED_RECORDING_SAMPLES = (
    SAMPLING_FREQUENCY
    * RECORDING_DURATION_SECONDS
)

WINDOW_DURATION_SECONDS = 1.0
WINDOW_SIZE = int(
    SAMPLING_FREQUENCY
    * WINDOW_DURATION_SECONDS
)

WINDOWS_PER_MEASUREMENT = 4

VIBRATION_CHANNEL_NAME = "vibration_1"
SPEED_CHANNEL_NAME = "speed"


# ============================================================
# PADERBORN 6203 BEARING GEOMETRY
# ============================================================

NUMBER_OF_ROLLING_ELEMENTS = 8
ROLLING_ELEMENT_DIAMETER_MM = 6.75
PITCH_DIAMETER_MM = 28.55
CONTACT_ANGLE_DEG = 0.0


# ============================================================
# SPECTRAL SETTINGS
# ============================================================

SEARCH_BAND_HZ = 3.0

# High-frequency resonance band for envelope extraction.
# This is a first benchmark choice, not claimed as universally optimal.
ENVELOPE_LOW_HZ = 5000
ENVELOPE_HIGH_HZ = 20000


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
# MAT HELPERS
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
        ROLLING_ELEMENT_DIAMETER_MM
        / PITCH_DIAMETER_MM
    ) * np.cos(theta)

    bpfo = (
        NUMBER_OF_ROLLING_ELEMENTS
        / 2.0
        * shaft
        * (
            1.0
            - ratio
        )
    )

    bpfi = (
        NUMBER_OF_ROLLING_ELEMENTS
        / 2.0
        * shaft
        * (
            1.0
            + ratio
        )
    )

    bsf = (
        PITCH_DIAMETER_MM
        / (
            2.0
            * ROLLING_ELEMENT_DIAMETER_MM
        )
        * shaft
        * (
            1.0
            - ratio ** 2
        )
    )

    ftf = (
        shaft
        / 2.0
        * (
            1.0
            - ratio
        )
    )

    return {
        "shaft": shaft,
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
    centered = (
        signal
        - np.mean(signal)
    )

    n = len(centered)

    window = np.hanning(
        n
    )

    spectrum = np.fft.rfft(
        centered
        * window
    )

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0
        / SAMPLING_FREQUENCY,
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


def get_peak_amplitude_near(
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

    if not np.any(mask):
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


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    signal: np.ndarray,
    rpm: float,
) -> dict:
    bearing = (
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
            np.abs(signal)
        )
    )

    peak_to_peak = float(
        np.ptp(signal)
    )

    crest_factor = (
        peak / rms
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
        get_peak_amplitude_near(
            raw_frequencies,
            raw_amplitude,
            bearing["shaft"],
        )
    )

    bpfo_raw = (
        get_peak_amplitude_near(
            raw_frequencies,
            raw_amplitude,
            bearing["bpfo"],
        )
    )

    bpfi_raw = (
        get_peak_amplitude_near(
            raw_frequencies,
            raw_amplitude,
            bearing["bpfi"],
        )
    )

    bsf_raw = (
        get_peak_amplitude_near(
            raw_frequencies,
            raw_amplitude,
            bearing["bsf"],
        )
    )

    ftf_raw = (
        get_peak_amplitude_near(
            raw_frequencies,
            raw_amplitude,
            bearing["ftf"],
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
        get_peak_amplitude_near(
            envelope_frequencies,
            envelope_amplitude,
            bearing["bpfo"],
        )
    )

    bpfi_envelope = (
        get_peak_amplitude_near(
            envelope_frequencies,
            envelope_amplitude,
            bearing["bpfi"],
        )
    )

    bsf_envelope = (
        get_peak_amplitude_near(
            envelope_frequencies,
            envelope_amplitude,
            bearing["bsf"],
        )
    )

    ftf_envelope = (
        get_peak_amplitude_near(
            envelope_frequencies,
            envelope_amplitude,
            bearing["ftf"],
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
# LOAD ONE MEASUREMENT
# ============================================================

def load_measurement(
    file_path: Path,
):
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

    if len(vibration) < EXPECTED_RECORDING_SAMPLES:
        raise ValueError(
            f"{file_path.name}: vibration has only "
            f"{len(vibration)} samples; expected at least "
            f"{EXPECTED_RECORDING_SAMPLES}."
        )

    # Paderborn documents a nominal 64 kHz sampling rate and
    # 4-second measurements. Some files contain a few extra host-service
    # samples, so use exactly the first 256000 samples for a uniform
    # four-second record.
    vibration = vibration[
        :EXPECTED_RECORDING_SAMPLES
    ]

    rpm = float(
        np.mean(
            speed
        )
    )

    return {
        "root_name": root_name,
        "vibration": vibration,
        "rpm": rpm,
        "original_vibration_samples": len(
            as_1d_array(
                vibration_channel.Data
            )
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 120)
    print("BUILD PADERBORN N15_M07_F10 DATASET")
    print("=" * 120)

    print()
    print(
        f"Nominal vibration fs : "
        f"{SAMPLING_FREQUENCY} Hz"
    )

    print(
        f"Record used          : "
        f"{RECORDING_DURATION_SECONDS} s "
        f"({EXPECTED_RECORDING_SAMPLES} samples)"
    )

    print(
        f"Window duration      : "
        f"{WINDOW_DURATION_SECONDS:.1f} s"
    )

    print(
        f"Windows/measurement  : "
        f"{WINDOWS_PER_MEASUREMENT}"
    )

    print(
        f"Envelope band        : "
        f"{ENVELOPE_LOW_HZ}-{ENVELOPE_HIGH_HZ} Hz"
    )

    print()
    print(
        "6203 theoretical frequencies at 1500 RPM:"
    )

    theoretical = (
        calculate_bearing_frequencies(
            1500.0
        )
    )

    for name, value in theoretical.items():
        print(
            f"  {name.upper():<5}: "
            f"{value:.3f} Hz"
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

        for measurement_index in range(
            1,
            MEASUREMENTS_PER_BEARING + 1,
        ):
            file_path = (
                bearing_dir
                / (
                    f"{OPERATING_CONDITION}_"
                    f"{bearing_code}_"
                    f"{measurement_index}.mat"
                )
            )

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Missing file: {file_path}"
                )

            loaded = load_measurement(
                file_path
            )

            vibration = loaded[
                "vibration"
            ]

            rpm = loaded[
                "rpm"
            ]

            print(
                f"{file_path.name:<34}"
                f" | original_n="
                f"{loaded['original_vibration_samples']:>6}"
                f" | used_n="
                f"{len(vibration):>6}"
                f" | rpm={rpm:>8.2f}"
            )

            for window_index in range(
                WINDOWS_PER_MEASUREMENT
            ):
                start = (
                    window_index
                    * WINDOW_SIZE
                )

                end = (
                    start
                    + WINDOW_SIZE
                )

                window = vibration[
                    start:end
                ]

                features = (
                    extract_features(
                        signal=window,
                        rpm=rpm,
                    )
                )

                record = {
                    "bearing_code": bearing_code,
                    "condition": condition,
                    "operating_condition": OPERATING_CONDITION,
                    "measurement_index": measurement_index,
                    "source_file": file_path.name,
                    "window_index": window_index,
                    "rpm": rpm,
                    "sampling_frequency": SAMPLING_FREQUENCY,
                    "window_duration_seconds": WINDOW_DURATION_SECONDS,
                    "original_vibration_samples": loaded[
                        "original_vibration_samples"
                    ],
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
        f"Dataset shape      : "
        f"{dataframe.shape}"
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
        "Samples per class:"
    )

    print(
        dataframe
        .groupby(
            "condition"
        )
        .size()
        .to_string()
    )

    print()
    print(
        "Mean features by class:"
    )

    print(
        dataframe
        .groupby(
            "condition"
        )[
            FEATURE_COLUMNS
        ]
        .mean()
        .to_string()
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
    print("PADERBORN DATASET BUILD COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
