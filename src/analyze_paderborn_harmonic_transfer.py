from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PADERBORN_DIR = PROJECT_ROOT / "data" / "raw" / "paderborn"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_harmonic_envelope_transfer_diagnostic.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_harmonic_envelope_transfer_summary.csv"
)


# ============================================================
# DATASET
# ============================================================

OPERATING_CONDITION = "N15_M07_F10"

BEARINGS = {
    "KA01": ("outer_race", "artificial"),
    "KA05": ("outer_race", "artificial"),
    "KA07": ("outer_race", "artificial"),
    "KI01": ("inner_race", "artificial"),
    "KI05": ("inner_race", "artificial"),
    "KI07": ("inner_race", "artificial"),
    "KA04": ("outer_race", "real_damage"),
    "KA16": ("outer_race", "real_damage"),
    "KI04": ("inner_race", "real_damage"),
    "KI14": ("inner_race", "real_damage"),
}

MEASUREMENTS_PER_BEARING = 20

SAMPLING_FREQUENCY = 64000
WINDOW_SIZE = 64000
MAX_WINDOWS_PER_MEASUREMENT = 4

VIBRATION_CHANNEL_NAME = "vibration_1"
SPEED_CHANNEL_NAME = "speed"


# ============================================================
# BEARING GEOMETRY
# ============================================================

NUMBER_OF_ROLLING_ELEMENTS = 8
ROLLING_ELEMENT_DIAMETER_MM = 6.75
PITCH_DIAMETER_MM = 28.55
CONTACT_ANGLE_DEG = 0.0


# ============================================================
# DIAGNOSTIC SETTINGS
# ============================================================

ENVELOPE_BANDS = {
    "1_5k": (1000, 5000),
    "5_10k": (5000, 10000),
    "10_20k": (10000, 20000),
    "5_20k": (5000, 20000),
}

HARMONICS = (1, 2, 3)

TARGET_HALF_WIDTH_HZ = 3.0
LOCAL_NOISE_HALF_WIDTH_HZ = 25.0
EPS = 1e-12


# ============================================================
# MAT HELPERS
# ============================================================

def load_root_struct(file_path: Path):
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
            f"{file_path.name}: expected one top-level variable, "
            f"found {public_keys}"
        )

    root_name = public_keys[0]
    return data[root_name]


def as_1d_array(value) -> np.ndarray:
    return np.asarray(value).reshape(-1)


def find_y_channel(root, channel_name: str):
    matches = []

    for channel in np.atleast_1d(root.Y):
        if str(getattr(channel, "Name", "")) == channel_name:
            matches.append(channel)

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one channel {channel_name!r}, "
            f"found {len(matches)}"
        )

    return matches[0]


# ============================================================
# BEARING FREQUENCIES
# ============================================================

def calculate_bearing_frequencies(rpm: float) -> dict:
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
        * (1.0 - ratio)
    )

    bpfi = (
        NUMBER_OF_ROLLING_ELEMENTS
        / 2.0
        * shaft
        * (1.0 + ratio)
    )

    return {
        "shaft": shaft,
        "bpfo": bpfo,
        "bpfi": bpfi,
    }


# ============================================================
# SPECTRAL HELPERS
# ============================================================

def calculate_fft(signal: np.ndarray):
    centered = signal - np.mean(signal)
    n = len(centered)

    window = np.hanning(n)

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

    return frequencies, amplitude


def peak_near(
    frequencies: np.ndarray,
    amplitude: np.ndarray,
    target_frequency: float,
) -> float:
    mask = (
        frequencies >= (
            target_frequency
            - TARGET_HALF_WIDTH_HZ
        )
    ) & (
        frequencies <= (
            target_frequency
            + TARGET_HALF_WIDTH_HZ
        )
    )

    if not np.any(mask):
        return 0.0

    return float(
        np.max(
            amplitude[mask]
        )
    )


def local_noise_floor(
    frequencies: np.ndarray,
    amplitude: np.ndarray,
    target_frequency: float,
) -> float:
    broad_mask = (
        frequencies >= (
            target_frequency
            - LOCAL_NOISE_HALF_WIDTH_HZ
        )
    ) & (
        frequencies <= (
            target_frequency
            + LOCAL_NOISE_HALF_WIDTH_HZ
        )
    )

    target_mask = (
        frequencies >= (
            target_frequency
            - TARGET_HALF_WIDTH_HZ
        )
    ) & (
        frequencies <= (
            target_frequency
            + TARGET_HALF_WIDTH_HZ
        )
    )

    noise_mask = (
        broad_mask
        & ~target_mask
    )

    values = amplitude[
        noise_mask
    ]

    if len(values) == 0:
        return EPS

    return float(
        np.median(values)
        + EPS
    )


def normalized_peak_ratio(
    frequencies: np.ndarray,
    amplitude: np.ndarray,
    target_frequency: float,
) -> float:
    peak = peak_near(
        frequencies,
        amplitude,
        target_frequency,
    )

    floor = local_noise_floor(
        frequencies,
        amplitude,
        target_frequency,
    )

    return float(
        peak / floor
    )


# ============================================================
# ENVELOPE
# ============================================================

def calculate_envelope(
    signal: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    nyquist = (
        SAMPLING_FREQUENCY
        / 2.0
    )

    low = low_hz / nyquist
    high = high_hz / nyquist

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

    envelope = np.abs(
        hilbert(
            filtered
        )
    )

    return (
        envelope
        - np.mean(
            envelope
        )
    )


# ============================================================
# SCORES
# ============================================================

def harmonic_score(
    frequencies: np.ndarray,
    amplitude: np.ndarray,
    base_frequency: float,
) -> float:
    ratios = []

    for harmonic in HARMONICS:
        target = (
            harmonic
            * base_frequency
        )

        ratio = normalized_peak_ratio(
            frequencies,
            amplitude,
            target,
        )

        ratios.append(
            np.log1p(
                ratio
            )
        )

    return float(
        np.mean(
            ratios
        )
    )


def inner_sideband_score(
    frequencies: np.ndarray,
    amplitude: np.ndarray,
    bpfi: float,
    shaft: float,
) -> float:
    values = []

    for harmonic in HARMONICS:
        center = (
            harmonic
            * bpfi
        )

        for sideband_index in (
            -1,
            1,
        ):
            target = (
                center
                + sideband_index
                * shaft
            )

            ratio = normalized_peak_ratio(
                frequencies,
                amplitude,
                target,
            )

            values.append(
                np.log1p(
                    ratio
                )
            )

    return float(
        np.mean(
            values
        )
    )


# ============================================================
# LOAD MEASUREMENT
# ============================================================

def load_measurement(
    file_path: Path,
):
    root = load_root_struct(
        file_path
    )

    vibration_channel = find_y_channel(
        root,
        VIBRATION_CHANNEL_NAME,
    )

    speed_channel = find_y_channel(
        root,
        SPEED_CHANNEL_NAME,
    )

    vibration = as_1d_array(
        vibration_channel.Data
    ).astype(float)

    speed = as_1d_array(
        speed_channel.Data
    ).astype(float)

    rpm = float(
        np.mean(
            speed
        )
    )

    windows_to_use = min(
        len(vibration)
        // WINDOW_SIZE,
        MAX_WINDOWS_PER_MEASUREMENT,
    )

    return (
        vibration,
        rpm,
        windows_to_use,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 130)
    print(
        "PADERBORN HARMONIC + ENVELOPE-BAND TRANSFER DIAGNOSTIC"
    )
    print("=" * 130)

    print()
    print(
        "Purpose: compare artificial vs real-damage bearings "
        "using normalized BPFO/BPFI harmonics and BPFI sidebands."
    )

    print(
        "Scores are normalized by a local spectral noise floor, "
        "so absolute vibration amplitude matters less."
    )

    records = []

    for (
        bearing_code,
        (
            condition,
            damage_origin,
        ),
    ) in BEARINGS.items():
        print()
        print("-" * 130)
        print(
            f"{bearing_code}"
            f" | {condition}"
            f" | {damage_origin}"
        )
        print("-" * 130)

        bearing_dir = (
            PADERBORN_DIR
            / bearing_code
        )

        if not bearing_dir.exists():
            raise FileNotFoundError(
                f"Missing folder:\n"
                f"{bearing_dir}"
            )

        bearing_window_count = 0

        for measurement_index in range(
            1,
            MEASUREMENTS_PER_BEARING
            + 1,
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
                    f"Missing file:\n"
                    f"{file_path}"
                )

            (
                vibration,
                rpm,
                windows_to_use,
            ) = load_measurement(
                file_path
            )

            bearing_freq = (
                calculate_bearing_frequencies(
                    rpm
                )
            )

            for window_index in range(
                windows_to_use
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

                for (
                    band_name,
                    (
                        low_hz,
                        high_hz,
                    ),
                ) in ENVELOPE_BANDS.items():
                    envelope = (
                        calculate_envelope(
                            window,
                            low_hz,
                            high_hz,
                        )
                    )

                    (
                        frequencies,
                        amplitude,
                    ) = calculate_fft(
                        envelope
                    )

                    bpfo_score = (
                        harmonic_score(
                            frequencies,
                            amplitude,
                            bearing_freq[
                                "bpfo"
                            ],
                        )
                    )

                    bpfi_score = (
                        harmonic_score(
                            frequencies,
                            amplitude,
                            bearing_freq[
                                "bpfi"
                            ],
                        )
                    )

                    inner_sideband = (
                        inner_sideband_score(
                            frequencies,
                            amplitude,
                            bearing_freq[
                                "bpfi"
                            ],
                            bearing_freq[
                                "shaft"
                            ],
                        )
                    )

                    inner_combined = float(
                        0.7
                        * bpfi_score
                        + 0.3
                        * inner_sideband
                    )

                    physics_prediction = (
                        "outer_race"
                        if bpfo_score
                        > inner_combined
                        else "inner_race"
                    )

                    records.append(
                        {
                            "bearing_code": bearing_code,
                            "condition": condition,
                            "damage_origin": damage_origin,
                            "source_file": file_path.name,
                            "measurement_index": measurement_index,
                            "window_index": window_index,
                            "rpm": rpm,
                            "band": band_name,
                            "band_low_hz": low_hz,
                            "band_high_hz": high_hz,
                            "bpfo_harmonic_score": bpfo_score,
                            "bpfi_harmonic_score": bpfi_score,
                            "bpfi_sideband_score": inner_sideband,
                            "inner_combined_score": inner_combined,
                            "score_margin_outer_minus_inner": (
                                bpfo_score
                                - inner_combined
                            ),
                            "physics_prediction": physics_prediction,
                            "physics_correct": (
                                physics_prediction
                                == condition
                            ),
                        }
                    )

                bearing_window_count += 1

        print(
            f"Processed windows: "
            f"{bearing_window_count}"
        )

    dataframe = pd.DataFrame(
        records
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary = (
        dataframe
        .groupby(
            [
                "damage_origin",
                "condition",
                "bearing_code",
                "band",
            ],
            as_index=False,
        )
        .agg(
            windows=(
                "physics_correct",
                "size",
            ),
            physics_accuracy=(
                "physics_correct",
                "mean",
            ),
            mean_bpfo_score=(
                "bpfo_harmonic_score",
                "mean",
            ),
            mean_bpfi_score=(
                "bpfi_harmonic_score",
                "mean",
            ),
            mean_bpfi_sideband_score=(
                "bpfi_sideband_score",
                "mean",
            ),
            mean_inner_combined_score=(
                "inner_combined_score",
                "mean",
            ),
            mean_margin_outer_minus_inner=(
                "score_margin_outer_minus_inner",
                "mean",
            ),
        )
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print("=" * 130)
    print("PER-BEARING / PER-BAND SUMMARY")
    print("=" * 130)

    print()
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("BAND SUMMARY")
    print("=" * 130)

    band_summary = (
        dataframe
        .groupby(
            [
                "damage_origin",
                "band",
            ],
            as_index=False,
        )
        .agg(
            accuracy=(
                "physics_correct",
                "mean",
            ),
            mean_abs_margin=(
                "score_margin_outer_minus_inner",
                lambda values: float(
                    np.mean(
                        np.abs(values)
                    )
                ),
            ),
        )
    )

    print()
    print(
        band_summary.to_string(
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

    print(
        SUMMARY_FILE
    )

    print()
    print("=" * 130)
    print("HARMONIC TRANSFER DIAGNOSTIC COMPLETE")
    print("=" * 130)


if __name__ == "__main__":
    main()
