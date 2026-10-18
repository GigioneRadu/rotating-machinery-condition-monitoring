from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "paderborn_ki16_diagnostic"
)

WINDOW_RESULTS_FILE = (
    OUTPUT_DIR
    / "paderborn_ki16_posthoc_window_scores.csv"
)

BEARING_SUMMARY_FILE = (
    OUTPUT_DIR
    / "paderborn_ki16_posthoc_bearing_summary.csv"
)


# ============================================================
# POST-HOC DIAGNOSTIC SET
# ============================================================

OPERATING_CONDITION = "N15_M07_F10"

BEARINGS = {
    "KI16": "inner_race",
    "KI18": "inner_race",
    "KI21": "inner_race",
}

MEASUREMENTS_PER_BEARING = 20

SAMPLING_FREQUENCY = 64000
WINDOW_SIZE = 64000
MAX_WINDOWS_PER_MEASUREMENT = 4

VIBRATION_CHANNEL_NAME = "vibration_1"
SPEED_CHANNEL_NAME = "speed"


# ============================================================
# FROZEN PHYSICS SETTINGS
# ============================================================

NUMBER_OF_ROLLING_ELEMENTS = 8
ROLLING_ELEMENT_DIAMETER_MM = 6.75
PITCH_DIAMETER_MM = 28.55
CONTACT_ANGLE_DEG = 0.0

ENVELOPE_LOW_HZ = 1000
ENVELOPE_HIGH_HZ = 5000

HARMONICS = (1, 2, 3)

TARGET_HALF_WIDTH_HZ = 3.0
LOCAL_NOISE_HALF_WIDTH_HZ = 25.0

INNER_HARMONIC_WEIGHT = 0.7
INNER_SIDEBAND_WEIGHT = 0.3

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
            f"{file_path.name}: expected exactly one top-level variable, "
            f"found {public_keys}"
        )

    return data[public_keys[0]]


def as_1d_array(value) -> np.ndarray:
    return np.asarray(value).reshape(-1)


def find_y_channel(root, channel_name: str):
    matches = []

    for channel in np.atleast_1d(root.Y):
        name = str(
            getattr(
                channel,
                "Name",
                "",
            )
        )

        if name == channel_name:
            matches.append(channel)

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one channel {channel_name!r}, "
            f"found {len(matches)}."
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

    values = amplitude[
        broad_mask
        & ~target_mask
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
    return float(
        peak_near(
            frequencies,
            amplitude,
            target_frequency,
        )
        / local_noise_floor(
            frequencies,
            amplitude,
            target_frequency,
        )
    )


# ============================================================
# ENVELOPE
# ============================================================

def calculate_envelope(signal: np.ndarray) -> np.ndarray:
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
# SCORE EXTRACTION
# ============================================================

def extract_window_scores(
    signal: np.ndarray,
    rpm: float,
) -> dict:
    bearing_freq = (
        calculate_bearing_frequencies(
            rpm
        )
    )

    envelope = calculate_envelope(
        signal
    )

    (
        frequencies,
        amplitude,
    ) = calculate_fft(
        envelope
    )

    bpfo_harmonics = {}
    bpfi_harmonics = {}
    bpfi_sidebands = {}

    for harmonic in HARMONICS:
        bpfo_target = (
            harmonic
            * bearing_freq["bpfo"]
        )

        bpfi_target = (
            harmonic
            * bearing_freq["bpfi"]
        )

        bpfo_harmonics[
            f"bpfo_h{harmonic}_ratio"
        ] = normalized_peak_ratio(
            frequencies,
            amplitude,
            bpfo_target,
        )

        bpfi_harmonics[
            f"bpfi_h{harmonic}_ratio"
        ] = normalized_peak_ratio(
            frequencies,
            amplitude,
            bpfi_target,
        )

        for direction_name, direction in [
            ("minus", -1),
            ("plus", 1),
        ]:
            sideband_target = (
                bpfi_target
                + direction
                * bearing_freq["shaft"]
            )

            bpfi_sidebands[
                f"bpfi_h{harmonic}_{direction_name}_1xshaft_ratio"
            ] = normalized_peak_ratio(
                frequencies,
                amplitude,
                sideband_target,
            )

    bpfo_score = float(
        np.mean(
            [
                np.log1p(
                    bpfo_harmonics[
                        f"bpfo_h{harmonic}_ratio"
                    ]
                )
                for harmonic in HARMONICS
            ]
        )
    )

    bpfi_score = float(
        np.mean(
            [
                np.log1p(
                    bpfi_harmonics[
                        f"bpfi_h{harmonic}_ratio"
                    ]
                )
                for harmonic in HARMONICS
            ]
        )
    )

    sideband_score = float(
        np.mean(
            [
                np.log1p(value)
                for value
                in bpfi_sidebands.values()
            ]
        )
    )

    inner_score = float(
        INNER_HARMONIC_WEIGHT
        * bpfi_score
        + INNER_SIDEBAND_WEIGHT
        * sideband_score
    )

    margin = float(
        bpfo_score
        - inner_score
    )

    result = {
        "shaft_hz": bearing_freq["shaft"],
        "bpfo_hz": bearing_freq["bpfo"],
        "bpfi_hz": bearing_freq["bpfi"],
        **bpfo_harmonics,
        **bpfi_harmonics,
        **bpfi_sidebands,
        "bpfo_harmonic_score": bpfo_score,
        "bpfi_harmonic_score": bpfi_score,
        "bpfi_sideband_score": sideband_score,
        "inner_combined_score": inner_score,
        "outer_minus_inner_margin": margin,
        "predicted_condition": (
            "outer_race"
            if margin > 0.0
            else "inner_race"
        ),
    }

    return result


# ============================================================
# LOAD MEASUREMENT
# ============================================================

def load_measurement(file_path: Path):
    root = load_root_struct(
        file_path
    )

    vibration = (
        as_1d_array(
            find_y_channel(
                root,
                VIBRATION_CHANNEL_NAME,
            ).Data
        )
        .astype(float)
    )

    speed = (
        as_1d_array(
            find_y_channel(
                root,
                SPEED_CHANNEL_NAME,
            ).Data
        )
        .astype(float)
    )

    rpm = float(
        np.mean(speed)
    )

    windows_to_use = min(
        len(vibration)
        // WINDOW_SIZE,
        MAX_WINDOWS_PER_MEASUREMENT,
    )

    if windows_to_use < 1:
        raise ValueError(
            f"{file_path.name}: not enough data for one complete window."
        )

    return (
        vibration,
        rpm,
        windows_to_use,
    )


# ============================================================
# FIGURES
# ============================================================

def save_bearing_margin_plot(
    dataframe: pd.DataFrame,
    bearing_code: str,
):
    subset = (
        dataframe[
            dataframe[
                "bearing_code"
            ] == bearing_code
        ]
        .copy()
        .sort_values(
            [
                "measurement_index",
                "window_index",
            ]
        )
    )

    x = np.arange(
        len(subset)
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        x,
        subset[
            "outer_minus_inner_margin"
        ].to_numpy(),
        marker="o",
        markersize=3,
        linewidth=1,
    )

    ax.axhline(
        0.0,
        linewidth=1,
    )

    ax.set_title(
        f"{bearing_code}: frozen physics margin by 1-second window"
    )

    ax.set_xlabel(
        "Window sequence"
    )

    ax.set_ylabel(
        "Outer score - inner score"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / f"{bearing_code}_margin_by_window.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_mean_score_plot(
    summary: pd.DataFrame,
):
    plot_df = summary.copy()

    x = np.arange(
        len(plot_df)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width / 2,
        plot_df[
            "mean_bpfo_score"
        ],
        width,
        label="BPFO score",
    )

    ax.bar(
        x + width / 2,
        plot_df[
            "mean_inner_score"
        ],
        width,
        label="Inner score",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        plot_df[
            "bearing_code"
        ]
    )

    ax.set_ylabel(
        "Mean normalized score"
    )

    ax.set_title(
        "KI16 vs KI18 vs KI21: mean frozen physics scores"
    )

    ax.legend()
    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / "ki16_ki18_ki21_mean_scores.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 130)
    print(
        "POST-HOC DIAGNOSTIC: KI16 vs KI18 vs KI21"
    )
    print("=" * 130)

    print()
    print(
        "IMPORTANT: this is diagnostic analysis after the frozen held-out test."
    )

    print(
        "It must NOT be presented as an untouched validation result."
    )

    print()
    print(
        f"Envelope band: "
        f"{ENVELOPE_LOW_HZ}-{ENVELOPE_HIGH_HZ} Hz"
    )

    print(
        f"Harmonics    : "
        f"{HARMONICS}"
    )

    records = []

    for (
        bearing_code,
        condition,
    ) in BEARINGS.items():
        print()
        print("-" * 130)

        print(
            f"{bearing_code}"
            f" | true={condition}"
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

                scores = extract_window_scores(
                    window,
                    rpm,
                )

                record = {
                    "bearing_code": bearing_code,
                    "condition": condition,
                    "source_file": file_path.name,
                    "measurement_index": measurement_index,
                    "window_index": window_index,
                    "rpm": rpm,
                    **scores,
                }

                record[
                    "correct"
                ] = (
                    record[
                        "predicted_condition"
                    ]
                    == condition
                )

                records.append(
                    record
                )

    dataframe = pd.DataFrame(
        records
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        WINDOW_RESULTS_FILE,
        index=False,
    )

    summary = (
        dataframe
        .groupby(
            [
                "condition",
                "bearing_code",
            ],
            as_index=False,
        )
        .agg(
            windows=(
                "correct",
                "size",
            ),
            accuracy=(
                "correct",
                "mean",
            ),
            mean_bpfo_h1_ratio=(
                "bpfo_h1_ratio",
                "mean",
            ),
            mean_bpfo_h2_ratio=(
                "bpfo_h2_ratio",
                "mean",
            ),
            mean_bpfo_h3_ratio=(
                "bpfo_h3_ratio",
                "mean",
            ),
            mean_bpfi_h1_ratio=(
                "bpfi_h1_ratio",
                "mean",
            ),
            mean_bpfi_h2_ratio=(
                "bpfi_h2_ratio",
                "mean",
            ),
            mean_bpfi_h3_ratio=(
                "bpfi_h3_ratio",
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
            mean_inner_score=(
                "inner_combined_score",
                "mean",
            ),
            mean_margin=(
                "outer_minus_inner_margin",
                "mean",
            ),
            std_margin=(
                "outer_minus_inner_margin",
                "std",
            ),
            min_margin=(
                "outer_minus_inner_margin",
                "min",
            ),
            max_margin=(
                "outer_minus_inner_margin",
                "max",
            ),
        )
    )

    summary.to_csv(
        BEARING_SUMMARY_FILE,
        index=False,
    )

    for bearing_code in BEARINGS:
        save_bearing_margin_plot(
            dataframe,
            bearing_code,
        )

    save_mean_score_plot(
        summary
    )

    print()
    print("=" * 130)
    print("BEARING SUMMARY")
    print("=" * 130)

    print()
    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("KI16 HARMONIC DETAIL")
    print("=" * 130)

    ki16 = summary[
        summary[
            "bearing_code"
        ] == "KI16"
    ]

    print()
    print(
        ki16.to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("FILES SAVED")
    print("=" * 130)

    print()
    print(
        WINDOW_RESULTS_FILE
    )

    print(
        BEARING_SUMMARY_FILE
    )

    print(
        FIGURES_DIR
    )

    print()
    print("=" * 130)
    print("KI16 POST-HOC DIAGNOSTIC COMPLETE")
    print("=" * 130)


if __name__ == "__main__":
    main()
