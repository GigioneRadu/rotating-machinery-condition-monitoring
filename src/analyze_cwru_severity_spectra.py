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

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cwru"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "cwru_severity_spectra"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_severity_spectral_diagnostics.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLING_FREQUENCY = 12000

NUMBER_OF_ROLLING_ELEMENTS = 9
BALL_DIAMETER_INCH = 0.3126
PITCH_DIAMETER_INCH = 1.537
CONTACT_ANGLE_DEG = 0.0

SEARCH_BAND_HZ = 3.0

# Compare several envelope bands because a fixed resonance band
# can itself create domain shift when fault severity changes.
ENVELOPE_BANDS = [
    (500, 1500),
    (1000, 3000),
    (1000, 5000),
    (3000, 5000),
]

MAX_PLOT_FREQUENCY = 500

# Use 0 HP initially so load is held constant and only severity changes.
FILES = {
    "inner_race": {
        0.007: "105.mat",
        0.014: "169.mat",
        0.021: "209.mat",
    },
    "ball_fault": {
        0.007: "118.mat",
        0.014: "185.mat",
        0.021: "222.mat",
    },
    "outer_race": {
        0.007: "130.mat",
        0.014: "197.mat",
        0.021: "234.mat",
    },
}


# ============================================================
# MATLAB HELPERS
# ============================================================

def cwru_prefix_from_filename(file_path: Path) -> str:
    experiment_id = int(file_path.stem)
    return f"X{experiment_id:03d}"


def find_exact_key(
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
                f"No key for prefix {prefix!r} "
                f"ending with {suffix!r}."
            )
        return None

    if required:
        raise KeyError(
            f"Ambiguous keys for {prefix!r} and {suffix!r}: "
            f"{matches}"
        )

    return None


def load_signal_and_rpm(file_path: Path):
    data = loadmat(file_path)

    prefix = cwru_prefix_from_filename(file_path)

    de_key = find_exact_key(
        data,
        prefix,
        "_DE_time",
        required=True,
    )

    rpm_key = find_exact_key(
        data,
        prefix,
        "RPM",
        required=False,
    )

    signal = (
        data[de_key]
        .squeeze()
        .astype(float)
    )

    if rpm_key is None:
        raise KeyError(
            f"{file_path.name} has no RPM key; "
            "this diagnostic expects the selected files to contain RPM."
        )

    rpm = float(
        np.asarray(data[rpm_key])
        .reshape(-1)[0]
    )

    return signal, rpm


# ============================================================
# BEARING FREQUENCIES
# ============================================================

def calculate_bearing_frequencies(rpm: float):
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
# FFT / ENVELOPE
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


def calculate_envelope(
    signal: np.ndarray,
    low_hz: float,
    high_hz: float,
):
    nyquist = SAMPLING_FREQUENCY / 2.0

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
        hilbert(filtered)
    )

    return envelope - np.mean(envelope)


def get_peak_near(
    frequencies: np.ndarray,
    amplitude: np.ndarray,
    target_frequency: float,
):
    mask = (
        (frequencies >= target_frequency - SEARCH_BAND_HZ)
        & (frequencies <= target_frequency + SEARCH_BAND_HZ)
    )

    if not np.any(mask):
        return np.nan, np.nan

    local_f = frequencies[mask]
    local_a = amplitude[mask]

    idx = int(
        np.argmax(local_a)
    )

    return (
        float(local_f[idx]),
        float(local_a[idx]),
    )


# ============================================================
# HARMONIC / SIDEBAND DIAGNOSTICS
# ============================================================

def collect_target_rows(
    condition: str,
    severity: float,
    filename: str,
    rpm: float,
    band_low: int,
    band_high: int,
    frequencies: np.ndarray,
    amplitude: np.ndarray,
    bearing_frequencies: dict,
):
    rows = []

    primary_name = {
        "inner_race": "bpfi",
        "outer_race": "bpfo",
        "ball_fault": "bsf",
    }[condition]

    primary_frequency = bearing_frequencies[
        primary_name
    ]

    shaft = bearing_frequencies[
        "shaft"
    ]

    targets = [
        ("fundamental", primary_frequency),
        ("2x", 2.0 * primary_frequency),
        ("3x", 3.0 * primary_frequency),
        ("minus_1x_shaft", primary_frequency - shaft),
        ("plus_1x_shaft", primary_frequency + shaft),
    ]

    for target_name, target_frequency in targets:
        peak_frequency, peak_amplitude = get_peak_near(
            frequencies,
            amplitude,
            target_frequency,
        )

        rows.append(
            {
                "condition": condition,
                "severity_inch": severity,
                "source_file": filename,
                "rpm": rpm,
                "envelope_band_low_hz": band_low,
                "envelope_band_high_hz": band_high,
                "fault_frequency_name": primary_name,
                "target_type": target_name,
                "target_frequency_hz": target_frequency,
                "measured_peak_frequency_hz": peak_frequency,
                "measured_peak_amplitude": peak_amplitude,
            }
        )

    return rows


# ============================================================
# PLOT ONE CONDITION / BAND
# ============================================================

def save_comparison_plot(
    condition: str,
    band_low: int,
    band_high: int,
    spectra: dict,
):
    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for severity, payload in spectra.items():
        frequencies = payload[
            "frequencies"
        ]

        amplitude = payload[
            "amplitude"
        ]

        mask = (
            frequencies
            <= MAX_PLOT_FREQUENCY
        )

        ax.plot(
            frequencies[mask],
            amplitude[mask],
            label=f"{severity:.3f} in",
        )

    # Use the 0.014 frequencies only for marker placement.
    # RPM differences across these 0 HP files are tiny.
    reference = spectra[0.014][
        "bearing_frequencies"
    ]

    primary_name = {
        "inner_race": "bpfi",
        "outer_race": "bpfo",
        "ball_fault": "bsf",
    }[condition]

    primary = reference[
        primary_name
    ]

    shaft = reference[
        "shaft"
    ]

    marker_targets = {
        primary_name.upper(): primary,
        f"2x {primary_name.upper()}": 2.0 * primary,
        f"3x {primary_name.upper()}": 3.0 * primary,
        "-1x shaft": primary - shaft,
        "+1x shaft": primary + shaft,
    }

    for label, frequency in marker_targets.items():
        if (
            frequency > 0
            and frequency <= MAX_PLOT_FREQUENCY
        ):
            ax.axvline(
                frequency,
                linestyle="--",
                linewidth=1,
                label=label,
            )

    ax.set_title(
        f"{condition} - Envelope Spectrum by Severity\n"
        f"Band {band_low}-{band_high} Hz"
    )

    ax.set_xlabel(
        "Frequency [Hz]"
    )

    ax.set_ylabel(
        "Envelope amplitude"
    )

    ax.set_xlim(
        0,
        MAX_PLOT_FREQUENCY,
    )

    ax.legend(
        fontsize=8
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    output_file = (
        FIGURES_DIR
        / condition
        / f"envelope_{band_low}_{band_high}.png"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_file,
        dpi=170,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 115)
    print("CWRU SEVERITY SPECTRAL DIAGNOSTICS")
    print("=" * 115)

    print()
    print(
        "Purpose: determine whether the 0.014-inch domain shift "
        "is related to the selected envelope resonance band, "
        "harmonics, or shaft-frequency sidebands."
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for condition, severity_files in FILES.items():
        print()
        print("=" * 115)
        print(
            f"CONDITION: {condition}"
        )
        print("=" * 115)

        loaded = {}

        for severity, filename in severity_files.items():
            file_path = (
                RAW_DATA_DIR
                / filename
            )

            if not file_path.exists():
                raise FileNotFoundError(
                    f"Missing file: {file_path}"
                )

            signal, rpm = (
                load_signal_and_rpm(
                    file_path
                )
            )

            loaded[
                severity
            ] = {
                "filename": filename,
                "signal": signal,
                "rpm": rpm,
                "bearing_frequencies": (
                    calculate_bearing_frequencies(
                        rpm
                    )
                ),
            }

            print(
                f"{severity:.3f} in"
                f" | file={filename}"
                f" | rpm={rpm:.0f}"
                f" | samples={len(signal)}"
            )

        for (
            band_low,
            band_high,
        ) in ENVELOPE_BANDS:
            print()
            print(
                f"Envelope band: "
                f"{band_low}-{band_high} Hz"
            )

            spectra = {}

            for severity, payload in loaded.items():
                envelope = calculate_envelope(
                    payload["signal"],
                    band_low,
                    band_high,
                )

                frequencies, amplitude = (
                    calculate_fft(
                        envelope
                    )
                )

                spectra[
                    severity
                ] = {
                    "frequencies": frequencies,
                    "amplitude": amplitude,
                    "bearing_frequencies": payload[
                        "bearing_frequencies"
                    ],
                }

                diagnostic_rows = (
                    collect_target_rows(
                        condition=condition,
                        severity=severity,
                        filename=payload[
                            "filename"
                        ],
                        rpm=payload[
                            "rpm"
                        ],
                        band_low=band_low,
                        band_high=band_high,
                        frequencies=frequencies,
                        amplitude=amplitude,
                        bearing_frequencies=payload[
                            "bearing_frequencies"
                        ],
                    )
                )

                rows.extend(
                    diagnostic_rows
                )

                primary_name = {
                    "inner_race": "bpfi",
                    "outer_race": "bpfo",
                    "ball_fault": "bsf",
                }[condition]

                fundamental_row = next(
                    row
                    for row in diagnostic_rows
                    if row["target_type"]
                    == "fundamental"
                )

                print(
                    f"  {severity:.3f} in"
                    f" | {primary_name.upper()} target="
                    f"{fundamental_row['target_frequency_hz']:.2f} Hz"
                    f" | measured="
                    f"{fundamental_row['measured_peak_frequency_hz']:.2f} Hz"
                    f" | amp="
                    f"{fundamental_row['measured_peak_amplitude']:.6f}"
                )

            save_comparison_plot(
                condition=condition,
                band_low=band_low,
                band_high=band_high,
                spectra=spectra,
            )

    dataframe = pd.DataFrame(
        rows
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print("=" * 115)
    print("0.014-INCH PRIMARY-FREQUENCY BAND COMPARISON")
    print("=" * 115)

    print()

    primary_rows = dataframe[
        dataframe[
            "target_type"
        ]
        == "fundamental"
    ].copy()

    comparison = (
        primary_rows
        .pivot_table(
            index=[
                "condition",
                "envelope_band_low_hz",
                "envelope_band_high_hz",
            ],
            columns="severity_inch",
            values="measured_peak_amplitude",
            aggfunc="mean",
        )
        .reset_index()
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print("=" * 115)
    print("FILES SAVED")
    print("=" * 115)

    print()
    print(
        "Spectral diagnostic CSV:"
    )

    print(
        OUTPUT_CSV
    )

    print()
    print(
        "Comparison figures:"
    )

    print(
        FIGURES_DIR
    )

    print()
    print("=" * 115)
    print("SPECTRAL DIAGNOSTICS COMPLETE")
    print("=" * 115)


if __name__ == "__main__":
    main()
