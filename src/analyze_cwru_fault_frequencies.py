from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert


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

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "cwru"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True,
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
# CWRU DRIVE-END BEARING GEOMETRY
# ============================================================

NUMBER_OF_ROLLING_ELEMENTS = 9

BALL_DIAMETER_INCH = 0.3126

PITCH_DIAMETER_INCH = 1.537

CONTACT_ANGLE_DEG = 0.0


# ============================================================
# ENVELOPE SETTINGS
# ============================================================

ENVELOPE_LOW_HZ = 1000
ENVELOPE_HIGH_HZ = 5000


# ============================================================
# FREQUENCY SEARCH SETTINGS
# ============================================================

SEARCH_BAND_HZ = 3.0


# ============================================================
# HELPERS
# ============================================================

def find_key(
    data: dict,
    suffix: str,
):
    matches = [
        key
        for key in data.keys()
        if key.endswith(suffix)
    ]

    if not matches:
        raise KeyError(
            f"No key ending with {suffix!r} found."
        )

    if len(matches) > 1:
        raise KeyError(
            f"Multiple keys ending with {suffix!r}: {matches}"
        )

    return matches[0]


# ============================================================
# LOAD CWRU DRIVE END SIGNAL
# ============================================================

def load_cwru_de_signal(
    file_path: Path,
):
    data = loadmat(
        file_path
    )

    de_key = find_key(
        data,
        "_DE_time",
    )

    rpm_key = find_key(
        data,
        "RPM",
    )

    signal = (
        data[de_key]
        .squeeze()
        .astype(float)
    )

    rpm = float(
        np.asarray(
            data[rpm_key]
        ).reshape(-1)[0]
    )

    return signal, rpm


# ============================================================
# BEARING FREQUENCIES
# ============================================================

def calculate_bearing_frequencies(
    rpm: float,
):
    shaft_frequency = (
        rpm / 60.0
    )

    theta = np.deg2rad(
        CONTACT_ANGLE_DEG
    )

    n = NUMBER_OF_ROLLING_ELEMENTS

    d = BALL_DIAMETER_INCH

    D = PITCH_DIAMETER_INCH


    bpfo = (
        n
        / 2.0
        * shaft_frequency
        * (
            1.0
            - (
                d / D
            )
            * np.cos(theta)
        )
    )


    bpfi = (
        n
        / 2.0
        * shaft_frequency
        * (
            1.0
            + (
                d / D
            )
            * np.cos(theta)
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
            - (
                (
                    d / D
                )
                * np.cos(theta)
            ) ** 2
        )
    )


    ftf = (
        shaft_frequency
        / 2.0
        * (
            1.0
            - (
                d / D
            )
            * np.cos(theta)
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
    sampling_frequency: int,
):
    signal = (
        signal
        - np.mean(signal)
    )

    n = len(
        signal
    )

    window = np.hanning(
        n
    )

    windowed_signal = (
        signal
        * window
    )

    spectrum = np.fft.rfft(
        windowed_signal
    )

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / sampling_frequency,
    )

    amplitude = (
        2.0
        * np.abs(spectrum)
        / np.sum(window)
    )

    return frequencies, amplitude


# ============================================================
# ENVELOPE SIGNAL
# ============================================================

def calculate_envelope(
    signal: np.ndarray,
    sampling_frequency: int,
):
    nyquist = (
        sampling_frequency
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

    envelope = (
        envelope
        - np.mean(envelope)
    )

    return envelope


# ============================================================
# AMPLITUDE NEAR TARGET FREQUENCY
# ============================================================

def get_peak_near_frequency(
    frequencies: np.ndarray,
    amplitude: np.ndarray,
    target_frequency: float,
    search_band_hz: float = SEARCH_BAND_HZ,
):
    mask = (
        frequencies
        >= target_frequency - search_band_hz
    ) & (
        frequencies
        <= target_frequency + search_band_hz
    )

    if not np.any(mask):
        return None, None

    local_frequencies = (
        frequencies[
            mask
        ]
    )

    local_amplitudes = (
        amplitude[
            mask
        ]
    )

    max_index = int(
        np.argmax(
            local_amplitudes
        )
    )

    peak_frequency = float(
        local_frequencies[
            max_index
        ]
    )

    peak_amplitude = float(
        local_amplitudes[
            max_index
        ]
    )

    return (
        peak_frequency,
        peak_amplitude,
    )


# ============================================================
# PRINT TARGET ANALYSIS
# ============================================================

def print_target_analysis(
    label: str,
    target_frequency: float,
    raw_frequencies: np.ndarray,
    raw_amplitude: np.ndarray,
    envelope_frequencies: np.ndarray,
    envelope_amplitude: np.ndarray,
):
    raw_peak_frequency, raw_peak_amplitude = (
        get_peak_near_frequency(
            raw_frequencies,
            raw_amplitude,
            target_frequency,
        )
    )

    envelope_peak_frequency, envelope_peak_amplitude = (
        get_peak_near_frequency(
            envelope_frequencies,
            envelope_amplitude,
            target_frequency,
        )
    )


    print(
        f"{label:<6}"
        f" theoretical={target_frequency:8.3f} Hz"
        f" | raw={raw_peak_frequency:8.3f} Hz"
        f" amp={raw_peak_amplitude:10.6f}"
        f" | envelope={envelope_peak_frequency:8.3f} Hz"
        f" amp={envelope_peak_amplitude:10.6f}"
    )


# ============================================================
# PLOT RAW FFT
# ============================================================

def save_raw_fft_plot(
    condition: str,
    frequencies: np.ndarray,
    amplitude: np.ndarray,
    bearing_frequencies: dict,
):
    plt.figure(
        figsize=(
            12,
            6,
        )
    )

    max_plot_frequency = 500

    mask = (
        frequencies
        <= max_plot_frequency
    )

    plt.plot(
        frequencies[
            mask
        ],
        amplitude[
            mask
        ],
    )


    for label in [
        "shaft",
        "bpfo",
        "bpfi",
        "bsf",
        "ftf",
    ]:

        plt.axvline(
            bearing_frequencies[
                label
            ],
            linestyle="--",
            linewidth=1,
            label=(
                f"{label.upper()} "
                f"{bearing_frequencies[label]:.1f} Hz"
            ),
        )


    plt.title(
        f"CWRU Real Data - Raw FFT - {condition}"
    )

    plt.xlabel(
        "Frequency [Hz]"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.xlim(
        0,
        max_plot_frequency,
    )

    plt.legend()

    plt.tight_layout()


    output_file = (
        FIGURES_DIR
        / f"{condition}_raw_fft.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
    )

    plt.close()


# ============================================================
# PLOT ENVELOPE FFT
# ============================================================

def save_envelope_fft_plot(
    condition: str,
    frequencies: np.ndarray,
    amplitude: np.ndarray,
    bearing_frequencies: dict,
):
    plt.figure(
        figsize=(
            12,
            6,
        )
    )

    max_plot_frequency = 500

    mask = (
        frequencies
        <= max_plot_frequency
    )

    plt.plot(
        frequencies[
            mask
        ],
        amplitude[
            mask
        ],
    )


    for label in [
        "shaft",
        "bpfo",
        "bpfi",
        "bsf",
        "ftf",
    ]:

        plt.axvline(
            bearing_frequencies[
                label
            ],
            linestyle="--",
            linewidth=1,
            label=(
                f"{label.upper()} "
                f"{bearing_frequencies[label]:.1f} Hz"
            ),
        )


    plt.title(
        f"CWRU Real Data - Envelope Spectrum - {condition}"
    )

    plt.xlabel(
        "Frequency [Hz]"
    )

    plt.ylabel(
        "Envelope amplitude"
    )

    plt.xlim(
        0,
        max_plot_frequency,
    )

    plt.legend()

    plt.tight_layout()


    output_file = (
        FIGURES_DIR
        / f"{condition}_envelope_fft.png"
    )

    plt.savefig(
        output_file,
        dpi=150,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 110)
    print("CWRU REAL DATA - FAULT FREQUENCY ANALYSIS")
    print("=" * 110)

    print()

    print(
        f"Sampling frequency : "
        f"{SAMPLING_FREQUENCY} Hz"
    )

    print(
        f"Envelope band      : "
        f"{ENVELOPE_LOW_HZ} - "
        f"{ENVELOPE_HIGH_HZ} Hz"
    )

    print(
        f"Search band        : "
        f"±{SEARCH_BAND_HZ:.1f} Hz"
    )


    # ========================================================
    # PROCESS EACH CONDITION
    # ========================================================

    for condition, filename in (
        CWRU_FILES.items()
    ):

        print()
        print("-" * 110)

        print(
            f"Condition: {condition}"
        )

        print(
            f"File     : {filename}"
        )


        file_path = (
            DATA_DIR
            / filename
        )


        signal, rpm = (
            load_cwru_de_signal(
                file_path
            )
        )


        bearing_frequencies = (
            calculate_bearing_frequencies(
                rpm
            )
        )


        print()

        print(
            f"RPM      : "
            f"{rpm:.0f}"
        )

        print(
            f"Shaft    : "
            f"{bearing_frequencies['shaft']:.3f} Hz"
        )

        print(
            f"BPFO     : "
            f"{bearing_frequencies['bpfo']:.3f} Hz"
        )

        print(
            f"BPFI     : "
            f"{bearing_frequencies['bpfi']:.3f} Hz"
        )

        print(
            f"BSF      : "
            f"{bearing_frequencies['bsf']:.3f} Hz"
        )

        print(
            f"FTF      : "
            f"{bearing_frequencies['ftf']:.3f} Hz"
        )


        # ====================================================
        # RAW FFT
        # ====================================================

        raw_frequencies, raw_amplitude = (
            calculate_fft(
                signal,
                SAMPLING_FREQUENCY,
            )
        )


        # ====================================================
        # ENVELOPE
        # ====================================================

        envelope_signal = (
            calculate_envelope(
                signal,
                SAMPLING_FREQUENCY,
            )
        )

        envelope_frequencies, envelope_amplitude = (
            calculate_fft(
                envelope_signal,
                SAMPLING_FREQUENCY,
            )
        )


        print()
        print(
            "Target frequency comparison:"
        )

        print_target_analysis(
            "SHAFT",
            bearing_frequencies[
                "shaft"
            ],
            raw_frequencies,
            raw_amplitude,
            envelope_frequencies,
            envelope_amplitude,
        )

        print_target_analysis(
            "BPFO",
            bearing_frequencies[
                "bpfo"
            ],
            raw_frequencies,
            raw_amplitude,
            envelope_frequencies,
            envelope_amplitude,
        )

        print_target_analysis(
            "BPFI",
            bearing_frequencies[
                "bpfi"
            ],
            raw_frequencies,
            raw_amplitude,
            envelope_frequencies,
            envelope_amplitude,
        )

        print_target_analysis(
            "BSF",
            bearing_frequencies[
                "bsf"
            ],
            raw_frequencies,
            raw_amplitude,
            envelope_frequencies,
            envelope_amplitude,
        )

        print_target_analysis(
            "FTF",
            bearing_frequencies[
                "ftf"
            ],
            raw_frequencies,
            raw_amplitude,
            envelope_frequencies,
            envelope_amplitude,
        )


        # ====================================================
        # SAVE PLOTS
        # ====================================================

        save_raw_fft_plot(
            condition,
            raw_frequencies,
            raw_amplitude,
            bearing_frequencies,
        )

        save_envelope_fft_plot(
            condition,
            envelope_frequencies,
            envelope_amplitude,
            bearing_frequencies,
        )


    print()
    print("=" * 110)

    print(
        "Analysis complete."
    )

    print(
        f"Figures saved to:"
    )

    print(
        FIGURES_DIR
    )

    print("=" * 110)


if __name__ == "__main__":
    main()