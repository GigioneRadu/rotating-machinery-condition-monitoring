from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PADERBORN_DIR = PROJECT_ROOT / "data" / "raw" / "paderborn"

RESULTS_FILE = PROJECT_ROOT / "data" / "processed" / "paderborn_frozen_heldout_real_damage_predictions.csv"
BEARING_SUMMARY_FILE = PROJECT_ROOT / "data" / "processed" / "paderborn_frozen_heldout_real_damage_by_bearing.csv"
OVERALL_SUMMARY_FILE = PROJECT_ROOT / "data" / "processed" / "paderborn_frozen_heldout_real_damage_summary.csv"
FIGURE_FILE = PROJECT_ROOT / "figures" / "paderborn_frozen_heldout_real_damage_confusion.png"

OPERATING_CONDITION = "N15_M07_F10"

BEARINGS = {
    "KA15": "outer_race",
    "KA22": "outer_race",
    "KA30": "outer_race",
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

NUMBER_OF_ROLLING_ELEMENTS = 8
ROLLING_ELEMENT_DIAMETER_MM = 6.75
PITCH_DIAMETER_MM = 28.55
CONTACT_ANGLE_DEG = 0.0

# FROZEN METHOD
ENVELOPE_LOW_HZ = 1000
ENVELOPE_HIGH_HZ = 5000
HARMONICS = (1, 2, 3)
TARGET_HALF_WIDTH_HZ = 3.0
LOCAL_NOISE_HALF_WIDTH_HZ = 25.0
INNER_HARMONIC_WEIGHT = 0.7
INNER_SIDEBAND_WEIGHT = 0.3
EPS = 1e-12

CLASS_ORDER = ["inner_race", "outer_race"]


def load_root_struct(file_path: Path):
    data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    public_keys = [key for key in data if not key.startswith("__")]
    if len(public_keys) != 1:
        raise ValueError(f"{file_path.name}: unexpected top-level variables {public_keys}")
    return data[public_keys[0]]


def as_1d_array(value):
    return np.asarray(value).reshape(-1)


def find_y_channel(root, channel_name):
    matches = [
        channel
        for channel in np.atleast_1d(root.Y)
        if str(getattr(channel, "Name", "")) == channel_name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one channel {channel_name!r}, found {len(matches)}")
    return matches[0]


def bearing_frequencies(rpm):
    shaft = rpm / 60.0
    theta = np.deg2rad(CONTACT_ANGLE_DEG)
    ratio = (ROLLING_ELEMENT_DIAMETER_MM / PITCH_DIAMETER_MM) * np.cos(theta)
    bpfo = NUMBER_OF_ROLLING_ELEMENTS / 2.0 * shaft * (1.0 - ratio)
    bpfi = NUMBER_OF_ROLLING_ELEMENTS / 2.0 * shaft * (1.0 + ratio)
    return {"shaft": shaft, "bpfo": bpfo, "bpfi": bpfi}


def calculate_fft(signal):
    centered = signal - np.mean(signal)
    n = len(centered)
    window = np.hanning(n)
    spectrum = np.fft.rfft(centered * window)
    frequencies = np.fft.rfftfreq(n, d=1.0 / SAMPLING_FREQUENCY)
    amplitude = 2.0 * np.abs(spectrum) / np.sum(window)
    return frequencies, amplitude


def peak_near(frequencies, amplitude, target):
    mask = (frequencies >= target - TARGET_HALF_WIDTH_HZ) & (frequencies <= target + TARGET_HALF_WIDTH_HZ)
    if not np.any(mask):
        return 0.0
    return float(np.max(amplitude[mask]))


def local_noise_floor(frequencies, amplitude, target):
    broad = (frequencies >= target - LOCAL_NOISE_HALF_WIDTH_HZ) & (frequencies <= target + LOCAL_NOISE_HALF_WIDTH_HZ)
    target_mask = (frequencies >= target - TARGET_HALF_WIDTH_HZ) & (frequencies <= target + TARGET_HALF_WIDTH_HZ)
    values = amplitude[broad & ~target_mask]
    if len(values) == 0:
        return EPS
    return float(np.median(values) + EPS)


def normalized_peak_ratio(frequencies, amplitude, target):
    return peak_near(frequencies, amplitude, target) / local_noise_floor(frequencies, amplitude, target)


def calculate_envelope(signal):
    nyquist = SAMPLING_FREQUENCY / 2.0
    b, a = butter(
        4,
        [ENVELOPE_LOW_HZ / nyquist, ENVELOPE_HIGH_HZ / nyquist],
        btype="bandpass",
    )
    filtered = filtfilt(b, a, signal)
    envelope = np.abs(hilbert(filtered))
    return envelope - np.mean(envelope)


def harmonic_score(frequencies, amplitude, base_frequency):
    values = []
    for harmonic in HARMONICS:
        ratio = normalized_peak_ratio(
            frequencies,
            amplitude,
            harmonic * base_frequency,
        )
        values.append(np.log1p(ratio))
    return float(np.mean(values))


def bpfi_sideband_score(frequencies, amplitude, bpfi, shaft):
    values = []
    for harmonic in HARMONICS:
        center = harmonic * bpfi
        for direction in (-1, 1):
            ratio = normalized_peak_ratio(
                frequencies,
                amplitude,
                center + direction * shaft,
            )
            values.append(np.log1p(ratio))
    return float(np.mean(values))


def evaluate_window(signal, rpm):
    freqs_theoretical = bearing_frequencies(rpm)
    envelope = calculate_envelope(signal)
    frequencies, amplitude = calculate_fft(envelope)

    bpfo_score = harmonic_score(frequencies, amplitude, freqs_theoretical["bpfo"])
    bpfi_score = harmonic_score(frequencies, amplitude, freqs_theoretical["bpfi"])
    sideband_score = bpfi_sideband_score(
        frequencies,
        amplitude,
        freqs_theoretical["bpfi"],
        freqs_theoretical["shaft"],
    )

    inner_score = (
        INNER_HARMONIC_WEIGHT * bpfi_score
        + INNER_SIDEBAND_WEIGHT * sideband_score
    )
    outer_score = bpfo_score
    margin = outer_score - inner_score
    predicted = "outer_race" if margin > 0 else "inner_race"

    return {
        "bpfo_harmonic_score": outer_score,
        "bpfi_harmonic_score": bpfi_score,
        "bpfi_sideband_score": sideband_score,
        "inner_combined_score": inner_score,
        "outer_minus_inner_margin": margin,
        "predicted_condition": predicted,
    }


def load_measurement(file_path):
    root = load_root_struct(file_path)
    vibration = as_1d_array(find_y_channel(root, VIBRATION_CHANNEL_NAME).Data).astype(float)
    speed = as_1d_array(find_y_channel(root, SPEED_CHANNEL_NAME).Data).astype(float)
    rpm = float(np.mean(speed))
    windows_to_use = min(len(vibration) // WINDOW_SIZE, MAX_WINDOWS_PER_MEASUREMENT)
    if windows_to_use < 1:
        raise ValueError(f"{file_path.name}: not enough samples for one 1-second window")
    return vibration, rpm, windows_to_use


def main():
    print()
    print("=" * 130)
    print("PADERBORN FROZEN HELD-OUT REAL-DAMAGE VALIDATION")
    print("=" * 130)
    print()
    print("No model is trained here. The decision rule is frozen.")
    print(f"Envelope band : {ENVELOPE_LOW_HZ}-{ENVELOPE_HIGH_HZ} Hz")
    print(f"Harmonics     : {HARMONICS}")
    print("Inner score   : 0.7 * BPFI harmonics + 0.3 * BPFI sidebands")
    print("Outer score   : BPFO harmonics")
    print()

    records = []

    for bearing_code, true_condition in BEARINGS.items():
        print("-" * 130)
        print(f"{bearing_code} | true={true_condition}")
        print("-" * 130)

        bearing_dir = PADERBORN_DIR / bearing_code
        if not bearing_dir.exists():
            raise FileNotFoundError(f"Missing folder:\n{bearing_dir}")

        bearing_records = []

        for measurement_index in range(1, MEASUREMENTS_PER_BEARING + 1):
            file_path = bearing_dir / (
                f"{OPERATING_CONDITION}_{bearing_code}_{measurement_index}.mat"
            )
            if not file_path.exists():
                raise FileNotFoundError(f"Missing file:\n{file_path}")

            vibration, rpm, windows_to_use = load_measurement(file_path)

            for window_index in range(windows_to_use):
                start = window_index * WINDOW_SIZE
                end = start + WINDOW_SIZE
                window = vibration[start:end]

                scores = evaluate_window(window, rpm)

                record = {
                    "bearing_code": bearing_code,
                    "condition": true_condition,
                    "source_file": file_path.name,
                    "measurement_index": measurement_index,
                    "window_index": window_index,
                    "rpm": rpm,
                    **scores,
                }
                record["correct"] = (
                    record["predicted_condition"] == true_condition
                )
                records.append(record)
                bearing_records.append(record)

        bearing_df = pd.DataFrame(bearing_records)
        print(f"Windows evaluated : {len(bearing_df)}")
        print(f"Accuracy          : {bearing_df['correct'].mean():.4f}")
        print(f"Mean margin       : {bearing_df['outer_minus_inner_margin'].mean():.4f}")
        print()

    results_df = pd.DataFrame(records)

    y_true = results_df["condition"]
    y_pred = results_df["predicted_condition"]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        average="macro",
        zero_division=0,
    )

    recalls = recall_score(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        average=None,
        zero_division=0,
    )
    recall_map = dict(zip(CLASS_ORDER, recalls))

    bearing_summary = (
        results_df
        .groupby(["condition", "bearing_code"], as_index=False)
        .agg(
            windows=("correct", "size"),
            accuracy=("correct", "mean"),
            mean_bpfo_score=("bpfo_harmonic_score", "mean"),
            mean_bpfi_score=("bpfi_harmonic_score", "mean"),
            mean_bpfi_sideband_score=("bpfi_sideband_score", "mean"),
            mean_inner_score=("inner_combined_score", "mean"),
            mean_margin=("outer_minus_inner_margin", "mean"),
        )
    )

    overall_summary = pd.DataFrame([{
        "heldout_bearings": len(BEARINGS),
        "evaluated_windows": len(results_df),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "recall_inner_race": recall_map["inner_race"],
        "recall_outer_race": recall_map["outer_race"],
        "envelope_low_hz": ENVELOPE_LOW_HZ,
        "envelope_high_hz": ENVELOPE_HIGH_HZ,
        "inner_harmonic_weight": INNER_HARMONIC_WEIGHT,
        "inner_sideband_weight": INNER_SIDEBAND_WEIGHT,
    }])

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(RESULTS_FILE, index=False)
    bearing_summary.to_csv(BEARING_SUMMARY_FILE, index=False)
    overall_summary.to_csv(OVERALL_SUMMARY_FILE, index=False)

    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_ORDER,
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    display.plot(ax=ax, values_format="d")
    ax.set_title("Frozen physics rule on unseen real-damage bearings")
    fig.tight_layout()
    fig.savefig(FIGURE_FILE, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("=" * 130)
    print("FINAL HELD-OUT RESULTS")
    print("=" * 130)
    print()
    print(f"Accuracy          : {accuracy:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"Recall inner_race : {recall_map['inner_race']:.4f}")
    print(f"Recall outer_race : {recall_map['outer_race']:.4f}")
    print()
    print("Classification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=CLASS_ORDER,
            digits=4,
            zero_division=0,
        )
    )

    print()
    print("=" * 130)
    print("PER-BEARING SUMMARY")
    print("=" * 130)
    print()
    print(bearing_summary.to_string(index=False))

    print()
    print("=" * 130)
    print("FILES SAVED")
    print("=" * 130)
    print()
    print(RESULTS_FILE)
    print(BEARING_SUMMARY_FILE)
    print(OVERALL_SUMMARY_FILE)
    print(FIGURE_FILE)
    print()
    print("=" * 130)
    print("FROZEN HELD-OUT REAL-DAMAGE VALIDATION COMPLETE")
    print("=" * 130)


if __name__ == "__main__":
    main()
