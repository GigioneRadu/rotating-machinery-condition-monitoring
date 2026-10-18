from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    recall_score,
)


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

TRAIN_DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_multi_bearing_n15_m07_f10_dataset.csv"
)

TEST_DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_stage1_heldout_test_dataset.csv"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_stage1_healthy_vs_damaged_results.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_stage1_healthy_vs_damaged_predictions.csv"
)

PER_BEARING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_stage1_healthy_vs_damaged_by_bearing.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "paderborn_stage1_healthy_vs_damaged"
)


# ============================================================
# TRAIN / HELD-OUT TEST SPLIT
# ============================================================

OPERATING_CONDITION = "N15_M07_F10"

TRAIN_HEALTHY_BEARINGS = {
    "K001",
    "K002",
    "K003",
}

TRAIN_DAMAGED_BEARINGS = {
    "KA01",
    "KA05",
    "KA07",
    "KI01",
    "KI05",
    "KI07",
}

TEST_BEARINGS = {
    "K004": "healthy",
    "K005": "healthy",
    "KA15": "damaged",
    "KA22": "damaged",
    "KA30": "damaged",
    "KI16": "damaged",
    "KI18": "damaged",
    "KI21": "damaged",
}

MEASUREMENTS_PER_BEARING = 20


# ============================================================
# SIGNAL SETTINGS
# ============================================================

SAMPLING_FREQUENCY = 64000
WINDOW_DURATION_SECONDS = 1.0
WINDOW_SIZE = int(
    SAMPLING_FREQUENCY
    * WINDOW_DURATION_SECONDS
)
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
# SPECTRAL SETTINGS
# ============================================================

SEARCH_BAND_HZ = 3.0
ENVELOPE_LOW_HZ = 5000
ENVELOPE_HIGH_HZ = 20000


# ============================================================
# MODEL SETTINGS
# ============================================================

RANDOM_STATE = 42
N_ESTIMATORS = 500

CLASS_ORDER = [
    "healthy",
    "damaged",
]


# ============================================================
# FEATURE GROUPS
# ============================================================

TIME_DOMAIN_FEATURES = [
    "rms",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
]

RAW_FREQUENCY_FEATURES = [
    "shaft_amplitude",
    "bpfo_raw",
    "bpfi_raw",
    "bsf_raw",
    "ftf_raw",
]

ENVELOPE_FEATURES = [
    "bpfo_envelope",
    "bpfi_envelope",
    "bsf_envelope",
    "ftf_envelope",
]

PHYSICS_FREQUENCY_FEATURES = (
    RAW_FREQUENCY_FEATURES
    + ENVELOPE_FEATURES
)

ALL_FEATURES = (
    TIME_DOMAIN_FEATURES
    + PHYSICS_FREQUENCY_FEATURES
)

FEATURE_GROUPS = {
    "time_domain_only": TIME_DOMAIN_FEATURES,
    "raw_frequency_only": RAW_FREQUENCY_FEATURES,
    "envelope_only": ENVELOPE_FEATURES,
    "physics_frequency_only": PHYSICS_FREQUENCY_FEATURES,
    "time_plus_envelope": (
        TIME_DOMAIN_FEATURES
        + ENVELOPE_FEATURES
    ),
    "all_features": ALL_FEATURES,
}


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
    return np.asarray(
        value
    ).reshape(-1)


def find_y_channel(
    root,
    channel_name: str,
):
    matches = []

    for channel in np.atleast_1d(
        root.Y
    ):
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
            f"Expected exactly one channel {channel_name!r}, "
            f"found {len(matches)}."
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
        * (1.0 - ratio)
    )

    bpfi = (
        NUMBER_OF_ROLLING_ELEMENTS
        / 2.0
        * shaft
        * (1.0 + ratio)
    )

    bsf = (
        PITCH_DIAMETER_MM
        / (
            2.0
            * ROLLING_ELEMENT_DIAMETER_MM
        )
        * shaft
        * (1.0 - ratio ** 2)
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
# FFT HELPERS
# ============================================================

def calculate_fft(
    signal: np.ndarray,
):
    centered = (
        signal
        - np.mean(
            signal
        )
    )

    n = len(
        centered
    )

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

    analytic = hilbert(
        filtered
    )

    envelope = np.abs(
        analytic
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

    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "crest_factor": crest_factor,
        "kurtosis": kurtosis_value,
        "shaft_amplitude": get_peak_amplitude_near(
            raw_frequencies,
            raw_amplitude,
            bearing["shaft"],
        ),
        "bpfo_raw": get_peak_amplitude_near(
            raw_frequencies,
            raw_amplitude,
            bearing["bpfo"],
        ),
        "bpfi_raw": get_peak_amplitude_near(
            raw_frequencies,
            raw_amplitude,
            bearing["bpfi"],
        ),
        "bsf_raw": get_peak_amplitude_near(
            raw_frequencies,
            raw_amplitude,
            bearing["bsf"],
        ),
        "ftf_raw": get_peak_amplitude_near(
            raw_frequencies,
            raw_amplitude,
            bearing["ftf"],
        ),
        "bpfo_envelope": get_peak_amplitude_near(
            envelope_frequencies,
            envelope_amplitude,
            bearing["bpfo"],
        ),
        "bpfi_envelope": get_peak_amplitude_near(
            envelope_frequencies,
            envelope_amplitude,
            bearing["bpfi"],
        ),
        "bsf_envelope": get_peak_amplitude_near(
            envelope_frequencies,
            envelope_amplitude,
            bearing["bsf"],
        ),
        "ftf_envelope": get_peak_amplitude_near(
            envelope_frequencies,
            envelope_amplitude,
            bearing["ftf"],
        ),
    }


# ============================================================
# TEST DATASET BUILD
# ============================================================

def load_measurement(
    file_path: Path,
):
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
        np.mean(
            speed
        )
    )

    windows_to_use = min(
        len(vibration)
        // WINDOW_SIZE,
        MAX_WINDOWS_PER_MEASUREMENT,
    )

    if windows_to_use < 1:
        raise ValueError(
            f"{file_path.name}: not enough data for one full 1-second window."
        )

    return (
        vibration,
        rpm,
        windows_to_use,
    )


def build_heldout_test_dataset():
    print()
    print("=" * 130)
    print(
        "BUILD HELD-OUT STAGE-1 TEST DATASET"
    )
    print("=" * 130)

    records = []

    for (
        bearing_code,
        binary_condition,
    ) in TEST_BEARINGS.items():
        print()
        print("-" * 130)

        print(
            f"{bearing_code}"
            f" | binary_condition={binary_condition}"
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

        bearing_windows = 0

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

                features = extract_features(
                    window,
                    rpm,
                )

                records.append(
                    {
                        "bearing_code": bearing_code,
                        "binary_condition": binary_condition,
                        "source_file": file_path.name,
                        "measurement_index": measurement_index,
                        "window_index": window_index,
                        "rpm": rpm,
                        **features,
                    }
                )

                bearing_windows += 1

        print(
            f"Windows collected: "
            f"{bearing_windows}"
        )

    test_df = pd.DataFrame(
        records
    )

    TEST_DATASET_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_df.to_csv(
        TEST_DATASET_FILE,
        index=False,
    )

    print()
    print(
        f"Held-out test dataset shape: "
        f"{test_df.shape}"
    )

    print()
    print(
        "Samples by binary condition:"
    )

    print(
        test_df
        .groupby(
            "binary_condition"
        )
        .size()
        .to_string()
    )

    return test_df


# ============================================================
# TRAIN DATASET
# ============================================================

def load_training_dataset():
    if not TRAIN_DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n"
            f"{TRAIN_DATASET_FILE}"
        )

    dataframe = pd.read_csv(
        TRAIN_DATASET_FILE
    )

    allowed = (
        TRAIN_HEALTHY_BEARINGS
        | TRAIN_DAMAGED_BEARINGS
    )

    dataframe = dataframe[
        dataframe[
            "bearing_code"
        ].isin(
            allowed
        )
    ].copy()

    dataframe[
        "binary_condition"
    ] = np.where(
        dataframe[
            "condition"
        ] == "healthy",
        "healthy",
        "damaged",
    )

    return dataframe


# ============================================================
# EVALUATION
# ============================================================

def create_model():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )


def main():
    print()
    print("=" * 130)
    print(
        "PADERBORN STAGE 1: HEALTHY VS DAMAGED"
    )
    print("=" * 130)

    print()
    print(
        "TRAIN healthy : K001 K002 K003"
    )

    print(
        "TRAIN damaged : KA01 KA05 KA07 KI01 KI05 KI07"
    )

    print()
    print(
        "HELD-OUT healthy: K004 K005"
    )

    print(
        "HELD-OUT damaged: KA15 KA22 KA30 KI16 KI18 KI21"
    )

    train_df = load_training_dataset()
    test_df = build_heldout_test_dataset()

    train_bearings = set(
        train_df[
            "bearing_code"
        ].unique()
    )

    test_bearings = set(
        test_df[
            "bearing_code"
        ].unique()
    )

    overlap = (
        train_bearings
        & test_bearings
    )

    if overlap:
        raise RuntimeError(
            f"Bearing leakage detected: "
            f"{sorted(overlap)}"
        )

    print()
    print("=" * 130)
    print("DATA SPLIT")
    print("=" * 130)

    print()
    print(
        f"Training rows     : "
        f"{len(train_df)}"
    )

    print(
        f"Training bearings : "
        f"{sorted(train_bearings)}"
    )

    print(
        f"Test rows         : "
        f"{len(test_df)}"
    )

    print(
        f"Test bearings     : "
        f"{sorted(test_bearings)}"
    )

    print()
    print(
        "Training class counts:"
    )

    print(
        train_df
        .groupby(
            "binary_condition"
        )
        .size()
        .to_string()
    )

    print()
    print(
        "Test class counts:"
    )

    print(
        test_df
        .groupby(
            "binary_condition"
        )
        .size()
        .to_string()
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []
    prediction_records = []
    bearing_records = []

    for (
        feature_group,
        feature_columns,
    ) in FEATURE_GROUPS.items():
        print()
        print("=" * 130)

        print(
            f"FEATURE GROUP: "
            f"{feature_group}"
        )

        print("=" * 130)

        model = create_model()

        model.fit(
            train_df[
                feature_columns
            ],
            train_df[
                "binary_condition"
            ],
        )

        predictions = model.predict(
            test_df[
                feature_columns
            ]
        )

        probabilities = model.predict_proba(
            test_df[
                feature_columns
            ]
        )

        confidence = np.max(
            probabilities,
            axis=1,
        )

        accuracy = accuracy_score(
            test_df[
                "binary_condition"
            ],
            predictions,
        )

        macro_f1 = f1_score(
            test_df[
                "binary_condition"
            ],
            predictions,
            labels=CLASS_ORDER,
            average="macro",
            zero_division=0,
        )

        recalls = recall_score(
            test_df[
                "binary_condition"
            ],
            predictions,
            labels=CLASS_ORDER,
            average=None,
            zero_division=0,
        )

        recall_map = dict(
            zip(
                CLASS_ORDER,
                recalls,
            )
        )

        print()
        print(
            f"Accuracy       : "
            f"{accuracy:.4f}"
        )

        print(
            f"Macro F1       : "
            f"{macro_f1:.4f}"
        )

        print(
            f"Recall healthy : "
            f"{recall_map['healthy']:.4f}"
        )

        print(
            f"Recall damaged : "
            f"{recall_map['damaged']:.4f}"
        )

        print(
            f"Mean confidence: "
            f"{np.mean(confidence):.4f}"
        )

        print()
        print(
            classification_report(
                test_df[
                    "binary_condition"
                ],
                predictions,
                labels=CLASS_ORDER,
                digits=4,
                zero_division=0,
            )
        )

        results.append(
            {
                "feature_group": feature_group,
                "n_features": len(
                    feature_columns
                ),
                "train_samples": len(
                    train_df
                ),
                "test_samples": len(
                    test_df
                ),
                "train_bearings": len(
                    train_bearings
                ),
                "test_bearings": len(
                    test_bearings
                ),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "recall_healthy": recall_map[
                    "healthy"
                ],
                "recall_damaged": recall_map[
                    "damaged"
                ],
                "mean_confidence": float(
                    np.mean(
                        confidence
                    )
                ),
                "min_confidence": float(
                    np.min(
                        confidence
                    )
                ),
                "max_confidence": float(
                    np.max(
                        confidence
                    )
                ),
            }
        )

        class_to_index = {
            class_name: index
            for index, class_name
            in enumerate(
                model.classes_
            )
        }

        test_reset = test_df.reset_index(
            drop=True
        )

        for row_index in range(
            len(
                test_reset
            )
        ):
            row = test_reset.iloc[
                row_index
            ]

            record = {
                "feature_group": feature_group,
                "bearing_code": row[
                    "bearing_code"
                ],
                "binary_condition": row[
                    "binary_condition"
                ],
                "source_file": row[
                    "source_file"
                ],
                "window_index": row[
                    "window_index"
                ],
                "predicted_condition": predictions[
                    row_index
                ],
                "correct": (
                    row[
                        "binary_condition"
                    ]
                    == predictions[
                        row_index
                    ]
                ),
                "prediction_confidence": confidence[
                    row_index
                ],
            }

            for class_name in model.classes_:
                record[
                    f"probability_{class_name}"
                ] = probabilities[
                    row_index,
                    class_to_index[
                        class_name
                    ],
                ]

            prediction_records.append(
                record
            )

        print(
            "Per-bearing:"
        )

        for bearing_code in sorted(
            test_bearings
        ):
            mask = (
                test_df[
                    "bearing_code"
                ].to_numpy()
                == bearing_code
            )

            true_values = (
                test_df.loc[
                    test_df[
                        "bearing_code"
                    ]
                    == bearing_code,
                    "binary_condition",
                ]
                .to_numpy()
            )

            pred_values = predictions[
                mask
            ]

            conf_values = confidence[
                mask
            ]

            bearing_accuracy = (
                accuracy_score(
                    true_values,
                    pred_values,
                )
            )

            true_class = (
                true_values[
                    0
                ]
            )

            print(
                f"  {bearing_code:<5}"
                f" | true={true_class:<7}"
                f" | accuracy="
                f"{bearing_accuracy:.4f}"
                f" | mean_conf="
                f"{np.mean(conf_values):.4f}"
            )

            bearing_records.append(
                {
                    "feature_group": feature_group,
                    "bearing_code": bearing_code,
                    "binary_condition": true_class,
                    "samples": len(
                        true_values
                    ),
                    "accuracy": bearing_accuracy,
                    "mean_confidence": float(
                        np.mean(
                            conf_values
                        )
                    ),
                    "min_confidence": float(
                        np.min(
                            conf_values
                        )
                    ),
                }
            )

        matrix = confusion_matrix(
            test_df[
                "binary_condition"
            ],
            predictions,
            labels=CLASS_ORDER,
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=CLASS_ORDER,
        )

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        display.plot(
            ax=ax,
            values_format="d",
        )

        ax.set_title(
            "Stage 1 held-out healthy vs damaged\n"
            f"{feature_group}"
        )

        fig.tight_layout()

        fig.savefig(
            FIGURES_DIR
            / f"confusion_{feature_group}.png",
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    results_df = pd.DataFrame(
        results
    ).sort_values(
        [
            "macro_f1",
            "accuracy",
        ],
        ascending=False,
    )

    predictions_df = pd.DataFrame(
        prediction_records
    )

    per_bearing_df = pd.DataFrame(
        bearing_records
    )

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    per_bearing_df.to_csv(
        PER_BEARING_FILE,
        index=False,
    )

    print()
    print("=" * 130)
    print("OVERALL SUMMARY")
    print("=" * 130)

    print()
    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("PER-BEARING SUMMARY")
    print("=" * 130)

    print()
    print(
        per_bearing_df
        .sort_values(
            [
                "feature_group",
                "binary_condition",
                "bearing_code",
            ]
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("FILES SAVED")
    print("=" * 130)

    print()
    print(
        TEST_DATASET_FILE
    )

    print(
        RESULTS_FILE
    )

    print(
        PREDICTIONS_FILE
    )

    print(
        PER_BEARING_FILE
    )

    print(
        FIGURES_DIR
    )

    print()
    print("=" * 130)
    print(
        "STAGE-1 HEALTHY VS DAMAGED EVALUATION COMPLETE"
    )
    print("=" * 130)


if __name__ == "__main__":
    main()
