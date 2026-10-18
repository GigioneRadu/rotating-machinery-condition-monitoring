from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_real_bearing_dataset.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "random_forest_cwru_real.joblib"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "cwru_model"
)

CONFUSION_MATRIX_FILE = (
    FIGURES_DIR
    / "confusion_matrix_load3_test.png"
)

FEATURE_IMPORTANCE_FILE = (
    FIGURES_DIR
    / "feature_importance_rf_impurity.png"
)

PERMUTATION_IMPORTANCE_FILE = (
    FIGURES_DIR
    / "feature_importance_permutation_test.png"
)

FEATURE_IMPORTANCE_CSV = (
    FIGURES_DIR
    / "feature_importance.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_real_model_predictions_load3.csv"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

TRAIN_LOADS = [0, 1, 2]
TEST_LOAD = 3

RANDOM_STATE = 42
N_ESTIMATORS = 500

# Signal metadata used when saving the model artifact.
SAMPLING_FREQUENCY = 12000
WINDOW_DURATION_SECONDS = 1.0


# ============================================================
# FEATURE SET
# ============================================================

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

TARGET_COLUMN = "condition"


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_FILE}\n\n"
            "Run src/build_cwru_dataset.py first."
        )

    dataframe = pd.read_csv(DATASET_FILE)

    required_columns = (
        FEATURE_COLUMNS
        + [
            TARGET_COLUMN,
            "source_file",
            "load_hp",
            "rpm",
            "window_index",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{missing_columns}"
        )

    return dataframe


# ============================================================
# SPLIT BY OPERATING CONDITION
# ============================================================

def split_dataset(dataframe: pd.DataFrame):
    """
    Important:
    We intentionally do NOT use a random train/test split.

    Training data:
        0 HP, 1 HP, 2 HP

    Test data:
        3 HP

    Therefore all 3 HP source recordings are unseen during training.
    """

    train_df = dataframe[
        dataframe["load_hp"].isin(TRAIN_LOADS)
    ].copy()

    test_df = dataframe[
        dataframe["load_hp"] == TEST_LOAD
    ].copy()

    if train_df.empty:
        raise ValueError(
            "Training set is empty."
        )

    if test_df.empty:
        raise ValueError(
            "Test set is empty."
        )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    return (
        train_df,
        test_df,
        X_train,
        y_train,
        X_test,
        y_test,
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_split_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    print()
    print("=" * 100)
    print("EXPERIMENT SPLIT")
    print("=" * 100)

    print()
    print(
        "Training loads : "
        + ", ".join(
            f"{load} HP"
            for load in TRAIN_LOADS
        )
    )

    print(
        f"Test load      : {TEST_LOAD} HP"
    )

    print()
    print(
        f"Training samples: {len(train_df)}"
    )

    print(
        f"Test samples    : {len(test_df)}"
    )

    print()
    print("Training samples per class:")

    print(
        train_df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Test samples per class:")

    print(
        test_df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Training source files:")

    print(
        train_df[
            [
                "source_file",
                TARGET_COLUMN,
                "load_hp",
                "rpm",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            by=[
                TARGET_COLUMN,
                "load_hp",
            ]
        )
        .to_string(index=False)
    )

    print()
    print("Test source files:")

    print(
        test_df[
            [
                "source_file",
                TARGET_COLUMN,
                "load_hp",
                "rpm",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            by=TARGET_COLUMN
        )
        .to_string(index=False)
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
):
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    print()
    print("=" * 100)
    print("TEST RESULTS - UNSEEN 3 HP LOAD")
    print("=" * 100)

    print()
    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Macro F1 : {macro_f1:.4f}"
    )

    print()
    print("Classification report:")

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    return (
        predictions,
        probabilities,
        accuracy,
        macro_f1,
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    model,
    y_test: pd.Series,
    predictions: np.ndarray,
):
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels = list(
        model.classes_
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    print()
    print("Confusion matrix:")
    print()

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=[
            f"true_{label}"
            for label in labels
        ],
        columns=[
            f"pred_{label}"
            for label in labels
        ],
    )

    print(
        matrix_dataframe.to_string()
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title(
        "CWRU Real Data - Test on Unseen 3 HP Load"
    )

    fig.tight_layout()

    fig.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# RANDOM FOREST IMPURITY IMPORTANCE
# ============================================================

def save_impurity_importance(
    model,
):
    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "rf_impurity_importance": model.feature_importances_,
        }
    ).sort_values(
        by="rf_impurity_importance",
        ascending=False,
    )

    print()
    print("=" * 100)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 100)

    print()

    print(
        importance_df.to_string(
            index=False
        )
    )

    plot_df = importance_df.sort_values(
        by="rf_impurity_importance",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        plot_df["feature"],
        plot_df["rf_impurity_importance"],
    )

    ax.set_xlabel(
        "Random Forest impurity importance"
    )

    ax.set_title(
        "CWRU Real Data - Random Forest Feature Importance"
    )

    fig.tight_layout()

    fig.savefig(
        FEATURE_IMPORTANCE_FILE,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    return importance_df


# ============================================================
# PERMUTATION IMPORTANCE ON HELD-OUT TEST LOAD
# ============================================================

def save_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=1,
        scoring="accuracy",
    )

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "permutation_mean": result.importances_mean,
            "permutation_std": result.importances_std,
        }
    ).sort_values(
        by="permutation_mean",
        ascending=False,
    )

    print()
    print("=" * 100)
    print("PERMUTATION IMPORTANCE - UNSEEN 3 HP TEST SET")
    print("=" * 100)

    print()

    print(
        importance_df.to_string(
            index=False
        )
    )

    plot_df = importance_df.sort_values(
        by="permutation_mean",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        plot_df["feature"],
        plot_df["permutation_mean"],
        xerr=plot_df["permutation_std"],
    )

    ax.set_xlabel(
        "Decrease in test accuracy after feature permutation"
    )

    ax.set_title(
        "CWRU Real Data - Permutation Importance on Unseen 3 HP Load"
    )

    fig.tight_layout()

    fig.savefig(
        PERMUTATION_IMPORTANCE_FILE,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    return importance_df


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    model,
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
):
    output = test_df[
        [
            "source_file",
            "condition",
            "load_hp",
            "rpm",
            "window_index",
        ]
    ].copy()

    output["predicted_condition"] = predictions

    output["correct"] = (
        output["condition"]
        == output["predicted_condition"]
    )

    class_to_probability_index = {
        class_name: index
        for index, class_name in enumerate(
            model.classes_
        )
    }

    for class_name in model.classes_:
        probability_index = (
            class_to_probability_index[
                class_name
            ]
        )

        output[
            f"probability_{class_name}"
        ] = probabilities[
            :,
            probability_index,
        ]

    output[
        "prediction_confidence"
    ] = np.max(
        probabilities,
        axis=1,
    )

    PREDICTIONS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print()
    print("=" * 100)
    print("PREDICTION CONFIDENCE BY TRUE CLASS")
    print("=" * 100)

    print()

    confidence_summary = (
        output
        .groupby("condition")[
            "prediction_confidence"
        ]
        .agg(
            [
                "mean",
                "min",
                "max",
            ]
        )
    )

    print(
        confidence_summary.to_string()
    )

    print()
    print("Misclassified windows:")

    errors = output[
        ~output["correct"]
    ]

    if errors.empty:
        print(
            "None"
        )
    else:
        print(
            errors.to_string(
                index=False
            )
        )


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    accuracy: float,
    macro_f1: float,
):
    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "train_loads": TRAIN_LOADS,
        "test_load": TEST_LOAD,
        "sampling_frequency": SAMPLING_FREQUENCY,
        "window_duration_seconds": WINDOW_DURATION_SECONDS,
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "classes": list(model.classes_),
    }

    joblib.dump(
        artifact,
        MODEL_FILE,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 100)
    print("TRAIN CWRU REAL-DATA RANDOM FOREST")
    print("=" * 100)

    dataframe = load_dataset()

    (
        train_df,
        test_df,
        X_train,
        y_train,
        X_test,
        y_test,
    ) = split_dataset(
        dataframe
    )

    print_split_summary(
        train_df,
        test_df,
    )

    print()
    print("=" * 100)
    print("TRAINING")
    print("=" * 100)

    print()
    print(
        f"Random Forest trees: {N_ESTIMATORS}"
    )

    print(
        f"Random state       : {RANDOM_STATE}"
    )

    model = train_model(
        X_train,
        y_train,
    )

    print()
    print("Training complete.")

    (
        predictions,
        probabilities,
        accuracy,
        macro_f1,
    ) = evaluate_model(
        model,
        X_test,
        y_test,
    )

    save_confusion_matrix(
        model,
        y_test,
        predictions,
    )

    impurity_importance = (
        save_impurity_importance(
            model
        )
    )

    permutation_importance_df = (
        save_permutation_importance(
            model,
            X_test,
            y_test,
        )
    )

    importance_merged = (
        impurity_importance
        .merge(
            permutation_importance_df,
            on="feature",
            how="outer",
        )
    )

    importance_merged.to_csv(
        FEATURE_IMPORTANCE_CSV,
        index=False,
    )

    save_predictions(
        model,
        test_df,
        predictions,
        probabilities,
    )

    save_model(
        model,
        accuracy,
        macro_f1,
    )

    print()
    print("=" * 100)
    print("FILES SAVED")
    print("=" * 100)

    print()
    print(f"Model:")
    print(MODEL_FILE)

    print()
    print("Confusion matrix:")
    print(CONFUSION_MATRIX_FILE)

    print()
    print("RF feature importance:")
    print(FEATURE_IMPORTANCE_FILE)

    print()
    print("Permutation importance:")
    print(PERMUTATION_IMPORTANCE_FILE)

    print()
    print("Feature importance CSV:")
    print(FEATURE_IMPORTANCE_CSV)

    print()
    print("Test predictions:")
    print(PREDICTIONS_FILE)

    print()
    print("=" * 100)
    print("TRAINING AND EVALUATION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
