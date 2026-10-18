from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


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

RESULTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_feature_ablation_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

LOADS = [0, 1, 2, 3]

RANDOM_STATE = 42
N_ESTIMATORS = 500

TARGET_COLUMN = "condition"


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
# LOAD DATA
# ============================================================

def load_dataset() -> pd.DataFrame:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_FILE}"
        )

    dataframe = pd.read_csv(DATASET_FILE)

    required_columns = [
        TARGET_COLUMN,
        "load_hp",
        *ALL_FEATURES,
    ]

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
# MODEL
# ============================================================

def create_model():
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )


# ============================================================
# EVALUATE ONE FEATURE GROUP
# ============================================================

def evaluate_feature_group(
    dataframe: pd.DataFrame,
    group_name: str,
    feature_columns: list[str],
):
    fold_results = []

    for test_load in LOADS:
        train_loads = [
            load
            for load in LOADS
            if load != test_load
        ]

        train_df = dataframe[
            dataframe["load_hp"].isin(train_loads)
        ]

        test_df = dataframe[
            dataframe["load_hp"] == test_load
        ]

        X_train = train_df[feature_columns]
        y_train = train_df[TARGET_COLUMN]

        X_test = test_df[feature_columns]
        y_test = test_df[TARGET_COLUMN]

        model = create_model()

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
        )

        fold_results.append(
            {
                "feature_group": group_name,
                "test_load_hp": test_load,
                "n_features": len(feature_columns),
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )

    return fold_results


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 105)
    print("CWRU FEATURE ABLATION - LEAVE-ONE-LOAD-OUT")
    print("=" * 105)

    dataframe = load_dataset()

    all_results = []

    for group_name, feature_columns in FEATURE_GROUPS.items():
        print()
        print("-" * 105)
        print(f"FEATURE GROUP: {group_name}")
        print("-" * 105)

        print(
            f"Features ({len(feature_columns)}): "
            + ", ".join(feature_columns)
        )

        fold_results = evaluate_feature_group(
            dataframe=dataframe,
            group_name=group_name,
            feature_columns=feature_columns,
        )

        all_results.extend(
            fold_results
        )

        group_df = pd.DataFrame(
            fold_results
        )

        for _, row in group_df.iterrows():
            print(
                f"Test {int(row['test_load_hp'])} HP"
                f" | accuracy={row['accuracy']:.4f}"
                f" | macro_f1={row['macro_f1']:.4f}"
            )

        print(
            f"Mean accuracy: "
            f"{group_df['accuracy'].mean():.4f}"
        )

        print(
            f"Mean Macro F1: "
            f"{group_df['macro_f1'].mean():.4f}"
        )

    results_df = pd.DataFrame(
        all_results
    )

    summary_df = (
        results_df
        .groupby(
            [
                "feature_group",
                "n_features",
            ],
            as_index=False,
        )
        .agg(
            mean_accuracy=("accuracy", "mean"),
            min_accuracy=("accuracy", "min"),
            max_accuracy=("accuracy", "max"),
            mean_macro_f1=("macro_f1", "mean"),
            min_macro_f1=("macro_f1", "min"),
            max_macro_f1=("macro_f1", "max"),
        )
        .sort_values(
            by=[
                "mean_accuracy",
                "mean_macro_f1",
            ],
            ascending=False,
        )
    )

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print()
    print("=" * 105)
    print("ABLATION SUMMARY")
    print("=" * 105)

    print()
    print(
        summary_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 105)
    print("INTERPRETATION GUIDE")
    print("=" * 105)

    print()
    print(
        "If time_domain_only remains near 1.0, the task can largely be solved "
        "from vibration severity/amplitude."
    )

    print(
        "If envelope_only remains strong, the bearing fault-frequency "
        "signatures carry substantial diagnostic information by themselves."
    )

    print(
        "If physics_frequency_only substantially outperforms raw_frequency_only, "
        "envelope analysis is adding useful fault-specific information."
    )

    print(
        "If all_features is much better than every reduced group, the classifier "
        "benefits from combining severity and physics-based features."
    )

    print()
    print("Detailed fold results saved to:")
    print(RESULTS_FILE)

    print()
    print("=" * 105)
    print("FEATURE ABLATION COMPLETE")
    print("=" * 105)


if __name__ == "__main__":
    main()
