from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_fault_severity_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_within_severity_leave_one_load_out_results.csv"
)

SEVERITIES = [0.007, 0.014, 0.021]
LOADS = [0, 1, 2, 3]

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


def create_model():
    return RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )


def main():
    print()
    print("=" * 120)
    print("CWRU WITHIN-SEVERITY LEAVE-ONE-LOAD-OUT")
    print("=" * 120)

    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_FILE}"
        )

    df = pd.read_csv(DATASET_FILE)

    rows = []

    for severity in SEVERITIES:
        severity_df = df[
            df["severity_inch"].round(3)
            == round(severity, 3)
        ].copy()

        print()
        print("=" * 120)
        print(
            f"SEVERITY: {severity:.3f} in"
        )
        print("=" * 120)

        for feature_group, features in FEATURE_GROUPS.items():
            fold_accuracies = []
            fold_f1s = []

            print()
            print(
                f"Feature group: {feature_group}"
            )

            for test_load in LOADS:
                train_df = severity_df[
                    severity_df["load_hp"]
                    != test_load
                ]

                test_df = severity_df[
                    severity_df["load_hp"]
                    == test_load
                ]

                X_train = train_df[features]
                y_train = train_df["condition"]

                X_test = test_df[features]
                y_test = test_df["condition"]

                model = create_model()
                model.fit(
                    X_train,
                    y_train,
                )

                pred = model.predict(
                    X_test
                )

                accuracy = accuracy_score(
                    y_test,
                    pred,
                )

                macro_f1 = f1_score(
                    y_test,
                    pred,
                    average="macro",
                )

                fold_accuracies.append(
                    accuracy
                )

                fold_f1s.append(
                    macro_f1
                )

                rows.append(
                    {
                        "severity_inch": severity,
                        "feature_group": feature_group,
                        "test_load_hp": test_load,
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                    }
                )

                print(
                    f"  Test {test_load} HP"
                    f" | accuracy={accuracy:.4f}"
                    f" | macro_f1={macro_f1:.4f}"
                )

            print(
                f"  Mean accuracy={sum(fold_accuracies)/len(fold_accuracies):.4f}"
                f" | Mean macro_f1={sum(fold_f1s)/len(fold_f1s):.4f}"
            )

    results_df = pd.DataFrame(
        rows
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary = (
        results_df
        .groupby(
            [
                "severity_inch",
                "feature_group",
            ],
            as_index=False,
        )
        .agg(
            mean_accuracy=(
                "accuracy",
                "mean",
            ),
            min_accuracy=(
                "accuracy",
                "min",
            ),
            mean_macro_f1=(
                "macro_f1",
                "mean",
            ),
            min_macro_f1=(
                "macro_f1",
                "min",
            ),
        )
        .sort_values(
            [
                "severity_inch",
                "mean_accuracy",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    print()
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print()

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 120)
    print("INTERPRETATION")
    print("=" * 120)
    print()
    print(
        "If 0.014-inch performs well here but poorly in cross-severity tests, "
        "the problem is domain shift between severities rather than an "
        "intrinsically unclassifiable 0.014-inch dataset."
    )
    print(
        "If 0.014-inch also performs poorly within severity, then the "
        "recordings themselves are harder to separate and should be "
        "investigated before further model tuning."
    )

    print()
    print("Saved:")
    print(OUTPUT_FILE)

    print()
    print("=" * 120)
    print("WITHIN-SEVERITY EVALUATION COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
