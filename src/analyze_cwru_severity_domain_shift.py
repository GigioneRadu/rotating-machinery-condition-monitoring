from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_fault_severity_dataset.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "cwru_severity_diagnostics"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_severity_diagnostic_summary.csv"
)


# ============================================================
# SETTINGS
# ============================================================

SEVERITIES = [
    0.007,
    0.014,
    0.021,
]

CONDITIONS = [
    "ball_fault",
    "inner_race",
    "outer_race",
]

FEATURES_TO_ANALYZE = [
    "rms",
    "kurtosis",
    "bpfo_envelope",
    "bpfi_envelope",
    "bsf_envelope",
    "ftf_envelope",
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset() -> pd.DataFrame:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_FILE}\n\n"
            "Run src/build_cwru_severity_dataset.py first."
        )

    dataframe = pd.read_csv(
        DATASET_FILE
    )

    required_columns = [
        "condition",
        "severity_inch",
        "load_hp",
        "rpm",
        "source_file",
        "window_index",
        *FEATURES_TO_ANALYZE,
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
# SUMMARY TABLE
# ============================================================

def build_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for condition in CONDITIONS:
        condition_df = dataframe[
            dataframe["condition"]
            == condition
        ]

        for severity in SEVERITIES:
            severity_df = condition_df[
                np.isclose(
                    condition_df[
                        "severity_inch"
                    ].to_numpy(),
                    severity,
                    rtol=0.0,
                    atol=1e-9,
                )
            ]

            for feature in FEATURES_TO_ANALYZE:
                values = (
                    severity_df[
                        feature
                    ]
                    .to_numpy()
                )

                rows.append(
                    {
                        "condition": condition,
                        "severity_inch": severity,
                        "feature": feature,
                        "count": len(values),
                        "mean": float(
                            np.mean(values)
                        ),
                        "std": float(
                            np.std(
                                values,
                                ddof=1,
                            )
                        ),
                        "min": float(
                            np.min(values)
                        ),
                        "q25": float(
                            np.quantile(
                                values,
                                0.25,
                            )
                        ),
                        "median": float(
                            np.median(
                                values
                            )
                        ),
                        "q75": float(
                            np.quantile(
                                values,
                                0.75,
                            )
                        ),
                        "max": float(
                            np.max(values)
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# DISTRIBUTION PLOTS
# ============================================================

def save_distribution_plot(
    dataframe: pd.DataFrame,
    condition: str,
    feature: str,
):
    condition_df = dataframe[
        dataframe["condition"]
        == condition
    ]

    data_by_severity = []

    labels = []

    for severity in SEVERITIES:
        values = (
            condition_df[
                np.isclose(
                    condition_df[
                        "severity_inch"
                    ].to_numpy(),
                    severity,
                    rtol=0.0,
                    atol=1e-9,
                )
            ][
                feature
            ]
            .to_numpy()
        )

        data_by_severity.append(
            values
        )

        labels.append(
            f"{severity:.3f}"
        )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.boxplot(
        data_by_severity,
        tick_labels=labels,
        showmeans=True,
    )

    ax.set_title(
        f"{condition} - {feature}"
    )

    ax.set_xlabel(
        "Fault severity [inch]"
    )

    ax.set_ylabel(
        feature
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    fig.tight_layout()

    output_file = (
        FIGURES_DIR
        / condition
        / f"{feature}.png"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_file,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# FEATURE MEAN PLOTS
# ============================================================

def save_feature_mean_plot(
    dataframe: pd.DataFrame,
    feature: str,
):
    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for condition in CONDITIONS:
        condition_df = dataframe[
            dataframe["condition"]
            == condition
        ]

        means = []
        stds = []

        for severity in SEVERITIES:
            values = (
                condition_df[
                    np.isclose(
                        condition_df[
                            "severity_inch"
                        ].to_numpy(),
                        severity,
                        rtol=0.0,
                        atol=1e-9,
                    )
                ][
                    feature
                ]
                .to_numpy()
            )

            means.append(
                float(
                    np.mean(values)
                )
            )

            stds.append(
                float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )
            )

        ax.errorbar(
            SEVERITIES,
            means,
            yerr=stds,
            marker="o",
            capsize=4,
            label=condition,
        )

    ax.set_title(
        f"Cross-severity feature behavior - {feature}"
    )

    ax.set_xlabel(
        "Fault severity [inch]"
    )

    ax.set_ylabel(
        feature
    )

    ax.legend()

    ax.grid(
        alpha=0.25,
    )

    fig.tight_layout()

    output_file = (
        FIGURES_DIR
        / f"comparison_{feature}.png"
    )

    fig.savefig(
        output_file,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# CLASS-SEPARATION TABLE
# ============================================================

def print_feature_separation(
    summary_df: pd.DataFrame,
):
    print()
    print("=" * 115)
    print("FEATURE MEANS BY SEVERITY AND CONDITION")
    print("=" * 115)

    for feature in FEATURES_TO_ANALYZE:
        print()
        print("-" * 115)
        print(
            f"FEATURE: {feature}"
        )
        print("-" * 115)

        feature_df = summary_df[
            summary_df["feature"]
            == feature
        ]

        pivot = (
            feature_df
            .pivot(
                index="severity_inch",
                columns="condition",
                values="mean",
            )
        )

        print()
        print(
            pivot.to_string()
        )


# ============================================================
# OVERLAP DIAGNOSTIC
# ============================================================

def print_severity_overlap_diagnostic(
    dataframe: pd.DataFrame,
):
    print()
    print("=" * 115)
    print("0.014-INCH DOMAIN-SHIFT DIAGNOSTIC")
    print("=" * 115)

    print()
    print(
        "For each feature and class, compare the 0.014-inch mean "
        "with the mean range formed by 0.007 and 0.021 inch."
    )

    print()

    for condition in CONDITIONS:
        print("-" * 115)
        print(
            f"Condition: {condition}"
        )
        print("-" * 115)

        condition_df = dataframe[
            dataframe["condition"]
            == condition
        ]

        for feature in FEATURES_TO_ANALYZE:
            means = {}

            for severity in SEVERITIES:
                values = (
                    condition_df[
                        np.isclose(
                            condition_df[
                                "severity_inch"
                            ].to_numpy(),
                            severity,
                            rtol=0.0,
                            atol=1e-9,
                        )
                    ][
                        feature
                    ]
                    .to_numpy()
                )

                means[
                    severity
                ] = float(
                    np.mean(values)
                )

            lower = min(
                means[0.007],
                means[0.021],
            )

            upper = max(
                means[0.007],
                means[0.021],
            )

            middle = means[
                0.014
            ]

            if (
                middle >= lower
                and middle <= upper
            ):
                relationship = (
                    "inside endpoint range"
                )
            else:
                relationship = (
                    "outside endpoint range"
                )

            print(
                f"{feature:<16}"
                f" 0.007={means[0.007]:>11.6f}"
                f" | 0.014={means[0.014]:>11.6f}"
                f" | 0.021={means[0.021]:>11.6f}"
                f" | {relationship}"
            )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 115)
    print("CWRU SEVERITY DOMAIN-SHIFT DIAGNOSTICS")
    print("=" * 115)

    dataframe = load_dataset()

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df = build_summary(
        dataframe
    )

    SUMMARY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Per-condition boxplots
    # --------------------------------------------------------

    for condition in CONDITIONS:
        for feature in FEATURES_TO_ANALYZE:
            save_distribution_plot(
                dataframe=dataframe,
                condition=condition,
                feature=feature,
            )

    # --------------------------------------------------------
    # Cross-condition mean +/- std plots
    # --------------------------------------------------------

    for feature in FEATURES_TO_ANALYZE:
        save_feature_mean_plot(
            dataframe=dataframe,
            feature=feature,
        )

    # --------------------------------------------------------
    # Console summaries
    # --------------------------------------------------------

    print_feature_separation(
        summary_df
    )

    print_severity_overlap_diagnostic(
        dataframe
    )

    print()
    print("=" * 115)
    print("FILES SAVED")
    print("=" * 115)

    print()
    print("Summary CSV:")
    print(
        SUMMARY_FILE
    )

    print()
    print("Figures:")
    print(
        FIGURES_DIR
    )

    print()
    print(
        "Generated plots:"
    )

    print(
        f"- {len(CONDITIONS) * len(FEATURES_TO_ANALYZE)} "
        f"per-condition boxplots"
    )

    print(
        f"- {len(FEATURES_TO_ANALYZE)} "
        f"cross-condition comparison plots"
    )

    print()
    print("=" * 115)
    print("SEVERITY DIAGNOSTICS COMPLETE")
    print("=" * 115)


if __name__ == "__main__":
    main()
