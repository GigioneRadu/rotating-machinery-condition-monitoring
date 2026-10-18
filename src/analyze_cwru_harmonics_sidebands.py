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

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_severity_spectral_diagnostics.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "cwru_harmonic_sideband_summary.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "cwru_harmonic_sideband_diagnostics"
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

TARGET_TYPES = [
    "fundamental",
    "2x",
    "3x",
    "minus_1x_shaft",
    "plus_1x_shaft",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n\n"
            "Run src/analyze_cwru_severity_spectra.py first."
        )

    dataframe = pd.read_csv(
        INPUT_FILE
    )

    required_columns = [
        "condition",
        "severity_inch",
        "source_file",
        "rpm",
        "envelope_band_low_hz",
        "envelope_band_high_hz",
        "fault_frequency_name",
        "target_type",
        "target_frequency_hz",
        "measured_peak_frequency_hz",
        "measured_peak_amplitude",
    ]

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return dataframe


# ============================================================
# BUILD WIDE SUMMARY
# ============================================================

def build_wide_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    pivot = (
        dataframe
        .pivot_table(
            index=[
                "condition",
                "severity_inch",
                "source_file",
                "rpm",
                "envelope_band_low_hz",
                "envelope_band_high_hz",
                "fault_frequency_name",
            ],
            columns="target_type",
            values="measured_peak_amplitude",
            aggfunc="first",
        )
        .reset_index()
    )

    for target_type in TARGET_TYPES:
        if target_type not in pivot.columns:
            pivot[
                target_type
            ] = np.nan

    # --------------------------------------------------------
    # Aggregate physics features
    # --------------------------------------------------------

    pivot[
        "harmonic_sum"
    ] = (
        pivot["fundamental"].fillna(0.0)
        + pivot["2x"].fillna(0.0)
        + pivot["3x"].fillna(0.0)
    )

    pivot[
        "sideband_sum"
    ] = (
        pivot["minus_1x_shaft"].fillna(0.0)
        + pivot["plus_1x_shaft"].fillna(0.0)
    )

    pivot[
        "total_fault_pattern_energy"
    ] = (
        pivot["harmonic_sum"]
        + pivot["sideband_sum"]
    )

    epsilon = 1e-12

    pivot[
        "harmonics_to_fundamental_ratio"
    ] = (
        (
            pivot["2x"].fillna(0.0)
            + pivot["3x"].fillna(0.0)
        )
        / (
            pivot["fundamental"].fillna(0.0)
            + epsilon
        )
    )

    pivot[
        "sidebands_to_fundamental_ratio"
    ] = (
        pivot["sideband_sum"]
        / (
            pivot["fundamental"].fillna(0.0)
            + epsilon
        )
    )

    pivot[
        "fundamental_fraction"
    ] = (
        pivot["fundamental"].fillna(0.0)
        / (
            pivot[
                "total_fault_pattern_energy"
            ]
            + epsilon
        )
    )

    return pivot


# ============================================================
# PRINT CONDITION SUMMARY
# ============================================================

def print_condition_summary(
    wide_df: pd.DataFrame,
):
    metrics = [
        "fundamental",
        "2x",
        "3x",
        "minus_1x_shaft",
        "plus_1x_shaft",
        "harmonic_sum",
        "sideband_sum",
        "total_fault_pattern_energy",
        "harmonics_to_fundamental_ratio",
        "sidebands_to_fundamental_ratio",
        "fundamental_fraction",
    ]

    for condition in CONDITIONS:
        print()
        print("=" * 120)
        print(
            f"CONDITION: {condition}"
        )
        print("=" * 120)

        condition_df = wide_df[
            wide_df["condition"]
            == condition
        ]

        for (
            band_low,
            band_high,
        ), band_df in condition_df.groupby(
            [
                "envelope_band_low_hz",
                "envelope_band_high_hz",
            ]
        ):
            print()
            print(
                f"Envelope band: "
                f"{int(band_low)}-{int(band_high)} Hz"
            )

            summary = (
                band_df
                .set_index(
                    "severity_inch"
                )[
                    metrics
                ]
                .sort_index()
            )

            print()
            print(
                summary.to_string(
                    float_format=lambda value: f"{value:.6f}"
                )
            )


# ============================================================
# FIND MOST SEVERITY-STABLE BAND
# ============================================================

def calculate_band_stability(
    wide_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    metrics = [
        "fundamental",
        "harmonic_sum",
        "sideband_sum",
        "total_fault_pattern_energy",
        "harmonics_to_fundamental_ratio",
        "sidebands_to_fundamental_ratio",
        "fundamental_fraction",
    ]

    for condition in CONDITIONS:
        condition_df = wide_df[
            wide_df["condition"]
            == condition
        ]

        for (
            band_low,
            band_high,
        ), band_df in condition_df.groupby(
            [
                "envelope_band_low_hz",
                "envelope_band_high_hz",
            ]
        ):
            for metric in metrics:
                values = (
                    band_df
                    .sort_values(
                        "severity_inch"
                    )[
                        metric
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                mean_abs = float(
                    np.mean(
                        np.abs(values)
                    )
                )

                std = float(
                    np.std(
                        values,
                        ddof=0,
                    )
                )

                cv = (
                    std / mean_abs
                    if mean_abs > 0.0
                    else np.nan
                )

                rows.append(
                    {
                        "condition": condition,
                        "envelope_band_low_hz": int(
                            band_low
                        ),
                        "envelope_band_high_hz": int(
                            band_high
                        ),
                        "metric": metric,
                        "mean_abs": mean_abs,
                        "std": std,
                        "coefficient_of_variation": cv,
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PLOT NORMALIZED PATTERN METRICS
# ============================================================

def save_ratio_plots(
    wide_df: pd.DataFrame,
):
    ratio_metrics = [
        "harmonics_to_fundamental_ratio",
        "sidebands_to_fundamental_ratio",
        "fundamental_fraction",
    ]

    for condition in CONDITIONS:
        condition_df = wide_df[
            wide_df["condition"]
            == condition
        ]

        for metric in ratio_metrics:
            fig, ax = plt.subplots(
                figsize=(10, 6)
            )

            for (
                band_low,
                band_high,
            ), band_df in condition_df.groupby(
                [
                    "envelope_band_low_hz",
                    "envelope_band_high_hz",
                ]
            ):
                band_df = (
                    band_df
                    .sort_values(
                        "severity_inch"
                    )
                )

                ax.plot(
                    band_df[
                        "severity_inch"
                    ],
                    band_df[
                        metric
                    ],
                    marker="o",
                    label=(
                        f"{int(band_low)}-"
                        f"{int(band_high)} Hz"
                    ),
                )

            ax.set_title(
                f"{condition} - {metric}"
            )

            ax.set_xlabel(
                "Fault severity [inch]"
            )

            ax.set_ylabel(
                metric
            )

            ax.legend()

            ax.grid(
                alpha=0.25
            )

            fig.tight_layout()

            output_file = (
                FIGURES_DIR
                / condition
                / f"{metric}.png"
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
# MAIN
# ============================================================

def main():
    print()
    print("=" * 120)
    print("CWRU HARMONIC AND SIDEBAND DIAGNOSTICS")
    print("=" * 120)

    dataframe = load_data()

    wide_df = build_wide_summary(
        dataframe
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    wide_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_condition_summary(
        wide_df
    )

    stability_df = (
        calculate_band_stability(
            wide_df
        )
    )

    print()
    print("=" * 120)
    print("MOST SEVERITY-STABLE NORMALIZED METRICS")
    print("=" * 120)

    print()

    normalized_metrics = [
        "harmonics_to_fundamental_ratio",
        "sidebands_to_fundamental_ratio",
        "fundamental_fraction",
    ]

    normalized_stability = (
        stability_df[
            stability_df[
                "metric"
            ]
            .isin(
                normalized_metrics
            )
        ]
        .sort_values(
            by=[
                "condition",
                "coefficient_of_variation",
            ]
        )
    )

    for condition in CONDITIONS:
        print()
        print(
            f"Condition: {condition}"
        )

        print(
            normalized_stability[
                normalized_stability[
                    "condition"
                ]
                == condition
            ]
            .head(8)
            .to_string(
                index=False
            )
        )

    save_ratio_plots(
        wide_df
    )

    print()
    print("=" * 120)
    print("FILES SAVED")
    print("=" * 120)

    print()
    print("Wide harmonic/sideband summary:")
    print(
        OUTPUT_FILE
    )

    print()
    print("Diagnostic ratio figures:")
    print(
        FIGURES_DIR
    )

    print()
    print("=" * 120)
    print("HARMONIC AND SIDEBAND DIAGNOSTICS COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
