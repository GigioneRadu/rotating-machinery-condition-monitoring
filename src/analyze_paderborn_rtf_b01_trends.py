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
    / "paderborn_rtf_b01_measurement_summary.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paderborn_rtf_b01_trend_diagnostic.csv"
)

FIGURES_DIR = (
    PROJECT_ROOT
    / "figures"
    / "paderborn_rtf_b01_trends"
)


# ============================================================
# HELPERS
# ============================================================

def rolling_median(series, window=15):
    return (
        series
        .rolling(
            window=window,
            center=True,
            min_periods=1,
        )
        .median()
    )


def rolling_mad(series, window=15):
    def mad(values):
        values = np.asarray(values, dtype=float)
        med = np.median(values)
        return np.median(np.abs(values - med))

    return (
        series
        .rolling(
            window=window,
            center=True,
            min_periods=1,
        )
        .apply(
            mad,
            raw=True,
        )
    )


def save_line_plot(
    df,
    columns,
    title,
    ylabel,
    filename,
):
    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    for column in columns:
        ax.plot(
            df["measurement_number"],
            df[column],
            label=column,
            linewidth=1.4,
        )

    ax.set_xlabel(
        "Measurement number"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR
        / filename,
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
        "PADERBORN RTF B01 TREND + OPERATING-CONDITION DIAGNOSTIC"
    )
    print("=" * 130)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing input file:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Robust rolling trends
    # --------------------------------------------------------

    for column in [
        "frontal_rms",
        "rear_rms",
        "frontal_kurtosis",
        "rear_kurtosis",
        "mean_temp_t1_c",
        "mean_temp_t2_c",
        "mean_abs_speed_rpm",
        "mean_abs_static_load_n",
        "peak_dynamic_load_n",
    ]:
        df[
            f"{column}_rollmed15"
        ] = rolling_median(
            df[column],
            window=15,
        )

        df[
            f"{column}_rollmad15"
        ] = rolling_mad(
            df[column],
            window=15,
        )

    # --------------------------------------------------------
    # Simple scale normalization for vibration
    # This is exploratory, not a final health indicator.
    # --------------------------------------------------------

    speed_safe = (
        df["mean_abs_speed_rpm"]
        .clip(lower=1.0)
    )

    load_safe = (
        df["mean_abs_static_load_n"]
        .clip(lower=1.0)
    )

    df[
        "frontal_rms_per_krpm"
    ] = (
        df["frontal_rms"]
        / (
            speed_safe / 1000.0
        )
    )

    df[
        "rear_rms_per_krpm"
    ] = (
        df["rear_rms"]
        / (
            speed_safe / 1000.0
        )
    )

    df[
        "frontal_rms_per_krpm_kn"
    ] = (
        df["frontal_rms"]
        / (
            (speed_safe / 1000.0)
            * (load_safe / 1000.0)
        )
    )

    df[
        "rear_rms_per_krpm_kn"
    ] = (
        df["rear_rms"]
        / (
            (speed_safe / 1000.0)
            * (load_safe / 1000.0)
        )
    )

    # --------------------------------------------------------
    # Time blocks
    # --------------------------------------------------------

    n = len(df)

    block_edges = np.linspace(
        0,
        n,
        6,
        dtype=int,
    )

    block_labels = [
        "Q1_early",
        "Q2",
        "Q3",
        "Q4",
        "Q5_late",
    ]

    time_block = np.empty(
        n,
        dtype=object,
    )

    for i in range(5):
        start = block_edges[i]
        end = block_edges[i + 1]

        time_block[
            start:end
        ] = block_labels[i]

    df[
        "time_block"
    ] = time_block

    # --------------------------------------------------------
    # Summary by time block
    # --------------------------------------------------------

    block_summary = (
        df
        .groupby(
            "time_block",
            sort=False,
        )
        .agg(
            measurements=(
                "measurement_number",
                "count",
            ),
            speed_mean=(
                "mean_abs_speed_rpm",
                "mean",
            ),
            static_load_mean=(
                "mean_abs_static_load_n",
                "mean",
            ),
            temp_t1_mean=(
                "mean_temp_t1_c",
                "mean",
            ),
            frontal_rms_mean=(
                "frontal_rms",
                "mean",
            ),
            rear_rms_mean=(
                "rear_rms",
                "mean",
            ),
            frontal_rms_per_krpm_mean=(
                "frontal_rms_per_krpm",
                "mean",
            ),
            rear_rms_per_krpm_mean=(
                "rear_rms_per_krpm",
                "mean",
            ),
            frontal_kurtosis_mean=(
                "frontal_kurtosis",
                "mean",
            ),
            rear_kurtosis_mean=(
                "rear_kurtosis",
                "mean",
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Correlations with operating conditions
    # --------------------------------------------------------

    correlation_columns = [
        "frontal_rms",
        "rear_rms",
        "frontal_kurtosis",
        "rear_kurtosis",
        "mean_abs_speed_rpm",
        "mean_abs_static_load_n",
        "peak_dynamic_load_n",
        "mean_temp_t1_c",
        "mean_temp_t2_c",
        "measurement_number",
    ]

    correlations = (
        df[
            correlation_columns
        ]
        .corr(
            method="spearman"
        )
    )

    # --------------------------------------------------------
    # Save plots
    # --------------------------------------------------------

    save_line_plot(
        df,
        [
            "frontal_rms",
            "rear_rms",
        ],
        "B01 vibration RMS over run",
        "RMS amplitude",
        "01_rms_over_run.png",
    )

    save_line_plot(
        df,
        [
            "frontal_kurtosis",
            "rear_kurtosis",
        ],
        "B01 kurtosis over run",
        "Kurtosis",
        "02_kurtosis_over_run.png",
    )

    save_line_plot(
        df,
        [
            "mean_abs_speed_rpm",
        ],
        "B01 rotational speed over run",
        "Speed (rpm)",
        "03_speed_over_run.png",
    )

    save_line_plot(
        df,
        [
            "mean_abs_static_load_n",
            "peak_dynamic_load_n",
        ],
        "B01 load over run",
        "Load (N)",
        "04_load_over_run.png",
    )

    save_line_plot(
        df,
        [
            "mean_temp_t1_c",
            "mean_temp_t2_c",
        ],
        "B01 bearing temperatures over run",
        "Temperature (°C)",
        "05_temperature_over_run.png",
    )

    save_line_plot(
        df,
        [
            "frontal_rms_per_krpm",
            "rear_rms_per_krpm",
        ],
        "B01 exploratory RMS normalized by speed",
        "RMS / krpm",
        "06_rms_per_krpm.png",
    )

    save_line_plot(
        df,
        [
            "frontal_rms_per_krpm_kn",
            "rear_rms_per_krpm_kn",
        ],
        "B01 exploratory RMS normalized by speed and static load",
        "RMS / (krpm·kN)",
        "07_rms_per_krpm_kn.png",
    )

    save_line_plot(
        df,
        [
            "frontal_rms_rollmed15",
            "rear_rms_rollmed15",
        ],
        "B01 rolling-median RMS trend",
        "Rolling median RMS",
        "08_rms_rolling_median.png",
    )

    # --------------------------------------------------------
    # Save data
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 130)
    print("TIME-BLOCK SUMMARY")
    print("=" * 130)

    print()
    print(
        block_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("SPEARMAN CORRELATIONS WITH VIBRATION")
    print("=" * 130)

    for target in [
        "frontal_rms",
        "rear_rms",
    ]:
        print()
        print(
            f"{target}:"
        )

        series = (
            correlations[
                target
            ]
            .drop(
                target
            )
            .sort_values(
                key=np.abs,
                ascending=False,
            )
        )

        print(
            series.to_string()
        )

    print()
    print("=" * 130)
    print("EARLY VS LATE CHANGE")
    print("=" * 130)

    early = df.iloc[
        : max(
            1,
            int(
                0.2
                * len(df)
            ),
        )
    ]

    late = df.iloc[
        int(
            0.8
            * len(df)
        ):
    ]

    comparison_columns = [
        "frontal_rms",
        "rear_rms",
        "frontal_rms_per_krpm",
        "rear_rms_per_krpm",
        "frontal_rms_per_krpm_kn",
        "rear_rms_per_krpm_kn",
        "frontal_kurtosis",
        "rear_kurtosis",
        "mean_temp_t1_c",
    ]

    rows = []

    for column in comparison_columns:
        early_median = float(
            early[
                column
            ].median()
        )

        late_median = float(
            late[
                column
            ].median()
        )

        ratio = (
            late_median
            / early_median
            if abs(
                early_median
            ) > 1e-12
            else np.nan
        )

        rows.append(
            {
                "metric": column,
                "early_median": early_median,
                "late_median": late_median,
                "late_over_early": ratio,
            }
        )

    comparison = pd.DataFrame(
        rows
    )

    print()
    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print("=" * 130)
    print("FILES SAVED")
    print("=" * 130)

    print()
    print(
        OUTPUT_FILE
    )

    print(
        FIGURES_DIR
    )

    print()
    print("=" * 130)
    print(
        "B01 TREND DIAGNOSTIC COMPLETE"
    )
    print("=" * 130)


if __name__ == "__main__":
    main()
