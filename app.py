from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Rotating Machinery Condition Monitoring",
    page_icon="⚙️",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

COMPARISON_FILE = (
    PROCESSED_DIR
    / "paderborn_rtf_b01_b03_comparison_revised.csv"
)

EXPERIMENT_FILES = {
    "B01": PROCESSED_DIR / "paderborn_rtf_b01_generic_monitoring.csv",
    "B02": PROCESSED_DIR / "paderborn_rtf_b02_generic_monitoring.csv",
    "B03": PROCESSED_DIR / "paderborn_rtf_b03_generic_monitoring.csv",
}

EXPERIMENT_DESCRIPTIONS = {
    "B01": (
        "Development / robustness case. The run contains interruption "
        "and restart behavior, so no simple official-stop lead time is reported."
    ),
    "B02": (
        "Frozen detector benchmark. The first official intervention occurred "
        "after accelerometer RMS exceeded 0.65 V twice."
    ),
    "B03": (
        "Frozen detector benchmark. The experiment stopped when bearing "
        "housing temperature exceeded 110 °C."
    ),
}


# ============================================================
# CONSTANTS
# ============================================================

WARNING_THRESHOLD = 3.0
ANOMALY_THRESHOLD = 6.0
V_TO_G = 10.0


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1450px;
        }

        .hero {
            padding: 1.25rem 1.4rem;
            border-radius: 16px;
            border: 1px solid rgba(128,128,128,0.22);
            margin-bottom: 1.4rem;
            background: linear-gradient(
                135deg,
                rgba(70,130,180,0.10),
                rgba(90,90,180,0.05)
            );
        }

        .hero-title {
            font-size: 2.25rem;
            font-weight: 750;
            margin-bottom: 0.25rem;
        }

        .hero-subtitle {
            font-size: 1.03rem;
            opacity: 0.78;
        }

        .status-card {
            padding: 1rem 1.1rem;
            border-radius: 14px;
            border: 1px solid rgba(128,128,128,0.22);
            min-height: 112px;
        }

        .status-label {
            font-size: 0.85rem;
            opacity: 0.72;
            margin-bottom: 0.3rem;
        }

        .status-value {
            font-size: 1.8rem;
            font-weight: 720;
            line-height: 1.1;
        }

        .status-note {
            font-size: 0.78rem;
            opacity: 0.66;
            margin-top: 0.35rem;
        }

        .normal-card {
            background: rgba(46, 160, 67, 0.08);
        }

        .warning-card {
            background: rgba(245, 166, 35, 0.10);
        }

        .anomaly-card {
            background: rgba(220, 53, 69, 0.10);
        }

        .info-card {
            background: rgba(13, 110, 253, 0.08);
        }

        div[data-testid="stMetricValue"] {
            font-size: 2rem;
        }

        .small-note {
            font-size: 0.84rem;
            opacity: 0.7;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

@st.cache_data
def load_comparison():
    df = pd.read_csv(COMPARISON_FILE)

    for column in [
        "warning_timestamp",
        "anomaly_timestamp",
        "official_first_action_time",
        "official_final_stop_time",
    ]:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    return df


@st.cache_data
def load_experiment(experiment):
    df = pd.read_csv(
        EXPERIMENT_FILES[experiment]
    )

    for column in [
        "timestamp_operating",
        "timestamp",
    ]:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    return df


def get_timestamp_column(df):
    for candidate in [
        "timestamp_operating",
        "timestamp",
    ]:
        if candidate in df.columns:
            return candidate

    return None


def fmt_measurement(value):
    if pd.isna(value):
        return "N/A"

    return f"M{int(value):04d}"


def fmt_minutes(value):
    if pd.isna(value):
        return "N/A"

    return f"{float(value):.1f} min"


def card(label, value, note="", css_class="info-card"):
    st.markdown(
        f"""
        <div class="status-card {css_class}">
            <div class="status-label">{label}</div>
            <div class="status-value">{value}</div>
            <div class="status-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# LOAD
# ============================================================

comparison = load_comparison()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">⚙️ Rotating Machinery Condition Monitoring</div>
        <div class="hero-subtitle">
            Condition-matched baseline monitoring on Paderborn
            run-to-failure bearing experiments
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TOP SUMMARY
# ============================================================

b02 = comparison.loc[
    comparison["experiment"] == "B02"
].iloc[0]

b03 = comparison.loc[
    comparison["experiment"] == "B03"
].iloc[0]

c1, c2, c3, c4 = st.columns(4)

with c1:
    card(
        "Validated experiments",
        "3",
        "B01 development + B02/B03 frozen evaluation",
        "info-card",
    )

with c2:
    card(
        "B02 early-warning lead",
        fmt_minutes(
            b02["lead_minutes_to_first_official_action"]
        ),
        "Before first official intervention",
        "normal-card",
    )

with c3:
    card(
        "B03 early-warning lead",
        fmt_minutes(
            b03["lead_minutes_to_first_official_action"]
        ),
        "Before automatic thermal stop",
        "normal-card",
    )

with c4:
    card(
        "Persistence rule",
        "5",
        "Consecutive in-support measurements",
        "warning-card",
    )


# ============================================================
# BENCHMARK TABLE
# ============================================================

st.subheader("Benchmark overview")

table = comparison[
    [
        "experiment",
        "measurements",
        "reference_start",
        "reference_end",
        "first_sustained_warning",
        "first_sustained_anomaly",
        "lead_minutes_to_first_official_action",
        "final_anomaly_score",
        "in_support_fraction",
    ]
].copy()

table["Reference"] = (
    "M"
    + table["reference_start"].astype(int).astype(str).str.zfill(4)
    + "–M"
    + table["reference_end"].astype(int).astype(str).str.zfill(4)
)

table["Warning"] = table[
    "first_sustained_warning"
].apply(fmt_measurement)

table["Anomaly"] = table[
    "first_sustained_anomaly"
].apply(fmt_measurement)

table["Lead to official action"] = table[
    "lead_minutes_to_first_official_action"
].apply(fmt_minutes)

table["In-support"] = (
    table["in_support_fraction"]
    * 100.0
).map(lambda value: f"{value:.1f}%")

table = table[
    [
        "experiment",
        "measurements",
        "Reference",
        "Warning",
        "Anomaly",
        "Lead to official action",
        "final_anomaly_score",
        "In-support",
    ]
]

table.columns = [
    "Experiment",
    "Measurements",
    "Reference baseline",
    "First warning",
    "First anomaly",
    "Lead to official action",
    "Final anomaly score",
    "In-support data",
]

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# DETAIL
# ============================================================

st.divider()

left, right = st.columns(
    [1, 3]
)

with left:
    experiment = st.selectbox(
        "Experiment",
        ["B01", "B02", "B03"],
        index=1,
    )

    unit_mode = st.radio(
        "Vibration unit",
        ["Volts", "g"],
        horizontal=True,
    )

with right:
    st.info(
        EXPERIMENT_DESCRIPTIONS[
            experiment
        ]
    )

df = load_experiment(
    experiment
)

row = comparison.loc[
    comparison["experiment"]
    == experiment
].iloc[0]

warning_measurement = int(
    row[
        "first_sustained_warning"
    ]
)

anomaly_measurement = int(
    row[
        "first_sustained_anomaly"
    ]
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    card(
        "First sustained warning",
        fmt_measurement(
            warning_measurement
        ),
        "score ≥ 3 for 5 consecutive in-support measurements",
        "warning-card",
    )

with m2:
    card(
        "First sustained anomaly",
        fmt_measurement(
            anomaly_measurement
        ),
        "score ≥ 6 for 5 consecutive in-support measurements",
        "anomaly-card",
    )

with m3:
    lead_value = row[
        "lead_minutes_to_first_official_action"
    ]

    card(
        "Lead to first official action",
        fmt_minutes(
            lead_value
        ),
        "N/A for B01 because of restart chronology",
        "normal-card" if not pd.isna(lead_value) else "info-card",
    )

with m4:
    card(
        "Final anomaly score",
        f"{row['final_anomaly_score']:.2f}",
        "Robust positive z-score",
        "anomaly-card",
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "Anomaly score",
        "Actual vs expected RMS",
        "Timeline & support",
    ]
)


# ============================================================
# TAB 1
# ============================================================

with tab1:
    fig, ax = plt.subplots(
        figsize=(13, 5)
    )

    x = df[
        "measurement_number"
    ]

    y = df[
        "condition_matched_anomaly_score"
    ]

    ax.plot(
        x,
        y,
        linewidth=1.2,
        label="Anomaly score",
    )

    ax.axhspan(
        0,
        WARNING_THRESHOLD,
        alpha=0.05,
        label="Normal zone",
    )

    ax.axhspan(
        WARNING_THRESHOLD,
        ANOMALY_THRESHOLD,
        alpha=0.07,
        label="Warning zone",
    )

    ax.axhspan(
        ANOMALY_THRESHOLD,
        max(
            float(y.max()) * 1.05,
            ANOMALY_THRESHOLD + 1,
        ),
        alpha=0.05,
        label="Anomaly zone",
    )

    ax.axhline(
        WARNING_THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label="Warning threshold",
    )

    ax.axhline(
        ANOMALY_THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label="Anomaly threshold",
    )

    ax.axvspan(
        int(row["reference_start"]),
        int(row["reference_end"]),
        alpha=0.12,
        label="Frozen reference",
    )

    ax.axvline(
        anomaly_measurement,
        linestyle=":",
        linewidth=1.5,
        label="First sustained anomaly",
    )

    ax.set_title(
        f"{experiment} condition-matched anomaly score"
    )

    ax.set_xlabel(
        "Measurement number"
    )

    ax.set_ylabel(
        "Positive robust z-score"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend(
        loc="upper left",
        ncol=2,
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)


# ============================================================
# TAB 2
# ============================================================

with tab2:
    channel = st.radio(
        "Channel",
        ["rear_rms", "frontal_rms"],
        horizontal=True,
    )

    factor = (
        V_TO_G
        if unit_mode == "g"
        else 1.0
    )

    ylabel = (
        "RMS acceleration (g)"
        if unit_mode == "g"
        else "RMS (V)"
    )

    actual = (
        df[channel]
        * factor
    )

    expected = (
        df[f"{channel}_expected"]
        * factor
    )

    fig, ax = plt.subplots(
        figsize=(13, 5)
    )

    ax.plot(
        df["measurement_number"],
        actual,
        linewidth=1.1,
        label="Actual RMS",
    )

    ax.plot(
        df["measurement_number"],
        expected,
        linewidth=1.1,
        label="Condition-matched expected RMS",
    )

    ax.axvline(
        anomaly_measurement,
        linestyle=":",
        linewidth=1.5,
        label="First sustained anomaly",
    )

    if experiment == "B02":
        official_threshold = (
            0.65 * factor
        )

        ax.axhline(
            official_threshold,
            linestyle="--",
            linewidth=1.0,
            label=(
                "Official RMS criterion "
                f"({official_threshold:.2f} {unit_mode})"
            ),
        )

    ax.set_title(
        f"{experiment} — {channel}: actual vs expected"
    )

    ax.set_xlabel(
        "Measurement number"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)

    if unit_mode == "g":
        st.caption(
            "Conversion uses the Paderborn experiment factor 10 g/V."
        )


# ============================================================
# TAB 3
# ============================================================

with tab3:
    support_fraction = (
        df[
            "operating_condition_in_support"
        ].mean()
        * 100.0
    )

    ood_count = int(
        (
            ~df[
                "operating_condition_in_support"
            ]
        ).sum()
    )

    s1, s2 = st.columns(2)

    with s1:
        card(
            "Operating-condition support",
            f"{support_fraction:.1f}%",
            "Measurements represented by the frozen reference operating envelope",
            "normal-card",
        )

    with s2:
        card(
            "OOD operating points",
            str(ood_count),
            "Measurements outside the frozen reference support",
            "warning-card" if ood_count else "normal-card",
        )

    st.write("")

    timestamp_column = get_timestamp_column(
        df
    )

    events = []

    for event_name, measurement_number in [
        ("First sustained warning", warning_measurement),
        ("First sustained anomaly", anomaly_measurement),
    ]:
        match = df[
            df["measurement_number"]
            == measurement_number
        ]

        timestamp_value = None

        if (
            timestamp_column
            and not match.empty
        ):
            timestamp_value = match.iloc[0][
                timestamp_column
            ]

        events.append(
            {
                "Event": event_name,
                "Measurement": fmt_measurement(
                    measurement_number
                ),
                "Timestamp": timestamp_value,
            }
        )

    if not pd.isna(
        row["official_first_action_time"]
    ):
        events.append(
            {
                "Event": "First official action",
                "Measurement": "—",
                "Timestamp": row[
                    "official_first_action_time"
                ],
            }
        )

    if (
        not pd.isna(
            row["official_final_stop_time"]
        )
        and row[
            "official_final_stop_time"
        ]
        != row[
            "official_first_action_time"
        ]
    ):
        events.append(
            {
                "Event": "Official final stop",
                "Measurement": "—",
                "Timestamp": row[
                    "official_final_stop_time"
                ],
            }
        )

    st.dataframe(
        pd.DataFrame(events),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# METHOD
# ============================================================

st.divider()

with st.expander(
    "How the detector works"
):
    st.markdown(
        """
        The detector uses a machine-specific frozen reference interval.

        For each new measurement:

        1. Shaft speed, static load and dynamic load define the operating state.
        2. The 10 closest reference measurements are selected.
        3. Their median RMS defines the expected vibration for those conditions.
        4. Actual and expected RMS are compared using a log-ratio residual.
        5. The residual is normalized with robust reference statistics.
        6. The larger score from the two accelerometer channels becomes the anomaly score.
        7. Warning threshold = 3; anomaly threshold = 6.
        8. An event is accepted only after 5 consecutive in-support measurements.

        The objective is therefore to detect deviation from the machine's own expected
        behavior rather than relying only on one fixed absolute vibration limit.
        """
    )


st.caption(
    "Research / portfolio prototype based on controlled run-to-failure benchmark data. "
    "Results are not industrial certification, and detected anomaly points should not "
    "be interpreted as exact physical fault-onset ground truth."
)
