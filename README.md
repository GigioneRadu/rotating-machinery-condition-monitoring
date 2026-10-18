# Rotating Machinery Condition Monitoring

A Python condition-monitoring project for rotating machinery, built around real run-to-failure bearing data from the Paderborn benchmark.

The system uses a **machine-specific, condition-matched baseline detector**. Instead of relying only on one fixed vibration limit, it asks:

> **Is the current machine vibrating more than expected for this speed and load?**

---
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/7a48008e-412e-4873-92d8-c6557ed2a19b" />


## Project status

The current version includes:

- real measured run-to-failure vibration data;
- two accelerometer channels;
- operating-condition matching;
- frozen commissioning baselines;
- robust anomaly scoring;
- `NORMAL / WARNING / ANOMALY` states;
- persistence logic using consecutive measurements;
- out-of-distribution operating-condition detection;
- comparison across three independent experiments;
- Streamlit dashboard;
- comparison against official experiment stop criteria.

This is a **research / portfolio prototype**, not an industrially certified diagnostic system.

---

## Main results

| Experiment | Frozen reference | First sustained warning | First sustained anomaly | Lead before first official action |
|---|---:|---:|---:|---:|
| B01 | M0134–M0190 | M0208 | M0208 | N/A |
| B02 | M0052–M0250 | M0864 | M0866 | **16.1 min** |
| B03 | M0001–M0200 | M0547 | M0549 | **19.7 min** |

### B02

The frozen detector produced a sustained anomaly at:

```text
M0866
2021-10-25 13:15:57
```

The first official intervention occurred at:

```text
2021-10-25 13:32:00
```

after accelerometer RMS exceeded `0.65 V` twice.

Early-warning lead:

```text
16.05 minutes
```

The test was later completed at `14:36` after the threshold had been exceeded four times.

### B03

The frozen detector produced a sustained anomaly at:

```text
M0549
2021-11-05 18:03:16
```

The experiment stopped at:

```text
2021-11-05 18:23:00
```

after bearing-housing temperature exceeded `110 °C`.

Early-warning lead:

```text
19.73 minutes
```

### B01

B01 was used mainly as a development and robustness case.

Its chronology contains interruption/restart behavior, so no simple official-stop lead time is reported.

The final post-restart configuration detected:

```text
First sustained warning : M0208
First sustained anomaly : M0208
```

---

## Dashboard

The Streamlit dashboard provides:

- benchmark overview;
- B01 / B02 / B03 selection;
- first warning and anomaly measurements;
- early-warning lead time;
- anomaly-score evolution;
- frozen-reference visualization;
- actual vs expected RMS;
- optional `Volts / g` display;
- operating-condition support;
- event timeline;
- detector methodology.

Run it with:

```powershell
uv run streamlit run app.py
```

---

## Core idea

Raw vibration amplitude depends on more than mechanical health. It can also change with:

- shaft speed;
- static load;
- dynamic load;
- operating regime.

Therefore the detector compares each measurement with **reference measurements recorded under similar operating conditions**.

Conceptually:

```text
machine-specific frozen reference
                ↓
operating conditions
(speed + static load + dynamic load)
                ↓
10 nearest reference measurements
                ↓
median expected vibration RMS
                ↓
actual RMS vs expected RMS
                ↓
robust residual normalization
                ↓
condition-matched anomaly score
                ↓
NORMAL / WARNING / ANOMALY
```

---

## Detector logic

### 1. Frozen reference baseline

The experiment-specific reference intervals are:

```text
B01 → M0134–M0190
B02 → M0052–M0250
B03 → M0001–M0200
```

These intervals are intentionally frozen and are not moved after detector evaluation.

### 2. Operating-condition representation

Each measurement is represented by:

```text
mean shaft speed
mean static load
peak dynamic load
```

These variables are standardized using reference-interval statistics only.

### 3. Condition matching

For every measurement, the detector finds:

```text
K = 10
```

nearest reference points in operating-condition space.

The median vibration RMS of these neighbors becomes the expected vibration level.

Two channels are monitored:

```text
frontal accelerometer RMS
rear accelerometer RMS
```

### 4. Residual

For each channel:

```text
residual = log(actual RMS / expected RMS)
```

The residual is normalized using robust reference statistics:

```text
median
MAD
robust sigma = 1.4826 × MAD
```

Only positive deviations contribute to the monitoring score.

### 5. Combined score

```text
score = max(
    frontal positive robust z-score,
    rear positive robust z-score
)
```

### 6. Monitoring states

```text
score < 3        → NORMAL
3 <= score < 6   → WARNING
score >= 6       → ANOMALY
```

### 7. Persistence rule

An event is accepted only after:

```text
5 consecutive in-support measurements
```

cross the relevant threshold.

This reduces sensitivity to isolated spikes.

### 8. Operating-condition support

The detector checks whether the current operating point is represented by the frozen reference interval.

Measurements outside that support are marked:

```text
OOD_CONDITION
```

rather than being treated as normal detector evidence.

---

## Why condition matching matters

In B02, around the degradation transition:

```text
M0870

frontal actual   ≈ 0.1158 V
frontal expected ≈ 0.0575 V

rear actual      ≈ 0.2333 V
rear expected    ≈ 0.1145 V
```

The absolute vibration was still below the official `0.65 V` intervention criterion, but it was already roughly twice the level expected for similar operating conditions.

That is the central idea of this project:

> Detect abnormal deviation from a machine's own expected behavior before a large absolute protection threshold is reached.

---

## Data characteristics

The Paderborn run-to-failure files used here have:

```text
Sampling frequency : 128 kHz
Sampling duration  : 1.6 s
Vibration unit     : Volts
Conversion factor  : 10 g/V
```

Each MAT file contains:

```text
accHorizFrontal_C
accHorizRear_A
measTime
```

Operating-condition CSV files include:

```text
setDynLoad / N
peak_dynLoad / N
setStatLoad / N
meanAbs_statLoad / N
setSpeed / rpm
meanAbs_speed / rpm
```

Temperature files include:

```text
Mean Abs. Temp. T1 / °C
Mean Abs. Temp. T2 / °C
Mean Room Temp. / °C
```

---

## Experiment summary

### B01

```text
Measurements            : 377
Frozen reference        : M0134–M0190
First sustained warning : M0208
First sustained anomaly : M0208
Final anomaly score     : 124.03
In-support fraction     : 83.6%
```

B01 contains interruption/restart behavior and was retained as a development / robustness case.

### B02

```text
Measurements            : 1116
Frozen reference        : M0052–M0250
First sustained warning : M0864
First sustained anomaly : M0866
Final anomaly score     : 54.04
In-support fraction     : 93.8%
```

Official experiment behavior:

```text
13:32 → first intervention after RMS > 0.65 V twice
14:36 → final completion after RMS > 0.65 V four times
```

Detector anomaly:

```text
13:15:57
```

Lead before first official intervention:

```text
16.05 minutes
```

### B03

```text
Measurements            : 614
Frozen reference        : M0001–M0200
First sustained warning : M0547
First sustained anomaly : M0549
Final anomaly score     : 53.76
In-support fraction     : 97.2%
```

Official experiment stop criterion:

```text
bearing housing temperature > 110 °C
```

Detector anomaly:

```text
18:03:16
```

Official stop:

```text
18:23:00
```

Lead:

```text
19.73 minutes
```

---

## Project structure

```text
02_rotating_machinery_condition_monitoring/
│
├── app.py
├── pyproject.toml
│
├── data/
│   ├── raw/
│   │   └── paderborn_run_to_failure/
│   │       ├── B01/
│   │       ├── B02/
│   │       └── B03/
│   │
│   └── processed/
│       ├── paderborn_rtf_b01_generic_monitoring.csv
│       ├── paderborn_rtf_b02_generic_monitoring.csv
│       ├── paderborn_rtf_b03_generic_monitoring.csv
│       └── paderborn_rtf_b01_b03_comparison_revised.csv
│
├── figures/
│   ├── paderborn_rtf_generic_monitoring/
│   └── paderborn_rtf_comparison/
│
└── src/
    ├── run_paderborn_rtf_monitoring.py
    ├── compare_paderborn_rtf_b01_b03_revised.py
    ├── analyze_paderborn_rtf_b01_measurements.py
    ├── analyze_paderborn_rtf_b02_measurements.py
    ├── analyze_paderborn_rtf_b03_measurements.py
    └── ...
```

The repository also contains exploratory scripts used to investigate benchmark structure, domain shift, spectral behavior, timeline gaps, temperature evolution and condition-matched monitoring.

---

## Installation

This project uses `uv`.

From the project directory:

```powershell
uv sync
```

If Streamlit is not already included:

```powershell
uv add streamlit
```

Main packages:

```text
numpy
pandas
scipy
matplotlib
scikit-learn
joblib
streamlit
```

---

## Run the generic detector

### B01

```powershell
uv run python src/run_paderborn_rtf_monitoring.py --experiment B01
```

### B02

```powershell
uv run python src/run_paderborn_rtf_monitoring.py --experiment B02
```

### B03

```powershell
uv run python src/run_paderborn_rtf_monitoring.py --experiment B03
```

Expected sustained anomaly measurements:

```text
B01 → M0208
B02 → M0866
B03 → M0549
```

---

## Generate the comparison

```powershell
uv run python src/compare_paderborn_rtf_b01_b03_revised.py
```

Output:

```text
data/processed/paderborn_rtf_b01_b03_comparison_revised.csv
```

Figures are generated under:

```text
figures/paderborn_rtf_comparison/
```

---

## Run the dashboard

```powershell
uv run streamlit run app.py
```

Streamlit normally opens at:

```text
http://localhost:8501
```

---

## Methodological decisions

### Frozen settings

B02 and B03 were evaluated with detector settings frozen before examining the final detector output.

The generic detector uses:

```text
K neighbors       = 10
warning threshold = 3
anomaly threshold = 6
persistence       = 5 measurements
```

### No exact fault-onset claim

The detected anomaly measurement is **not claimed to be the exact physical instant when mechanical damage begins**.

It is the first sustained point where measured vibration deviates strongly from expected machine-specific behavior according to this detector.

### Controlled benchmark data

These are controlled run-to-failure benchmark experiments.

They are not equivalent to uncontrolled industrial field data.

### B01 limitation

Because B01 includes interruption/restart behavior, its chronology is not used for a simple early-warning lead-time claim.

### No industrial certification

The system has not been validated across:

- different machine designs;
- different bearing families;
- sensor mounting variations;
- industrial noise sources;
- long-term maintenance histories;
- production environments.

---

## What this project demonstrated

Several important engineering lessons emerged during development.

### High benchmark accuracy can be misleading

Earlier classification experiments showed that excellent performance inside one controlled benchmark subset does not guarantee robustness under:

- unseen fault severity;
- unseen physical bearing;
- different damage mechanism;
- different operating domain.

### High confidence does not guarantee validity

Out-of-domain predictions can still be wrong with high classifier confidence.

### Machine-specific baselines are useful

A more realistic monitoring architecture is:

```text
commission machine
        ↓
learn machine-specific reference behavior
        ↓
monitor deviation over time
        ↓
raise warning / anomaly
        ↓
perform detailed diagnosis after abnormal behavior is established
```

### Operating conditions matter

Raw RMS can change because of speed and load, not only because of degradation.

Condition matching helps separate these effects.

---

## Current architecture

This project currently focuses on **Stage 1 — condition monitoring**:

```text
NORMAL
  ↓
WARNING
  ↓
ANOMALY
```

A future diagnostic stage could attempt:

```text
outer-race fault
inner-race fault
ball fault
cage fault
unknown / unsupported condition
```

An industrial system should be allowed to return:

```text
UNKNOWN
```

rather than forcing every anomaly into a known fault class.

---

## Possible next improvements

- order-domain features using rotational speed;
- spectral and envelope indicators;
- online streaming sensor input;
- automatic commissioning workflow;
- baseline-quality checks;
- machine configuration profiles;
- persistent monitoring database;
- alert history;
- maintenance annotations;
- additional run-to-failure experiments;
- uncertainty estimation;
- sensor-fault detection;
- multi-machine dashboard;
- edge-PC / industrial mini-PC deployment.

---

## Tech stack

```text
Python 3.12
uv
NumPy
Pandas
SciPy
Matplotlib
scikit-learn
Streamlit
```

---

## Conclusion

The strongest supported conclusion from the current experiments is:

> A frozen machine-specific condition-matched baseline detector identified persistent vibration anomalies in multiple independent controlled run-to-failure experiments. In B02 and B03, those anomalies appeared before the official experiment intervention or stop criteria.

The project therefore evolved from a simple fault-classification exercise into a more realistic condition-monitoring architecture focused on machine-specific reference behavior, operating-condition awareness and persistent anomaly detection.

---

## Disclaimer

This repository is an educational and research-oriented prototype.

It must not be used as the sole basis for maintenance, safety or shutdown decisions on real machinery without appropriate engineering validation, sensor qualification, calibration and domain-specific safety procedures.
