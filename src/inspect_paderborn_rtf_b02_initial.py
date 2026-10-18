from pathlib import Path

from scipy.io import loadmat
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

B02_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "paderborn_run_to_failure"
    / "B02"
)

VIBRATION_DIR = (
    B02_DIR
    / "vibrationData"
)

OPERATING_FILE = (
    B02_DIR
    / "B02_operatingConditions.csv"
)

TEMPERATURE_FILE = (
    B02_DIR
    / "B02_meanTemperatures.csv"
)

LOG_FILE = (
    B02_DIR
    / "B02_log.pdf"
)


# ============================================================
# HELPERS
# ============================================================

def summarize_mat(file_path: Path):
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

    print()
    print(f"MAT FILE: {file_path.name}")
    print(f"Top-level variables: {public_keys}")

    for key in public_keys:
        value = np.asarray(
            data[key]
        )

        print(
            f"  {key:<25}"
            f" shape={value.shape}"
            f" dtype={value.dtype}"
        )

        if (
            value.size > 0
            and np.issubdtype(
                value.dtype,
                np.number,
            )
        ):
            flat = value.reshape(-1)

            print(
                f"    preview="
                f"{np.array2string(flat[:6], precision=6, separator=', ')}"
            )


def inspect_csv(
    file_path: Path,
    label: str,
):
    print()
    print("=" * 120)
    print(label)
    print("=" * 120)

    if not file_path.exists():
        print(
            f"MISSING: {file_path.name}"
        )
        return None

    df = pd.read_csv(
        file_path
    )

    print(
        f"Shape: {df.shape}"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    print()
    print(
        "First 3 rows:"
    )

    print(
        df.head(3).to_string(
            index=False
        )
    )

    print()
    print(
        "Last 3 rows:"
    )

    print(
        df.tail(3).to_string(
            index=False
        )
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 120)
    print("PADERBORN RUN-TO-FAILURE B02 INITIAL INSPECTION")
    print("=" * 120)

    if not B02_DIR.exists():
        raise FileNotFoundError(
            f"B02 folder not found:\n{B02_DIR}"
        )

    print()
    print("TOP-LEVEL CONTENTS")

    for item in sorted(
        B02_DIR.iterdir()
    ):
        marker = (
            "[DIR ]"
            if item.is_dir()
            else "[FILE]"
        )

        print(
            f"{marker} {item.name}"
        )

    if not VIBRATION_DIR.exists():
        raise FileNotFoundError(
            f"Missing vibrationData folder:\n"
            f"{VIBRATION_DIR}"
        )

    mat_files = sorted(
        VIBRATION_DIR.glob(
            "*.mat"
        )
    )

    print()
    print("=" * 120)
    print("VIBRATION DATA")
    print("=" * 120)

    print(
        f"MAT files: {len(mat_files)}"
    )

    if not mat_files:
        raise FileNotFoundError(
            f"No MAT files found in:\n"
            f"{VIBRATION_DIR}"
        )

    print(
        f"First MAT: {mat_files[0].name}"
    )

    print(
        f"Last MAT : {mat_files[-1].name}"
    )

    summarize_mat(
        mat_files[0]
    )

    if len(mat_files) > 1:
        summarize_mat(
            mat_files[-1]
        )

    operating = inspect_csv(
        OPERATING_FILE,
        "B02 OPERATING CONDITIONS",
    )

    temperatures = inspect_csv(
        TEMPERATURE_FILE,
        "B02 MEAN TEMPERATURES",
    )

    print()
    print("=" * 120)
    print("CONSISTENCY CHECK")
    print("=" * 120)

    print(
        f"Log PDF exists          : "
        f"{LOG_FILE.exists()}"
    )

    print(
        f"MAT count               : "
        f"{len(mat_files)}"
    )

    if operating is not None:
        print(
            f"Operating rows          : "
            f"{len(operating)}"
        )

    if temperatures is not None:
        print(
            f"Temperature rows        : "
            f"{len(temperatures)}"
        )

    if (
        operating is not None
        and temperatures is not None
    ):
        print(
            f"Counts aligned          : "
            f"{len(mat_files) == len(operating) == len(temperatures)}"
        )

    print()
    print("=" * 120)
    print("B02 INITIAL INSPECTION COMPLETE")
    print("=" * 120)


if __name__ == "__main__":
    main()
