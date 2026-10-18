from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

B01_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "paderborn_run_to_failure"
    / "B01"
)

VIBRATION_DIR = (
    B01_DIR
    / "vibrationData"
)

OPERATING_CONDITIONS_FILE = (
    B01_DIR
    / "B01_operatingConditions.csv"
)

TEMPERATURES_FILE = (
    B01_DIR
    / "B01_meanTemperatures.csv"
)


# ============================================================
# HELPERS
# ============================================================

def summarize_value(value, max_items=8):
    array = np.asarray(value)

    summary = {
        "type": type(value).__name__,
        "shape": array.shape,
        "dtype": str(array.dtype),
    }

    if array.size == 0:
        summary["preview"] = "<empty>"
        return summary

    flat = array.reshape(-1)

    if np.issubdtype(array.dtype, np.number):
        preview = flat[:max_items]

        summary["preview"] = np.array2string(
            preview,
            precision=6,
            separator=", ",
        )

        try:
            summary["min"] = float(
                np.nanmin(array)
            )

            summary["max"] = float(
                np.nanmax(array)
            )

            summary["mean"] = float(
                np.nanmean(array)
            )
        except Exception:
            pass

    else:
        summary["preview"] = str(
            flat[:max_items]
        )

    return summary


def print_struct_fields(
    obj,
    prefix="",
    depth=0,
    max_depth=3,
):
    indent = "    " * depth

    if depth > max_depth:
        print(
            f"{indent}{prefix}<max depth reached>"
        )
        return

    field_names = getattr(
        obj,
        "_fieldnames",
        None,
    )

    if field_names:
        print(
            f"{indent}{prefix}"
            f"{type(obj).__name__} "
            f"fields={field_names}"
        )

        for field_name in field_names:
            value = getattr(
                obj,
                field_name,
            )

            child_prefix = (
                f"{field_name}: "
            )

            if getattr(
                value,
                "_fieldnames",
                None,
            ):
                print_struct_fields(
                    value,
                    prefix=child_prefix,
                    depth=depth + 1,
                    max_depth=max_depth,
                )

            elif isinstance(
                value,
                np.ndarray,
            ) and value.dtype == object:
                print(
                    f"{'    ' * (depth + 1)}"
                    f"{child_prefix}"
                    f"object array "
                    f"shape={value.shape}"
                )

                flattened = value.reshape(-1)

                for index, item in enumerate(
                    flattened[:10]
                ):
                    print_struct_fields(
                        item,
                        prefix=f"[{index}] ",
                        depth=depth + 2,
                        max_depth=max_depth,
                    )

            else:
                summary = summarize_value(
                    value
                )

                print(
                    f"{'    ' * (depth + 1)}"
                    f"{child_prefix}"
                    f"type={summary['type']}, "
                    f"shape={summary['shape']}, "
                    f"dtype={summary['dtype']}, "
                    f"preview={summary.get('preview')}"
                )

        return

    summary = summarize_value(
        obj
    )

    print(
        f"{indent}{prefix}"
        f"type={summary['type']}, "
        f"shape={summary['shape']}, "
        f"dtype={summary['dtype']}, "
        f"preview={summary.get('preview')}"
    )


def inspect_mat_file(
    file_path: Path,
):
    print()
    print("=" * 130)
    print(
        f"MAT FILE: {file_path.name}"
    )
    print("=" * 130)

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
    print(
        f"Top-level variables: "
        f"{public_keys}"
    )

    for key in public_keys:
        value = data[
            key
        ]

        print()
        print(
            f"VARIABLE: {key}"
        )

        print_struct_fields(
            value,
            prefix="",
            depth=0,
            max_depth=4,
        )


def inspect_csv(
    file_path: Path,
    label: str,
):
    print()
    print("=" * 130)
    print(label)
    print("=" * 130)

    if not file_path.exists():
        print(
            f"MISSING: {file_path}"
        )
        return

    dataframe = pd.read_csv(
        file_path
    )

    print()
    print(
        f"Shape: {dataframe.shape}"
    )

    print()
    print(
        "Columns:"
    )

    for column in dataframe.columns:
        print(
            f"  {column}"
        )

    print()
    print(
        "First 10 rows:"
    )

    print(
        dataframe
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Last 5 rows:"
    )

    print(
        dataframe
        .tail(5)
        .to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 130)
    print(
        "PADERBORN RUN-TO-FAILURE B01 CONTENT INSPECTION"
    )
    print("=" * 130)

    if not VIBRATION_DIR.exists():
        raise FileNotFoundError(
            f"Missing folder:\n"
            f"{VIBRATION_DIR}"
        )

    mat_files = sorted(
        VIBRATION_DIR.glob(
            "*.mat"
        )
    )

    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files found in:\n"
            f"{VIBRATION_DIR}"
        )

    print()
    print(
        f"MAT files found: "
        f"{len(mat_files)}"
    )

    print(
        f"First file: "
        f"{mat_files[0].name}"
    )

    print(
        f"Last file : "
        f"{mat_files[-1].name}"
    )

    # Inspect beginning and end of run.
    inspect_mat_file(
        mat_files[0]
    )

    if len(mat_files) > 1:
        inspect_mat_file(
            mat_files[-1]
        )

    inspect_csv(
        OPERATING_CONDITIONS_FILE,
        "B01 OPERATING CONDITIONS",
    )

    inspect_csv(
        TEMPERATURES_FILE,
        "B01 MEAN TEMPERATURES",
    )

    print()
    print("=" * 130)
    print(
        "CONTENT INSPECTION COMPLETE"
    )
    print("=" * 130)


if __name__ == "__main__":
    main()
