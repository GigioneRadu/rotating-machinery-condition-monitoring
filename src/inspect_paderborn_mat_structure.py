from pathlib import Path
import numpy as np
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PADERBORN_DIR = PROJECT_ROOT / "data" / "raw" / "paderborn"

FILES_TO_INSPECT = [
    PADERBORN_DIR / "K001" / "N15_M07_F10_K001_1.mat",
    PADERBORN_DIR / "KA01" / "N15_M07_F10_KA01_1.mat",
    PADERBORN_DIR / "KI01" / "N15_M07_F10_KI01_1.mat",
]

def describe(value, indent=0, name=None, depth=0):
    p = " " * indent
    label = f"{name}: " if name else ""
    print(f"{p}{label}type={type(value).__name__}")

    if depth >= 5:
        return

    if isinstance(value, np.ndarray):
        print(f"{p}  shape={value.shape}")
        print(f"{p}  dtype={value.dtype}")

        if value.dtype.names:
            print(f"{p}  fields={list(value.dtype.names)}")
            if value.size:
                first = value.flat[0]
                for field in value.dtype.names:
                    describe(first[field], indent + 4, field, depth + 1)
            return

        if value.dtype == object:
            if value.size:
                describe(value.flat[0], indent + 4, "first_element", depth + 1)
            return

        if np.issubdtype(value.dtype, np.number):
            flat = value.reshape(-1)
            print(f"{p}  preview={flat[:10]}")
            return

    if hasattr(value, "_fieldnames"):
        print(f"{p}  fields={value._fieldnames}")
        for field in value._fieldnames:
            describe(getattr(value, field), indent + 4, field, depth + 1)
        return

    try:
        print(f"{p}  value={value}")
    except Exception:
        pass

def inspect_file(file_path: Path):
    print()
    print("=" * 110)
    print(f"FILE: {file_path}")
    print("=" * 110)

    if not file_path.exists():
        print("ERROR: file does not exist.")
        return

    try:
        data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        print(f"ERROR while loading MAT file: {exc}")
        return

    keys = [k for k in data if not k.startswith("__")]

    print(f"Top-level variables ({len(keys)}):")
    for key in keys:
        print(f"  - {key}")

    for key in keys:
        print("-" * 110)
        describe(data[key], 0, key, 0)

def main():
    print()
    print("=" * 110)
    print("PADERBORN MAT FILE STRUCTURE INSPECTION")
    print("=" * 110)

    for file_path in FILES_TO_INSPECT:
        inspect_file(file_path)

    print()
    print("=" * 110)
    print("PADERBORN INSPECTION COMPLETE")
    print("=" * 110)

if __name__ == "__main__":
    main()
