from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
B01_DIR = PROJECT_ROOT / "data" / "raw" / "paderborn_run_to_failure" / "B01"
VIBRATION_DIR = B01_DIR / "vibrationData"

def human_size(n):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"

def main():
    print()
    print("=" * 120)
    print("PADERBORN RUN-TO-FAILURE B01 STRUCTURE INSPECTION")
    print("=" * 120)

    if not B01_DIR.exists():
        raise FileNotFoundError(f"B01 folder not found:\n{B01_DIR}")

    print("\nTOP-LEVEL CONTENTS")
    for item in sorted(B01_DIR.iterdir()):
        if item.is_dir():
            print(f"[DIR ] {item.name}")
        else:
            print(f"[FILE] {item.name:<40} {human_size(item.stat().st_size)}")

    if not VIBRATION_DIR.exists():
        raise FileNotFoundError(f"vibrationData folder not found:\n{VIBRATION_DIR}")

    files = sorted(p for p in VIBRATION_DIR.rglob("*") if p.is_file())
    dirs = sorted(p for p in VIBRATION_DIR.rglob("*") if p.is_dir())

    print("\n" + "=" * 120)
    print("VIBRATION DATA SUMMARY")
    print("=" * 120)
    print(f"Total files    : {len(files)}")
    print(f"Subdirectories : {len(dirs)}")

    ext_counts = {}
    for p in files:
        ext = p.suffix.lower() if p.suffix else "<no extension>"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    print("\nFILE EXTENSIONS")
    for ext, count in sorted(ext_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"{ext:<15} {count}")

    print("\nFIRST 30 FILES")
    for i, p in enumerate(files[:30], start=1):
        rel = p.relative_to(VIBRATION_DIR)
        print(f"{i:>3}. {str(rel):<85} {human_size(p.stat().st_size)}")

    if len(files) > 30:
        print(f"... and {len(files) - 30} more files")

    if files:
        sizes = [p.stat().st_size for p in files]
        print("\nFILE SIZE STATISTICS")
        print(f"Smallest file : {human_size(min(sizes))}")
        print(f"Largest file  : {human_size(max(sizes))}")
        print(f"Average size  : {human_size(sum(sizes) / len(sizes))}")
        print(f"Total size    : {human_size(sum(sizes))}")

    print("\n" + "=" * 120)
    print("AUXILIARY FILES")
    print("=" * 120)

    for name in [
        "B01_log.pdf",
        "B01_meanTemperatures.csv",
        "B01_operatingConditions.csv",
    ]:
        p = B01_DIR / name
        if p.exists():
            print(f"FOUND   {name:<35} {human_size(p.stat().st_size)}")
        else:
            print(f"MISSING {name}")

    print("\n" + "=" * 120)
    print("INSPECTION COMPLETE")
    print("=" * 120)

if __name__ == "__main__":
    main()
