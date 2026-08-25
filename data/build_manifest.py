import os
import csv
from pathlib import Path
Raw_DIR = Path("data/raw/CASIA2")
AU_DIR = Raw_DIR / "Au"
TP_DIR = Raw_DIR / "Tp"
OUTPUT_CSV = Path("data/manifest.csv")

VALID_EXTENSIONS = {".jpg",".jpeg",".png",".tif",".tiff",".bmp"}


def collect_images(folder: Path,label: int):

    records =[]
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS:
            records.append((str(path),label))
    return records

def main():
    if not AU_DIR.exists() or not TP_DIR.exists():
        raise FileNotFoundError(
            f"Expected folders not found. Checked: {AU_DIR} and {TP_DIR}. "
            "Double check your data/raw structure."
        )
        
    authentic_records = collect_images(AU_DIR,label =0)
    tampered_records = collect_images(TP_DIR, label =1)

    all_records = authentic_records + tampered_records

    with open(OUTPUT_CSV, "w",newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath","label"])
        writer.writerows(all_records)

    print(f"Authentic images found: {len(authentic_records)}")
    print(f"Tampered images found: {len(tampered_records)}")
    print(f"Total: {len(all_records)}")
    print(f"Manifest written to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
