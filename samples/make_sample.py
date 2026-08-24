"""Generate a prescription image so the pipeline can be tested without touching
real patient data.

    python samples/make_sample.py                        # fully printed
    python samples/make_sample.py --handwriting rxhandbd # printed form, real
                                                         # handwritten drug names

The second form is the realistic case: a printed clinic letterhead with the
prescription written by hand underneath. It pastes real word images from
RxHandBD (see benchmark_handwriting.py for the download) and prints the ground
truth, so you can check what the pipeline recovered against what was written.

Uses cv2 only - no new dependency, and no PHI ever enters the repo.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

LINES = [
    ("CITY MULTISPECIALITY CLINIC", 1.0),
    ("Patient Name : Ramesh Kumar", 0.75),
    ("Age : 43     Sex : M", 0.75),
    ("UHID : MH-2291/25     Date : 14/07/2025", 0.7),
    ("Consultant : Dr. Anitha Rao", 0.7),
    ("BP : 138/86   Pulse : 92   SpO2 : 97%", 0.7),
    ("Temp : 98.6   Wt : 71.5 kg", 0.7),
    ("Diagnosis : Acute bronchitis with allergic rhinitis", 0.7),
    ("Rx", 0.85),
    ("Tab Azithromycin 500mg 1-0-0 x 3 days", 0.7),
    ("Cap Amoxicillin 250 mg 1-0-1 x 5 days", 0.7),
    ("Syrup Ambroxol 10ml tds", 0.7),
    ("Advice : steam inhalation, review after 5 days", 0.7),
]


def paste_handwriting(img, dataset, y_start, count=3):
    """Paste real handwritten drug words onto the printed form. Returns their labels."""
    import csv
    import random

    rows = list(csv.DictReader((dataset / "Test_Labels.csv").open()))
    random.seed(7)
    chosen = random.sample(rows, count)
    truths = []
    y = y_start
    for i, row in enumerate(chosen):
        word = cv2.imread(str(dataset / "Test_Set" / row["Images"]))
        if word is None:
            continue
        # The crops carry their own grey paper tone; stretch each to white-backed
        # ink so it sits on the printed form instead of showing as a pasted tile.
        gray = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lo, hi = gray.min(), max(gray.max(), gray.min() + 1)
        gray = np.clip((gray - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        word = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        scale = 74 / word.shape[0]
        word = cv2.resize(word, (int(word.shape[1] * scale), 74),
                          interpolation=cv2.INTER_CUBIC)
        cv2.putText(img, f"{i + 1}.", (60, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (25, 25, 25), 2, cv2.LINE_AA)
        h, w = word.shape[:2]
        img[y:y + h, 110:110 + w] = word            # the handwritten drug name
        cv2.putText(img, "1-0-1  x 5 days", (110 + w + 30, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 25, 25), 2, cv2.LINE_AA)
        truths.append(row["Text"].strip())
        y += 100
    return truths


def main(out=Path(__file__).parent / "sample_report.png"):
    step = lambda scale: int(52 * scale) + 22
    height = 70 + sum(step(s) for _, s in LINES) + 40   # size the page to the text, not vice versa
    img = np.full((height, 1100, 3), 250, np.uint8)
    y = 70
    for text, scale in LINES:
        cv2.putText(img, text, (60, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (25, 25, 25), 2,
                    cv2.LINE_AA)
        y += step(scale)
    # A little sensor-grade grain, so preprocessing has something real to remove.
    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out), img)
    print(f"wrote {out}")


def main_mixed(dataset, out):
    """Printed letterhead and patient details, handwritten prescription."""
    printed = [line for line in LINES
               if not line[0].startswith(("Tab ", "Cap ", "Syrup ", "Advice"))]
    advice = next(line for line in LINES if line[0].startswith("Advice"))
    step = lambda scale: int(52 * scale) + 22
    height = 70 + sum(step(s) for _, s in printed) + 420
    img = np.full((height, 1100, 3), 250, np.uint8)
    y = 70
    rx_y = None
    for text, scale in printed:
        cv2.putText(img, text, (60, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (25, 25, 25), 2,
                    cv2.LINE_AA)
        if text == "Rx":
            rx_y = y + 20
        y += step(scale)
    truths = paste_handwriting(img, dataset, rx_y)
    cv2.putText(img, advice[0], (60, rx_y + 100 * len(truths) + 60),
                cv2.FONT_HERSHEY_SIMPLEX, advice[1], (25, 25, 25), 2, cv2.LINE_AA)
    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out), img)
    print(f"wrote {out}")
    print("handwritten drug names on this page:", ", ".join(truths))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", type=Path,
                    default=Path(__file__).parent / "sample_report.png")
    ap.add_argument("--handwriting", type=Path,
                    help="unzipped RxHandBD folder; writes a printed form with real "
                         "handwritten drug names on it")
    args = ap.parse_args()
    if args.handwriting:
        out = args.out
        if out.name == "sample_report.png":
            out = out.with_name("sample_mixed.png")
        main_mixed(args.handwriting, out)
    else:
        main(args.out)
