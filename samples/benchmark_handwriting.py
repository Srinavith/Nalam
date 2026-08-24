"""Score the recognisers on real handwritten prescriptions.

The synthetic sample in this folder is printed text, which flatters the pipeline.
This measures it on RxHandBD: cropped words cut from real medical prescriptions,
MIT-licensed, with ground-truth transcriptions.

    curl -L -o RxHandBD.zip https://zenodo.org/records/18478741/files/RxHandBD.zip
    unzip RxHandBD.zip -d rxhandbd
    python samples/benchmark_handwriting.py rxhandbd --engine all

Engines:
  easyocr   what the pipeline uses for printed text
  trocr     the handwriting recogniser, raw output
  pipeline  trocr + cleanup + formulary snapping - what actually ships

--formulary swaps the vocabulary. Quote the held-out number (a formulary built
from the train split only) unless you are describing a clinic that really does
have its own complete drug list.
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

import cv2  # noqa: E402

import ocr  # noqa: E402
from ocr import edit_distance  # noqa: E402


def score(name, truths, predictions, seconds, show=0):
    exact = sum(p.lower() == t.lower() for p, t in zip(predictions, truths))
    errors = sum(edit_distance(t.lower(), p.lower()) for p, t in zip(predictions, truths))
    characters = sum(len(t) for t in truths)
    print(f"{name:34} exact={exact / len(truths):6.1%}  CER={errors / characters:6.1%}"
          f"  {seconds:5.1f}s")
    for truth, prediction in list(zip(truths, predictions))[:show]:
        hit = "HIT " if truth.lower() == prediction.lower() else "    "
        print(f"    {hit}{truth:18} -> {prediction!r}")
    return errors / characters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path, help="unzipped RxHandBD folder")
    ap.add_argument("--engine", default="all", choices=["easyocr", "trocr", "pipeline", "all"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--formulary", type=Path, help="override the formulary file")
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    if args.formulary:
        ocr.FORMULARY_PATH = args.formulary
        ocr.load_formulary.cache_clear()

    labels = args.dataset / "Test_Labels.csv"
    if not labels.exists():
        sys.exit(f"{labels} not found - point me at the unzipped RxHandBD folder")
    rows = list(csv.DictReader(labels.open()))
    random.seed(0)
    rows = random.sample(rows, min(args.limit, len(rows)))

    images = [cv2.imread(str(args.dataset / "Test_Set" / r["Images"])) for r in rows]
    truths = [r["Text"].strip() for r in rows]
    print(f"{len(rows)} handwritten words, formulary={ocr.FORMULARY_PATH.name} "
          f"({len(ocr.load_formulary())} entries)\n")

    if args.engine in ("easyocr", "all"):
        reader = ocr.get_reader()
        started = time.time()
        predictions = [" ".join(reader.readtext(im, detail=0, paragraph=False)).strip()
                       for im in images]
        score("easyocr (printed-text engine)", truths, predictions, time.time() - started,
              args.show)

    if args.engine in ("trocr", "pipeline", "all"):
        started = time.time()
        raw = ocr.read_handwriting([ocr.prepare_scribble(im) for im in images])
        elapsed = time.time() - started
        if args.engine in ("trocr", "all"):
            score("trocr (handwriting, raw)", truths, raw, elapsed, args.show)
        if args.engine in ("pipeline", "all"):
            snapped = [ocr.snap_to_formulary(t) for t in raw]
            score("pipeline (trocr + formulary snap)", truths, snapped, elapsed, args.show)


if __name__ == "__main__":
    main()
