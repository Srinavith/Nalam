"""NALAM OCR pipeline: OpenCV preprocessing -> EasyOCR (PyTorch) -> structured fields.

Two halves that are deliberately independent:
  read_image()/run_ocr()  - pixels to raw text lines (needs the heavy deps)
  parse_fields()          - text lines to a clinical record dict (pure python, testable)

Keeping parse_fields() dependency-free is what makes test_ocr.py runnable without
downloading a 100MB model.
"""
import difflib
import functools
import os
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent

# ponytail: cv2/numpy are imported inside the pixel functions, not at module scope,
# so parse_fields() (and its test) run on a bare python with no vision stack.

# --- calibration knobs -------------------------------------------------------
# Scans are never the ideal on paper: phone photos are skewed, fax printouts are
# blown out, handwriting is faint. Tune these per clinic rather than editing code.
MIN_CONFIDENCE = float(os.getenv("NALAM_MIN_CONFIDENCE", "0.35"))
# Below this EasyOCR confidence a box is treated as handwriting and re-read by TrOCR.
# EasyOCR is reliable on print and near-useless on a doctor's hand; TrOCR is the
# reverse. Routing per box, rather than per document, handles the usual real slip:
# a printed letterhead with a handwritten prescription under it.
TROCR_BELOW = float(os.getenv("NALAM_TROCR_BELOW", "0.75"))
# Boxes this weak are noise, not text, and are dropped before either recogniser.
MIN_DETECT = float(os.getenv("NALAM_MIN_DETECT", "0.03"))
HANDWRITING = os.getenv("NALAM_HANDWRITING", "auto")   # auto | always | never
DESKEW_LIMIT_DEG = float(os.getenv("NALAM_DESKEW_LIMIT", "15"))
UPSCALE_TO_WIDTH = int(os.getenv("NALAM_UPSCALE_WIDTH", "1600"))
MAX_UPSCALE = float(os.getenv("NALAM_MAX_UPSCALE", "4"))
LANGS = os.getenv("NALAM_OCR_LANGS", "en").split(",")

_reader = None
_trocr = {}


def get_reader():
    """Lazy singleton. EasyOCR loads ~100MB of PyTorch weights; do it once, on demand."""
    global _reader
    if _reader is None:
        import easyocr
        import torch

        gpu = torch.cuda.is_available()
        # ponytail: EasyOCR's gpu flag is CUDA-only; Apple MPS falls back to CPU.
        _reader = easyocr.Reader(LANGS, gpu=gpu)
    return _reader


def estimate_skew(gray):
    """Return the page's rotation in degrees (positive = counter-clockwise).

    Otsu picks the ink out of the paper - thresholding at "> 0" would select the
    whole page, since inverted white paper is 5, not 0, and the resulting box
    measures the scan border instead of the text. findNonZero gives (x, y) points,
    which is the order minAreaRect expects; np.where would hand it (y, x) and
    silently transpose the angle.
    """
    import cv2

    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Bleed the glyphs of a line into one blob so the box follows the text baseline.
    ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)), iterations=2)
    points = cv2.findNonZero(ink)
    if points is None or len(points) < 50:
        return 0.0
    angle = cv2.minAreaRect(points)[-1]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    return -angle


def _default_trocr():
    # Measured: a trocr-small checkpoint fine-tuned on RxHandBD still trails
    # zero-shot trocr-base on the held-out test split (46% CER vs 42%) - not
    # enough capacity in the small model to make up for lower generalisation.
    # trocr-base is the better default until a fine-tune of trocr-base itself
    # is run (see backend/finetune.py --model microsoft/trocr-base-handwritten).
    return os.getenv("NALAM_TROCR_MODEL", "microsoft/trocr-base-handwritten")


# The pad and the prescription pipeline want different recognisers. A model
# fine-tuned on prescription words gets sharply better at drug names and worse at
# everything else - it read a hand-drawn "HI" as "t". So the pad keeps the general
# handwriting model unless told otherwise, and the page pipeline uses the
# fine-tuned one when a checkpoint exists.
GENERAL_TROCR = os.getenv("NALAM_PAD_MODEL", "microsoft/trocr-base-handwritten")


def get_trocr(name=None):
    """Lazy, per-model cache of the handwriting recogniser. (processor, model, device)."""
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    name = name or os.getenv("NALAM_TROCR_MODEL") or _default_trocr()
    if name not in _trocr:
        processor = TrOCRProcessor.from_pretrained(name)
        model = VisionEncoderDecoderModel.from_pretrained(name).eval()
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        _trocr[name] = (processor, model, device)
    return _trocr[name]


def _as_pil(crops):
    import numpy as np
    from PIL import Image

    images = []
    for crop in crops:
        arr = np.asarray(crop)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        images.append(Image.fromarray(arr.astype("uint8")).convert("RGB"))
    return images


def read_handwriting(crops, batch_size=8, model_name=None):
    """Run TrOCR over a list of BGR/grayscale crops. Returns a list of strings."""
    import numpy as np
    import torch
    from PIL import Image

    if not crops:
        return []
    processor, model, device = get_trocr(model_name)
    images = _as_pil(crops)

    out = []
    for i in range(0, len(images), batch_size):
        pixels = processor(images=images[i:i + batch_size],
                           return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = model.generate(pixels, max_new_tokens=32)
        out += processor.batch_decode(ids, skip_special_tokens=True)
    return [clean_handwriting(t) for t in out]


TRAILING_JUNK = re.compile(r"[\s.\-_,;:|]+$")


def clean_handwriting(text):
    """Strip the artefacts TrOCR reliably adds: a trailing ' .' and split 'rn' for 'm'."""
    text = text.strip()
    text = TRAILING_JUNK.sub("", text)
    text = re.sub(r"rn", "m", text)      # TrOCR habitually splits a written "m" into "rn"
    text = re.sub(r"[^\w\s%/.\-]", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _to_gray(image):
    import cv2

    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def is_blank(image, ink_fraction=0.0008):
    """True when there is essentially nothing drawn - an empty pad, not a failed read."""
    import cv2
    import numpy as np

    gray = _to_gray(image)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if np.mean(gray) < 127:                       # light ink on a dark ground
        ink = cv2.bitwise_not(ink)
    return (np.count_nonzero(ink) / ink.size) < ink_fraction


def prepare_scribble(image, target_height=96, pad_ratio=0.18):
    """Tidy a pad drawing for the recogniser: crop to the ink, pad, normalise size.

    TrOCR is trained on tight crops of a written line. A raw canvas is mostly empty
    white with a small scrawl somewhere in it, which reads far worse than the same
    strokes cropped to their bounding box.
    """
    import cv2
    import numpy as np

    gray = _to_gray(image)
    if np.mean(gray) < 127:                       # white-on-black pad: flip it
        gray = cv2.bitwise_not(gray)

    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    points = cv2.findNonZero(ink)
    if points is not None:
        x, y, w, h = cv2.boundingRect(points)
        pad = max(8, int(pad_ratio * h))
        H, W = gray.shape
        gray = gray[max(0, y - pad):min(H, y + h + pad),
                    max(0, x - pad):min(W, x + w + pad)]

    if gray.size and gray.shape[0] < target_height:
        scale = target_height / gray.shape[0]
        gray = cv2.resize(gray, (max(1, int(gray.shape[1] * scale)), target_height),
                          interpolation=cv2.INTER_CUBIC)
    return gray


def deskew(gray):
    """Rotate the page upright using the minimum-area box of the dark pixels."""
    import cv2

    angle = estimate_skew(gray)
    if abs(angle) < 0.1 or abs(angle) > DESKEW_LIMIT_DEG:
        return gray  # a wild angle means the box latched onto noise, not text
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE, borderValue=255)


def preprocess(image):
    """Grayscale -> upscale -> denoise -> CLAHE -> deskew. Returns a uint8 image."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    if w < UPSCALE_TO_WIDTH:
        # Capped: hitting a fixed target width blows a small crop up 12x, which is
        # slow and just interpolates blur into the strokes. Enlarge, don't inflate.
        scale = min(UPSCALE_TO_WIDTH / w, MAX_UPSCALE)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, h=7)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return deskew(gray)


LINE_TOLERANCE = float(os.getenv("NALAM_LINE_TOLERANCE", "0.6"))


def group_boxes(result, tolerance=None):
    """Merge OCR boxes that sit on the same visual line.

    The detector emits boxes, not lines: "Age" and ":43" come back separately, as
    do "BP :" and "138/86". Every field pattern below expects a label and its value
    in the same string, so rebuilding the physical lines here - once, for every
    caller - is what makes the parser work on real output instead of tidy fixtures.

    `result` is EasyOCR's [(bbox, text, conf), ...]. Returns a list of lines,
    top-to-bottom, each a list of box dicts ordered left-to-right.
    """
    tolerance = LINE_TOLERANCE if tolerance is None else tolerance
    boxes = []
    for bbox, text, conf in result:
        if not text.strip():
            continue
        ys = [pt[1] for pt in bbox]
        xs = [pt[0] for pt in bbox]
        boxes.append({"cy": (min(ys) + max(ys)) / 2, "height": max(1.0, max(ys) - min(ys)),
                      "left": min(xs), "right": max(xs), "top": min(ys), "bottom": max(ys),
                      "text": text.strip(), "conf": float(conf)})
    boxes.sort(key=lambda b: b["cy"])

    lines = []
    for box in boxes:
        # Same line if the vertical centres are within a fraction of the text height.
        if lines and abs(box["cy"] - lines[-1]["cy"]) <= tolerance * box["height"]:
            lines[-1]["items"].append(box)
            lines[-1]["cy"] = sum(i["cy"] for i in lines[-1]["items"]) / len(lines[-1]["items"])
        else:
            lines.append({"cy": box["cy"], "items": [box]})

    return [sorted(line["items"], key=lambda b: b["left"]) for line in lines]


def group_lines(result, tolerance=None):
    """Text-only view of group_boxes: [(line_text, mean_confidence), ...]."""
    return [(" ".join(i["text"] for i in line),
             sum(i["conf"] for i in line) / len(line))
            for line in group_boxes(result, tolerance)]


# Words that appear on essentially every clinical form. If EasyOCR read one of
# these, it read a printed label correctly and there is nothing for TrOCR to fix.
LABEL_WORDS = (
    "name", "patient", "age", "sex", "gender", "uhid", "mrn", "reg", "date", "doctor",
    "dr", "consultant", "physician", "diagnosis", "impression", "complaint", "advice",
    "plan", "follow", "instruction", "clinic", "hospital", "centre", "center",
    "bp", "blood", "pressure", "pulse", "temp", "spo", "weight", "tab", "tablet",
    "cap", "capsule", "syrup", "syr", "inj", "injection", "drop", "ointment",
    "mg", "ml", "kg", "day", "week", "month",
)


def looks_like_text(text):
    """Is this EasyOCR read plausibly correct, or is it noise from handwriting?

    Confidence alone is the wrong signal: EasyOCR reports 0.6-0.7 on perfectly good
    printed labels, so routing on confidence sends clean text to TrOCR and makes the
    printed path worse. What actually separates the two cases is whether the string
    means anything - a form label, a measurement, or a real drug name.
    """
    stripped = text.strip().lower()
    if not stripped:
        return False
    if any(word in stripped for word in LABEL_WORDS):
        return True
    alphanumeric = [c for c in stripped if c.isalnum()]
    if alphanumeric and sum(c.isdigit() for c in alphanumeric) / len(alphanumeric) >= 0.5:
        return True                                    # "138/86", ":43", "97%"
    if any(key == stripped for key, _ in load_formulary()):
        return True
    return snap_to_formulary(stripped).lower() != stripped


def crop_union(image, boxes, pad=None):
    """Cut the bounding box of a run of adjacent boxes out of the preprocessed page.

    Padding scales with text height: detectors clip ascenders and descenders, and a
    recogniser handed a clipped word reads a different word.
    """
    xs = [boxes[0]["left"], boxes[-1]["right"]] + [b["left"] for b in boxes] + \
         [b["right"] for b in boxes]
    ys = [b["top"] for b in boxes] + [b["bottom"] for b in boxes]
    height = max(1.0, max(ys) - min(ys))
    pad = int(0.25 * height) if pad is None else pad
    xs = [int(v) for v in xs]
    ys = [int(v) for v in ys]
    h, w = image.shape[:2]
    x0, x1 = max(0, min(xs) - pad), min(w, max(xs) + pad)
    y0, y1 = max(0, min(ys) - pad), min(h, max(ys) + pad)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return image[y0:y1, x0:x1]


def run_ocr(image):
    """Detect text, then read each part of a line with whichever recogniser suits it.

    EasyOCR does detection and recognition; only its recognition collapses on
    handwriting (~73% character error on real prescriptions). So its detector is
    kept, and the stretches it reads as nonsense are cropped and re-read by TrOCR.

    The unit of re-reading is a *run of adjacent boxes*, not one box. A detector
    tuned on print carves a handwritten word into fragments ("Neuroxen" -> "n"),
    and handing TrOCR those fragments just relays the damage; cropping the whole
    handwritten stretch of the line gives it the word back. This is what lets one
    page carry a printed letterhead and a handwritten prescription.

    Returns [(line_text, confidence), ...] in reading order.
    """
    page = preprocess(image)
    detected = get_reader().readtext(page, detail=1, paragraph=False)
    detected = [d for d in detected if d[2] >= MIN_DETECT]
    if not detected:
        return []

    lines = group_boxes(detected)
    if HANDWRITING == "never":
        return [(" ".join(i["text"] for i in line),
                 sum(i["conf"] for i in line) / len(line))
                for line in ([b for b in ln if b["conf"] >= MIN_CONFIDENCE] for ln in lines)
                if line]

    # Split every line into alternating printed / handwritten runs.
    segments, crops = [], []
    for line in lines:
        run = []
        for box in line:
            handwritten = (HANDWRITING == "always"
                           or (box["conf"] < TROCR_BELOW and not looks_like_text(box["text"])))
            if run and run[-1][0] == handwritten:
                run[-1][1].append(box)
            else:
                run.append([handwritten, [box]])
        segments.append(run)
        for handwritten, boxes in run:
            if handwritten:
                crop = crop_union(page, boxes)
                # Same preparation the scribble pad uses: crop to the ink and
                # normalise its height. TrOCR is trained on tight, consistently
                # sized lines, and reads a loose or small crop noticeably worse.
                crops.append(prepare_scribble(crop) if crop is not None else None)

    readings = iter(read_handwriting([c for c in crops if c is not None]))
    filled = [next(readings) if c is not None else "" for c in crops]

    out, cursor = [], 0
    for run in segments:
        parts, confidences = [], []
        for handwritten, boxes in run:
            if handwritten:
                text = filled[cursor]
                cursor += 1
                if text:
                    parts.append(text)
                    # TrOCR gives no per-box score; report the routing floor so the
                    # reviewer can see this line was a guess, not a confident read.
                    confidences.append(TROCR_BELOW * 0.5)
            else:
                kept = [b for b in boxes if b["conf"] >= MIN_CONFIDENCE]
                if kept:
                    parts.append(" ".join(b["text"] for b in kept))
                    confidences += [b["conf"] for b in kept]
        if parts:
            out.append((" ".join(parts), sum(confidences) / len(confidences)))
    return out


# --- text -> structured record ----------------------------------------------
LABELLED = {
    "patient_name": r"(?:patient(?:'?s)?\s*name|patient|name)\s*[:\-]\s*(.+)",
    "age": r"\bage\s*[:\-]?\s*(\d{1,3})",
    "sex": r"\b(?:sex|gender)\s*[:\-]?\s*(male|female|m|f|other)\b",
    "patient_id": r"\b(?:uhid|mrn|patient\s*id|reg(?:istration)?\s*no)\s*[:\-]?\s*([A-Za-z0-9\-/]+)",
    "visit_date": r"\b(?:date|dated|visit\s*date|doa)\s*[:\-]?\s*(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{2,4})",
    "doctor": r"\b(?:dr\.?|doctor|physician|consultant)\s*[:\-]?\s*([A-Za-z][A-Za-z.\s]{2,40})",
    # The colon is optional here: OCR routinely loses it, and these two labels are
    # distinctive enough that a bare "Diagnosis Acute bronchitis" is not ambiguous.
    "diagnosis": r"\b(?:diagnosis|impression|provisional\s*diagnosis|complaint)\s*[:\-]?\s+(.{3,})",
    "advice": r"\b(?:advice|plan|follow\s*up|instructions)\s*[:\-]?\s+(.{3,})",
}

VITALS = {
    "bp": r"\b(?:b\.?p\.?|blood\s*pressure)\s*[:\-]?\s*(\d{2,3}\s*/\s*\d{2,3})",
    "pulse": r"\b(?:pulse|hr|heart\s*rate)\s*[:\-]?\s*(\d{2,3})",
    "temp": r"\b(?:temp(?:erature)?)\s*[:\-]?\s*(\d{2,3}(?:\.\d)?)",
    "spo2": r"\b(?:sp[o0][2zs]|sa[o0]2|o2\s*sat)\s*[:\-]?\s*(\d{2,3})\s*%?",
    "weight": r"\b(?:weight|wt)\s*[:\-]?\s*(\d{1,3}(?:\.\d)?)\s*(?:kg)?",
}

# "Tab Amoxicillin 500mg 1-0-1 x 5 days" and friends - and the handwritten form,
# "1. Neuroxen 1-0-1 x 5 days", where the numbered list replaces the dosage form.
# A numbered match must carry a frequency or a duration (see parse_fields), or every
# numbered line on the page would look like a prescription.
MED_RE = re.compile(
    r"(?:\b(?P<form>tab(?:let)?|cap(?:sule)?|syr(?:up)?|inj(?:ection)?|oint(?:ment)?|drops?)\.?"
    r"|(?:^|\s)(?P<index>\d{1,2})[.)])\s+"
    # A name word must start with a letter, which is what keeps "500mg" / "10ml" out of it
    # while still allowing "Vitamin D3". The lookahead stops it swallowing a frequency.
    r"(?P<name>[A-Za-z][A-Za-z0-9\-]*"
    r"(?:\s+(?!od\b|bd\b|tds\b|qid\b|hs\b|sos\b|stat\b|x\b)[A-Za-z][A-Za-z0-9\-]*){0,3})\s*"
    r"(?P<dose>\d+(?:\.\d+)?\s*(?:mg|ml|mcg|g|iu)\b)?\s*"
    r"(?P<freq>\d\s*-\s*\d\s*-\s*\d|(?:od|bd|tds|qid|hs|sos|stat)\b)?\s*"
    r"(?:x?\s*(?P<duration>\d{1,2}\s*(?:days?|weeks?|months?)))?",
    re.I)

# OCR reads "10ml" as "1Oml" and "500mg" as "5O0mg" constantly. Only applied to a
# token that is already glued to a unit, so real words are never touched.
DIGIT_LOOKALIKES = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "S": "5"})
DOSE_TOKEN = re.compile(r"\b([0-9OolIS]{1,5})(\s*)(mg|ml|mcg|iu)\b", re.I)


def normalize_digits(line):
    return DOSE_TOKEN.sub(lambda m: m.group(1).translate(DIGIT_LOOKALIKES) + m.group(2) + m.group(3),
                          line)


MAX_SNAP_RATIO = float(os.getenv("NALAM_MAX_SNAP_RATIO", "0.34"))
MIN_SNAP_LENGTH = int(os.getenv("NALAM_MIN_SNAP_LENGTH", "5"))
FORMULARY_PATH = Path(os.getenv("NALAM_FORMULARY", BASE / "formulary.txt"))
# How close a read has to be to a formulary entry before it is snapped to it.
# Higher = fewer corrections but fewer wrong ones. 0 disables snapping entirely.
LEXICON_CUTOFF = float(os.getenv("NALAM_LEXICON_CUTOFF", "0.6"))


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@functools.lru_cache(maxsize=1)
def load_formulary():
    """The clinic's drug vocabulary, as [(lowercased, original), ...]."""
    if not FORMULARY_PATH.exists():
        return ()
    # Deduplicated on the lowercased key: a formulary listing both "Cefotil" and
    # "cefotil" would otherwise look like two equally-close matches and trip the
    # ambiguity guard in snap_to_formulary, refusing a correction that is certain.
    entries = {}
    for line in FORMULARY_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.setdefault(line.lower(), line)
    return tuple(entries.items())


def snap_to_formulary(word, cutoff=None):
    """Correct a garbled drug name to the nearest real one, or return it unchanged.

    A recogniser reading handwriting produces near-misses ("cofotil" for "Cefotil").
    Prescriptions are drawn from a known formulary, so the closest entry is almost
    always the intended drug - this is worth more accuracy than any model setting.
    Nothing is snapped silently: parse_fields keeps the original read alongside, and
    a doctor confirms every record before it is exported.
    """
    cutoff = LEXICON_CUTOFF if cutoff is None else cutoff
    entries = load_formulary()
    if not word or not entries or cutoff <= 0:
        return word

    lowered = word.lower()
    keys = [key for key, _ in entries]
    candidates = difflib.get_close_matches(lowered, keys, n=5, cutoff=cutoff)
    if not candidates:
        return word

    # difflib ranks by matching subsequences, which happily maps a long garbled read
    # onto a short unrelated entry ("mexciture" -> "meur"). Re-rank by edit distance
    # and refuse anything that would rewrite more than MAX_SNAP_RATIO of the word: a
    # wrong drug name is far worse than an uncorrected one.
    ranked = sorted(candidates, key=lambda c: (edit_distance(lowered, c),
                                               abs(len(c) - len(lowered))))
    best = ranked[0]
    distance = edit_distance(lowered, best)
    if distance > MAX_SNAP_RATIO * max(len(lowered), len(best)):
        return word
    # Ambiguous: two formulary entries are equally close, so there is no evidence
    # for either. Guessing a drug name from a coin flip is worse than not guessing.
    if len(ranked) > 1 and edit_distance(lowered, ranked[1]) == distance:
        return word
    if distance and len(lowered) < MIN_SNAP_LENGTH:
        return word          # short reads have too little signal to correct safely
    return dict(entries)[best]


SEX_MAP = {"m": "Male", "f": "Female", "male": "Male", "female": "Female", "other": "Other"}


def _clean(value):
    return re.sub(r"\s{2,}", " ", value).strip(" .,:;-|")


def parse_fields(lines):
    """Map raw OCR lines onto a clinical record.

    `lines` is a list of strings (or (text, conf) tuples straight from run_ocr).
    Returns {fields..., "medications": [...], "vitals": {...}, "raw_text": str}.
    Every value is a best guess for a human to correct in the dashboard - the
    doctor is the last mile, not the model.
    """
    lines = [l[0] if isinstance(l, (tuple, list)) else l for l in lines]
    text = "\n".join(lines)
    record = {k: "" for k in LABELLED}
    record["medications"] = []
    record["vitals"] = {}

    for line in lines:
        for key, pattern in LABELLED.items():
            if record[key]:
                continue
            m = re.search(pattern, line, re.I)
            if m:
                record[key] = _clean(m.group(1))
        for key, pattern in VITALS.items():
            if key in record["vitals"]:
                continue
            m = re.search(pattern, normalize_digits(line), re.I)
            if m:
                record["vitals"][key] = _clean(m.group(1)).replace(" ", "")

        for m in MED_RE.finditer(normalize_digits(line)):
            # A numbered line only counts as a prescription if it has dosing on it.
            if m.group("index") and not (m.group("freq") or m.group("duration")):
                continue
            if m.group("name"):
                as_read = _clean(m.group("name")).title()
                corrected = snap_to_formulary(as_read)
                entry = {
                    "name": corrected,
                    "dose": _clean(m.group("dose") or ""),
                    "frequency": _clean(m.group("freq") or ""),
                    "duration": _clean(m.group("duration") or ""),
                }
                if corrected.lower() != as_read.lower():
                    entry["as_read"] = as_read     # never hide what the machine saw
                record["medications"].append(entry)

    if record["sex"]:
        record["sex"] = SEX_MAP.get(record["sex"].lower(), record["sex"])
    if record["patient_name"]:
        # OCR loves trailing junk from the next column ("Ramesh K Age 43").
        record["patient_name"] = _clean(
            re.split(r"\b(?:age|sex|gender|uhid|mrn|date)\b", record["patient_name"], flags=re.I)[0])
    record["raw_text"] = text
    return record


def digitize(image):
    """Full pipeline: image array -> record dict with a mean-confidence score."""
    pairs = run_ocr(image)
    record = parse_fields(pairs)
    record["confidence"] = round(sum(c for _, c in pairs) / len(pairs), 3) if pairs else 0.0
    return record


def read_image(path_or_bytes):
    import cv2
    import numpy as np

    if isinstance(path_or_bytes, (bytes, bytearray)):
        buf = np.frombuffer(path_or_bytes, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(path_or_bytes), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unreadable image (corrupt file or unsupported format)")
    return img
