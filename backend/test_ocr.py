"""One runnable check on the only non-trivial pure logic: the field parser.

    python3 backend/test_ocr.py

No pytest, no fixtures, no vision stack needed - it fails loudly if a regex rots.
"""
from ocr import group_lines, looks_like_text, parse_fields, snap_to_formulary

SAMPLE = [
    "CITY MULTISPECIALITY CLINIC",
    "Patient Name : Ramesh Kumar   Age 43",
    "Sex: M     UHID: MH-2291/25",
    "Date: 14/07/2025",
    "Consultant: Dr. Anitha Rao",
    "BP: 138/86 mmHg   Pulse 92   SpO2 97%   Temp 98.6   Wt 71.5 kg",
    "Diagnosis: Acute bronchitis with allergic rhinitis",
    "Rx",
    "Tab Azithromycin 500mg 1-0-0 x 3 days",
    "Cap Amoxicillin 250 mg 1-0-1 x 5 days",
    "Syrup Ambroxol 10ml tds",
    "Advice: steam inhalation, review after 5 days",
]


def main():
    r = parse_fields(SAMPLE)

    assert r["patient_name"] == "Ramesh Kumar", r["patient_name"]
    assert r["age"] == "43", r["age"]
    assert r["sex"] == "Male", r["sex"]
    assert r["patient_id"] == "MH-2291/25", r["patient_id"]
    assert r["visit_date"] == "14/07/2025", r["visit_date"]
    assert "Anitha Rao" in r["doctor"], r["doctor"]
    assert r["diagnosis"].startswith("Acute bronchitis"), r["diagnosis"]
    assert r["advice"].startswith("steam inhalation"), r["advice"]

    assert r["vitals"] == {"bp": "138/86", "pulse": "92", "spo2": "97",
                           "temp": "98.6", "weight": "71.5"}, r["vitals"]

    names = [m["name"] for m in r["medications"]]
    assert names == ["Azithromycin", "Amoxicillin", "Ambroxol"], names
    assert r["medications"][0]["dose"].replace(" ", "") == "500mg"
    assert r["medications"][0]["frequency"] == "1-0-0"
    assert r["medications"][0]["duration"] == "3 days"
    assert r["medications"][2]["frequency"].lower() == "tds"

    # A blank page must not crash or invent data.
    empty = parse_fields([])
    assert empty["patient_name"] == "" and empty["medications"] == []

    # (text, confidence) tuples straight from run_ocr() work too.
    assert parse_fields([("Age: 7", 0.9)])["age"] == "7"

    check_real_ocr_output()
    check_handwriting_support()
    checked_deskew = check_deskew()
    print(f"test_ocr: all assertions passed{'' if checked_deskew else ' (deskew skipped: no cv2)'}")


def check_handwriting_support():
    """The handwriting path: formulary correction, routing, and numbered scripts."""
    # Corrects a near-miss to a real drug...
    assert snap_to_formulary("Paracetmol").lower() == "paracetamol"
    assert snap_to_formulary("Amoxicilin").lower() == "amoxicillin"
    # ...but never invents one. A wrong drug name is worse than an uncorrected read.
    assert snap_to_formulary("xyzzyplugh") == "xyzzyplugh"      # nothing close
    assert snap_to_formulary("mexciture") == "mexciture"        # too far to trust
    assert snap_to_formulary("Nex") == "Nex"                    # too short to judge
    assert snap_to_formulary("Callbor") == "Callbor"            # ambiguous: several tie

    # Routing: printed labels and measurements must stay with EasyOCR, so that
    # enabling handwriting support cannot degrade a printed report.
    for printed in ("Patient Name :", "BP : 138/86", "Diagnosis", "138/86", "97%",
                    "Tab Azithromycin", "Age", "Temp : 98.6"):
        assert looks_like_text(printed), printed
    for scrawl in ("Ntnww)", "(us )", "Jxk", ""):
        assert not looks_like_text(scrawl), scrawl

    # A handwritten script is a numbered list with no "Tab"/"Cap" prefix.
    script = parse_fields(["1. Neuroxen 1-0-1 x 5 days",
                           "2. Calboral 500mg 1-0-0 x 3 days"])
    assert [m["name"].lower() for m in script["medications"]] == ["neuroxen", "calboral"]
    assert script["medications"][1]["dose"].replace(" ", "") == "500mg"
    # ...but a numbered list without dosing is prose, not a prescription.
    assert parse_fields(["1. Patient reports fever", "2. Advised rest"])["medications"] == []

    # A corrected name always carries the original read with it.
    corrected = parse_fields(["1. Paracetmol 1-0-1 x 3 days"])["medications"][0]
    assert corrected["name"].lower() == "paracetamol"
    assert corrected["as_read"] == "Paracetmol"


def check_deskew():
    """A tilted page must come back straight, or group_lines() merges the wrong rows.

    Skipped when cv2 is absent - the parser checks above are the ones that must
    always run. Regression guard: the first version thresholded at "> 0" (selecting
    the whole page, not the ink) and fed minAreaRect (y, x) points, so it silently
    returned ~0 for every angle and every skewed scan parsed as garbage.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return False

    from ocr import deskew, estimate_skew

    page = np.full((600, 900), 255, np.uint8)
    for i, line in enumerate(["Patient Name : Ramesh Kumar", "Age : 43   Sex : M",
                              "BP : 138/86   Pulse : 92", "Diagnosis : bronchitis"]):
        cv2.putText(page, line, (60, 120 + i * 110), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 3,
                    cv2.LINE_AA)

    for applied in (-11, -5, 0, 3, 7):
        m = cv2.getRotationMatrix2D((450, 300), applied, 1.0)
        tilted = cv2.warpAffine(page, m, (900, 600), borderValue=255)
        estimated = estimate_skew(tilted)
        assert abs(estimated - applied) < 1.0, f"{applied}deg read as {estimated:.2f}"
        assert abs(estimate_skew(deskew(tilted))) < 1.0, f"{applied}deg not corrected"
    return True


def box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def check_real_ocr_output():
    """Regression cases taken from an actual EasyOCR run on samples/sample_report.png.

    The detector emits fragments, not lines - "Age" and ":43" arrive as separate
    boxes, and "BP :" arrives *after* the "138/86" it belongs to. Everything here
    failed before group_lines() existed.
    """
    fragments = [
        (box(60, 180, 130, 215), "Age", 1.00),
        (box(140, 182, 200, 214), ":43", 1.00),
        (box(240, 181, 330, 215), "Sex : M", 0.93),
        (box(300, 358, 420, 392), "138/86", 0.60),      # value box sorts before its label
        (box(60, 360, 130, 390), "BP :", 0.96),
        (box(600, 359, 720, 391), "Spoz : 97%", 0.73),  # SpO2 misread as Spoz
        (box(60, 474, 190, 508), "Diagnosis", 0.67),    # colon lost by the OCR
        (box(200, 475, 700, 507), "Acute bronchitis with allergic rhinitis", 0.92),
        (box(60, 598, 120, 632), "Cap", 1.00),
        (box(130, 599, 600, 631), "Amoxicillin 250 mg 1-0-1 x 5 days", 0.63),
        (box(60, 715, 400, 749), "Syrup Ambroxol 1Oml tds", 0.98),  # "10ml" read as "1Oml"
    ]

    lines = group_lines(fragments)
    texts = [t for t, _ in lines]
    assert texts[0] == "Age :43 Sex : M", texts[0]
    assert texts[1] == "BP : 138/86 Spoz : 97%", texts[1]   # reordered left-to-right
    assert texts[2].startswith("Diagnosis Acute bronchitis"), texts[2]

    r = parse_fields(lines)
    assert r["age"] == "43", r["age"]
    assert r["sex"] == "Male", r["sex"]
    assert r["diagnosis"] == "Acute bronchitis with allergic rhinitis", r["diagnosis"]
    assert r["vitals"]["bp"] == "138/86", r["vitals"]
    assert r["vitals"]["spo2"] == "97", r["vitals"]

    meds = {m["name"]: m for m in r["medications"]}
    assert set(meds) == {"Amoxicillin", "Ambroxol"}, list(meds)
    assert meds["Amoxicillin"]["frequency"] == "1-0-1", meds["Amoxicillin"]
    assert meds["Amoxicillin"]["duration"] == "5 days", meds["Amoxicillin"]
    assert meds["Ambroxol"]["dose"].replace(" ", "") == "10ml", meds["Ambroxol"]

    # Two medications written on one line must both be picked up.
    two = parse_fields(["Tab Pantoprazole 40mg od   Tab Ondansetron 4mg sos"])
    assert [m["name"] for m in two["medications"]] == ["Pantoprazole", "Ondansetron"], two


if __name__ == "__main__":
    main()
