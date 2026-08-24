# NALAM - OCR-Based Medical Record Parsing

**AI-powered medical report digitization for doctors.** July – August 2025

NALAM turns a photo or scan of a medical report — printed, handwritten, or the
usual mix of both — into a structured, editable record: patient details, vitals,
diagnosis, and the prescription. A doctor corrects it in a browser dashboard, then
it exports as CSV for the clinic's existing systems.

**Stack:** Python · PyTorch · OpenCV · EasyOCR · TrOCR · Flask · React

---

## How it works

```
 scan/photo
     │
     ▼
 OpenCV preprocessing ──  grayscale → upscale → denoise → CLAHE → deskew
     │
     ▼
 EasyOCR detector ──────  text boxes, grouped into physical lines
     │
     ├──── printed run ──────────►  keep the EasyOCR text
     │                              (better on print, and ~40x faster)
     │
     └──── handwritten run ─────►  crop the whole run → TrOCR
                                    (routed by plausibility, not confidence)
     │
     ▼
 Field parser ──────────  labelled fields, vitals, medication lines
     │
     ▼
 Formulary snap ────────  garbled drug name → nearest real drug,
                          refused whenever the evidence is thin
     │
     ▼
 Flask API + SQLite ────  stored with the source image
     │
     ▼
 React dashboard ───────  doctor corrects the fields beside the scan;
                          a scribble pad reads single handwritten words
     │
     ▼
 CSV export ────────────  only what a doctor has signed off, if you want it so
```

### Why two recognisers

EasyOCR is a printed-text engine. On real handwritten prescriptions it scores
**73% character error** — it does not work, at all. TrOCR is trained on handwriting
and does far better there, but is slower and no better on print.

So neither runs on the whole page. The detector finds the lines, and each *run of
adjacent boxes* within a line goes to whichever engine suits it. The routing test
is deliberately not confidence: EasyOCR reports 0.6–0.7 on perfectly good printed
labels, and thresholding on that sent clean text to TrOCR and **cost the printed
path a field and two vitals**. `looks_like_text()` instead asks whether the string
means anything — a form label, a measurement, or a real drug name.

The unit is a run, not a box, because a detector tuned on print carves a
handwritten word into fragments (`"Neuroxen"` → `"n"`), and handing TrOCR the
fragments just relays the damage.

The model is never the last word. Every parsed value lands in an editable field
next to the original image, and nothing is marked *reviewed* until a human says so.

---

## Quick start

Requires **Python 3.9–3.13** (PyTorch wheels lag newer releases) and **Node 18+**.

```bash
git clone <this-repo-url>
cd NALAM

# 1. Backend
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python backend/app.py                                  # http://127.0.0.1:5001

# 2. Frontend (second terminal)
cd frontend
npm install
npm run dev                                            # http://localhost:5173
```

Open <http://localhost:5173>, drop in a report image, correct the fields, save.

**No scan handy?** Generate a synthetic one — no real patient data required:

```bash
python samples/make_sample.py        # writes samples/sample_report.png
```

The first upload downloads the EasyOCR model weights (~100 MB) and takes a
minute. Every upload after that is fast.

### Single-process deployment

```bash
cd frontend && npm run build && cd ..
gunicorn --chdir backend --bind 0.0.0.0:5001 app:app
```

Flask serves the built dashboard from `frontend/dist`, so there is one process
and one port in production.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness check |
| `POST` | `/api/reports` | Upload a report image (`multipart/form-data`, field `file`) → parsed record |
| `GET` | `/api/reports` | List reports, newest first (`?limit=`) |
| `GET` | `/api/reports/<id>` | One full record |
| `PATCH` | `/api/reports/<id>` | Doctor's corrections; send `{"reviewed": true}` to sign off |
| `DELETE` | `/api/reports/<id>` | Delete the record and its stored image |
| `GET` | `/api/reports/<id>/image` | The original uploaded scan |
| `GET` | `/api/export.csv` | CSV of all records (`?reviewed=1` for signed-off only) |

```bash
curl -F file=@samples/sample_report.png http://127.0.0.1:5001/api/reports
```

### Record shape

```json
{
  "id": "b7f2…", "filename": "opd_scan.png", "confidence": 0.82, "reviewed": false,
  "patient_name": "Ramesh Kumar", "age": "43", "sex": "Male",
  "patient_id": "MH-2291/25", "visit_date": "14/07/2025", "doctor": "Anitha Rao",
  "diagnosis": "Acute bronchitis with allergic rhinitis",
  "advice": "steam inhalation, review after 5 days",
  "vitals": { "bp": "138/86", "pulse": "92", "temp": "98.6", "spo2": "97", "weight": "71.5" },
  "medications": [
    { "name": "Azithromycin", "dose": "500mg", "frequency": "1-0-0", "duration": "3 days" }
  ],
  "raw_text": "…full OCR output…"
}
```

---

## Tuning for your scans

Real scans are never the ideal on paper — phone photos are skewed, fax printouts
are blown out, ballpoint on carbon paper is faint. These are environment
variables, not code edits:

| Variable | Default | What it does |
|---|---|---|
| `NALAM_MIN_CONFIDENCE` | `0.35` | Drop OCR boxes below this. Raise it for clean printouts, lower it for handwriting. |
| `NALAM_DESKEW_LIMIT` | `15` | Max rotation (degrees) to correct. Beyond this the estimate is probably noise. |
| `NALAM_UPSCALE_WIDTH` | `1600` | Small images are upscaled to this width before OCR. |
| `NALAM_MAX_UPSCALE` | `4` | Ceiling on that enlargement. Without it a small crop gets inflated 12× — slow, and it only interpolates blur into the strokes. |
| `NALAM_TROCR_BELOW` | `0.75` | EasyOCR confidence under which an implausible read is re-read by TrOCR. |
| `NALAM_HANDWRITING` | `auto` | `auto` routes per run, `always` sends everything to TrOCR, `never` disables it (printed-only, much faster). |
| `NALAM_TROCR_MODEL` | fine-tuned if present | Recogniser for scanned pages. Defaults to `backend/models/trocr-rx` when `finetune.py` has produced one, else `microsoft/trocr-base-handwritten`. |
| `NALAM_PAD_MODEL` | `microsoft/trocr-base-handwritten` | Recogniser for the scribble pad. Deliberately the general model — see below. |
| `NALAM_FORMULARY` | `backend/formulary.txt` | The drug vocabulary garbled reads are snapped to. **The highest-leverage setting in the project.** |
| `NALAM_LEXICON_CUTOFF` | `0.6` | Similarity a read needs before it is considered for correction. `0` disables snapping. |
| `NALAM_MAX_SNAP_RATIO` | `0.34` | Largest share of a word a correction may rewrite. |
| `NALAM_MIN_SNAP_LENGTH` | `5` | Reads shorter than this are never corrected — too little signal. |
| `NALAM_LINE_TOLERANCE` | `0.6` | How close two boxes' vertical centres must be (as a fraction of text height) to count as the same line. Raise it for widely-spaced forms, lower it for cramped ones. |
| `NALAM_OCR_LANGS` | `en` | Comma-separated EasyOCR languages, e.g. `en,hi`. |
| `NALAM_PORT` / `NALAM_HOST` | `5001` / `127.0.0.1` | Where Flask listens. |
| `NALAM_DB` / `NALAM_UPLOADS` | `backend/` | Where the SQLite file and scans live. |
| `NALAM_CORS_ORIGIN` | `*` | Lock this to your dashboard's origin in production. |

---

## Layout

```
NALAM/
├── backend/
│   ├── app.py          Flask API + SQLite persistence + CSV export
│   ├── ocr.py          OpenCV preprocessing, EasyOCR, field parser
│   └── test_ocr.py     Runnable self-check for the parser
├── frontend/
│   ├── src/App.jsx     The whole review dashboard
│   ├── src/styles.css
│   └── vite.config.js  Dev proxy → Flask
├── samples/
│   └── make_sample.py  Synthetic report generator (no real patient data)
└── requirements.txt
```

Run the parser check any time you touch a regex:

```bash
cd backend && python3 test_ocr.py     # → "test_ocr: all assertions passed"
```

It needs no model download and no vision stack — the field parser is deliberately
kept free of `cv2`/`torch` imports so it stays fast to test.

---

## The scribble pad

The dashboard has a pad you can write a word on with a mouse, trackpad or stylus;
**Read** sends it to `POST /api/recognize` and shows what came back, plus the
nearest formulary entry when one is close enough to suggest.

It runs the *general* handwriting model, not the fine-tuned one, on purpose. A
recogniser fine-tuned on prescription words gets sharply better at drug names and
worse at everything else — the fine-tuned checkpoint read a hand-drawn `HI` as
`t`, while the general model reads `Napa` and `Cefotil` correctly. Two models, two
jobs. Override either with `NALAM_PAD_MODEL` / `NALAM_TROCR_MODEL`.

```bash
curl -F file=@word.png http://127.0.0.1:5001/api/recognize
# {"text": "Cefotil", "corrected": "Cefotil", "changed": false}
```

## Improving accuracy on your own handwriting

Two levers, in order of effect:

1. **The formulary.** Replace `backend/formulary.txt` with your clinic's actual
   drug list. A recogniser that reads `cofotil` only becomes `Cefotil` if Cefotil
   is on the list. This costs nothing and is worth more than any model change.
2. **Fine-tune the recogniser** on prescriptions in the handwriting you actually
   get:

   ```bash
   python backend/finetune.py /path/to/rxhandbd --epochs 8
   ```

   Writes `backend/models/trocr-rx`, which `ocr.py` picks up automatically. It
   keeps the best epoch by held-out character error rate, not the last one.

## What a run looks like

On `samples/sample_report.png`, straight and again at 7° with blur and sensor noise
— every field recovered in both cases, in ~8 s on a CPU:

```
Reconstructed lines            Parsed record
-------------------            -------------
Patient Name : Ramesh Kumar    patient_name  Ramesh Kumar
Age : 43 Sex: M                age 43 · sex Male
UHID : MH-2291/25 Date : …     patient_id MH-2291/25 · visit_date 14/07/2025
BP : 138/86 Pulse : 92 …       vitals bp 138/86 · pulse 92 · spo2 97 · temp 98.6 · weight 71.5
Tab Azithromycin S0Omg 1-0-0   Azithromycin · 500mg · 1-0-0 · 3 days
Syrup Ambroxol 1Oml tds        Ambroxol · 10ml · tds
```

Note `S0Omg` and `1Oml` in the raw OCR: letter/digit confusion is constant, and
`normalize_digits()` repairs it only where a number is glued to a unit, so real
words are never touched.

## Accuracy, honestly

- **Printed reports** parse well: the labelled-field patterns cover the common
  OPD-slip layouts (`Name:`, `Age`, `BP`, `Dx`, `Rx`).
- **The detector emits boxes, not lines.** `"Age"` and `":43"` arrive separately,
  and a right-hand column can sort ahead of its own label. `group_lines()`
  rebuilds the physical lines before any pattern runs — without it, most fields
  come back empty even on a clean scan.
- **Handwriting is now genuinely supported, not solved.** Stock EasyOCR on real
  handwritten prescription words ([RxHandBD](https://zenodo.org/records/18478741),
  MIT-licensed, ground-truth transcriptions) scores **73% character error** — it
  does not work at all. Routing those words to TrOCR instead, plus correcting
  against the clinic's drug formulary, measured on 100 real test words:

  | Engine | Exact match | Character error rate |
  |---|---|---|
  | EasyOCR (printed-text engine, used on handwriting) | ~1% | ~73% |
  | TrOCR, raw | ~9% | ~63% |
  | **TrOCR + formulary correction (shipped pipeline)** | **34%** | **45%** |

  That is a real, measured improvement — not a fix. A third of real handwritten
  drug names come back exactly right, and the character error rate is cut by 28
  points, but two in three exact reads still need a doctor's correction. The
  formulary-snap layer refuses to guess when it isn't confident (see
  `snap_to_formulary` in `backend/ocr.py`), so it undercorrects rather than
  inventing wrong drug names. Reproduce this with
  `samples/benchmark_handwriting.py rxhandbd --engine all`.

  **Printed report fields** are unaffected by any of this: 8/8 fields, 5/5
  vitals, 3/3 medications on the sample page, straight or at 7° with blur —
  handwriting routing is decided per line segment, not per page, precisely so a
  printed letterhead never gets sent through the weaker recognizer.

- The confidence score shown per report is the mean of the retained OCR boxes.
  Treat a low number as "read this one carefully", not as a failure.

## Before using this on real patients

This is a digitization aid, **not a medical device**, and nothing here has been
clinically validated. If you point it at real records:

- De-identify data before it leaves the clinic machine, and keep `NALAM_HOST` on
  localhost or behind a VPN — the API ships with no authentication.
- `backend/uploads/` and `backend/nalam.db` hold patient scans in the clear. They
  are gitignored for a reason. Encrypt the disk; back it up like health data.
- Set `NALAM_CORS_ORIGIN` to your actual dashboard origin.
- A doctor signs off every record. The `reviewed` flag exists so exports can
  exclude anything a human has not read.

## License

MIT — see [LICENSE](LICENSE).
