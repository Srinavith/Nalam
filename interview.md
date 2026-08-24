# NALAM — Interview Prep

## One-line pitch

OCR pipeline that turns a photo of a medical report — printed, handwritten, or a mix — into a structured, doctor-reviewed record (patient info, vitals, diagnosis, prescription), exportable as CSV. Two OCR engines, routed per line-segment by content, not by a single blanket model.

## Architecture (pipeline order)

```
scan/photo
  → OpenCV preprocessing (grayscale, upscale, denoise, CLAHE, deskew)
  → EasyOCR detector (finds text boxes, grouped into physical lines)
      ├─ printed run      → keep EasyOCR's text (better + ~40x faster on print)
      └─ handwritten run  → crop the whole run → TrOCR reads it
  → field parser (regex-based: name, age, vitals, diagnosis, meds, dosage)
  → formulary snap (garbled drug name → nearest real drug, or refuse)
  → Flask API + SQLite (record stored with source image)
  → React dashboard (doctor edits fields next to the scan; scribble pad for single words)
  → CSV export (only records marked "reviewed", if you filter for it)
```

Nothing is final until a human confirms it — the UI is built around review, not blind automation.

## File-by-file

**backend/ocr.py** — the core pipeline. Two independent halves on purpose: `read_image()`/`run_ocr()` (pixels → raw lines, needs cv2/torch) and `parse_fields()` (lines → structured dict, pure Python). That split is why `test_ocr.py` runs with no model download. Also holds `looks_like_text()` (routing logic), `snap_to_formulary()` (drug-name correction), deskew/preprocessing.

**backend/app.py** — Flask API: upload, list, get, patch (doctor edits), delete, CSV export, `/api/recognize` (scribble pad endpoint). Storage is stdlib `sqlite3` with the parsed record as a JSON blob — no ORM, prescription schema isn't stable enough to normalize.

**backend/finetune.py** — fine-tunes TrOCR on RxHandBD prescription words, writes a checkpoint to `backend/models/trocr-rx`. Keeps the best epoch by held-out CER, not the last one. Not used by default right now — see below.

**backend/formulary.txt** — the drug vocabulary a garbled handwriting read gets snapped against. Plain text, one entry per line. This is the single highest-leverage file in the project — swap it for a real clinic's drug list and accuracy jumps without touching any code.

**backend/test_ocr.py** — one runnable assertion-based check on the parser (no pytest). Deliberately dependency-free so it runs in a couple seconds.

**frontend/src/App.jsx** — the whole dashboard: upload, report list, side-by-side review (scan image + editable fields), the scribble pad component, CSV export links. No component library, no state manager — plain React state, ~260 lines.

**frontend/src/styles.css** — 15 lines. Plain, unstyled-looking HTML on purpose (see below).

## Key design decisions (be ready to explain these out loud)

**Why two OCR engines?**
EasyOCR is a printed-text engine — measured at ~73% character error rate on real handwritten prescriptions (RxHandBD dataset). It effectively does not work on handwriting. TrOCR (a transformer trained on handwriting) does much better there but is slower and no better on clean print. So the page is split: printed text stays with EasyOCR, handwritten runs get cropped out and re-read by TrOCR.

**Why route by "plausibility" (`looks_like_text()`) instead of by confidence score?**
Tried confidence-thresholding first — EasyOCR reports 0.6–0.7 confidence on perfectly clean printed labels like "Diagnosis" or "BP:", so a naive threshold sent good printed text to TrOCR and broke the printed path (lost a field and two vitals in testing). `looks_like_text()` instead checks whether the string is a real word/number/label — a much better signal than the engine's own confidence.

**Why route per line-segment, not per detected box?**
The detector is tuned on print and fragments handwritten words — e.g. "Neuroxen" gets split into boxes like "n" and "euroxen". Re-OCRing each fragment separately just relays the damage. Grouping adjacent boxes into one run before cropping gives TrOCR the whole word.

**Why the formulary-snap layer, and what stops it from inventing wrong drug names?**
Prescriptions draw from a known, finite drug list — so a garbled read like "cofotil" can be corrected to "Cefotil" if that's a real formulary entry. Guarded three ways: max edit-distance ratio (won't snap something too different), minimum word length (short garbled reads aren't corrected — too little signal), and ambiguity refusal (if two formulary entries are equally close, leave it alone). The logic: a wrong drug name is worse than an uncorrected one.

**Why fine-tune trocr-small, not trocr-base?**
Speed on Apple Silicon (MPS/CPU, no CUDA). trocr-base fine-tuning was too slow to iterate on; trocr-small trains fast enough to actually run epochs and check held-out CER between them.

**Why isn't the fine-tuned checkpoint used in production, then?**
Measured it: the trocr-small checkpoint fine-tuned on RxHandBD still trailed zero-shot trocr-base on the real test split (46% CER vs 45%) — trocr-small doesn't have enough raw capacity to make up for its worse starting point, even after fine-tuning. So the pipeline defaults to plain trocr-base-handwritten. This is exactly the kind of thing worth measuring instead of assuming — fine-tuning is not automatically better than a bigger zero-shot model. The honest fix is fine-tuning trocr-base itself (`finetune.py --model microsoft/trocr-base-handwritten`), which needs a GPU to be practical.

**Why does the scribble pad use the general TrOCR model, not the fine-tuned one?**
Fine-tuning specializes the model — it gets much better at prescription drug names and noticeably worse at everything else. Concrete example hit during dev: the fine-tuned checkpoint read a hand-drawn "HI" as "t". The pad is for live demo of arbitrary handwriting, so it deliberately uses the general, non-specialized model.

**Why plain unstyled-looking HTML for the UI?**
Deliberate — the point of the demo is to show the pipeline working, not to show off CSS. Simple, obviously-functional, no framework theater.

## Honest numbers

- **Printed reports:** all fields, vitals, and medications recover reliably (verified against synthetic and skewed/noisy test images).
- **Handwriting, raw EasyOCR baseline:** ~73% character error rate — essentially non-functional. This is the number that justified building the second engine.
- **Handwriting, current pipeline (TrOCR (zero-shot, trocr-base-handwritten) + formulary snap), measured on 100 real RxHandBD test words:** 34% exact match, 45.2% character error rate. This is a real, measured improvement over the 73% baseline, but handwriting recognition on out-of-vocabulary drug names remains a genuinely hard, unsolved problem — be upfront about that if asked. The formulary-snap layer is doing real, disclosed work here (it's explicitly *not* pretending to be a general handwriting model that just works).
- If pushed on why it's not higher: the dataset is real doctors' handwriting with huge inter-writer variance, tiny/ambiguous strokes, and drug names that aren't dictionary words — this is a known-hard subfield (see clinical handwriting OCR literature). More fine-tuning epochs, more training data, and a larger/clinic-specific formulary are the levers, in that order.

## Anticipated questions

**"What's the accuracy on handwriting and why isn't it higher?"**
~45% CER with TrOCR + formulary correction, down from ~73% CER for plain EasyOCR on the same handwritten words. It's a hard problem — doctors' handwriting has huge variance, and drug names aren't in a normal language model's vocabulary. The formulary-snap layer does a lot of the real lifting by constraining the answer space to actual drugs.

**"How would you scale this to a real clinic?"**
Swap SQLite for Postgres once you need concurrent writers; the JSON-blob storage pattern still works fine there. Add auth (there is none right now — noted explicitly in the README as a pre-production gap). Move the OCR pipeline to a queue/worker instead of synchronous request handling, since a page can take several seconds to process.

**"What would you do differently with more time?"**
More fine-tuning epochs and more labeled handwriting data; try a larger/different handwriting model (or an ensemble); build a clinic-specific formulary ingestion tool instead of a static text file; add authentication and audit logging before this touches real patient data.

**"How does the formulary correction work, and what happens if a drug isn't in the list?"**
Edit-distance matching against a list of known drug names, capped by a maximum allowed distance ratio and a minimum word length, and refused outright if two entries are equally close. If a drug isn't in the formulary, the raw (uncorrected) OCR read is kept — the system never fabricates a plausible-sounding drug name it can't back up against the list.

**"Why Flask + SQLite instead of a bigger stack?"**
This is a single-clinic-scale prototype — SQLite with a JSON blob column is enough throughput and zero ops overhead. The architecture doesn't lock you in: swapping to Postgres is a connection-string change, not a rewrite, because the storage layer is already just "store this JSON blob."

**"Why did you build two separate OCR paths instead of one model for everything?"**
Because one model for everything was measurably worse at both jobs. Tried a single-engine approach first, measured it, and the data (73% CER on handwriting) forced the split. This project's whole shape came from measuring, not assuming.

**"What's the hardest bug you hit building this?"**
Confidence-based routing quietly breaking the printed-text path — it looked reasonable in code review but real EasyOCR confidence scores on clean printed labels were low enough to misroute good text to the wrong engine. Caught it by testing on real page images, not just isolated word crops.

## 90-second live demo script

1. Upload a printed sample report → point out every field (name, age, vitals, diagnosis, prescription) auto-populated in the review panel next to the scanned image.
2. Open the scribble pad → write a short word by hand → click recognize → show the TrOCR read (and formulary suggestion if it fires).
3. Edit one field manually to show the human-in-the-loop review step, mark it reviewed.
4. Click CSV export → show the structured record ready to leave the browser.
