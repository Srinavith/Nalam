"""NALAM Flask API: upload a medical report, get structured fields back, let a
doctor correct them, export the corrected set as CSV.

Storage is stdlib sqlite3 with the parsed record kept as a JSON blob - the schema
of a prescription is not stable enough to be worth normalising, and one file with
no ORM is one fewer thing to debug at 3am.
"""
import base64
import csv
import io
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, g, jsonify, request, send_file, send_from_directory

import ocr

BASE = Path(__file__).resolve().parent
UPLOADS = Path(os.getenv("NALAM_UPLOADS", BASE / "uploads"))
DB_PATH = Path(os.getenv("NALAM_DB", BASE / "nalam.db"))
FRONTEND_DIST = BASE.parent / "frontend" / "dist"
ALLOWED = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MAX_BYTES = 16 * 1024 * 1024

UPLOADS.mkdir(parents=True, exist_ok=True)
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES


@app.after_request
def cors(resp):
    # ponytail: three headers instead of the flask-cors dependency.
    resp.headers["Access-Control-Allow-Origin"] = os.getenv("NALAM_CORS_ORIGIN", "*")
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,PATCH,DELETE,OPTIONS"
    return resp


def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute("""CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            stored_file TEXT NOT NULL,
            created_at TEXT NOT NULL,
            reviewed INTEGER NOT NULL DEFAULT 0,
            confidence REAL NOT NULL DEFAULT 0,
            record TEXT NOT NULL)""")
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    conn = g.pop("db", None)
    if conn is not None:
        conn.commit()
        conn.close()


def row_to_dict(row):
    return {
        "id": row["id"],
        "filename": row["filename"],
        "created_at": row["created_at"],
        "reviewed": bool(row["reviewed"]),
        "confidence": row["confidence"],
        **json.loads(row["record"]),
    }


@app.get("/api/health")
def health():
    return jsonify(status="ok", time=datetime.now(timezone.utc).isoformat())


# Measured on real data, not estimated - see samples/benchmark_handwriting.py
# and the printed-report checks in backend/test_ocr.py. Update these only by
# re-running the benchmark; do not hand-edit the numbers.
BENCHMARK = {
    "printed": {"field_accuracy": 1.00, "sample": "synthetic printed OPD slip, 8 fields + 5 vitals + 3 meds"},
    "handwritten": {"exact_match": 0.34, "cer": 0.452, "baseline_cer": 0.732,
                    "sample": "100 real prescription words, RxHandBD test split"},
}


@app.get("/api/stats")
def stats():
    return jsonify(BENCHMARK)


@app.post("/api/reports")
def create_report():
    """Upload one report image. Returns the parsed record for immediate review."""
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify(error="No file uploaded under form field 'file'"), 400
    ext = Path(upload.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify(error=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED)}"), 415

    blob = upload.read()
    if not blob:
        return jsonify(error="Uploaded file is empty"), 400

    report_id = uuid.uuid4().hex
    stored = UPLOADS / f"{report_id}{ext}"          # never trust the client's filename on disk
    stored.write_bytes(blob)

    try:
        record = ocr.digitize(ocr.read_image(blob))
    except Exception as exc:
        stored.unlink(missing_ok=True)
        app.logger.exception("OCR failed for %s", upload.filename)
        return jsonify(error=f"OCR failed: {exc}"), 422

    confidence = record.pop("confidence", 0.0)
    db().execute(
        "INSERT INTO reports (id, filename, stored_file, created_at, reviewed, confidence, record)"
        " VALUES (?,?,?,?,0,?,?)",
        (report_id, upload.filename, stored.name,
         datetime.now(timezone.utc).isoformat(timespec="seconds"),
         confidence, json.dumps(record)))
    db().commit()
    return jsonify(id=report_id, filename=upload.filename, reviewed=False,
                   confidence=confidence, **record), 201


@app.get("/api/reports")
def list_reports():
    rows = db().execute(
        "SELECT * FROM reports ORDER BY created_at DESC LIMIT ?",
        (min(int(request.args.get("limit", 200)), 1000),)).fetchall()
    return jsonify([row_to_dict(r) for r in rows])


@app.get("/api/reports/<report_id>")
def get_report(report_id):
    row = db().execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
    if row is None:
        return jsonify(error="Report not found"), 404
    return jsonify(row_to_dict(row))


@app.patch("/api/reports/<report_id>")
def update_report(report_id):
    """Doctor's corrections. Whatever keys are sent overwrite the parsed ones."""
    row = db().execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
    if row is None:
        return jsonify(error="Report not found"), 404
    patch = request.get_json(silent=True)
    if not isinstance(patch, dict):
        return jsonify(error="Body must be a JSON object"), 400

    record = json.loads(row["record"])
    reviewed = bool(patch.pop("reviewed", row["reviewed"]))
    for key in ("id", "filename", "created_at", "confidence"):
        patch.pop(key, None)
    record.update(patch)
    db().execute("UPDATE reports SET record=?, reviewed=? WHERE id=?",
                 (json.dumps(record), int(reviewed), report_id))
    db().commit()
    return jsonify(id=report_id, filename=row["filename"], created_at=row["created_at"],
                   confidence=row["confidence"], reviewed=reviewed, **record)


@app.delete("/api/reports/<report_id>")
def delete_report(report_id):
    row = db().execute("SELECT * FROM reports WHERE id=?", (report_id,)).fetchone()
    if row is None:
        return jsonify(error="Report not found"), 404
    (UPLOADS / row["stored_file"]).unlink(missing_ok=True)
    db().execute("DELETE FROM reports WHERE id=?", (report_id,))
    db().commit()
    return "", 204


@app.get("/api/reports/<report_id>/image")
def report_image(report_id):
    row = db().execute("SELECT stored_file FROM reports WHERE id=?", (report_id,)).fetchone()
    if row is None:
        return jsonify(error="Report not found"), 404
    return send_from_directory(UPLOADS, row["stored_file"])


@app.post("/api/recognize")
def recognize():
    """Read a single handwritten word or line - used by the scribble pad.

    Accepts either a multipart 'file' or JSON {"image": "data:image/png;base64,..."}.
    Goes straight to the handwriting recogniser: the input is already one cropped
    line, so there is nothing to detect or lay out. Returns what the model read and
    what the formulary correction makes of it, never one without the other.
    """
    blob = None
    upload = request.files.get("file")
    if upload is not None and upload.filename:
        blob = upload.read()
    else:
        payload = request.get_json(silent=True) or {}
        data_url = payload.get("image", "")
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        if data_url:
            try:
                blob = base64.b64decode(data_url, validate=True)
            except Exception:
                return jsonify(error="image must be valid base64"), 400
    if not blob:
        return jsonify(error="Send a 'file' upload or JSON {\"image\": \"data:image/png;base64,...\"}"), 400
    if len(blob) > MAX_BYTES:
        return jsonify(error="Image too large"), 413

    try:
        image = ocr.read_image(blob)
        if ocr.is_blank(image):
            return jsonify(text="", corrected="", note="nothing drawn"), 200
        # The pad gets the general handwriting model: someone writing here writes
        # any word, not only the drug names the pipeline's model is tuned for.
        text = ocr.read_handwriting([ocr.prepare_scribble(image)],
                                    model_name=ocr.GENERAL_TROCR)[0]
    except Exception as exc:
        app.logger.exception("recognize failed")
        return jsonify(error=f"Recognition failed: {exc}"), 422

    corrected = ocr.snap_to_formulary(text) if text else ""
    return jsonify(text=text, corrected=corrected,
                   changed=bool(text) and corrected.lower() != text.lower())


CSV_COLUMNS = ["id", "created_at", "reviewed", "confidence", "patient_name", "age", "sex",
               "patient_id", "visit_date", "doctor", "diagnosis", "advice",
               "bp", "pulse", "temp", "spo2", "weight", "medications"]


@app.get("/api/export.csv")
def export_csv():
    """One row per report. ?reviewed=1 exports only what a doctor has signed off."""
    query = "SELECT * FROM reports"
    params = ()
    if request.args.get("reviewed") in ("1", "true", "yes"):
        query += " WHERE reviewed=1"
    rows = db().execute(query + " ORDER BY created_at DESC", params).fetchall()

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        item = row_to_dict(row)
        item.update(item.pop("vitals", {}))
        item["medications"] = "; ".join(
            " ".join(filter(None, (m.get("name"), m.get("dose"), m.get("frequency"),
                                   m.get("duration"))))
            for m in item.get("medications", []))
        writer.writerow(item)
    return send_file(io.BytesIO(buf.getvalue().encode()), mimetype="text/csv",
                     as_attachment=True, download_name="nalam_export.csv")


@app.errorhandler(413)
def too_large(_e):
    return jsonify(error=f"File exceeds the {MAX_BYTES // (1024 * 1024)}MB limit"), 413


@app.get("/", defaults={"path": ""})
@app.get("/<path:path>")
def serve_frontend(path):
    """Serves the built React dashboard when `npm run build` has been run."""
    if not FRONTEND_DIST.exists():
        return jsonify(message="NALAM API is running. Build the frontend or use the Vite dev server.",
                       endpoints=["/api/health", "/api/reports", "/api/export.csv"])
    target = FRONTEND_DIST / path
    return send_from_directory(FRONTEND_DIST, path if path and target.is_file() else "index.html")


if __name__ == "__main__":
    app.run(host=os.getenv("NALAM_HOST", "127.0.0.1"),
            port=int(os.getenv("NALAM_PORT", "5001")),
            debug=os.getenv("NALAM_DEBUG") == "1")
