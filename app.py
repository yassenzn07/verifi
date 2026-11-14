"""
Verifi - health misinformation checker (Flask backend).

How it fits together:
  * A user asks a health question. If a verified answer exists it is returned with its source;
    if not, the question is saved to a review queue for volunteer specialists (doctors).
  * A separate text checker scores a pasted text (for example an AI answer) with a small
    classifier trained on CoAID (see detector.py).
  * Storage is one SQLite file, so no external database server is needed (PythonAnywhere free plan).
"""
import difflib
import json
import logging
import os
import re
import secrets
import sqlite3
import time
from datetime import datetime, timedelta, timezone

from flask import Flask, g, jsonify, render_template, request, session
from werkzeug.security import check_password_hash, generate_password_hash

import detector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
DB_PATH = os.environ.get("VERIFI_DB", os.path.join(INSTANCE_DIR, "verifi.db"))
SEED_PATH = os.path.join(BASE_DIR, "facts_seed.json")
log = logging.getLogger("verifi")

# Active team members shown on the About page (edit this list to change the names on the site).
TEAM = ["Yassen", "Nour", "Hassan", "Eva", "Ayinezo"]

# --- limits -------------------------------------------------------------------
MAX_QUERY = 300
MAX_TEXT = 5000
MAX_NOTE = 1000
MAX_SOURCE = 300
HIGH_RISK = 0.75          # probability >= this  -> high risk
UNCERTAIN = 0.40          # probability >= this  -> uncertain (needs expert review)
LOGIN_WINDOW = 600        # seconds
LOGIN_MAX_PER_EMAIL = 5
LOGIN_MAX_TOTAL = 40
NEW_CLAIMS_PER_HOUR = 100   # protects the review queue from being flooded
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

app = Flask(__name__)


# --- secret key (must be the same in every worker, or logins silently fail) ------
def _load_secret_key():
    key = os.environ.get("SECRET_KEY")
    if key:
        return key
    os.makedirs(INSTANCE_DIR, exist_ok=True)
    path = os.path.join(INSTANCE_DIR, "secret_key")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip()


app.config.update(
    SECRET_KEY=_load_secret_key(),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    # PythonAnywhere serves HTTPS, so cookies are Secure by default; local dev uses FLASK_DEBUG=1 (plain http)
    SESSION_COOKIE_SECURE=os.environ.get("FLASK_DEBUG") != "1",
    MAX_CONTENT_LENGTH=64 * 1024,
)


# --- database -----------------------------------------------------------------
def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, timeout=10)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS facts(
    id INTEGER PRIMARY KEY, question TEXT UNIQUE NOT NULL, answer TEXT NOT NULL,
    source TEXT NOT NULL, origin TEXT NOT NULL DEFAULT 'seed', created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS claims(
    id INTEGER PRIMARY KEY, text TEXT UNIQUE NOT NULL, hits INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'pending', verdict TEXT, note TEXT, reviewed_by INTEGER,
    created_at TEXT NOT NULL, reviewed_at TEXT);
CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT, role TEXT NOT NULL,
    status TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS doctor_codes(
    id INTEGER PRIMARY KEY, code_hash TEXT NOT NULL, label TEXT, active INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS events(
    id INTEGER PRIMARY KEY, kind TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS login_failures(
    id INTEGER PRIMARY KEY, ident TEXT NOT NULL, ts REAL NOT NULL);
"""


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=10)
    db.executescript(SCHEMA)
    cols = {r[1] for r in db.execute("PRAGMA table_info(facts)")}
    if "origin" not in cols:          # database created by an older version
        db.execute("ALTER TABLE facts ADD COLUMN origin TEXT NOT NULL DEFAULT 'seed'")
    try:
        with open(SEED_PATH, encoding="utf-8") as f:
            seeds = json.load(f)
    except (OSError, ValueError):
        seeds = []
        log.warning("facts_seed.json missing or invalid; starting with no seed facts")
    for s in seeds:
        db.execute("INSERT OR IGNORE INTO facts(question, answer, source, origin, created_at) VALUES(?,?,?,?,?)",
                   (normalize(s["question"]), s["answer"], s.get("source", ""), "seed", now()))
    if db.execute("SELECT COUNT(*) FROM doctor_codes").fetchone()[0] == 0:
        env_codes = [c.strip() for c in os.environ.get("VERIFI_DOCTOR_CODES", "").split(",") if c.strip()]
        if not env_codes:
            env_codes = ["CERT-12345"]
            log.warning("VERIFI_DOCTOR_CODES not set: seeded the DEMO doctor code CERT-12345. "
                        "Set real codes before the project is shared publicly.")
        for c in env_codes:
            db.execute("INSERT INTO doctor_codes(code_hash, label) VALUES(?,?)",
                       (generate_password_hash(c, method="pbkdf2:sha256"), "seeded"))
    db.commit()
    db.close()


# --- text helpers (work for Arabic and English) --------------------------------
_AR_MARKS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")   # tashkeel + tatweel
_STOP = {"does", "do", "is", "are", "can", "the", "a", "an", "of", "to", "it", "in", "will", "you", "i",
         "هل", "ما", "ماذا", "في", "من", "هو", "هي", "على", "عن"}


def normalize(text):
    """Lower-case, strip Arabic diacritics, unify common Arabic letter variants, keep letters/digits only."""
    t = _AR_MARKS.sub("", text.lower())
    t = re.sub("[إأآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return " ".join(re.findall(r"\w+", t))


def _tokens(text):
    out = []
    for w in normalize(text).split():
        if w in _STOP:
            continue
        if w.isascii():
            if len(w) > 3 and w.endswith("es"):
                w = w[:-2]
            elif len(w) > 3 and w.endswith("s"):
                w = w[:-1]
        elif len(w) > 4 and w.startswith("ال"):
            w = w[2:]
        out.append(w)
    return out


def find_fact(db, query):
    """Best matching verified fact, tolerant of wording ('coffee causes sleeping' ~ 'does coffee cause sleeping')."""
    qt = set(_tokens(query))
    qn = normalize(query)
    best, best_score = None, 0.0
    for row in db.execute("SELECT question, answer, source FROM facts"):
        ft = set(_tokens(row["question"]))
        jaccard = len(qt & ft) / len(qt | ft) if (qt | ft) else 0.0
        seq = difflib.SequenceMatcher(None, qn, normalize(row["question"])).ratio()
        score = max(jaccard, seq if seq >= 0.85 else 0.0)
        if score > best_score:
            best, best_score = row, score
    return best if best_score >= 0.8 else None


def json_body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else None


def bad(msg, status=400, code="bad_request"):
    return jsonify({"ok": False, "error": msg, "code": code}), status


def log_event(db, kind):
    db.execute("INSERT INTO events(kind, created_at) VALUES(?,?)", (kind, now()))


def current_doctor(db):
    uid = session.get("doctor_id")
    if not uid:
        return None
    return db.execute("SELECT * FROM users WHERE id=? AND role='doctor' AND status='verified'", (uid,)).fetchone()


# --- pages --------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html", team=TEAM)


# --- ask a question / search ------------------------------------------------------
@app.route("/api/search", methods=["POST"])
def api_search():
    data = json_body()
    if data is None:
        return bad("Expected JSON body.", 415, "bad_json")
    query = str(data.get("query", "")).strip()
    if len(query) > MAX_QUERY:
        return bad(f"Question is too long (max {MAX_QUERY} characters).", 400, "too_long")
    key = normalize(query)
    if not key:
        return jsonify({"ok": True, "verified": False, "code": "empty",
                        "result": "Please write a health question or claim to check."})

    db = get_db()
    fact = find_fact(db, query)
    if fact:
        log_event(db, "search_verified")
        db.commit()
        return jsonify({"ok": True, "verified": True, "code": "verified", "answer": fact["answer"],
                        "result": "Verified fact: " + fact["answer"], "source": fact["source"]})

    row = db.execute("SELECT id FROM claims WHERE text=?", (key,)).fetchone()
    if row:
        db.execute("UPDATE claims SET hits = hits + 1 WHERE id=?", (row["id"],))
        code = "queued"
    else:
        since = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
        recent = db.execute("SELECT COUNT(*) FROM claims WHERE created_at > ?", (since,)).fetchone()[0]
        if recent >= NEW_CLAIMS_PER_HOUR:
            code = "queue_full"
        else:
            db.execute("INSERT INTO claims(text, created_at) VALUES(?,?)", (key, now()))
            code = "queued"
    log_event(db, "search_unverified")
    db.commit()
    msg = ("Not in our verified database yet. This question was saved and will be checked by a volunteer specialist."
           if code == "queued" else
           "Not in our verified database yet. The review queue is very busy right now, please try again later.")
    return jsonify({"ok": True, "verified": False, "code": code, "result": msg})


# --- text checker -------------------------------------------------------------
@app.route("/api/detect", methods=["POST"])
def api_detect():
    data = json_body()
    if data is None:
        return bad("Expected JSON body.", 415, "bad_json")
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"ok": True, "status": "empty", "code": "empty", "verdict": "Please enter text to analyze."})
    if len(text) > MAX_TEXT:
        return bad(f"Text is too long (max {MAX_TEXT} characters).", 400, "too_long")

    r = detector.score_text(text)
    if r["status"] == "insufficient":
        return jsonify({"ok": True, "status": "insufficient", "code": "insufficient", "n_words": r["n_words"],
                        "verdict": f"Not enough text to analyze reliably (need at least {detector.MIN_WORDS} words). "
                                   "No score was produced."})
    if r["status"] == "unsupported":
        return jsonify({"ok": True, "status": "unsupported", "code": "unsupported", "n_words": r["n_words"],
                        "verdict": "The text checker currently understands English health text only. "
                                   "No score was produced."})

    p = r["probability"]
    if p >= HIGH_RISK:
        risk, risk_level, kind = "high", "High Risk", "detect_flagged"
        verdict = "This text strongly resembles known health misinformation."
    elif p >= UNCERTAIN:
        risk, risk_level, kind = "uncertain", "Uncertain", "detect_ok"
        verdict = "Mixed signals. Treat as unverified and have an expert review it."
    else:
        risk, risk_level, kind = "low", "Low Risk", "detect_ok"
        verdict = "No strong misinformation patterns found. This is NOT proof the text is true."

    db = get_db()
    log_event(db, kind)
    db.commit()
    return jsonify({"ok": True, "status": "ok", "code": "ok", "risk": risk, "risk_level": risk_level,
                    "verdict": verdict, "probability": round(p * 100, 1), "drivers": r["drivers"],
                    "n_words": r["n_words"], "is_fake": kind == "detect_flagged"})


@app.route("/api/model-info")
def api_model_info():
    return jsonify(detector.model_info())


# --- dashboard ----------------------------------------------------------------
@app.route("/api/stats")
def api_stats():
    db = get_db()
    ev = {r["kind"]: r["n"] for r in db.execute("SELECT kind, COUNT(*) n FROM events GROUP BY kind")}
    return jsonify({
        "total_tests": sum(ev.values()),
        "fake_detected": ev.get("detect_flagged", 0),
        "verified_facts": ev.get("search_verified", 0),
        "pending_claims": db.execute("SELECT COUNT(*) FROM claims WHERE status='pending'").fetchone()[0],
        "members": db.execute("SELECT COUNT(*) FROM users").fetchone()[0],
    })


# --- community ----------------------------------------------------------------
@app.route("/api/community/join", methods=["POST"])
def api_join():
    data = json_body()
    if data is None:
        return bad("Expected JSON body.", 415, "bad_json")
    email = str(data.get("email", "")).strip().lower()
    name = str(data.get("name", "")).strip()[:80]
    role = str(data.get("role", "")).strip()
    if len(email) > 254 or not EMAIL_RE.match(email):
        return bad("Please enter a valid email address.", 400, "bad_email")
    if role not in ("patient", "doctor"):
        return bad("Role must be 'patient' or 'doctor'.", 400, "bad_role")

    db = get_db()
    if db.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
        return jsonify({"ok": True, "code": "already",
                        "message": "This email is already registered. Welcome back!"})
    status = "active" if role == "patient" else "unverified"
    db.execute("INSERT INTO users(email, name, role, status, created_at) VALUES(?,?,?,?,?)",
               (email, name, role, status, now()))
    db.commit()
    if role == "patient":
        return jsonify({"ok": True, "code": "joined", "message": "You're in! Thanks for joining the Verifi community."})
    return jsonify({"ok": True, "code": "joined_doctor",
                    "message": "Registered. To unlock the review tools, sign in on the Specialists page "
                               "with your authorization code."})


# --- specialist (doctor) portal ---------------------------------------------------
@app.route("/api/doctor/login", methods=["POST"])
def api_doctor_login():
    data = json_body()
    if data is None:
        return bad("Expected JSON body.", 415, "bad_json")
    email = str(data.get("email", "")).strip().lower()
    code = str(data.get("code", "")).strip()
    if len(email) > 254 or not EMAIL_RE.match(email) or not code or len(code) > 128:
        return bad("Enter your email and authorization code.", 400, "missing")

    db = get_db()
    since = time.time() - LOGIN_WINDOW
    per_email = db.execute("SELECT COUNT(*) FROM login_failures WHERE ident=? AND ts>?", (email, since)).fetchone()[0]
    total = db.execute("SELECT COUNT(*) FROM login_failures WHERE ts>?", (since,)).fetchone()[0]
    if per_email >= LOGIN_MAX_PER_EMAIL or total >= LOGIN_MAX_TOTAL:
        return bad("Too many failed attempts. Please wait a few minutes and try again.", 429, "throttled")

    ok = any(check_password_hash(r["code_hash"], code)
             for r in db.execute("SELECT code_hash FROM doctor_codes WHERE active=1"))
    if not ok:
        db.execute("INSERT INTO login_failures(ident, ts) VALUES(?,?)", (email, time.time()))
        db.execute("DELETE FROM login_failures WHERE ts<?", (time.time() - 86400,))
        db.commit()
        return bad("Invalid email or authorization code.", 401, "invalid_login")

    user = db.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if user is None:
        cur = db.execute("INSERT INTO users(email, role, status, created_at) VALUES(?,?,?,?)",
                         (email, "doctor", "verified", now()))
        uid = cur.lastrowid
    else:
        uid = user["id"]
        db.execute("UPDATE users SET role='doctor', status='verified' WHERE id=?", (uid,))
    db.commit()
    session.clear()
    session["doctor_id"] = uid
    return jsonify({"ok": True, "email": email})


@app.route("/api/doctor/logout", methods=["POST"])
def api_doctor_logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/doctor/me")
def api_doctor_me():
    doc = current_doctor(get_db())
    return jsonify({"logged_in": bool(doc), "email": doc["email"] if doc else None})


@app.route("/api/doctor/claims")
def api_doctor_claims():
    db = get_db()
    if not current_doctor(db):
        return bad("Sign in as a verified specialist first.", 401, "need_login")
    rows = db.execute("SELECT id, text, hits, created_at FROM claims WHERE status='pending' "
                      "ORDER BY hits DESC, id ASC LIMIT 50").fetchall()
    return jsonify({"ok": True, "claims": [dict(r) for r in rows]})


@app.route("/api/doctor/review", methods=["POST"])
def api_doctor_review():
    db = get_db()
    doc = current_doctor(db)
    if not doc:
        return bad("Sign in as a verified specialist first.", 401, "need_login")
    data = json_body()
    if data is None:
        return bad("Expected JSON body.", 415, "bad_json")
    verdict = str(data.get("verdict", ""))
    note = str(data.get("note", "")).strip()
    cite = str(data.get("source", "")).strip()
    try:
        claim_id = int(data.get("claim_id"))
    except (TypeError, ValueError):
        return bad("Invalid claim.", 400, "invalid_claim")
    if verdict not in ("accurate", "misleading", "false"):
        return bad("Verdict must be accurate, misleading or false.", 400, "bad_verdict")
    if not note or len(note) > MAX_NOTE or len(cite) > MAX_SOURCE:
        return bad(f"Add a short explanation (up to {MAX_NOTE} characters).", 400, "need_note")

    claim = db.execute("SELECT * FROM claims WHERE id=? AND status='pending'", (claim_id,)).fetchone()
    if not claim:
        return bad("Claim not found or already reviewed.", 404, "not_found")
    label = {"accurate": "Accurate", "misleading": "Misleading", "false": "False"}[verdict]
    db.execute("UPDATE claims SET status='reviewed', verdict=?, note=?, reviewed_by=?, reviewed_at=? WHERE id=?",
               (verdict, note, doc["id"], now(), claim_id))
    db.execute("INSERT OR REPLACE INTO facts(question, answer, source, origin, created_at) VALUES(?,?,?,?,?)",
               (claim["text"], f"{label}. {note}", cite or "Reviewed by a verified specialist", "doctor", now()))
    db.commit()
    return jsonify({"ok": True})


init_db()   # also runs when PythonAnywhere imports this module

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")
