"""
Verifi - health misinformation & AI-hallucination checking hub (Flask backend).

Storage: SQLite (a single file, no external server) -> works on PythonAnywhere's free plan.
Detector: TF-IDF + logistic regression trained on CoAID, scored in pure Python (see detector.py).
"""
import difflib
import logging
import os
import re
import secrets
import sqlite3
import time
from datetime import datetime, timezone

from flask import Flask, g, jsonify, render_template, request, session
from werkzeug.security import check_password_hash, generate_password_hash

import detector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
DB_PATH = os.environ.get("VERIFI_DB", os.path.join(INSTANCE_DIR, "verifi.db"))
log = logging.getLogger("verifi")

# --- limits -------------------------------------------------------------------
MAX_QUERY = 300
MAX_TEXT = 5000
MAX_NOTE = 1000
HIGH_RISK = 0.75          # probability >= this  -> "High Risk"
UNCERTAIN = 0.40          # probability >= this  -> "Uncertain" (needs expert review)
LOGIN_WINDOW = 600        # seconds
LOGIN_MAX_PER_EMAIL = 5
LOGIN_MAX_TOTAL = 40
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

SEED_FACTS = [
    ("does coffee cause sleeping",
     "Common misconception! Caffeine in coffee stimulates the central nervous system, promoting alertness rather than sleep."),
    ("does banana raise blood sugar",
     "Bananas contain natural sugars and fiber; blood sugar impact depends on portion size and individual health status."),
]

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
    source TEXT NOT NULL, created_at TEXT NOT NULL);
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
    for q, a in SEED_FACTS:
        db.execute("INSERT OR IGNORE INTO facts(question, answer, source, created_at) VALUES(?,?,?,?)",
                   (q, a, "seed", now()))
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


# --- helpers ------------------------------------------------------------------
_STOP = {"does", "do", "is", "are", "can", "the", "a", "an", "of", "to", "it", "in", "will", "you", "i"}


def _tokens(text):
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOP:
            continue
        if len(w) > 3 and w.endswith("es"):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s"):
            w = w[:-1]
        out.append(w)
    return out


def normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def find_fact(db, query):
    """Best matching verified fact, tolerant of wording ('coffee causes sleeping' ~ 'does coffee cause sleeping')."""
    qt = set(_tokens(query))
    best, best_score = None, 0.0
    for row in db.execute("SELECT question, answer, source FROM facts"):
        ft = set(_tokens(row["question"]))
        jaccard = len(qt & ft) / len(qt | ft) if (qt | ft) else 0.0
        seq = difflib.SequenceMatcher(None, normalize(query), normalize(row["question"])).ratio()
        score = max(jaccard, seq if seq >= 0.85 else 0.0)
        if score > best_score:
            best, best_score = row, score
    return best if best_score >= 0.8 else None


def json_body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else None


def bad(msg, code=400):
    return jsonify({"ok": False, "error": msg}), code


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
    return render_template("index.html")


# --- search & claim logging -----------------------------------------------------
@app.route("/api/search", methods=["POST"])
def api_search():
    data = json_body()
    if data is None:
        return bad("Expected JSON body.", 415)
    query = str(data.get("query", "")).strip()
    if not query:
        return jsonify({"ok": True, "verified": False, "result": "Please enter a medical claim or question to verify."})
    if len(query) > MAX_QUERY:
        return bad(f"Claim is too long (max {MAX_QUERY} characters).")

    db = get_db()
    fact = find_fact(db, query)
    if fact:
        log_event(db, "search_verified")
        db.commit()
        return jsonify({"ok": True, "verified": True, "result": "Verified fact: " + fact["answer"],
                        "source": "Verifi expert-verified database" if fact["source"] == "seed" else "Reviewed by a verified doctor"})

    key = normalize(query)
    row = db.execute("SELECT id FROM claims WHERE text=?", (key,)).fetchone()
    if row:
        db.execute("UPDATE claims SET hits = hits + 1 WHERE id=?", (row["id"],))
    else:
        db.execute("INSERT INTO claims(text, created_at) VALUES(?,?)", (key, now()))
    log_event(db, "search_unverified")
    db.commit()
    return jsonify({"ok": True, "verified": False,
                    "result": "Not in our verified database yet. This claim was saved to the review queue "
                              "and will be checked by a verified doctor."})


# --- misinformation detector ----------------------------------------------------
@app.route("/api/detect", methods=["POST"])
def api_detect():
    data = json_body()
    if data is None:
        return bad("Expected JSON body.", 415)
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"ok": True, "status": "empty", "verdict": "Please enter text to analyze."})
    if len(text) > MAX_TEXT:
        return bad(f"Text is too long (max {MAX_TEXT} characters).")

    r = detector.score_text(text)
    if r["status"] == "insufficient":
        return jsonify({"ok": True, "status": "insufficient", "n_words": r["n_words"],
                        "verdict": f"Not enough text to analyze reliably (need at least {detector.MIN_WORDS} words). "
                                   "No score was produced."})

    p = r["probability"]
    if p >= HIGH_RISK:
        risk, verdict, kind = "High Risk", "This text strongly resembles known health misinformation.", "detect_flagged"
    elif p >= UNCERTAIN:
        risk, verdict, kind = "Uncertain", "Mixed signals. Treat as unverified and have an expert review it.", "detect_ok"
    else:
        risk, verdict, kind = "Low Risk", "No strong misinformation patterns found. This is NOT proof the text is true.", "detect_ok"

    db = get_db()
    log_event(db, kind)
    db.commit()
    return jsonify({"ok": True, "status": "ok", "risk_level": risk, "verdict": verdict,
                    "probability": round(p * 100, 1), "drivers": r["drivers"], "n_words": r["n_words"],
                    "is_fake": kind == "detect_flagged"})


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
        return bad("Expected JSON body.", 415)
    email = str(data.get("email", "")).strip().lower()
    name = str(data.get("name", "")).strip()[:80]
    role = str(data.get("role", "")).strip()
    if len(email) > 254 or not EMAIL_RE.match(email):
        return bad("Please enter a valid email address.")
    if role not in ("patient", "doctor"):
        return bad("Role must be 'patient' or 'doctor'.")

    db = get_db()
    if db.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
        return jsonify({"ok": True, "message": "This email is already registered. Welcome back!"})
    status = "active" if role == "patient" else "unverified"
    db.execute("INSERT INTO users(email, name, role, status, created_at) VALUES(?,?,?,?,?)",
               (email, name, role, status, now()))
    db.commit()
    msg = ("You're in! Thanks for joining the Verifi community." if role == "patient" else
           "Registered. To unlock the review tools, sign in on the Doctor Portal with your authorization code.")
    return jsonify({"ok": True, "message": msg})


# --- doctor portal ------------------------------------------------------------
@app.route("/api/doctor/login", methods=["POST"])
def api_doctor_login():
    data = json_body()
    if data is None:
        return bad("Expected JSON body.", 415)
    email = str(data.get("email", "")).strip().lower()
    code = str(data.get("code", "")).strip()
    if len(email) > 254 or not EMAIL_RE.match(email) or not code or len(code) > 128:
        return bad("Enter your email and authorization code.")

    db = get_db()
    since = time.time() - LOGIN_WINDOW
    per_email = db.execute("SELECT COUNT(*) FROM login_failures WHERE ident=? AND ts>?", (email, since)).fetchone()[0]
    total = db.execute("SELECT COUNT(*) FROM login_failures WHERE ts>?", (since,)).fetchone()[0]
    if per_email >= LOGIN_MAX_PER_EMAIL or total >= LOGIN_MAX_TOTAL:
        return bad("Too many failed attempts. Please wait a few minutes and try again.", 429)

    ok = any(check_password_hash(r["code_hash"], code)
             for r in db.execute("SELECT code_hash FROM doctor_codes WHERE active=1"))
    if not ok:
        db.execute("INSERT INTO login_failures(ident, ts) VALUES(?,?)", (email, time.time()))
        db.execute("DELETE FROM login_failures WHERE ts<?", (time.time() - 86400,))
        db.commit()
        return bad("Invalid email or authorization code.", 401)

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
        return bad("Sign in as a verified doctor first.", 401)
    rows = db.execute("SELECT id, text, hits, created_at FROM claims WHERE status='pending' "
                      "ORDER BY hits DESC, id ASC LIMIT 50").fetchall()
    return jsonify({"ok": True, "claims": [dict(r) for r in rows]})


@app.route("/api/doctor/review", methods=["POST"])
def api_doctor_review():
    db = get_db()
    doc = current_doctor(db)
    if not doc:
        return bad("Sign in as a verified doctor first.", 401)
    data = json_body()
    if data is None:
        return bad("Expected JSON body.", 415)
    verdict = str(data.get("verdict", ""))
    note = str(data.get("note", "")).strip()
    try:
        claim_id = int(data.get("claim_id"))
    except (TypeError, ValueError):
        return bad("Invalid claim.")
    if verdict not in ("accurate", "misleading", "false"):
        return bad("Verdict must be accurate, misleading or false.")
    if not note or len(note) > MAX_NOTE:
        return bad(f"Add a short explanation (up to {MAX_NOTE} characters).")

    claim = db.execute("SELECT * FROM claims WHERE id=? AND status='pending'", (claim_id,)).fetchone()
    if not claim:
        return bad("Claim not found or already reviewed.", 404)
    label = {"accurate": "Accurate", "misleading": "Misleading", "false": "False"}[verdict]
    db.execute("UPDATE claims SET status='reviewed', verdict=?, note=?, reviewed_by=?, reviewed_at=? WHERE id=?",
               (verdict, note, doc["id"], now(), claim_id))
    db.execute("INSERT OR REPLACE INTO facts(question, answer, source, created_at) VALUES(?,?,?,?)",
               (claim["text"], f"{label}. {note}", f"doctor:{doc['id']}", now()))
    db.commit()
    return jsonify({"ok": True})


init_db()   # also runs when PythonAnywhere imports this module

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")
