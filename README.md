# Verifi

Check health claims before you trust or share them. Junior Academy Innovation Challenge, team 7122.

Flask + SQLite. Arabic and English. No paid services, no heavy ML libraries on the server.

## What it does

| Page | How it works |
|---|---|
| Ask | A user asks a health question. If a specialist-verified answer exists it is shown **with its source**. If not, the question is **saved to the database** and queued for a volunteer specialist. |
| Check a text | A small classifier (TF-IDF + logistic regression trained on [CoAID](https://github.com/cuilimeng/CoAID), scored in pure Python in `detector.py`) estimates how much a pasted text looks like known health misinformation. Texts under 8 words, or not in English, get no score instead of a made-up one. |
| Specialists | Doctors sign in with an email and authorization code and review the queue (accurate / misleading / false + explanation + optional source). A reviewed question becomes a verified answer for the next person who asks. |
| Join | Doctors and patients register with an email (stored in `users`). |
| About | How it works, limits, and the active team (edit the `TEAM` list in `app.py`). |

The "Verifi so far" numbers come from the `events` table, so every visitor sees the same numbers and they survive restarts.

## Run locally

```bash
pip install -r requirements.txt
FLASK_DEBUG=1 python app.py          # http://127.0.0.1:5000
python -m unittest discover -s tests -v
```

`FLASK_DEBUG=1` only switches off "Secure" cookies so specialist sign-in works over plain http locally.

## Configuration (environment variables)

| Variable | Purpose |
|---|---|
| `VERIFI_DOCTOR_CODES` | Comma-separated authorization codes, read once when the database is first created. If unset, the **demo code `CERT-12345`** is seeded. Replace it before sharing the site. |
| `SECRET_KEY` | Session signing key. If unset, one is generated once and stored in `instance/secret_key` (shared by all workers). |
| `VERIFI_DB` | Optional path for the SQLite file (default `instance/verifi.db`). |

To change codes after the first run, delete `instance/verifi.db` (this also erases saved questions and users).

## Adding verified answers

Edit `facts_seed.json`. Each entry has a `question`, an `answer` and a `source`. New entries are added the next time the app starts; existing ones are never overwritten. The two entries shipped now are **demo entries that no doctor has reviewed**, and their source field says so. Replace them with doctor-reviewed answers (with a real source) before relying on them.

## Deploy on PythonAnywhere

1. Bash console: `cd` into the project, then `git pull`.
2. Web tab: **Reload**.

The `instance/` folder and the database are created automatically. To set real codes, add this at the top of the WSGI file, before the app is imported:

```python
import os
os.environ["VERIFI_DOCTOR_CODES"] = "your-code-1,your-code-2"
```

## Re-train the detector

```bash
pip install scikit-learn pandas
python train_detector.py     # downloads CoAID, prints held-out metrics, writes detector_model.json
```

Held-out results of the shipped model (CoAID, split by article, 1,746 texts): accuracy 94.4%; misinformation class precision 0.78, recall 0.81, F1 0.79.

## Known limitations

- The detector learned **COVID-19** misinformation from English news articles and fact-check claims. It estimates whether text *looks like* that misinformation. It does not check facts, can be wrong on other health topics, and a low score is not proof that a text is true.
- Only 27 short misinformation *claims* exist in the training data, so accuracy on short claims is not well measured.
- Doctor access uses shared authorization codes. Medical licenses are not verified automatically yet (planned: AI vision check of uploaded certificates, with a human confirming).
- Registration stores the email only; no confirmation emails are sent.
- Verifi gives general information, not medical advice.

## Data credit

CoAID: Limeng Cui and Dongwon Lee, *CoAID: COVID-19 Healthcare Misinformation Dataset*, 2020 (arXiv:2006.00885).
