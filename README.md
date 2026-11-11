# Verifi

Health-misinformation and AI-hallucination checking hub — Junior Academy Innovation Challenge (Milestone 4).

Flask + SQLite. No paid services, no heavy ML libraries on the server.

## What it does

| Section | How it works |
|---|---|
| Search & Verify | Matches a claim against the verified-facts table (wording-tolerant). Unknown claims are **saved to the database** and queued for doctor review. |
| AI Detection | TF-IDF + logistic regression trained on [CoAID](https://github.com/cuilimeng/CoAID), scored in pure Python (`detector.py`). Texts under 8 words get no score instead of a made-up one. |
| Doctor Portal | Email + authorization code. Signed-in doctors review the queue (accurate / misleading / false + explanation); reviewed claims become verified facts for future searches. |
| Community | Doctors and patients register with an email (stored in `users`). |
| Dashboard | Counts come from the `events` table, so they are shared by all visitors and survive restarts. |

## Run locally

```bash
pip install -r requirements.txt
FLASK_DEBUG=1 python app.py          # http://127.0.0.1:5000
python -m unittest discover -s tests -v
```

`FLASK_DEBUG=1` only switches off "Secure" cookies so sign-in works over plain http locally.

## Configuration (environment variables)

| Variable | Purpose |
|---|---|
| `VERIFI_DOCTOR_CODES` | Comma-separated authorization codes, read once when the database is first created. If unset, the **demo code `CERT-12345`** is seeded — replace it before sharing the site. |
| `SECRET_KEY` | Session signing key. If unset, one is generated once and stored in `instance/secret_key` (shared by all workers). |
| `VERIFI_DB` | Optional path for the SQLite file (default `instance/verifi.db`). |

To change codes after the first run, delete `instance/verifi.db` (this also erases saved claims/users) or insert a new hashed row into `doctor_codes`.

## Deploy on PythonAnywhere

1. In a Bash console: `cd` into the project and `git pull`.
2. Web tab -> **Reload**.

The `instance/` folder and database are created automatically on first load. To set real codes, add this at the top of the WSGI file, before the app is imported:

```python
import os
os.environ["VERIFI_DOCTOR_CODES"] = "your-code-1,your-code-2"
```

## Re-train the detector

```bash
pip install scikit-learn pandas
python train_detector.py     # downloads CoAID, prints held-out metrics, writes detector_model.json
```

Held-out results of the shipped model (CoAID, split by article, 1,746 texts): accuracy 94.4%, misinformation class precision 0.78 / recall 0.81 / F1 0.79.

## Known limitations (read before trusting a score)

- The detector learned **COVID-19** misinformation from news articles and fact-check claims. It estimates whether text *looks like* that misinformation; it does not check facts and can be wrong on other health topics. A low score is not proof that text is true.
- Only 27 short misinformation *claims* exist in the training data, so accuracy on short claims is not well measured.
- Doctor authorization is by shared codes. There is no automatic verification of medical licenses yet (planned: AI vision check of uploaded certificates, with human confirmation).
- Registration stores the email only; no confirmation emails are sent.

## Data credit

CoAID: Limeng Cui and Dongwon Lee, *CoAID: COVID-19 Healthcare Misinformation Dataset*, 2020 (arXiv:2006.00885).
