"""Run with:  python -m unittest discover -s tests -v   (from the project folder)"""
import os
import sys
import tempfile
import unittest

_tmp = tempfile.mkdtemp()
os.environ["VERIFI_DB"] = os.path.join(_tmp, "test.db")
os.environ["SECRET_KEY"] = "test-secret"
os.environ["VERIFI_DOCTOR_CODES"] = "TEST-CODE-1"
os.environ["FLASK_DEBUG"] = "1"          # plain-http cookies for the test client
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as verifi  # noqa: E402

MISINFO = "Drinking bleach or hot water every morning cures COVID-19 and the government is hiding it from you"
RELIABLE = "Regular exercise and a balanced diet can help reduce the risk of heart disease according to health experts"


class VerifiTests(unittest.TestCase):
    def setUp(self):
        self.c = verifi.app.test_client()

    def post(self, path, body):
        return self.c.post(path, json=body)

    def test_01_search_fuzzy_match_and_non_json(self):
        r = self.post("/api/search", {"query": "coffee causes sleeping"}).get_json()
        self.assertTrue(r["verified"])
        self.assertEqual(self.c.post("/api/search", data="x").status_code, 415)

    def test_02_unverified_claim_is_persisted(self):
        q = "Garlic water cures every infection"
        r = self.post("/api/search", {"query": q}).get_json()
        self.assertFalse(r["verified"])
        self.post("/api/search", {"query": q})
        with verifi.app.app_context():
            row = verifi.get_db().execute("SELECT hits, status FROM claims WHERE text=?", (verifi.normalize(q),)).fetchone()
        self.assertEqual((row["hits"], row["status"]), (2, "pending"))

    def test_03_detector(self):
        short = self.post("/api/detect", {"text": "cure"}).get_json()
        self.assertEqual(short["status"], "insufficient")
        self.assertNotIn("probability", short)
        bad = self.post("/api/detect", {"text": MISINFO}).get_json()
        good = self.post("/api/detect", {"text": RELIABLE}).get_json()
        self.assertEqual(bad["risk_level"], "High Risk")
        self.assertEqual(good["risk_level"], "Low Risk")
        # same length must not decide the outcome
        self.assertNotEqual(bad["probability"], good["probability"])

    def test_04_stats_come_from_database(self):
        before = self.c.get("/api/stats").get_json()
        self.post("/api/detect", {"text": MISINFO})
        after = self.c.get("/api/stats").get_json()
        self.assertEqual(after["total_tests"], before["total_tests"] + 1)
        self.assertEqual(after["fake_detected"], before["fake_detected"] + 1)

    def test_05_community_join(self):
        self.assertEqual(self.post("/api/community/join", {"email": "nope", "role": "patient"}).status_code, 400)
        self.assertEqual(self.post("/api/community/join", {"email": "a@b.co", "role": "hacker"}).status_code, 400)
        r = self.post("/api/community/join", {"email": "Pat@Example.com", "name": "Pat", "role": "patient"})
        self.assertTrue(r.get_json()["ok"])
        again = self.post("/api/community/join", {"email": "pat@example.com", "role": "patient"}).get_json()
        self.assertIn("already", again["message"])

    def test_06_doctor_flow(self):
        self.assertEqual(self.c.get("/api/doctor/claims").status_code, 401)
        self.assertEqual(self.post("/api/doctor/login", {"email": "doc@x.org", "code": "WRONG"}).status_code, 401)
        ok = self.post("/api/doctor/login", {"email": "doc@x.org", "code": "TEST-CODE-1"})
        self.assertEqual(ok.status_code, 200)
        self.assertTrue(self.c.get("/api/doctor/me").get_json()["logged_in"])      # session survives next request
        q = "Cold showers cure the flu"
        self.post("/api/search", {"query": q})
        claims = self.c.get("/api/doctor/claims").get_json()["claims"]
        cid = [x["id"] for x in claims if x["text"] == verifi.normalize(q)][0]
        self.assertEqual(self.post("/api/doctor/review", {"claim_id": cid, "verdict": "false", "note": ""}).status_code, 400)
        self.assertTrue(self.post("/api/doctor/review", {"claim_id": cid, "verdict": "false",
                                                         "note": "No evidence; the flu is caused by a virus."}).get_json()["ok"])
        r = self.post("/api/search", {"query": q}).get_json()          # reviewed claim now answers future searches
        self.assertTrue(r["verified"])
        self.assertIn("False.", r["result"])
        self.post("/api/doctor/logout", {})
        self.assertEqual(self.c.get("/api/doctor/claims").status_code, 401)

    def test_07_login_throttle(self):
        codes = [self.post("/api/doctor/login", {"email": "brute@x.org", "code": f"bad{i}"}).status_code for i in range(7)]
        self.assertEqual(codes[:5], [401] * 5)
        self.assertEqual(codes[5], 429)
        # a locked-out email cannot get in even with the right code
        self.assertEqual(self.post("/api/doctor/login", {"email": "brute@x.org", "code": "TEST-CODE-1"}).status_code, 429)


    def test_08_arabic_claims_are_stored_separately(self):
        a, b = "هل الثوم يعالج الإنفلونزا؟", "هل الحجامة تعالج السكري"
        for q in (a, b, a):
            r = self.post("/api/search", {"query": q}).get_json()
            self.assertFalse(r["verified"])
            self.assertEqual(r["code"], "queued")
        with verifi.app.app_context():
            rows = verifi.get_db().execute("SELECT text, hits FROM claims WHERE text LIKE '%الثوم%' OR text LIKE '%الحجامه%'").fetchall()
        self.assertEqual(len(rows), 2)                       # two different Arabic claims -> two rows
        self.assertTrue(all(r["text"].strip() for r in rows))   # never an empty key
        self.assertEqual(sorted(r["hits"] for r in rows), [1, 2])

    def test_09_punctuation_only_is_not_stored(self):
        r = self.post("/api/search", {"query": "?!؟ ..."}).get_json()
        self.assertEqual(r["code"], "empty")
        with verifi.app.app_context():
            n = verifi.get_db().execute("SELECT COUNT(*) FROM claims WHERE text=''").fetchone()[0]
        self.assertEqual(n, 0)

    def test_10_detector_rejects_arabic_without_a_score(self):
        before = self.c.get("/api/stats").get_json()["total_tests"]
        r = self.post("/api/detect", {"text": "الثوم يقتل الفيروس في الجسم خلال يوم واحد ويعالج كل الأمراض تماما"}).get_json()
        self.assertEqual(r["status"], "unsupported")
        self.assertNotIn("probability", r)
        self.assertEqual(self.c.get("/api/stats").get_json()["total_tests"], before)   # not counted as a test

    def test_11_team_page_lists_active_members_only(self):
        html = self.c.get("/").get_data(as_text=True)
        for name in ("Yassen", "Nour", "Hassan", "Eva", "Ayinezo"):
            self.assertIn(name, html)
        self.assertNotIn("Jana", html)

    def test_12_sources_are_returned_and_doctor_can_cite(self):
        seed = self.post("/api/search", {"query": "does banana raise blood sugar"}).get_json()
        self.assertIn("source", seed)
        self.post("/api/doctor/login", {"email": "doc2@x.org", "code": "TEST-CODE-1"})
        q = "Honey cures diabetes"
        self.post("/api/search", {"query": q})
        cid = [x["id"] for x in self.c.get("/api/doctor/claims").get_json()["claims"]
               if x["text"] == verifi.normalize(q)][0]
        self.post("/api/doctor/review", {"claim_id": cid, "verdict": "false", "note": "Honey does not cure diabetes.",
                                          "source": "Example Hospital patient leaflet"})
        r = self.post("/api/search", {"query": q}).get_json()
        self.assertEqual(r["source"], "Example Hospital patient leaflet")
        self.post("/api/doctor/logout", {})

    def test_13_queue_cap(self):
        old = verifi.NEW_CLAIMS_PER_HOUR
        verifi.NEW_CLAIMS_PER_HOUR = 0
        try:
            r = self.post("/api/search", {"query": "A brand new question about vitamins"}).get_json()
        finally:
            verifi.NEW_CLAIMS_PER_HOUR = old
        self.assertEqual(r["code"], "queue_full")


if __name__ == "__main__":
    unittest.main()
