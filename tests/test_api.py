from fastapi.testclient import TestClient

from app.main import app

CRITICAL = {"patient_id": "PAT-9821", "age": 68, "heart_rate": 135, "systolic_bp": 175,
            "diastolic_bp": 105, "spo2": 88, "temperature": 38.8, "pain_level": 9}
STABLE = {"patient_id": "PAT-0001", "age": 30, "heart_rate": 72, "systolic_bp": 118,
          "diastolic_bp": 76, "spo2": 98, "temperature": 36.6, "pain_level": 1}


def test_flow():
    with TestClient(app) as c:
        h = c.get("/health").json()
        assert h["status"] == "healthy" and h["model_loaded"] is True

        assert c.get("/api/v1/bi/er-status").json()["hospital_kpis"]["total_admissions"] == 0

        r = c.post("/api/v1/events/patient-checkin", json=CRITICAL).json()
        assert r["triage_level"] == 1 and r["estimated_wait_minutes"] == 0
        r = c.post("/api/v1/events/patient-checkin", json=STABLE).json()
        assert r["triage_level"] == 3

        bi = c.get("/api/v1/bi/er-status").json()
        assert bi["hospital_kpis"]["total_admissions"] == 2
        assert bi["level_breakdown"]["level_1_critical"] == 1


def test_validation():
    with TestClient(app) as c:
        assert c.post("/api/v1/events/patient-checkin", json={**CRITICAL, "spo2": 150}).status_code == 422
