"""TriageFlow ML-Engine — microservicio de triaje con ML y métricas BI en tiempo real."""
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("triageflow")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).parent / "triage_model.joblib"))
WAIT_MINUTES = {1: 0, 2: 15, 3: 45}  # tiempo de espera estimado por nivel
MAX_EVENTS = int(os.getenv("MAX_EVENTS", "10000"))  # tope del buffer en memoria

state: dict = {"model": None, "features": None}
events: list[dict] = []  # buffer OLAP en memoria
events_lock = Lock()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Carga el modelo una sola vez al iniciar el contenedor."""
    try:
        artifact = joblib.load(MODEL_PATH)
        state["model"] = artifact["model"]
        state["features"] = artifact["features"]
        logger.info("Modelo cargado desde %s", MODEL_PATH)
    except Exception:
        logger.exception("No se pudo cargar el modelo")
    yield


app = FastAPI(
    title="TriageFlow ML-Engine",
    description="Triaje de urgencias con RandomForest y KPIs de saturación en tiempo real.",
    version="1.0.0",
    lifespan=lifespan,
)


class PatientCheckin(BaseModel):
    patient_id: str = Field(..., min_length=1, max_length=64, examples=["PAT-9821"])
    age: int = Field(..., ge=0, le=120, examples=[68])
    heart_rate: int = Field(..., ge=20, le=250, examples=[135])
    systolic_bp: int = Field(..., ge=40, le=300, examples=[175])
    diastolic_bp: int = Field(..., ge=20, le=200, examples=[105])
    spo2: float = Field(..., ge=50, le=100, examples=[88])
    temperature: float = Field(..., ge=30, le=45, examples=[38.8])
    pain_level: int = Field(..., ge=0, le=10, examples=[9])


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "TriageFlow-ML",
        "model_loaded": state["model"] is not None,
        "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
    }


@app.post("/api/v1/events/patient-checkin",
        responses={
        503: {
            "description": "El modelo ML no está disponible",
        }
    },
)
def patient_checkin(p: PatientCheckin):
    model = state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo ML no disponible")

    x = np.array([[p.age, p.heart_rate, p.systolic_bp, p.diastolic_bp,
                   p.spo2, p.temperature, p.pain_level]], dtype=float)
    probs = model.predict_proba(x)[0]
    idx = int(np.argmax(probs))
    level = int(model.classes_[idx])
    confidence = round(float(probs[idx]) * 100, 2)

    with events_lock:
        events.append({"level": level, "confidence": confidence, "spo2": p.spo2})
        if len(events) > MAX_EVENTS:
            del events[: len(events) - MAX_EVENTS]

    return {
        "status": "TRIAGED_BY_ML",
        "patient_id": p.patient_id,
        "triage_level": level,
        "model_confidence": confidence,
        "estimated_wait_minutes": WAIT_MINUTES[level],
    }


@app.get("/api/v1/bi/er-status")
def er_status():
    with events_lock:
        snapshot = list(events)
    total = len(snapshot)
    count = lambda lvl: sum(1 for e in snapshot if e["level"] == lvl)  # noqa: E731

    return {
        "hospital_kpis": {
            "total_admissions": total,
            "critical_cases_ratio": round(100 * count(1) / total, 2) if total else 0.0,
            "ml_mean_confidence_pct": round(sum(e["confidence"] for e in snapshot) / total, 2) if total else 0.0,
            "average_room_spo2": round(sum(e["spo2"] for e in snapshot) / total, 2) if total else 0.0,
        },
        "level_breakdown": {
            "level_1_critical": count(1),
            "level_2_urgent": count(2),
            "level_3_stable": count(3),
        },
    }
