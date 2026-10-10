# TriageFlow ML-Engine

Microservicio HealthTech/MLOps que asigna nivel de triaje (1 crítico, 2 urgente, 3 estable) a pacientes de urgencias con un `RandomForestClassifier` y expone KPIs de saturación en tiempo real. Laboratorio 4 — Contenedores y Despliegue con Docker (UdeA, 2026).

**Stack:** Python 3.14 · FastAPI + Uvicorn · scikit-learn · NumPy · Joblib · Docker multi-stage (`python:3.14-slim`) · puerto 8000

## Endpoints

| Método | Ruta                             | Descripción                               |
| ------ | -------------------------------- | ----------------------------------------- |
| GET    | `/health`                        | Estado del servicio y del modelo          |
| POST   | `/api/v1/events/patient-checkin` | Ingesta de signos vitales + inferencia ML |
| GET    | `/api/v1/bi/er-status`           | KPIs agregados de urgencias               |
