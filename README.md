# TriageFlow ML-Engine

Microservicio HealthTech/MLOps que asigna nivel de triaje (1 crítico, 2 urgente, 3 estable) a pacientes de urgencias con un `RandomForestClassifier` y expone KPIs de saturación en tiempo real. Laboratorio 4 — Contenedores y Despliegue con Docker (UdeA, 2026).

**Stack:** Python 3.14 · FastAPI + Uvicorn · scikit-learn · NumPy · Joblib · Docker multi-stage (`python:3.14-slim`) · puerto 8000

## Endpoints

| Método | Ruta                             | Descripción                               |
| ------ | -------------------------------- | ----------------------------------------- |
| GET    | `/health`                        | Estado del servicio y del modelo          |
| POST   | `/api/v1/events/patient-checkin` | Ingesta de signos vitales + inferencia ML |
| GET    | `/api/v1/bi/er-status`           | KPIs agregados de urgencias               |

Swagger UI: http://localhost:8000/docs

Ejemplo:

```bash
curl -X POST http://localhost:8000/api/v1/events/patient-checkin \
  -H "Content-Type: application/json" \
  -d '{"patient_id":"PAT-9821","age":68,"heart_rate":135,"systolic_bp":175,"diastolic_bp":105,"spo2":88,"temperature":38.8,"pain_level":9}'
```

## Ejecutar con Docker

```bash
docker build -t TU_USUARIO/triageflow-ml:v1 .
docker run -d -p 8000:8000 --name triage-app TU_USUARIO/triageflow-ml:v1
docker ps && docker images

docker login
docker push TU_USUARIO/triageflow-ml:v1
```

Probar en un entorno limpio: `docker pull TU_USUARIO/triageflow-ml:v1 && docker run -p 8000:8000 TU_USUARIO/triageflow-ml:v1`

## Desarrollo local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest httpx
python ml/train_model.py          # regenera app/triage_model.joblib
uvicorn app.main:app --reload
pytest -q tests
```

## Estructura

```
app/main.py              API FastAPI
app/triage_model.joblib  modelo pre-entrenado
ml/train_model.py        entrenamiento offline (datos sintéticos)
tests/test_api.py        pruebas de los endpoints
Dockerfile               build multi-stage, usuario no-root, HEALTHCHECK
```

> Nota: El modelo se entrena con datos clínicos **simulados**; es un proyecto académico, no una herramienta de decisión médica.
