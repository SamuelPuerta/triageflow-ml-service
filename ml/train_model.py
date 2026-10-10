"""Entrenamiento offline del modelo de triaje de TriageFlow.

Simula datos clínicos multivariables, entrena un RandomForestClassifier y
serializa el artefacto en app/triage_model.joblib.

Uso:  python ml/train_model.py
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# El orden de las características es un contrato con app/main.py
FEATURES = [
    "age", "heart_rate", "systolic_bp", "diastolic_bp",
    "spo2", "temperature", "pain_level",
]
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "app" / "triage_model.joblib"
SEED = 42


def simulate_patients(n: int, rng: np.random.Generator) -> np.ndarray:
    """Genera signos vitales plausibles (mezcla de pacientes estables y graves)."""
    severity = rng.beta(1.2, 2.2, size=n)  # 0 = sano, 1 = muy grave
    age = np.clip(rng.normal(45 + 25 * severity, 18), 1, 100)
    heart_rate = np.clip(rng.normal(75 + 65 * severity, 12), 35, 200)
    systolic = np.clip(rng.normal(120 + 55 * (severity - 0.3), 15), 60, 240)
    diastolic = np.clip(systolic * rng.normal(0.62, 0.05), 40, 140)
    spo2 = np.clip(rng.normal(98 - 14 * severity, 2), 60, 100)
    temperature = np.clip(rng.normal(36.7 + 2.2 * severity, 0.5), 34, 42)
    pain = np.clip(np.round(rng.normal(2 + 7 * severity, 1.8)), 0, 10)
    return np.column_stack([age, heart_rate, systolic, diastolic, spo2, temperature, pain])


def label_triage(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Etiqueta con reglas clínicas simplificadas + ruido (1 crítico, 2 urgente, 3 estable)."""
    age, hr, sbp, dbp, spo2, temp, pain = X.T
    score = (
        3.0 * (spo2 < 90) + 1.5 * ((spo2 >= 90) & (spo2 < 94))
        + 2.0 * ((hr > 130) | (hr < 45)) + 1.0 * ((hr > 110) & (hr <= 130))
        + 2.0 * ((sbp > 180) | (sbp < 90)) + 1.0 * ((sbp > 150) & (sbp <= 180))
        + 1.0 * (dbp > 100)
        + 1.0 * ((temp > 38.5) | (temp < 35.5)) + 0.5 * ((temp > 37.8) & (temp <= 38.5))
        + 1.5 * (pain >= 8) + 0.5 * ((pain >= 5) & (pain < 8))
        + 0.5 * (age > 65)
        + rng.normal(0, 0.4, size=len(X))
    )
    return np.where(score >= 4.5, 1, np.where(score >= 2.0, 2, 3))


def main() -> None:
    rng = np.random.default_rng(SEED)
    X = simulate_patients(6000, rng)
    y = label_triage(X, rng)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    model = RandomForestClassifier( 
        n_estimators=150, max_depth=10, min_samples_leaf=3, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print(f"Distribución de clases: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(classification_report(y_test, model.predict(X_test), digits=3))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, OUTPUT_PATH, compress=3)
    print(f"Modelo guardado en {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
