from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from diabeteseye.crew import DiabetesEyeCrew


def _make_json_serializable(obj):
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_make_json_serializable(v) for v in obj)
    return obj


def main() -> None:
    sample_payload = {
        "patient_id": "demo-patient-001",
        "exam_id": "demo-exam-001",
        "images": [
            {
                "uri": str(ROOT / "data" / "IDRiD_01.jpg"),
                "type": "fundus",
                "eye": "OD",
            }
        ],
        "clinical": {
            "hba1c": 8.7,
            "bp_systolic": 138,
            "bp_diastolic": 82,
            "duration_years": 8,
            "age": 62,
            "gender": "F",
            "bmi": 28.5,
            "smoker": False,
            "medications": ["Metformina 500mg 2x/dia", "Lisinopril 10mg/dia"],
            "last_eye_exam": "2024-06-15",
            "family_history": "Pai com DM tipo 2, mãe com hipertensão"
        },
    }

    crew = DiabetesEyeCrew().diabeteseye()
    output = crew.kickoff(inputs=sample_payload)

    final_payload = output.json_dict or output.to_dict() or {"raw": output.raw}
    print("Final output:")
    print(json.dumps(_make_json_serializable(final_payload), indent=2, ensure_ascii=False))

    print("\nTask outputs:")
    for task_output in output.tasks_output:
        name = task_output.name or task_output.description
        content = task_output.json_dict or _make_json_serializable(task_output.to_dict()) or task_output.raw
        print(f"- {name}: {json.dumps(content, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
