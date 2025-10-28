# Agents package for DiabetesEye
from .agents import (
    ClinicalIntegratorAgent,
    DRClassifierAgent,
    FollowUpPlannerAgent,
    PipelineAgent,
    PreprocessingAgent,
    ProgressionPredictorAgent,
)


def build_crew():  # noqa: D401 - compatibility wrapper
    from ..crew import build_crew as _build

    return _build()


__all__ = [
    "PipelineAgent",
    "PreprocessingAgent",
    "DRClassifierAgent",
    "ProgressionPredictorAgent",
    "ClinicalIntegratorAgent",
    "FollowUpPlannerAgent",
    "build_crew",
]