from dataclasses import dataclass
from typing import Optional


@dataclass
class ExerciseResult:
    """Typed result returned by Exercise.evaluate()."""

    angle: Optional[float]
    rep_count: int
    stage: Optional[str]  # "up" | "down" | None
    system_stage: str  # "waiting" | "aligning" | "ready" | "counting"
    side: Optional[str] = None  # "left" | "right" | None
