"""Enums and shared types for the ACE Active Carver application."""

from enum import Enum

class SimulationState(Enum):
    """Enum representing the state of a simulation after execution."""

    COMPLETED = "completed"
    UNCERTAIN = "uncertain"
    FAILED = "failed"
