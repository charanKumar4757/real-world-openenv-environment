from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class Observation(BaseModel):
    # ── Core cognitive state ──────────────────────────────────────
    cognitive_score: int = Field(
        default=100,
        description="Current cognitive capacity (0=exhausted, 100=fully fresh)"
    )
    fatigue_level: str = Field(
        default="low",
        description="Fatigue level: low | medium | high | overwhelmed"
    )
    decision_count: int = Field(
        default=0,
        description="Total number of decisions made this session"
    )
    emotional_state: str = Field(
        default="neutral",
        description="Emotional state: neutral | stressed | overwhelmed | guarded"
    )
    spillover_level: int = Field(
        default=0,
        description="Emotional spillover intensity from recent stressful tasks (0-100)"
    )
    recovery_prediction: int = Field(
        default=0,
        description="Predicted minutes until cognitive score reaches safe threshold (70+)"
    )

    # ── Decision management ───────────────────────────────────────
    decision_debt: int = Field(
        default=0,
        description="Number of deferred decisions handled by autopilot pending review"
    )
    debt_items: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of specific deferred decision items"
    )
    autopilot_active: bool = Field(
        default=False,
        description="Whether AI autopilot is currently handling low-stakes decisions"
    )

    # ── Tasks and team ────────────────────────────────────────────
    pending_tasks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Tasks waiting to be handled, each with id/urgency/cognitive_cost/flags"
    )
    team_scores: Dict[str, int] = Field(
        default_factory=dict,
        description="Cognitive capacity scores of all team members including User"
    )

    # ── Trust and transparency ────────────────────────────────────
    human_trust_score: int = Field(
        default=80,
        description="User's trust level in the AI system (0-100)"
    )
    transparency_used: bool = Field(
        default=False,
        description="Whether AI explained its reasoning before automating a decision"
    )

    # ── Environment flags ─────────────────────────────────────────
    shield_active: bool = Field(
        default=False,
        description="Whether emotional contagion shield is protecting the user"
    )
    recovery_window_reserved: bool = Field(
        default=False,
        description="Whether a protected recovery window has been blocked in the schedule"
    )
    upcoming_event: Optional[str] = Field(
        default=None,
        description="Description of upcoming high-stakes event (e.g. client presentation)"
    )
    minutes_to_event: int = Field(
        default=999,
        description="Minutes remaining before the next high-stakes event"
    )

    # ── Advanced features ─────────────────────────────────────────
    predicted_regret: int = Field(
        default=0,
        description="Estimated regret cost if current decisions are deferred or automated"
    )
    resilience_score: int = Field(
        default=0,
        description="Cognitive resilience score from stress test simulation"
    )
    pattern_memory_triggered: bool = Field(
        default=False,
        description="Whether the historical fatigue pattern memory algorithm has fired"
    )
    fragmentation_index: int = Field(
        default=0,
        description="Attention fragmentation level due to context switching (0-100)"
    )
    context_switches: int = Field(
        default=0,
        description="Number of recent attention context switches"
    )
    interruptions: int = Field(
        default=0,
        description="Number of interruptions affecting sustained focus"
    )

    # ── PHASE 3 REQUIRED FIELD ────────────────────────────────────
    situation_summary: str = Field(
        default="",
        description=(
            "Plain-English summary of the current situation for the agent. "
            "Describes cognitive state, emotional risks, pending priorities, "
            "and recommended next action. Generated fresh each step."
        )
    )


class Action(BaseModel):
    action_type: str = Field(
        ...,
        description=(
            "Action to perform. One of: "
            "calculate_cognitive_score | predict_recovery | activate_autopilot | "
            "reorder_tasks | delay_task | trigger_recovery_mode | "
            "redistribute_team_load | review_debt | isolate_stressful_task | "
            "forecast_regret | provide_transparency | reserve_recovery_window | "
            "final_answer"
        )
    )
    target_task_id: Optional[str] = Field(
        default=None,
        description="Task ID for task-specific actions (e.g. isolate, autopilot, delay)"
    )
    target_user: Optional[str] = Field(
        default=None,
        description="Team member name for delegation actions (e.g. redistribute_team_load)"
    )
    summary: Optional[str] = Field(
        default=None,
        description="Optional human-readable summary of why this action was chosen"
    )


class TaskDef(BaseModel):
    task_id: str
    difficulty: str
    cognitive_cost: int
    urgency: int
    is_stressful: bool
    is_low_stakes: bool


class RewardModel(BaseModel):
    reward: float = Field(..., description="Step reward value (strictly between 0 and 1)")
    reason: str = Field(..., description="Plain-English reason for this reward value")