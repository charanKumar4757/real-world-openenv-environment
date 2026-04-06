from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class Observation(BaseModel):
    cognitive_score: int = Field(default=100, description="Current cognitive score (0-100)")
    fatigue_level: str = Field(default="low", description="Fatigue level: low, medium, high")
    decision_count: int = Field(default=0, description="Number of decisions made so far")
    recovery_prediction: int = Field(default=0, description="Predicted minutes until full recovery")
    decision_debt: int = Field(default=0, description="Number of deferred or autopilot decisions pending review")
    debt_items: List[Dict[str, Any]] = Field(default_factory=list, description="Specific items placed in debt")
    emotional_state: str = Field(default="neutral", description="Emotional state: neutral, stressed, overwhelmed")
    spillover_level: int = Field(default=0, description="Emotional spillover score from recent stressful tasks")
    pending_tasks: List[Dict[str, Any]] = Field(default_factory=list, description="List of tasks with details")
    team_scores: Dict[str, int] = Field(default_factory=dict, description="Cognitive scores of team members")
    autopilot_active: bool = Field(default=False, description="Whether autopilot is handling low stakes tasks")
    shield_active: bool = Field(default=False, description="Whether emotional contagion shield is currently protecting the user")
    upcoming_event: Optional[str] = Field(None, description="Upcoming high stakes event scenario")
    resilience_score: int = Field(default=0, description="Predicted cognitive resilience score for stress tests")
    pattern_memory_triggered: bool = Field(default=False, description="Whether memory algorithm fired based on past history")
    fragmentation_index: int = Field(default=0, description="How fragmented the user's attention is due to context switching")
    context_switches: int = Field(default=0, description="Number of recent context switches")
    interruptions: int = Field(default=0, description="Number of interruptions affecting focus")
    predicted_regret: int = Field(default=0, description="Predicted regret cost of deferring or automating decisions")
    human_trust_score: int = Field(default=80, description="How much the user trusts the AI system")
    transparency_used: bool = Field(default=False, description="Whether the AI explained or justified sensitive automation")
    recovery_window_reserved: bool = Field(default=False, description="Whether the AI reserved a protected recovery window")
    minutes_to_event: int = Field(default=0, description="Minutes left before the next high-stakes event")

class Action(BaseModel):
    action_type: str = Field(..., description="Action to perform: calculate_cognitive_score, predict_recovery, activate_autopilot, assign_task, reorder_tasks, delay_task, trigger_recovery_mode, redistribute_team_load, review_debt, execute_task, isolate_stressful_task, apply_pattern_memory, simulate_stress_test, final_answer, forecast_regret, provide_transparency, reserve_recovery_window")
    target_task_id: Optional[str] = Field(None, description="Task ID if action is task-specific")
    target_user: Optional[str] = Field(None, description="Team member to assign task to")
    summary: Optional[str] = Field(None, description="Summary or specific detail for the action")

class TaskDef(BaseModel):
    task_id: str
    difficulty: str
    cognitive_cost: int
    urgency: int
    is_stressful: bool
    is_low_stakes: bool

class RewardModel(BaseModel):
    reward: float
    reason: str
