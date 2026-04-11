from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
 
 
class Observation(BaseModel):
    # Core cognitive state
    cognitive_score: int = Field(default=100, description="Current cognitive score (0-100). Below 30 = danger zone.")
    fatigue_level: str = Field(default="low", description="Fatigue level: low, medium, high, overwhelmed")
    decision_count: int = Field(default=0, description="Number of decisions made so far in this session")
    emotional_state: str = Field(default="neutral", description="Emotional state: neutral, guarded, stressed, overwhelmed")
    spillover_level: int = Field(default=0, description="Emotional spillover from recent stressful tasks (0-100)")
 
    # Recovery and debt
    recovery_prediction: int = Field(default=0, description="Predicted minutes until cognitive score reaches safe level (70+)")
    decision_debt: int = Field(default=0, description="Number of deferred decisions waiting for review")
    debt_items: List[Dict[str, Any]] = Field(default_factory=list, description="Specific deferred task items")
 
    # Tasks and team
    pending_tasks: List[Dict[str, Any]] = Field(default_factory=list, description="List of pending tasks with id, urgency, cognitive_cost, is_stressful, is_low_stakes")
    team_scores: Dict[str, int] = Field(default_factory=dict, description="Cognitive scores of available team members")
 
    # Status flags
    autopilot_active: bool = Field(default=False, description="Whether autopilot is handling low-stakes tasks")
    shield_active: bool = Field(default=False, description="Whether emotional contagion shield is active")
    recovery_window_reserved: bool = Field(default=False, description="Whether a protected recovery window has been reserved")
    transparency_used: bool = Field(default=False, description="Whether AI explained its automation decisions to the user")
    pattern_memory_triggered: bool = Field(default=False, description="Whether historical fatigue pattern was detected and applied")
 
    # Event context
    upcoming_event: Optional[str] = Field(None, description="Description of upcoming high-stakes event if any")
    minutes_to_event: int = Field(default=999, description="Minutes remaining before next high-stakes event")
 
    # Scores
    resilience_score: int = Field(default=0, description="Cognitive resilience score from stress simulation")
    predicted_regret: int = Field(default=0, description="Predicted regret cost if high-value tasks are deferred")
    human_trust_score: int = Field(default=80, description="User trust in the AI system (0-100)")
    fragmentation_index: int = Field(default=0, description="Attention fragmentation from context switching (0-100)")
    context_switches: int = Field(default=0, description="Number of recent context switches")
    interruptions: int = Field(default=0, description="Number of recent interruptions")
 
    # KEY ADDITION: plain English summary the LLM agent can read and understand
    situation_summary: str = Field(
        default="",
        description="Plain English summary of the current situation to help the agent understand what is happening and what action is most appropriate."
    )
 
 
class Action(BaseModel):
    action_type: str = Field(
        ...,
        description=(
            "Action to perform. Valid values: "
            "calculate_cognitive_score, predict_recovery, activate_autopilot, "
            "assign_task, reorder_tasks, delay_task, trigger_recovery_mode, "
            "redistribute_team_load, review_debt, execute_task, "
            "isolate_stressful_task, apply_pattern_memory, simulate_stress_test, "
            "final_answer, forecast_regret, provide_transparency, reserve_recovery_window"
        )
    )
    target_task_id: Optional[str] = Field(None, description="Task ID if action targets a specific task")
    target_user: Optional[str] = Field(None, description="Team member name if assigning a task to someone")
    summary: Optional[str] = Field(None, description="Optional explanation or reasoning for this action")
 
 
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
 