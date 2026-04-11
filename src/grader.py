"""
grader.py — Task graders for ACIE-HADO environment
====================================================
FIXED:
- Broken walrus operator (was: if x := y < 0, now: x = y; if x < 0)
- Scores now strictly between 0.01 and 0.99 (not 0.0, not 1.0)
- Penalty cap so scores never go negative and hit 0.0
"""

from typing import Dict, Any, List


def clamp_score(score: float) -> float:
    """Score must be strictly between 0 and 1. 0.0 and 1.0 are rejected by validator."""
    score = float(score)
    if score <= 0.0:
        return 0.01
    if score >= 1.0:
        return 0.99
    return round(score, 2)


def appears_in_order(action_history: List[str], required_sequence: List[str]) -> bool:
    """Check if required actions appear in the correct order (not necessarily adjacent)."""
    idx = 0
    for action in action_history:
        if idx < len(required_sequence) and action == required_sequence[idx]:
            idx += 1
    return idx == len(required_sequence)


def repetition_penalty(action_history: List[str]) -> float:
    """Penalize consecutive repeated actions. Capped at 0.3 so it never destroys a score."""
    penalty = 0.0
    for i in range(1, len(action_history)):
        if action_history[i] == action_history[i - 1]:
            penalty += 0.1
    return min(penalty, 0.3)


def grade_easy(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
    """
    Easy Task: Student Fatigue Scenario
    Goal: detect fatigue → automate one low-stakes task → finish efficiently
    Good agent scores: 0.65 - 0.90
    """
    score = 0.0
    action_history = info.get("action_history", [])

    # Correct sequence (worth 0.35)
    if appears_in_order(action_history, ["calculate_cognitive_score", "activate_autopilot", "final_answer"]):
        score += 0.35

    # Individual action bonuses
    if "calculate_cognitive_score" in action_history:
        score += 0.10   # agent assessed situation first
    if "activate_autopilot" in action_history:
        score += 0.15   # agent automated low-stakes task
    if "final_answer" in action_history:
        score += 0.05   # clean episode end

    # Efficiency: finished in 4 steps or fewer
    if len(action_history) <= 4:
        score += 0.10

    # State health: cognitive score survived
    if final_state.get("cognitive_score", 0) >= 35:
        score += 0.10

    # Penalty: stressful tasks left unhandled
    pending = final_state.get("pending_tasks", [])
    if any(t.get("is_stressful") for t in pending):
        score -= 0.15

    # Penalty: repetitive useless actions
    score -= repetition_penalty(action_history)

    # Penalty: negative total reward means bad choices
    total_reward = info.get("total_reward", 0.0)  # FIXED: was broken walrus operator
    if total_reward < 0:
        score -= 0.15

    return clamp_score(score)  # FIXED: strictly between 0.01 and 0.99


def grade_medium(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
    """
    Medium Task: Work Queue Optimization with Emotional Spillover
    Goal: look ahead → handle stress → reorder → automate → finish
    Good agent scores: 0.50 - 0.85
    """
    score = 0.0
    action_history = info.get("action_history", [])

    # Ideal sequence (worth 0.30)
    ideal = [
        "forecast_regret",
        "trigger_recovery_mode",
        "isolate_stressful_task",
        "reorder_tasks",
        "activate_autopilot",
        "final_answer"
    ]
    if appears_in_order(action_history, ideal):
        score += 0.30

    # Individual action bonuses
    if "forecast_regret" in action_history:
        score += 0.05
    if "trigger_recovery_mode" in action_history:
        score += 0.10
    if "isolate_stressful_task" in action_history:
        score += 0.10
    if "reorder_tasks" in action_history:
        score += 0.10
    if "activate_autopilot" in action_history:
        score += 0.10
    if "final_answer" in action_history:
        score += 0.05

    # State quality
    if final_state.get("cognitive_score", 0) >= 20:
        score += 0.05
    if final_state.get("emotional_state") in ["neutral", "guarded"]:
        score += 0.05

    # Penalties
    total_reward = info.get("total_reward", 0.0)  # FIXED: was broken walrus operator
    if total_reward < 0:
        score -= 0.10

    score -= repetition_penalty(action_history)

    return clamp_score(score)  # FIXED: strictly between 0.01 and 0.99


def grade_hard(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
    """
    Hard Task: Team Routing Under Crisis
    Goal: predict recovery → stabilize → isolate stressor → delegate → finish
    Good agent scores: 0.45 - 0.80
    """
    score = 0.0
    action_history = info.get("action_history", [])

    # Ideal sequence (worth 0.25)
    ideal = [
        "forecast_regret",
        "predict_recovery",
        "trigger_recovery_mode",
        "isolate_stressful_task",
        "activate_autopilot",
        "redistribute_team_load",
        "final_answer"
    ]
    if appears_in_order(action_history, ideal):
        score += 0.25

    # Individual action bonuses
    if "forecast_regret" in action_history:
        score += 0.05
    if "predict_recovery" in action_history:
        score += 0.08   # critical for hard task
    if "trigger_recovery_mode" in action_history:
        score += 0.08
    if "isolate_stressful_task" in action_history:
        score += 0.08
    if "activate_autopilot" in action_history:
        score += 0.08
    if "redistribute_team_load" in action_history:
        score += 0.10   # key action for hard task
    if "final_answer" in action_history:
        score += 0.05

    # State quality
    if final_state.get("cognitive_score", 0) >= 20:
        score += 0.05
    if final_state.get("human_trust_score", 0) >= 40:
        score += 0.05   # trust must be preserved
    if final_state.get("decision_debt", 10) <= 3:
        score += 0.04   # debt under control

    # Penalties
    total_reward = info.get("total_reward", 0.0)  # FIXED: was broken walrus operator
    if total_reward < 0:
        score -= 0.10

    score -= repetition_penalty(action_history)

    return clamp_score(score)  # FIXED: strictly between 0.01 and 0.99