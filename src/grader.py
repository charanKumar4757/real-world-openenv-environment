from typing import Dict, Any, List


def clamp_score(score: float) -> float:
    """
    CRITICAL: Validator requires score strictly between 0 and 1.
    0.0 and 1.0 are NOT allowed.
    """
    score = float(score)
    if score <= 0.0:
        return 0.01
    if score >= 1.0:
        return 0.99
    return score


def appears_in_order(action_history: List[str], required_sequence: List[str]) -> bool:
    """Check if required actions appear in order (not necessarily consecutive)."""
    idx = 0
    for action in action_history:
        if idx < len(required_sequence) and action == required_sequence[idx]:
            idx += 1
    return idx == len(required_sequence)


def repetition_penalty(action_history: List[str]) -> float:
    """Penalize repeating the same action back-to-back."""
    penalty = 0.0
    for i in range(1, len(action_history)):
        if action_history[i] == action_history[i - 1]:
            penalty += 0.1
    return min(penalty, 0.3)  # cap so it never destroys the score


def grade_easy(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
    """
    Easy task: Student Fatigue Scenario.
    Agent must detect fatigue, use autopilot for low-stakes task, finish efficiently.
    """
    score = 0.0
    action_history = info.get("action_history", [])

    # Core sequence check (worth 0.35)
    if appears_in_order(action_history, ["calculate_cognitive_score", "activate_autopilot", "final_answer"]):
        score += 0.35

    # Individual action bonuses
    if "calculate_cognitive_score" in action_history:
        score += 0.10
    if "activate_autopilot" in action_history:
        score += 0.15
    if "final_answer" in action_history:
        score += 0.05

    # Efficiency bonus
    if len(action_history) <= 4:
        score += 0.10

    # State health
    if final_state.get("cognitive_score", 0) >= 35:
        score += 0.10

    # Penalty: stressful tasks left behind
    pending = final_state.get("pending_tasks", [])
    if [t for t in pending if t.get("is_stressful")]:
        score -= 0.15

    # Penalty: repetitive actions
    score -= repetition_penalty(action_history)

    # Penalty: negative cumulative reward
    total_reward = info.get("total_reward", 0.0)
    if total_reward < 0:
        score -= 0.15

    # CRITICAL: clamp strictly between 0 and 1 (not 0.0, not 1.0)
    return clamp_score(score)


def grade_medium(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
    """
    Medium task: Work Queue Optimization with emotional spillover.
    """
    score = 0.0
    action_history = info.get("action_history", [])

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

    # Individual bonuses
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

    # State checks
    if final_state.get("cognitive_score", 0) >= 20:
        score += 0.05
    if final_state.get("emotional_state") in ["neutral", "guarded"]:
        score += 0.05

    # Penalties
    total_reward = info.get("total_reward", 0.0)
    if total_reward < 0:
        score -= 0.10

    score -= repetition_penalty(action_history)

    # CRITICAL: clamp strictly between 0 and 1
    return clamp_score(score)


def grade_hard(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
    """
    Hard task: Team Routing Under Crisis.
    Agent must predict recovery, shield emotions, redistribute work, preserve trust.
    """
    score = 0.0
    action_history = info.get("action_history", [])

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

    # Individual bonuses
    if "forecast_regret" in action_history:
        score += 0.05
    if "predict_recovery" in action_history:
        score += 0.08
    if "trigger_recovery_mode" in action_history:
        score += 0.08
    if "isolate_stressful_task" in action_history:
        score += 0.08
    if "activate_autopilot" in action_history:
        score += 0.08
    if "redistribute_team_load" in action_history:
        score += 0.10
    if "final_answer" in action_history:
        score += 0.05

    # State checks
    if final_state.get("cognitive_score", 0) >= 20:
        score += 0.05
    if final_state.get("human_trust_score", 0) >= 40:
        score += 0.05
    if final_state.get("decision_debt", 10) <= 3:
        score += 0.04

    # Penalties
    total_reward = info.get("total_reward", 0.0)
    if total_reward < 0:
        score -= 0.10

    score -= repetition_penalty(action_history)

    # CRITICAL: clamp strictly between 0 and 1
    return clamp_score(score)