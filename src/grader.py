from typing import Dict, Any, List

def appears_in_order(action_history: List[str], required_sequence: List[str]) -> bool:
    idx = 0
    for action in action_history:
        if idx < len(required_sequence) and action == required_sequence[idx]:
            idx += 1
    return idx == len(required_sequence)

def repetition_penalty(action_history: List[str]) -> float:
    penalty = 0.0
    for i in range(1, len(action_history)):
        if action_history[i] == action_history[i - 1]:
            penalty += 0.1
    return penalty

def grade_easy(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
    score = 0.0
    action_history = info.get("action_history", [])

    # Correct sequence: assess -> automate -> finish
    if appears_in_order(action_history, ["calculate_cognitive_score", "activate_autopilot", "final_answer"]):
        score += 0.5

    # Low-stakes automation used
    if "activate_autopilot" in action_history:
        score += 0.2

    # Finished efficiently
    if len(action_history) <= 4:
        score += 0.2

    # Final state should not be too damaged
    if final_state.get("cognitive_score", 0) >= 35:
        score += 0.1

    score -= repetition_penalty(action_history)

    # Extra rules
    total_reward = info.get("total_reward", 0.0)
    if total_reward < 0:
        score -= 0.2

    if final_state.get("pending_tasks", []):
        score -= 0.15

    return max(0.0, min(1.0, score))


def grade_medium(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
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
        score += 0.6

    if "forecast_regret" in action_history:
        score += 0.05
    if "trigger_recovery_mode" in action_history:
        score += 0.05
    if "isolate_stressful_task" in action_history:
        score += 0.05
    if "reorder_tasks" in action_history:
        score += 0.05
    if "activate_autopilot" in action_history:
        score += 0.05
    if "final_answer" in action_history:
        score += 0.05

    if final_state.get("cognitive_score", 0) >= 20:
        score += 0.05

    score -= repetition_penalty(action_history)

    if total_reward := info.get("total_reward", 0.0) < 0:
        score -= 0.1

    return max(0.0, min(1.0, score))


def grade_hard(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
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
        score += 0.6

    if "forecast_regret" in action_history:
        score += 0.05
    if "predict_recovery" in action_history:
        score += 0.05
    if "trigger_recovery_mode" in action_history:
        score += 0.05
    if "isolate_stressful_task" in action_history:
        score += 0.05
    if "activate_autopilot" in action_history:
        score += 0.05
    if "redistribute_team_load" in action_history:
        score += 0.05
    if "final_answer" in action_history:
        score += 0.05

    if final_state.get("cognitive_score", 0) >= 20:
        score += 0.05

    score -= repetition_penalty(action_history)

    if total_reward := info.get("total_reward", 0.0) < 0:
        score -= 0.1

    return max(0.0, min(1.0, score))
