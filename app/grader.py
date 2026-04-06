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

    # Expected medium flow
    preferred = [
        "trigger_recovery_mode",
        "reorder_tasks",
        "activate_autopilot",
        "final_answer"
    ]
    if appears_in_order(action_history, preferred):
        score += 0.4

    # Stress handling
    if "trigger_recovery_mode" in action_history or "isolate_stressful_task" in action_history:
        score += 0.2

    # Reordering
    if "reorder_tasks" in action_history:
        score += 0.15

    # Cognitive protection
    if final_state.get("cognitive_score", 0) >= 30:
        score += 0.15

    # Debt control
    if final_state.get("decision_debt", 0) <= 2:
        score += 0.1

    score -= repetition_penalty(action_history)

    # Extra rules
    total_reward = info.get("total_reward", 0.0)
    if total_reward < 0:
        score -= 0.2

    if final_state.get("pending_tasks", []):
        score -= 0.15

    return max(0.0, min(1.0, score))


def grade_hard(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
    score = 0.0
    action_history = info.get("action_history", [])

    preferred_sequence = [
        "predict_recovery",
        "isolate_stressful_task",
        "activate_autopilot",
        "redistribute_team_load",
        "review_debt",
        "final_answer"
    ]

    if appears_in_order(action_history, preferred_sequence):
        score += 0.45

    if "predict_recovery" in action_history:
        score += 0.10

    if "isolate_stressful_task" in action_history:
        score += 0.10

    if "activate_autopilot" in action_history:
        score += 0.10

    if "redistribute_team_load" in action_history:
        score += 0.10

    if "review_debt" in action_history or final_state.get("decision_debt", 0) == 0:
        score += 0.10

    if final_state.get("cognitive_score", 0) >= 30:
        score += 0.10

    score -= repetition_penalty(action_history)

    # Extra rules
    total_reward = info.get("total_reward", 0.0)
    if total_reward < 0:
        score -= 0.2

    if final_state.get("pending_tasks", []):
        score -= 0.15

    return max(0.0, min(1.0, score))
