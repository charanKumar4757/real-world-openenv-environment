from typing import Dict, Any, List

def calculate_reward(action_type: str, state: Dict[str, Any], action_kwargs: Dict[str, Any] = None) -> float:
    if action_kwargs is None:
        action_kwargs = {}

    reward = 0.0
    target_user = action_kwargs.get("target_user")
    action_history = action_kwargs.get("action_history", [])

    # -----------------------------
    # Positive learning signals
    # -----------------------------
    if action_type in ["calculate_cognitive_score", "predict_recovery"]:
        reward += 0.2

    elif action_type == "activate_autopilot":
        if state["cognitive_score"] < 50:
            reward += 0.5
        else:
            reward -= 0.3

    elif action_type in ["delay_task", "trigger_recovery_mode"]:
        if state.get("spillover_level", 0) > 20 or state["emotional_state"] in ["stressed", "overwhelmed"]:
            reward += 0.6
        else:
            reward -= 0.2

    elif action_type in ["assign_task", "redistribute_team_load"]:
        if target_user and state.get("team_scores", {}).get(target_user, 0) > 50:
            reward += 0.6
        else:
            reward -= 0.4

    elif action_type == "reorder_tasks":
        if state.get("fatigue_level") in ["high", "overwhelmed"] and len(state.get("pending_tasks", [])) > 1:
            reward += 0.5
        else:
            reward += 0.1

    elif action_type == "review_debt":
        if state.get("decision_debt", 0) > 0:
            if state.get("cognitive_score", 0) >= 40:
                reward += 0.4
            else:
                reward -= 0.2
        else:
            reward -= 0.1

    elif action_type == "final_answer":
        pending_tasks = state.get("pending_tasks", [])
        stressful_tasks = [t for t in pending_tasks if t.get("is_stressful")]
        low_stakes_tasks = [t for t in pending_tasks if t.get("is_low_stakes")]

        if len(pending_tasks) == 0:
            reward += 1.0
        elif len(stressful_tasks) == 0 and len(low_stakes_tasks) == 0:
            reward += 0.5
        elif len(stressful_tasks) == 0:
            reward += 0.2
        else:
            reward -= 0.5

    elif action_type == "forecast_regret":
        if len(state.get("pending_tasks", [])) > 0:
            reward += 0.25

    # Use predicted_regret in decisions
    predicted_regret = state.get("predicted_regret", 0)
    if predicted_regret > 50:
        # High regret risk: penalize risky actions, reward protective ones
        if action_type == "activate_autopilot":
            reward -= 0.2  # Don't autopilot when regret is high
        elif action_type in ["isolate_stressful_task", "trigger_recovery_mode", "delay_task"]:
            reward += 0.1  # Reward protective actions
    elif predicted_regret < 20:
        # Low regret risk: reward efficient actions
        if action_type == "activate_autopilot":
            reward += 0.1  # Safe to autopilot
        elif action_type == "final_answer":
            reward += 0.1  # Safe to finish

    elif action_type == "provide_transparency":
        reward += 0.2

    elif action_type == "reserve_recovery_window":
        if state.get("minutes_to_event", 999) <= 45 and state.get("cognitive_score", 100) < 50:
            reward += 0.5
        else:
            reward -= 0.1

    # -----------------------------
    # Penalties
    # -----------------------------
    if state.get("decision_debt", 0) > 3:
        reward -= 0.3

    if state.get("human_trust_score", 100) < 40:
        reward -= 0.3

    if action_type == "execute_task" and state.get("cognitive_score", 0) < 20:
        reward -= 0.4

    # Repeated action penalty
    if len(action_history) >= 2:
        if action_history[-1] == action_type and action_history[-2] == action_type:
            reward -= 0.5

    # Mild penalty for using same action too often overall
    repeated_count = action_history.count(action_type)
    if repeated_count >= 3:
        reward -= 0.2

    return round(float(reward), 2)
