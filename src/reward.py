from typing import Dict, Any, List


def calculate_reward(action_type: str, state: Dict[str, Any], action_kwargs: Dict[str, Any] = None) -> float:
    """
    Calculate the reward for a given action based on the current environment state.
    
    Rewards are context-sensitive — the same action can be good or bad depending on state.
    This is the core learning signal for the RL agent.
    """
    if action_kwargs is None:
        action_kwargs = {}

    reward = 0.0
    target_user = action_kwargs.get("target_user")
    action_history = action_kwargs.get("action_history", [])

    cognitive_score = state.get("cognitive_score", 50)
    fatigue_level = state.get("fatigue_level", "medium")
    emotional_state = state.get("emotional_state", "neutral")
    spillover_level = state.get("spillover_level", 0)
    decision_debt = state.get("decision_debt", 0)
    pending_tasks = state.get("pending_tasks", [])
    team_scores = state.get("team_scores", {})

    # ─────────────────────────────────────────────
    # 1. calculate_cognitive_score
    #    Good: agent is checking state before acting (always useful first step)
    # ─────────────────────────────────────────────
    if action_type == "calculate_cognitive_score":
        reward += 0.2  # Always a useful diagnostic action

    # ─────────────────────────────────────────────
    # 2. predict_recovery
    #    Good: when cognitive score is low (planning ahead)
    #    Neutral: when score is already high
    # ─────────────────────────────────────────────
    elif action_type == "predict_recovery":
        if cognitive_score < 50:
            reward += 0.25  # Smart forward planning when tired
        else:
            reward += 0.1   # Not bad, just not urgent

    # ─────────────────────────────────────────────
    # 3. activate_autopilot
    #    Good: when cognitive score is low (correct use)
    #    Bad: when score is high (unnecessary, wastes autonomy)
    # ─────────────────────────────────────────────
    elif action_type == "activate_autopilot":
        if cognitive_score < 50:
            reward += 0.5   # Correct: conserves mental energy
        elif cognitive_score < 70:
            reward += 0.1   # Borderline acceptable
        else:
            reward -= 0.2   # Unnecessary when user is fresh

    # ─────────────────────────────────────────────
    # 4. trigger_recovery_mode / delay_task
    #    Good: when stressed or high spillover
    #    Bad: when everything is already fine
    # ─────────────────────────────────────────────
    elif action_type in ["delay_task", "trigger_recovery_mode"]:
        if spillover_level > 20 or emotional_state in ["stressed", "overwhelmed"]:
            reward += 0.6   # Correct: user needs protection
        elif cognitive_score < 30:
            reward += 0.4   # Also valid when exhausted
        else:
            reward -= 0.2   # Unnecessary delay

    # ─────────────────────────────────────────────
    # 5. redistribute_team_load
    #    FIXED: reward based on whether ANY team member is available (score > 50)
    #    Original bug: required target_user to be passed, which env never did
    # ─────────────────────────────────────────────
    elif action_type == "redistribute_team_load":
        # Check if there's any capable team member (score > 50)
        capable_members = {u: s for u, s in team_scores.items() if s > 50}
        if capable_members:
            reward += 0.6   # Good: smart team delegation
        else:
            reward -= 0.2   # Bad: no one to delegate to

    # ALSO handle specific assign_task with target_user
    elif action_type == "assign_task":
        if target_user and team_scores.get(target_user, 0) > 50:
            reward += 0.6   # Good: chose a capable person
        elif target_user and team_scores.get(target_user, 0) <= 50:
            reward -= 0.4   # Bad: assigned to an already tired person
        else:
            reward -= 0.2   # No target specified

    # ─────────────────────────────────────────────
    # 6. reorder_tasks
    #    Good: when there are multiple tasks and agent is tired
    # ─────────────────────────────────────────────
    elif action_type == "reorder_tasks":
        if fatigue_level in ["high", "overwhelmed"] and len(pending_tasks) > 1:
            reward += 0.5   # Smart prioritization under pressure
        elif len(pending_tasks) > 1:
            reward += 0.2   # Useful even when not exhausted
        else:
            reward += 0.05  # Not much to reorder

    # ─────────────────────────────────────────────
    # 7. isolate_stressful_task
    #    Good: when there are stressful tasks and spillover is active
    # ─────────────────────────────────────────────
    elif action_type == "isolate_stressful_task":
        stressful = [t for t in pending_tasks if t.get("is_stressful")]
        if stressful and spillover_level > 10:
            reward += 0.5   # Correct: protect from emotional contagion
        elif stressful:
            reward += 0.25  # Still useful even without current spillover
        else:
            reward -= 0.1   # Nothing stressful to isolate

    # ─────────────────────────────────────────────
    # 8. review_debt
    #    Good: when debt is high AND score is sufficient to review
    # ─────────────────────────────────────────────
    elif action_type == "review_debt":
        if decision_debt > 0 and cognitive_score >= 40:
            reward += 0.4   # Good time to clear debt
        elif decision_debt > 0 and cognitive_score < 40:
            reward -= 0.2   # Too tired to review — will make it worse
        else:
            reward -= 0.1   # No debt to review

    # ─────────────────────────────────────────────
    # 9. final_answer
    #    Good: when all stressful/low-stakes tasks are handled
    #    Bad: when major tasks are left unresolved
    # ─────────────────────────────────────────────
    elif action_type == "final_answer":
        stressful_left = [t for t in pending_tasks if t.get("is_stressful")]
        low_stakes_left = [t for t in pending_tasks if t.get("is_low_stakes")]

        if len(pending_tasks) == 0:
            reward += 1.0   # Perfect: everything handled
        elif not stressful_left and not low_stakes_left:
            reward += 0.5   # Only medium tasks remain — acceptable
        elif not stressful_left:
            reward += 0.2   # Stressful handled, some low-stakes remain
        else:
            reward -= 0.5   # Ended episode with stressful tasks still pending

    # ─────────────────────────────────────────────
    # 10. forecast_regret
    #    Good: when there are pending tasks (useful planning)
    # ─────────────────────────────────────────────
    elif action_type == "forecast_regret":
        if len(pending_tasks) > 0:
            reward += 0.25  # Forward planning is always valuable

    # ─────────────────────────────────────────────
    # 11. provide_transparency
    #    Modest reward — builds trust
    # ─────────────────────────────────────────────
    elif action_type == "provide_transparency":
        reward += 0.2

    # ─────────────────────────────────────────────
    # 12. reserve_recovery_window
    #    Good: when close to a deadline and cognitively depleted
    # ─────────────────────────────────────────────
    elif action_type == "reserve_recovery_window":
        minutes_to_event = state.get("minutes_to_event", 999)
        if minutes_to_event <= 45 and cognitive_score < 50:
            reward += 0.5   # Smart: protect recovery before deadline
        else:
            reward -= 0.1   # Not needed right now

    # ─────────────────────────────────────────────
    # Global penalties (apply regardless of action)
    # ─────────────────────────────────────────────

    # Penalty for high decision debt (it's draining the user)
    if decision_debt > 3:
        reward -= 0.2

    # Penalty for lost trust
    if state.get("human_trust_score", 100) < 40:
        reward -= 0.2

    # Penalty for executing tasks when severely exhausted
    if action_type == "execute_task" and cognitive_score < 20:
        reward -= 0.4

    # Repeated same action 3 times in a row → the agent is stuck
    if len(action_history) >= 3:
        if action_history[-1] == action_type and action_history[-2] == action_type:
            reward -= 0.5

    # Mild penalty for using same action too often overall (but only after 3 times)
    repeated_count = action_history.count(action_type)
    if repeated_count >= 3:
        reward -= 0.15

    return round(float(reward), 2)