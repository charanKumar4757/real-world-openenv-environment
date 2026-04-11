from typing import Dict, Any, List


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
    return min(penalty, 0.4)  # cap at 0.4 so it doesn't destroy the score


def grade_easy(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
    """
    Easy task: Student Fatigue Scenario.
    Agent must detect fatigue, use autopilot for low-stakes task, finish efficiently.
    Target score: 0.6 - 0.9 for a good agent.
    """
    score = 0.0
    action_history = info.get("action_history", [])

    # Core sequence: assess → automate → finish (worth 0.4)
    if appears_in_order(action_history, ["calculate_cognitive_score", "activate_autopilot", "final_answer"]):
        score += 0.4

    # Individual action bonuses (each worth 0.1)
    if "calculate_cognitive_score" in action_history:
        score += 0.1  # Agent correctly assessed the situation

    if "activate_autopilot" in action_history:
        score += 0.15  # Agent correctly automated a low-stakes task

    if "final_answer" in action_history:
        score += 0.05  # Agent properly ended the episode

    # Efficiency bonus: finished in 4 steps or fewer
    if len(action_history) <= 4:
        score += 0.1

    # State health: cognitive score should not be too low at end
    if final_state.get("cognitive_score", 0) >= 35:
        score += 0.1

    # Penalize if stressful tasks were left behind unhandled
    pending = final_state.get("pending_tasks", [])
    stressful_left = [t for t in pending if t.get("is_stressful")]
    if stressful_left:
        score -= 0.2

    # Penalize repetitive actions
    score -= repetition_penalty(action_history)

    # Penalize if total cumulative reward was negative (agent made bad choices)
    total_reward = info.get("total_reward", 0.0)
    if total_reward < 0:
        score -= 0.2

    return max(0.0, min(1.0, round(score, 2)))


def grade_medium(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
    """
    Medium task: Work Queue Optimization.
    Agent must handle emotional spillover, reorder tasks, and automate low-stakes tasks.
    Target score: 0.5 - 0.85 for a good agent.
    """
    score = 0.0
    action_history = info.get("action_history", [])

    # Ideal sequence (worth 0.35)
    ideal = [
        "forecast_regret",
        "trigger_recovery_mode",
        "isolate_stressful_task",
        "reorder_tasks",
        "activate_autopilot",
        "final_answer"
    ]
    if appears_in_order(action_history, ideal):
        score += 0.35

    # Individual action bonuses
    if "forecast_regret" in action_history:
        score += 0.05  # Agent looked ahead before acting

    if "trigger_recovery_mode" in action_history:
        score += 0.1   # Agent handled the emotional stress correctly

    if "isolate_stressful_task" in action_history:
        score += 0.1   # Agent protected user from spillover

    if "reorder_tasks" in action_history:
        score += 0.1   # Agent optimized the task queue

    if "activate_autopilot" in action_history:
        score += 0.1   # Agent automated low-stakes work

    if "final_answer" in action_history:
        score += 0.05  # Clean episode end

    # State quality checks
    if final_state.get("cognitive_score", 0) >= 20:
        score += 0.05  # Cognitive score survived

    if final_state.get("emotional_state") in ["neutral", "guarded"]:
        score += 0.05  # Emotional state improved

    # FIXED: Correct walrus operator usage (was broken before)
    total_reward = info.get("total_reward", 0.0)
    if total_reward < 0:
        score -= 0.1

    # Penalize repetition
    score -= repetition_penalty(action_history)

    return max(0.0, min(1.0, round(score, 2)))


def grade_hard(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
    """
    Hard task: Team Routing Under Crisis.
    Agent must predict recovery, shield emotions, redistribute work, and preserve trust.
    Target score: 0.45 - 0.80 for a good agent.
    """
    score = 0.0
    action_history = info.get("action_history", [])

    # Ideal sequence (worth 0.30)
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
        score += 0.30

    # Individual action bonuses
    if "forecast_regret" in action_history:
        score += 0.05

    if "predict_recovery" in action_history:
        score += 0.08  # Critical for the hard scenario

    if "trigger_recovery_mode" in action_history:
        score += 0.08

    if "isolate_stressful_task" in action_history:
        score += 0.08

    if "activate_autopilot" in action_history:
        score += 0.08

    if "redistribute_team_load" in action_history:
        score += 0.10  # Key action for the hard task

    if "final_answer" in action_history:
        score += 0.05

    # State quality checks
    if final_state.get("cognitive_score", 0) >= 20:
        score += 0.05

    # Trust must be preserved
    if final_state.get("human_trust_score", 0) >= 40:
        score += 0.05

    # Decision debt should not be excessive
    if final_state.get("decision_debt", 10) <= 3:
        score += 0.04

    # FIXED: Correct check (was broken walrus operator before)
    total_reward = info.get("total_reward", 0.0)
    if total_reward < 0:
        score -= 0.1

    # Penalize repetition
    score -= repetition_penalty(action_history)

    return max(0.0, min(1.0, round(score, 2)))