"""
grader.py — Real outcome graders for ACIE-HADO
================================================
Phase 3 fix: graders now check ACTUAL state changes (did cognitive score
improve? were stressful tasks actually removed?) not just action name presence.
All scores strictly between 0.01 and 0.99.
"""
from typing import Dict, Any, List


def clamp_score(score: float) -> float:
    """Score must be strictly between 0 and 1. 0.0 and 1.0 are rejected."""
    score = float(score)
    if score <= 0.0:
        return 0.01
    if score >= 1.0:
        return 0.99
    return round(score, 2)


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
    return min(penalty, 0.3)


def grade_easy(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
    """
    Easy Task: Student Fatigue Scenario.
    REAL OUTCOMES checked:
    - Did cognitive score stay above 35? (was 45 at start)
    - Was the low-stakes task actually automated? (low_stakes_automated > 0)
    - Did the episode end cleanly? (final_answer used)
    - Were no stressful tasks left unhandled?
    """
    score = 0.0
    action_history = info.get("action_history", [])

    # ── REAL OUTCOME CHECKS (Phase 3 requirement) ─────────────────

    # 1. Cognitive score actually maintained or improved
    final_cog = info.get("final_cognitive_score", final_state.get("cognitive_score", 0))
    initial_cog = info.get("initial_cognitive_score", 45)
    if final_cog >= 40:
        score += 0.20   # score survived — real outcome ✅
    elif final_cog >= 30:
        score += 0.10   # partial credit

    # 2. Low-stakes task was actually automated (not just action name present)
    if info.get("low_stakes_automated", 0) > 0:
        score += 0.20   # actual automation happened — real outcome ✅
    elif "activate_autopilot" in action_history:
        score += 0.05   # action taken but env may not have found a task

    # 3. No stressful tasks were left pending (real state check)
    pending = final_state.get("pending_tasks", [])
    stressful_left = [t for t in pending if t.get("is_stressful")]
    if not stressful_left:
        score += 0.15   # real outcome: no dangerous tasks unhandled ✅

    # 4. Episode ended properly
    if "final_answer" in action_history:
        score += 0.10

    # 5. Sequence bonus (action quality)
    if appears_in_order(action_history, ["calculate_cognitive_score", "activate_autopilot", "final_answer"]):
        score += 0.15

    # 6. Efficiency
    if len(action_history) <= 4:
        score += 0.10

    # ── PENALTIES ─────────────────────────────────────────────────

    # Trust must not be destroyed
    if info.get("human_trust_score", 80) < 40:
        score -= 0.10

    # Negative total reward = consistently bad decisions
    total_reward = info.get("total_reward", 0.0)
    if total_reward < 0:
        score -= 0.15

    # Repetition penalty
    score -= repetition_penalty(action_history)

    return clamp_score(score)


def grade_medium(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
    """
    Medium Task: Work Queue Optimization with Emotional Spillover.
    REAL OUTCOMES checked:
    - Was emotional state actually improved? (stressed → neutral/guarded)
    - Were stressful tasks actually resolved?
    - Did cognitive score survive above 20?
    - Was task queue actually reordered?
    """
    score = 0.0
    action_history = info.get("action_history", [])

    # ── REAL OUTCOME CHECKS ───────────────────────────────────────

    # 1. Emotional state actually improved (real state check)
    final_emotion = info.get("emotional_state", final_state.get("emotional_state", "stressed"))
    if final_emotion in ["neutral", "guarded"]:
        score += 0.20   # real outcome: stress resolved ✅
    elif final_emotion == "stressed":
        score += 0.05   # partial: still stressed but not worse

    # 2. Stressful tasks actually removed from queue
    resolved = info.get("stressful_tasks_resolved", 0)
    if resolved > 0:
        score += 0.20   # real outcome: stressor handled ✅
    pending = final_state.get("pending_tasks", [])
    if not [t for t in pending if t.get("is_stressful")]:
        score += 0.10   # double-check: queue confirms no stressful tasks

    # 3. Cognitive score survived
    final_cog = info.get("final_cognitive_score", final_state.get("cognitive_score", 0))
    if final_cog >= 25:
        score += 0.15
    elif final_cog >= 15:
        score += 0.05

    # 4. Low-stakes work was automated
    if info.get("low_stakes_automated", 0) > 0:
        score += 0.10

    # 5. Action sequence quality
    ideal = [
        "forecast_regret", "trigger_recovery_mode", "isolate_stressful_task",
        "reorder_tasks", "activate_autopilot", "final_answer"
    ]
    if appears_in_order(action_history, ideal):
        score += 0.15
    else:
        # Partial credit for using key actions
        for act in ["trigger_recovery_mode", "isolate_stressful_task", "reorder_tasks"]:
            if act in action_history:
                score += 0.03

    # ── PENALTIES ─────────────────────────────────────────────────

    total_reward = info.get("total_reward", 0.0)
    if total_reward < 0:
        score -= 0.10

    if info.get("human_trust_score", 80) < 40:
        score -= 0.10

    score -= repetition_penalty(action_history)

    return clamp_score(score)


def grade_hard(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
    """
    Hard Task: Team Routing Under Crisis.
    REAL OUTCOMES checked:
    - Was cognitive score protected above 20 despite starting at 18?
    - Were stressful tasks actually delegated or resolved?
    - Was human trust preserved above 40?
    - Was decision debt kept under control?
    - Was the episode finished with key tasks handled?
    """
    score = 0.0
    action_history = info.get("action_history", [])

    # ── REAL OUTCOME CHECKS ───────────────────────────────────────

    # 1. Cognitive score actually recovered (started at 18 — should go up)
    final_cog = info.get("final_cognitive_score", final_state.get("cognitive_score", 0))
    initial_cog = info.get("initial_cognitive_score", 18)
    cog_change = final_cog - initial_cog
    if cog_change > 10:
        score += 0.20   # real outcome: meaningful recovery ✅
    elif cog_change > 0:
        score += 0.10   # some improvement
    elif final_cog >= 20:
        score += 0.05   # at least didn't crash

    # 2. Stressful tasks actually resolved (delegated or isolated)
    resolved = info.get("stressful_tasks_resolved", 0)
    if resolved >= 1:
        score += 0.20   # real outcome: stressor removed ✅
    pending = final_state.get("pending_tasks", [])
    stressful_left = [t for t in pending if t.get("is_stressful")]
    if not stressful_left:
        score += 0.10

    # 3. Human trust preserved
    trust = info.get("human_trust_score", final_state.get("human_trust_score", 80))
    if trust >= 50:
        score += 0.10   # trust maintained ✅
    elif trust >= 40:
        score += 0.05

    # 4. Decision debt controlled
    debt = info.get("decision_debt", final_state.get("decision_debt", 10))
    if debt <= 3:
        score += 0.08

    # 5. Key actions used
    if "redistribute_team_load" in action_history:
        score += 0.08   # delegation happened
    if "predict_recovery" in action_history:
        score += 0.05
    if "trigger_recovery_mode" in action_history:
        score += 0.05

    # 6. Full sequence bonus
    ideal = [
        "forecast_regret", "predict_recovery", "trigger_recovery_mode",
        "isolate_stressful_task", "activate_autopilot", "redistribute_team_load", "final_answer"
    ]
    if appears_in_order(action_history, ideal):
        score += 0.10

    # ── PENALTIES ─────────────────────────────────────────────────

    # Low-stakes tasks left unautomated when agent had resources
    if info.get("low_stakes_automated", 0) == 0 and final_cog > 40:
        score -= 0.05

    total_reward = info.get("total_reward", 0.0)
    if total_reward < 0:
        score -= 0.10

    if trust < 30:
        score -= 0.10   # severe trust damage

    score -= repetition_penalty(action_history)

    return clamp_score(score)