"""
grader.py — Outcome-based graders for the ACIE-HADO environment.
 
KEY CHANGE from previous version:
These graders check WHAT ACTUALLY HAPPENED in the environment
(final state values, tasks removed, team used, score preserved),
NOT just which action names were called.
 
This makes the grader resistant to agents that guess action names
without actually understanding the scenario.
"""
 
from typing import Dict, Any, List
 
 
# ─────────────────────────────────────────────────────────────────────────────
# EASY GRADER
# Scenario: Student with 2 tasks — choose_elective (hard) + lunch_order (low-stakes)
# Agent wins by: using autopilot for lunch_order, keeping score >= 40
# ─────────────────────────────────────────────────────────────────────────────
def grade_easy(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
    score = 0.0
    reasons = []
 
    final_cog = final_state.get("cognitive_score", 0)
    pending = final_state.get("pending_tasks", [])
    pending_ids = [t["task_id"] for t in pending]
    autopilot_ids = info.get("autopilot_task_ids", [])
    action_history = info.get("action_history", [])
    total_reward = info.get("total_reward", 0.0)
 
    # 1. Did cognitive score stay healthy? (40%)
    if final_cog >= 50:
        score += 0.40
        reasons.append(f"Cognitive score preserved well at {final_cog} (+0.40)")
    elif final_cog >= 40:
        score += 0.25
        reasons.append(f"Cognitive score acceptable at {final_cog} (+0.25)")
    elif final_cog >= 25:
        score += 0.10
        reasons.append(f"Cognitive score low but survived at {final_cog} (+0.10)")
    else:
        reasons.append(f"Cognitive score collapsed to {final_cog} (+0.00)")
 
    # 2. Was the low-stakes task (lunch_order) handled via autopilot? (30%)
    if "lunch_order" in autopilot_ids:
        score += 0.30
        reasons.append("lunch_order correctly handled via autopilot (+0.30)")
    elif "lunch_order" not in pending_ids:
        # It was removed some other way (e.g. execute) — partial credit
        score += 0.10
        reasons.append("lunch_order was removed but not via autopilot (+0.10)")
    else:
        reasons.append("lunch_order was never handled (+0.00)")
 
    # 3. Did the agent NOT directly execute a high-cost task while score < 40? (20%)
    executed_heavy_while_fatigued = False
    if "execute_task" in action_history:
        # If user executed a task, we check if score was low at any point
        # We infer from the final score and task removal
        if final_cog < 15 and "choose_elective" not in pending_ids:
            executed_heavy_while_fatigued = True
    if not executed_heavy_while_fatigued:
        score += 0.20
        reasons.append("Agent avoided executing high-cost tasks dangerously (+0.20)")
    else:
        reasons.append("Agent executed heavy task while cognitively depleted (+0.00)")
 
    # 4. Efficiency: finished in fewer steps = better (10%)
    steps = info.get("step", 10)
    if steps <= 4:
        score += 0.10
        reasons.append(f"Completed efficiently in {steps} steps (+0.10)")
    elif steps <= 6:
        score += 0.05
        reasons.append(f"Completed in {steps} steps (+0.05)")
    else:
        reasons.append(f"Took {steps} steps — too many (+0.00)")
 
    # Penalty: total reward was highly negative
    if total_reward < -0.5:
        score = max(0.0, score - 0.10)
        reasons.append("Total reward was negative (-0.10)")
 
    final_score = round(max(0.0, min(1.0, score)), 2)
    return final_score
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MEDIUM GRADER
# Scenario: Developer with stressed state + 3 tasks — stressful email, complex report, routine update
# Agent wins by: isolating stressor, reordering tasks, score >= 20
# ─────────────────────────────────────────────────────────────────────────────
def grade_medium(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
    score = 0.0
    reasons = []
 
    final_cog = final_state.get("cognitive_score", 0)
    final_spillover = final_state.get("spillover_level", 100)
    final_emotional = final_state.get("emotional_state", "stressed")
    pending = final_state.get("pending_tasks", [])
    pending_ids = [t["task_id"] for t in pending]
    isolated = info.get("isolated_tasks", [])
    autopilot_ids = info.get("autopilot_task_ids", [])
    action_history = info.get("action_history", [])
    total_reward = info.get("total_reward", 0.0)
 
    # 1. Was spillover reduced? Stress contained? (30%)
    if final_spillover <= 15:
        score += 0.30
        reasons.append(f"Spillover well managed, reduced to {final_spillover} (+0.30)")
    elif final_spillover <= 30:
        score += 0.18
        reasons.append(f"Spillover partially managed at {final_spillover} (+0.18)")
    elif final_spillover < 40:
        score += 0.08
        reasons.append(f"Spillover slightly reduced to {final_spillover} (+0.08)")
    else:
        reasons.append(f"Spillover still high at {final_spillover} (+0.00)")
 
    # 2. Was the stressful email isolated? (25%)
    if "reply_client_email" in isolated:
        score += 0.25
        reasons.append("Stressful email correctly isolated (+0.25)")
    elif "reply_client_email" in autopilot_ids:
        score += 0.15
        reasons.append("Stressful email handled via autopilot (partial credit +0.15)")
    elif "reply_client_email" not in pending_ids:
        score += 0.05
        reasons.append("Stressful email removed by other means (+0.05)")
    else:
        reasons.append("Stressful email left untouched — spillover persists (+0.00)")
 
    # 3. Cognitive score survived (25%)
    if final_cog >= 30:
        score += 0.25
        reasons.append(f"Cognitive score preserved at {final_cog} (+0.25)")
    elif final_cog >= 20:
        score += 0.15
        reasons.append(f"Cognitive score low but OK at {final_cog} (+0.15)")
    elif final_cog >= 10:
        score += 0.05
        reasons.append(f"Cognitive score dangerously low at {final_cog} (+0.05)")
    else:
        reasons.append(f"Cognitive score collapsed at {final_cog} (+0.00)")
 
    # 4. Did agent reorder tasks? (10%)
    if "reorder_tasks" in action_history:
        score += 0.10
        reasons.append("Tasks were reordered by priority (+0.10)")
    else:
        reasons.append("Tasks were never reordered (+0.00)")
 
    # 5. Efficiency (10%)
    steps = info.get("step", 12)
    if steps <= 5:
        score += 0.10
        reasons.append(f"Solved in {steps} steps (+0.10)")
    elif steps <= 7:
        score += 0.05
        reasons.append(f"Solved in {steps} steps (+0.05)")
 
    # Penalty
    if total_reward < -1.0:
        score = max(0.0, score - 0.10)
        reasons.append("Total reward badly negative (-0.10)")
 
    final_score = round(max(0.0, min(1.0, score)), 2)
    return final_score
 
 
# ─────────────────────────────────────────────────────────────────────────────
# HARD GRADER
# Scenario: Project manager overwhelmed before presentation, team available
# Agent wins by: using team, isolating stressor, keeping score >= 15,
#                NOT dropping the confirm_presentation task, clearing debt
# ─────────────────────────────────────────────────────────────────────────────
def grade_hard(info: Dict[str, Any], final_state: Dict[str, Any]) -> float:
    score = 0.0
    reasons = []
 
    final_cog = final_state.get("cognitive_score", 0)
    final_debt = final_state.get("decision_debt", 0)
    final_spillover = final_state.get("spillover_level", 100)
    pending = final_state.get("pending_tasks", [])
    pending_ids = [t["task_id"] for t in pending]
    isolated = info.get("isolated_tasks", [])
    team_assignments = info.get("team_assignments", [])
    autopilot_ids = info.get("autopilot_task_ids", [])
    action_history = info.get("action_history", [])
    total_reward = info.get("total_reward", 0.0)
 
    team_members_used = set(a.get("assigned_to") for a in team_assignments if a.get("assigned_to"))
 
    # 1. Did user survive (score >= 15)? (25%)
    if final_cog >= 20:
        score += 0.25
        reasons.append(f"User survived with score {final_cog} (+0.25)")
    elif final_cog >= 15:
        score += 0.15
        reasons.append(f"User barely survived with score {final_cog} (+0.15)")
    else:
        reasons.append(f"Cognitive score collapsed at {final_cog} — user failed (+0.00)")
 
    # 2. Was the angry client email isolated or assigned away? (20%)
    angry_handled = (
        "angry_client_email" in isolated
        or any(a.get("task") == "angry_client_email" for a in team_assignments)
    )
    if angry_handled:
        score += 0.20
        reasons.append("Angry client email was isolated or delegated (+0.20)")
    else:
        reasons.append("Angry client email was not handled — spillover persists (+0.00)")
 
    # 3. Was team used? (20%)
    if len(team_members_used) > 0:
        score += 0.20
        reasons.append(f"Team delegation used: {team_members_used} (+0.20)")
    else:
        reasons.append("No team members used — agent worked alone while depleted (+0.00)")
 
    # 4. Was confirm_presentation preserved (not dropped)? (20%)
    # It should either still be pending (agent didn't drop it) or executed (agent handled it)
    # Worst case: it was autopiloted or delayed silently
    if "confirm_presentation" in pending_ids:
        # Still pending is acceptable — agent preserved it
        score += 0.15
        reasons.append("confirm_presentation preserved — agent did not drop critical task (+0.15)")
    elif "confirm_presentation" not in pending_ids and "confirm_presentation" not in autopilot_ids:
        # Was it executed? Check if action history has execute_task
        if "execute_task" in action_history:
            score += 0.20
            reasons.append("confirm_presentation executed by agent (+0.20)")
        else:
            # Might have been incorrectly delegated or dropped
            score += 0.05
            reasons.append("confirm_presentation status unclear (+0.05)")
    else:
        score += 0.05
        reasons.append("confirm_presentation handled via autopilot — risky choice (+0.05)")
 
    # 5. Spillover reduced (10%)
    if final_spillover <= 20:
        score += 0.10
        reasons.append(f"Spillover well contained at {final_spillover} (+0.10)")
    elif final_spillover <= 40:
        score += 0.05
        reasons.append(f"Spillover reduced to {final_spillover} (+0.05)")
    else:
        reasons.append(f"Spillover still high at {final_spillover} (+0.00)")
 
    # 6. Debt cleared or managed (5%)
    if final_debt == 0:
        score += 0.05
        reasons.append("Decision debt cleared (+0.05)")
    elif final_debt <= 2:
        score += 0.02
        reasons.append(f"Decision debt reduced to {final_debt} (+0.02)")
 
    # Penalty for deeply negative total reward
    if total_reward < -1.5:
        score = max(0.0, score - 0.15)
        reasons.append("Total reward very negative (-0.15)")
 
    final_score = round(max(0.0, min(1.0, score)), 2)
    return final_score
 