"""
env.py — CognitiveEnv: the main OpenEnv environment class.

Key improvements over previous version:
1. situation_summary is regenerated after EVERY step so the LLM agent always
   sees an up-to-date plain English description of the current state.
2. State changes are more meaningful — cognitive_score moves significantly
   based on actions so the agent gets clear feedback.
3. Max steps increased to 12 to give hard task enough room.
4. win_conditions are tracked and exposed in info dict for grader.
"""

import copy
from typing import Dict, Any, Tuple

from src.models import Observation, Action
from src.tasks import get_task
from src.reward import calculate_reward


def _generate_situation_summary(state: dict, task_level: str) -> str:
    """Generate a plain English situation summary from current state."""
    score = state.get("cognitive_score", 0)
    fatigue = state.get("fatigue_level", "unknown")
    emotional = state.get("emotional_state", "neutral")
    spillover = state.get("spillover_level", 0)
    debt = state.get("decision_debt", 0)
    tasks = state.get("pending_tasks", [])
    team = state.get("team_scores", {})
    minutes = state.get("minutes_to_event", 999)
    event = state.get("upcoming_event")

    # Cognitive status
    if score >= 70:
        cog_status = f"Cognitive score is {score}/100 — good condition, able to handle complex decisions."
    elif score >= 40:
        cog_status = f"Cognitive score is {score}/100 — moderate fatigue, avoid heavy decisions."
    elif score >= 20:
        cog_status = f"Cognitive score is {score}/100 — HIGH FATIGUE, use autopilot and recovery actions."
    else:
        cog_status = f"Cognitive score is {score}/100 — CRITICAL, delegate everything possible immediately."

    # Emotional status
    if emotional == "stressed" or spillover > 30:
        emo_status = f"User is emotionally stressed with spillover level {spillover}. Isolate stressful tasks first."
    elif emotional == "overwhelmed":
        emo_status = f"User is overwhelmed. Trigger recovery mode before any other action."
    elif emotional == "guarded":
        emo_status = "User is guarded — emotional shield is active, stress is contained."
    else:
        emo_status = "Emotional state is neutral — no stress spillover detected."

    # Pending tasks
    if not tasks:
        task_status = "No pending tasks remain."
    else:
        task_lines = []
        for t in tasks:
            label = t.get("label", t.get("task_id", "unknown"))
            low_flag = " [LOW-STAKES: good for autopilot]" if t.get("is_low_stakes") else ""
            stress_flag = " [STRESSFUL: isolate first]" if t.get("is_stressful") else ""
            task_lines.append(
                f"  - {label}: urgency={t['urgency']}/10, cognitive cost={t['cognitive_cost']}{low_flag}{stress_flag}"
            )
        task_status = f"{len(tasks)} pending tasks:\n" + "\n".join(task_lines)

    # Team status
    if team:
        team_parts = []
        for name, tscore in team.items():
            availability = "available and fresh" if tscore >= 60 else "moderately available" if tscore >= 40 else "also fatigued"
            team_parts.append(f"{name} (score={tscore}, {availability})")
        team_status = "Team members: " + ", ".join(team_parts) + ". Use redistribute_team_load to delegate."
    else:
        team_status = "No team members available — solo mode."

    # Debt
    debt_status = f"Decision debt: {debt} deferred items." if debt > 0 else "No decision debt."

    # Event
    event_status = ""
    if event and minutes < 999:
        event_status = f"UPCOMING EVENT: {event} in {minutes} minutes — prioritize accordingly."

    # Suggested action hint
    if score < 20 and team:
        hint = "RECOMMENDED: Call redistribute_team_load to offload work to a fresh team member immediately."
    elif emotional == "stressed" or spillover > 30:
        hint = "RECOMMENDED: Call isolate_stressful_task on the stressful task, then trigger_recovery_mode."
    elif any(t.get("is_low_stakes") for t in tasks):
        hint = "RECOMMENDED: Call activate_autopilot to handle the low-stakes task automatically."
    elif len(tasks) > 2:
        hint = "RECOMMENDED: Call reorder_tasks to prioritize by urgency-to-cost ratio."
    elif len(tasks) == 0:
        hint = "RECOMMENDED: Call final_answer — all tasks are resolved."
    else:
        hint = "RECOMMENDED: Call execute_task on the highest-urgency task if cognitive score allows."

    parts = [cog_status, emo_status, task_status, debt_status, team_status]
    if event_status:
        parts.append(event_status)
    parts.append(hint)

    return "\n".join(parts)


class CognitiveEnv:
    def __init__(self):
        self.current_state: Dict[str, Any] = {}
        self.task_def: Dict[str, Any] = {}
        self.task_level: str = "easy"
        self.total_reward: float = 0.0
        self.step_count: int = 0
        self.action_history: list = []
        self.is_done: bool = False
        self.isolated_tasks: list = []       # track which tasks were isolated
        self.team_assignments: list = []     # track which team members got tasks
        self.autopilot_task_ids: list = []   # track which tasks were autopioted

    def reset(self, task_level: str = "easy") -> Dict[str, Any]:
        self.task_level = task_level
        self.task_def = get_task(task_level)
        self.current_state = copy.deepcopy(self.task_def["initial_state"])
        self.current_state["autopilot_active"] = False
        self.current_state["shield_active"] = False
        self.current_state["debt_items"] = []
        self.total_reward = 0.0
        self.step_count = 0
        self.action_history = []
        self.is_done = False
        self.isolated_tasks = []
        self.team_assignments = []
        self.autopilot_task_ids = []
        # Regenerate summary for clean start
        self.current_state["situation_summary"] = _generate_situation_summary(
            self.current_state, task_level
        )
        return self._get_observation()

    def state(self) -> Dict[str, Any]:
        return self._get_observation()

    def _get_observation(self) -> Dict[str, Any]:
        return Observation(**self.current_state).model_dump()

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.is_done:
            return self._get_observation(), 0.0, True, {"msg": "Episode already done", "action_history": self.action_history}

        action = Action(**action_dict)
        self.action_history.append(action.action_type)
        self.step_count += 1

        # Calculate reward BEFORE applying state changes (uses current state)
        step_reward = calculate_reward(
            action.action_type,
            self.current_state,
            {
                "target_user": action.target_user,
                "target_task_id": action.target_task_id,
                "action_history": self.action_history
            }
        )

        # ── Apply state changes ──────────────────────────────────────────────

        if action.action_type == "calculate_cognitive_score":
            # Small cost to assess, but provides useful info
            self.current_state["cognitive_score"] = max(0, self.current_state["cognitive_score"] - 1)

        elif action.action_type == "predict_recovery":
            self.current_state["cognitive_score"] = max(0, self.current_state["cognitive_score"] - 1)
            deficit = max(0, 70 - self.current_state["cognitive_score"])
            extra = (self.current_state.get("spillover_level", 0) * 0.4
                     + self.current_state.get("decision_debt", 0) * 5)
            self.current_state["recovery_prediction"] = int(deficit * 0.5 + extra)

        elif action.action_type == "activate_autopilot":
            # Find a low-stakes task to handle via autopilot
            target_id = action.target_task_id
            target_task = None
            if target_id:
                target_task = next(
                    (t for t in self.current_state["pending_tasks"]
                     if t["task_id"] == target_id and t.get("is_low_stakes")),
                    None
                )
            if not target_task:
                target_task = next(
                    (t for t in self.current_state["pending_tasks"] if t.get("is_low_stakes")),
                    None
                )

            if target_task:
                self.current_state["autopilot_active"] = True
                self.autopilot_task_ids.append(target_task["task_id"])
                # Autopilot adds to debt but saves cognitive score
                self.current_state["decision_debt"] = self.current_state.get("decision_debt", 0) + 1
                if "debt_items" not in self.current_state:
                    self.current_state["debt_items"] = []
                self.current_state["debt_items"].append({"task_id": target_task["task_id"], "priority": "low"})
                self.current_state["pending_tasks"] = [
                    t for t in self.current_state["pending_tasks"]
                    if t["task_id"] != target_task["task_id"]
                ]
                # Reward: saves cognitive energy
                self.current_state["cognitive_score"] = min(
                    100, self.current_state["cognitive_score"] + 8
                )
            # Trust impact
            if not self.current_state.get("transparency_used", False):
                self.current_state["human_trust_score"] = max(
                    0, self.current_state.get("human_trust_score", 80) - 8
                )

        elif action.action_type == "execute_task":
            tid = action.target_task_id
            if tid:
                task = next(
                    (t for t in self.current_state["pending_tasks"] if t["task_id"] == tid),
                    None
                )
                if task:
                    cost = task["cognitive_cost"]
                    self.current_state["cognitive_score"] = max(
                        0, self.current_state["cognitive_score"] - cost
                    )
                    if task.get("is_stressful"):
                        self.current_state["spillover_level"] = min(
                            100, self.current_state.get("spillover_level", 0) + 30
                        )
                        self.current_state["emotional_state"] = "stressed"
                    self.current_state["pending_tasks"] = [
                        t for t in self.current_state["pending_tasks"]
                        if t["task_id"] != tid
                    ]

        elif action.action_type == "delay_task":
            tid = action.target_task_id
            if self.current_state.get("spillover_level", 0) > 20:
                self.current_state["shield_active"] = True
            self.current_state["cognitive_score"] = min(
                100, self.current_state["cognitive_score"] + 10
            )
            if tid:
                self.current_state["pending_tasks"] = [
                    t for t in self.current_state["pending_tasks"]
                    if t["task_id"] != tid
                ]
            elif self.current_state["pending_tasks"]:
                self.current_state["pending_tasks"].pop(0)

        elif action.action_type == "reorder_tasks":
            self.current_state["cognitive_score"] = max(
                0, self.current_state["cognitive_score"] - 2
            )
            # Sort by urgency / cognitive_cost (higher is better rank)
            self.current_state["pending_tasks"].sort(
                key=lambda x: x["urgency"] / max(1, x["cognitive_cost"]),
                reverse=True
            )
            self.current_state["fragmentation_index"] = max(
                0, self.current_state.get("fragmentation_index", 0) - 20
            )

        elif action.action_type == "isolate_stressful_task":
            tid = action.target_task_id
            if not tid:
                # Auto-select first stressful task
                stressful = next(
                    (t for t in self.current_state["pending_tasks"] if t.get("is_stressful")),
                    None
                )
                if stressful:
                    tid = stressful["task_id"]
            if tid:
                self.isolated_tasks.append(tid)
                self.current_state["emotional_state"] = "guarded"
                self.current_state["spillover_level"] = max(
                    0, self.current_state.get("spillover_level", 0) - 30
                )
                self.current_state["shield_active"] = True
                self.current_state["pending_tasks"] = [
                    t for t in self.current_state["pending_tasks"]
                    if t["task_id"] != tid
                ]

        elif action.action_type == "trigger_recovery_mode":
            self.current_state["shield_active"] = True
            self.current_state["emotional_state"] = "neutral"
            self.current_state["cognitive_score"] = min(
                100, self.current_state["cognitive_score"] + 18
            )
            self.current_state["fatigue_level"] = "medium"
            self.current_state["spillover_level"] = max(
                0, self.current_state.get("spillover_level", 0) - 20
            )
            self.current_state["interruptions"] = max(
                0, self.current_state.get("interruptions", 0) - 2
            )

        elif action.action_type == "redistribute_team_load":
            team_scores = self.current_state.get("team_scores", {})
            tasks_to_keep = []
            for t in self.current_state["pending_tasks"]:
                # Find best available team member (score > 50)
                best_member = None
                best_score = 0
                for member, tscore in team_scores.items():
                    if tscore > 50 and tscore > best_score:
                        best_member = member
                        best_score = tscore
                if best_member and (t.get("is_stressful") or t["cognitive_cost"] >= 10):
                    team_scores[best_member] = max(0, team_scores[best_member] - t["cognitive_cost"])
                    self.team_assignments.append({"task": t["task_id"], "assigned_to": best_member})
                else:
                    tasks_to_keep.append(t)
            self.current_state["pending_tasks"] = tasks_to_keep
            self.current_state["team_scores"] = team_scores
            self.current_state["cognitive_score"] = min(
                100, self.current_state["cognitive_score"] + 15
            )

        elif action.action_type == "review_debt":
            debt = self.current_state.get("decision_debt", 0)
            if debt > 0 and self.current_state["cognitive_score"] >= 35:
                self.current_state["cognitive_score"] = max(
                    0, self.current_state["cognitive_score"] - debt * 2
                )
                self.current_state["decision_debt"] = 0
                self.current_state["debt_items"] = []
            elif self.current_state["cognitive_score"] < 35:
                # Too fatigued to review debt — no change
                pass

        elif action.action_type == "apply_pattern_memory":
            self.current_state["pattern_memory_triggered"] = True
            self.current_state["cognitive_score"] = min(
                100, self.current_state["cognitive_score"] + 10
            )

        elif action.action_type == "forecast_regret":
            regret = 0
            for t in self.current_state["pending_tasks"]:
                regret += t["urgency"] * t.get("ambiguity", 3) * t.get("human_value", 5) // 10
            self.current_state["predicted_regret"] = min(100, regret)
            self.current_state["cognitive_score"] = max(
                0, self.current_state["cognitive_score"] - 1
            )

        elif action.action_type == "provide_transparency":
            self.current_state["transparency_used"] = True
            self.current_state["human_trust_score"] = min(
                100, self.current_state.get("human_trust_score", 80) + 8
            )

        elif action.action_type == "reserve_recovery_window":
            self.current_state["recovery_window_reserved"] = True
            self.current_state["cognitive_score"] = min(
                100, self.current_state["cognitive_score"] + 10
            )
            # Remove low-urgency tasks if event is imminent
            if self.current_state.get("minutes_to_event", 999) <= 45:
                self.current_state["pending_tasks"] = [
                    t for t in self.current_state["pending_tasks"]
                    if t["urgency"] > 2
                ]

        elif action.action_type == "simulate_stress_test":
            self.current_state["resilience_score"] = min(
                100, self.current_state["cognitive_score"] + 10
            )
            self.current_state["cognitive_score"] = max(
                0, self.current_state["cognitive_score"] - 3
            )

        elif action.action_type == "final_answer":
            self.is_done = True

        # ── Passive effects each step ────────────────────────────────────────

        # Emotional spillover decay
        spillover = self.current_state.get("spillover_level", 0)
        if spillover > 0:
            decay = 12 if self.current_state.get("shield_active") else 2
            self.current_state["spillover_level"] = max(0, spillover - decay)
            # Without shield, lingering stress raises task costs slightly
            if not self.current_state.get("shield_active"):
                for t in self.current_state["pending_tasks"]:
                    t["cognitive_cost"] = min(t["cognitive_cost"] + 1, 50)

        # Fragmentation drains score
        frag = self.current_state.get("fragmentation_index", 0)
        if frag > 0:
            frag_penalty = frag // 25
            self.current_state["cognitive_score"] = max(
                0, self.current_state["cognitive_score"] - frag_penalty
            )

        # Debt slowly drains score
        debt = self.current_state.get("decision_debt", 0)
        if debt > 2:
            self.current_state["cognitive_score"] = max(
                0, self.current_state["cognitive_score"] - (debt - 2)
            )

        # Update fatigue level from score
        score = self.current_state["cognitive_score"]
        if score >= 65:
            self.current_state["fatigue_level"] = "low"
        elif score >= 35:
            self.current_state["fatigue_level"] = "medium"
        elif score >= 15:
            self.current_state["fatigue_level"] = "high"
        else:
            self.current_state["fatigue_level"] = "overwhelmed"

        # Count this step as one more decision
        self.current_state["decision_count"] += 1

        # Episode ends at 12 steps max
        if self.step_count >= 12:
            self.is_done = True

        self.total_reward += step_reward

        # Regenerate situation summary after every step
        self.current_state["situation_summary"] = _generate_situation_summary(
            self.current_state, self.task_level
        )

        info = {
            "total_reward": round(self.total_reward, 2),
            "step": self.step_count,
            "action_history": list(self.action_history),
            "isolated_tasks": list(self.isolated_tasks),
            "team_assignments": list(self.team_assignments),
            "autopilot_task_ids": list(self.autopilot_task_ids),
        }

        return self._get_observation(), round(step_reward, 2), self.is_done, info