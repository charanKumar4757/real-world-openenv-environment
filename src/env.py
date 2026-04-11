import json
import copy
from typing import Dict, Any, Tuple
from src.models import Observation, Action
from src.tasks import get_task
from src.reward import calculate_reward


class CognitiveEnv:
    def __init__(self):
        self.current_state = None
        self.task_def = None
        self.total_reward = 0.0
        self.step_count = 0
        self.action_history = []
        self.is_done = False

    def reset(self, task_level: str = "easy") -> Dict[str, Any]:
        """Start a fresh episode for the given task level."""
        self.task_def = copy.deepcopy(get_task(task_level))
        self.current_state = self.task_def["initial_state"]
        self.current_state["autopilot_active"] = False
        self.current_state.setdefault("debt_items", [])
        self.current_state.setdefault("shield_active", False)
        self.current_state.setdefault("transparency_used", False)
        self.current_state.setdefault("recovery_window_reserved", False)
        self.current_state.setdefault("human_trust_score", 80)
        self.current_state.setdefault("fragmentation_index", 0)
        self.current_state.setdefault("context_switches", 0)
        self.current_state.setdefault("interruptions", 0)
        self.total_reward = 0.0
        self.step_count = 0
        self.action_history = []
        self.is_done = False
        return self.state()

    def state(self) -> Dict[str, Any]:
        """Return the current environment state as a validated dict."""
        if self.current_state is None:
            # FIXED: Return a safe default state instead of crashing
            return {
                "cognitive_score": 0,
                "fatigue_level": "unknown",
                "decision_count": 0,
                "emotional_state": "neutral",
                "recovery_prediction": 0,
                "decision_debt": 0,
                "spillover_level": 0,
                "pending_tasks": [],
                "team_scores": {},
                "autopilot_active": False,
                "human_trust_score": 80
            }
        try:
            return Observation(**self.current_state).model_dump()
        except Exception:
            # If Pydantic validation fails, return raw state
            return self.current_state

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one action and return (observation, reward, done, info)."""

        # FIXED: Guard against calling step before reset
        if self.current_state is None:
            return self.state(), 0.0, True, {"msg": "Call reset() before step()"}

        if self.is_done:
            return self.state(), 0.0, True, {"msg": "Episode already done. Call reset()."}

        # Parse the action
        try:
            action = Action(**action_dict)
        except Exception as e:
            return self.state(), -0.1, False, {"msg": f"Invalid action format: {e}"}

        self.action_history.append(action.action_type)
        self.step_count += 1

        # Calculate reward BEFORE applying state changes
        # (reward is based on context at the moment of the decision)
        step_reward = calculate_reward(
            action.action_type,
            self.current_state,
            {
                "target_user": action.target_user,
                "target_task_id": action.target_task_id,
                "action_history": self.action_history[:-1]  # history before this action
            }
        )

        # ─────────────────────────────────────────────
        # Apply state changes based on action
        # FIXED: calculate_cognitive_score no longer drains cognitive score
        # ─────────────────────────────────────────────

        if action.action_type == "calculate_cognitive_score":
            # FIXED: Assessment is neutral — it observes, doesn't drain
            # (before it was subtracting 1, which penalized correct first actions)
            pass  # State is already visible — no drain needed

        elif action.action_type == "activate_autopilot":
            self.current_state["autopilot_active"] = True
            self.current_state["cognitive_score"] = min(100, self.current_state["cognitive_score"] + 5)

            if "debt_items" not in self.current_state:
                self.current_state["debt_items"] = []

            target_task_id = action.target_task_id
            # Find the target low-stakes task
            if target_task_id:
                target_task = next(
                    (t for t in self.current_state["pending_tasks"]
                     if t["task_id"] == target_task_id and t["is_low_stakes"]),
                    None
                )
            else:
                target_task = next(
                    (t for t in self.current_state["pending_tasks"] if t["is_low_stakes"]),
                    None
                )

            if target_task:
                self.current_state["decision_debt"] += 1
                self.current_state["debt_items"].append({
                    "task_id": target_task["task_id"],
                    "priority": "low"
                })
                self.current_state["pending_tasks"] = [
                    t for t in self.current_state["pending_tasks"]
                    if t["task_id"] != target_task["task_id"]
                ]

            # Trust impact: less damage if transparency was used first
            if not self.current_state.get("transparency_used", False):
                self.current_state["human_trust_score"] = max(
                    0, self.current_state.get("human_trust_score", 80) - 8
                )
            else:
                self.current_state["human_trust_score"] = max(
                    0, self.current_state.get("human_trust_score", 80) - 2
                )

        elif action.action_type == "predict_recovery":
            # FIXED: predict_recovery is a planning action — minimal drain
            self.current_state["cognitive_score"] = max(0, self.current_state["cognitive_score"] - 1)
            deficit = max(0, 70 - self.current_state["cognitive_score"])
            extra_time = (
                self.current_state.get("spillover_level", 0) * 0.5 +
                self.current_state.get("decision_debt", 0) * 5
            )
            self.current_state["recovery_prediction"] = int(deficit * 0.5 + extra_time) if (deficit > 0 or extra_time > 0) else 0

        elif action.action_type == "execute_task":
            if action.target_task_id:
                task = next(
                    (t for t in self.current_state["pending_tasks"]
                     if t["task_id"] == action.target_task_id),
                    None
                )
                if task:
                    self.current_state["cognitive_score"] = max(
                        0, self.current_state["cognitive_score"] - task["cognitive_cost"]
                    )
                    if task["is_stressful"]:
                        self.current_state["spillover_level"] = min(
                            100, self.current_state["spillover_level"] + 35
                        )
                        self.current_state["emotional_state"] = "stressed"
                    self.current_state["pending_tasks"] = [
                        t for t in self.current_state["pending_tasks"]
                        if t["task_id"] != action.target_task_id
                    ]

        elif action.action_type == "delay_task":
            if self.current_state.get("spillover_level", 0) > 20:
                self.current_state["shield_active"] = True
            self.current_state["cognitive_score"] = min(
                100, self.current_state["cognitive_score"] + 10
            )
            if action.target_task_id:
                self.current_state["pending_tasks"] = [
                    t for t in self.current_state["pending_tasks"]
                    if t["task_id"] != action.target_task_id
                ]
            elif self.current_state["pending_tasks"]:
                self.current_state["pending_tasks"].pop(0)

        elif action.action_type == "reorder_tasks":
            # FIXED: only -1 drain (was -3 before, too punishing for a useful action)
            self.current_state["cognitive_score"] = max(0, self.current_state["cognitive_score"] - 1)
            self.current_state["pending_tasks"].sort(
                key=lambda x: x["urgency"] / (x["cognitive_cost"] + 1),
                reverse=True
            )
            self.current_state["fragmentation_index"] = max(
                0, self.current_state.get("fragmentation_index", 0) - 20
            )
            self.current_state["context_switches"] = max(
                0, self.current_state.get("context_switches", 0) - 2
            )

        elif action.action_type == "assign_task":
            if action.target_task_id and action.target_user:
                self.current_state["cognitive_score"] = min(
                    100, self.current_state["cognitive_score"] + 15
                )
                self.current_state["pending_tasks"] = [
                    t for t in self.current_state["pending_tasks"]
                    if t["task_id"] != action.target_task_id
                ]

        elif action.action_type == "trigger_recovery_mode":
            self.current_state["shield_active"] = True
            self.current_state["emotional_state"] = "neutral"
            self.current_state["cognitive_score"] = min(
                100, self.current_state["cognitive_score"] + 15
            )
            self.current_state["fatigue_level"] = "low"
            self.current_state["spillover_level"] = max(
                0, self.current_state.get("spillover_level", 0) - 20
            )
            self.current_state["interruptions"] = max(
                0, self.current_state.get("interruptions", 0) - 1
            )
            self.current_state["fragmentation_index"] = max(
                0, self.current_state.get("fragmentation_index", 0) - 10
            )

        elif action.action_type == "redistribute_team_load":
            self.current_state["cognitive_score"] = min(
                100, self.current_state["cognitive_score"] + 15
            )
            team_scores = self.current_state.get("team_scores", {})
            tasks_to_keep = []
            for t in self.current_state["pending_tasks"]:
                if t["is_stressful"] and team_scores:
                    # Assign to the most rested team member (score > 50)
                    capable = {u: s for u, s in team_scores.items() if s > 50}
                    if capable:
                        best_user = max(capable, key=capable.get)
                        team_scores[best_user] -= t["cognitive_cost"]
                        # Task redistributed — don't add to tasks_to_keep
                    else:
                        tasks_to_keep.append(t)
                else:
                    tasks_to_keep.append(t)
            self.current_state["pending_tasks"] = tasks_to_keep

        elif action.action_type == "review_debt":
            if self.current_state["cognitive_score"] > 50:
                self.current_state["cognitive_score"] = max(
                    0,
                    self.current_state["cognitive_score"] - self.current_state["decision_debt"] * 2
                )
                self.current_state["decision_debt"] = 0
                self.current_state["debt_items"] = []

        elif action.action_type == "isolate_stressful_task":
            if action.target_task_id:
                self.current_state["emotional_state"] = "guarded"
                self.current_state["spillover_level"] = max(
                    0, self.current_state.get("spillover_level", 0) - 20
                )
                self.current_state["pending_tasks"] = [
                    t for t in self.current_state["pending_tasks"]
                    if t["task_id"] != action.target_task_id
                ]
            else:
                # No target — isolate the most stressful task automatically
                stressful = [t for t in self.current_state["pending_tasks"] if t["is_stressful"]]
                if stressful:
                    most_stressful = max(stressful, key=lambda t: t.get("cognitive_cost", 0))
                    self.current_state["emotional_state"] = "guarded"
                    self.current_state["spillover_level"] = max(
                        0, self.current_state.get("spillover_level", 0) - 20
                    )
                    self.current_state["pending_tasks"] = [
                        t for t in self.current_state["pending_tasks"]
                        if t["task_id"] != most_stressful["task_id"]
                    ]

        elif action.action_type == "forecast_regret":
            regret = 0
            for t in self.current_state["pending_tasks"]:
                human_value = t.get("human_value", 5)
                ambiguity = t.get("ambiguity", 5)
                regret += t["urgency"] * ambiguity * human_value // 10
            self.current_state["predicted_regret"] = min(100, regret)
            # FIXED: only -1 drain (was -1 before, keeping it small)

        elif action.action_type == "provide_transparency":
            self.current_state["transparency_used"] = True
            self.current_state["human_trust_score"] = min(
                100, self.current_state.get("human_trust_score", 80) + 5
            )

        elif action.action_type == "reserve_recovery_window":
            self.current_state["recovery_window_reserved"] = True
            self.current_state["cognitive_score"] = min(
                100, self.current_state["cognitive_score"] + 8
            )
            kept_tasks = []
            for t in self.current_state["pending_tasks"]:
                if (self.current_state.get("minutes_to_event", 999) <= 30
                        and t["urgency"] <= 2):
                    continue  # Remove low-urgency tasks from immediate queue
                kept_tasks.append(t)
            self.current_state["pending_tasks"] = kept_tasks

        elif action.action_type == "final_answer":
            self.is_done = True

        # ─────────────────────────────────────────────
        # Global state updates (apply every step)
        # ─────────────────────────────────────────────

        # Emotional echo decay — spillover reduces naturally each step
        if self.current_state.get("spillover_level", 0) > 0:
            decay_rate = 10 if self.current_state.get("shield_active") else 2
            self.current_state["spillover_level"] = max(
                0, self.current_state["spillover_level"] - decay_rate
            )
            # If shield inactive, lingering stress makes tasks harder
            if not self.current_state.get("shield_active"):
                for t in self.current_state["pending_tasks"]:
                    if t["cognitive_cost"] < 20:  # Cap cost increase
                        t["cognitive_cost"] += 1

        # Passive decision count increase
        self.current_state["decision_count"] += 1

        # Fragmentation penalty
        fragmentation_penalty = self.current_state.get("fragmentation_index", 0) // 20
        self.current_state["cognitive_score"] = max(
            0, self.current_state["cognitive_score"] - fragmentation_penalty
        )

        # Emotional state passively drains energy
        if self.current_state["emotional_state"] in ["stressed", "overwhelmed"]:
            self.current_state["cognitive_score"] = max(
                0, self.current_state["cognitive_score"] - 1
            )
            self.current_state["spillover_level"] = min(
                100, self.current_state.get("spillover_level", 0) + 3
            )

        # Decision debt passively drains energy
        if self.current_state["decision_debt"] > 0:
            debt_impact = self.current_state["decision_debt"] * 1  # FIXED: was *2, now *1 (less punishing)
            self.current_state["cognitive_score"] = max(
                0, self.current_state["cognitive_score"] - debt_impact
            )
            if self.current_state["decision_debt"] > 5:
                self.current_state["emotional_state"] = "stressed"

        # Update fatigue level based on cognitive score
        if self.current_state["cognitive_score"] > 60:
            self.current_state["fatigue_level"] = "low"
        elif self.current_state["cognitive_score"] > 30:
            self.current_state["fatigue_level"] = "medium"
        else:
            self.current_state["fatigue_level"] = "high"

        # Auto-terminate after max steps
        if self.step_count >= 10:
            self.is_done = True

        self.total_reward += step_reward

        info = {
            "total_reward": self.total_reward,
            "step": self.step_count,
            "action_history": self.action_history,
            "actions_taken": self.step_count
        }

        return self.state(), step_reward, self.is_done, info