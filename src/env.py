import copy
from typing import Dict, Any, Tuple
from src.models import Observation, Action
from src.tasks import get_task
from src.reward import calculate_reward


def generate_situation_summary(state: Dict[str, Any]) -> str:
    """
    Generate a plain-English situation summary for the agent.
    This is the 'situation_summary' field required by Phase 3.
    It describes what is happening and what the agent should consider doing.
    """
    parts = []
    score = state.get("cognitive_score", 50)
    fatigue = state.get("fatigue_level", "medium")
    emotion = state.get("emotional_state", "neutral")
    spillover = state.get("spillover_level", 0)
    debt = state.get("decision_debt", 0)
    pending = state.get("pending_tasks", [])
    team = state.get("team_scores", {})
    minutes = state.get("minutes_to_event", 999)
    trust = state.get("human_trust_score", 80)

    # Cognitive state description
    if score < 25:
        parts.append(f"User is severely cognitively depleted (score={score}/100).")
    elif score < 50:
        parts.append(f"User is mentally fatigued (score={score}/100, fatigue={fatigue}).")
    else:
        parts.append(f"User has moderate cognitive capacity (score={score}/100).")

    # Emotional state
    if emotion in ["stressed", "overwhelmed"] or spillover > 30:
        parts.append(
            f"Emotional state is {emotion} with spillover level {spillover}/100 — "
            "recovery or isolation actions are recommended."
        )

    # Deadline pressure
    if minutes <= 60:
        parts.append(
            f"High-stakes event in {minutes} minutes — "
            "protect recovery time and delegate non-critical tasks."
        )

    # Pending task overview
    stressful = [t for t in pending if t.get("is_stressful")]
    low_stakes = [t for t in pending if t.get("is_low_stakes")]
    urgent = sorted(pending, key=lambda t: t.get("urgency", 0), reverse=True)

    if stressful:
        ids = ", ".join(t["task_id"] for t in stressful)
        parts.append(f"Stressful tasks present: {ids} — consider isolating to prevent emotional contagion.")
    if low_stakes:
        ids = ", ".join(t["task_id"] for t in low_stakes)
        parts.append(f"Low-stakes tasks ({ids}) can be safely automated via autopilot.")
    if urgent:
        top = urgent[0]
        parts.append(f"Highest urgency task: {top['task_id']} (urgency={top.get('urgency', '?')}).")

    # Decision debt
    if debt > 0:
        parts.append(
            f"Decision debt is {debt} — "
            f"{'review when score recovers above 50' if score < 50 else 'safe to review now'}."
        )

    # Team availability
    capable = {u: s for u, s in team.items() if u != "User" and s > 50}
    if capable and score < 40:
        best = max(capable, key=capable.get)
        parts.append(
            f"Team member {best} has cognitive score {capable[best]}/100 and can handle delegation."
        )

    # Trust warning
    if trust < 50:
        parts.append(
            "Human trust score is low — use provide_transparency before automating further."
        )

    # Recommended action hint
    if score < 30 and emotion in ["stressed", "overwhelmed"]:
        parts.append("Recommended: trigger_recovery_mode first, then isolate stressful tasks.")
    elif low_stakes and score < 50:
        parts.append("Recommended: activate_autopilot for low-stakes tasks to conserve energy.")
    elif stressful:
        parts.append("Recommended: isolate_stressful_task to prevent emotional spillover.")
    elif not pending:
        parts.append("All tasks handled — use final_answer to end the episode.")

    return " ".join(parts)


class CognitiveEnv:
    def __init__(self):
        self.current_state = None
        self.task_def = None
        self.total_reward = 0.0
        self.step_count = 0
        self.action_history = []
        self.is_done = False
        # Track actual outcome changes for real grading
        self.initial_cognitive_score = 0
        self.stressful_tasks_resolved = 0
        self.low_stakes_automated = 0

    def reset(self, task_level: str = "easy") -> Dict[str, Any]:
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
        self.current_state.setdefault("spillover_level", 0)
        self.current_state.setdefault("recovery_prediction", 0)
        self.current_state.setdefault("minutes_to_event", 999)
        self.current_state.setdefault("predicted_regret", 0)
        self.current_state.setdefault("resilience_score", 0)
        self.current_state.setdefault("pattern_memory_triggered", False)
        self.total_reward = 0.0
        self.step_count = 0
        self.action_history = []
        self.is_done = False
        self.initial_cognitive_score = self.current_state["cognitive_score"]
        self.stressful_tasks_resolved = 0
        self.low_stakes_automated = 0
        return self.state()

    def state(self) -> Dict[str, Any]:
        if self.current_state is None:
            return {
                "cognitive_score": 0, "fatigue_level": "unknown",
                "decision_count": 0, "emotional_state": "neutral",
                "recovery_prediction": 0, "decision_debt": 0,
                "spillover_level": 0, "pending_tasks": [], "team_scores": {},
                "autopilot_active": False, "human_trust_score": 80,
                "situation_summary": "Environment not initialized. Call reset() first."
            }
        # Generate fresh situation summary every time state is called
        self.current_state["situation_summary"] = generate_situation_summary(self.current_state)
        try:
            return Observation(**self.current_state).model_dump()
        except Exception:
            return self.current_state

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.current_state is None:
            return self.state(), 0.0, True, {"msg": "Call reset() before step()"}
        if self.is_done:
            return self.state(), 0.0, True, {"msg": "Episode done. Call reset()."}

        try:
            action = Action(**action_dict)
        except Exception as e:
            return self.state(), -0.1, False, {"msg": f"Invalid action: {e}"}

        self.action_history.append(action.action_type)
        self.step_count += 1

        # Calculate reward BEFORE state changes
        step_reward = calculate_reward(
            action.action_type,
            self.current_state,
            {
                "target_user": action.target_user,
                "target_task_id": action.target_task_id,
                "action_history": self.action_history[:-1]
            }
        )

        # ── Apply state changes ───────────────────────────────────

        if action.action_type == "calculate_cognitive_score":
            pass  # Pure observation — no drain

        elif action.action_type == "predict_recovery":
            self.current_state["cognitive_score"] = max(0, self.current_state["cognitive_score"] - 1)
            deficit = max(0, 70 - self.current_state["cognitive_score"])
            extra = (self.current_state.get("spillover_level", 0) * 0.5 +
                     self.current_state.get("decision_debt", 0) * 5)
            self.current_state["recovery_prediction"] = int(deficit * 0.5 + extra)

        elif action.action_type == "activate_autopilot":
            self.current_state["autopilot_active"] = True
            self.current_state["cognitive_score"] = min(100, self.current_state["cognitive_score"] + 5)
            if "debt_items" not in self.current_state:
                self.current_state["debt_items"] = []
            tid = action.target_task_id
            target = next(
                (t for t in self.current_state["pending_tasks"]
                 if (t["task_id"] == tid if tid else t["is_low_stakes"])),
                next((t for t in self.current_state["pending_tasks"] if t["is_low_stakes"]), None)
            )
            if target:
                self.current_state["decision_debt"] += 1
                self.current_state["debt_items"].append({"task_id": target["task_id"], "priority": "low"})
                self.current_state["pending_tasks"] = [
                    t for t in self.current_state["pending_tasks"]
                    if t["task_id"] != target["task_id"]
                ]
                self.low_stakes_automated += 1
            trust_drain = 2 if self.current_state.get("transparency_used") else 8
            self.current_state["human_trust_score"] = max(
                0, self.current_state.get("human_trust_score", 80) - trust_drain
            )

        elif action.action_type == "trigger_recovery_mode":
            self.current_state["shield_active"] = True
            self.current_state["emotional_state"] = "neutral"
            self.current_state["cognitive_score"] = min(100, self.current_state["cognitive_score"] + 15)
            self.current_state["fatigue_level"] = "low"
            self.current_state["spillover_level"] = max(
                0, self.current_state.get("spillover_level", 0) - 20
            )
            self.current_state["fragmentation_index"] = max(
                0, self.current_state.get("fragmentation_index", 0) - 10
            )

        elif action.action_type == "isolate_stressful_task":
            tid = action.target_task_id
            stressful = [t for t in self.current_state["pending_tasks"] if t["is_stressful"]]
            if tid:
                target = next((t for t in stressful if t["task_id"] == tid), None)
            else:
                target = max(stressful, key=lambda t: t.get("cognitive_cost", 0)) if stressful else None
            if target:
                self.current_state["emotional_state"] = "guarded"
                self.current_state["spillover_level"] = max(
                    0, self.current_state.get("spillover_level", 0) - 20
                )
                self.current_state["pending_tasks"] = [
                    t for t in self.current_state["pending_tasks"]
                    if t["task_id"] != target["task_id"]
                ]
                self.stressful_tasks_resolved += 1

        elif action.action_type == "reorder_tasks":
            self.current_state["cognitive_score"] = max(0, self.current_state["cognitive_score"] - 1)
            self.current_state["pending_tasks"].sort(
                key=lambda x: x["urgency"] / (x["cognitive_cost"] + 1), reverse=True
            )
            self.current_state["fragmentation_index"] = max(
                0, self.current_state.get("fragmentation_index", 0) - 20
            )

        elif action.action_type == "redistribute_team_load":
            self.current_state["cognitive_score"] = min(100, self.current_state["cognitive_score"] + 15)
            team = self.current_state.get("team_scores", {})
            kept = []
            for t in self.current_state["pending_tasks"]:
                if t["is_stressful"]:
                    capable = {u: s for u, s in team.items() if s > 50}
                    if capable:
                        best = max(capable, key=capable.get)
                        team[best] -= t["cognitive_cost"]
                        self.stressful_tasks_resolved += 1
                    else:
                        kept.append(t)
                else:
                    kept.append(t)
            self.current_state["pending_tasks"] = kept

        elif action.action_type == "delay_task":
            self.current_state["cognitive_score"] = min(100, self.current_state["cognitive_score"] + 10)
            if self.current_state.get("spillover_level", 0) > 20:
                self.current_state["shield_active"] = True
            if action.target_task_id:
                self.current_state["pending_tasks"] = [
                    t for t in self.current_state["pending_tasks"]
                    if t["task_id"] != action.target_task_id
                ]
            elif self.current_state["pending_tasks"]:
                self.current_state["pending_tasks"].pop(0)

        elif action.action_type == "review_debt":
            if self.current_state["cognitive_score"] > 50:
                self.current_state["cognitive_score"] = max(
                    0,
                    self.current_state["cognitive_score"] - self.current_state["decision_debt"] * 2
                )
                self.current_state["decision_debt"] = 0
                self.current_state["debt_items"] = []

        elif action.action_type == "forecast_regret":
            regret = sum(
                t["urgency"] * t.get("ambiguity", 5) * t.get("human_value", 5) // 10
                for t in self.current_state["pending_tasks"]
            )
            self.current_state["predicted_regret"] = min(100, regret)

        elif action.action_type == "provide_transparency":
            self.current_state["transparency_used"] = True
            self.current_state["human_trust_score"] = min(
                100, self.current_state.get("human_trust_score", 80) + 5
            )

        elif action.action_type == "reserve_recovery_window":
            self.current_state["recovery_window_reserved"] = True
            self.current_state["cognitive_score"] = min(100, self.current_state["cognitive_score"] + 8)
            self.current_state["pending_tasks"] = [
                t for t in self.current_state["pending_tasks"]
                if not (self.current_state.get("minutes_to_event", 999) <= 30 and t["urgency"] <= 2)
            ]

        elif action.action_type == "final_answer":
            self.is_done = True

        # ── Global updates every step ─────────────────────────────

        if self.current_state.get("spillover_level", 0) > 0:
            decay = 10 if self.current_state.get("shield_active") else 2
            self.current_state["spillover_level"] = max(0, self.current_state["spillover_level"] - decay)
            if not self.current_state.get("shield_active"):
                for t in self.current_state["pending_tasks"]:
                    if t["cognitive_cost"] < 20:
                        t["cognitive_cost"] += 1

        self.current_state["decision_count"] += 1

        frag = self.current_state.get("fragmentation_index", 0) // 20
        self.current_state["cognitive_score"] = max(0, self.current_state["cognitive_score"] - frag)

        if self.current_state["emotional_state"] in ["stressed", "overwhelmed"]:
            self.current_state["cognitive_score"] = max(0, self.current_state["cognitive_score"] - 1)
            self.current_state["spillover_level"] = min(
                100, self.current_state.get("spillover_level", 0) + 3
            )

        if self.current_state["decision_debt"] > 0:
            self.current_state["cognitive_score"] = max(
                0, self.current_state["cognitive_score"] - self.current_state["decision_debt"]
            )
            if self.current_state["decision_debt"] > 5:
                self.current_state["emotional_state"] = "stressed"

        score = self.current_state["cognitive_score"]
        if score > 60:
            self.current_state["fatigue_level"] = "low"
        elif score > 30:
            self.current_state["fatigue_level"] = "medium"
        else:
            self.current_state["fatigue_level"] = "high"

        if self.step_count >= 10:
            self.is_done = True

        self.total_reward += step_reward

        info = {
            "total_reward": self.total_reward,
            "step": self.step_count,
            "action_history": self.action_history,
            "actions_taken": self.step_count,
            # Real outcome tracking for graders
            "initial_cognitive_score": self.initial_cognitive_score,
            "final_cognitive_score": self.current_state["cognitive_score"],
            "cognitive_score_change": self.current_state["cognitive_score"] - self.initial_cognitive_score,
            "stressful_tasks_resolved": self.stressful_tasks_resolved,
            "low_stakes_automated": self.low_stakes_automated,
            "decision_debt": self.current_state["decision_debt"],
            "human_trust_score": self.current_state.get("human_trust_score", 80),
            "emotional_state": self.current_state["emotional_state"],
            "pending_tasks_remaining": len(self.current_state.get("pending_tasks", [])),
        }

        return self.state(), step_reward, self.is_done, info