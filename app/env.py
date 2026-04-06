import json
import copy
from typing import Dict, Any, Tuple
from app.models import Observation, Action
from app.tasks import get_task
from app.reward import calculate_reward

class CognitiveEnv:
    def __init__(self):
        self.current_state = None
        self.task_def = None
        self.total_reward = 0.0
        self.step_count = 0
        self.action_history = []
        self.is_done = False

    def reset(self, task_level: str = "easy") -> Dict[str, Any]:
        self.task_def = copy.deepcopy(get_task(task_level))
        self.current_state = self.task_def["initial_state"]
        self.current_state["autopilot_active"] = False
        self.total_reward = 0.0
        self.step_count = 0
        self.action_history = []
        self.is_done = False
        return self.state()

    def state(self) -> Dict[str, Any]:
        # Validate through Pydantic model and return as dict
        return Observation(**self.current_state).model_dump()

    def step(self, action_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.is_done:
            return self.state(), 0.0, True, {"msg": "Episode already done"}

        action = Action(**action_dict)
        self.action_history.append(action.action_type)
        self.step_count += 1
        
        step_reward = calculate_reward(
            action.action_type,
            self.current_state,
            {
                "target_user": action.target_user,
                "target_task_id": action.target_task_id,
                "action_history": self.action_history
            }
        )

        # Apply state changes based on action
        if action.action_type == "calculate_cognitive_score":
            self.current_state["cognitive_score"] = max(0, self.current_state["cognitive_score"] - 1)
            
        elif action.action_type == "activate_autopilot":
            self.current_state["autopilot_active"] = True
            self.current_state["cognitive_score"] = min(100, self.current_state["cognitive_score"] + 5)
            # Automatically handle only the targeted low-stakes task
            if "debt_items" not in self.current_state:
                self.current_state["debt_items"] = []
            target_task_id = action.target_task_id
            if not target_task_id:
                target_task = next((t for t in self.current_state["pending_tasks"] if t["is_low_stakes"]), None)
            else:
                target_task = next((t for t in self.current_state["pending_tasks"] if t["task_id"] == target_task_id and t["is_low_stakes"]), None)
            if target_task is None:
                target_task = next((t for t in self.current_state["pending_tasks"] if t["is_low_stakes"]), None)

            tasks_to_keep = []
            for t in self.current_state["pending_tasks"]:
                if target_task is not None and t["task_id"] == target_task["task_id"]:
                    self.current_state["decision_debt"] += 1
                    self.current_state["debt_items"].append({
                        "task_id": t["task_id"],
                        "priority": "low"
                    })
                else:
                    tasks_to_keep.append(t)
            self.current_state["pending_tasks"] = tasks_to_keep

            # Trust Preservation Impact
            if not self.current_state.get("transparency_used", False):
                self.current_state["human_trust_score"] = max(0, self.current_state.get("human_trust_score", 80) - 10)
            else:
                self.current_state["human_trust_score"] = max(0, self.current_state.get("human_trust_score", 80) - 2)
            
        elif action.action_type == "predict_recovery":
            self.current_state["cognitive_score"] = max(0, self.current_state["cognitive_score"] - 2)
            # Estimate how many minutes until cognitive score reaches 70 (safe threshold)
            deficit = max(0, 70 - self.current_state["cognitive_score"])
            extra_time = self.current_state.get("spillover_level", 0) * 0.5 + self.current_state.get("decision_debt", 0) * 5
            self.current_state["recovery_prediction"] = int(deficit * 0.5 + extra_time) if deficit > 0 or extra_time > 0 else 0
            
        elif action.action_type == "execute_task":
            if action.target_task_id:
                task = next((t for t in self.current_state["pending_tasks"] if t["task_id"] == action.target_task_id), None)
                if task:
                    self.current_state["cognitive_score"] = max(0, self.current_state["cognitive_score"] - task["cognitive_cost"])
                    if task["is_stressful"]:
                        self.current_state["spillover_level"] = min(100, self.current_state["spillover_level"] + 35)
                        self.current_state["emotional_state"] = "stressed"
                        
                    self.current_state["pending_tasks"] = [
                        t for t in self.current_state["pending_tasks"] if t["task_id"] != action.target_task_id
                    ]
            
        elif action.action_type == "delay_task":
            if self.current_state.get("spillover_level", 0) > 20:
                self.current_state["shield_active"] = True
                
            self.current_state["cognitive_score"] = min(100, self.current_state["cognitive_score"] + 10)
            if action.target_task_id:
                self.current_state["pending_tasks"] = [
                    t for t in self.current_state["pending_tasks"] if t["task_id"] != action.target_task_id
                ]
            else:
                if len(self.current_state["pending_tasks"]) > 0:
                    self.current_state["pending_tasks"].pop(0)

        elif action.action_type == "reorder_tasks":
            self.current_state["cognitive_score"] = max(0, self.current_state["cognitive_score"] - 3)
            # Sort by urgency vs cognitive cost gravity
            self.current_state["pending_tasks"].sort(
                key=lambda x: x["urgency"] / (x["cognitive_cost"] + 1), reverse=True
            )
            # Fragmentation Feature Update
            self.current_state["fragmentation_index"] = max(0, self.current_state.get("fragmentation_index", 0) - 20)
            self.current_state["context_switches"] = max(0, self.current_state.get("context_switches", 0) - 2)
            
        elif action.action_type == "assign_task":
            if action.target_task_id and action.target_user:
                self.current_state["cognitive_score"] = min(100, self.current_state["cognitive_score"] + 15)
                self.current_state["pending_tasks"] = [
                    t for t in self.current_state["pending_tasks"] if t["task_id"] != action.target_task_id
                ]
                
        elif action.action_type == "trigger_recovery_mode":
            self.current_state["shield_active"] = True
            
            self.current_state["emotional_state"] = "neutral"
            # Spillover is no longer instantly zeroed. Emotional echo decay will handle it.
            self.current_state["cognitive_score"] = min(100, self.current_state["cognitive_score"] + 15)
            self.current_state["fatigue_level"] = "low"
            # Fragmentation reduction
            self.current_state["interruptions"] = max(0, self.current_state.get("interruptions", 0) - 1)
            self.current_state["fragmentation_index"] = max(0, self.current_state.get("fragmentation_index", 0) - 10)
            
        elif action.action_type == "redistribute_team_load":
            self.current_state["cognitive_score"] = min(100, self.current_state["cognitive_score"] + 15)
            team_scores = self.current_state.get("team_scores", {})
            tasks_to_keep = []
            for t in self.current_state["pending_tasks"]:
                if len(team_scores) > 0 and t["is_stressful"]:
                    # Fair delegation: find a safe teammate (score > 50) rather than dumping all tasks
                    assigned = False
                    for user, score in team_scores.items():
                        if score > 50:
                            team_scores[user] -= t["cognitive_cost"]
                            assigned = True
                            break
                    if not assigned:
                        tasks_to_keep.append(t)
                else:
                    tasks_to_keep.append(t)
            self.current_state["pending_tasks"] = tasks_to_keep
            
        elif action.action_type == "review_debt":
            if self.current_state["cognitive_score"] > 50:
                self.current_state["cognitive_score"] = max(0, self.current_state["cognitive_score"] - self.current_state["decision_debt"] * 2)
                self.current_state["decision_debt"] = 0
                self.current_state["debt_items"] = []
                
        elif action.action_type == "final_answer":
            self.is_done = True
            
        elif action.action_type == "isolate_stressful_task":
            if action.target_task_id:
                self.current_state["emotional_state"] = "guarded"
                self.current_state["spillover_level"] = max(0, self.current_state.get("spillover_level", 0) - 20)
                self.current_state["pending_tasks"] = [
                    t for t in self.current_state["pending_tasks"] if t["task_id"] != action.target_task_id
                ]
                
        elif action.action_type == "apply_pattern_memory":
            self.current_state["pattern_memory_triggered"] = True
            self.current_state["cognitive_score"] = min(100, self.current_state["cognitive_score"] + 10)
            
        elif action.action_type == "simulate_stress_test":
            self.current_state["resilience_score"] = 68
            self.current_state["cognitive_score"] = max(0, self.current_state["cognitive_score"] - 2)
            
        elif action.action_type == "forecast_regret":
            regret = 0
            for t in self.current_state["pending_tasks"]:
                human_value = t.get("human_value", 5)
                ambiguity = t.get("ambiguity", 5)
                regret += t["urgency"] * ambiguity * human_value // 10
            self.current_state["predicted_regret"] = min(100, regret)
            self.current_state["cognitive_score"] = max(0, self.current_state["cognitive_score"] - 1)
            
        elif action.action_type == "provide_transparency":
            self.current_state["transparency_used"] = True
            self.current_state["human_trust_score"] = min(100, self.current_state.get("human_trust_score", 80) + 5)
            
        elif action.action_type == "reserve_recovery_window":
            self.current_state["recovery_window_reserved"] = True
            self.current_state["cognitive_score"] = min(100, self.current_state["cognitive_score"] + 8)
            # Remove only optional low-value tasks from immediate attention
            kept_tasks = []
            for t in self.current_state["pending_tasks"]:
                if self.current_state.get("minutes_to_event", 999) <= 30 and t["urgency"] <= 2:
                    continue
                kept_tasks.append(t)
            self.current_state["pending_tasks"] = kept_tasks
            
        # Emotional Echo Decay
        if self.current_state.get("spillover_level", 0) > 0:
            decay_rate = 10 if self.current_state.get("shield_active") else 2
            self.current_state["spillover_level"] = max(0, self.current_state["spillover_level"] - decay_rate)
            # If shield inactive, lingering stress contaminates pending tasks
            if not self.current_state.get("shield_active"):
                for t in self.current_state["pending_tasks"]:
                    t["cognitive_cost"] += 1

        # Passive cognitive drain per decision / action
        self.current_state["decision_count"] += 1
        
        # Fragmentation impact
        fragmentation_penalty = self.current_state.get("fragmentation_index", 0) // 20
        self.current_state["cognitive_score"] = max(0, self.current_state["cognitive_score"] - fragmentation_penalty)
        if self.current_state["emotional_state"] in ["stressed", "overwhelmed"]:
            self.current_state["cognitive_score"] = max(0, self.current_state["cognitive_score"] - 1)
            self.current_state["spillover_level"] = min(100, self.current_state.get("spillover_level", 0) + 5)
            
        # Penalty for holding too much decision debt alters performance
        if self.current_state["decision_debt"] > 0:
            debt_impact = self.current_state["decision_debt"] * 2
            self.current_state["cognitive_score"] = max(0, self.current_state["cognitive_score"] - debt_impact)
            if self.current_state["decision_debt"] > 3:
                self.current_state["emotional_state"] = "stressed"
        
        # Update fatigue level dynamically
        if self.current_state["cognitive_score"] > 60:
            self.current_state["fatigue_level"] = "low"
        elif self.current_state["cognitive_score"] > 30:
            self.current_state["fatigue_level"] = "medium"
        else:
            self.current_state["fatigue_level"] = "high"
            
        # Check termination condition
        if self.step_count >= 10:
            self.is_done = True

        self.total_reward += step_reward
        
        info = {
            "total_reward": self.total_reward,
            "step": self.step_count,
            "action_history": self.action_history
        }
        
        return self.state(), step_reward, self.is_done, info
