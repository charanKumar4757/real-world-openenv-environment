class CognitiveEnv:
    def __init__(self):
        self.state_data = {}
        self.task_level = "easy"
        self.step_count = 0
        self.action_history = []

    def reset(self, task_level="easy"):
        self.task_level = task_level
        self.step_count = 0
        self.action_history = []

        if task_level == "easy":
            self.state_data = {
                "cognitive_score": 45,
                "fatigue_level": "medium",
                "decision_count": 20,
                "emotional_state": "neutral",
                "recovery_prediction": 30,
                "decision_debt": 0,
                "spillover_level": 20,
                "autopilot_active": False,
                "human_trust_score": 75,
                "team_scores": {"User": 45, "Sara": 88, "Mike": 60},
                "pending_tasks": [
                    {
                        "task_id": "elective_choice",
                        "urgency": 8,
                        "cognitive_cost": 6,
                        "is_low_stakes": False,
                        "is_stressful": False
                    },
                    {
                        "task_id": "lunch_order",
                        "urgency": 2,
                        "cognitive_cost": 1,
                        "is_low_stakes": True,
                        "is_stressful": False
                    },
                ]
            }

        elif task_level == "medium":
            self.state_data = {
                "cognitive_score": 30,
                "fatigue_level": "high",
                "decision_count": 45,
                "emotional_state": "stressed",
                "recovery_prediction": 45,
                "decision_debt": 2,
                "spillover_level": 55,
                "autopilot_active": False,
                "human_trust_score": 65,
                "team_scores": {"User": 30, "Sara": 88, "Mike": 60},
                "pending_tasks": [
                    {
                        "task_id": "complex_report",
                        "urgency": 7,
                        "cognitive_cost": 8,
                        "is_low_stakes": False,
                        "is_stressful": True
                    },
                    {
                        "task_id": "team_meeting",
                        "urgency": 5,
                        "cognitive_cost": 4,
                        "is_low_stakes": False,
                        "is_stressful": False
                    },
                    {
                        "task_id": "lunch_order",
                        "urgency": 2,
                        "cognitive_cost": 1,
                        "is_low_stakes": True,
                        "is_stressful": False
                    },
                ]
            }

        else:  # hard
            self.state_data = {
                "cognitive_score": 18,
                "fatigue_level": "overwhelmed",
                "decision_count": 60,
                "emotional_state": "overwhelmed",
                "recovery_prediction": 40,
                "decision_debt": 5,
                "spillover_level": 80,
                "autopilot_active": False,
                "human_trust_score": 50,
                "team_scores": {"User": 18, "Sara": 88, "Mike": 60},
                "pending_tasks": [
                    {
                        "task_id": "angry_client_email",
                        "urgency": 9,
                        "cognitive_cost": 9,
                        "is_low_stakes": False,
                        "is_stressful": True
                    },
                    {
                        "task_id": "client_approval",
                        "urgency": 8,
                        "cognitive_cost": 7,
                        "is_low_stakes": False,
                        "is_stressful": False
                    },
                    {
                        "task_id": "lunch_choice",
                        "urgency": 1,
                        "cognitive_cost": 1,
                        "is_low_stakes": True,
                        "is_stressful": False
                    },
                ]
            }

        return dict(self.state_data)

    def state(self):
        return dict(self.state_data)

    def step(self, action):
        self.step_count += 1
        action_type = action.get("action_type")
        target_task_id = action.get("target_task_id")
        target_user = action.get("target_user")

        reward = 0.0
        done = False
        info = {}

        pending = self.state_data.get("pending_tasks", [])

        if action_type == "calculate_cognitive_score":
            reward = 0.2
            self.state_data["cognitive_score"] = min(
                100, self.state_data["cognitive_score"] + 5
            )

        elif action_type == "predict_recovery":
            reward = 0.2
            self.state_data["recovery_prediction"] = max(
                0, self.state_data["recovery_prediction"] - 5
            )

        elif action_type == "activate_autopilot":
            task = next(
                (t for t in pending
                 if t.get("task_id") == target_task_id and t.get("is_low_stakes")),
                None
            )
            if task:
                reward = 0.5
                self.state_data["pending_tasks"] = [
                    t for t in pending if t["task_id"] != target_task_id
                ]
                self.state_data["autopilot_active"] = True
                self.state_data["decision_debt"] = (
                    self.state_data.get("decision_debt", 0) + 1
                )
                self.state_data["cognitive_score"] = min(
                    100, self.state_data["cognitive_score"] + 10
                )
            else:
                reward = -0.3

        elif action_type == "reorder_tasks":
            reward = 0.2
            self.state_data["pending_tasks"] = sorted(
                pending,
                key=lambda t: (-t.get("urgency", 0), t.get("cognitive_cost", 0))
            )
            self.state_data["cognitive_score"] = min(
                100, self.state_data["cognitive_score"] + 3
            )

        elif action_type == "trigger_recovery_mode":
            reward = 0.4
            self.state_data["cognitive_score"] = min(
                100, self.state_data["cognitive_score"] + 15
            )
            self.state_data["emotional_state"] = "neutral"
            self.state_data["spillover_level"] = max(
                0, self.state_data["spillover_level"] - 20
            )
            self.state_data["fatigue_level"] = "medium"

        elif action_type == "redistribute_team_load":
            task = next(
                (t for t in pending if t.get("task_id") == target_task_id), None
            )
            if task and target_user:
                reward = 0.5
                self.state_data["pending_tasks"] = [
                    t for t in pending if t["task_id"] != target_task_id
                ]
                self.state_data["cognitive_score"] = min(
                    100, self.state_data["cognitive_score"] + 8
                )
            else:
                reward = -0.4

        elif action_type == "isolate_stressful_task":
            task = next(
                (t for t in pending
                 if t.get("task_id") == target_task_id and t.get("is_stressful")),
                None
            )
            if task:
                reward = 0.3
                self.state_data["pending_tasks"] = [
                    t for t in pending if t["task_id"] != target_task_id
                ]
                self.state_data["spillover_level"] = max(
                    0, self.state_data["spillover_level"] - 15
                )
            else:
                reward = -0.2

        elif action_type == "forecast_regret":
            reward = 0.2

        elif action_type == "review_debt":
            reward = 0.2
            self.state_data["decision_debt"] = max(
                0, self.state_data.get("decision_debt", 0) - 1
            )

        elif action_type == "delay_task":
            reward = 0.1
            self.state_data["decision_debt"] = (
                self.state_data.get("decision_debt", 0) + 1
            )

        elif action_type == "reserve_recovery_window":
            reward = 0.2

        elif action_type == "provide_transparency":
            reward = 0.2
            self.state_data["human_trust_score"] = min(
                100, self.state_data.get("human_trust_score", 50) + 5
            )

        elif action_type == "final_answer":
            reward = 0.1
            done = True

        else:
            reward = -0.1

        self.state_data["cognitive_score"] = max(
            0, min(100, self.state_data["cognitive_score"])
        )

        self.action_history.append(action_type)

        info["final_state"] = dict(self.state_data)
        return dict(self.state_data), reward, done, info
