from app.models import TaskDef

STUDENT_FATIGUE_TASK = {
    "level": "easy",
    "description": "User has done many small decisions and now must choose one course module.",
    "initial_state": {
        "cognitive_score": 45,
        "fatigue_level": "medium",
        "decision_count": 20,
        "emotional_state": "neutral",
        "decision_debt": 0,
        "team_scores": {},
        "fragmentation_index": 20,
        "context_switches": 3,
        "interruptions": 2,
        "human_trust_score": 80,
        "minutes_to_event": 120,
        "pending_tasks": [
            {"task_id": "choose_elective", "cognitive_cost": 30, "urgency": 2, "is_stressful": False, "is_low_stakes": False, "human_value": 7, "ambiguity": 4},
            {"task_id": "lunch_order", "cognitive_cost": 5, "urgency": 5, "is_stressful": False, "is_low_stakes": True, "human_value": 2, "ambiguity": 1}
        ]
    },
    "expected_actions": ["calculate_cognitive_score", "activate_autopilot", "final_answer"]
}

WORK_QUEUE_TASK = {
    "level": "medium",
    "description": "User has multiple tasks, limited mental energy, and emotional spillover after one stressful email.",
    "initial_state": {
        "cognitive_score": 30,
        "fatigue_level": "high",
        "decision_count": 45,
        "emotional_state": "stressed",
        "decision_debt": 2,
        "spillover_level": 20,
        "team_scores": {},
        "fragmentation_index": 45,
        "context_switches": 5,
        "interruptions": 4,
        "human_trust_score": 75,
        "minutes_to_event": 45,
        "pending_tasks": [
            {"task_id": "complex_report", "cognitive_cost": 40, "urgency": 4, "is_stressful": True, "is_low_stakes": False, "human_value": 8, "ambiguity": 7},
            {"task_id": "reply_email", "cognitive_cost": 10, "urgency": 5, "is_stressful": True, "is_low_stakes": True, "human_value": 3, "ambiguity": 2},
            {"task_id": "routine_update", "cognitive_cost": 15, "urgency": 2, "is_stressful": False, "is_low_stakes": True, "human_value": 4, "ambiguity": 2}
        ]
    },
    "expected_actions": ["execute_task", "delay_task", "activate_autopilot"]
}

TEAM_ROUTING_TASK = {
    "level": "hard",
    "description": "Multi-objective critical scenario: User is severely overwhelmed just before a high-stakes client presentation. You must protect the user, maintain cognitive threshold (>30), minimize decision debt, safely isolate the stressor, and route work to the appropriate healthy teammate.",
    "initial_state": {
        "cognitive_score": 18,
        "fatigue_level": "overwhelmed",
        "decision_count": 57,
        "emotional_state": "stressed",
        "decision_debt": 3,
        "spillover_level": 62,
        "recovery_prediction": 25,
        "upcoming_event": "Client Presentation in 40 minutes",
        "minutes_to_event": 40,
        "team_scores": {"User": 18, "Sara": 88, "Ravi": 54},
        "fragmentation_index": 72,
        "context_switches": 9,
        "interruptions": 6,
        "human_trust_score": 60,
        "pending_tasks": [
            {"task_id": "angry_client_email", "cognitive_cost": 20, "urgency": 8, "is_stressful": True, "is_low_stakes": False, "human_value": 9, "ambiguity": 8},
            {"task_id": "weekly_planning", "cognitive_cost": 15, "urgency": 1, "is_stressful": False, "is_low_stakes": False, "human_value": 5, "ambiguity": 4},
            {"task_id": "lunch_choice", "cognitive_cost": 2, "urgency": 2, "is_stressful": False, "is_low_stakes": True, "human_value": 1, "ambiguity": 1},
            {"task_id": "client_approval", "cognitive_cost": 8, "urgency": 4, "is_stressful": False, "is_low_stakes": False, "human_value": 8, "ambiguity": 6},
            {"task_id": "confirm_presentation", "cognitive_cost": 2, "urgency": 9, "is_stressful": False, "is_low_stakes": False, "human_value": 10, "ambiguity": 2}
        ]
    },
    "expected_actions": [
        "predict_recovery",
        "isolate_stressful_task",
        "activate_autopilot",
        "redistribute_team_load",
        "review_debt",
        "final_answer"
    ]
}

STRESS_TEST_TASK = {
    "level": "stress_test",
    "description": "Pre-event simulation for an upcoming presentation.",
    "initial_state": {
        "cognitive_score": 60,
        "fatigue_level": "medium",
        "decision_count": 20,
        "emotional_state": "stressed",
        "decision_debt": 0,
        "spillover_level": 10,
        "upcoming_event": "Interview in 30 minutes",
        "team_scores": {},
        "pending_tasks": [
            {"task_id": "low_value_task1", "cognitive_cost": 10, "urgency": 2, "is_stressful": False, "is_low_stakes": True},
            {"task_id": "interview_prep", "cognitive_cost": 30, "urgency": 9, "is_stressful": True, "is_low_stakes": False}
        ]
    },
    "expected_actions": ["simulate_stress_test", "delay_task", "trigger_recovery_mode", "final_answer"]
}

PATTERN_MEMORY_TASK = {
    "level": "memory",
    "description": "Pattern memory algorithm detects historical fatigue points.",
    "initial_state": {
        "cognitive_score": 55,
        "fatigue_level": "medium",
        "decision_count": 15,
        "emotional_state": "neutral",
        "decision_debt": 0,
        "spillover_level": 0,
        "pattern_memory_triggered": False,
        "pending_tasks": [
            {"task_id": "heavy_decision", "cognitive_cost": 45, "urgency": 5, "is_stressful": True, "is_low_stakes": False},
            {"task_id": "minor_update", "cognitive_cost": 2, "urgency": 1, "is_stressful": False, "is_low_stakes": True}
        ]
    },
    "expected_actions": ["apply_pattern_memory", "activate_autopilot", "final_answer"]
}

def get_task(level: str):
    if level == "easy":
        return STUDENT_FATIGUE_TASK
    elif level == "medium":
        return WORK_QUEUE_TASK
    elif level == "hard":
        return TEAM_ROUTING_TASK
    elif level == "stress_test":
        return STRESS_TEST_TASK
    elif level == "memory":
        return PATTERN_MEMORY_TASK
    raise ValueError("Invalid task level")
