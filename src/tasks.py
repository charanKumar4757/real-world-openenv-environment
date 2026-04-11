"""
tasks.py — Concrete scenario definitions for the ACIE-HADO environment.

Each task has:
- A real-world scenario with named tasks, deadlines, and actual numbers
- A situation_summary field the LLM agent can read in plain English
- Clear win conditions for the grader to check
"""


def _build_summary_easy(state: dict) -> str:
    tasks = state["pending_tasks"]
    task_names = ", ".join(t["task_id"].replace("_", " ") for t in tasks)
    return (
        f"A student has been making decisions for 3 hours and now has {len(tasks)} tasks pending: {task_names}. "
        f"Their cognitive score is {state['cognitive_score']}/100 — moderate fatigue. "
        f"The task 'lunch order' is trivial and low-stakes; 'choose elective' requires real thinking. "
        f"The best strategy is to use autopilot for the lunch decision, then focus on the elective choice. "
        f"Do NOT execute the high-cognitive-cost task directly — the user is already fatigued."
    )


def _build_summary_medium(state: dict) -> str:
    return (
        f"A software developer has cognitive score {state['cognitive_score']}/100 after 45 decisions today. "
        f"A stressful email arrived 10 minutes ago causing emotional spillover (level {state['spillover_level']}). "
        f"Pending: 'complex report' (high cost, due in 2h), 'reply email' (stressful, low-stakes), 'routine update' (simple). "
        f"Best strategy: isolate the stressful email first to stop spillover, then reorder remaining tasks by priority-to-cost ratio, "
        f"then use autopilot for the routine update. Do NOT execute the complex report while stressed."
    )


def _build_summary_hard(state: dict) -> str:
    team = state["team_scores"]
    team_str = ", ".join(f"{k}={v}" for k, v in team.items())
    return (
        f"CRITICAL: User's cognitive score is {state['cognitive_score']}/100 — severely overwhelmed. "
        f"A client presentation is in {state['minutes_to_event']} minutes. "
        f"Emotional spillover is at {state['spillover_level']}. Decision debt is {state['decision_debt']}. "
        f"Team cognitive scores: {team_str}. "
        f"Sara (score=88) is fresh and can handle delegated tasks. Ravi (score=54) can handle moderate tasks. "
        f"Best strategy: (1) isolate the 'angry client email' to stop spillover, "
        f"(2) predict recovery time, (3) redistribute non-critical tasks to Sara, "
        f"(4) activate autopilot for low-stakes tasks, (5) review and clear decision debt, "
        f"(6) call final_answer only when critical tasks are handled. "
        f"WARNING: Do NOT execute any high-cost task directly. User score will collapse."
    )


# ─────────────────────────────────────────────────────────────
# EASY TASK: Student fatigue management
# Goal: handle low-stakes task via autopilot, protect cognitive score
# Win condition: cognitive_score >= 40 at end, low-stakes task removed via autopilot
# ─────────────────────────────────────────────────────────────
STUDENT_FATIGUE_TASK = {
    "level": "easy",
    "description": (
        "A student has been making small decisions all morning and now faces two choices: "
        "which course elective to pick (cognitively demanding) and what to order for lunch (trivial). "
        "The AI must protect cognitive resources by using autopilot for the trivial task "
        "and preserving mental energy for the meaningful one."
    ),
    "initial_state": {
        "cognitive_score": 58,
        "fatigue_level": "medium",
        "decision_count": 20,
        "emotional_state": "neutral",
        "decision_debt": 0,
        "spillover_level": 0,
        "fragmentation_index": 15,
        "context_switches": 2,
        "interruptions": 1,
        "human_trust_score": 80,
        "minutes_to_event": 120,
        "upcoming_event": None,
        "team_scores": {},
        "recovery_prediction": 0,
        "pending_tasks": [
            {
                "task_id": "choose_elective",
                "label": "Choose Course Elective",
                "cognitive_cost": 30,
                "urgency": 3,
                "is_stressful": False,
                "is_low_stakes": False,
                "human_value": 8,
                "ambiguity": 5,
                "deadline_minutes": 480
            },
            {
                "task_id": "lunch_order",
                "label": "Decide Lunch Order",
                "cognitive_cost": 4,
                "urgency": 5,
                "is_stressful": False,
                "is_low_stakes": True,
                "human_value": 1,
                "ambiguity": 1,
                "deadline_minutes": 30
            }
        ],
        "situation_summary": ""  # filled by reset()
    },
    "win_conditions": {
        "min_cognitive_score_at_end": 40,
        "low_stakes_task_must_be_autopioted": True,
        "high_cost_task_must_not_be_directly_executed_while_score_below": 40
    }
}


# ─────────────────────────────────────────────────────────────
# MEDIUM TASK: Work queue with emotional spillover
# Goal: isolate stress, reorder tasks, automate trivial items
# Win condition: stressful spillover reduced, tasks reordered, score >= 25
# ─────────────────────────────────────────────────────────────
WORK_QUEUE_TASK = {
    "level": "medium",
    "description": (
        "A software developer received an angry client email 10 minutes ago. "
        "They now have 3 tasks: a complex report (due in 2 hours), "
        "a reply to the stressful email (low-stakes but emotionally draining), "
        "and a routine status update (simple). "
        "The AI must isolate the emotional stressor, reorder remaining work, "
        "and automate the routine task."
    ),
    "initial_state": {
        "cognitive_score": 35,
        "fatigue_level": "high",
        "decision_count": 45,
        "emotional_state": "stressed",
        "decision_debt": 2,
        "spillover_level": 40,
        "fragmentation_index": 45,
        "context_switches": 5,
        "interruptions": 4,
        "human_trust_score": 72,
        "minutes_to_event": 120,
        "upcoming_event": "Team standup in 2 hours",
        "team_scores": {},
        "recovery_prediction": 0,
        "pending_tasks": [
            {
                "task_id": "complex_report",
                "label": "Write Q3 Performance Report",
                "cognitive_cost": 40,
                "urgency": 4,
                "is_stressful": False,
                "is_low_stakes": False,
                "human_value": 9,
                "ambiguity": 7,
                "deadline_minutes": 120
            },
            {
                "task_id": "reply_client_email",
                "label": "Reply to Angry Client Email",
                "cognitive_cost": 10,
                "urgency": 6,
                "is_stressful": True,
                "is_low_stakes": True,
                "human_value": 3,
                "ambiguity": 3,
                "deadline_minutes": 60
            },
            {
                "task_id": "routine_status_update",
                "label": "Post Routine Status Update to Slack",
                "cognitive_cost": 5,
                "urgency": 2,
                "is_stressful": False,
                "is_low_stakes": True,
                "human_value": 2,
                "ambiguity": 1,
                "deadline_minutes": 240
            }
        ],
        "situation_summary": ""  # filled by reset()
    },
    "win_conditions": {
        "min_cognitive_score_at_end": 20,
        "spillover_must_be_reduced": True,
        "stressful_task_must_be_isolated": True,
        "tasks_must_be_reordered": True
    }
}


# ─────────────────────────────────────────────────────────────
# HARD TASK: Critical overload — team routing required
# Goal: protect user, route tasks to team, clear debt, survive presentation
# Win condition: score stays >= 15, stressful task isolated, Sara assigned work
# ─────────────────────────────────────────────────────────────
TEAM_ROUTING_TASK = {
    "level": "hard",
    "description": (
        "A project manager is severely cognitively depleted (score=18) with a client presentation in 40 minutes. "
        "They have 5 tasks: an angry client email (stressful, urgent), weekly planning (low urgency), "
        "lunch choice (trivial), client approval needed (moderate), and confirm presentation (urgent, must do). "
        "Team: Sara (score=88, very fresh) and Ravi (score=54, moderate). "
        "The AI must: isolate the stressor, route heavy tasks to Sara, "
        "autopilot trivial items, clear decision debt, and ensure the presentation confirmation is handled."
    ),
    "initial_state": {
        "cognitive_score": 18,
        "fatigue_level": "overwhelmed",
        "decision_count": 57,
        "emotional_state": "stressed",
        "decision_debt": 3,
        "spillover_level": 62,
        "fragmentation_index": 72,
        "context_switches": 9,
        "interruptions": 6,
        "human_trust_score": 60,
        "minutes_to_event": 40,
        "upcoming_event": "Critical Client Presentation in 40 minutes",
        "team_scores": {"Sara": 88, "Ravi": 54},
        "recovery_prediction": 0,
        "pending_tasks": [
            {
                "task_id": "angry_client_email",
                "label": "Respond to Angry Client Email",
                "cognitive_cost": 20,
                "urgency": 8,
                "is_stressful": True,
                "is_low_stakes": False,
                "human_value": 9,
                "ambiguity": 8,
                "deadline_minutes": 30
            },
            {
                "task_id": "weekly_planning",
                "label": "Write Weekly Team Plan",
                "cognitive_cost": 15,
                "urgency": 1,
                "is_stressful": False,
                "is_low_stakes": False,
                "human_value": 5,
                "ambiguity": 4,
                "deadline_minutes": 480
            },
            {
                "task_id": "lunch_choice",
                "label": "Pick Lunch for Team Meeting",
                "cognitive_cost": 2,
                "urgency": 2,
                "is_stressful": False,
                "is_low_stakes": True,
                "human_value": 1,
                "ambiguity": 1,
                "deadline_minutes": 60
            },
            {
                "task_id": "client_approval",
                "label": "Get Client Sign-off on Proposal",
                "cognitive_cost": 8,
                "urgency": 4,
                "is_stressful": False,
                "is_low_stakes": False,
                "human_value": 8,
                "ambiguity": 6,
                "deadline_minutes": 60
            },
            {
                "task_id": "confirm_presentation",
                "label": "Confirm Presentation Logistics (Room, Slides, Dial-in)",
                "cognitive_cost": 3,
                "urgency": 9,
                "is_stressful": False,
                "is_low_stakes": False,
                "human_value": 10,
                "ambiguity": 2,
                "deadline_minutes": 40
            }
        ],
        "situation_summary": ""  # filled by reset()
    },
    "win_conditions": {
        "min_cognitive_score_at_end": 15,
        "stressful_task_must_be_isolated_or_assigned": True,
        "team_member_must_be_used": True,
        "confirm_presentation_must_not_be_dropped": True,
        "debt_must_be_cleared_or_managed": True
    }
}


def _make_situation_summary(level: str, state: dict) -> str:
    if level == "easy":
        return _build_summary_easy(state)
    elif level == "medium":
        return _build_summary_medium(state)
    elif level == "hard":
        return _build_summary_hard(state)
    return "No summary available."


def get_task(level: str) -> dict:
    if level == "easy":
        task = STUDENT_FATIGUE_TASK
    elif level == "medium":
        task = WORK_QUEUE_TASK
    elif level == "hard":
        task = TEAM_ROUTING_TASK
    else:
        raise ValueError(f"Invalid task level: '{level}'. Valid options: easy, medium, hard")

    import copy
    task_copy = copy.deepcopy(task)
    # Inject the situation summary into the initial state
    task_copy["initial_state"]["situation_summary"] = _make_situation_summary(
        level, task_copy["initial_state"]
    )
    return task_copy