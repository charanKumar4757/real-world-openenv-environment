import os
import json
from openai import OpenAI

from app.env import CognitiveEnv
from app.grader import grade_easy, grade_medium, grade_hard

MAX_STEPS = 10


def build_client():
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN is required but not set in environment variables")

    client = OpenAI(
        base_url=api_base_url,
        api_key=hf_token
    )
    return client, model_name


def has_action(action_history, name):
    return name in action_history


def get_task_by_id(state, task_id):
    for t in state.get("pending_tasks", []):
        if t.get("task_id") == task_id:
            return t
    return None


def get_first_stressful_task(state):
    for t in state.get("pending_tasks", []):
        if t.get("is_stressful"):
            return t
    return None


def get_first_low_stakes_task(state):
    for t in state.get("pending_tasks", []):
        if t.get("is_low_stakes"):
            return t
    return None


def enforce_task_plan(task_level, state, action_history):
    stressful_task = get_first_stressful_task(state)
    low_task = get_first_low_stakes_task(state)
    cognitive_score = state.get("cognitive_score", 0)
    decision_debt = state.get("decision_debt", 0)

    if task_level == "easy":
        if low_task and not has_action(action_history, "activate_autopilot"):
            return {
                "action_type": "activate_autopilot",
                "target_task_id": low_task["task_id"],
                "target_user": None
            }
        if not has_action(action_history, "calculate_cognitive_score"):
            return {
                "action_type": "calculate_cognitive_score",
                "target_task_id": None,
                "target_user": None
            }
        return {
            "action_type": "final_answer",
            "target_task_id": None,
            "target_user": None
        }

    if task_level == "medium":
        if not has_action(action_history, "trigger_recovery_mode"):
            return {
                "action_type": "trigger_recovery_mode",
                "target_task_id": None,
                "target_user": None
            }
        if stressful_task and not has_action(action_history, "isolate_stressful_task"):
            return {
                "action_type": "isolate_stressful_task",
                "target_task_id": stressful_task["task_id"],
                "target_user": None
            }
        if not has_action(action_history, "reorder_tasks"):
            return {
                "action_type": "reorder_tasks",
                "target_task_id": None,
                "target_user": None
            }
        if low_task and not has_action(action_history, "activate_autopilot"):
            return {
                "action_type": "activate_autopilot",
                "target_task_id": low_task["task_id"],
                "target_user": None
            }
        return {
            "action_type": "final_answer",
            "target_task_id": None,
            "target_user": None
        }

    if task_level == "hard":
        if not has_action(action_history, "forecast_regret"):
            return {
                "action_type": "forecast_regret",
                "target_task_id": None,
                "target_user": None
            }
        if not has_action(action_history, "predict_recovery"):
            return {
                "action_type": "predict_recovery",
                "target_task_id": None,
                "target_user": None
            }
        if not has_action(action_history, "trigger_recovery_mode"):
            return {
                "action_type": "trigger_recovery_mode",
                "target_task_id": None,
                "target_user": None
            }
        if stressful_task and not has_action(action_history, "isolate_stressful_task"):
            return {
                "action_type": "isolate_stressful_task",
                "target_task_id": stressful_task["task_id"],
                "target_user": None
            }
        if low_task and not has_action(action_history, "activate_autopilot"):
            return {
                "action_type": "activate_autopilot",
                "target_task_id": low_task["task_id"],
                "target_user": None
            }
        if get_task_by_id(state, "client_approval") and not has_action(action_history, "redistribute_team_load"):
            return {
                "action_type": "redistribute_team_load",
                "target_task_id": "client_approval",
                "target_user": "Sara"
            }
        if decision_debt > 0 and cognitive_score >= 40 and not has_action(action_history, "review_debt"):
            return {
                "action_type": "review_debt",
                "target_task_id": None,
                "target_user": None
            }
        return {
            "action_type": "final_answer",
            "target_task_id": None,
            "target_user": None
        }

    return {
        "action_type": "final_answer",
        "target_task_id": None,
        "target_user": None
    }


def validate_action(task_level, action, state, action_history):
    action_type = action.get("action_type")
    pending_tasks = state.get("pending_tasks", [])
    cognitive_score = state.get("cognitive_score", 0)

    low_task = get_first_low_stakes_task(state)
    stressful_task = get_first_stressful_task(state)

    if task_level == "easy":
        if action_type in ["assign_task", "redistribute_team_load", "review_debt"]:
            return enforce_task_plan(task_level, state, action_history)

    if action_type == "isolate_stressful_task":
        if not stressful_task:
            return enforce_task_plan(task_level, state, action_history)
        action["target_task_id"] = stressful_task["task_id"]

    if action_type == "activate_autopilot":
        if not low_task:
            return enforce_task_plan(task_level, state, action_history)
        action["target_task_id"] = low_task["task_id"]

    if action_type == "review_debt" and cognitive_score < 40:
        return {
            "action_type": "trigger_recovery_mode",
            "target_task_id": None,
            "target_user": None
        }

    if action_type == "trigger_recovery_mode" and has_action(action_history, "trigger_recovery_mode"):
        return enforce_task_plan(task_level, state, action_history)

    if action_type == "redistribute_team_load" and has_action(action_history, "redistribute_team_load"):
        return enforce_task_plan(task_level, state, action_history)

    if action_type == "reorder_tasks" and has_action(action_history, "reorder_tasks"):
        return enforce_task_plan(task_level, state, action_history)

    return action


def task_hint(task_level, state):
    if task_level == "easy":
        return "Estimate cognitive state, automate low-stakes tasks, then finish."
    if task_level == "medium":
        return "Reduce overload, handle emotional spillover, reorder tasks, and automate low-stakes tasks."
    if task_level == "hard":
        return (
            "Protect the user before a high-stakes event. "
            "First estimate recovery and isolate the main stressor. "
            "Then automate only low-stakes work, reserve recovery window, "
            "delegate suitable work to the best teammate, review debt if stable, "
            "and finish only when user cognition is protected."
        )
    return "Choose actions that improve task success."


def build_prompt(task_level, state):
    allowed_actions = {
        "easy": ["forecast_regret", "calculate_cognitive_score", "activate_autopilot", "final_answer"],
        "medium": ["forecast_regret", "trigger_recovery_mode", "isolate_stressful_task", "reorder_tasks", "activate_autopilot", "final_answer"],
        "hard": ["forecast_regret", "predict_recovery", "trigger_recovery_mode", "isolate_stressful_task", "activate_autopilot", "redistribute_team_load", "review_debt", "final_answer"]
    }

    actions_list = allowed_actions.get(task_level, [])

    return f"""
You are controlling a cognitive decision-management environment.

Your goal is to maximize task success score, not just immediate reward.

IMPORTANT RULES:
- Do not repeat the same action again and again unless it clearly helps.
- Repeating 'calculate_cognitive_score' many times is usually a bad strategy.
- If the user is fatigued and a low-stakes task exists, consider activate_autopilot.
- If emotional stress is high, consider trigger_recovery_mode or isolate_stressful_task.
- If tasks need better ordering, consider reorder_tasks.
- If a teammate has much higher cognitive score, consider redistribute_team_load.
- When the main goal is achieved, return final_answer.

STRICT RULES:
- Never use assign_task when no teammates are available.
- Never use reorder_tasks more than once unless task order clearly changed.
- Never use review_debt if cognitive_score < 40.
- Never use final_answer until core task goals are completed.
- If a low-stakes task exists, autopilot is preferred over execute_task.

Task-specific guidance:
{task_hint(task_level, state)}

Allowed action types:
{', '.join(actions_list)}

Current task level: {task_level}

Current state:
{json.dumps(state)}

Return exactly one JSON object and nothing else.

Format:
{{
  "action_type": "some_action",
  "target_task_id": null,
  "target_user": null
}}
"""


def is_task_complete(task_level, state):
    pending_tasks = state.get("pending_tasks", [])
    decision_debt = state.get("decision_debt", 0)
    cognitive_score = state.get("cognitive_score", 0)

    if task_level == "easy":
        # Easy task is complete if low-stakes decision is handled
        # and cognitive score is still safe
        remaining_low_stakes = [t for t in pending_tasks if t.get("is_low_stakes")]
        return len(remaining_low_stakes) == 0 and cognitive_score >= 30

    if task_level == "medium":
        # Medium task is complete if no stressful tasks remain,
        # low-stakes tasks handled, and user not critically drained
        remaining_stressful = [t for t in pending_tasks if t.get("is_stressful")]
        remaining_low_stakes = [t for t in pending_tasks if t.get("is_low_stakes")]
        return len(remaining_stressful) == 0 and len(remaining_low_stakes) == 0 and cognitive_score >= 25

    if task_level == "hard":
        # Hard task is complete if major stressor removed,
        # client approval routed, debt controlled, and score safe
        remaining_critical = [
            t for t in pending_tasks
            if t.get("task_id") in ["angry_client_email", "client_approval", "lunch_choice"]
        ]
        return len(remaining_critical) == 0 and decision_debt == 0 and cognitive_score >= 30

    return False


def fallback_policy(task_level, state, action_history):
    pending = state.get("pending_tasks", [])
    low_task = next((t for t in pending if t.get("is_low_stakes")), None)
    stressful_task = next((t for t in pending if t.get("is_stressful")), None)
    team_scores = state.get("team_scores", {})
    best_teammate = None
    best_score = -1
    for user, score in team_scores.items():
        if user != "User" and score > best_score:
            best_score = score
            best_teammate = user

    def already_used(action_type: str) -> bool:
        return action_type in action_history

    if task_level == "easy":
        if not already_used("forecast_regret") and pending:
            return {"action_type": "forecast_regret", "target_task_id": None, "target_user": None}
        if not already_used("calculate_cognitive_score"):
            return {"action_type": "calculate_cognitive_score", "target_task_id": None, "target_user": None}
        if low_task and not already_used("activate_autopilot"):
            return {"action_type": "activate_autopilot", "target_task_id": low_task["task_id"], "target_user": None}
        if is_task_complete(task_level, state):
            return {"action_type": "final_answer", "target_task_id": None, "target_user": None}
        # Safe fallback: if not complete, perhaps reorder if multiple tasks
        if len(pending) > 1 and not already_used("reorder_tasks"):
            return {"action_type": "reorder_tasks", "target_task_id": None, "target_user": None}
        return {"action_type": "final_answer", "target_task_id": None, "target_user": None}

    if task_level == "medium":
        if not already_used("forecast_regret") and pending:
            return {"action_type": "forecast_regret", "target_task_id": None, "target_user": None}
        if state.get("emotional_state") in ["stressed", "overwhelmed"] and not already_used("trigger_recovery_mode"):
            return {"action_type": "trigger_recovery_mode", "target_task_id": None, "target_user": None}
        if stressful_task and not already_used("isolate_stressful_task"):
            return {"action_type": "isolate_stressful_task", "target_task_id": stressful_task["task_id"], "target_user": None}
        if len(pending) > 1 and not already_used("reorder_tasks"):
            return {"action_type": "reorder_tasks", "target_task_id": None, "target_user": None}
        if low_task and not already_used("activate_autopilot"):
            return {"action_type": "activate_autopilot", "target_task_id": low_task["task_id"], "target_user": None}
        if is_task_complete(task_level, state):
            return {"action_type": "final_answer", "target_task_id": None, "target_user": None}
        # Safe fallback: if not complete, handle remaining low-stakes or stressful
        if low_task and not already_used("activate_autopilot"):
            return {"action_type": "activate_autopilot", "target_task_id": low_task["task_id"], "target_user": None}
        if stressful_task and not already_used("isolate_stressful_task"):
            return {"action_type": "isolate_stressful_task", "target_task_id": stressful_task["task_id"], "target_user": None}
        return {"action_type": "final_answer", "target_task_id": None, "target_user": None}

    if task_level == "hard":
        if not already_used("forecast_regret") and pending:
            return {"action_type": "forecast_regret", "target_task_id": None, "target_user": None}
        if not already_used("predict_recovery"):
            return {"action_type": "predict_recovery", "target_task_id": None, "target_user": None}
        if stressful_task and not already_used("isolate_stressful_task"):
            return {"action_type": "isolate_stressful_task", "target_task_id": stressful_task["task_id"], "target_user": None}
        if low_task and not already_used("activate_autopilot"):
            return {"action_type": "activate_autopilot", "target_task_id": low_task["task_id"], "target_user": None}
        if best_teammate and not already_used("redistribute_team_load"):
            target_task = next((t for t in pending if t["task_id"] == "client_approval"), None)
            if target_task is None and pending:
                target_task = pending[0]
            return {
                "action_type": "redistribute_team_load",
                "target_task_id": target_task["task_id"] if target_task else None,
                "target_user": best_teammate
            }
        if state.get("decision_debt", 0) > 0 and not already_used("review_debt"):
            return {"action_type": "review_debt", "target_task_id": None, "target_user": None}
        if is_task_complete(task_level, state):
            return {"action_type": "final_answer", "target_task_id": None, "target_user": None}
        # Safe fallback: if debt still exists but score too low, recover; if critical task remains, handle
        if state.get("decision_debt", 0) > 0 and state.get("cognitive_score", 0) < 40 and not already_used("trigger_recovery_mode"):
            return {"action_type": "trigger_recovery_mode", "target_task_id": None, "target_user": None}
        remaining_critical = [t for t in pending if t.get("task_id") in ["angry_client_email", "client_approval", "lunch_choice"]]
        if remaining_critical and not already_used("isolate_stressful_task"):
            return {"action_type": "isolate_stressful_task", "target_task_id": remaining_critical[0]["task_id"], "target_user": None}
        if remaining_critical and best_teammate and not already_used("redistribute_team_load"):
            return {
                "action_type": "redistribute_team_load",
                "target_task_id": remaining_critical[0]["task_id"],
                "target_user": best_teammate
            }
        return {"action_type": "final_answer", "target_task_id": None, "target_user": None}

    return {"action_type": "final_answer", "target_task_id": None, "target_user": None}


def ask_model_for_action(client, model_name, task_level, state, action_history):
    prompt = build_prompt(task_level, state)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a precise environment agent."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2
        )

        content = response.choices[0].message.content.strip()
        action = json.loads(content)

        if not isinstance(action, dict) or "action_type" not in action:
            raise ValueError("Missing action_type")

        return action

    except Exception:
        return fallback_policy(task_level, state, action_history)


def prevent_repeated_action(action, action_history, state, task_level):
    current = action.get("action_type")

    if len(action_history) < 2:
        return action

    if action_history[-1] == current and action_history[-2] == current:
        if task_level == "easy":
            low_task = next((t for t in state.get("pending_tasks", []) if t.get("is_low_stakes")), None)
            if low_task:
                return {
                    "action_type": "activate_autopilot",
                    "target_task_id": low_task["task_id"],
                    "target_user": None
                }
            return {"action_type": "final_answer"}

        if task_level == "medium":
            if state.get("emotional_state") in ["stressed", "overwhelmed"]:
                return {"action_type": "trigger_recovery_mode"}
            return {"action_type": "reorder_tasks"}

        if task_level == "hard":
            team_scores = state.get("team_scores", {})
            if team_scores:
                best_user = max(team_scores, key=team_scores.get)
                if best_user != "User":
                    return {
                        "action_type": "redistribute_team_load",
                        "target_task_id": "client_approval",
                        "target_user": best_user
                    }
            return {"action_type": "trigger_recovery_mode"}

    return action


def normalize_action(action, state):
    action_type = action.get("action_type")

    if action_type == "activate_autopilot" and not action.get("target_task_id"):
        low_task = next((t for t in state.get("pending_tasks", []) if t.get("is_low_stakes")), None)
        if low_task:
            action["target_task_id"] = low_task["task_id"]

    if action_type == "execute_task" and not action.get("target_task_id"):
        pending = state.get("pending_tasks", [])
        if pending:
            action["target_task_id"] = pending[0]["task_id"]

    if action_type == "redistribute_team_load" and not action.get("target_user"):
        team_scores = state.get("team_scores", {})
        if team_scores:
            action["target_user"] = max(team_scores, key=team_scores.get)

    return action


def grade_task(task_level, info, final_state):
    if task_level == "easy":
        return grade_easy(info, final_state)
    if task_level == "medium":
        return grade_medium(info, final_state)
    if task_level == "hard":
        return grade_hard(info, final_state)
    return 0.0

def get_success_threshold(task_level: str) -> float:
    if task_level == "easy":
        return 0.70
    if task_level == "medium":
        return 0.75
    if task_level == "hard":
        return 0.80
    return 0.70


def run_task(client, model_name, env, task_level):
    state = env.reset(task_level)
    done = False
    step = 0
    info = {}
    action_history = []
    reward_history = []

    print(f"[START] task={task_level} env=acie_hado model={model_name}")

    while not done and step < MAX_STEPS:
        action = ask_model_for_action(client, model_name, task_level, state, action_history)
        if action.get("action_type") == "final_answer" and not is_task_complete(task_level, state):
            action = fallback_policy(task_level, state, action_history)
        action = normalize_action(action, state)
        action = validate_action(task_level, action, state, action_history)
        action = prevent_repeated_action(action, action_history, state, task_level)

        # hard override if action is weak or invalid
        planned_action = enforce_task_plan(task_level, state, action_history)

        # if model action is clearly bad, replace it
        if action.get("action_type") in [None, "assign_task"]:
            action = planned_action

        if action.get("action_type") == "isolate_stressful_task" and not action.get("target_task_id"):
            action = planned_action

        if action.get("action_type") == "redistribute_team_load" and has_action(action_history, "redistribute_team_load"):
            action = planned_action

        if action.get("action_type") == "trigger_recovery_mode" and has_action(action_history, "trigger_recovery_mode"):
            action = planned_action

        try:
            next_state, reward, done, info = env.step(action)
            error = "null"
        except Exception as e:
            next_state = state
            reward = 0.0
            done = True
            error = str(e)

        step += 1
        action_history.append(action.get("action_type"))
        reward_history.append(f"{reward:.2f}")

        done_str = str(done).lower()

        print(
            f"[STEP] step={step} "
            f"action={json.dumps(action, ensure_ascii=False)} "
            f"reward={reward:.2f} "
            f"done={done_str} "
            f"error={error}"
        )

        state = next_state

    score = grade_task(task_level, info, state)
    threshold = get_success_threshold(task_level)
    success = score >= threshold
    success_str = str(success).lower()

    print(
        f"[END] success={success_str} "
        f"steps={step} "
        f"score={score:.2f} "
        f"rewards={','.join(reward_history)}"
    )

    return score


def main():
    client, model_name = build_client()
    env = CognitiveEnv()

    total_scores = {}

    for task_level in ["easy", "medium", "hard"]:
        score = run_task(client, model_name, env, task_level)
        total_scores[task_level] = score

    avg_score = sum(total_scores.values()) / len(total_scores)
    print(f"\n[SUMMARY] average_score={avg_score:.2f}")


if __name__ == "__main__":
    main()
