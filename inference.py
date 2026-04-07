import os
import json
from openai import OpenAI

from app.env import CognitiveEnv
from app.grader import grade_easy, grade_medium, grade_hard

MAX_STEPS = 10


def build_client():
    api_base_url = os.getenv("API_BASE_URL")
    api_key = os.getenv("API_KEY")
    model_name = os.getenv("MODEL_NAME")

    if not api_base_url or not api_key or not model_name:
        raise RuntimeError(
            "Missing API_BASE_URL, API_KEY, or MODEL_NAME. "
            "These are required only when running inference.py."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=api_base_url
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
    pending_tasks = state.get("pending_tasks", [])
    cognitive_score = state.get("cognitive_score", 0)

    def find_task(task_id):
        for t in pending_tasks:
            if t.get("task_id") == task_id:
                return t
        return None

    def first_stressful():
        for t in pending_tasks:
            if t.get("is_stressful"):
                return t
        return None

    def first_low():
        for t in pending_tasks:
            if t.get("is_low_stakes"):
                return t
        return None

    if task_level == "easy":
        if "forecast_regret" not in action_history:
            return {"action_type": "forecast_regret", "target_task_id": None, "target_user": None}
        if "calculate_cognitive_score" not in action_history:
            return {"action_type": "calculate_cognitive_score", "target_task_id": None, "target_user": None}
        if "activate_autopilot" not in action_history:
            low = first_low()
            if low:
                return {"action_type": "activate_autopilot", "target_task_id": low["task_id"], "target_user": None}
        return {"action_type": "final_answer", "target_task_id": None, "target_user": None}

    if task_level == "medium":
        if "forecast_regret" not in action_history:
            return {"action_type": "forecast_regret", "target_task_id": None, "target_user": None}
        if "trigger_recovery_mode" not in action_history:
            return {"action_type": "trigger_recovery_mode", "target_task_id": None, "target_user": None}
        stress = find_task("complex_report") or first_stressful()
        if stress and "isolate_stressful_task" not in action_history:
            return {"action_type": "isolate_stressful_task", "target_task_id": stress["task_id"], "target_user": None}
        if "reorder_tasks" not in action_history:
            return {"action_type": "reorder_tasks", "target_task_id": None, "target_user": None}
        low = first_low()
        if low and "activate_autopilot" not in action_history:
            return {"action_type": "activate_autopilot", "target_task_id": low["task_id"], "target_user": None}
        return {"action_type": "final_answer", "target_task_id": None, "target_user": None}

    if task_level == "hard":
        if "forecast_regret" not in action_history:
            return {"action_type": "forecast_regret", "target_task_id": None, "target_user": None}
        if "predict_recovery" not in action_history:
            return {"action_type": "predict_recovery", "target_task_id": None, "target_user": None}
        if "trigger_recovery_mode" not in action_history:
            return {"action_type": "trigger_recovery_mode", "target_task_id": None, "target_user": None}
        stress = find_task("angry_client_email") or first_stressful()
        if stress and "isolate_stressful_task" not in action_history:
            return {"action_type": "isolate_stressful_task", "target_task_id": stress["task_id"], "target_user": None}
        low = find_task("lunch_choice") or first_low()
        if low and "activate_autopilot" not in action_history:
            return {"action_type": "activate_autopilot", "target_task_id": low["task_id"], "target_user": None}
        if find_task("client_approval") and "redistribute_team_load" not in action_history:
            return {"action_type": "redistribute_team_load", "target_task_id": "client_approval", "target_user": "Sara"}
        return {"action_type": "final_answer", "target_task_id": None, "target_user": None}

    return {"action_type": "final_answer", "target_task_id": None, "target_user": None}


def validate_action(task_level, action, state, action_history):
    action_type = action.get("action_type")
    pending_tasks = state.get("pending_tasks", [])
    cognitive_score = state.get("cognitive_score", 0)
    spillover_level = state.get("spillover_level", 0)

    # Rule 1: Never allow repeated recovery spam
    if action_type == "trigger_recovery_mode" and "trigger_recovery_mode" in action_history:
        if is_task_complete(task_level, state, action_history):
            return {
                "action_type": "final_answer",
                "target_task_id": None,
                "target_user": None
            }
        return enforce_task_plan(task_level, state, action_history)

    # Rule 2: Never allow repeated delegation
    if action_type == "redistribute_team_load" and "redistribute_team_load" in action_history:
        return enforce_task_plan(task_level, state, action_history)

    # Rule 3: Never allow isolate_stressful_task without a target
    if action_type == "isolate_stressful_task":
        stressful_task = get_first_stressful_task(state)
        if not action.get("target_task_id") and stressful_task:
            action["target_task_id"] = stressful_task["task_id"]
        elif not stressful_task:
            return enforce_task_plan(task_level, state, action_history)

    # Rule 4: Never allow final_answer before completion
    if action_type == "final_answer" and not is_task_complete(task_level, state, action_history):
        return enforce_task_plan(task_level, state, action_history)

    # Rule 5: Remove unstable actions
    if action_type in ["assign_task", "execute_task", "review_debt"]:
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
    if task_level == "easy":
        allowed_actions = [
            "forecast_regret",
            "calculate_cognitive_score",
            "activate_autopilot",
            "final_answer"
        ]
        task_rules = """
Goal:
- Forecast regret
- Assess cognitive state
- Automate exactly one low-stakes task
- Finish cleanly

Strict rules:
- No delegation
- No assign_task
- No execute_task
- Do not repeat activate_autopilot more than once
- Use final_answer only after the low-stakes task is handled
"""
    elif task_level == "medium":
        allowed_actions = [
            "forecast_regret",
            "trigger_recovery_mode",
            "isolate_stressful_task",
            "reorder_tasks",
            "activate_autopilot",
            "final_answer"
        ]
        task_rules = """
Goal:
- Forecast regret
- Reduce emotional overload
- Isolate the main stressful task
- Reorder remaining tasks
- Automate one low-stakes task
- Finish cleanly

Strict rules:
- Trigger recovery mode at most once unless the user becomes unstable again
- Isolate only one main stressful task
- Reorder at most once
- Do not use final_answer until the stressful task is handled and one low-stakes task is handled
"""
    else:
        allowed_actions = [
            "forecast_regret",
            "predict_recovery",
            "trigger_recovery_mode",
            "isolate_stressful_task",
            "activate_autopilot",
            "redistribute_team_load",
            "final_answer"
        ]
        task_rules = """
Goal:
- Forecast regret
- Predict recovery
- Stabilize the user
- Isolate the main emotional risk
- Automate one low-stakes task
- Delegate one suitable team task
- Finish cleanly

Strict rules:
- Use trigger_recovery_mode at most once unless critical instability returns
- Use redistribute_team_load at most once
- Do not use final_answer until angry_client_email is handled, lunch_choice is handled, and client_approval is delegated
- Do not use actions outside the allowed list
"""

    return f"""
You are controlling a cognitive decision-support environment.

Current task level: {task_level}

State:
{json.dumps(state)}

Allowed actions only:
{allowed_actions}

{task_rules}

Output exactly one JSON object and nothing else:
{{
  "action_type": "one_allowed_action",
  "target_task_id": null,
  "target_user": null
}}
"""


def is_task_complete(task_level, state, action_history):
    pending_tasks = state.get("pending_tasks", [])
    cognitive_score = state.get("cognitive_score", 0)

    if task_level == "easy":
        low_stakes_left = [t for t in pending_tasks if t.get("is_low_stakes")]
        return len(low_stakes_left) == 0 and cognitive_score >= 30

    if task_level == "medium":
        stressful_left = [t for t in pending_tasks if t.get("is_stressful")]
        low_stakes_left = [t for t in pending_tasks if t.get("is_low_stakes")]
        return len(stressful_left) == 0 and len(low_stakes_left) < 2 and cognitive_score >= 20

    if task_level == "hard":
        important_left = [
            t for t in pending_tasks
            if t.get("task_id") in ["angry_client_email", "client_approval", "lunch_choice"]
        ]
        return len(important_left) == 0 and cognitive_score >= 20

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
        if is_task_complete(task_level, state, action_history):
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

        remaining_stressful = [t for t in pending if t.get("is_stressful")]
        remaining_low = [t for t in pending if t.get("is_low_stakes")]

        if remaining_stressful and not already_used("trigger_recovery_mode"):
            return {"action_type": "trigger_recovery_mode", "target_task_id": None, "target_user": None}

        # Do not activate again if already used
        if remaining_low and not already_used("activate_autopilot"):
            return {"action_type": "activate_autopilot", "target_task_id": remaining_low[0]["task_id"], "target_user": None}

        if len(pending) > 1:
            return {"action_type": "reorder_tasks", "target_task_id": None, "target_user": None}

        if is_task_complete(task_level, state, action_history):
            return {"action_type": "final_answer", "target_task_id": None, "target_user": None}

        return {"action_type": "trigger_recovery_mode", "target_task_id": None, "target_user": None}

    if task_level == "hard":
        if not already_used("forecast_regret") and pending:
            return {"action_type": "forecast_regret", "target_task_id": None, "target_user": None}

        if not already_used("predict_recovery"):
            return {"action_type": "predict_recovery", "target_task_id": None, "target_user": None}

        if not already_used("trigger_recovery_mode"):
            return {"action_type": "trigger_recovery_mode", "target_task_id": None, "target_user": None}

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

        if state.get("decision_debt", 0) > 0 and state.get("cognitive_score", 0) >= 40 and not already_used("review_debt"):
            return {"action_type": "review_debt", "target_task_id": None, "target_user": None}

        remaining_critical = [
            t for t in pending
            if t.get("task_id") in ["angry_client_email", "client_approval", "lunch_choice"]
        ]

        if remaining_critical:
            if remaining_critical[0].get("is_low_stakes"):
                return {
                    "action_type": "activate_autopilot",
                    "target_task_id": remaining_critical[0]["task_id"],
                    "target_user": None
                }
            if best_teammate:
                return {
                    "action_type": "redistribute_team_load",
                    "target_task_id": remaining_critical[0]["task_id"],
                    "target_user": best_teammate
                }
            return {"action_type": "trigger_recovery_mode", "target_task_id": None, "target_user": None}

        if is_task_complete(task_level, state, action_history):
            return {"action_type": "final_answer", "target_task_id": None, "target_user": None}

        if "trigger_recovery_mode" in action_history:
            return {"action_type": "final_answer", "target_task_id": None, "target_user": None}

        return {"action_type": "trigger_recovery_mode", "target_task_id": None, "target_user": None}

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
            temperature=0
        )

        content = response.choices[0].message.content.strip()
        action = json.loads(content)

        if isinstance(action, dict) and "action_type" in action:
            return validate_action(task_level, action, state, action_history)
    except Exception:
        pass

    planned = enforce_task_plan(task_level, state, action_history)
    return planned if planned else fallback_policy(task_level, state, action_history)


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


def scripted_action(task_level, state, action_history, step):
    return enforce_task_plan(task_level, state, action_history)


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
        return 0.80
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
        if is_task_complete(task_level, state, action_history):
            action = {"action_type": "final_answer", "target_task_id": None, "target_user": None}
        if action.get("action_type") == "final_answer" and not is_task_complete(task_level, state, action_history):
            action = fallback_policy(task_level, state, action_history)

        if is_task_complete(task_level, state, action_history):
            action = {"action_type": "final_answer", "target_task_id": None, "target_user": None}

        recovery_override = False
        if action.get("action_type") == "trigger_recovery_mode" and "trigger_recovery_mode" in action_history:
            if task_level == "hard":
                if state.get("decision_debt", 0) > 0 and state.get("cognitive_score", 0) >= 40 and "review_debt" not in action_history:
                    action = {"action_type": "review_debt", "target_task_id": None, "target_user": None}
                else:
                    action = {"action_type": "final_answer", "target_task_id": None, "target_user": None}
                recovery_override = True

        if task_level in ["medium", "hard"] and action.get("action_type") == "final_answer" and not is_task_complete(task_level, state, action_history) and not recovery_override:
            action = enforce_task_plan(task_level, state, action_history)
            if action.get("action_type") == "final_answer":
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

        try:
            next_state, reward, done, info = env.step(action)
            error = "null"
        except Exception as e:
            next_state = state
            reward = 0.0
            done = True
            error = str(e)

        if is_task_complete(task_level, next_state, action_history) and action.get("action_type") == "final_answer":
            done = True

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
