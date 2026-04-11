"""
inference.py — Baseline inference script for ACIE-HADO OpenEnv environment.

MANDATORY FORMAT (do not change):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables:
    API_BASE_URL   - LLM API endpoint (default: HuggingFace router)
    MODEL_NAME     - Model to use (default: Qwen2.5-72B-Instruct)
    HF_TOKEN       - HuggingFace API key (required)
"""

import os
import json
import sys
from openai import OpenAI
from src.env import CognitiveEnv
from src.grader import grade_easy, grade_medium, grade_hard

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = "acie-hado"
MAX_STEPS = 10

# ── Client ────────────────────────────────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "no-key"
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an AI agent managing human cognitive resources in a simulation environment.

ENVIRONMENT: ACIE-HADO (Adaptive Cognitive Intelligence Environment for Human-AI Decision Optimization)

YOUR GOAL: Help a human user manage their cognitive load by taking actions that protect their mental energy,
reduce emotional spillover, automate trivial decisions, and route work to available team members.

OBSERVATION: Each step you receive a JSON observation with:
- cognitive_score (0-100): User's mental energy. Below 30 = danger zone.
- fatigue_level: low / medium / high / overwhelmed
- emotional_state: neutral / guarded / stressed / overwhelmed
- spillover_level (0-100): Emotional contamination from stressful tasks
- pending_tasks: List of tasks with task_id, urgency, cognitive_cost, is_stressful, is_low_stakes
- team_scores: Dict of team member names and their cognitive scores
- decision_debt: Number of deferred decisions
- situation_summary: PLAIN ENGLISH explanation of the situation — READ THIS CAREFULLY

AVAILABLE ACTIONS (respond with exactly one):
- calculate_cognitive_score: Assess current cognitive state (small cost)
- predict_recovery: Estimate when user will recover (small cost)
- activate_autopilot: Automatically handle a low-stakes task (set target_task_id)
- isolate_stressful_task: Remove a stressful task from active queue to stop spillover (set target_task_id)
- trigger_recovery_mode: Emergency recovery — restores score and clears stress
- reorder_tasks: Sort tasks by urgency/cost ratio for optimal order
- execute_task: Directly complete a task (ONLY for non-stressful tasks when score > 35, set target_task_id)
- delay_task: Postpone a task to save cognitive resources (set target_task_id)
- redistribute_team_load: Assign heavy/stressful tasks to fresh team members
- review_debt: Clear deferred decisions (only when score > 35)
- provide_transparency: Explain AI actions to user (builds trust)
- reserve_recovery_window: Block time for recovery before a high-stakes event
- forecast_regret: Estimate risk of deferring current tasks
- final_answer: End the episode (call only when tasks are handled or you are done)

STRATEGY RULES:
1. ALWAYS read situation_summary — it tells you exactly what to do.
2. If cognitive_score < 30, use trigger_recovery_mode or redistribute_team_load FIRST.
3. If spillover_level > 30, use isolate_stressful_task FIRST.
4. Use activate_autopilot for tasks where is_low_stakes=true.
5. NEVER use execute_task when cognitive_score < 30 — it will drain the user further.
6. Use redistribute_team_load when team members are available and score > 50.
7. Call final_answer when all critical tasks are handled.

RESPONSE FORMAT: Respond with ONLY a JSON object, no other text:
{
  "action_type": "<action_name>",
  "target_task_id": "<task_id or null>",
  "target_user": "<team member name or null>",
  "reasoning": "<one sentence explanation>"
}"""


def get_action_from_llm(observation: dict, step_num: int) -> dict:
    """Ask the LLM what action to take given the current observation."""
    obs_text = json.dumps(observation, indent=2)
    user_msg = f"Step {step_num}. Current observation:\n{obs_text}\n\nWhat action do you take? Respond with JSON only."

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=300,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ]
        )
        content = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        action = json.loads(content.strip())
        return {
            "action_type": action.get("action_type", "calculate_cognitive_score"),
            "target_task_id": action.get("target_task_id"),
            "target_user": action.get("target_user"),
        }
    except Exception as e:
        # Fallback to safe default action
        return {"action_type": "calculate_cognitive_score", "target_task_id": None, "target_user": None}


def get_grade(task_level: str, info: dict, final_obs: dict) -> float:
    if task_level == "easy":
        return grade_easy(info, final_obs)
    elif task_level == "medium":
        return grade_medium(info, final_obs)
    elif task_level == "hard":
        return grade_hard(info, final_obs)
    return 0.0


def run_task(task_level: str) -> float:
    """Run one full episode for a given task level. Returns final score 0.0-1.0."""
    env = CognitiveEnv()
    observation = env.reset(task_level)

    rewards = []
    done = False
    last_error = None
    info = {}

    print(f"[START] task={task_level} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    step_num = 0
    while not done and step_num < MAX_STEPS:
        step_num += 1

        # Get action from LLM
        action = get_action_from_llm(observation, step_num)
        action_str = action.get("action_type", "calculate_cognitive_score")
        if action.get("target_task_id"):
            action_str += f"(target={action['target_task_id']})"
        if action.get("target_user"):
            action_str += f"(user={action['target_user']})"

        # Step the environment
        try:
            observation, reward, done, info = env.step(action)
            rewards.append(reward)
            last_error = None
        except Exception as e:
            last_error = str(e).replace("\n", " ")
            rewards.append(0.0)
            done = True

        error_str = last_error if last_error else "null"
        done_str = "true" if done else "false"

        print(
            f"[STEP] step={step_num} action={action_str} "
            f"reward={reward:.2f} done={done_str} error={error_str}",
            flush=True
        )

    # Grade the episode
    final_score = get_grade(task_level, info, observation)
    success = final_score >= 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={'true' if success else 'false'} steps={step_num} "
        f"score={final_score:.2f} rewards={rewards_str}",
        flush=True
    )

    return final_score


def main():
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set. LLM calls will fail. Set HF_TOKEN env var.", file=sys.stderr)

    tasks = ["easy", "medium", "hard"]
    scores = {}

    for task in tasks:
        try:
            score = run_task(task)
            scores[task] = score
        except Exception as e:
            print(f"[END] success=false steps=0 score=0.00 rewards=", flush=True)
            scores[task] = 0.0

    avg = sum(scores.values()) / len(scores)
    print(f"\n# Final Results: easy={scores.get('easy',0):.2f} medium={scores.get('medium',0):.2f} hard={scores.get('hard',0):.2f} average={avg:.2f}", flush=True)


if __name__ == "__main__":
    main()
