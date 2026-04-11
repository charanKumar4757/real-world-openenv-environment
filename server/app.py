"""
app.py — FastAPI + Gradio UI for the ACIE-HADO OpenEnv environment.

Improvements over previous version:
1. Gradio UI has clickable action buttons (no more raw JSON input)
2. Live cognitive score displayed prominently
3. Task queue shown as a readable table
4. Situation summary shown in plain English
5. Reward history visible after each step
6. All OpenEnv API endpoints preserved for validator
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.env import CognitiveEnv
from src.grader import grade_easy, grade_medium, grade_hard

import gradio as gr
from gradio.routes import mount_gradio_app

app = FastAPI(title="ACIE-HADO: Adaptive Cognitive Intelligence Environment")
env = CognitiveEnv()


# ── Request Models ────────────────────────────────────────────────────────────
class ActionRequest(BaseModel):
    action_type: str
    target_task_id: str | None = None
    target_user: str | None = None
    summary: str | None = None


# ── API Routes (required by OpenEnv validator) ────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "ACIE-HADO OpenEnv environment is running",
        "environment": "CognitiveEnv",
        "tasks": ["easy", "medium", "hard"],
        "ui": "/ui"
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/reset")
def reset(task_level: str = "easy"):
    try:
        return env.reset(task_level)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    try:
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(action: ActionRequest):
    try:
        observation, reward, done, info = env.step(action.model_dump())
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/grade")
def grade(task_level: str = "easy"):
    """Grade the current episode."""
    try:
        final_state = env.state()
        info = {
            "total_reward": env.total_reward,
            "step": env.step_count,
            "action_history": env.action_history,
            "isolated_tasks": env.isolated_tasks,
            "team_assignments": env.team_assignments,
            "autopilot_task_ids": env.autopilot_task_ids,
        }
        if task_level == "easy":
            score = grade_easy(info, final_state)
        elif task_level == "medium":
            score = grade_medium(info, final_state)
        elif task_level == "hard":
            score = grade_hard(info, final_state)
        else:
            raise ValueError(f"Invalid task level: {task_level}")
        return {"task_level": task_level, "score": score, "info": info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── UI State ──────────────────────────────────────────────────────────────────
reward_history = []
current_task_level = "easy"


def _format_tasks_as_table(tasks: list) -> str:
    if not tasks:
        return "✅ No pending tasks"
    lines = ["| Task | Urgency | Cog Cost | Type |", "|------|---------|----------|------|"]
    for t in tasks:
        label = t.get("label", t.get("task_id", "?"))
        urgency = t.get("urgency", 0)
        cost = t.get("cognitive_cost", 0)
        flags = []
        if t.get("is_low_stakes"):
            flags.append("🟢 Low-stakes")
        if t.get("is_stressful"):
            flags.append("🔴 Stressful")
        if not flags:
            flags.append("🔵 Normal")
        lines.append(f"| {label} | {urgency}/10 | {cost} | {', '.join(flags)} |")
    return "\n".join(lines)


def _score_bar(score: int) -> str:
    filled = int(score / 10)
    empty = 10 - filled
    if score >= 60:
        color = "🟩"
    elif score >= 35:
        color = "🟨"
    else:
        color = "🟥"
    return color * filled + "⬛" * empty + f"  {score}/100"


def _format_team(team: dict) -> str:
    if not team:
        return "No team available"
    return " | ".join(f"{k}: {v}/100" for k, v in team.items())


# ── UI Functions ──────────────────────────────────────────────────────────────
def ui_reset(task_level: str):
    global reward_history, current_task_level
    reward_history = []
    current_task_level = task_level
    obs = env.reset(task_level)
    return _build_ui_outputs(obs, f"✅ Environment reset for task: **{task_level}**", 0.0)


def ui_step(action_type: str, target_task: str, target_user: str):
    action = {
        "action_type": action_type,
        "target_task_id": target_task.strip() if target_task.strip() else None,
        "target_user": target_user.strip() if target_user.strip() else None,
    }
    obs, reward, done, info = env.step(action)
    reward_history.append(reward)

    status_msg = f"**Action:** `{action_type}` → **Reward:** `{reward:+.2f}`"
    if done:
        # Auto grade
        if current_task_level == "easy":
            grade_score = grade_easy(info, obs)
        elif current_task_level == "medium":
            grade_score = grade_medium(info, obs)
        else:
            grade_score = grade_hard(info, obs)
        status_msg += f"\n\n🏁 **Episode done!** Final grade: **{grade_score:.2f}/1.00**"

    return _build_ui_outputs(obs, status_msg, reward)


def ui_get_state():
    obs = env.state()
    return _build_ui_outputs(obs, "📊 Current state fetched", 0.0)


def _build_ui_outputs(obs: dict, status: str, last_reward: float):
    cog_score = obs.get("cognitive_score", 0)
    score_display = _score_bar(cog_score)
    situation = obs.get("situation_summary", "No summary available.")
    tasks_md = _format_tasks_as_table(obs.get("pending_tasks", []))
    team_display = _format_team(obs.get("team_scores", {}))

    state_details = (
        f"**Fatigue:** {obs.get('fatigue_level', '?')}  |  "
        f"**Emotional:** {obs.get('emotional_state', '?')}  |  "
        f"**Spillover:** {obs.get('spillover_level', 0)}  |  "
        f"**Debt:** {obs.get('decision_debt', 0)}  |  "
        f"**Steps:** {env.step_count}/12"
    )

    history_str = ""
    if reward_history:
        history_str = " → ".join(
            f"{r:+.2f}" for r in reward_history
        ) + f"  (total: {sum(reward_history):+.2f})"
    else:
        history_str = "No steps taken yet"

    action_hist = ", ".join(env.action_history) if env.action_history else "None"

    return (
        score_display,       # cognitive score bar
        situation,           # situation summary
        tasks_md,            # task table
        team_display,        # team scores
        state_details,       # state details row
        status,              # last action status
        history_str,         # reward history
        action_hist,         # action history
    )


# ── Gradio Blocks UI ──────────────────────────────────────────────────────────
with gr.Blocks(title="ACIE-HADO: Cognitive Load AI Environment", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# 🧠 ACIE-HADO: Adaptive Cognitive Intelligence Environment
**An AI agent learns to protect human cognitive resources by managing task queues, emotional spillover, decision debt, and team load.**

Choose a task level, reset the environment, then click actions to see how the agent interacts with the simulation.
""")

    # ── Controls ──
    with gr.Row():
        task_level_dd = gr.Dropdown(
            choices=["easy", "medium", "hard"],
            value="easy",
            label="Task Level",
            scale=1
        )
        reset_btn = gr.Button("🔄 Reset Environment", variant="primary", scale=2)

    # ── Score and Situation ──
    gr.Markdown("### Current State")
    with gr.Row():
        with gr.Column(scale=1):
            score_display = gr.Textbox(label="🧠 Cognitive Score", interactive=False, lines=1)
            state_details = gr.Textbox(label="Status Details", interactive=False, lines=2)
            team_display = gr.Textbox(label="👥 Team Cognitive Scores", interactive=False, lines=1)
        with gr.Column(scale=2):
            situation_box = gr.Textbox(
                label="📋 Situation Summary (what the agent sees)",
                interactive=False,
                lines=6
            )

    # ── Task Queue ──
    gr.Markdown("### Pending Tasks")
    task_table = gr.Markdown(label="Tasks")

    # ── Actions ──
    gr.Markdown("### Take an Action")
    with gr.Row():
        action_radio = gr.Radio(
            choices=[
                "calculate_cognitive_score",
                "predict_recovery",
                "activate_autopilot",
                "isolate_stressful_task",
                "trigger_recovery_mode",
                "reorder_tasks",
                "execute_task",
                "delay_task",
                "redistribute_team_load",
                "review_debt",
                "provide_transparency",
                "reserve_recovery_window",
                "forecast_regret",
                "apply_pattern_memory",
                "final_answer",
            ],
            label="Select Action",
            value="calculate_cognitive_score"
        )

    with gr.Row():
        target_task_input = gr.Textbox(
            label="Target Task ID (optional — e.g. lunch_order, angry_client_email)",
            placeholder="Leave blank to auto-select",
            scale=2
        )
        target_user_input = gr.Textbox(
            label="Target Team Member (optional — e.g. Sara, Ravi)",
            placeholder="Leave blank if not assigning",
            scale=1
        )

    step_btn = gr.Button("▶ Take Step", variant="primary")
    state_btn = gr.Button("👁 Refresh State")

    # ── Output ──
    gr.Markdown("### Results")
    status_box = gr.Markdown(label="Last Action Result")
    with gr.Row():
        reward_history_box = gr.Textbox(label="📈 Reward History", interactive=False, lines=2)
        action_history_box = gr.Textbox(label="📜 Action History", interactive=False, lines=2)

    # ── Task ID quick reference ──
    with gr.Accordion("Task ID Reference", open=False):
        gr.Markdown("""
**Easy task IDs:** `choose_elective`, `lunch_order`

**Medium task IDs:** `complex_report`, `reply_client_email`, `routine_status_update`

**Hard task IDs:** `angry_client_email`, `weekly_planning`, `lunch_choice`, `client_approval`, `confirm_presentation`

**Team members (hard only):** `Sara`, `Ravi`
""")

    # ── Wire buttons ──
    outputs = [
        score_display, situation_box, task_table, team_display,
        state_details, status_box, reward_history_box, action_history_box
    ]

    reset_btn.click(fn=ui_reset, inputs=[task_level_dd], outputs=outputs)
    step_btn.click(
        fn=ui_step,
        inputs=[action_radio, target_task_input, target_user_input],
        outputs=outputs
    )
    state_btn.click(fn=ui_get_state, inputs=[], outputs=outputs)


# ── Mount UI and validator entry ──────────────────────────────────────────────
app = mount_gradio_app(app, demo, path="/ui")


def main():
    print("ACIE-HADO server ready for OpenEnv validation.")


if __name__ == "__main__":
    main()