"""
app.py — ACIE-HADO Gradio UI + FastAPI OpenEnv endpoints
=========================================================
Phase 3 fix: UI now has CLICKABLE ACTION BUTTONS (not just a JSON text box).
Each button is pre-wired to a specific action and updates the state display.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.env import CognitiveEnv
import gradio as gr
from gradio.routes import mount_gradio_app
import json

app = FastAPI(title="ACIE-HADO: Real World OpenEnv Environment")
env = CognitiveEnv()


# ── Request model ─────────────────────────────────────────────────
class ActionRequest(BaseModel):
    action_type: str
    target_task_id: str | None = None
    target_user: str | None = None


# ── API Routes (required by OpenEnv validator) ────────────────────
@app.get("/")
def root():
    return {"status": "ok", "message": "ACIE-HADO OpenEnv environment is running"}

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
        if env.current_state is None:
            raise HTTPException(status_code=400, detail="Call /reset first.")
        observation, reward, done, info = env.step(action.model_dump())
        return {"observation": observation, "reward": reward, "done": done, "info": info}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── UI helpers ────────────────────────────────────────────────────

def format_state(obs: dict) -> str:
    if not obs or not obs.get("cognitive_score"):
        return "Click 'Reset Environment' to start."
    lines = [
        f"Cognitive Score:  {obs.get('cognitive_score', '?')}/100",
        f"Fatigue Level:    {obs.get('fatigue_level', '?')}",
        f"Emotional State:  {obs.get('emotional_state', '?')}",
        f"Spillover Level:  {obs.get('spillover_level', 0)}/100",
        f"Decision Debt:    {obs.get('decision_debt', 0)} deferred",
        f"Trust Score:      {obs.get('human_trust_score', '?')}/100",
        f"Autopilot:        {'ON' if obs.get('autopilot_active') else 'OFF'}",
        f"Recovery ETA:     {obs.get('recovery_prediction', 0)} min",
        "",
    ]
    pending = obs.get("pending_tasks", [])
    lines.append(f"Pending Tasks ({len(pending)}):")
    for t in pending:
        flags = []
        if t.get("is_stressful"): flags.append("STRESSFUL")
        if t.get("is_low_stakes"): flags.append("low-stakes")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        lines.append(f"  - {t['task_id']} | urgency={t['urgency']} | cost={t['cognitive_cost']}{flag_str}")

    team = obs.get("team_scores", {})
    if team:
        lines.append("")
        lines.append("Team Scores:")
        for user, s in team.items():
            bar = "HIGH" if s > 60 else ("MED" if s > 30 else "LOW")
            lines.append(f"  - {user}: {s}/100 [{bar}]")

    summary = obs.get("situation_summary", "")
    if summary:
        lines.append("")
        lines.append("Situation:")
        # Word-wrap the summary at 80 chars
        words = summary.split()
        line = ""
        for w in words:
            if len(line) + len(w) + 1 > 78:
                lines.append(f"  {line}")
                line = w
            else:
                line = f"{line} {w}".strip()
        if line:
            lines.append(f"  {line}")

    return "\n".join(lines)


def ui_reset(task_level):
    try:
        obs = env.reset(task_level)
        history = "No actions yet."
        feedback = f"Environment reset. Task: {task_level.upper()}. Use the buttons below to step."
        return format_state(obs), feedback, history
    except Exception as e:
        return f"Error: {e}", str(e), ""


def ui_do_action(action_type, target_task_id=None, target_user=None):
    """Core step function used by all clickable buttons."""
    if env.current_state is None:
        return (
            "Click 'Reset Environment' first.",
            "ERROR: Environment not initialized.",
            ""
        )
    try:
        action = {"action_type": action_type, "target_task_id": target_task_id, "target_user": target_user}
        obs, reward, done, info = env.step(action)
        history = "\n".join(
            f"Step {i+1}: {a}" for i, a in enumerate(env.action_history)
        )
        sign = "+" if reward >= 0 else ""
        status = "EPISODE DONE — click Reset to start again." if done else "Continue."
        feedback = f"Action: {action_type}  |  Reward: {sign}{reward:.2f}  |  {status}"
        return format_state(obs), feedback, history
    except Exception as e:
        return format_state(env.state()) if env.current_state else "", f"Error: {e}", ""


def ui_custom_step(action_json):
    """Allow custom JSON action input."""
    if not action_json or not action_json.strip():
        return format_state(env.state()) if env.current_state else "", "Enter action JSON first.", ""
    try:
        action = json.loads(action_json)
        return ui_do_action(
            action.get("action_type", "final_answer"),
            action.get("target_task_id"),
            action.get("target_user")
        )
    except json.JSONDecodeError as e:
        return format_state(env.state()) if env.current_state else "", f"Invalid JSON: {e}", ""


def ui_get_state():
    if env.current_state is None:
        return "Click Reset first.", "Not started.", ""
    obs = env.state()
    history = "\n".join(
        f"Step {i+1}: {a}" for i, a in enumerate(env.action_history)
    ) or "No actions yet."
    return format_state(obs), "State refreshed.", history


def ui_run_simulation():
    """Automatically run the optimal action sequence for all 3 tasks."""
    from src.grader import grade_easy, grade_medium, grade_hard
    results = []
    sequences = {
        "easy": [
            {"action_type": "calculate_cognitive_score"},
            {"action_type": "activate_autopilot"},
            {"action_type": "final_answer"},
        ],
        "medium": [
            {"action_type": "forecast_regret"},
            {"action_type": "trigger_recovery_mode"},
            {"action_type": "isolate_stressful_task"},
            {"action_type": "reorder_tasks"},
            {"action_type": "activate_autopilot"},
            {"action_type": "final_answer"},
        ],
        "hard": [
            {"action_type": "forecast_regret"},
            {"action_type": "predict_recovery"},
            {"action_type": "trigger_recovery_mode"},
            {"action_type": "isolate_stressful_task"},
            {"action_type": "activate_autopilot"},
            {"action_type": "redistribute_team_load"},
            {"action_type": "final_answer"},
        ]
    }
    graders = {"easy": grade_easy, "medium": grade_medium, "hard": grade_hard}
    total = 0.0
    for level, actions in sequences.items():
        env.reset(level)
        info = {}
        for action in actions:
            _, _, done, info = env.step(action)
            if done:
                break
        final_state = env.state()
        raw = graders[level](info, final_state)
        score = max(0.01, min(0.99, raw))
        total += score
        status = "PASS" if score >= 0.40 else "FAIL"
        results.append(f"{level.upper():8s}  score={score:.2f}  [{status}]")
    avg = total / 3
    results.append(f"\nAverage score: {avg:.2f}")
    return "\n".join(results)


# ── Gradio UI ─────────────────────────────────────────────────────
with gr.Blocks(title="ACIE-HADO Environment") as demo:

    gr.Markdown("## ACIE-HADO: Human-AI Decision Optimization Environment")
    gr.Markdown("An RL environment where AI agents learn to manage human cognitive load and decision fatigue.")

    with gr.Row():

        # Left column — controls
        with gr.Column(scale=1):
            gr.Markdown("### Setup")
            task_dd = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Task Level"
            )
            reset_btn = gr.Button("Reset Environment", variant="primary")
            get_state_btn = gr.Button("Refresh State")
            run_sim_btn = gr.Button("Run Full Simulation (all 3 tasks)")

            gr.Markdown("### Quick Actions")
            gr.Markdown("Click any button to execute that action immediately:")

            # CLICKABLE ACTION BUTTONS — Phase 3 requirement
            btn_calc = gr.Button("Calculate Cognitive Score")
            btn_predict = gr.Button("Predict Recovery")
            btn_forecast = gr.Button("Forecast Regret")
            btn_recovery = gr.Button("Trigger Recovery Mode")
            btn_isolate = gr.Button("Isolate Stressful Task")
            btn_reorder = gr.Button("Reorder Tasks")
            btn_autopilot = gr.Button("Activate Autopilot")
            btn_redistribute = gr.Button("Redistribute Team Load")
            btn_transparency = gr.Button("Provide Transparency")
            btn_reserve = gr.Button("Reserve Recovery Window")
            btn_final = gr.Button("Final Answer (end episode)", variant="stop")

            gr.Markdown("### Custom Action (JSON)")
            action_input = gr.Textbox(
                label="Custom action JSON",
                placeholder='{"action_type": "isolate_stressful_task", "target_task_id": "angry_client_email"}',
                lines=3
            )
            custom_btn = gr.Button("Run Custom Action")

        # Right column — state display
        with gr.Column(scale=2):
            gr.Markdown("### Environment State")
            state_display = gr.Textbox(
                label="Current State",
                lines=22,
                interactive=False,
                value="Click 'Reset Environment' to begin."
            )
            feedback_display = gr.Textbox(
                label="Last Action Feedback",
                lines=3,
                interactive=False
            )
            history_display = gr.Textbox(
                label="Action History",
                lines=6,
                interactive=False,
                value="No actions yet."
            )

    sim_output = gr.Textbox(
        label="Simulation Results",
        lines=7,
        interactive=False
    )

    # ── Wire up all buttons ───────────────────────────────────────
    OUTS = [state_display, feedback_display, history_display]

    reset_btn.click(fn=ui_reset, inputs=[task_dd], outputs=OUTS)
    get_state_btn.click(fn=ui_get_state, inputs=[], outputs=OUTS)
    run_sim_btn.click(fn=ui_run_simulation, inputs=[], outputs=[sim_output])
    custom_btn.click(fn=ui_custom_step, inputs=[action_input], outputs=OUTS)

    # Each clickable button calls ui_do_action with its action_type baked in
    btn_calc.click(fn=lambda: ui_do_action("calculate_cognitive_score"), inputs=[], outputs=OUTS)
    btn_predict.click(fn=lambda: ui_do_action("predict_recovery"), inputs=[], outputs=OUTS)
    btn_forecast.click(fn=lambda: ui_do_action("forecast_regret"), inputs=[], outputs=OUTS)
    btn_recovery.click(fn=lambda: ui_do_action("trigger_recovery_mode"), inputs=[], outputs=OUTS)
    btn_isolate.click(fn=lambda: ui_do_action("isolate_stressful_task"), inputs=[], outputs=OUTS)
    btn_reorder.click(fn=lambda: ui_do_action("reorder_tasks"), inputs=[], outputs=OUTS)
    btn_autopilot.click(fn=lambda: ui_do_action("activate_autopilot"), inputs=[], outputs=OUTS)
    btn_redistribute.click(fn=lambda: ui_do_action("redistribute_team_load"), inputs=[], outputs=OUTS)
    btn_transparency.click(fn=lambda: ui_do_action("provide_transparency"), inputs=[], outputs=OUTS)
    btn_reserve.click(fn=lambda: ui_do_action("reserve_recovery_window"), inputs=[], outputs=OUTS)
    btn_final.click(fn=lambda: ui_do_action("final_answer"), inputs=[], outputs=OUTS)


# ── Mount UI at /ui ───────────────────────────────────────────────
app = mount_gradio_app(app, demo, path="/ui")


def main():
    print("ACIE-HADO server ready for OpenEnv validation.")


if __name__ == "__main__":
    main()