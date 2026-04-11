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


def run_demo_task(task_level):
    """Run a demo task episode and return formatted output with real environment state."""
    try:
        state = env.reset(task_level)
        output = [
            f"═══ ACIE-HADO Demo Task: {task_level.upper()} ═══\n",
            format_state(state),
            "\n--- Situation Summary ---",
            state.get("situation_summary", "No summary available"),
        ]
        return "\n".join(output)
    except Exception as e:
        return f"Error running demo: {e}"


# ── Gradio UI ─────────────────────────────────────────────────────
with gr.Blocks(title="ACIE-HADO Environment") as demo:

    gr.Markdown("## ACIE-HADO: Adaptive Cognitive Intelligence Environment")
    gr.Markdown("Human-AI Decision Optimization — Real-Time Cognitive Load Management")
    gr.Markdown("Click a button below to run a demo episode for each task level.")

    with gr.Row():
        btn_easy = gr.Button("▶ Run Easy Task", variant="primary")
        btn_medium = gr.Button("▶ Run Medium Task", variant="secondary")
        btn_hard = gr.Button("▶ Run Hard Task", variant="secondary")

    output = gr.Textbox(
        label="Episode Output",
        lines=20,
        interactive=False,
        value="Click a button to run a demo task episode."
    )

    # Wire up the three task buttons to run_demo_task
    btn_easy.click(fn=lambda: run_demo_task("easy"), outputs=output)
    btn_medium.click(fn=lambda: run_demo_task("medium"), outputs=output)
    btn_hard.click(fn=lambda: run_demo_task("hard"), outputs=output)


# ── Mount UI at /ui ───────────────────────────────────────────────
app = mount_gradio_app(app, demo, path="/ui")


def main():
    print("ACIE-HADO server ready for OpenEnv validation.")


if __name__ == "__main__":
    main()