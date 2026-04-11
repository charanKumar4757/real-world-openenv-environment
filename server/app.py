from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.env import CognitiveEnv

import gradio as gr
from gradio.routes import mount_gradio_app
import json

app = FastAPI(title="ACIE-HADO: Real World OpenEnv Environment")
env = CognitiveEnv()


# ---------- Request Model ----------
class ActionRequest(BaseModel):
    action_type: str
    target_task_id: str | None = None
    target_user: str | None = None


# ---------- API Routes (Required by OpenEnv validator) ----------
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
        # FIXED: Guard against stepping before reset
        if env.current_state is None:
            raise HTTPException(
                status_code=400,
                detail="Environment not initialized. Call /reset first."
            )
        observation, reward, done, info = env.step(action.model_dump())
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- UI HELPER FUNCTIONS ----------

def format_state_display(state: dict) -> str:
    """Format the state dict into a readable string for the UI."""
    if not state:
        return "No state yet. Click 'Reset Environment' first."
    lines = []
    lines.append(f"🧠 Cognitive Score: {state.get('cognitive_score', '?')}/100")
    lines.append(f"😴 Fatigue Level: {state.get('fatigue_level', '?')}")
    lines.append(f"😤 Emotional State: {state.get('emotional_state', '?')}")
    lines.append(f"⚡ Spillover Level: {state.get('spillover_level', 0)}/100")
    lines.append(f"📋 Decision Debt: {state.get('decision_debt', 0)} deferred decisions")
    lines.append(f"🤖 Autopilot Active: {state.get('autopilot_active', False)}")
    lines.append(f"🤝 Human Trust Score: {state.get('human_trust_score', '?')}/100")
    lines.append(f"⏱️ Recovery Prediction: {state.get('recovery_prediction', 0)} minutes")

    pending = state.get("pending_tasks", [])
    lines.append(f"\n📌 Pending Tasks ({len(pending)}):")
    for t in pending:
        stress = "⚠️ STRESSFUL" if t.get("is_stressful") else ""
        low = "✅ low-stakes" if t.get("is_low_stakes") else ""
        lines.append(f"  • {t['task_id']} | urgency={t['urgency']} | cost={t['cognitive_cost']} {stress}{low}")

    team = state.get("team_scores", {})
    if team:
        lines.append(f"\n👥 Team Scores:")
        for user, score in team.items():
            bar = "🟢" if score > 60 else ("🟡" if score > 30 else "🔴")
            lines.append(f"  {bar} {user}: {score}/100")

    return "\n".join(lines)


def ui_reset(task_level):
    """Reset the environment and return the initial state."""
    try:
        observation = env.reset(task_level)
        state_display = format_state_display(observation)
        valid_actions = get_valid_actions_hint(observation)
        feedback = f"✅ Environment reset for task: {task_level.upper()}\n\n{valid_actions}"
        history = "No actions taken yet."
        return state_display, feedback, history
    except Exception as e:
        return f"Error: {e}", f"Error resetting: {e}", ""


def get_valid_actions_hint(state: dict) -> str:
    """Give the user a hint about what actions make sense right now."""
    hints = ["💡 Suggested next actions:"]
    cognitive = state.get("cognitive_score", 50)
    emotional = state.get("emotional_state", "neutral")
    spillover = state.get("spillover_level", 0)
    pending = state.get("pending_tasks", [])

    if cognitive > 60:
        hints.append('  → {"action_type": "calculate_cognitive_score"} — assess situation first')
        hints.append('  → {"action_type": "reorder_tasks"} — prioritize tasks')
    elif cognitive < 40:
        hints.append('  → {"action_type": "trigger_recovery_mode"} — user is tired, rest first')
        hints.append('  → {"action_type": "activate_autopilot"} — automate low-stakes tasks')

    if spillover > 20 or emotional in ["stressed", "overwhelmed"]:
        hints.append('  → {"action_type": "isolate_stressful_task", "target_task_id": "stressful_email"} — protect from emotional spillover')

    team = state.get("team_scores", {})
    capable = [u for u, s in team.items() if s > 60]
    if capable and cognitive < 30:
        hints.append(f'  → {{"action_type": "redistribute_team_load"}} — delegate to {capable[0]}')

    return "\n".join(hints)


def ui_get_state():
    """Return current state of the environment."""
    try:
        # FIXED: Guard against state() before reset
        if env.current_state is None:
            return "Click 'Reset Environment' first.", "Not started yet.", ""
        observation = env.state()
        state_display = format_state_display(observation)
        history = "\n".join([f"Step {i+1}: {a}" for i, a in enumerate(env.action_history)]) if env.action_history else "No actions taken yet."
        return state_display, "State fetched.", history
    except Exception as e:
        return f"Error: {e}", f"Error: {e}", ""


def ui_step(action_json):
    """Execute one action step in the environment."""
    try:
        # FIXED: Guard against step before reset
        if env.current_state is None:
            return (
                "⚠️ Please click 'Reset Environment' first before stepping.",
                "❌ Error: Environment not initialized.",
                ""
            )

        if not action_json or not action_json.strip():
            return (
                format_state_display(env.state()),
                "❌ Error: Please enter an action in the JSON box first.",
                ""
            )

        action = json.loads(action_json)
        observation, reward, done, info = env.step(action)

        state_display = format_state_display(observation)
        history = "\n".join([f"Step {i+1}: {a}" for i, a in enumerate(env.action_history)])

        reward_emoji = "✅" if reward > 0 else ("⚠️" if reward == 0 else "❌")
        status = "🏁 EPISODE DONE" if done else "🔄 Continue"

        feedback_lines = [
            f"{reward_emoji} Reward: {reward:+.2f}",
            f"{status}",
            f"Total reward so far: {info.get('total_reward', 0):.2f}",
            f"Step: {info.get('step', 0)}/10",
            "",
        ]

        if done:
            feedback_lines.append("Episode complete! Click Reset to start a new task.")
        else:
            feedback_lines.append(get_valid_actions_hint(observation))

        feedback = "\n".join(feedback_lines)
        return state_display, feedback, history

    except json.JSONDecodeError as e:
        return (
            format_state_display(env.state()) if env.current_state else "",
            f"❌ Invalid JSON: {e}\n\nExample: {{\"action_type\": \"calculate_cognitive_score\"}}",
            ""
        )
    except Exception as e:
        return (
            format_state_display(env.state()) if env.current_state else "",
            f"❌ Error: {e}",
            ""
        )


def ui_run_all():
    """Run a full simulation automatically across all 3 tasks."""
    from src.grader import grade_easy, grade_medium, grade_hard

    results = []
    all_actions = {
        "easy": [
            {"action_type": "calculate_cognitive_score"},
            {"action_type": "activate_autopilot", "target_task_id": None},
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

    for task_level, actions in all_actions.items():
        env.reset(task_level)
        info = {}
        for action in actions:
            _, _, done, info = env.step(action)
            if done:
                break
        final_state = env.state()
        score = graders[task_level](info, final_state)
        score = max(0.0, min(1.0, score))
        results.append(f"{'🟢' if score >= 0.4 else '🔴'} {task_level.upper()} task: score = {score:.2f} {'✅ PASS' if score >= 0.4 else '❌ FAIL'}")

    results.append(f"\n📊 Average: {sum([float(r.split('= ')[1].split()[0]) for r in results[:3]]) / 3:.2f}")
    return "\n".join(results)


# ---------- GRADIO UI ----------
with gr.Blocks(title="ACIE-HADO: Cognitive AI Environment") as demo:
    gr.Markdown("""
# 🧠 ACIE-HADO: Human-AI Decision Optimization Environment
**An RL environment where AI agents learn to manage human cognitive load**

---
""")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Controls")
            task_level = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Select Task Level"
            )
            reset_btn = gr.Button("🔄 Reset Environment", variant="primary")
            state_btn = gr.Button("📊 Get State")
            run_all_btn = gr.Button("🚀 Run Full Simulation (All 3 Tasks)", variant="secondary")

            gr.Markdown("### 🎮 Manual Step")
            gr.Markdown("""**Available actions:**
- `calculate_cognitive_score`
- `predict_recovery`
- `activate_autopilot`
- `trigger_recovery_mode`
- `isolate_stressful_task`
- `reorder_tasks`
- `redistribute_team_load`
- `forecast_regret`
- `provide_transparency`
- `reserve_recovery_window`
- `final_answer`
""")
            action_input = gr.Textbox(
                label="Action (JSON)",
                placeholder='{"action_type": "calculate_cognitive_score"}',
                lines=3
            )
            step_btn = gr.Button("▶️ Step", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### 🔍 Environment State")
            current_observation = gr.Textbox(
                label="Current State",
                lines=18,
                interactive=False,
                value="Click '🔄 Reset Environment' to begin."
            )

            gr.Markdown("### 💬 Feedback")
            step_feedback = gr.Textbox(
                label="Step Feedback & Suggestions",
                lines=8,
                interactive=False
            )

            gr.Markdown("### 📜 Action History")
            action_history = gr.Textbox(
                label="Actions Taken This Episode",
                lines=5,
                interactive=False,
                value="No actions taken yet."
            )

    # Simulation results panel
    sim_results = gr.Textbox(
        label="🏆 Full Simulation Results",
        lines=6,
        interactive=False,
        visible=True
    )

    # Wire up buttons
    reset_btn.click(
        fn=ui_reset,
        inputs=[task_level],
        outputs=[current_observation, step_feedback, action_history]
    )

    step_btn.click(
        fn=ui_step,
        inputs=[action_input],
        outputs=[current_observation, step_feedback, action_history]
    )

    state_btn.click(
        fn=ui_get_state,
        inputs=[],
        outputs=[current_observation, step_feedback, action_history]
    )

    run_all_btn.click(
        fn=ui_run_all,
        inputs=[],
        outputs=[sim_results]
    )


# ---------- MOUNT UI AT /ui ----------
app = mount_gradio_app(app, demo, path="/ui")


# ---------- REQUIRED FOR VALIDATOR ----------
def main():
    print("ACIE-HADO server ready for OpenEnv validation.")


if __name__ == "__main__":
    main()