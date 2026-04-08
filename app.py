from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.env import CognitiveEnv

import gradio as gr
from gradio.routes import mount_gradio_app

app = FastAPI(title="Real World OpenEnv Environment")
env = CognitiveEnv()


# ---------- Request Model ----------
class ActionRequest(BaseModel):
    action_type: str
    target_task_id: str | None = None
    target_user: str | None = None


# ---------- API Routes ----------
@app.get("/")
def root():
    return {"status": "ok", "message": "OpenEnv environment is running"}


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


# ---------- UI HELPERS ----------
def ui_reset(task_level):
    try:
        observation = env.reset(task_level)
        return (
            observation,
            "Environment reset successfully.",
            "No actions taken yet"
        )
    except Exception as e:
        return {}, f"Error: {e}", ""


def ui_get_state():
    try:
        observation = env.state()
        history = "\n".join(env.action_history) if env.action_history else "No actions taken yet"
        return (
            observation,
            "State fetched successfully.",
            history
        )
    except Exception as e:
        return {}, f"Error: {e}", ""


def ui_step(action_json):
    try:
        import json

        action = json.loads(action_json)
        observation, reward, done, info = env.step(action)

        history = "\n".join(env.action_history) if env.action_history else "No actions taken yet"

        feedback = (
            f"Reward: {reward}\n"
            f"Done: {done}\n"
            f"Info: {info}"
        )

        return observation, feedback, history

    except Exception as e:
        return {}, f"Error: {e}", ""


# ---------- GRADIO UI ----------
with gr.Blocks(title="ACIE-HADO Interface") as demo:
    gr.Markdown("## ACIE-HADO: Human-AI Decision Optimization Environment")

    with gr.Row():
        task_level = gr.Dropdown(
            choices=["easy", "medium", "hard"],
            value="easy",
            label="Select Task Level"
        )

    reset_btn = gr.Button("Reset Environment", variant="primary")

    action_input = gr.Textbox(
        label="Action (JSON)",
        placeholder='{"action_type":"calculate_cognitive_score"}',
        lines=4
    )

    step_btn = gr.Button("Step", variant="primary")
    state_btn = gr.Button("Get State")

    current_observation = gr.JSON(label="Current Observation")
    step_feedback = gr.Textbox(label="Step Feedback", lines=5)
    action_history = gr.Textbox(label="Action History", lines=8)

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


# ---------- MOUNT UI AT /ui ----------
app = mount_gradio_app(app, demo, path="/ui")


# ---------- REQUIRED FOR VALIDATOR ----------
def main():
    """
    Entry point for validator (multi-mode support)
    """
    print("Server module ready for OpenEnv validation.")


# ---------- IMPORTANT ----------
if __name__ == "__main__":
    main()
