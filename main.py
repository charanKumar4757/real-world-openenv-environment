import os
import json
from typing import Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from gradio.routes import mount_gradio_app

from app.env import CognitiveEnv

# ============================================
# FastAPI App Setup
# ============================================
app = FastAPI(title="ACIE-HADO Environment")

# Global environment instance
env = CognitiveEnv()

# ============================================
# Request/Response Models
# ============================================
class ResetRequest(BaseModel):
    task_level: str = "easy"

class StepRequest(BaseModel):
    action_type: str
    target_task_id: Optional[str] = None
    target_user: Optional[str] = None

# ============================================
# API Endpoints
# ============================================
@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "ACIE-HADO environment is running"}

@app.post("/reset")
def reset_env(task_level: str = "easy"):
    """Reset environment and return initial observation."""
    try:
        obs = env.reset(task_level=task_level)
        return {
            "status": "ok",
            "observation": obs,
            "task_level": task_level
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/state")
def get_state():
    """Get current environment state."""
    try:
        state = env.state()
        return {
            "status": "ok",
            "state": state,
            "step_count": env.step_count,
            "total_reward": env.total_reward
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/step")
def step_env(
    action_type: str,
    target_task_id: Optional[str] = None,
    target_user: Optional[str] = None
):
    """Take a step in the environment with the given action."""
    try:
        action_dict = {
            "action_type": action_type,
            "target_task_id": target_task_id,
            "target_user": target_user
        }
        obs, reward, done, info = env.step(action_dict)
        return {
            "status": "ok",
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info,
            "step_count": env.step_count,
            "total_reward": env.total_reward
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ============================================
# Gradio UI (mounted at /ui)
# ============================================
def format_observation(obs_dict):
    """Format observation dictionary for display."""
    lines = []
    lines.append(f"Cognitive Score: {obs_dict.get('cognitive_score', 0)}/100")
    lines.append(f"Fatigue Level: {obs_dict.get('fatigue_level', 'unknown').capitalize()}")
    lines.append(f"Emotional State: {obs_dict.get('emotional_state', 'unknown').capitalize()}")
    lines.append(f"Recovery Prediction: {obs_dict.get('recovery_prediction', 0)} min")
    lines.append(f"Decision Debt: {obs_dict.get('decision_debt', 0)} items")
    lines.append(f"Autopilot Active: {obs_dict.get('autopilot_active', False)}")
    lines.append(f"Pending Tasks: {len(obs_dict.get('pending_tasks', []))}")
    return "\n".join(lines)

def ui_reset_environment(task_dropdown):
    """UI handler for reset."""
    try:
        obs = env.reset(task_level=task_dropdown)
        formatted_obs = format_observation(obs)
        return formatted_obs, "Environment reset for " + task_dropdown + " task", "No actions taken yet"
    except Exception as e:
        return "", f"Error: {str(e)}", ""

def ui_get_state():
    """UI handler for get state."""
    try:
        obs = env.state()
        formatted_obs = format_observation(obs)
        return formatted_obs, json.dumps(obs, indent=2), "\n".join(env.action_history) if env.action_history else "No actions taken yet"
    except Exception as e:
        return "", f"Error: {str(e)}", ""

def ui_step_environment(action_text):
    """UI handler for step."""
    try:
        action_dict = json.loads(action_text)
    except json.JSONDecodeError:
        try:
            obs = env.state()
            return format_observation(obs), f"Invalid JSON action: {action_text}", "\n".join(env.action_history) if env.action_history else ""
        except:
            return "", f"Invalid JSON action: {action_text}", ""
    
    try:
        obs, reward, done, info = env.step(action_dict)
        formatted_obs = format_observation(obs)
        feedback = f"Action: {action_dict.get('action_type', 'unknown')}\nReward: {reward:.2f}\nDone: {done}"
        history = "\n".join(env.action_history) if env.action_history else "No actions taken yet"
        return formatted_obs, feedback, history
    except Exception as e:
        try:
            obs = env.state()
            return format_observation(obs), f"Error: {str(e)}", "\n".join(env.action_history) if env.action_history else ""
        except:
            return "", f"Error: {str(e)}", ""

with gr.Blocks(title="ACIE-HADO Interface") as demo:
    gr.Markdown("## ACIE-HADO: Human-AI Decision Optimization Environment")
    gr.Markdown("**Adaptive Cognitive Intelligence Environment for Human-AI Decision Optimization**")

    gr.Markdown("### Environment Settings")
    task_dropdown = gr.Dropdown(
        choices=["easy", "medium", "hard"],
        value="easy",
        label="Select Task Level"
    )
    reset_button = gr.Button("Reset Environment", variant="primary")

    gr.Markdown("### Take Action")
    action_input = gr.Textbox(
        label="Action (JSON) *",
        placeholder='{"action_type": "calculate_cognitive_score"}',
        lines=3
    )
    step_button = gr.Button("Step", variant="primary")

    gr.Markdown("### State Management")
    get_state_button = gr.Button("Get State")

    gr.Markdown("### Environment Status")
    obs_box = gr.Textbox(
        label="Current Observation",
        lines=8,
        interactive=False
    )

    feedback_box = gr.Textbox(
        label="Step Feedback",
        lines=4,
        interactive=False
    )

    action_history_box = gr.Textbox(
        label="Action History",
        lines=6,
        interactive=False
    )

    # Button handlers
    reset_button.click(
        fn=ui_reset_environment,
        inputs=[task_dropdown],
        outputs=[obs_box, feedback_box, action_history_box]
    )

    step_button.click(
        fn=ui_step_environment,
        inputs=[action_input],
        outputs=[obs_box, feedback_box, action_history_box]
    )

    get_state_button.click(
        fn=ui_get_state,
        inputs=[],
        outputs=[obs_box, feedback_box, action_history_box]
    )

# Mount Gradio UI at /ui
app = mount_gradio_app(app, demo, path="/ui")

# ============================================
# Run with uvicorn
# ============================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)