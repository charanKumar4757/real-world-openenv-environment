import os
import json
import gradio as gr
from app.env import CognitiveEnv

# Global environment instance
env = None
current_task_level = "easy"

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

def reset_environment(task_dropdown):
    """Reset the environment with selected task level."""
    global env, current_task_level
    current_task_level = task_dropdown
    
    if env is None:
        env = CognitiveEnv()
    
    obs = env.reset(task_level=current_task_level)
    
    formatted_obs = format_observation(obs)
    return formatted_obs, "Environment reset for " + current_task_level + " task", "No actions taken yet"

def get_state():
    """Get current environment state."""
    global env
    if env is None:
        return "Environment not initialized", "Please reset first", "No actions taken yet"
    
    obs = env.state()
    formatted_obs = format_observation(obs)
    return formatted_obs, json.dumps(obs, indent=2), "\n".join(env.action_history) if env.action_history else "No actions taken yet"

def step_environment(action_text):
    """Step the environment with given action."""
    global env
    if env is None:
        return "Environment not initialized", "Please reset first", "Error"
    
    try:
        action_dict = json.loads(action_text)
    except json.JSONDecodeError:
        return format_observation(env.state()), f"Invalid JSON action: {action_text}", "\n".join(env.action_history)
    
    try:
        obs, reward, done, info = env.step(action_dict)
        formatted_obs = format_observation(obs)
        feedback = f"Action: {action_dict.get('action_type', 'unknown')}\nReward: {reward:.2f}\nDone: {done}"
        history = "\n".join(env.action_history) if env.action_history else "No actions taken yet"
        return formatted_obs, feedback, history
    except Exception as e:
        return format_observation(env.state()), f"Error: {str(e)}", "\n".join(env.action_history)

with gr.Blocks(title="ACIE-HADO Interface") as demo:
    gr.Markdown("## ACIE-HADO: Human-AI Decision Optimization Environment")

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
    with gr.Row():
        get_state_button = gr.Button("Get State")
        # Space for alignment

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
        fn=reset_environment,
        inputs=[task_dropdown],
        outputs=[obs_box, feedback_box, action_history_box]
    )

    step_button.click(
        fn=step_environment,
        inputs=[action_input],
        outputs=[obs_box, feedback_box, action_history_box]
    )

    get_state_button.click(
        fn=get_state,
        inputs=[],
        outputs=[obs_box, feedback_box, action_history_box]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)