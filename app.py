import os
import uuid
import gradio as gr

# -----------------------------
# Simple environment state
# Replace later with your real env logic from env.py
# -----------------------------
env_state = {
    "status": "Reset",
    "episode_id": "-",
    "step_count": 0,
    "observation": "No observation yet",
    "history": []
}

def reset_environment():
    env_state["status"] = "Reset"
    env_state["episode_id"] = str(uuid.uuid4())[:8]
    env_state["step_count"] = 0
    env_state["observation"] = "Environment reset successfully."
    env_state["history"] = []

    current_state = (
        f"Status: {env_state['status']}\n"
        f"Episode ID: {env_state['episode_id']}\n"
        f"Step Count: {env_state['step_count']}"
    )

    return current_state, env_state["observation"], "No actions taken yet"

def get_state():
    current_state = (
        f"Status: {env_state['status']}\n"
        f"Episode ID: {env_state['episode_id']}\n"
        f"Step Count: {env_state['step_count']}"
    )

    history_text = "\n".join(env_state["history"]) if env_state["history"] else "No actions taken yet"
    return current_state, env_state["observation"], history_text

def step_environment(message):
    if env_state["episode_id"] == "-":
        env_state["episode_id"] = str(uuid.uuid4())[:8]

    env_state["status"] = "Running"
    env_state["step_count"] += 1

    clean_message = (message or "").strip()
    if not clean_message:
        clean_message = "No input provided"

    env_state["observation"] = f"Processed action: {clean_message}"
    env_state["history"].append(f"Step {env_state['step_count']}: {clean_message}")

    current_state = (
        f"Status: {env_state['status']}\n"
        f"Episode ID: {env_state['episode_id']}\n"
        f"Step Count: {env_state['step_count']}"
    )

    history_text = "\n".join(env_state["history"]) if env_state["history"] else "No actions taken yet"
    return current_state, env_state["observation"], history_text

with gr.Blocks(title="HumanAgent Interface") as demo:
    gr.Markdown("## HumanAgent Interface")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Take Action")
            message_input = gr.Textbox(
                label="Message *",
                placeholder="Enter message...",
                lines=3
            )
            step_button = gr.Button("Step", variant="primary")

            with gr.Row():
                reset_button = gr.Button("Reset Environment")
                get_state_button = gr.Button("Get State")

            current_state_box = gr.Textbox(
                label="Current State",
                lines=6,
                interactive=False,
                value="Status: Reset\nEpisode ID: -\nStep Count: 0"
            )

        with gr.Column(scale=2):
            current_observation_box = gr.Textbox(
                label="Current Observation",
                lines=6,
                interactive=False,
                value="No observation yet"
            )

            action_history_box = gr.Textbox(
                label="Action History",
                lines=12,
                interactive=False,
                value="No actions taken yet"
            )

    step_button.click(
        fn=step_environment,
        inputs=[message_input],
        outputs=[current_state_box, current_observation_box, action_history_box]
    )

    reset_button.click(
        fn=reset_environment,
        inputs=[],
        outputs=[current_state_box, current_observation_box, action_history_box]
    )

    get_state_button.click(
        fn=get_state,
        inputs=[],
        outputs=[current_state_box, current_observation_box, action_history_box]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)