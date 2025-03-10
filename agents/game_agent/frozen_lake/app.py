import gradio as gr
import numpy as np
import gymnasium as gym
from PIL import Image
import matplotlib.pyplot as plt

# Initialize FrozenLake environment
env = gym.make("FrozenLake-v1", render_mode="rgb_array")
state, _ = env.reset()

action_mapping = {
    "Left": 3,
    "Down": 1,
    "Right": 2,
    "Up": 0,
}

def render_env():
    """Render the environment and return as an image."""
    frame = env.render()
    image = Image.fromarray(frame)
    return image

def step(action):
    """Take a step in the environment."""
    global state
    action_index = action_mapping[action]
    state, reward, done, _, _ = env.step(action_index)
    
    image = render_env()
    message = f"State: {state}, Reward: {reward}, Done: {done}"
    
    if done:
        env.reset()
        message += " - Resetting environment"
    
    return image, message

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Play Frozen Lake!")
    image_display = gr.Image()
    action_buttons = gr.Radio(choices=list(action_mapping.keys()), label="Select Action")
    submit_button = gr.Button("Step")
    output_text = gr.Textbox(label="Game State")
    
    submit_button.click(fn=step, inputs=action_buttons, outputs=[image_display, output_text])
    
    # Show initial state
    image_display.update(render_env())

demo.launch()
