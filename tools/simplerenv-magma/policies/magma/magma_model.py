import numpy as np
from PIL import Image
import random
import torch
import torchvision
import json
import sys
import os
from transformers import AutoModelForVision2Seq, AutoProcessor
from magma.image_processing_magma import MagmaImageProcessor
from magma.processing_magma import MagmaProcessor
from magma.modeling_magma import MagmaForCausalLM
from transforms3d.euler import euler2axangle

class MagmaInference:
    def __init__(self, model_name, policy_setup, action_scale=1.0, sticky_gripper_num_repeat=10, unnorm_key=None, sample=False):
        if policy_setup == "widowx_bridge":
            self.unnorm_key = "bridge_orig" if unnorm_key is None else unnorm_key
        elif policy_setup == "google_robot":
            self.unnorm_key = "fractal20220817_data" if unnorm_key is None else unnorm_key
        self.sticky_gripper_num_repeat = sticky_gripper_num_repeat

        self.real_vla = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to("cuda")

        self.processor = MagmaProcessor.from_pretrained(model_name, trust_remote_code=True) 
        self.vla = MagmaForCausalLM.from_pretrained(
            model_name,
	        device_map="cuda", 
	        low_cpu_mem_usage=True,        
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        self.task_description = None

        self.policy_setup = policy_setup
        self.action_scale = action_scale
        self.sample = sample

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.action_norm_stats = self.real_vla.get_action_stats(self.unnorm_key)
        self.n_action_bins = 256
        self.vocab_size = self.processor.tokenizer.vocab_size
        self.bins = np.linspace(-1, 1, self.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

    def reset(self, task_description):
        self.task_description = task_description

    def step(self, image: np.ndarray, task_description: str | None = None):
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        convs = [
            {"role": "user", "content": f"<image>\nWhat action should the robot take to {self.task_description}?"},
        ]
        convs = [
            {
                "role": "system",
                "content": "You are agent that can see, talk and act.", 
            },            
        ] + convs            
        prompt = self.processor.tokenizer.apply_chat_template(
            convs,
            tokenize=False,
            add_generation_prompt=True
        )
        if self.vla.config.mm_use_image_start_end:
            prompt = prompt.replace("<image>", "<image_start><image><image_end>")
            
        image = Image.fromarray(image)

        # resize image to 256x256
        # image = image.resize((512, 512))
        image = image.resize((256, 256))
        inputs = self.processor(images=image, texts=prompt, return_tensors="pt")
        inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
        inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
        inputs = inputs.to("cuda").to(torch.bfloat16)

        self.vla.generation_config.pad_token_id = self.processor.tokenizer.pad_token_id
        with torch.inference_mode():
            output_ids = self.vla.generate(
                **inputs, 
                temperature=0.7, 
                do_sample=self.sample, 
                num_beams=1, 
                max_new_tokens=1000, 
                use_cache=True,
            )
            action_ids = output_ids[0, -8:-1].cpu().tolist()

            if random.random() < 0.1:
                print("Action ids", action_ids)

        predicted_action_ids = np.array(action_ids).astype(np.int64)
        discretized_actions = self.vocab_size - predicted_action_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
        normalized_actions = self.bin_centers[discretized_actions]

        # Unnormalize actions
        mask = self.action_norm_stats.get("mask", np.ones_like(self.action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(self.action_norm_stats["q99"]), np.array(self.action_norm_stats["q01"])
        raw_actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        raw_action = {
            "world_vector": np.array(raw_actions[:3]),
            "rotation_delta": np.array(raw_actions[3:6]),
            "open_gripper": np.array(raw_actions[6:7]),  # range [0, 1]; 1 = open; 0 = close
        }
        # print(raw_action)
        # Process raw_action to obtain the action for the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = action_rotation_ax * action_rotation_angle
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            current_gripper_action = raw_action["open_gripper"]

            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and not self.sticky_action_is_on:
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])

        return raw_action, action