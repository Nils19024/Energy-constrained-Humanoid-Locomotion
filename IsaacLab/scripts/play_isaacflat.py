#!/usr/bin/env python3
"""
Final, working launcher script that saves the recorded video inside
the agent's log directory for better organization.
"""

import os
import sys
import argparse
import torch
import json
import gymnasium

from isaaclab.app import AppLauncher

ISAACLAB_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ISAACLAB_DIR, "..", ".."))
OMNISAFE_REPO_DIR = os.path.join(PROJECT_ROOT, "omnisafe")

def main():
    parser = argparse.ArgumentParser(description="Play a trained OmniSafe agent in Isaac Lab.")
    parser.add_argument("--log_dir", required=True, type=str, help="Path to the run directory (e.g., .../seed-0-...).")
    parser.add_argument("--model_name", type=str, default="epoch-0.pt", help="The name of the saved model file.")
    parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video in steps.")
    parser.add_argument("--video_name", type=str, default="agent_video", help="Base name of the output video file (without extension).")
    
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    
    args.headless = True
    args.enable_cameras = True
    
    device = "cuda:0"

    print("[INFO] Launching Isaac Sim in HEADLESS mode with cameras ENABLED...")
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    if OMNISAFE_REPO_DIR not in sys.path:
        sys.path.insert(0, OMNISAFE_REPO_DIR)

    print("[INFO] Isaac Sim is running. Importing libraries...")
    from omnisafe.envs.play_isaac_g1_env import PlayIsaacFlatEnv
    from omnisafe.models import ActorCritic
    from omnisafe.common.normalizer import Normalizer
    from omnisafe.utils.config import ModelConfig

    print("[INFO] Creating the environment with 'rgb_array' render mode...")
    env = PlayIsaacFlatEnv(env_id="Isaac-Velocity-Flat-G1-v0-Play", num_envs=16, device=device, render_mode="rgb_array")

    # --- GEÄNDERTE ZEILE: Video-Ordner anpassen ---
    # Das Video wird jetzt in einem "videos"-Unterordner innerhalb des Log-Verzeichnisses des Agenten gespeichert.
    video_folder = os.path.join(args.log_dir, "videos")
    
    print(f"[INFO] Wrapping environment for video recording. Videos will be saved in: {video_folder}")
    video_kwargs = {
        "video_folder": video_folder,
        "name_prefix": args.video_name,
        "step_trigger": lambda step: step == 0,
        "video_length": args.video_length,
        "disable_logger": True,
    }
    env = gymnasium.wrappers.RecordVideo(env, **video_kwargs)
    
    config_path = os.path.join(args.log_dir, "config.json")
    model_path = os.path.join(args.log_dir, "torch_save", args.model_name)

    print(f"[INFO] Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    model_config_obj = ModelConfig(**config["model_cfgs"])
    epochs = config["train_cfgs"]["epochs"]

    print("[INFO] Re-creating the ActorCritic model structure...")
    actor_critic = ActorCritic(
        obs_space=env.observation_space,
        act_space=env.action_space,
        model_cfgs=model_config_obj,
        epochs=epochs,
    ).to(device)

    print("[INFO] Re-creating the observation normalizer...")
    obs_normalizer = Normalizer(shape=env.observation_space.shape).to(device)

    print(f"[INFO] Loading model and normalizer state from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    actor_critic.load_state_dict(checkpoint['pi'], strict=False)
    obs_normalizer.load_state_dict(checkpoint['obs_normalizer'])
    
    print("[INFO] Agent components loaded and ready!")
    print(f"[INFO] Running inference for {args.video_length} steps to record one video...")
    
    obs, info = env.reset()
    
    for i in range(args.video_length):
        normalized_obs = obs_normalizer.normalize(obs.to(device))
        with torch.no_grad():
            action, _, _ = actor_critic.step(normalized_obs, deterministic=False)
        print("Action:", action)
        obs, reward, terminated, truncated, info = env.step(action)
        cost = info.get('cost', 0.0)
        
        if (i + 1) % 100 == 0:
            print(f"  Step {i + 1}/{args.video_length}")

    print("[INFO] Inference complete.")
    
    env.close()
    
    print(f"[INFO] Video saved successfully in folder: {video_folder}")
    print("[INFO] Closing simulation.")
    simulation_app.close()

if __name__ == "__main__":
    main()