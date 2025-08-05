# FILE: /work/dlclarge2/schmidtn-schmidt-IsaacSim/New/IsaacLab/scripts/train_isaacflat.py

#!/usr/bin/env python3
"""
Final, definitive launcher script to train an OmniSafe agent within Isaac Lab.
This script respects the Isaac Lab application lifecycle by initializing the
simulation app BEFORE importing any packages that depend on it.
"""

import os
import sys
import argparse

# Import from isaaclab *before* modifying the path.
from isaaclab.app import AppLauncher

# Define paths at the top level, but do not use them until inside main().
ISAACLAB_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ISAACLAB_DIR, "..", ".."))
OMNISAFE_REPO_DIR = os.path.join(PROJECT_ROOT, "omnisafe")

def main():
    """Main function to launch Isaac Sim and start training."""
    parser = argparse.ArgumentParser(description="Train an OmniSafe agent with Isaac Lab.")
    parser.add_argument("--algo", required=True, help="The OmniSafe algorithm to use for training.")
    parser.add_argument("--env-id", required=True, help="The environment ID to train on.")
    args, _ = parser.parse_known_args()

    # Launch Isaac Sim
    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    if OMNISAFE_REPO_DIR not in sys.path:
        sys.path.insert(0, OMNISAFE_REPO_DIR)

    print("[INFO] Isaac Sim is running. Importing OmniSafe...")
    import omnisafe
    from omnisafe.envs.train_isaac_g1_env import TrainIsaacFlatEnv 


    num_envs = 16  
    steps_per_env_per_epoch = 512
    steps_per_epoch = num_envs * steps_per_env_per_epoch
    total_steps = steps_per_epoch * 1000

    train_terminal_cfgs = {}
    
    # In main() von train_isaacflat.py
    custom_cfgs = {
        "train_cfgs": {
            "vector_env_nums": num_envs,
            "device": "cuda:0",
            "total_steps": total_steps,
            "torch_threads": 16,
        },
        "algo_cfgs": {
            "steps_per_epoch": steps_per_epoch,
            "reward_normalize": False,
            "cost_normalize": True,
            "obs_normalize": True,
            # VORSCHLAG 3: Erhöhe die Policy-Entropie, um die Exploration zu fördern
            "entropy_coef": 0.01,
            # VORSCHLAG 4: Reduziere die Ziel-KL-Divergenz für stabilere Policy-Updates
            "target_kl": 0.02,
            "update_iters": 10,
            # VORSCHLAG 4: Erhöhe die Batch-Größe für stabilere Gradienten
            "batch_size": 512,
            "use_critic_norm": False,
            # VORSCHLAG 4: Füge die Gradientennorm-Begrenzung hinzu und setze sie auf einen gängigen Wert
            "use_max_grad_norm": True,
            "max_grad_norm": 1.0,
        },
        "model_cfgs": {
            "actor_type": "gaussian_learning",
            # VORSCHLAG 3: Verwende ein fixes, höheres Rauschen für konsistentere Exploration
            "std_range": [0.1, 0.1],
            "actor": {
                "hidden_sizes": [256, 128, 128],
                "activation": "relu",
                # VORSCHLAG 4: Passe die Lernraten an gängige Werte an
                "lr": 0.0003, # 3e-4 ist ein gängiger Startwert
            },
            "critic": {
                "hidden_sizes": [256, 128, 128],
                "activation": "relu",
                # VORSCHLAG 4: Gleiche die Lernrate der des Actors an
                "lr": 0.0003,
            }
        },
        "logger_cfgs": {
            "log_dir": os.path.join(os.getcwd(), "train_logs"),
            "save_model_freq": 25
        },
    }

    
    print(f"[INFO] Starting OmniSafe training for algo '{args.algo}' on env '{args.env_id}'...")
    print(f"[INFO] Manual Configs: {custom_cfgs}")

    # Call the agent directly, which is what `omnisafe_app` would do internally.
    agent = omnisafe.Agent(
        algo=args.algo,
        env_id=args.env_id,
        train_terminal_cfgs=train_terminal_cfgs,
        custom_cfgs=custom_cfgs
    )
    
    # Start the learning process
    agent.learn()

    print(f"[INFO] OmniSafe training finished.")
    simulation_app.close()
    print("[INFO] Training finished and simulation closed.")

if __name__ == "__main__":
    main()