# /work/dlclarge2/schmidtn-schmidt-IsaacSim/New/omnisafe/train_velocities.py

import omnisafe

custom_cfgs = {
    'train_cfgs': {
        "total_steps": 5000000,
        "vector_env_nums": 16,
        "parallel": 1,
        "torch_threads": 8,
        "device": "cuda:0",
    },
    'lagrange_cfgs': {
        'cost_limit': 50,
    },
    "model_cfgs": {
        "actor": {
            "hidden_sizes": [256, 128, 128],
            "lr": 0.00001, 
        },
        "critic": {
            "hidden_sizes": [256, 128, 128],
            "lr": 0.00001,
        },
        "exploration_noise_anneal": True,
    },
    "algo_cfgs": {
        "update_iters": 10,
        "target_kl": 0.02,
    },
    "logger_cfgs": {
        "save_model_freq": 10,
    }
}

print("Initialisiere den OmniSafe-Agenten...")
agent_v1 = omnisafe.Agent(
    algo="PPOLag",
    env_id="SafetyHumanoidVelocity-v0",
    custom_cfgs=custom_cfgs
)

log_dir = agent_v1.agent.logger.log_dir
print(f"Agent initialisiert. Log-Verzeichnis ist: {log_dir}")

# Den Befehl zum Initialisieren der Log-Datei senden
print("Sende Befehl zur Initialisierung der Log-Datei...")
env_adapter = agent_v1.agent._env
env_adapter.call('set_log_dir', log_dir)
print("Befehl gesendet.")

print("\nStarte das Training...")
agent_v1.learn()

print("Training beendet.")