from ray import tune
from rocket_gym import RocketMeister10


tune.run(
    "SAC", # reinforced learning agent
    name = "Training1",
    # to resume training from a checkpoint, set the path accordingly:
    resume = False, # you can resume from checkpoint
    restore = r'checkpoint',
    checkpoint_freq = 100,
    checkpoint_at_end = True,
    local_dir = r'./ray_results/',
    config={
        "env": RocketMeister10,
        "num_workers": 30,
        "num_cpus_per_worker": 0.5,
        "env_config":{
            "max_steps": 1000,
            "export_frames": False,
            "export_states": True,
            "reward_mode": "continuous",
            # "env_flipped": True,
            # "env_flipmode": True,
        }
    },
    stop = {
        "timesteps_total": 5_000_000,
    },
)