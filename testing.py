
import tensorflow.keras as keras
import rocket_gym 


# ─── INITIALIZE AND RUN ENVIRONMENT ─────────────────────────────────────────────
env_config = {
    'gui': True,
    # 'env_name': 'default',
    # 'env_name': 'empty',
    'env_name': 'level1',
    # 'env_name': 'level2',
    # 'env_name': 'random',
    # 'camera_mode': 'centered',
    # 'env_flipped': False,
    # 'env_flipmode': False,
    # 'export_frames': True,
    'export_states': False,
    # 'export_highscore': False,
    # 'export_string': 'human',
    'max_steps': 1000,
    'gui_reward_total': True,
    'gui_echo_distances': True,
    'gui_level': True,
    'gui_velocity': True,
    'gui_goal_ang': True,
    'gui_frames_remaining': True,
    'gui_draw_echo_points': True,
    'gui_draw_echo_vectors': True,
    'gui_draw_goal_points': True,
}

agent = keras.rl.core.Agent(processor=None)

agent.fit(rocket_gym.RocketMeister10(env_config), 10, action_repetition=1, callbacks=None, verbose=1, visualize=False, nb_max_start_steps=0, start_step_policy=None, log_interval=10000, nb_max_episode_steps=None)