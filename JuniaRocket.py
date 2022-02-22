import numpy as np
import pygame
#import gym
from rocket_gym import RocketMeister10

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

env = RocketMeister10(env_config)
observation = env.reset()
env.render()

for _ in range(1000):
  #env.clock.tick(10)
  get_event = pygame.event.get()  
  action = env.action_space.sample() # your agent here (this takes random actions)  
  observation, reward, done, info = env.step(action)
  if done:
    observation = env.reset()
  print (observation)
  env.render()

env.close()
pygame.quit()