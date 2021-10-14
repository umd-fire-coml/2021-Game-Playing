from src.video import Video
import os.path
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

def test_video():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    Video.generate_video(env, 1, 'video.mp4')
    assert(os.path.exists('video.mp4'))
