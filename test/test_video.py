from video import Video
from IPython import display as ipythondisplay
from IPython.display import HTML
from base64 import b64encode
import glob
import os.path
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

def test_video():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    Video.generate_video(env, 1)
    mp4list = glob.glob('video/*.mp4')
    mp4 = mp4list[0]
    assert(os.path.exists(glob.glob('video/*.mp4')[0]))
