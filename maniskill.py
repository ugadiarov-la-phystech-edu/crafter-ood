import gym
import gymnasium
import mani_skill.envs
import numpy as np


class ManiSkillEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, task, seed):
        self._task = task
        self._seed = seed
        self.render_mode = self.metadata["render.modes"][0]
        self._env = gymnasium.make(
            task,
            obs_mode="rgbd",
            control_mode="pd_joint_delta_pos",
            render_mode='rgb_array',
        )
        self.observation_space = gym.spaces.Box(0, 255, (128, 128, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, self._env.action_space.shape, dtype=np.float32)
        self._last_frame = None
        self._env.reset(seed=self._seed)

    @staticmethod
    def _unravel(step_result):
        unravel_result = [step_result[0]['sensor_data']['base_camera']['rgb'][0]]
        for x in step_result[1:-1]:
            unravel_result.append(x[0] if hasattr(x, '__len__') else x)

        info = {key: value[0] if hasattr(value, '__len__') else value for key, value in step_result[-1].items()}
        if 'success' in info:
            info['is_success'] = bool(info['success'])

        unravel_result.append(info)

        return unravel_result

    def render(self, mode=None):
        return self._last_frame

    def reset(self, seed=None, options=None):
        last_frame, _ = self._unravel(self._env.reset())
        self._last_frame = last_frame.numpy()
        return self._last_frame

    def step(self, action):
        last_frame, reward, terminated, truncated, info = self._unravel(self._env.step(action))
        self._last_frame = last_frame.numpy()
        return self._last_frame, float(reward), bool(terminated) or bool(truncated), info
