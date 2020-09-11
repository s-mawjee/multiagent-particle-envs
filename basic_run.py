import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
import numpy as np
import gym
from multiagent.wrappers.monitor import Monitor
from multiagent.wrappers.time_limit import TimeLimit


class OneHotEncodeActionWrapper(gym.Wrapper):
    def __init__(self, env, movement_rate=0.5):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self._movement_rate = movement_rate

    def step(self, action):
        onehot_actions = []
        for i in range(len(action)):
            onehot_action = np.zeros(self.env.action_space[i].n)
            onehot_action[action[i]] = self._movement_rate
            onehot_actions.append(onehot_action)
        next_states, rewards, dones, info = self.env.step(onehot_actions)
        return next_states, rewards, dones, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def create_env(
    scenario_name="simple_spread",
    benchmark=False,
    number_of_agents=None,
    number_of_landmarks=None,
    communication_dim=None,
):
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(
        number_of_agents, number_of_landmarks, communication_dim
    )
    # create multi-agent environment
    if benchmark:
        env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            scenario.benchmark_data,
        )
    else:
        env = MultiAgentEnv(
            world, scenario.reset_world, scenario.reward, scenario.observation
        )
    return env


if __name__ == "__main__":

    # Create env
    env_name = "simple_spread"
    env_ = create_env(env_name)
    env_ = OneHotEncodeActionWrapper(env_, movement_rate=0.5)
    env_ = TimeLimit(env_, max_episode_steps=50)

    for eps in range(2):
        env = Monitor(
            env_,
            f"./videos/{env_name}/eps_{eps}",
            video_callable=lambda episode_id: episode_id == 0,
            force=True,
        )

        print(f"Starting eps:{eps}")
        print("*" * 50)
        dones = [False for _ in range(env.n)]
        time_step = 0

        obs = env.reset()
        while not all(dones):
            # get random action for each agent
            actions = [env.action_space[i].sample() for i in range(env.n)]
            next_obs, rewards, dones, info = env.step(actions)
            time_step += 1
            obs = next_obs
            print(time_step, rewards, dones)
        print("*" * 50)
    env.close()
