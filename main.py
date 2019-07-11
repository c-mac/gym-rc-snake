import argparse
import gym
import time
import datetime
import torch

from agent.vpg_agent import VPGAgent
from agent.ppo_agent import PPOAgent
from agent.network import fc, lstm
from stats import Stats

from gym import logger
from wrappers.snake import SnakePerspectiveWithPrevActions
from gym.envs.registration import register
from gym.wrappers.monitoring.video_recorder import VideoRecorder

register(id="snake-rc-v0", entry_point="gym_rc_snake.envs:SnakeRCEnv")

AGENTS = {"ppo": PPOAgent, "vpg": VPGAgent}


def test(env, agent, num_episodes, stats, render=False):
    with torch.no_grad():
        episode_stats = []
        for t in range(num_episodes):
            episode_stats.append(one_episode(env, agent, render=render))
    stats.record([e[0] for e in episode_stats], [e[1] for e in episode_stats])


def one_episode(env, agent, pause=0.02, train=False, render=False):
    recorder = False
    if render:
        recorder = VideoRecorder(env, base_path=f"video/{datetime.datetime.utcnow()}")
    if recorder:
        print(recorder.ansi_mode)
    cum_reward = 0
    length = 0
    ob = env.reset()
    old_ob = ob
    while True:
        action = agent.act(ob)
        ob, reward, done, info = env.step(action)

        if train:
            agent.update_value(old_ob, action, reward, ob, done)

        old_ob = ob

        cum_reward += reward
        length += 1

        if render:
            env.render()
            recorder.capture_frame()
        # time.sleep(pause)

        if done:
            env.close()
            if recorder:
                recorder.close()
            return (cum_reward, length)


def train(env, agent, num_episodes):
    for t in range(num_episodes):
        one_episode(env, agent, train=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        "--env_id", default="snake-rc-v0", help="The environment to run"
    )
    parser.add_argument("--render", action="store_true", help="Render test episodes")
    parser.add_argument(
        "--test_episodes", default=100, type=int, help="How many episodes to test"
    )
    parser.add_argument(
        "--training_episodes",
        default=250,
        type=int,
        help="How many episodes to train before testing again",
    )
    parser.add_argument(
        "--epochs", default=1000, type=int, help="How many epochs to train"
    )
    parser.add_argument("--agent", default="ppo", help="Which agent to use")

    args = parser.parse_args()

    logger.set_level(logger.DEBUG)

    env = gym.make(args.env_id)
    env.reset()
    if args.env_id == "snake-rc-v0":
        env = SnakePerspectiveWithPrevActions(env)

    network_fn = fc(sum(env.observation_space.shape), env.action_space.n)

    agent = AGENTS[args.agent](
        action_space=env.action_space,
        network_fn=network_fn,
        network_name=f"savepoints/{args.agent}-{args.env_id}.pkl",
    )
    stats = Stats(args.env_id)

    for _ in range(args.epochs):
        test(env, agent, args.test_episodes, stats, args.render)
        train(env, agent, args.training_episodes)

    env.close()
