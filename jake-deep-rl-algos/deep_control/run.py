import gym
import numpy as np
import torch

from . import envs, utils


def run_env(agent, env, episodes, max_steps, render=False, verbosity=1, discount=1.0):
    episode_return_history = []
    if render:
        env.render("rgb_array")
    for episode in range(episodes):
        episode_return = 0.0
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result
        done, info = False, {}
        for step_num in range(max_steps):
            if done:
                break
            action = agent.forward(state)
            step_result = env.step(action)
            # Handle both old gym (4 returns) and new gym (5 returns)
            if len(step_result) == 5:
                state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                state, reward, done, info = step_result
            if render:
                env.render("rgb_array")
            episode_return += reward * (discount ** step_num)
        if verbosity:
            print(f"Episode {episode}:: {episode_return}")
        episode_return_history.append(episode_return)
    return torch.tensor(episode_return_history)


def exploration_noise(action, random_process):
    return np.clip(action + random_process.sample(), -1.0, 1.0)


def evaluate_agent(
    agent, env, eval_episodes, max_episode_steps, render=False, verbosity=0
):
    agent.eval()
    returns = run_env(
        agent, env, eval_episodes, max_episode_steps, render, verbosity=verbosity
    )
    agent.train()
    mean_return = returns.mean()
    return mean_return


def collect_experience_by_steps(
    agent,
    env,
    buffer,
    num_steps,
    current_state=None,
    current_done=None,
    steps_this_ep=None,
    max_rollout_length=None,
    random_process=None,
):
    if current_state is None:
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result
    else:
        state = current_state
    if current_done is None:
        done = False
    else:
        done = current_done
    if steps_this_ep is None:
        steps_this_ep = 0
    for step in range(num_steps):
        if done:
            result = env.reset()
            state = result[0] if isinstance(result, tuple) else result
            steps_this_ep = 0

        # collect a new transition
        action = agent.collection_forward(state)
        if random_process is not None:
            action = exploration_noise(action, random_process, env.action_space.high[0])
        step_result = env.step(action)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, info = step_result
        buffer.push(state, action, reward, next_state, done)
        state = next_state

        steps_this_ep += 1
        if max_rollout_length and steps_this_ep >= max_rollout_length:
            done = True
    return state, done, steps_this_ep


def collect_experience_by_rollouts(
    agent,
    env,
    buffer,
    num_rollouts,
    max_rollout_length,
    random_process=None,
):
    for rollout in range(num_rollouts):
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result
        done = False
        step_num = 0
        while not done:
            action = agent.collection_forward(state)
            if random_process is not None:
                action = exploration_noise(action, random_process, env.action_space.high[0])
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, info = step_result
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            step_num += 1
            if step_num >= max_rollout_length:
                done = True


def warmup_buffer(buffer, env, warmup_steps, max_episode_steps):
    # use warmp up steps to add random transitions to the buffer
    result = env.reset()
    state = result[0] if isinstance(result, tuple) else result
    done = False
    for _ in range(warmup_steps):
        rand_action = env.action_space.sample()
        step_result = env.step(rand_action)
        # Handle both old gym (4 returns) and new gym (5 returns)
        if len(step_result) == 5:
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            next_state, reward, done, info = step_result
        buffer.push(state, rand_action, reward, next_state, done)
        state = next_state
        if done:
            result = env.reset()
            state = result[0] if isinstance(result, tuple) else result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", type=int, default=1)
    parser.add_argument("--env", type=str)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--save", type=str)
    parser.add_argument("--algo", type=str)
    parser.add_argument("--max_steps", type=int, default=300)
    args = parser.parse_args()

    agent, env = envs.load_env(args.env, args.algo)
    agent.load(args.agent)
    run_env(agent, env, args.episodes, args.max_steps, args.render, verbosity=1)
