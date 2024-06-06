# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import gymnasium as gym
import torch
import numpy as np

from c_qnet import QNet, MODEL_DIR

total_reward = []

def test(env, q, num_episodes):
    global total_reward
    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation, _ = env.reset()

        episode_steps = 0

        done = False

        while not done:
            episode_steps += 1
            action = q.get_action(observation, epsilon=0.0)

            next_observation, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(
            i, episode_steps, episode_reward
        ))
        total_reward.append(episode_reward)


def main_play(num_episodes, env_name):
    global total_reward
    # 'model' 폴더 경로 설정
    folder_path = 'models'

    # 폴더 내 .pt 파일 리스트 가져오기
    pt_files = [file for file in os.listdir(folder_path) if file.endswith('.pth')]

    for model_name in pt_files:
        print(model_name)
        env = gym.make(env_name, render_mode="human")

        q = QNet(n_features=8, n_actions=4)
        model_params = torch.load(os.path.join(MODEL_DIR, model_name.format(env_name)))
        q.load_state_dict(model_params)

        test(env, q, num_episodes=num_episodes)

        env.close()

    # 평균 계산
    mean_reward = np.mean(total_reward)
    # 표준편차 계산
    std_reward = np.std(total_reward)

    print("Dueling DQN : [Test] Mean of Episode rewards:", mean_reward)
    print("Dueling DQN : [Test] Standard Dev. of Episode rewards:", std_reward)


if __name__ == "__main__":
    NUM_EPISODES = 3
    ENV_NAME = "LunarLander-v2"

    main_play(num_episodes=NUM_EPISODES, env_name=ENV_NAME)
