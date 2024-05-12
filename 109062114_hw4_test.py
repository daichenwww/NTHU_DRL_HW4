from osim.env import L2M2019Env
import time
from tqdm import tqdm
# import cv2
import torch
from torchvision import transforms
import gym
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

### SAC
class ActorNet(torch.nn.Module):

    def __init__(self):
        super(ActorNet, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(339, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 22 * 2),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = self.main(x)
        mu, log_std = x.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        # make sure the std is not too small
        std = torch.clamp(std, 1e-5, 1e-4)
        dist = torch.distributions.Normal(mu, std)
        transforms = [torch.distributions.transforms.TanhTransform()]
        dist = torch.distributions.transformed_distribution.TransformedDistribution(dist, transforms)
        action = dist.rsample()
        # action = mu
        log_prob = dist.log_prob(action).sum(-1)
        log_prob -= (2 * (np.log(2) - action - torch.nn.functional.softplus(-2 * action))).sum(-1) # refers to Open AI Spinning up
        return action, log_prob

class Agent(object):

    def __init__(self):
        self.device = device
        self.actor = ActorNet().to(self.device)
        self.load("109062114_hw4_data")

        self.skip = 4
        self.skip_count = 4
        self.last_action = [0 for _ in range(22)]

    def load(self, path):
        self.actor.load_state_dict(torch.load(path, map_location=self.device))

    def act(self, observation):
        if self.skip_count < self.skip:
            self.skip_count += 1
            return self.last_action
        self.skip_count = 0
        observation = self.observation_preprocessing(observation)
        self.last_action = self.choose_action(observation)
        return self.last_action
    
    def choose_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action = self.actor(observation)[0].cpu().numpy()
        # action range [-1, 1] to [0, 1]
        action = (action + 1) / 2
        # print(action)
        return action

    # def extract_features(self, V):
    #     # Extract features from the observation's v_tgt_field using a pre-trained ResNet18 model
    #     # 2x11x11 -> 3x11x11
    #     V = cv2.merge([V[0], V[1], np.zeros_like(V[0])])
    #     # 3x11x11 -> 3x224x224
    #     V = cv2.resize(V, (224, 224))
    #     V = transforms.ToTensor()(V).to(dtype=torch.float32)
    #     V = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(V)
    #     V = V.unsqueeze(0)
    #     V = self.resnet(V.to(self.device))
    #     return V.cpu().detach().numpy().flatten()

    def observation_preprocessing(self, observation):
        # put all the observation into a single vector
        V = np.array([])
        # recursively flatten the nested dictionary
        def flatten(d):
            for k, v in d.items():
                # if the key is v_tgt_field, extract features
                # if k == "v_tgt_field":
                    # yield self.extract_features(v)
                    # use 0 for now
                    # yield np.zeros(2*11*11)
                if isinstance(v, dict):
                    yield from flatten(v)
                elif isinstance(v, list):
                    for i in v:
                        yield i
                else:
                    yield v
        for i in flatten(observation):
            V = np.append(V, i)

        # V = torch.tensor(V, dtype=torch.float32) 
        # V = V.unsqueeze(0) 
        # print(V.shape)  # torch.Size([1097])
        return V

if __name__ == '__main__':

    env = L2M2019Env(visualize=True, difficulty=2)
    time_limit = 120
    max_timesteps = env.spec.timestep_limit
    agent = Agent()

    pbar = tqdm(range(3))
    for episode in pbar:
        obs = env.reset()
        episode_reward = 0
        timestep = 0
        start_time = time.time()

        while True:
            action = agent.act(obs)
            reward_ = 0
            for i in range(8):
                next_obs, reward, done, info = env.step(action)
                reward_ += reward
                if done:
                    break
            reward = reward_
            obs = next_obs
            episode_reward += reward
            timestep += 1
            if timestep >= max_timesteps:
                print(f"Max timestep reached for episode {episode}")
                break

            if time.time() - start_time > time_limit:
                print(f"Time limit reached for episode {episode}")
                break

            if done:
                break

        pbar.set_description(f"Episode {episode} reward: {episode_reward}")