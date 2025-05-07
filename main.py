#!/usr/bin/env python
# main.py ────────────────────────────────────────────────────────────────────
# Solves BipedalWalker-v3 with PPO and exports the actor as ONNX.
# Author: Mark Sere – May 2025
# ---------------------------------------------------------------------------
import os, math, random, time, argparse, shutil, glob, io, base64, warnings
from collections import deque, namedtuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import onnx

# ────── helper: tiny tqdm fallback ─────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:                        # keep the file self-contained
    def tqdm(x, **k): return x

# ────── hyper-parameters (can be overridden from CLI) ──────────────────────
CFG = dict(
    seed=31,
    env_id="BipedalWalker-v3",
    hardcore=False,            # set True for the hardcore variant
    total_episodes=1200,       # ~15 min on a T4 / M1
    max_steps=1600,            # env max is 1600 for BipedalWalker-v3
    update_every=2048,         # min transitions before a PPO update
    epochs=4,
    batch_size=128,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    lr=1e-4,
    betas=(0.9, 0.999),
    entropy_coef=1e-3,
    value_coef=0.5,
    max_grad_norm=0.5,
    hidden_size=128,
    logdir="runs",
    video=False,               # set True to record rollout.mp4
)

# ────── reproducibility ────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ────── Actor / Critic networks ────────────────────────────────────────────
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
        )
        self.mu_head     = nn.Linear(hidden, act_dim)
        self.logstd_head = nn.Linear(hidden, act_dim)

    def forward(self, x):
        h = self.net(x)
        mu      = self.mu_head(h)
        log_std = torch.clamp(self.logstd_head(h), -5, 2)   # keep numerics sane
        std     = log_std.exp()
        return mu, std


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.actor  = Actor(obs_dim, act_dim, hidden)
        self.critic = Critic(obs_dim, hidden)

    # step for interaction ---------------------------------------------------
    @torch.no_grad()
    def act(self, obs):
        mu, std = self.actor(obs)
        dist    = Normal(mu, std)
        action  = dist.sample()
        logp    = dist.log_prob(action).sum(-1)
        return action.cpu().numpy(), logp, dist

    # evaluate a batch -------------------------------------------------------
    def evaluate(self, obs, act):
        mu, std = self.actor(obs)
        dist    = Normal(mu, std)
        logp    = dist.log_prob(act).sum(-1)
        entropy = dist.entropy().sum(-1)
        value   = self.critic(obs).squeeze(-1)
        return logp, entropy, value

# ────── Rollout storage (GAE) ──────────────────────────────────────────────
Transition = namedtuple("Transition",
                        "obs act logp reward done value")

class RolloutBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma, gae_lambda):
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.max_size   = size
        self.gamma      = gamma
        self.lam        = gae_lambda
        self.reset()

    def reset(self):
        self.ptr = 0
        self.data = []

    def add(self, *args):
        self.data.append(Transition(*args))
        self.ptr += 1

    def __len__(self): return self.ptr

    # returns processed tensors ---------------------------------------------
    def compute_returns_adv(self):
        obs   = torch.tensor(np.stack([t.obs   for t in self.data]),
                             dtype=torch.float32)
        acts  = torch.tensor(np.stack([t.act   for t in self.data]),
                             dtype=torch.float32)
        logps = torch.tensor(np.stack([t.logp  for t in self.data]),
                             dtype=torch.float32)
        rews  = [t.reward for t in self.data]
        dones = [t.done   for t in self.data]
        vals  = torch.tensor(np.stack([t.value for t in self.data]),
                             dtype=torch.float32)

        # GAE lambda advantage ------------------------------------------------
        advs, gae = [], 0.0
        next_value = 0.0
        for step in reversed(range(len(rews))):
            delta = rews[step] + self.gamma * next_value * (1. - dones[step]) - vals[step].item()
            gae   = delta + self.gamma * self.lam * (1. - dones[step]) * gae
            advs.insert(0, gae)
            next_value = vals[step].item()
        advs   = torch.tensor(advs, dtype=torch.float32)
        returns = advs + vals
        # normalise advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        self.reset()
        return obs, acts, logps, returns, advs

# ────── PPO trainer ────────────────────────────────────────────────────────
class PPOAgent:
    def __init__(self, env, cfg):
        self.env   = env
        self.cfg   = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        self.ac = ActorCritic(obs_dim, act_dim, cfg.hidden_size).to(self.device)
        self.optim = optim.Adam(self.ac.parameters(),
                                lr=cfg.lr, betas=cfg.betas)

        buf_size = cfg.update_every
        self.buf = RolloutBuffer(obs_dim, act_dim, buf_size,
                                 cfg.gamma, cfg.gae_lambda)

        # logging
        self.ep_returns = deque(maxlen=100)

    # -----------------------------------------------------------------------
    def collect_rollout(self):
        obs, _ = self.env.reset()
        done, ep_ret = False, 0.0
        steps = 0

        while len(self.buf) < self.cfg.update_every:
            obs_t  = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            action, logp, _ = self.ac.act(obs_t)
            value = self.ac.critic(obs_t).item()

            next_obs, reward, done, _, _ = self.env.step(action.squeeze(0))
            self.buf.add(obs, action, logp.cpu().numpy(), reward, done, value)
            obs  = next_obs
            ep_ret += reward
            steps += 1

            if done or steps >= self.cfg.max_steps:
                self.ep_returns.append(ep_ret)
                obs, _  = self.env.reset()
                done, ep_ret, steps = False, 0.0, 0

    # -----------------------------------------------------------------------
    def update(self):
        obs, acts, logps_old, returns, advs = self.buf.compute_returns_adv()
        obs, acts, logps_old, returns, advs = \
            [t.to(self.device) for t in (obs, acts, logps_old, returns, advs)]

        for _ in range(self.cfg.epochs):
            idxs = torch.randperm(len(obs))
            for start in range(0, len(obs), self.cfg.batch_size):
                end   = start + self.cfg.batch_size
                mb    = idxs[start:end]

                logp, entropy, value = self.ac.evaluate(obs[mb], acts[mb])
                ratio = (logp - logps_old[mb]).exp()

                surr1 = ratio * advs[mb]
                surr2 = torch.clamp(ratio, 1. - self.cfg.clip_eps,
                                             1. + self.cfg.clip_eps) * advs[mb]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = ((returns[mb] - value) ** 2).mean()
                entropy_loss= -entropy.mean()

                loss = (policy_loss +
                        self.cfg.value_coef * value_loss +
                        self.cfg.entropy_coef * entropy_loss)

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(),
                                         self.cfg.max_grad_norm)
                self.optim.step()

    # -----------------------------------------------------------------------
    def train(self):
        for ep in tqdm(range(1, self.cfg.total_episodes + 1)):
            self.collect_rollout()
            self.update()

            if ep % 10 == 0 and self.ep_returns:
                avg = np.mean(self.ep_returns)
                print(f"[{ep:4d}/{self.cfg.total_episodes}] "
                      f"avg-100-return: {avg:7.1f}")
                # early stopping when comfortably above 300
                if avg > 300:
                    print("Solved environment 🎉")
                    break

    # -----------------------------------------------------------------------
    def save_actor_onnx(self, fname="submission_actor.onnx"):
        dummy = torch.randn(1, self.env.observation_space.shape[0]).to(self.device)
        torch.onnx.export(self.ac.actor, dummy, fname, opset_version=13)
        print(f"ONNX actor exported → {fname} ({os.path.abspath(fname)})")

    # -----------------------------------------------------------------------
    @torch.no_grad()
    def watch(self, episodes=3):
        env = gym.make(self.cfg.env_id, render_mode="human",
                       max_episode_steps=self.cfg.max_steps,
                       hardcore=self.cfg.hardcore)
        for ep in range(episodes):
            obs, _ = env.reset()
            done, ep_ret = False, 0.0
            while not done:
                obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                action, *_ = self.ac.act(obs_t)
                obs, reward, done, _, _ = env.step(action.squeeze(0))
                ep_ret += reward
            print(f"Episode {ep+1}: return {ep_ret:.1f}")
        env.close()

# ────── main / CLI ─────────────────────────────────────────────────────────
def parse_cfg():
    p = argparse.ArgumentParser()
    for k, v in CFG.items():
        arg = f"--{k.replace('_','-')}"
        if isinstance(v, bool):
            p.add_argument(arg, action="store_true" if not v else "store_false")
        else:
            p.add_argument(arg, type=type(v), default=v)
    return argparse.Namespace(**{**CFG, **vars(p.parse_args())})

def main():
    cfg = parse_cfg()
    set_seed(cfg.seed)

    env = gym.make(cfg.env_id,
                   hardcore=cfg.hardcore,
                   max_episode_steps=cfg.max_steps)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if cfg.video:
        env = gym.wrappers.RecordVideo(
            env, cfg.logdir, episode_trigger=lambda i: i==0)

    agent = PPOAgent(env, cfg)
    start = time.time()
    agent.train()
    dur = (time.time() - start) / 60
    print(f"Training finished in {dur:.1f} min")

    agent.save_actor_onnx()
    agent.watch()

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()