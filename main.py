import random, time
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import gymnasium as gym
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# ───── hyper‑parameters ────────────────────────────────────────────
SEED = 31
ENV_ID = "BipedalWalker-v3"
HARDCORE = False
TOTAL_EPISODES = 1200
MAX_STEPS = 1600
UPDATE_EVERY = 2048
EPOCHS = 4
BATCH_SIZE = 128
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 1e-4
BETAS = (0.9, 0.999)
ENTROPY_COEF = 1e-3
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5
HIDDEN = 128

# ───── reproducibility ────────────────────────────────────────────────────

def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ───── networks ───────────────────────────────────────────────────────────

class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh()
        )
        self.mu_head = nn.Linear(HIDDEN, act_dim)
        self.logsig_head = nn.Linear(HIDDEN, act_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu_head(h), self.logsig_head(h)


class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, 1)
        )

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        mu_logits, sig_logits = self.actor(obs)
        mu = torch.tanh(mu_logits)
        sigma = F.softplus(sig_logits) + 1e-5
        dist = Normal(mu, sigma)
        act = dist.sample()
        logp = dist.log_prob(act).sum(-1)
        return torch.clamp(act, -1, 1).cpu().numpy(), logp, dist

    def evaluate(self, obs: torch.Tensor, act: torch.Tensor):
        mu_logits, sig_logits = self.actor(obs)
        mu = torch.tanh(mu_logits)
        sigma = F.softplus(sig_logits) + 1e-5
        dist = Normal(mu, sigma)
        logp = dist.log_prob(act).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return logp, entropy, value


# ───── rollout buffer ─────────────────────────────────────────────────────

Transition = namedtuple("Transition", "obs act logp rew done val")


class RolloutBuffer:
    def __init__(self):
        self.gamma = GAMMA
        self.lam = GAE_LAMBDA
        self.reset()

    def reset(self):
        self.data = []

    def add(self, *args):
        self.data.append(Transition(*args))

    def __len__(self):
        return len(self.data)

    def compute(self):
        obs = torch.tensor(np.stack([t.obs for t in self.data]), dtype=torch.float32)
        acts = torch.tensor(np.stack([t.act for t in self.data]), dtype=torch.float32)
        logps = torch.tensor(np.stack([t.logp for t in self.data]), dtype=torch.float32)
        rews = [t.rew for t in self.data]
        dones = [t.done for t in self.data]
        vals = torch.tensor(np.stack([t.val for t in self.data]), dtype=torch.float32)

        advs, gae, next_val = [], 0.0, 0.0
        for i in reversed(range(len(rews))):
            delta = rews[i] + self.gamma * next_val * (1 - dones[i]) - vals[i].item()
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advs.insert(0, gae)
            next_val = vals[i].item()
        advs = torch.tensor(advs, dtype=torch.float32)
        returns = advs + vals
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        self.reset()
        return obs, acts, logps, returns, advs


# ───── PPO agent ──────────────────────────────────────────────────────────

class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.ac = ActorCritic(obs_dim, act_dim).to(self.device)
        self.opt = optim.Adam(self.ac.parameters(), lr=LR, betas=BETAS)
        self.buf = RolloutBuffer()
        self.ep_returns = deque(maxlen=100)

    def collect(self):
        obs, _ = self.env.reset()
        done, ep_ret, steps = False, 0.0, 0
        while len(self.buf) < UPDATE_EVERY:
            o_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            act, logp, _ = self.ac.act(o_t)
            val = self.ac.critic(o_t).item()
            nxt, rew, done, _, _ = self.env.step(act.squeeze(0))
            self.buf.add(obs, act, logp, rew, done, val)
            obs, ep_ret, steps = nxt, ep_ret + rew, steps + 1
            if done or steps >= MAX_STEPS:
                self.ep_returns.append(ep_ret)
                obs, _ = self.env.reset()
                done, ep_ret, steps = False, 0.0, 0

    def update(self):
        obs, acts, logps_old, rets, advs = self.buf.compute()
        obs, acts, logps_old, rets, advs = [t.to(self.device) for t in (obs, acts, logps_old, rets, advs)]
        for _ in range(EPOCHS):
            idx = torch.randperm(len(obs))
            for start in range(0, len(obs), BATCH_SIZE):
                mb = idx[start:start + BATCH_SIZE]
                logp, ent, value = self.ac.evaluate(obs[mb], acts[mb])
                ratio = (logp - logps_old[mb]).exp()
                surr1 = ratio * advs[mb]
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advs[mb]
                loss = (-torch.min(surr1, surr2).mean() +
                        VALUE_COEF * ((rets[mb] - value) ** 2).mean() -
                        ENTROPY_COEF * ent.mean())
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), MAX_GRAD_NORM)
                self.opt.step()

    def train(self):
        for ep in tqdm(range(1, TOTAL_EPISODES + 1)):
            self.collect()
            self.update()
            if ep % 10 == 0 and self.ep_returns:
                avg = np.mean(self.ep_returns)
                print(f"[{ep}/{TOTAL_EPISODES}] avg100 {avg:.1f}")
                if avg > 300:
                    break

    def save_actor_onnx(self, fname="submission_actor.onnx"):
        dummy = torch.randn(1, self.env.observation_space.shape[0]).to(self.device)
        torch.onnx.export(self.ac.actor, dummy, fname, opset_version=13)

    @torch.no_grad()
    def watch(self, n=3):
        e = gym.make(ENV_ID, render_mode="human", max_episode_steps=MAX_STEPS, hardcore=HARDCORE)
        for _ in range(n):
            o, _ = e.reset()
            d, r_tot = False, 0.0
            while not d:
                a, *_ = self.ac.act(torch.tensor(o, dtype=torch.float32).unsqueeze(0).to(self.device))
                o, r, d, _, _ = e.step(a.squeeze(0))
                r_tot += r
            print(r_tot)
        e.close()


# ───── main entry ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_seed(SEED)
    env = gym.make(ENV_ID, hardcore=HARDCORE, max_episode_steps=MAX_STEPS)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    agent = PPOAgent(env)
    start = time.time()
    agent.train()
    print(f"training finished in {(time.time() - start) / 60:.1f} min")
    agent.save_actor_onnx()
    agent.watch()