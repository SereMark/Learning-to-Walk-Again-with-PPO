import random, gymnasium as gym, numpy as np, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from collections import deque, namedtuple
from torch.distributions import Normal
from tqdm import tqdm

SEED              = 31
ENV_ID            = "BipedalWalker-v3"
HARDCORE          = False
MAX_EPISODES      = 1200
MAX_STEPS         = 1600
UPDATE_EVERY      = 8192
EPOCHS            = 4
MINIBATCH_SIZE    = 256
GAMMA             = 0.99
GAE_LAMBDA        = 0.95
CLIP_EPS          = 0.2
LR_START          = 3e-4
ENTROPY_COEF      = 1e-2
VALUE_COEF        = 0.5
MAX_GRAD_NORM     = 0.5
HIDDEN            = 128

ONNX_EXPORT       = "submission_actor.onnx"

def orthogonal_init(layer, gain: float = 1.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain)
        nn.init.constant_(layer.bias, 0)

class RunningNorm:
    def __init__(self, shape, eps=1e-4):
        self.mean = torch.zeros(shape)
        self.var  = torch.ones(shape)
        self.count = eps
    def update(self, x: torch.Tensor):
        batch_mean = x.mean(0)
        batch_var  = x.var(0, unbiased=False)
        batch_count = x.size(0)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count
    def __call__(self, x: torch.Tensor):
        return (x - self.mean) / (self.var.sqrt() + 1e-8)

class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN), nn.Tanh(),
            nn.Linear(HIDDEN, HIDDEN), nn.Tanh()
        )
        self.mu_head     = nn.Linear(HIDDEN, act_dim)
        self.logsig_head = nn.Linear(HIDDEN, act_dim)
        self.apply(lambda m: orthogonal_init(m, np.sqrt(2)))
        orthogonal_init(self.mu_head, 0.01)
        orthogonal_init(self.logsig_head, 0.01)

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
        self.apply(lambda m: orthogonal_init(m, np.sqrt(2)))
        orthogonal_init(self.net[-1], 1.0)
    def forward(self, x): return self.net(x).squeeze(-1)

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor  = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim)

    def _dist(self, obs: torch.Tensor):
        mu_logits, logsig_logits = self.actor(obs)
        sigma = F.softplus(logsig_logits) + 1e-5
        return Normal(mu_logits, sigma)

    @torch.no_grad()
    def act(self, obs: torch.Tensor):
        dist = self._dist(obs)
        raw  = dist.rsample()
        act  = torch.tanh(raw)
        logp = dist.log_prob(raw).sum(-1) - torch.log(1 - act.pow(2) + 1e-6).sum(-1)
        return act.cpu().numpy(), logp, dist

    def evaluate(self, obs: torch.Tensor, act: torch.Tensor):
        dist = self._dist(obs)
        raw_act = torch.atanh(torch.clamp(act, -0.999, 0.999))
        logp = dist.log_prob(raw_act).sum(-1) - torch.log(1 - act.pow(2) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs)
        return logp, entropy, value

Transition = namedtuple("Transition", "obs act logp rew done val")

class RolloutBuffer:
    def __init__(self): self.reset()
    def reset(self): self.data = []
    def add(self, *args): self.data.append(Transition(*args))
    def __len__(self): return len(self.data)

    def compute(self, gamma=GAMMA, lam=GAE_LAMBDA):
        obs  = torch.tensor(np.stack([t.obs for t in self.data]), dtype=torch.float32)
        acts = torch.tensor(np.stack([t.act for t in self.data]), dtype=torch.float32)
        logp = torch.tensor(np.stack([t.logp for t in self.data]), dtype=torch.float32)
        rews = [t.rew for t in self.data]
        dones= [t.done for t in self.data]
        vals = torch.tensor(np.stack([t.val for t in self.data]), dtype=torch.float32)

        advs, gae, next_val = [], 0.0, 0.0
        for i in reversed(range(len(rews))):
            delta = rews[i] + gamma * next_val * (1 - dones[i]) - vals[i].item()
            gae   = delta + gamma * lam * (1 - dones[i]) * gae
            advs.insert(0, gae)
            next_val = vals[i].item()
        advs = torch.tensor(advs, dtype=torch.float32)
        returns = advs + vals
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        self.reset()
        return obs, acts, logp, returns, advs

class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cpu")
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        self.ac = ActorCritic(obs_dim, act_dim).to(self.device)
        self.opt = optim.Adam(self.ac.parameters(), lr=LR_START, eps=1e-5)
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(
            self.opt, lambda f: 1 - f)
        self.buf = RolloutBuffer()
        self.obs_norm = RunningNorm(obs_dim)
        self.ep_returns = deque(maxlen=100)

    def _proc_obs(self, o):
        o_t = torch.tensor(o, dtype=torch.float32).to(self.device)
        o_n = self.obs_norm(o_t)
        return o_t, o_n.unsqueeze(0)

    def collect(self):
        obs, _ = self.env.reset(seed=SEED)
        self.obs_norm.update(torch.tensor(obs))
        done, ep_ret, steps = False, 0.0, 0

        while len(self.buf) < UPDATE_EVERY:
            o_t, o_in = self._proc_obs(obs)
            act, logp, _ = self.ac.act(o_in)
            val = self.ac.critic(o_in).item()
            nxt, rew, done, _, _ = self.env.step(act.squeeze(0))
            self.buf.add(o_t.numpy(), act, logp, rew, done, val)

            obs, ep_ret, steps = nxt, ep_ret + rew, steps + 1
            self.obs_norm.update(torch.tensor(obs))
            if done or steps >= MAX_STEPS:
                self.ep_returns.append(ep_ret)
                obs, _ = self.env.reset()
                self.obs_norm.update(torch.tensor(obs))
                done, ep_ret, steps = False, 0.0, 0

    def update(self, epoch_i):
        obs, acts, logp_old, rets, advs = self.buf.compute()
        obs, acts, logp_old, rets, advs = [t.to(self.device) for t in (obs, acts, logp_old, rets, advs)]
        num_samples = len(obs)
        for _ in range(EPOCHS):
            idx = torch.randperm(num_samples)
            for start in range(0, num_samples, MINIBATCH_SIZE):
                mb = idx[start:start+MINIBATCH_SIZE]
                logp, ent, val = self.ac.evaluate(obs[mb], acts[mb])
                ratio = (logp - logp_old[mb]).exp()
                surr1 = ratio * advs[mb]
                surr2 = torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * advs[mb]
                loss = (-torch.min(surr1, surr2).mean()
                        + VALUE_COEF * (rets[mb] - val).pow(2).mean()
                        - ENTROPY_COEF * ent.mean())
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), MAX_GRAD_NORM)
                self.opt.step()
        self.lr_scheduler.step(epoch_i / MAX_EPISODES)

    def train(self):
        for ep in tqdm(range(1, MAX_EPISODES + 1), ncols=80):
            self.collect()
            self.update(ep)
            if ep % 10 == 0 and self.ep_returns:
                avg = np.mean(self.ep_returns)
                if avg > 300:
                    print("Solved, stopping early.")
                    break

    class _ExportActor(nn.Module):
        def __init__(self, base_actor): super().__init__(); self.base = base_actor
        def forward(self, x):
            mu, logsig = self.base(x)
            return torch.tanh(mu), logsig

    def export_onnx(self, fname=ONNX_EXPORT):
        dummy = torch.randn(1, self.env.observation_space.shape[0])
        torch.onnx.export(self._ExportActor(self.ac.actor), dummy, fname, opset_version=13)
        print(f"[+] exported actor → {fname}")

    @torch.no_grad()
    def watch(self, episodes=3):
        watch_env = gym.make(ENV_ID, render_mode="human", max_episode_steps=MAX_STEPS, hardcore=HARDCORE)
        for ep in range(episodes):
            o, _ = watch_env.reset()
            done, ret = False, 0.0
            while not done:
                _, o_in = self._proc_obs(o)
                act, *_ = self.ac.act(o_in)
                o, r, done, _, _ = watch_env.step(act.squeeze(0))
                ret += r
            print(f"Episode {ep+1}: {ret:.1f}")
        watch_env.close()

if __name__ == "__main__":
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    env = gym.make(ENV_ID, max_episode_steps=MAX_STEPS, hardcore=HARDCORE)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    agent = PPOAgent(env)
    agent.train()
    agent.export_onnx()
    agent.watch()