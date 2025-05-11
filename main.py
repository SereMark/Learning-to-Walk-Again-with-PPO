import os, random, torch, torch.optim as optim, torch.nn as nn, torch.nn.functional as F, numpy as np, gymnasium as gym, onnx
from gymnasium.wrappers import RecordEpisodeStatistics
from torch.utils.data import Dataset, DataLoader
from gymnasium import logger as gymlogger
from torch.distributions import Normal
from time import strftime
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gymlogger.set_level(gym.logger.ERROR)

def set_global_seed(seed_value=None):
    if seed_value is not None:
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)

class Logger:
    def __init__(self, log_filename="log.txt", hyperparameters_dict=None):
        self.log_file_path = log_filename
        current_time_str = strftime("%Y-%m-%dT%H-%M-%S")
        with open(self.log_file_path, 'w') as f:
            f.write(f"Timestamp: {current_time_str}\n")
            if hyperparameters_dict:
                f.write("Hyperparameters:\n")
                for key, value in hyperparameters_dict.items():
                    f.write(f"  {key}: {value}\n")

    def log_metrics(self, step_type, step_num, metrics_dict):
        log_entry = f"{step_type.upper()}, {step_num}"
        for key, value in metrics_dict.items():
            log_entry += f", {key}, {value:.4f}" if isinstance(value, float) else f", {key}, {value}"
        with open(self.log_file_path, 'a') as f:
            f.write(log_entry + "\n")

def create_gym_env(env_id='BipedalWalker-v3', hardcore=False, max_ep_steps=1600):
    env = gym.make(env_id, hardcore=hardcore, max_episode_steps=max_ep_steps)
    env = RecordEpisodeStatistics(env)
    return env, env.action_space.shape[0], env.observation_space.shape[0]

class Transition:
    def __init__(self, state_np, action_tensor, reward_float, done_bool, log_prob_tensor, value_float):
        self.state_np = state_np
        self.action_tensor = action_tensor
        self.reward_float = reward_float
        self.done_bool = done_bool
        self.log_prob_tensor = log_prob_tensor
        self.value_float = value_float
        self.advantage = 0.0
        self.gae_return = 0.0

class Episode:
    def __init__(self):
        self.transitions = []
        self.raw_reward_sum = 0.0

    def add_transition(self, transition_obj):
        self.transitions.append(transition_obj)

    def size(self):
        return len(self.transitions)

class BufferDataset(Dataset):
    def __init__(self, transitions_list):
        self.transitions_list = transitions_list

    def __len__(self):
        return len(self.transitions_list)

    def __getitem__(self, idx):
        t = self.transitions_list[idx]
        return t.state_np, t.action_tensor, t.log_prob_tensor, t.advantage, t.gae_return

class RolloutBuffer:
    def __init__(self, num_steps_for_batch, minibatch_size):
        self.num_steps_for_batch = num_steps_for_batch
        self.minibatch_size = minibatch_size
        self.storage = []

    def add_transition_object(self, trans_obj):
        self.storage.append(trans_obj)

    def compute_gae_and_returns(self, last_value_estimate, gamma, gae_lambda):
        advantage_running = 0.0
        if not self.storage:
            return

        for i in reversed(range(len(self.storage))):
            t = self.storage[i]
            next_val = last_value_estimate if i == len(self.storage) - 1 else self.storage[i+1].value_float
            next_non_terminal = 1.0 - float(t.done_bool)
            delta = t.reward_float + gamma * next_val * next_non_terminal - t.value_float
            advantage_running = delta + gamma * gae_lambda * next_non_terminal * advantage_running
            t.advantage = advantage_running
            t.gae_return = t.advantage + t.value_float

    def make_dataloader(self):
        if len(self.storage) < self.minibatch_size:
            return None
        dataset = BufferDataset(self.storage)
        return DataLoader(dataset, batch_size=self.minibatch_size, shuffle=True, drop_last=True)

    def clear_storage(self):
        self.storage.clear()

    def __len__(self):
        return len(self.storage)

class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=[256, 256]):
        super().__init__()
        if not isinstance(hidden_dim, list): hidden_dim = [hidden_dim]

        layers = []
        input_d = state_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(input_d, h_dim))
            layers.append(nn.ReLU())
            input_d = h_dim
        self.net = nn.Sequential(*layers)
        self.mu_logits_head = nn.Linear(input_d, action_dim)
        self.sigma_logits_head = nn.Linear(input_d, action_dim)
        torch.nn.init.orthogonal_(self.mu_logits_head.weight, gain=0.01)
        torch.nn.init.constant_(self.mu_logits_head.bias, 0.0)
        torch.nn.init.orthogonal_(self.sigma_logits_head.weight, gain=0.01)
        torch.nn.init.constant_(self.sigma_logits_head.bias, -0.5)

    def forward(self, state_tensor_batch):
        features = self.net(state_tensor_batch)
        return self.mu_logits_head(features), self.sigma_logits_head(features)

class CriticNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=[256, 256]):
        super().__init__()
        if not isinstance(hidden_dim, list): hidden_dim = [hidden_dim]

        layers = []
        input_d = state_dim
        for h_dim in hidden_dim:
            layers.append(nn.Linear(input_d, h_dim))
            layers.append(nn.ReLU())
            input_d = h_dim
        layers.append(nn.Linear(input_d, 1))
        self.net = nn.Sequential(*layers)
        torch.nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        torch.nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, state_tensor_batch):
        return self.net(state_tensor_batch)

class ActorCriticNet(nn.Module):
    def __init__(self, state_dim, action_dim, actor_hidden_dims, critic_hidden_dims):
        super().__init__()
        self.actor = ActorNet(state_dim, action_dim, actor_hidden_dims)
        self.critic = CriticNet(state_dim, critic_hidden_dims)

    def _get_distribution(self, state_tensor_batch):
        mu_logits, sigma_logits = self.actor(state_tensor_batch)
        mu = torch.tanh(mu_logits)
        sigma = F.softplus(sigma_logits) + 1e-5
        return Normal(mu, sigma)

    @torch.no_grad()
    def act_for_step(self, state_tensor_batch):
        self.eval()
        dist = self._get_distribution(state_tensor_batch)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        value = self.critic(state_tensor_batch)
        return action, log_prob, value

    def evaluate_actions(self, state_tensor_batch, action_tensor_batch):
        dist = self._get_distribution(state_tensor_batch)
        log_prob = dist.log_prob(action_tensor_batch).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        new_value = self.critic(state_tensor_batch)
        return log_prob, new_value, entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim, actor_hidden_dims, critic_hidden_dims, lr, logger, config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.logger = logger
        self.config = config
        self.current_entropy_coeff = config.entropy_initial
        self.ac_model = ActorCriticNet(state_dim, action_dim, actor_hidden_dims, critic_hidden_dims).to(device)
        self.optimizer = optim.Adam(self.ac_model.parameters(), lr=lr, betas=config.adam_betas, weight_decay=config.adam_weight_decay)

    def act(self, state_np):
        state_t = torch.from_numpy(state_np.astype(np.float32)).to(device).unsqueeze(0)
        action_t, log_prob_t, value_t = self.ac_model.act_for_step(state_t)
        return action_t, log_prob_t, value_t

    def train_on_buffer(self, rollout_buffer):
        self.ac_model.train()
        metrics_sum = {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        num_batches_processed = 0
        dataloader = rollout_buffer.make_dataloader()
        if dataloader is None:
            return {k: 0 for k in metrics_sum}
        for _ in range(self.config.ppo_epochs):
            for states_np, actions_b, old_log_probs_b, advantages_b, gae_returns_b in dataloader:
                states_t = torch.tensor(np.array(states_np), dtype=torch.float32, device=device)
                actions_t = actions_b.squeeze(1).to(device)
                old_log_probs_t = old_log_probs_b.squeeze(1).to(device)
                advantages_t = advantages_b.float().to(device).unsqueeze(1)
                gae_returns_t = gae_returns_b.float().to(device).unsqueeze(1)
                curr_log_probs, state_values_new, entropy = self.ac_model.evaluate_actions(states_t, actions_t)
                adv_norm = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
                ratios = torch.exp(curr_log_probs - old_log_probs_t.detach())
                surr1 = ratios * adv_norm
                surr2 = torch.clamp(ratios, 1.0 - self.config.clip_epsilon, 1.0 + self.config.clip_epsilon) * adv_norm
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(state_values_new, gae_returns_t)
                entropy_loss = -self.current_entropy_coeff * entropy.mean()
                total_loss = policy_loss + self.config.value_loss_coeff * value_loss + entropy_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac_model.parameters(), 0.5)
                self.optimizer.step()
                metrics_sum["total_loss"] += total_loss.item()
                metrics_sum["policy_loss"] += policy_loss.item()
                metrics_sum["value_loss"] += value_loss.item()
                metrics_sum["entropy"] += entropy.mean().item()
                num_batches_processed += 1
        return {k: v / num_batches_processed for k, v in metrics_sum.items()} if num_batches_processed > 0 else {k: 0 for k in metrics_sum}

    def export_actor_to_onnx(self, filename="actor.onnx"):
        self.ac_model.actor.eval()
        dummy_input = torch.randn(1, self.state_dim, device=device)
        torch.onnx.export(self.ac_model.actor, dummy_input, filename, input_names=['input_state'], output_names=['mu_logits', 'sigma_logits'], opset_version=11, export_params=True, do_constant_folding=True, dynamic_axes={'input_state': {0: 'batch_size'}, 'mu_logits': {0: 'batch_size'}, 'sigma_logits': {0: 'batch_size'}})
        onnx_model = onnx.load(filename)
        onnx.checker.check_model(onnx_model)

    def save_model_state(self, filepath):
        torch.save(self.ac_model.state_dict(), filepath)

    def load_model_state(self, filepath):
        if os.path.exists(filepath):
            self.ac_model.load_state_dict(torch.load(filepath, map_location=device))

class TrainingRunner:
    def __init__(self, env, agent, buffer, logger, config):
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        self.config = config
        self.num_ppo_update_calls = 0
        self.all_episode_stats = []
        self.best_rolling_avg_reward = -float('inf')

    def _update_learning_params(self):
        self.num_ppo_update_calls += 1
        progress = min(1.0, self.num_ppo_update_calls / float(self.config.total_expected_updates))
        current_lr = self.config.lr_initial * (1.0 - progress)
        for param_group in self.agent.optimizer.param_groups:
            param_group['lr'] = max(current_lr, 1e-6)
        coeff_range = self.config.entropy_initial - self.config.entropy_final
        current_entropy_coeff = self.config.entropy_initial - coeff_range * progress
        self.agent.current_entropy_coeff = max(current_entropy_coeff, self.config.entropy_final)
        return current_lr, self.agent.current_entropy_coeff

    def run_training_loop(self):
        total_steps_for_update = 0
        for ep_idx in tqdm(range(self.config.num_episodes), desc="Training Progress"):
            state_np, info = self.env.reset(seed=self.config.seed + ep_idx if self.config.seed is not None else None)
            ep_raw_reward = 0.0
            for ep_step in range(self.config.env_max_steps):
                action_t, log_prob_t, value_t = self.agent.act(state_np)
                action_np = action_t.squeeze(0).cpu().numpy()
                next_state_np, raw_reward, terminated, truncated, info = self.env.step(np.clip(action_np, self.env.action_space.low, self.env.action_space.high))
                ep_raw_reward += raw_reward
                done = terminated or truncated
                self.buffer.add_transition_object(Transition(state_np, action_t, self.config.reward_scaler(raw_reward), done, log_prob_t, value_t.squeeze().item()))
                state_np = next_state_np
                total_steps_for_update += 1
                if done:
                    break
            ep_stats = info.get("episode", {"r": ep_raw_reward, "l": ep_step + 1})
            self.all_episode_stats.append({"raw_reward": ep_stats["r"], "length": ep_stats["l"]})
            if len(self.all_episode_stats) >= 100:
                current_rolling_avg = np.mean([s["raw_reward"] for s in self.all_episode_stats[-100:]])
                if current_rolling_avg > self.best_rolling_avg_reward:
                    self.best_rolling_avg_reward = current_rolling_avg
                    tqdm.write(f"Ep {ep_idx}: New best 100-ep avg: {current_rolling_avg:.2f}. Saving model...")
                    self.agent.save_model_state(self.config.best_model_path)
            if total_steps_for_update >= self.buffer.num_steps_for_batch:
                current_lr, current_entropy_coeff = self._update_learning_params()
                last_val_estimate = 0.0
                if not done:
                    _, _, last_val_t = self.agent.act(state_np)
                    last_val_estimate = last_val_t.squeeze().item()
                self.buffer.compute_gae_and_returns(last_val_estimate, self.config.gamma, self.config.gae_lambda)
                avg_metrics = self.agent.train_on_buffer(self.buffer)
                log_data = {
                    **avg_metrics,
                    "current_entropy_coeff": current_entropy_coeff,
                    "current_lr": current_lr,
                    "roll_avg_rew100": np.mean([s["raw_reward"] for s in self.all_episode_stats[-100:]]) if len(self.all_episode_stats) > 0 else 0.0,
                    "roll_avg_len100": np.mean([s["length"] for s in self.all_episode_stats[-100:]]) if len(self.all_episode_stats) > 0 else 0.0,
                    "hist_best_roll_avg_rew": self.best_rolling_avg_reward
                }
                self.logger.log_metrics("update", self.num_ppo_update_calls, log_data)
                self.buffer.clear_storage()
                total_steps_for_update = 0
        self.env.close()

class Config:
    def __init__(self):
        self.env_id = 'BipedalWalker-v3'
        self.num_episodes = 12000
        self.env_max_steps = 1600
        self.actor_hidden_dims = [256, 256]
        self.critic_hidden_dims = [256, 256]
        self.lr_initial = 1.5e-4
        self.clip_epsilon = 0.2
        self.ppo_epochs = 10
        self.value_loss_coeff = 0.5
        self.entropy_initial = 0.0001
        self.entropy_final = 0.000001
        self.adam_betas = (0.9, 0.999)
        self.adam_weight_decay = 1e-5
        self.transitions_per_update = 4096
        self.minibatch_size = 128
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.reward_scaler = lambda r: r
        self.seed = 42
        self.best_model_path = "bipedalwalker_ppo_best_ac_state.pth"
        self.best_onnx_path = "bipedalwalker_best_actor.onnx"
        self.final_model_path = "bipedalwalker_ppo_final_ac_state.pth"
        self.final_onnx_path = "bipedalwalker_final_actor.onnx"
        self.total_env_steps_estimate = self.num_episodes * self.env_max_steps
        self.total_expected_updates = max(1, int(self.total_env_steps_estimate // self.transitions_per_update))

if __name__ == "__main__":
    config = Config()
    set_global_seed(config.seed)
    hyperparam_log_dict = {k: v for k, v in vars(config).items() if not k.startswith('_')}
    main_logger = Logger(log_filename=f"log_ppo_{config.env_id.lower()}.txt", hyperparameters_dict=hyperparam_log_dict)
    train_env, action_dim, obs_dim = create_gym_env(config.env_id, max_ep_steps=config.env_max_steps)
    rollout_buffer = RolloutBuffer(num_steps_for_batch=config.transitions_per_update, minibatch_size=config.minibatch_size)
    ppo_agent = PPOAgent(state_dim=obs_dim, action_dim=action_dim, actor_hidden_dims=config.actor_hidden_dims, critic_hidden_dims=config.critic_hidden_dims, lr=config.lr_initial, logger=main_logger, config=config)
    trainer = TrainingRunner(env=train_env, agent=ppo_agent, buffer=rollout_buffer, logger=main_logger, config=config)
    print(f"Device: {device}")
    trainer.run_training_loop()
    ppo_agent.save_model_state(filepath=config.final_model_path)
    ppo_agent.export_actor_to_onnx(filename=config.final_onnx_path)
    ppo_agent.load_model_state(filepath=config.best_model_path)
    ppo_agent.export_actor_to_onnx(filename=config.best_onnx_path)