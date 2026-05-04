import numpy as np
import random
from collections import deque
from abc import ABC, abstractmethod

class BaseScheduler(ABC):
    def __init__(self, n_sites, budget_per_hour=60):
        self.n_sites = n_sites
        self.budget_per_hour = budget_per_hour
        self.reset()

    def reset(self):
        self.crawl_counts = np.zeros(self.n_sites, dtype=int)
        self.change_counts = np.zeros(self.n_sites, dtype=int)
        self.last_crawl_time = np.zeros(self.n_sites)
        self.total_changes_detected = 0
        self.total_crawls_done = 0
        self.freshness_log = []
        self.crawl_log = []

    @abstractmethod
    def select_next_site(self, current_time): pass

    def update(self, site_id, result, current_time):
        self.crawl_counts[site_id] += 1
        self.total_crawls_done += 1
        self.last_crawl_time[site_id] = current_time
        if result.price_changed:
            self.change_counts[site_id] += 1
            self.total_changes_detected += 1
        self.crawl_log.append((current_time, site_id))

    def get_change_rate(self, site_id):
        if self.crawl_counts[site_id] == 0:
            return 0.5
        return self.change_counts[site_id] / self.crawl_counts[site_id]


class AdaptiveCrawlScheduler(BaseScheduler):
    """Adaptive Crawling: interval = base_interval / change_rate"""
    def __init__(self, n_sites, budget_per_hour=60, base_interval=1.0,
                 window_size=10, min_interval=0.1, max_interval=5.0):
        super().__init__(n_sites, budget_per_hour)
        self.base_interval = base_interval
        self.window_size = window_size
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.history = {i: deque(maxlen=window_size) for i in range(n_sites)}
        self.next_crawl_time = np.zeros(n_sites)

    def reset(self):
        super().reset()
        self.history = {i: deque(maxlen=self.window_size) for i in range(self.n_sites)} if hasattr(self, 'window_size') else {}
        self.next_crawl_time = np.zeros(self.n_sites)

    def _compute_interval(self, site_id):
        h = list(self.history.get(site_id, []))
        if not h:
            return self.base_interval
        rate = max(0.01, sum(h) / len(h))
        return float(np.clip(self.base_interval / rate, self.min_interval, self.max_interval))

    def select_next_site(self, current_time):
        overdue = current_time - self.next_crawl_time
        return int(np.argmax(overdue))

    def update(self, site_id, result, current_time):
        super().update(site_id, result, current_time)
        if site_id not in self.history:
            self.history[site_id] = deque(maxlen=self.window_size)
        self.history[site_id].append(int(result.price_changed))
        self.next_crawl_time[site_id] = current_time + self._compute_interval(site_id)

    def get_intervals(self):
        return np.array([self._compute_interval(i) for i in range(self.n_sites)])


class PriorityQueueScheduler(BaseScheduler):
    """Priority Queue: score = (elapsed * change_prob) / response_time"""
    def __init__(self, n_sites, budget_per_hour=60, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__(n_sites, budget_per_hour)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.response_times = {i: deque(maxlen=20) for i in range(n_sites)}
        self.mean_response = np.ones(n_sites)

    def reset(self):
        super().reset()
        self.response_times = {i: deque(maxlen=20) for i in range(self.n_sites)}
        self.mean_response = np.ones(self.n_sites)

    def _score(self, site_id, current_time):
        elapsed = max(0.01, current_time - self.last_crawl_time[site_id])
        cp = self.get_change_rate(site_id)
        rt = max(0.1, self.mean_response[site_id])
        return (self.alpha * elapsed * self.beta * cp) / (self.gamma * rt)

    def select_next_site(self, current_time):
        return int(np.argmax([self._score(i, current_time) for i in range(self.n_sites)]))

    def update(self, site_id, result, current_time):
        super().update(site_id, result, current_time)
        self.response_times[site_id].append(result.response_time)
        self.mean_response[site_id] = np.mean(list(self.response_times[site_id]))

    def get_scores(self, current_time):
        return np.array([self._score(i, current_time) for i in range(self.n_sites)])


class PoissonScheduler(BaseScheduler):
    """Poisson Process: optimal_interval = 1 / sqrt(2 * lambda * cost)"""
    def __init__(self, n_sites, budget_per_hour=60, crawl_cost=0.1,
                 prior_lambda=0.3, learning_rate=0.1):
        super().__init__(n_sites, budget_per_hour)
        self.crawl_cost = crawl_cost
        self.prior_lambda = prior_lambda
        self.learning_rate = learning_rate
        self.lambda_estimates = np.full(n_sites, prior_lambda)
        self.next_crawl_time = np.zeros(n_sites)

    def reset(self):
        super().reset()
        self.lambda_estimates = np.full(self.n_sites, self.prior_lambda if hasattr(self,'prior_lambda') else 0.3)
        self.next_crawl_time = np.zeros(self.n_sites)

    def _update_lambda(self, site_id, changed, elapsed):
        if elapsed > 0:
            obs = (1.0 / elapsed) if changed else 0.0
            self.lambda_estimates[site_id] = np.clip(
                (1 - self.learning_rate) * self.lambda_estimates[site_id] + self.learning_rate * obs,
                0.01, 10.0)

    def _optimal_interval(self, site_id):
        lam = self.lambda_estimates[site_id]
        return float(np.clip(1.0 / np.sqrt(2.0 * lam * self.crawl_cost), 0.05, 8.0))

    def select_next_site(self, current_time):
        return int(np.argmax(current_time - self.next_crawl_time))

    def update(self, site_id, result, current_time):
        elapsed = current_time - self.last_crawl_time[site_id]
        super().update(site_id, result, current_time)
        self._update_lambda(site_id, result.price_changed, elapsed)
        self.next_crawl_time[site_id] = current_time + self._optimal_interval(site_id)

    def get_lambda_estimates(self):
        return self.lambda_estimates.copy()

    def get_optimal_intervals(self):
        return np.array([self._optimal_interval(i) for i in range(self.n_sites)])


class UCBBanditScheduler(BaseScheduler):
    """UCB1 Bandit: score = avg_reward + sqrt(2*ln(N)/n_i)"""
    def __init__(self, n_sites, budget_per_hour=60, exploration_c=2.0):
        super().__init__(n_sites, budget_per_hour)
        self.exploration_c = exploration_c
        self.cumulative_reward = np.zeros(n_sites)

    def reset(self):
        super().reset()
        self.cumulative_reward = np.zeros(self.n_sites)

    def _ucb_score(self, site_id):
        n_total = max(1, self.total_crawls_done)
        n_site = max(1, self.crawl_counts[site_id])
        avg = self.cumulative_reward[site_id] / n_site
        bonus = np.sqrt(self.exploration_c * np.log(n_total) / n_site)
        return avg + bonus

    def select_next_site(self, current_time):
        unvisited = np.where(self.crawl_counts == 0)[0]
        if len(unvisited) > 0:
            return int(unvisited[0])
        return int(np.argmax([self._ucb_score(i) for i in range(self.n_sites)]))

    def update(self, site_id, result, current_time):
        super().update(site_id, result, current_time)
        self.cumulative_reward[site_id] += 1.0 if result.price_changed else 0.05

    def get_ucb_scores(self):
        return np.array([self._ucb_score(i) for i in range(self.n_sites)])


class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buf = deque(maxlen=capacity)
    def push(self, s, a, r, ns):
        self.buf.append((s, a, r, ns))
    def sample(self, n):
        return random.sample(self.buf, min(n, len(self.buf)))
    def __len__(self):
        return len(self.buf)


class SimpleDQN:
    def __init__(self, s_dim, a_dim, h=64, lr=0.001):
        self.lr = lr
        self.W1 = np.random.randn(s_dim, h) * 0.1; self.b1 = np.zeros(h)
        self.W2 = np.random.randn(h, h) * 0.1;     self.b2 = np.zeros(h)
        self.W3 = np.random.randn(h, a_dim) * 0.1; self.b3 = np.zeros(a_dim)
        self.tW1,self.tb1=self.W1.copy(),self.b1.copy()
        self.tW2,self.tb2=self.W2.copy(),self.b2.copy()
        self.tW3,self.tb3=self.W3.copy(),self.b3.copy()
    def relu(self, x): return np.maximum(0, x)
    def forward(self, x, tgt=False):
        W1,b1=(self.tW1,self.tb1) if tgt else (self.W1,self.b1)
        W2,b2=(self.tW2,self.tb2) if tgt else (self.W2,self.b2)
        W3,b3=(self.tW3,self.tb3) if tgt else (self.W3,self.b3)
        return self.relu(self.relu(x@W1+b1)@W2+b2)@W3+b3
    def update_target(self):
        self.tW1,self.tb1=self.W1.copy(),self.b1.copy()
        self.tW2,self.tb2=self.W2.copy(),self.b2.copy()
        self.tW3,self.tb3=self.W3.copy(),self.b3.copy()
    def train_step(self, states, actions, targets):
        h1=self.relu(states@self.W1+self.b1)
        h2=self.relu(h1@self.W2+self.b2)
        q=h2@self.W3+self.b3
        lg=np.zeros_like(q)
        for i,a in enumerate(actions): lg[i,a]=2*(q[i,a]-targets[i])/len(states)
        dW3=h2.T@lg; db3=lg.sum(0)
        dh2=lg@self.W3.T*( (h1@self.W2+self.b2)>0)
        dW2=h1.T@dh2; db2=dh2.sum(0)
        dh1=dh2@self.W2.T*((states@self.W1+self.b1)>0)
        dW1=states.T@dh1; db1=dh1.sum(0)
        for p,g in [(self.W3,dW3),(self.b3,db3),(self.W2,dW2),(self.b2,db2),
                    (self.W1,dW1),(self.b1,db1)]:
            p -= self.lr * np.clip(g, -1, 1)


class DQNScheduler(BaseScheduler):
    """DQN: MDP formulation, state=[elapsed,change_rate,resp_time] per site"""
    def __init__(self, n_sites, budget_per_hour=60, hidden_dim=64, lr=0.001,
                 gamma=0.95, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay=0.995, batch_size=32, target_update_freq=50):
        super().__init__(n_sites, budget_per_hour)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.dqn = SimpleDQN(n_sites * 3, n_sites, hidden_dim, lr)
        self.replay = ReplayBuffer(5000)
        self.resp_est = np.ones(n_sites)
        self.step_count = 0
        self._last_state = None

    def reset(self):
        super().reset()
        if hasattr(self, 'dqn'):
            self.replay = ReplayBuffer(5000)
            self.resp_est = np.ones(self.n_sites)
            self.step_count = 0
            self._last_state = None

    def _state(self, t):
        s = []
        for i in range(self.n_sites):
            s += [(t - self.last_crawl_time[i])/10., self.get_change_rate(i), self.resp_est[i]/5.]
        return np.array(s, dtype=np.float32)

    def select_next_site(self, current_time):
        state = self._state(current_time)
        self._last_state = state
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_sites)
        return int(np.argmax(self.dqn.forward(state.reshape(1,-1))[0]))

    def update(self, site_id, result, current_time):
        old = self._last_state
        super().update(site_id, result, current_time)
        self.resp_est[site_id] = 0.8*self.resp_est[site_id] + 0.2*result.response_time
        reward = (1.5 if result.price_changed else 0.0) - result.response_time*0.1
        ns = self._state(current_time)
        if old is not None:
            self.replay.push(old, site_id, reward, ns)
        if len(self.replay) >= self.batch_size:
            batch = self.replay.sample(self.batch_size)
            S = np.array([b[0] for b in batch], dtype=np.float32)
            A = [b[1] for b in batch]
            R = np.array([b[2] for b in batch], dtype=np.float32)
            NS = np.array([b[3] for b in batch], dtype=np.float32)
            tgt = R + self.gamma * np.max(self.dqn.forward(NS, tgt=True), axis=1)
            self.dqn.train_step(S, A, tgt)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.dqn.update_target()