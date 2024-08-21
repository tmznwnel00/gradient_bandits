#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Artem Oboturov(oboturov@gmail.com)                             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

matplotlib.use('Agg')

def laplace_mech(v, sensitivity, epsilon):
    return v + np.random.laplace(loc=0, scale=sensitivity/epsilon)

def score(data, option):
    return data.value_counts()[option]/1000

def exponential(x, R, u, sensitivity, epsilon):
    # Calculate the score for each element of R
    scores = [u(x, r) for r in R]
    
    # Calculate the probability for each element, based on its score
    probabilities = [np.exp(epsilon * score / (2 * sensitivity)) for score in scores]
    
    # Normalize the probabilties so they sum to 1
    probabilities = probabilities / np.linalg.norm(probabilities, ord=1)

    # Choose an element from R based on the probabilities
    return np.random.choice(R, 1, p=probabilities)[0]

def report_noisy_max(x, R, u, sensitivity, epsilon):
    # Calculate the score for each element of R
    scores = [u(x, r) for r in R]

    # Add noise to each score
    noisy_scores = [laplace_mech(score, sensitivity, epsilon) for score in scores]

    # Find the index of the maximum score
    max_idx = np.argmax(noisy_scores)
    
    # Return the element corresponding to that index
    return R[max_idx]

class Bandit:
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @UCB_param: if not None, use UCB algorithm to select action
    # @gradient: if True, use gradient based bandit algorithm
    # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1, sample_averages=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, true_reward=0., noise_type=None, noise_scale=0.1, 
                 dp_UCB=False, epsilon_dp_UCB=0.1, time_size=1000):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial
        self.noise_type = noise_type  # 'gaussian', 'laplace', or None
        self.noise_scale = noise_scale  # Scale of the noise
        self.dp_UCB = dp_UCB
        self.epsilon_dp_UCB = epsilon_dp_UCB
        self.time_size = time_size

        if self.dp_UCB:
            # self.mu = [0.70]*self.k
            # self.mu[0] = 0.75
            self.mu = np.random.randn(self.k) + self.true_reward
            self.mu = np.array(self.mu)
            self.u_star = max(self.mu)
            self.k_n = np.ones(self.k)
            self.noisy_sums = self.mu.copy()
            self.sums = self.mu.copy()
            self.r = (self.u_star - self.mu).sum()

            self.gamma =  self.k*(np.log(self.time_size)**2)*(np.log((self.k * self.time_size * np.log(self.time_size)) / 0.1))/self.epsilon_dp_UCB

            self.epsilon_dp_UCB = self.epsilon_dp_UCB/self.k
            self.epsilon1 = self.epsilon_dp_UCB / np.log(self.time_size)

            self.logn_2 = int(np.log(self.time_size-self.k)/np.log(2))
            self.alpha = np.zeros((self.logn_2 + 1, self.k))
            self.alpha_hat = np.zeros((self.logn_2 + 1, self.k))
        
            
    def reset(self):
        # real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward
        
        # custom reward for expermient of dp_ucb
        # self.q_true = [0.70]*self.k
        # self.q_true[0] = 0.75

        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)

        self.time = 0
        
    # get an action for this bandit
    def act(self):
        if self.dp_UCB:
            if self.time < self.k:
                return self.time
            else:
                return np.argmax(self.noisy_sums/self.k_n + np.sqrt(2*np.log(self.time)/self.k_n) + self.gamma/self.k_n)
            
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation + \
                self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation == q_best)[0])

        if self.gradient:
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)
        
        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    # take an action, update estimation for this action
    def step(self, action):
        if self.dp_UCB:
            if self.time < self.k:
                reward = self.mu[self.time]
                regret = self.u_star - reward
            else:
                p = 1 / (1 + np.exp(-self.mu[action]))
                reward = np.random.binomial(n=1, p=p)
                reward_stream_at_t = np.zeros(self.k)
                reward_stream_at_t[action] = np.random.binomial(n=1, p=p)
                if self.time+1 == self.k+1:
                    reward_stream_at_t += self.mu.copy()
                # update the statistics
                # number of pulls
                self.k_n[action] += 1
                #noisy sums using the tree mechanism
                binary_rep = np.array(list(np.binary_repr(self.time+1 - self.k, width = self.logn_2 + 1))).astype(int)
                i = np.min(np.nonzero(np.flip(binary_rep)))
                self.alpha[i] = self.alpha[:i].sum(axis=0) + reward_stream_at_t
                for j in range(i):
                    self.alpha[j] = 0
                    self.alpha_hat[j] = 0
                self.alpha_hat[i] = self.alpha[i] + np.random.laplace(loc=0.0, scale=self.noise_scale, size=self.k)
                self.noisy_sums = self.alpha_hat.T.dot(np.flip(binary_rep))
                self.sums[action] += self.mu[action]
                # update total reward
                #total_reward += reward
                regret = self.u_star - reward
                
            self.time += 1
            self.action_count[action] += 1
            self.average_reward += (reward - self.average_reward) / self.time
        else:
            reward = np.random.randn() + self.q_true[action]
            self.time += 1
            self.action_count[action] += 1
            self.average_reward += (reward - self.average_reward) / self.time
            if self.sample_averages:
                # update estimation using sample averages
                self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
            elif self.gradient:
                one_hot = np.zeros(self.k)
                one_hot[action] = 1
                if self.gradient_baseline:
                    baseline = self.average_reward
                else:
                    baseline = 0

                noise = 0
                if self.noise_type == 'gaussian':
                    noise = np.random.normal(scale=self.noise_scale, size=self.k)
                elif self.noise_type == 'laplace':
                    noise = np.random.laplace(scale=self.noise_scale, size=self.k)
                self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob) + noise
                # self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob) 
            else:
                # update estimation with constant step size
                self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
            regret = self.q_true[self.best_action] - reward
        return reward, regret


def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    cumulative_regret = np.zeros((len(bandits), runs, time))
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            cumulative_regret_r = np.zeros(time)

            for t in range(time):
                action = bandit.act()  
                reward, regret = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
                if t == 0:
                    cumulative_regret_r[t] = regret
                else:
                    cumulative_regret_r[t] = cumulative_regret_r[t-1] + regret
            cumulative_regret[i, r, :] = cumulative_regret_r
    
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    mean_cumulative_regret = cumulative_regret.mean(axis=1)
    return mean_best_action_counts, mean_rewards, mean_cumulative_regret

def average_reward(runs=2000, time=2000):
    bandits = []
    bandits.append(Bandit(epsilon=0.1, UCB_param=1, sample_averages=True))
    bandits.append(Bandit(epsilon=0.1, UCB_param=2, sample_averages=True))
    bandits.append(Bandit(epsilon=0.1, UCB_param=4, sample_averages=True))
    # bandits.append(Bandit(dp_UCB=True, epsilon_dp_UCB=0.1, time_size=time, noise_scale=1.0))
    _, average_rewards, _ = simulate(runs, time, bandits)

    plt.plot(average_rewards[0], label=r'UCB $\epsilon = 0.1$, $c = 1$')
    plt.plot(average_rewards[1], label=r'UCB $\epsilon = 0.1$, $c = 2$')
    plt.plot(average_rewards[2], label=r'UCB $\epsilon = 0.1$, $c = 4$')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()

    plt.savefig('images/average_reward.png')
    plt.close()


def optimal_action(runs=2000, time=1000):
    bandits = []
    # bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=True, true_reward=4))

    # bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=True, true_reward=4, noise_type='gaussian', noise_scale=0.1))

    # bandits.append(Bandit(gradient=True, step_size=0.1, gradient_baseline=True, true_reward=4, noise_type='laplace', noise_scale=1.0))
    # bandits.append(Bandit(gradient=True, step_size=0.5, gradient_baseline=True, true_reward=4, noise_type='laplace', noise_scale=1.0))
    # bandits.append(Bandit(gradient=True, step_size=1.0, gradient_baseline=True, true_reward=4, noise_type='laplace', noise_scale=1.0))
    # bandits.append(Bandit(dp_UCB=True, epsilon_dp_UCB=0.1, time_size=time, noise_scale=1.0))

    # bandits.append(Bandit(epsilon=0, UCB_param=2, sample_averages=True))
    bandits.append(Bandit(epsilon=0.1, UCB_param=1, sample_averages=True))
    bandits.append(Bandit(epsilon=0.1, UCB_param=2, sample_averages=True))
    bandits.append(Bandit(epsilon=0.1, UCB_param=4, sample_averages=True))
    # bandits.append(Bandit(epsilon=0.5, UCB_param=2, sample_averages=True))

    best_action_counts, _, _ = simulate(runs, time, bandits)
    labels = [
        # r'UCB $\epsilon = 0$, $c = 2$',
        r'UCB $\epsilon = 0.1$, $c = 1$',
        r'UCB $\epsilon = 0.1$, $c = 2$',
        r'UCB $\epsilon = 0.1$, $c = 4$',
        # r'UCB $\epsilon = 0.5$, $c = 2$',
    ]
    # best_action_counts, _ = simulate(runs, time, bandits)
    # labels = [r'$\alpha = 0.1$, with baseline',
    #           r'$\alpha = 0.1$, without baseline',
    #           r'$\alpha = 0.4$, with baseline',
    #           r'$\alpha = 0.4$, without baseline']

    for i in range(len(bandits)):
        plt.plot(best_action_counts[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    plt.savefig('images/optimal_action.png')
    plt.close()

def cumulative_regret(runs=2000, time=1000):
    bandits = []
    # bandits.append(Bandit(epsilon=0, UCB_param=2, sample_averages=True))
    # bandits.append(Bandit(epsilon=0.1, sample_averages=True))
    # bandits.append(Bandit(gradient=True, step_size=1.0, gradient_baseline=True, noise_type='laplace', noise_scale=1.0))
    # bandits.append(Bandit(gradient=True, step_size=2.0, gradient_baseline=True, noise_type='laplace', noise_scale=1.0))
    # bandits.append(Bandit(dp_UCB=True, epsilon_dp_UCB=0.1, time_size=time, noise_scale=1.0))

    bandits.append(Bandit(epsilon=0.1, UCB_param=1, sample_averages=True))
    bandits.append(Bandit(epsilon=0.1, UCB_param=2, sample_averages=True))
    bandits.append(Bandit(epsilon=0.1, UCB_param=4, sample_averages=True))

    _, _, cumulative_regret = simulate(runs, time, bandits)

    plt.plot(cumulative_regret[0], label=r'UCB $\epsilon = 0.1$, $c = 1$')
    plt.plot(cumulative_regret[1], label=r'UCB $\epsilon = 0.1$, $c = 2$')
    plt.plot(cumulative_regret[2], label=r'UCB $\epsilon = 0.1$, $c = 4$')
    plt.xlabel('Steps')
    plt.ylabel('Cumulative regret')
    plt.legend()

    plt.savefig('images/cumulative_regret.png')
    plt.close()

if __name__ == '__main__':
    cumulative_regret()