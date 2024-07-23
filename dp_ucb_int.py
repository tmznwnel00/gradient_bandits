import numpy as np

def dp_ucb_int(epsilon, v, K, T, probs):
    rewards = np.zeros(K)
    pulls = np.zeros(K)
    regrets = []

    def bernoulli_reward(p):
        return np.random.rand() < p

    def laplace_noise(b):
        if b <= 0:
            return 0
        return np.random.laplace(0, b)

    f = int(1 / epsilon)
    for t in range(1, T + 1):
        if t <= K * f:
            arm = (t - 1) % K
            reward = bernoulli_reward(probs[arm])
            rewards[arm] += reward
            pulls[arm] += 1
        else:
            x_hat = np.array([
                rewards[a] / pulls[a] + 
                laplace_noise(1 / ((1 - v) * pulls[a])) + 
                np.sqrt(2 * np.log(t) / pulls[a]) 
                if pulls[a] > 0 else 0 
                for a in range(K)
            ])
            arm = np.argmax(x_hat)

            reward = bernoulli_reward(probs[arm])
            rewards[arm] += reward
            pulls[arm] += 1

        regret = max(probs) - rewards.sum() / t
        regrets.append(regret)

    return regrets
