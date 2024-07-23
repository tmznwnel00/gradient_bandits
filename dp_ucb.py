import numpy as np

def hybrid_mechanism_sum(rewards, current_arm, K):
    """Implement the Hybrid Mechanism with zero noise for non-played arms."""
    noise = np.random.laplace(0, 1)  # Adjust the scale as necessary
    
    # Return noise for only the current arm, zero for others
    return sum(rewards[current_arm]) + noise

def dp_ucb(K, T, epsilon, probs):
    rewards = [[] for _ in range(K)]
    pulls = np.zeros(K)
    regrets = []

    def bernoulli_reward(p):
        return np.random.rand() < p

    for t in range(1, T + 1):
        if t <= K:
            arm = t - 1
            reward = bernoulli_reward(probs[arm])
            rewards[arm].append(reward)
            pulls[arm] += 1
        else:
            # Compute UCB values for each arm
            s = np.array([hybrid_mechanism_sum(rewards, a, K) for a in range(K)])
            n_prime = pulls - 2**np.floor(np.log2(pulls)).astype(int)
            nu = (4 * np.sqrt(8) / epsilon) * np.log(t) * (np.log2(n_prime + 1) + 1)

            ucb_values = s / (pulls + 1e-9) + nu / (pulls + 1e-9)  # Added small value to avoid division by zero
            arm = np.argmax(ucb_values)

            reward = bernoulli_reward(probs[arm])
            rewards[arm].append(reward)
            pulls[arm] += 1

        # Compute regret
        regret = max(probs) - sum([sum(rewards[a]) for a in range(K)]) / t
        regrets.append(regret)

    return regrets
