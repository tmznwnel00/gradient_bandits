import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
# from scipy.stats import beta

from thompson_sampling import Thompson

# colour-blind friendly palette
cb_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00', '#66c2a5']

# rounds to iterate over
time_steps = 1000
# experiments to average over
experiments = 500
# bandits with known probabilites
bandit_probs = [0.1, 0.4, 0.45, 0.6, 0.61]


# initialize class
ts = Thompson(bandit_probs, time_steps)

# To remove noise, we average the quantities of interest across 500 experiments.

avg_cumsum_rewards = np.zeros((ts.n_bandits, ts.steps))
avg_cumsum_penalties = np.zeros((ts.n_bandits, ts.steps))
avg_regret = np.zeros(ts.steps)

# run!
for i in range(experiments):
    ts.run_experiment()
    avg_cumsum_rewards += ts.cumsum_rewards
    avg_cumsum_penalties += ts.cumsum_penalties
    avg_regret += ts.regret
    print(f"{i+1}/{experiments} concluded", end='\r')
avg_cumsum_rewards /= experiments
avg_cumsum_penalties /= experiments
avg_regret /= experiments

avg_rewards = np.sum(avg_cumsum_rewards, axis=0) / (np.arange(1, ts.steps + 1))

ts2 = Thompson(bandit_probs, time_steps, random=True)

avg_cumsum_rewards2 = np.zeros((ts2.n_bandits, ts2.steps))
avg_cumsum_penalties2 = np.zeros((ts2.n_bandits, ts2.steps))
avg_regret2 = np.zeros(ts2.steps)

# run!
for i in range(experiments):
    ts2.run_experiment()
    avg_cumsum_rewards2 += ts2.cumsum_rewards2
    avg_cumsum_penalties2 += ts2.cumsum_penalties2
    avg_regret2 += ts2.regret
    print(f"{i+1}/{experiments} concluded", end='\r')
avg_cumsum_rewards2 /= experiments
avg_cumsum_penalties2 /= experiments
avg_regret2 /= experiments

avg_rewards2 = np.sum(avg_cumsum_rewards2, axis=0) / (np.arange(1, ts2.steps + 1))

# approximate expectation value of the beta distribution
# (slightly modified to avoid division by zero errors)

def expec_val(alpha, beta):
    
    return (1 + alpha) / (1 + alpha + beta)

# here a plot of the E_k = alpha_k / (alpha_k, beta_k) for all bandits k, at 5 equidistant points in time

x = np.arange(ts.steps)

fig, ax = plt.subplots(1, figsize=(15, 10))
    
# looping over bandits 
for bandit in range(ts.n_bandits):
    ax.plot(x, expec_val(avg_cumsum_rewards[bandit, x], avg_cumsum_penalties[bandit, x]),
            color=cb_color_cycle[bandit], lw=4, alpha=0.6,
            label=f"p={bandit_probs[bandit]}_1")
    ax.plot(x, expec_val(avg_cumsum_rewards2[bandit, x], avg_cumsum_penalties2[bandit, x]),
            color=cb_color_cycle[bandit+5], lw=4, alpha=0.6,
            label=f"p={bandit_probs[bandit]}_2")
    ax.axhline(y=ts.success_probs[bandit],
               color=cb_color_cycle[bandit], ls ='-.', lw=2, alpha=0.6)
ax.set_xlim(0,ts.steps)
ax.set_ylim(0,1)
ax.legend(loc='upper right')
plt.xlabel("Iterations")
plt.ylabel("Estimated success probabilities")
plt.title(f"Average over {experiments} experiments")
plt.show()


# # here a plot of ts.regret against time
# x = np.arange(ts.steps)

# # fig = plt.figure(figsize=(10, 6))
# fig, ax = plt.subplots(1, figsize=(15, 10))
    
# plt.plot(x, avg_regret[x],
#          color=cb_color_cycle[0], lw=4, alpha=0.6, label="orignal")

# plt.plot(x, avg_regret2[x],
#          color=cb_color_cycle[1], lw=4, alpha=0.6, label="random")

# plt.xlim(0,ts.steps)
# plt.ylim(0,0.5)
# ax.legend(loc='upper right')
# plt.xlabel("Iterations")
# plt.ylabel("Regret")
# plt.title(f"Average over {experiments} experiments")
# plt.show()

# plt.plot(x, avg_rewards,
#          color=cb_color_cycle[1], lw=4, alpha=0.6)

# plt.xlim(0, ts.steps)
# plt.ylim(0, 1)
# plt.xlabel("Iterations")
# plt.ylabel("Average Reward")
# plt.title(f"Average Reward over {experiments} experiments")
# plt.show()