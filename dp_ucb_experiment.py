import numpy as np
import matplotlib.pyplot as plt
import logging
import time
from dp_ucb_bound import dp_ucb_bound
from dp_ucb import dp_ucb
from dp_ucb_int import dp_ucb_int

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to run experiments
def run_experiment(algorithm, K, T, epsilon, probs, v=1.1):
    results = []
    for i in range(100):
        if i % 10 == 0:
            logging.info(f'Running {algorithm} experiment run {i+1}/100')
        if algorithm == 'dp_ucb_bound':
            results.append(dp_ucb_bound(K, T, epsilon, probs))
        elif algorithm == 'dp_ucb':
            results.append(dp_ucb(K, T, epsilon, probs))
        elif algorithm == 'dp_ucb_int':
            results.append(dp_ucb_int(epsilon, v, K, T, probs))
    avg_results = np.mean(results, axis=0)
    return avg_results

# Experiment parameters
# T = 100000  # Number of time steps
T = 10000
epsilons = [0.1, 1]  # Privacy levels
v = 1.1  # Privacy rate for DP-UCB-Int

# # Scenario 1: Two arms with expectations 0.9 and 0.6
# probs_scenario_1 = [0.9, 0.6]
# K_scenario_1 = len(probs_scenario_1)

# for epsilon in epsilons:
#     logging.info(f'Starting experiments for epsilon = {epsilon} (Scenario 1: Two arms)')
#     start_time = time.time()
#     regret_dp_ucb_bound = run_experiment('dp_ucb_bound', K_scenario_1, T, epsilon, probs_scenario_1)
#     regret_dp_ucb = run_experiment('dp_ucb', K_scenario_1, T, epsilon, probs_scenario_1)
#     regret_dp_ucb_int = run_experiment('dp_ucb_int', K_scenario_1, T, epsilon, probs_scenario_1, v)
#     end_time = time.time()
#     logging.info(f'Experiments for epsilon = {epsilon} (Scenario 1: Two arms) completed in {end_time - start_time} seconds')

#     plt.figure(figsize=(12, 6))
#     plt.plot(regret_dp_ucb_int, label=f'DP-UCB-Int ({epsilon})')
#     plt.plot(regret_dp_ucb, label=f'DP-UCB ({epsilon})')
#     plt.plot(regret_dp_ucb_bound, label=f'DP-UCB-Bound ({epsilon})')
#     plt.xlabel('Time steps')
#     plt.ylabel('Average Regret')
#     plt.legend()
#     plt.title(f'Regret Comparison for Two Arms (ε={epsilon})')
#     plt.grid(True)
#     # plt.show()
#     plt.savefig('images/dp_ucb.png')
#     plt.close()

# Scenario 2: Ten arms with expectations 0.1 (8 arms), 0.55, and 0.2
probs_scenario_2 = [0.1] * 8 + [0.55, 0.2]
K_scenario_2 = len(probs_scenario_2)

for epsilon in epsilons:
    logging.info(f'Starting experiments for epsilon = {epsilon} (Scenario 2: Ten arms)')
    start_time = time.time()
    regret_dp_ucb_bound = run_experiment('dp_ucb_bound', K_scenario_2, T, epsilon, probs_scenario_2)
    regret_dp_ucb = run_experiment('dp_ucb', K_scenario_2, T, epsilon, probs_scenario_2)
    regret_dp_ucb_int = run_experiment('dp_ucb_int', K_scenario_2, T, epsilon, probs_scenario_2, v)
    end_time = time.time()
    logging.info(f'Experiments for epsilon = {epsilon} (Scenario 2: Ten arms) completed in {end_time - start_time} seconds')

    plt.figure(figsize=(12, 6))
    plt.plot(regret_dp_ucb_int, label=f'DP-UCB-Int ({epsilon})')
    plt.plot(regret_dp_ucb, label=f'DP-UCB ({epsilon})')
    plt.plot(regret_dp_ucb_bound, label=f'DP-UCB-Bound ({epsilon})')
    plt.xlabel('Time steps')
    plt.ylabel('Average Regret')
    plt.legend()
    plt.title(f'Regret Comparison for Ten Arms (ε={epsilon})')
    plt.grid(True)
    # plt.show()
    plt.savefig('images/dp_ucb.png')
    plt.close()
