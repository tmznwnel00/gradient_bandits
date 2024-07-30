import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

def compute_regret_DP_UCB(mu, epsilon, n = 10000):
    mu = np.array(mu)
    # Optimal arm's mean
    u_star = max(mu)
    # number of arms
    n_arms = len(mu)
    # Step count for each arm, we start by exploring each action once
    k_n = np.ones(n_arms)
    # Noisy sums for each arms, init with means since each arm is played once first
    noisy_sums = mu.copy()
    sums = mu.copy()
    # regret
    regret = list((u_star - mu).cumsum())
    r = (u_star - mu).sum()
    actions = []
    
    
    # private bonus
    k = n_arms
    gamma =  k*(np.log(n)**2)*(np.log((k * n * np.log(n)) / 0.1))/epsilon
    
    # each tree mechanism is initiated with epsilon/k
    epsilon = epsilon/n_arms
    epsilon1 = epsilon / np.log(n)
    logn_2 = int(np.log(n-n_arms)/np.log(2))
    alpha = np.zeros((logn_2 + 1, n_arms))
    alpha_hat = np.zeros((logn_2 + 1, n_arms))
    for t in range(n_arms+1, n+1):
        # Select action according to UCB Criteria, make it noisy!
        a = np.argmax(noisy_sums/k_n + np.sqrt(2*np.log(t)/k_n) + gamma/k_n)
        actions.append(a)
        # Sample the reward
        #reward = mu[a]
        reward = np.random.binomial(n=1, p= mu[a])
        reward_stream_at_t = np.zeros(n_arms)
        reward_stream_at_t[a] = np.random.binomial(n=1, p=mu[a])
        if t == n_arms+1:
            reward_stream_at_t += mu.copy()
        # update the statistics
        # number of pulls
        k_n[a] += 1
        #noisy sums using the tree mechanism
        binary_rep = np.array(list(np.binary_repr(t - n_arms, width = logn_2 + 1))).astype(int)
        i = np.min(np.nonzero(np.flip(binary_rep)))
        alpha[i] = alpha[:i].sum(axis=0) + reward_stream_at_t
        for j in range(i):
            alpha[j] = 0
            alpha_hat[j] = 0
        noise = np.random.laplace(loc=0.0, scale=1/epsilon1, size=n_arms)
        alpha_hat[i] = alpha[i] + noise
        noisy_sums = alpha_hat.T.dot(np.flip(binary_rep))
        sums[a] += mu[a]
        # update total reward
        #total_reward += reward
        r += u_star - reward
        regret.append(r)
        
    return regret, actions

def generate_private_results(f, list_mu, epsilon, n_iter, n_mc):
    return np.array([f(mu=list_mu, epsilon = epsilon, n = n_iter)[0] for i in range(n_mc)])

n_iter = 10**3
n_mc = 20
list_mu1 = [0.75,0.70,0.70,0.70,0.70]
list_mu2 = [0.75,0.625,0.5,0.375,0.25]
list_mu3 = [0.75,0.53125,0.375,0.28125,0.25]
list_mu4 = [0.75,0.71875,0.625,0.46875,0.25]
eps1, eps2, eps3, eps4 = 0.1, 0.25, 0.5, 1

# list_mu = [list_mu1, list_mu2, list_mu3, list_mu4]
list_mu = [list_mu1]
epss = [eps1]

for i in range(len(list_mu)):
    list_mu_i = list_mu[i]
    for j in range(len(epss)):
        eps_j = epss[j]
        # Generate results
        gen_res_private_ucb_i_eps_j = generate_private_results(compute_regret_DP_UCB, list_mu_i, eps_j, n_iter, n_mc)
        print(len(gen_res_private_ucb_i_eps_j[0]))
    
        # Plotting
        fig = plt.figure()
        fig.set_size_inches(w=4.6, h=3.5)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.arange(len(gen_res_private_ucb_i_eps_j.mean(axis=0))), gen_res_private_ucb_i_eps_j.mean(axis=0),
                color='k', ls='solid', label=r"DP-UCB($\gamma = {:.1f}$)".format(eps_j))
        
        # Uncomment the following lines if you want to add std error bars
        # std_err = gen_res_private_ucb_i_eps_j.std(axis=0) / np.sqrt(gen_res_private_ucb_i_eps_j.shape[0])
        # ax.fill_between(np.arange(len(gen_res_private_ucb_i_eps_j.mean(axis=0))),
        #                 gen_res_private_ucb_i_eps_j.mean(axis=0) - std_err,
        #                 gen_res_private_ucb_i_eps_j.mean(axis=0) + std_err,
        #                 color='k', alpha=0.2)

        plt.xlabel(r"Step $t$")
        plt.ylabel(r"Regret")
        ax.legend(loc='lower right')
        plt.yscale('log')
        
        # Display the plot
        plt.title(f'List Mu = {list_mu_i}, Eps = {eps_j}')
        # plt.show()
