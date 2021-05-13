
def gelman_rubin(n_chains, n_ensembles, chain_mean_list, chain_var_list):
    # computes ratio of between chain to within chain variance
    
    B, W, theta_hat = 0, 0, 0
    for chain in range(n_chains):
        theta_hat += chain_mean_list[chain]
        W += chain_var_list[chain]**2
        
    theta_hat = theta_hat/n_chains
    W = W/n_chains
    
    for chain in range(n_chains):
        B += (chain_mean_list[chain] - theta_hat)**2
        
    B = B * n_ensembles/(n_chains-1)
    
    return B/W