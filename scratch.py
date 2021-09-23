
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
    
'''def find_rep(av_fracs, pixels):
    # identify representative pixels for each pft
    # also return fraction represented and dominance (1 or 0)
    av_fracs_copy = np.copy(av_fracs)
    max_frac = np.nanmax(av_fracs_copy, axis=0)
    max_frac_dom = np.ones(len(max_frac))*np.nan
    
    av_fracs_copy[av_fracs_copy==max_frac] = np.nan
    second_max_frac = np.nanmax(av_fracs_copy, axis=0)
    second_max_frac_dom = np.ones(len(second_max_frac))*np.nan
    
    av_fracs_copy[av_fracs_copy==second_max_frac] = np.nan
    third_max_frac = np.nanmax(av_fracs_copy, axis=0)
    third_max_frac_dom = np.ones(len(third_max_frac))*np.nan
    
    av_fracs_copy[av_fracs_copy==third_max_frac] = np.nan
    fourth_max_frac = np.nanmax(av_fracs_copy, axis=0)
    fourth_max_frac_dom = np.ones(len(fourth_max_frac))*np.nan
    
    av_fracs_copy[av_fracs_copy==fourth_max_frac] = np.nan
    fifth_max_frac = np.nanmax(av_fracs_copy, axis=0)
    fifth_max_frac_dom = np.ones(len(fifth_max_frac))*np.nan
    
    rep_pixels = []
    for i in range(len(max_frac)):
        
        if max_frac[i]>0:
            max_frac_rows = np.where(av_fracs[:,i]==max_frac[i])
            second_max_frac_rows = np.where(av_fracs[:,i]==second_max_frac[i])
            third_max_frac_rows = np.where(av_fracs[:,i]==third_max_frac[i])
            fourth_max_frac_rows = np.where(av_fracs[:,i]==fourth_max_frac[i])
            fifth_max_frac_rows = np.where(av_fracs[:,i]==fifth_max_frac[i])
            
            max_frac_dom[i] = 1 if max_frac[i]==np.max(av_fracs[max_frac_rows]) else 0
            second_max_frac_dom[i] = 1 if second_max_frac[i]==np.max(av_fracs[second_max_frac_rows]) else 0
            third_max_frac_dom[i] = 1 if third_max_frac[i]==np.max(av_fracs[third_max_frac_rows]) else 0
            fourth_max_frac_dom[i] = 1 if fourth_max_frac[i]==np.max(av_fracs[fourth_max_frac_rows]) else 0
            fifth_max_frac_dom[i] = 1 if fifth_max_frac[i]==np.max(av_fracs[fifth_max_frac_rows]) else 0
        
            rep_pixels.append([[pixels[row] for row in max_frac_rows[0]][0], 
                [pixels[row] for row in second_max_frac_rows[0]][0], 
                [pixels[row] for row in third_max_frac_rows[0]][0], 
                [pixels[row] for row in fourth_max_frac_rows[0]][0], 
                [pixels[row] for row in fifth_max_frac_rows[0]][0]])
        
        else: rep_pixels.append(['-9999','-9999','-9999','-9999','-9999'])
        
    mxs = [max_frac, second_max_frac, third_max_frac, fourth_max_frac, fifth_max_frac]
    mxfracdoms = [max_frac_dom, second_max_frac_dom, third_max_frac_dom, fourth_max_frac_dom, fifth_max_frac_dom]
    return rep_pixels, mxs, mxfracdoms'''