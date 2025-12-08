import numpy as np 
from scipy.spatial import KDTree

def kl_div(d1,d2,k=1):
    """
    Computes the KL divergence between two sets of sampled points 
    via the k-nearest-neighbours method of 
    Wang, Qing and Kulkarni, Sanjeev R. and Verdu, Sergio. 
    In: IEEE Transactions on Information Theory (2009)

    Parameters
    ----------
    d1 : array
        Samples from a distribution
    d2 : array
        Samples from a distribution
    k : int
        number of nearest neighbours to consider. As a rule,larger k reduces variance, small k reduces bias
    
    Returns
    ------
    klout : float
        estimate of the KL divergence between the two empirical distributions d1,d2
    """
        
    #check d1, d2 in the right shape
    batch_samples = d1.shape[0]
    
    #GET NEAREST NEIGHBOURS
    #this indexes the values to their nearest neighbours
    tree1 = KDTree(d1)
    tree2 = KDTree(d2)

    #rho = tree1.query(d1, k=[k+1])[0] #2nd closesst point to x in data1 (since closest is alwayss itself)
    #nu = tree2.query(d1, k=[k])[0] #closest point in data2

    return (2/batch_samples)*np.log(tree2.query(d1, k=[k])[0]/(tree1.query(d1, k=[k+1])[0]+1e-10)).sum() + np.log(batch_samples/(batch_samples-1))


if __name__=="__main__":
   
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    #k=1
    
    plt.figure()
    for k in np.linspace(1,5,5):

        vals = np.zeros(4)
        lgvals = np.linspace(1,4,4)

        dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.ones(2), torch.eye(2))
        for lg in enumerate(lgvals):
            #p = dist.sample((1,int(10**lg[1]))).reshape((int(10**lg[1]),2))
            p = generate_samples(int(10**lg[1]),mode="initial")
            vals[lg[0]] = kl_div(p,k)
        
        plt.plot(lgvals,vals,label=f"k={k}")
        #plt.title(f"KL between final and initial state with k={k}")

    #compute the actual
    xs_test = np.linspace(-8,8,1000)
    kl = (16/1000)*np.sum(q_marginal_initial(xs_test)*((-(xs_test**4)+(xs_test**2-1)**2)/4+np.log(pf_norm/pi_norm)))
    plt.plot(lgvals,kl*np.ones(4),label="kl ref")
    plt.legend()
    plt.xlabel("number of samples (power of ten)")
    plt.savefig(f"kldescent/tests/boundary_test.png")
    