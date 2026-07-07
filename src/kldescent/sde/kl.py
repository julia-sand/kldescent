import numpy as np 
from scipy.spatial import KDTree

def kl_div(d1,d2,k=1,tree1=None,return_tree=False):
    """
    Estimate the KL divergence between two empirical distributions using
    the k-nearest neighbours method proposed by:

    Wang, Q., Kulkarni, S. R., & Verdú, S. (2009),
    IEEE Transactions on Information Theory.

    Parameters
    ----------
    d1 : np.ndarray
        Samples drawn from the first distribution.

    d2 : np.ndarray
        Samples drawn from the second distribution.

    k : int
        Number of nearest neighbours to use. Larger values reduce variance,
        while smaller values reduce bias.

    tree1 : KDTree, optional
        Precomputed KDTree for `d1`. If provided, it is reused to save computation.

    return_tree : bool, optional
        If True, also return the KDTree built from `d1` for reuse.
        Default is False.

    Returns
    -------
    klout : float
        Estimated KL divergence between the empirical distributions of `d1` and `d2`.

    tree1 : KDTree, optional
        Returned only if `return_tree` is True. KDTree constructed from `d1`.
    """
        
    #get d1, d2 shapes
    batch_samples = d1.shape[0] #number of samples n
    d = d1.shape[1] #dimension
    
    #GET NEAREST NEIGHBOURS
    #this indexes the values to their nearest neighbours
    if tree1 is None:
        tree1 = KDTree(d1)
    
    tree2 = KDTree(d2)

    #rho = tree1.query(d1, k=[k+1])[0] #2nd closesst point to x in data1 (since closest is alwayss itself)
    #nu = tree2.query(d1, k=[k])[0] #closest point in data2
    kl_out = (d/batch_samples)*np.log(tree2.query(d1, k=[k])[0]/(tree1.query(d1, k=[k+1])[0]+1e-10)).sum() + np.log(batch_samples/(batch_samples-1))

    if return_tree:
        return kl_out,tree1
    else:
        return kl_out 

def static_kl_divergence(xinit,xref,boundary):
    """
    Computes the KL divergence between the realised final distribution 
    and the reference distribution 

    Parameters
    ----------
    xinit : array
        Samples from a distribution
    xref : array
        Samples from a distribution
    boundary : class
        contains the reference distribution 
    Returns
    -------
    kl_static_from_ref : float
        estimate of the KL divergence between the achieved final distibution and the reference final distribution
    kl_static : float
        estimate of the KL divergence between the empirical distributions found by the neural network and the benchmark drift. 
    """
    
    if boundary.gauss_boundary is not None:

        Sig1 = np.cov(xinit[...,0],xinit[...,1])
        
        kl_static_from_ref = kl_between_gaussians(Sig1,boundary.final_full_distribution_inverse_cov(),boundary.final_full_distribution_covdet())

        Sig2 = np.cov(xref[...,0],xref[...,1])

        kl_static = kl_between_gaussians(Sig1,np.linalg.inv(Sig2),np.linalg.det(Sig2))
    
        return kl_static_from_ref, kl_static

    else: 
        d2 = boundary.generate_samples(xinit.shape[0],mode="final")
        kl_static_from_ref, tree1 = kl_div(xinit,d2,tree1=None,return_tree=True)
        kl_static = kl_div(xinit,xref,tree1=tree1,return_tree=False)
        return kl_static_from_ref, kl_static

def kl_between_gaussians(Sig1,InvSig2,Sig2_det):
    """Compute kl divergence between two gaussian distributions with zero mean"""
    
    return (1/2)*(np.log(Sig2_det/np.linalg.det(Sig1))-2+np.linalg.trace(InvSig2@Sig1))

if __name__=="__main__":
   
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    
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
    