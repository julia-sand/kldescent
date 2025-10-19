import numpy as np

import src.sde.boundary as boundaries
#Proximal operator solution for fokkerplanck equation
import pdb

def afun(Tf):
    return -(Tf - 1 + np.exp(-Tf))/Tf

def bfun(Tf):
    return -(1 - (1-np.exp(-Tf))*(3-np.exp(-Tf))/(2*Tf))


def proximal_fpk(x0,y0,
                    num_steps=100,
                    max_iters=1000,
                    eps=0.05,
                    tol=0.00001):

    #set up sampler
    Generator = np.random.default_rng(seed=42)
    
    samples = x0.shape[0] #number of points in the cloud
    x_evolution = x0.copy()

    dt = Tf/num_steps

    #preallocate
    x_evolution_new = np.empty_like(x_evolution) #np.zeros((samples,num_steps))

    power= 1/(1+(beta*eps/dt))
    yvals = y0.copy()

    for k in range(num_steps):
        
        #step of sde evolution 
        x_evolution_new = x_evolution + np.sqrt(2*dt/beta)*Generator.standard_normal(size=samples)
        #

        c_k = np.zeros((samples,samples))
        gamma_k = np.zeros((samples,samples))

        for i in range(samples):
            for j in range(samples):
                c_k[i,j] = (x_evolution_new[i]-x_evolution[j])**2

        gamma_k = np.exp(-c_k/(2*eps))
        #pdb.set_trace()

        gamma_kt = gamma_k.T 

        z_samples = Generator.standard_normal(size=x_evolution.shape)

        z_old = np.exp((dt*z_samples)/eps) 
        y_old = yvals/(gamma_k@z_old)

        #preallocate
        y_new = np.empty_like(y_old)
        z_new = np.empty_like(z_old)

        xi = np.exp(-1)*np.ones_like(x_evolution)

        iters = 0 

        while iters<=max_iters: 
            z_new = (xi / (gamma_kt@y_old))**power
            y_new = yvals/(gamma_k@z_new)

            if ((np.sum(np.abs(y_new - y_old))<tol) and
                    (np.sum(np.abs(z_new - z_old))<tol)):
                break
            else: 
                iters += 1
            
            #update iterates
            y_old = y_new
            z_old = z_new
        
        #set new running solutions
        yvals = z_new* (gamma_kt@y_new)
        x_evolution = x_evolution_new

    return x_evolution_new,yvals

if __name__=="__main__":
        
    boundary = boundaries.UnderdampedGaussian()
    Tf = boundary.params["Tf"]
    epsilon = boundary.params["epsilon"]

    alpha = afun(Tf)
    beta = -1/alpha
    nsamples=600

    #get initial solution
    x0 = boundary.generate_samples_q(nsamples,"initial").flatten()
    y0 = boundary.q_marginal_initial(x0).flatten()
    
    x_end, y_end = proximal_fpk(x0,y0)

    import matplotlib.pyplot as plt 

    plt.scatter(x_end, y_end)
    #plt.plot(np.linspace(0,8),afun(np.linspace(0,8)))
    plt.savefig("test_fpk_prox1.png")
    plt.close()