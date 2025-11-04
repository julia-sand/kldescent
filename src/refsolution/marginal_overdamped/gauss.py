#reference solution for gaussian full bridge
import numpy as np 
import src.sde.boundary as boundaries


class Gaussian_Overdamped_Ref():

    def __init__(self):   
        
        self.boundary = boundaries.UnderdampedGaussian()

        self.mu0 = self.boundary.gauss_boundary["muinitial"]
        self.muT =  self.boundary.gauss_boundary["mufinal"]
        self.sigma0 =  self.boundary.gauss_boundary["sigmainitial"]
        self.sigmaT =  self.boundary.gauss_boundary["sigmafinal"]
        self.A = self.boundary.afun(self.boundary.params["Tf"])
        self.B = self.boundary.bfun(self.boundary.params["Tf"])
        self.Tf2 = self.boundary.params["Tf"]*(self.boundary.params["epsilon"]**2)
        self.beta = 1/self.A

        self.sqrt_ = np.sqrt(self.Tf2**2 + (self.sigma0*self.sigmaT)**2)

    def m1(self,t):
        num = self.muT*t + self.mu0(self.Tf2-t)
        return 0 #num/Tf

    def v1(self,t):
        return 0 

    def m2(self,t):
        #num = (t*sqrt_ + (Tf2-t)*sigma0**2)**2
        #return (num - (t*Tf2)**2)/((Tf2*sigma0)**2)
        coeff1 = 1/((self.Tf2*self.beta*self.sigma0)**2)
        t1 = (self.Tf2 - t)*self.beta*self.sigmaT**2
        t2 = t*(-self.Tf2 + np.sqrt((self.beta*self.sigma0*self.sigmaT)**2+self.Tf2**2))*((self.Tf2-t)*self.beta*self.sigma0**2+t*(self.Tf2+np.sqrt((self.beta*self.sigma0*self.sigmaT)**2+self.Tf2**2)))
        return coeff1*(t1+t2)

    def v2(self,t):
        num = self.sqrt_ - self.Tf2 - self.sigma0**2
        denom = 2*(t*self.sqrt_ + (self.Tf2-t)*self.sigma0 - t*self.Tf2)
        return self.beta*num/denom 
        #v20_temp = self.Tf2 + (self.sigma0**2)*self.beta - np.sqrt((self.beta*self.sigma0*self.sigmaT)**2+self.Tf2**2)
        #v20 = v20_temp/(2*self.Tf2*(self.sigma0**2))
        #return self.beta*v20/(self.beta-2*t*v20)

    def phi_t(self,q,t):
        return np.exp((self.v2(t)*q**2)/2)

    def hatphi_t(self,q,t):
        return np.exp(-(self.v2(t)*q**2)/2)*np.exp(-(q**2)/(2*self.m2(t)))#/np.sqrt(2*m2(t)*np.pi)

def normalise(x,y):
    idx = np.argsort(x)
    return y/np.trapz(y[idx],x[idx])

if __name__=="__main__":
    import matplotlib.pyplot as plt
    q = np.linspace(-4,4)
    plt.plot(q,normalise(q,phi_t(q,0)*hatphi_t(q,0)))
    plt.plot(q,boundary.q_marginal_initial(q))
    plt.savefig("test_overdamped.png")

    plt.close()