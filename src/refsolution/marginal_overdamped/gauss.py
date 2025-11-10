#reference solution for gaussian full bridge
import numpy as np 
import src.sde.boundary as boundaries


class Gaussian_Overdamped_Ref():

    def __init__(self):   
        
        self.boundary = boundaries.UnderdampedGaussian()

        self.mu0 = 0#self.boundary.gauss_boundary["muinitial"]
        self.muT =  0#self.boundary.gauss_boundary["mufinal"]
        self.sigma0 = self.boundary.gauss_boundary["sigmainitial"]
        self.sigmaT = self.boundary.gauss_boundary["sigmafinal"]
        self.A = self.boundary.afun(self.boundary.params["Tf"])
        self.B = self.boundary.bfun(self.boundary.params["Tf"])
        self.Tf2 = self.boundary.params["Tf"]*(self.boundary.params["epsilon"]**2)
        #self.beta = 1/self.A
        self.sqrt_ = np.sqrt(self.Tf2**2 + (self.sigma0*self.sigmaT)**2)

    def m1(self,t):
        num = self.muT*t + self.mu0(self.Tf2-t)
        return 0 #num/Tf

    def v1(self,t):
        return 0 

    def m2(self,t):
        return self.sigma0**2/(1+2*self.B*t*self.alpha2())**3
        #(((self.Tf2-t)*(self.sigma0**(2/3))+t*(self.sigmaT**(2/3)))/self.Tf2)**3

    def alpha2(self):
        return ((1-(self.sigmaT/self.sigma0)**(2/3))/(2*self.B*self.Tf2))
    
    def v2(self,t):
        v20 = self.alpha2()
        return v20/(1-2*self.B*t*v20)
        
    def phi_t(self,q,t):
        return np.exp(-(self.B/self.A)*self.v2(t)*(q**2)/2)

    def hatphi_t(self,q,t):
        return np.exp((self.B/self.A)*(self.v2(t)*q**2)/2)*np.exp(-(q**2)/(2*self.m2(t)))

def normalise(x,y):
    idx = np.argsort(x)
    return y/np.trapz(y[idx],x[idx])

if __name__=="__main__":
    import matplotlib.pyplot as plt
    refsol1 = Gaussian_Overdamped_Ref()
    ts = np.linspace(0,0.2)
    qax= np.linspace(-4,4)
    #plt.plot(ts,refsol1.m2(ts))
    plt.plot(ts,refsol1.v2(ts))
    #plt.plot(qax,normalise(qax,refsol1.phi_t(qax,0.2)))
    #plt.plot(qax,normalise(qax,refsol1.hatphi_t(qax,0.2)))
    #plt.plot(qax,refsol1.phi_t(qax,0.2)*refsol1.hatphi_t(qax,0.2))
    plt.savefig("test_variance_overdamped.png")

    plt.close()