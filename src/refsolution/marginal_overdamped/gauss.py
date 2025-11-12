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
        self.sq = np.sqrt(self.Tf2**2 + (self.sigma0*self.sigmaT)**2)
        #self.beta = 1/self.A
    
    def denominator(self,t):
        return 2*(t*self.sq + (self.Tf2-t)*(self.sigma0**2)-t*self.Tf2)

    def m1(self,t):
        num = self.muT*t + self.mu0(self.Tf2-t)
        return 0 #num/Tf

    def v1(self,t):
        return 0 

    def m2(self,t):
        denom = (self.Tf2*self.sigma0)**2
        return ((t*self.sq + ((self.Tf2-t)*self.sigma0**2))**2 - (t*self.Tf2)**2)/denom
        
    def alpha2(self):
        return ((1-(self.sigmaT/self.sigma0)**(2/3))/(2*self.B*self.Tf2))
    
    def v2(self,t):
        num = self.sq-self.Tf2-(self.sigma0**2)
        return num /self.denominator(t)
        
    def phi_t(self,q,t):
        return np.exp((self.B/self.A)*self.v2(t)*(q**2)/2)

    def hatphi_t(self,q,t):
        return np.exp(-(self.B/self.A)*(self.v2(t)*q**2)/2)*np.exp(-(q**2)/(2*self.m2(t)))

    def rho(self,q,t):
        return np.exp(-(q**2)/(2*self.m2(t)))

    def value_fun(self,q,t):
        return self.v2(t)*(q**2)/2

def normalise(x,y):
    idx = np.argsort(x)
    return y/np.trapz(y[idx],x[idx])

if __name__=="__main__":
    import matplotlib.pyplot as plt
    refsol1 = Gaussian_Overdamped_Ref()
    ts = np.linspace(0,0.2)
    qax= np.linspace(-4,4)
    #plt.plot(ts,refsol1.m2(ts))
    plt.plot(ts,refsol1.m2(ts))
    #plt.plot(qax,normalise(qax,refsol1.phi_t(qax,0.2)))
    #plt.plot(qax,normalise(qax,refsol1.hatphi_t(qax,0.2)))
    #plt.plot(qax,refsol1.phi_t(qax,0.2)*refsol1.hatphi_t(qax,0.2))
    plt.savefig("test_variance_overdamped.png")

    plt.close()