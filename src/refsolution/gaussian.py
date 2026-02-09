"""Solve the gaussian case non-perturbatively"""

import numpy as np 
from scipy.integrate import solve_bvp 

from utilfunctions import save_array

#from ..utils.parser import gaussian_params,boundary_params

class GaussBVP():

    def __init__(self,boundary):
        self.boundary = boundary
        self.epsilon = self.boundary.params["epsilon"]

    def rho_final(self,QT,CT,PT):
        return CT/np.sqrt(QT*PT)

    def final_v11(self,QT,CT,PT):
        return (1/self.boundary.gauss_boundary["sigmafinal"]**2) - (1/(QT*(1-self.rho_final(QT,CT,PT)**2)))

    def final_v12(self,QT,CT,PT):
        return np.sqrt(1/(PT*QT))*self.rho_final(QT,CT,PT)/(1-self.rho_final(QT,CT,PT)**2)

    def final_v22(self,QT,CT,PT):
        return 1 - (1/(PT*(1-self.rho_final(QT,CT,PT)**2)))


    def evolution(self, x, y):
        
        return np.vstack((2*(y[1]**2) - 2*((y[4]*y[2]/y[3])**2), 
                            y[1] + 2*y[2]*y[1] -self.epsilon*y[0] + 2*y[4]*(y[2]**2)/y[3],
                            2.0*(y[2]-y[1]*self.epsilon),
                            2.0*self.epsilon*y[4],
                            self.epsilon*y[5] - 2*y[3]*y[1] - y[4] - 2*y[2]*y[4],
                            2 - 2*y[5] - 4*y[4]*y[1] - ((4*(y[4]**2)*y[2])/(y[3]))))
            
    def bc(self, ya, yb):
        return np.array([ya[3] - self.boundary.gauss_boundary["sigmainitial"]**2,  #initial sigma
                            ya[4] , #initial cross corr
                            ya[5] - 1, #initial mom var
                            yb[0] - self.final_v11(yb[3],yb[4],yb[5]),#final value funs
                            yb[1] - self.final_v12(yb[3],yb[4],yb[5]),
                            yb[2] - self.final_v12(yb[3],yb[4],yb[5])])

    def get_sol(self,):
        
        t_vec = np.linspace(0,self.boundary.params["Tf"],num=1000)

        #initial guess
        y0 = np.ones((6, t_vec.size))
        
        sol1 = solve_bvp(self.evolution, self.bc, t_vec, y0)

        return sol1 

    def save_csvs(self,sol1,save_location,tnum=1000):
        
        t_vec = np.linspace(0,self.boundary.params["Tf"],num=tnum)

        arr_to_save = np.vstack([t_vec,               #time
                    sol1.sol(t_vec)[0], #v11
                    sol1.sol(t_vec)[1], #v12
                    sol1.sol(t_vec)[2]]) #v22
                    
        cumulants_to_save =  np.vstack([t_vec,           
                                sol1.sol(t_vec)[3],#c11 / pos var
                                sol1.sol(t_vec)[4],#c12 / xcorr
                                sol1.sol(t_vec)[5] #c22 / mom var
                                ])

        save_array(arr_to_save, save_location, filename="underdamped",header="t,v11,v12,v22")
        save_array(cumulants_to_save, save_location, filename="underdamped_cumulants",header="t,c11,c12,c22")
        
        return
    
if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    from ..sde.boundary import UnderdampedGaussian

    bvp1 = GaussBVP(boundary=UnderdampedGaussian())
    sol1 = bvp1.get_sol()

    x_plot = np.linspace(0, 5, 100)
    for i in range(6):
        y_plot_a = sol1.sol(x_plot)[i]
    
        plt.plot(x_plot, y_plot_a, label=f'V{i}')
    
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("temp1.png")
    plt.close
