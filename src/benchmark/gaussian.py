"""Solve the gaussian case non-perturbatively"""

import numpy as np 
from scipy.integrate import solve_bvp 

from .utilfunctions import save_array

class GaussBVP():

    def __init__(self,
                      #muinitial,
                      #mufinal,
                      sigmainitial,
                      sigmafinal,
                      epsilon,
                      Tf,
                      M):
        """Set shared parameters"""
        #self.muinitial = muinitial 
        #self.mufinal = mufinal
        
        self.sigmainitial = sigmainitial 
        self.sigmafinal = sigmafinal
        self.epsilon = epsilon
        self.Tf = Tf 
        self.M = M #weighting of boundary conditions
        
    def drho_final(self,QT,CT,PT):
        return np.array([(-1/2)*PT*CT/((QT*PT)**(3/2)),
                        1/np.sqrt(QT*PT),
                        (-1/2)*QT*CT/((QT*PT)**(3/2))])

    def rho_final(self,QT,CT,PT):
        return CT/np.sqrt(QT*PT)

    def drho_denom(self,QT,CT,PT):
        return self.drho_final(QT,CT,PT)*(-2*self.rho_final(QT,CT,PT))
      
    def rho_denom(self,QT,CT,PT):
        return 1-self.rho_final(QT,CT,PT)**2

    def dfinal_v11(self,QT,CT,PT):

        return np.array([-self.drho_denom(QT,CT,PT)[0]/(QT*self.rho_denom(QT,CT,PT)**2)-1/((QT**2)*self.rho_denom(QT,CT,PT)),
                        -self.drho_denom(QT,CT,PT)[1]/(QT*self.rho_denom(QT,CT,PT)**2),
                        -self.drho_denom(QT,CT,PT)[2]/(QT*self.rho_denom(QT,CT,PT)**2)])

    def marginal_half_final_v11(self,QT):
        return (1/self.sigmafinal**2) - 1/QT

    def final_v11(self,QT,CT,PT):
        return (1/self.sigmafinal**2) - (1/(QT*self.rho_denom(QT,CT,PT)))

    def dfinal_v12(self,QT,CT,PT):

        return np.array([-(CT/(QT*PT))*self.drho_denom(QT,CT,PT)[0]*(CT/(QT*PT))/(self.rho_denom(QT,CT,PT)**2)-(CT/((QT**2)*PT))/(self.rho_denom(QT,CT,PT)),
                        -(CT/(QT*PT))*self.drho_denom(QT,CT,PT)[1]/(QT*self.rho_denom(QT,CT,PT)**2)+(1/((QT)*PT))/(self.rho_denom(QT,CT,PT)),
                        -(CT/(QT*PT))*self.drho_denom(QT,CT,PT)[2]/(QT*self.rho_denom(QT,CT,PT)**2)-(CT/(QT*(PT**2)))/(self.rho_denom(QT,CT,PT))])

    def final_v12(self,QT,CT,PT):
        return (CT/(QT*PT))/self.rho_denom(QT,CT,PT)
        #np.sqrt(1/(PT*QT))*self.rho_final(QT,CT,PT)/(1-self.rho_final(QT,CT,PT)**2)

    def dfinal_v22(self,QT,CT,PT):

        return np.array([-self.drho_denom(QT,CT,PT)[0]/(PT*self.rho_denom(QT,CT,PT)**2),
                        -self.drho_denom(QT,CT,PT)[1]/(PT*self.rho_denom(QT,CT,PT)**2),
                        -self.drho_denom(QT,CT,PT)[2]/(PT*self.rho_denom(QT,CT,PT)**2)-1/((PT**2)*self.rho_denom(QT,CT,PT))])

    def final_v22(self,QT,CT,PT):
        return 1 - (1/(PT*self.rho_denom(QT,CT,PT)))

    def evolution_jacobian(self,t,y):
        """compute jacobian of the evolution equations
        TODO: check these are correct, currently not used because they cause errors"""

        f0_deriv = np.array([np.zeros_like(y[0]),
                            4*(y[1]),
                            -4*((y[4]/y[3])**2)*y[2],
                            4*(y[4]*y[2])**2/(y[3]**3),
                            -4*((y[2]/y[3])**2)*y[4],
                            np.zeros_like(y[0])])
        f1_deriv = np.array([- self.epsilon*np.ones_like(y[0]),
                            1 + 2*y[2],
                            2*y[1] + 4*y[4]*y[2]/y[3],
                            -2*y[4]*((y[2]/y[3])**2),
                            2*(y[2]**2)/y[3],
                            np.zeros_like(y[0])])
        f2_deriv = np.array([np.zeros_like(y[0]),
                            -2*self.epsilon*np.ones_like(y[0]),
                            2.0*np.ones_like(y[0]),
                            np.zeros_like(y[0]),
                            np.zeros_like(y[0]),
                            np.zeros_like(y[0])])
        f3_deriv = np.array([np.zeros_like(y[0]),
                            np.zeros_like(y[0]),
                            np.zeros_like(y[0]),
                            np.zeros_like(y[0]),
                            2.0*self.epsilon*np.ones_like(y[0]),
                            np.zeros_like(y[0])])
        f4_deriv = np.array([np.zeros_like(y[0]),
                            -2*y[3],
                            -2*y[4],
                            -2*y[1],
                            - 1 - 2*y[2],
                            self.epsilon*np.ones_like(y[0])])      
        f5_deriv = np.array([np.zeros_like(y[0]),
                            - 4*y[4],
                            - 4*(y[4]**2)/y[3],
                            4*((y[4]/y[3])**2)*y[2],
                            - 4*y[1] - 8*y[4]*y[2]/y[3],
                            - 2*np.ones_like(y[5])])
        
        return np.array([f0_deriv,
                        f1_deriv,
                        f2_deriv,
                        f3_deriv,
                        f4_deriv,
                        f5_deriv])

    def evolution(self, t, y):
        
        return np.vstack((2*(y[1]**2) -2*((y[4]*y[2]/y[3])**2), 
                            y[1] + 2*y[2]*y[1] - self.epsilon*y[0] + 2*y[4]*(y[2]**2)/y[3],
                            2.0*(y[2]-y[1]*self.epsilon),
                            2.0*self.epsilon*y[4],
                            self.epsilon*y[5] - 2*y[3]*y[1] - y[4] - 2*y[2]*y[4],
                            2 - 2*y[5] - 4*(y[1]*y[4]) - 4*(y[4]**2)*y[2]/y[3]))
                            
              
    def bc_marginal_half(self, ya, yb):
        return np.array([ya[3] - self.sigmainitial**2,  #initial sigma
                            ya[4] , #initial cross corr
                            ya[5] - 1, #initial mom var
                            yb[0] - self.marginal_half_final_v11(yb[3]), #final value funs
                            yb[1] ,
                            yb[2]])

    def bc_half(self, ya, yb):
        return np.array([ya[3] - self.sigmainitial**2,  #initial sigma
                            ya[4] , #initial cross corr
                            ya[5] - 1, #initial mom var
                            yb[0] - self.M*self.final_v11(yb[3],yb[4],yb[5]),#final value funs
                            yb[1] - self.M*self.final_v12(yb[3],yb[4],yb[5]),
                            yb[2] - self.M*self.final_v22(yb[3],yb[4],yb[5])])

    def bc_full(self, ya, yb):
        return np.array([ya[3] - self.sigmainitial**2,  #initial sigma
                            ya[4] , #initial cross corr
                            ya[5] - 1, #initial mom var
                            yb[3] - self.sigmafinal**2, #final value funs
                            yb[4] ,
                            yb[5] - 1])

    def bc_jacobian(self,):
        pass 

    def get_marginal_half_sol(self,):
        
        t_vec = np.linspace(0,self.Tf,num=10000)

        #initial guess
        y0 = np.tile(np.array([-1,1,1,1,0,1]),(len(t_vec),1)).T
        
        sol1 = solve_bvp(self.evolution, self.bc_marginal_half, t_vec, y0,
                        fun_jac=self.evolution_jacobian,
                        verbose=2)

        return sol1 

    def get_half_sol(self,):
        
        t_vec = np.linspace(0,self.Tf,num=10000)

        #initial guess
        y0 = np.tile(np.array([-1,0,0,1,0,1]),(len(t_vec),1)).T
        
        sol1 = solve_bvp(self.evolution, self.bc_half, t_vec, y0,
                        fun_jac=self.evolution_jacobian,
                        max_nodes=100000,
                        tol=1e-8,
                        bc_tol=1e-8,
                        verbose=2)

        return sol1 

    def get_full_sol(self,):
        
        t_vec = np.linspace(0,self.Tf,num=50000)

        #initial guess
        y0 = np.tile(np.array([1,1,1,1,0,1]),(len(t_vec),1)).T
        
        sol1 = solve_bvp(self.evolution, self.bc_full, t_vec, y0,
                        #fun_jac=self.evolution_jacobian,
                        verbose=2)

        return sol1 

    def get_u2(self,sol1,t_vec):

        return (2/self.epsilon)*(sol1.sol(t_vec)[1] + (sol1.sol(t_vec)[4]*sol1.sol(t_vec)[2]/sol1.sol(t_vec)[3]))

    def save_csvs(self,sol1,save_location,tnum=1000):
        
        t_vec = np.linspace(0,self.Tf,num=tnum) #sol1.x
        #np.linspace(0,self.Tf,num=tnum)

        arr_to_save = np.concatenate((t_vec[:,np.newaxis],  #time
                                sol1.sol(t_vec)[0][:,np.newaxis],  #v11
                                sol1.sol(t_vec)[1][:,np.newaxis],  #v12
                                sol1.sol(t_vec)[2][:,np.newaxis],  #v22
                                self.get_u2(sol1,t_vec)[:,np.newaxis]),axis=1)  #U2
                    
        cumulants_to_save =  np.hstack((t_vec[:,np.newaxis],           
                                sol1.sol(t_vec)[3][:,np.newaxis],  #c11 / pos var
                                sol1.sol(t_vec)[4][:,np.newaxis],  #c12 / xcorr
                                sol1.sol(t_vec)[5][:,np.newaxis])) #c22 / mom var                

        save_array(arr_to_save, save_location, filename="underdamped",header="t,v11,v12,v22,u2")
        save_array(cumulants_to_save, save_location, filename="underdamped_cumulants",header="t,c11,c12,c22")
        
        return
    
if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    from ..sde.boundary import UnderdampedGaussian

    bvp1 = GaussBVP(boundary=UnderdampedGaussian())
    sol1 = bvp1.get_sol()
    bvp1.save_csvs(sol1,".")

    #plot the solution 
    x_plot = np.linspace(0, 5, 100)
    for i in range(6):
        y_plot_a = sol1.sol(x_plot)[i]
    
        plt.plot(x_plot, y_plot_a, label=f'V{i}')
    
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("temp1.png")
    plt.close
