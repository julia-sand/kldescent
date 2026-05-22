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
        """
        Initialize shared parameters for the model.

        Parameters
        ----------
        sigmainitial : float
            Initial variance of the distribution.
        sigmafinal : float
            Final variance of the reference distribution.
        epsilon : float
            Small parameter controlling the perturbation scale.
        Tf : float
            Final time of the process.
        M : float
            Weighting factor for enforcing boundary conditions.
        """

        #self.muinitial = muinitial 
        #self.mufinal = mufinal
        
        self.sigmainitial = sigmainitial 
        self.sigmafinal = sigmafinal
        self.epsilon = epsilon
        self.Tf = Tf 
        self.M = M #weighting of boundary conditions
      
    def _free_dynamics_guess(self, t_vec):
        """Initial guess from free (u=0) dynamics at the current self.epsilon."""
        P = np.ones_like(t_vec)
        C = self.epsilon * (1 - np.exp(-t_vec))
        Q = self.sigmainitial**2 + 2*(self.epsilon**2)*(t_vec + np.exp(-t_vec) - 1)
        v11_0 = 1/self.sigmafinal**2 - 1/self.sigmainitial**2  # exact eps=0 value
        return np.vstack([
            v11_0 * np.ones_like(t_vec),  # v11
            np.zeros_like(t_vec),          # v12
            np.zeros_like(t_vec),          # v22
            Q, C, P
        ])

    def _step(self, sol_prev, bc_fn, eps_from, eps_to, t_vec,
              final_kwargs, max_bisections, depth):
        """Take one continuation step from eps_from to eps_to with bisection fallback."""
        is_final = (eps_to == self.epsilon)
        kwargs = final_kwargs if is_final else {
            **final_kwargs,
            "tol": max(final_kwargs["tol"] * 1e2, 1e-4),
            "bc_tol": max(final_kwargs["bc_tol"] * 1e2, 1e-4),
        }
        self.epsilon = eps_to
        y0 = sol_prev.sol(t_vec)
        sol = solve_bvp(self.evolution, bc_fn, t_vec, y0, **kwargs)
        if not sol.success:
            if depth >= max_bisections:
                raise RuntimeError(
                    f"solve_bvp did not converge at epsilon={eps_to:.4f} "
                    f"after {max_bisections} bisections: {sol.message}"
                )
            eps_mid = (eps_from + eps_to) / 2
            sol_mid = self._step(sol_prev, bc_fn, eps_from, eps_mid, t_vec,
                                 final_kwargs, max_bisections, depth + 1)
            sol = self._step(sol_mid, bc_fn, eps_mid, eps_to, t_vec,
                             final_kwargs, max_bisections, depth + 1)
        return sol

    def _continuation_solve(self, bc_fn, eps_start=0.2, n_steps=5,
                            tol=1e-6, bc_tol=1e-6, max_nodes=100000,
                            max_bisections=4, **bvp_kwargs):
        """
        Solve BVP by stepping epsilon from eps_start up to self.epsilon,
        with adaptive bisection if any step fails to converge.

        Parameters
        ----------
        bc_fn : callable
            Boundary condition function to pass to solve_bvp.
        eps_start : float, optional
            Starting epsilon for continuation. Default is 0.2.
        n_steps : int, optional
            Number of evenly-spaced epsilon steps. Default is 5.
        tol : float, optional
            Residual tolerance for the final solve. Intermediate steps use
            a looser tolerance (100x) to reduce failures. Default is 1e-6.
        bc_tol : float, optional
            Boundary condition tolerance for the final solve. Default is 1e-6.
        max_nodes : int, optional
            Maximum mesh nodes passed to solve_bvp. Default is 100000.
        max_bisections : int, optional
            Maximum bisection depth when a step fails. Default is 4.
        **bvp_kwargs
            Additional keyword arguments forwarded to solve_bvp.

        Returns
        -------
        sol : OdeSolution
            Solution object from solve_bvp at self.epsilon.
        """
        eps_target = self.epsilon
        t_vec = np.linspace(0, self.Tf, num=10000)

        final_kwargs = dict(fun_jac=self.evolution_jacobian,
                            tol=tol, bc_tol=bc_tol, max_nodes=max_nodes,
                            **bvp_kwargs)

        if eps_target <= eps_start:
            self.epsilon = eps_target
            y0 = self._free_dynamics_guess(t_vec)
            sol = solve_bvp(self.evolution, bc_fn, t_vec, y0, **final_kwargs)
            if not sol.success:
                raise RuntimeError(
                    f"solve_bvp did not converge at epsilon={eps_target:.4f}: {sol.message}"
                )
            return sol

        eps_seq = np.linspace(eps_start, eps_target, n_steps + 1)

        self.epsilon = eps_seq[0]
        y0 = self._free_dynamics_guess(t_vec)
        init_kwargs = {**final_kwargs,
                       "tol": max(tol * 1e2, 1e-4),
                       "bc_tol": max(bc_tol * 1e2, 1e-4)}
        sol = solve_bvp(self.evolution, bc_fn, t_vec, y0, **init_kwargs)
        if not sol.success:
            raise RuntimeError(
                f"solve_bvp did not converge at initial epsilon={eps_seq[0]:.4f}: {sol.message}"
            )

        for eps_from, eps_to in zip(eps_seq[:-1], eps_seq[1:]):
            sol = self._step(sol, bc_fn, eps_from, eps_to, t_vec,
                             final_kwargs, max_bisections, depth=0)

        self.epsilon = eps_target
        return sol

    def rho_final(self,QT,CT,PT):
        return CT/np.sqrt(QT*PT)

    def rho_denom(self,QT,CT,PT):
        return 1-self.rho_final(QT,CT,PT)**2

    def marginal_half_final_v11(self,QT):
        return (1/self.sigmafinal**2) - 1/QT

    def final_v11(self,QT,CT,PT):
        return (1/self.sigmafinal**2) - (1/(QT*self.rho_denom(QT,CT,PT)))

    def final_v12(self,QT,CT,PT):
        return (CT/(QT*PT))/self.rho_denom(QT,CT,PT)
        #np.sqrt(1/(PT*QT))*self.rho_final(QT,CT,PT)/(1-self.rho_final(QT,CT,PT)**2)

    def final_v22(self,QT,CT,PT):
        return 1 - (1/(PT*self.rho_denom(QT,CT,PT)))

    def evolution_jacobian(self,t,y):
        """compute jacobian of the evolution equations"""

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

        return self._continuation_solve(self.bc_marginal_half,
                                        verbose=2)

    def get_half_sol(self,):

        return self._continuation_solve(self.bc_half,
                                        tol=1e-8,
                                        bc_tol=1e-8,
                                        verbose=2)

    def get_full_sol(self,):

        return self._continuation_solve(self.bc_full,
                                        verbose=2)

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
